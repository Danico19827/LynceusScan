# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Node -- Decimate point cloud.

Reduces the point density of every tile, pruning datasets for a first
exploration or for grids coarser than the source density. Output is a
thinned stream (points are physically removed): downstream nodes
(DTM/DSM/...) keep working on the decimated cloud.

Modes: ``ratio`` keeps roughly the requested percentage (quantized to a
1-in-k stride, so 30% keeps 1/3); ``adaptive`` picks k per tile to land
near ``target_points``. Tiles already at or below target pass through; a
single point is always kept so the stream never empties.

The decimation is deterministic: ``nth`` keeps every k-th point in file
order and ``random`` samples a seed-per-tile subset with the same count,
so re-runs replicate the exact same output on every machine.
"""

from __future__ import annotations

from lynceus.nodes._point_source import run_clean
from lynceus.nodes.ports import PortType

NODE_ID = "lynceus.nodes.lidar.clean.decimate"
NODE_NAME = "Decimate"
NODE_CATEGORY = "LiDAR"

NODE_SUBCATEGORY = "Cleaning"

NODE_DESCRIPTION = (
    "<b>Decimate</b> -- Point Density Reduction<br><br>"
    "Prunes every tile to a fraction of its points, keeping the pipeline "
    "fast while exploring very dense clouds. Use it when the source LiDAR "
    "density is far above what the downstream products need (e.g. grid "
    "metrics at 1 m or coarser).<br><br>"
    "<b>Modes:</b> <i>ratio</i> keeps a fixed percentage of points "
    "(default 25%); <i>adaptive</i> targets a fixed number of points per "
    "tile instead. When the tile already fits the target nothing is lost "
    "(the node is a pass-through).<br><br>"
    "<b>Strategy:</b> <i>nth</i> keeps every k-th point in file order — "
    "fast and deterministic. <i>random</i> keeps the same count as a "
    "random sample seeded per tile, avoiding any residual striping in "
    "structured clouds.<br><br>"
    "<b>Tip:</b> the tile cache below this node is untouched, so changing "
    "the ratio only reprocesses this node, never the source tiling."
)

INPUTS = (PortType.POINT_CLOUD,)
OUTPUTS = (PortType.POINT_CLOUD,)

PROCESSING_SPECS = {
    "tile_task": "tile_decimate",
    "input_port": "point_cloud",
    "output_port": "point_cloud",
    "passes_flags": ("classified",),
    "point_cloud_dir": "decimated",
    "output_globs": ("decimated/**/*.laz",),
    "config_schema": {
        "mode": {
            "type": "str",
            "default": "ratio",
            "options": ["ratio", "adaptive"],
            "description": "Decimation mode: ratio keeps a fixed percentage; adaptive targets a fixed number of points per tile.",
            "impact": "ratio is the simplest way to thin a dense cloud (start at 25%). adaptive is more predictable across heterogeneous data: it keeps roughly the requested point count per tile regardless of its original density.",
            "group": "Sampling",
        },
        "sampling_ratio": {
            "type": "float",
            "default": 25.0,
            "minimum": 0.0,
            "maximum": 100.0,
            "description": "Percentage of points to keep in ratio mode.",
            "impact": "50 keeps every 2nd point, 25 every 4th, 10 every 10th. Lower values cut faster but can starve dense-cells below the grid resolution; a tiny minimum is always kept so the stream never empties.",
            "group": "Sampling",
        },
        "target_points": {
            "type": "int",
            "default": 500000,
            "minimum": 1,
            "maximum": 100000000,
            "description": "Target points per tile in adaptive mode.",
            "impact": "Tiles smaller than the target pass through unchanged (no loss); denser tiles are thinned just enough to reach it. Useful to bound the downstream processing time per tile.",
            "group": "Sampling",
        },
        "strategy": {
            "type": "str",
            "default": "nth",
            "options": ["nth", "random"],
            "description": "Which points to drop when thinning.",
            "impact": "nth keeps every k-th point in file order: fastest and fully deterministic, ideal for huge files. random keeps the same count but picks a different (still deterministic) subset per tile, removing potential striping in regularly structured clouds.",
            "group": "Strategy",
        },
    },
}


def get_config_defaults() -> dict:
    return {k: v["default"] for k, v in PROCESSING_SPECS["config_schema"].items()}


_SEED_CACHE: dict[str, int] = {}


def _stable_seed(tile_id: str) -> int:
    import hashlib

    key = str(tile_id)
    seed = _SEED_CACHE.get(key)
    if seed is None:
        seed = int(hashlib.sha1(key.encode()).hexdigest()[:12], 16)
        if len(_SEED_CACHE) < 4096:
            _SEED_CACHE[key] = seed
    return seed


def _keep_mask(record, ctx: dict):
    """Boolean keep mask (True = point is retained)."""
    import numpy as np

    n = len(record)
    if n <= 1:
        return np.ones(n, dtype=bool)

    mode = ctx.get("mode", "ratio")
    if mode == "adaptive":
        target = int(ctx.get("target_points", 0) or 0)
        step = 1 if target <= 0 else max(1, -(-n // target))
    else:
        ratio = float(ctx.get("sampling_ratio", 25.0) or 0.0)
        ratio = max(0.0, min(ratio, 100.0))
        if ratio >= 100.0:
            return np.ones(n, dtype=bool)
        step = max(2, int(round(100.0 / ratio))) if ratio > 0 else n + 1

    if step <= 1:
        return np.ones(n, dtype=bool)

    keep = np.zeros(n, dtype=bool)
    count = max(1, int(np.ceil(n / step)))
    if ctx.get("strategy", "nth") == "random":
        # Same point count as nth, but a seeded random subset (reproducible
        # per tile across machines), avoiding striping in structured clouds.
        rng = np.random.default_rng(_stable_seed(ctx.get("tile_id", "")))
        keep[rng.choice(n, size=count, replace=False)] = True
    else:
        keep[::step] = True
    if "key_point" in set(record.point_format.dimension_names):
        # Model key points are never decimated (ASPRS 1.4): they ride
        # along on top of the sampled count.
        keep = keep | np.asarray(record.key_point, dtype=bool)
    return keep


def tile_decimate(tile: dict, ctx: dict) -> dict:
    """Thin each tile's points (deterministic, non-destructive removal)."""
    import numpy as np

    def mask_fn(record) -> np.ndarray:
        return _keep_mask(record, ctx)

    return run_clean(
        tile, ctx, NODE_ID, "decimated", mask_fn,
        force_remove=True,
    )