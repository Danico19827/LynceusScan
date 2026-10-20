# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Node -- Filter by QA Flags (vendor overlap/synthetic culling).

Removes points carrying LAS 1.4 vendor QA flags (flight-line overlap,
synthetic) or isolates the overlap strips for separate processing. The
`withheld` flag needs no option: every cleaning node already excludes it
unconditionally. Key points are never touched (decimation protects them).
"""

from __future__ import annotations

from lynceus.nodes._point_source import MissingDimension, run_clean
from lynceus.nodes.ports import PortType

NODE_ID = "lynceus.nodes.lidar.clean.filter_qa_flags"
NODE_NAME = "Filter by QA Flags"
NODE_CATEGORY = "LiDAR"

NODE_SUBCATEGORY = "Cleaning"

NODE_DESCRIPTION = (
    "<b>Filter by QA Flags</b> -- Vendor Flag Culling<br><br>"
    "Removes points flagged by the vendor (flight-line overlap, synthetic) "
    "or keeps only the overlap strips so they can be processed separately, "
    "as the engine warning suggests for area metrics. Flagged points keep "
    "their real class (LAS 1.4 flags never destroy it), so this filters by "
    "flag, never by class.<br><br>"
    "<b>Tips:</b> Remove overlap before area metrics (density doubles in "
    "strips). Use only-overlap to audit or process the strips on their own "
    "branch. This node always removes (never tags): review on a sample "
    "before chaining deletes."
)

INPUTS = (PortType.POINT_CLOUD,)
OUTPUTS = (PortType.POINT_CLOUD,)

PROCESSING_SPECS = {
    "tile_task": "tile_filter_qa_flags",
    "input_port": "point_cloud",
    "output_port": "point_cloud",
    "passes_flags": ("classified",),
    "point_cloud_dir": "clean_flags",
    "output_globs": ("clean_flags/**/*.laz",),
    "config_schema": {
        "mode": {
            "type": "str",
            "default": "remove_overlap",
            "options": ["remove_overlap", "remove_synthetic", "remove_all_qa",
                        "only_overlap"],
            "description": "Which flagged points to remove.",
            "impact": "remove_overlap drops flight-line strips (density is doubled there); remove_synthetic drops post-processed points; remove_all_qa drops both; only_overlap keeps just the strips for separate processing. Dropped points are deleted permanently.",
            "group": "Flags",
        },
    },
}


def get_config_defaults() -> dict:
    return {k: v["default"] for k, v in PROCESSING_SPECS["config_schema"].items()}


def tile_filter_qa_flags(tile: dict, ctx: dict) -> dict:
    """Cull vendor-flagged points; drop them unconditionally."""
    import numpy as np

    def mask_fn(record) -> np.ndarray:
        dims = set(record.point_format.dimension_names)
        has_overlap = "overlap" in dims
        has_synthetic = "synthetic" in dims
        if not has_overlap and not has_synthetic:
            raise MissingDimension(
                "source has no overlap/synthetic dimensions; "
                "QA flag filtering skipped"
            )
        overlap = (np.asarray(record.overlap, dtype=bool) if has_overlap
                   else np.zeros(len(np.asarray(record.x)), dtype=bool))
        synthetic = (np.asarray(record.synthetic, dtype=bool) if has_synthetic
                     else np.zeros(len(np.asarray(record.x)), dtype=bool))
        mode = str(ctx.get("mode", "remove_overlap"))
        if mode == "remove_overlap":
            if not has_overlap:
                raise MissingDimension(
                    "source has no overlap dimension; QA flag filtering skipped"
                )
            return ~overlap
        if mode == "remove_synthetic":
            if not has_synthetic:
                raise MissingDimension(
                    "source has no synthetic dimension; QA flag filtering skipped"
                )
            return ~synthetic
        if mode == "remove_all_qa":
            return ~(overlap | synthetic)
        if mode == "only_overlap":
            if not has_overlap:
                raise MissingDimension(
                    "source has no overlap dimension; QA flag filtering skipped"
                )
            return overlap
        raise ValueError(f"{NODE_ID}: unknown mode {mode!r}")

    return run_clean(tile, ctx, NODE_ID, "clean_flags", mask_fn,
                     force_remove=True)
