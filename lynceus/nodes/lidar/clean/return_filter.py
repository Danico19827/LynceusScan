# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Node -- Filter by Return (keep by pulse position).

Keeps only the selected returns of each pulse and physically deletes the
rest. Return counts come from the file itself (return_number /
number_of_returns on LAS 1.4, return_num / num_returns on legacy
formats, auto-detected); sources without return dimensions pass through
with a warning instead of breaking the run.
"""

from __future__ import annotations

from lynceus.nodes._point_source import MissingDimension, run_clean
from lynceus.nodes.ports import PortType

NODE_ID = "lynceus.nodes.lidar.clean.return_filter"
NODE_NAME = "Filter by Return"
NODE_CATEGORY = "LiDAR"

NODE_SUBCATEGORY = "Cleaning"

NODE_DESCRIPTION = (
    "<b>Filter by Return</b> -- Keep by Pulse Position<br><br>"
    "Keeps only the selected returns of each pulse (first/last/single/"
    "multiple) and deletes the rest permanently. Single-return pulses are "
    "solid surfaces; multiple returns mark porous targets (canopy, wires, "
    "fences).<br><br>"
    "<b>Tips:</b> First returns for surfaces and canopy tops, last returns "
    "for ground under vegetation. This node always removes (never tags): "
    "review on a sample before chaining deletes."
)

INPUTS = (PortType.POINT_CLOUD,)
OUTPUTS = (PortType.POINT_CLOUD,)

PROCESSING_SPECS = {
    "tile_task": "tile_return_filter",
    "input_port": "point_cloud",
    "output_port": "point_cloud",
    "passes_flags": ("classified",),
    "point_cloud_dir": "clean_returns",
    "output_globs": ("clean_returns/**/*.laz",),
    "config_schema": {
        "return_mode": {
            "type": "str",
            "default": "first",
            "options": ["first", "last", "single", "multiple"],
            "description": "Which returns to keep by pulse position.",
            "impact": "First returns outline surfaces and canopy tops; last returns reach the ground under vegetation. Single keeps only solid-surface pulses; multiple keeps only porous targets (canopy, wires). Dropped points are deleted permanently.",
            "group": "Returns",
        },
    },
}


def get_config_defaults() -> dict:
    return {k: v["default"] for k, v in PROCESSING_SPECS["config_schema"].items()}


def tile_return_filter(tile: dict, ctx: dict) -> dict:
    """Keep selected returns per pulse; drop the rest unconditionally."""
    import numpy as np

    def mask_fn(record) -> np.ndarray:
        dims = set(record.point_format.dimension_names)
        if "return_number" in dims and "number_of_returns" in dims:
            ret = np.asarray(record.return_number, dtype=np.int64)
            num = np.asarray(record.number_of_returns, dtype=np.int64)
        elif "return_num" in dims and "num_returns" in dims:
            # Legacy PDRF 0-5 carry the same pulse position under
            # shorter dimension names (auto-detected like scan_angle).
            ret = np.asarray(record.return_num, dtype=np.int64)
            num = np.asarray(record.num_returns, dtype=np.int64)
        else:
            raise MissingDimension(
                "source has no return dimensions; return filtering skipped"
            )
        mode = str(ctx.get("return_mode", "first"))
        if mode == "first":
            return ret == 1
        if mode == "last":
            return ret == num
        if mode == "single":
            return num == 1
        if mode == "multiple":
            return num > 1
        raise ValueError(f"{NODE_ID}: unknown return_mode {mode!r}")

    return run_clean(tile, ctx, NODE_ID, "clean_returns", mask_fn,
                     force_remove=True)
