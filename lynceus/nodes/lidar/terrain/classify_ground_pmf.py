# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Variant PMF of the Classify Ground strategy family (no NODE_ID)."""

from __future__ import annotations

from lynceus.nodes.ports import PortType

VARIANT_OF = "lynceus.nodes.lidar.terrain.classify_ground"
VARIANT_KEY = "pmf"
VARIANT_LABEL = "PMF"

INPUTS = (PortType.POINT_CLOUD,)
OUTPUTS = (PortType.POINT_CLOUD,)

PROCESSING_SPECS = {
    "tile_task": "tile_classify_ground_pmf",
    "input_port": "point_cloud",
    "output_port": "point_cloud",
    "point_cloud_dir": "classified",
    "provides": {"classified": True},
    "output_globs": ("classified/**/*.laz",),
    "config_schema": {
        "pmf_slope": {
            "type": "float",
            "default": 0.3,
            "minimum": 0.1,
            "maximum": 1.0,
            "description": "Slope threshold for adaptive window growth.",
            "impact": "Controls how aggressively the filter adapts to terrain slope. Lower values are stricter (may miss ground on steep slopes). Higher values tolerate steeper slopes but may misclassify low vegetation as ground.",
            "group": "PMF Algorithm",
        },
        "pmf_intercept": {
            "type": "float",
            "default": 0.3,
            "minimum": 0.1,
            "maximum": 1.0,
            "description": "Intercept threshold for window growth equation.",
            "impact": "Sets the minimum height difference needed to trigger window growth. Lower values allow the filter to grow windows sooner (more aggressive ground detection). Higher values require larger height differences.",
            "group": "PMF Algorithm",
        },
        "pmf_max_window_m": {
            "type": "float",
            "default": 8.0,
            "minimum": 4,
            "maximum": 20,
            "description": "Maximum filter window size (meters).",
            "impact": "Larger values allow the filter to remove taller objects (multi-story buildings, tall trees) but increase processing time and risk over-smoothing on complex slopes.",
            "group": "PMF Algorithm",
        },
        "pmf_initial_window_m": {
            "type": "float",
            "default": 2.0,
            "minimum": 1,
            "maximum": 5,
            "description": "Initial filter window size (meters).",
            "impact": "The starting window size for the progressive filter. Smaller values are more aggressive on small features. Larger values skip fine detail and only remove bigger objects.",
            "group": "PMF Algorithm",
        },
        "pmf_cell_size": {
            "type": "float",
            "default": 1.0,
            "minimum": 0.5,
            "maximum": 5.0,
            "description": "Grid cell size (meters) for the internal PMF raster.",
            "impact": "Smaller cells capture more terrain detail but increase memory and processing time. A good rule is to match or halve the point spacing.",
            "group": "Grid",
        },
        "output_class": {
            "type": "int",
            "default": 2,
            "minimum": 1,
            "maximum": 32,
            "description": "LAS classification code assigned to ground points.",
            "impact": "Standard value is 2 (ground). Some workflows use different codes (e.g. 8 for ground in combined datasets).",
            "group": "Output",
        },
    },
}

_FAMILY_ID = "lynceus.nodes.lidar.terrain.classify_ground"


def get_config_defaults() -> dict:
    return {
        "pmf_slope": 0.3,
        "pmf_intercept": 0.3,
        "pmf_max_window_m": 8.0,
        "pmf_initial_window_m": 2.0,
        "pmf_cell_size": 1.0,
        "output_class": 2,
    }


def tile_classify_ground_pmf(tile: dict, ctx: dict) -> dict:
    """Process one tile: classify ground with PMF and write a classified LAZ."""
    from lynceus.nodes.lidar.terrain._classify_base import (
        run_ground_classification,
    )
    from lynceus.processing.raster import progressive_morphological_filter

    def mask(x, y, z):
        return progressive_morphological_filter(
            x,
            y,
            z,
            cell_size=float(ctx.get("pmf_cell_size", ctx.get("cell_size", 1.0))),
            slope=float(ctx.get("pmf_slope", 0.3)),
            intercept=float(ctx.get("pmf_intercept", 0.3)),
            initial_window_m=float(ctx.get("pmf_initial_window_m", 2)),
            max_window_m=float(ctx.get("pmf_max_window_m", 8)),
            memory_budget_bytes=ctx.get("_raster_grid_memory_budget_bytes"),
            tile_id=tile.get("tile_id"),
        )

    return run_ground_classification(tile, ctx, mask, node_id=_FAMILY_ID)
