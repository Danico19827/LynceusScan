# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Variant SMRF of the Classify Ground strategy family (no NODE_ID)."""

from __future__ import annotations

from lynceus.nodes.ports import PortType

VARIANT_OF = "lynceus.nodes.lidar.terrain.classify_ground"
VARIANT_KEY = "smrf"
VARIANT_LABEL = "SMRF"

INPUTS = (PortType.POINT_CLOUD,)
OUTPUTS = (PortType.POINT_CLOUD,)

PROCESSING_SPECS = {
    "tile_task": "tile_classify_ground_smrf",
    "input_port": "point_cloud",
    "output_port": "point_cloud",
    "point_cloud_dir": "classified",
    "provides": {"classified": True},
    "output_globs": ("classified/**/*.laz",),
    "config_schema": {
        "smrf_cell_size": {
            "type": "float",
            "default": 1.0,
            "minimum": 0.5,
            "maximum": 5.0,
            "description": "Grid cell size (meters) for the internal surface.",
            "impact": "Smaller cells capture more terrain detail but increase memory and processing time. A good rule is to match or halve the point spacing.",
            "group": "Grid",
        },
        "smrf_slope": {
            "type": "float",
            "default": 0.2,
            "minimum": 0.05,
            "maximum": 1.0,
            "description": "Reference slope for adaptive window growth.",
            "impact": "Lower values test gentle cells earlier (aggressive on flat/agricultural ground). Steep relief is stripped by design here: use PMF or CSF for slopes.",
            "group": "SMRF Algorithm",
        },
        "smrf_window_m": {
            "type": "float",
            "default": 18.0,
            "minimum": 4,
            "maximum": 40,
            "description": "Maximum filter window size (meters).",
            "impact": "Larger values remove bigger objects but cost time and may bridge real relief on complex slopes.",
            "group": "SMRF Algorithm",
        },
        "smrf_threshold": {
            "type": "float",
            "default": 0.4,
            "minimum": 0.1,
            "maximum": 2.0,
            "description": "Fixed elevation threshold (meters) at every scale.",
            "impact": "Unlike PMF this threshold does not grow with the window: lower values strip more aggressively at all scales.",
            "group": "SMRF Algorithm",
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
        "smrf_cell_size": 1.0,
        "smrf_slope": 0.2,
        "smrf_window_m": 18.0,
        "smrf_threshold": 0.4,
        "output_class": 2,
    }


def tile_classify_ground_smrf(tile: dict, ctx: dict) -> dict:
    """Process one tile: classify ground with SMRF and write a classified LAZ."""
    from lynceus.nodes.lidar.terrain._classify_base import (
        run_ground_classification,
    )
    from lynceus.processing.raster import simple_morphological_filter

    def mask(x, y, z):
        return simple_morphological_filter(
            x,
            y,
            z,
            cell_size=float(ctx.get("smrf_cell_size", ctx.get("cell_size", 1.0))),
            slope=float(ctx.get("smrf_slope", 0.2)),
            max_window_m=float(ctx.get("smrf_window_m", 18)),
            threshold=float(ctx.get("smrf_threshold", 0.4)),
            memory_budget_bytes=ctx.get("_raster_grid_memory_budget_bytes"),
            tile_id=tile.get("tile_id"),
        )

    return run_ground_classification(tile, ctx, mask, node_id=_FAMILY_ID)
