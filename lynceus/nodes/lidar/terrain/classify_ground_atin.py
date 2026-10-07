# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Variant ATIN of the Classify Ground strategy family (no NODE_ID)."""

from __future__ import annotations

from lynceus.nodes.ports import PortType

VARIANT_OF = "lynceus.nodes.lidar.terrain.classify_ground"
VARIANT_KEY = "atin"
VARIANT_LABEL = "ATIN"

INPUTS = (PortType.POINT_CLOUD,)
OUTPUTS = (PortType.POINT_CLOUD,)

PROCESSING_SPECS = {
    "tile_task": "tile_classify_ground_atin",
    "input_port": "point_cloud",
    "output_port": "point_cloud",
    "point_cloud_dir": "classified",
    "provides": {"classified": True},
    "output_globs": ("classified/**/*.laz",),
    "config_schema": {
        "atin_seed_m": {
            "type": "float",
            "default": 20.0,
            "minimum": 5.0,
            "maximum": 50.0,
            "description": "Seed spacing (meters) for the initial TIN.",
            "impact": "Must exceed the largest building: blocks without a ground seed grow a roof into the surface. Smaller spacing densifies detail but costs time and risks roof seeds.",
            "group": "ATIN Algorithm",
        },
        "atin_max_angle": {
            "type": "float",
            "default": 6.0,
            "minimum": 1.0,
            "maximum": 30.0,
            "description": "Maximum facet angle (degrees) for densification.",
            "impact": "Cells seen under a steeper sightline from the facet vertices stay non-ground. Lower values are stricter on relief edges; higher values follow rough terrain but may keep low vegetation.",
            "group": "ATIN Algorithm",
        },
        "atin_max_dist": {
            "type": "float",
            "default": 1.0,
            "minimum": 0.2,
            "maximum": 5.0,
            "description": "Maximum facet distance (meters) for densification.",
            "impact": "Cells further above the facet plane stay non-ground, and the final labeling band uses the same value. Match it to the relief noise floor.",
            "group": "ATIN Algorithm",
        },
        "atin_iterations": {
            "type": "int",
            "default": 10,
            "minimum": 1,
            "maximum": 50,
            "description": "Maximum TIN densification passes.",
            "impact": "Stops early when no cell joins. Higher caps only matter on difficult tiles; they never change converged results.",
            "group": "ATIN Algorithm",
        },
        "atin_cell_size": {
            "type": "float",
            "default": 1.0,
            "minimum": 0.5,
            "maximum": 5.0,
            "description": "Grid cell size (meters) for the internal ATIN surface.",
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
        "atin_seed_m": 20.0,
        "atin_max_angle": 6.0,
        "atin_max_dist": 1.0,
        "atin_iterations": 10,
        "atin_cell_size": 1.0,
        "output_class": 2,
    }


def tile_classify_ground_atin(tile: dict, ctx: dict) -> dict:
    """Process one tile: classify ground with ATIN and write classified LAZ."""
    from lynceus.nodes.lidar.terrain._classify_base import (
        run_ground_classification,
    )
    from lynceus.processing.raster import adaptive_tin_filter

    def mask(x, y, z):
        return adaptive_tin_filter(
            x,
            y,
            z,
            cell_size=float(ctx.get("atin_cell_size", ctx.get("cell_size", 1.0))),
            seed_m=float(ctx.get("atin_seed_m", 20.0)),
            max_angle_deg=float(ctx.get("atin_max_angle", 6.0)),
            max_dist_m=float(ctx.get("atin_max_dist", 1.0)),
            max_iterations=int(ctx.get("atin_iterations", 10)),
            memory_budget_bytes=ctx.get("_raster_grid_memory_budget_bytes"),
            tile_id=tile.get("tile_id"),
        )

    return run_ground_classification(tile, ctx, mask, node_id=_FAMILY_ID)
