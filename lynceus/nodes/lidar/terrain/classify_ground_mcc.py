# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Variant MCC of the Classify Ground strategy family (no NODE_ID)."""

from __future__ import annotations

from lynceus.nodes.ports import PortType

VARIANT_OF = "lynceus.nodes.lidar.terrain.classify_ground"
VARIANT_KEY = "mcc"
VARIANT_LABEL = "MCC"

INPUTS = (PortType.POINT_CLOUD,)
OUTPUTS = (PortType.POINT_CLOUD,)

PROCESSING_SPECS = {
    "tile_task": "tile_classify_ground_mcc",
    "input_port": "point_cloud",
    "output_port": "point_cloud",
    "point_cloud_dir": "classified",
    "provides": {"classified": True},
    "output_globs": ("classified/**/*.laz",),
    "config_schema": {
        "mcc_scale": {
            "type": "float",
            "default": 1.5,
            "minimum": 0.5,
            "maximum": 5.0,
            "description": "Thin-plate scale (meters) for the interpolated surface.",
            "impact": "Larger values fit a stiffer plate that bridges bigger objects (tall trees, buildings) but may skip sharp ridges. Match it to the size of the objects on the ground.",
            "group": "MCC Algorithm",
        },
        "mcc_threshold": {
            "type": "float",
            "default": 0.3,
            "minimum": 0.1,
            "maximum": 1.0,
            "description": "Curvature threshold (meters) above the plate.",
            "impact": "Points higher than this above the interpolated surface are non-ground. Lower values remove more (risk on rough terrain); higher values keep more (risk under dense canopy).",
            "group": "MCC Algorithm",
        },
        "mcc_domains": {
            "type": "int",
            "default": 3,
            "minimum": 1,
            "maximum": 5,
            "description": "Number of scale domains (coarsening passes).",
            "impact": "Each extra domain refits a stiffer plate over a wider area, catching larger objects. More domains cost time; 3 follows the reference paper.",
            "group": "MCC Algorithm",
        },
        "mcc_iterations": {
            "type": "int",
            "default": 10,
            "minimum": 1,
            "maximum": 50,
            "description": "Maximum refit iterations per scale domain.",
            "impact": "The surface refits until fewer than 0.5% of cells change. Higher caps only matter on difficult tiles; they never change converged results.",
            "group": "MCC Algorithm",
        },
        "mcc_cell_size": {
            "type": "float",
            "default": 1.0,
            "minimum": 0.5,
            "maximum": 5.0,
            "description": "Grid cell size (meters) for the internal MCC surface.",
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
        "mcc_scale": 1.5,
        "mcc_threshold": 0.3,
        "mcc_domains": 3,
        "mcc_iterations": 10,
        "mcc_cell_size": 1.0,
        "output_class": 2,
    }


def tile_classify_ground_mcc(tile: dict, ctx: dict) -> dict:
    """Process one tile: classify ground with MCC and write a classified LAZ."""
    from lynceus.nodes.lidar.terrain._classify_base import (
        run_ground_classification,
    )
    from lynceus.processing.raster import multiscale_curvature_filter

    def mask(x, y, z):
        return multiscale_curvature_filter(
            x,
            y,
            z,
            cell_size=float(ctx.get("mcc_cell_size", ctx.get("cell_size", 1.0))),
            scale=float(ctx.get("mcc_scale", 1.5)),
            threshold=float(ctx.get("mcc_threshold", 0.3)),
            domains=int(ctx.get("mcc_domains", 3)),
            max_iterations=int(ctx.get("mcc_iterations", 10)),
            memory_budget_bytes=ctx.get("_raster_grid_memory_budget_bytes"),
            tile_id=tile.get("tile_id"),
        )

    return run_ground_classification(tile, ctx, mask, node_id=_FAMILY_ID)
