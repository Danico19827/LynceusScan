# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Variant CSF of the Classify Ground strategy family (no NODE_ID)."""

from __future__ import annotations

from lynceus.nodes.ports import PortType

VARIANT_OF = "lynceus.nodes.lidar.terrain.classify_ground"
VARIANT_KEY = "csf"
VARIANT_LABEL = "CSF"

INPUTS = (PortType.POINT_CLOUD,)
OUTPUTS = (PortType.POINT_CLOUD,)

PROCESSING_SPECS = {
    "tile_task": "tile_classify_ground_csf",
    "input_port": "point_cloud",
    "output_port": "point_cloud",
    "point_cloud_dir": "classified",
    "provides": {"classified": True},
    "output_globs": ("classified/**/*.laz",),
    "config_schema": {
        "csf_resolution": {
            "type": "float",
            "default": 0.5,
            "minimum": 0.2,
            "maximum": 2.0,
            "description": "Cloth grid resolution (meters).",
            "impact": "Finer cloth follows small features but costs memory and time. Match or halve the point spacing.",
            "group": "CSF Algorithm",
        },
        "csf_threshold": {
            "type": "float",
            "default": 0.5,
            "minimum": 0.1,
            "maximum": 2.0,
            "description": "Maximum cloth-to-point distance for ground (meters).",
            "impact": "Lower values are stricter (cleaner ground, risk of stripping relief); higher values keep more points as ground.",
            "group": "CSF Algorithm",
        },
        "csf_rigidness": {
            "type": "int",
            "default": 3,
            "minimum": 1,
            "maximum": 3,
            "description": "Cloth stiffness: 1 flat, 2 relief, 3 steep.",
            "impact": "Higher values drape tighter over relief and keep steep ground; lower values bridge small objects better on flat terrain.",
            "group": "CSF Algorithm",
        },
        "csf_iterations": {
            "type": "int",
            "default": 300,
            "minimum": 50,
            "maximum": 1000,
            "description": "Settling iterations of the cloth simulation.",
            "impact": "More iterations settle complex scenes better at linear time cost. Rarely needs changing.",
            "group": "CSF Algorithm",
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
        "csf_resolution": 0.5,
        "csf_threshold": 0.5,
        "csf_rigidness": 3,
        "csf_iterations": 300,
        "output_class": 2,
    }


def tile_classify_ground_csf(tile: dict, ctx: dict) -> dict:
    """Process one tile: classify ground with CSF and write a classified LAZ."""
    from lynceus.nodes.lidar.terrain._classify_base import (
        run_ground_classification,
    )
    from lynceus.processing.raster import cloth_simulation_filter

    def mask(x, y, z):
        return cloth_simulation_filter(
            x,
            y,
            z,
            cell_size=float(ctx.get("csf_resolution", 0.5)),
            class_threshold=float(ctx.get("csf_threshold", 0.5)),
            rigidness=int(ctx.get("csf_rigidness", 3)),
            iterations=int(ctx.get("csf_iterations", 300)),
            memory_budget_bytes=ctx.get("_raster_grid_memory_budget_bytes"),
            tile_id=tile.get("tile_id"),
        )

    return run_ground_classification(tile, ctx, mask, node_id=_FAMILY_ID)
