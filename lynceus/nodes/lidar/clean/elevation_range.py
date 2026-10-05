# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Node -- Elevation Range filter.

Keeps points whose Z falls inside [min_elevation, max_elevation]. Outliers
below the terrain (e.g. birds flying low, dropdown artifacts) and above the
survey (isolated wires/signals) are either tagged as noise (class 7,
non-destructive) or physically removed depending on ``action``.
"""

from __future__ import annotations

from lynceus.nodes._point_source import run_clean
from lynceus.nodes.ports import PortType

NODE_ID = "lynceus.nodes.lidar.clean.elevation_range"
NODE_NAME = "Elevation Range"
NODE_CATEGORY = "LiDAR"

NODE_SUBCATEGORY = "Cleaning"

NODE_DESCRIPTION = (
    "<b>Elevation Range</b> -- Z Outlier Filter<br><br>"
    "Keeps points with an elevation inside the configured range, tagging or "
    "removing outliers: isolated low points below the terrain and high "
    "returns above the survey (noise, birds, cables close to the sensor).<br><br>"
    "<b>Tips:</b> Set <i>min elevation</i> slightly below the lowest expected "
    "ground (drop artifacts sit below the real surface). Adjust <i>max "
    "elevation</i> to the tallest legitimate feature (tree tops, buildings). "
    "<i>classify</i> (default) is reversible and never deletes ground; use "
    "<i>remove</i> only after reviewing the tagged classes."
)

INPUTS = (PortType.POINT_CLOUD,)
OUTPUTS = (PortType.POINT_CLOUD,)

PROCESSING_SPECS = {
    "tile_task": "tile_elevation_range",
    "input_port": "point_cloud",
    "output_port": "point_cloud",
    "passes_flags": ("classified",),
    "point_cloud_dir": "clean_elevation",
    "output_globs": ("clean_elevation/**/*.laz",),
    "config_schema": {
        "min_elevation": {
            "type": "float",
            "default": -100.0,
            "minimum": -100000.0,
            "maximum": 100000.0,
            "description": "Lowest accepted elevation (meters). Points below are outliers.",
            "impact": "Lower values keep more points but let drop-artifacts and low noise survive. Raise it to cut isolated points below the true terrain.",
            "group": "Range",
        },
        "max_elevation": {
            "type": "float",
            "default": 100000.0,
            "minimum": -100000.0,
            "maximum": 100000.0,
            "description": "Highest accepted elevation (meters). Points above are outliers.",
            "impact": "Lower values remove high returns (noise, birds, near-sensor signals) but also real features like tall trees or building antennae. Set it to the tallest legitimate object in the survey.",
            "group": "Range",
        },
        "action": {
            "type": "str",
            "default": "classify",
            "options": ["classify", "remove"],
            "description": "How to handle out-of-range points.",
            "impact": "classify (default) tags them with the target class keeping the cloud complete and reversible; ground (class 2) is never overwritten while protect ground is on. remove physically deletes the points, permanently changing point counts.",
            "group": "Behavior",
        },
        "protect_ground": {
            "type": "bool",
            "default": True,
            "description": "Keep ground points (class 2) untouched when tagging.",
            "impact": "On (default) preserves current behavior: ground is never retagged, even outside the range. Off lets the node retag ground as noise so a later remove can delete it — bare-earth products lose those points and degrade with a warning.",
            "group": "Behavior",
        },
        "target_class": {
            "type": "int",
            "default": 7,
            "minimum": 0,
            "maximum": 31,
            "description": "LAS class assigned to out-of-range points when action=classify.",
            "impact": "ASPRS reserves class 7 (low point/noise) for noise and 18 (high noise) for high isolated returns. Any non-ground class will be overwritten, including water, buildings and vegetation.",
            "group": "Output",
        },
    },
}


def get_config_defaults() -> dict:
    return {k: v["default"] for k, v in PROCESSING_SPECS["config_schema"].items()}


def tile_elevation_range(tile: dict, ctx: dict) -> dict:
    """Filter points by elevation (Z outlier removal or tagging)."""
    import numpy as np

    def mask_fn(record) -> np.ndarray:
        z = np.asarray(record.z)
        lo = ctx.get("min_elevation", -100.0)
        hi = ctx.get("max_elevation", 100000.0)
        return (z >= lo) & (z <= hi)

    return run_clean(
        tile, ctx, NODE_ID, "clean_elevation", mask_fn,
        cls_value=int(ctx.get("target_class", 7)),
        protect_ground=bool(ctx.get("protect_ground", True)),
    )