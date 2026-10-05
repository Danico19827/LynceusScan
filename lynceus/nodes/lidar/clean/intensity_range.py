# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Node -- Intensity Range filter.

Keeps points whose intensity falls inside [min_intensity, max_intensity].
Very low returns (weak backscatter, mostly noise far from the sensor) and
very high saturated returns can be tagged as noise or removed.
"""

from __future__ import annotations

from lynceus.nodes._point_source import run_clean
from lynceus.nodes.ports import PortType

NODE_ID = "lynceus.nodes.lidar.clean.intensity_range"
NODE_NAME = "Intensity Range"
NODE_CATEGORY = "LiDAR"

NODE_SUBCATEGORY = "Cleaning"

NODE_DESCRIPTION = (
    "<b>Intensity Range</b> -- Return Strength Filter<br><br>"
    "Keeps points whose intensity (returned signal strength) falls inside "
    "the configured range. Extremely weak returns are usually atmospheric "
    "noise or edge hits with unreliable geometry; saturated returns "
    "may bias metrics on reflective surfaces.<br><br>"
    "<b>Tips:</b> Inspect the intensity histogram first (Intensity Ortho "
    "viewer) and set the bounds slightly inside the real signal peak. "
    "Over-aggressive filtering of low intensity can remove real returns from "
    "dark or wet surfaces. <i>classify</i> (default) keeps the cloud intact."
)

INPUTS = (PortType.POINT_CLOUD,)
OUTPUTS = (PortType.POINT_CLOUD,)

PROCESSING_SPECS = {
    "tile_task": "tile_intensity_range",
    "input_port": "point_cloud",
    "output_port": "point_cloud",
    "passes_flags": ("classified",),
    "point_cloud_dir": "clean_intensity",
    "output_globs": ("clean_intensity/**/*.laz",),
    "config_schema": {
        "min_intensity": {
            "type": "int",
            "default": 0,
            "minimum": 0,
            "maximum": 65535,
            "description": "Lowest accepted intensity (counts). Points below are tagged.",
            "impact": "Raises to cut atmospheric noise and weak edge returns. Be careful: dark surfaces (asphalt, water) legitimately return low intensity.",
            "group": "Range",
        },
        "max_intensity": {
            "type": "int",
            "default": 65535,
            "minimum": 0,
            "maximum": 65535,
            "description": "Highest accepted intensity (counts). Points above are tagged.",
            "impact": "Lowers to remove saturated returns that compress into a single reflectance bin, which can bias intensity-based metrics.",
            "group": "Range",
        },
        "action": {
            "type": "str",
            "default": "classify",
            "options": ["classify", "remove"],
            "description": "How to handle out-of-range points.",
            "impact": "classify (default) tags them with the target class keeping the cloud complete and reversible; ground (class 2) is never overwritten while protect ground is on. remove physically deletes the points.",
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
            "impact": "Class 7 (low point/noise) is the standard tag for intensity outliers.",
            "group": "Output",
        },
    },
}


def get_config_defaults() -> dict:
    return {k: v["default"] for k, v in PROCESSING_SPECS["config_schema"].items()}


def tile_intensity_range(tile: dict, ctx: dict) -> dict:
    """Filter points by intensity (weak/saturated return tagging or removal)."""
    import numpy as np

    def mask_fn(record) -> np.ndarray:
        # Native uint16 compare: no int64 upcast (4x bandwidth for nothing).
        intensity = np.asarray(record.intensity)
        lo = int(ctx.get("min_intensity", 0))
        hi = int(ctx.get("max_intensity", 65535))
        lo = max(0, min(65535, lo))
        hi = max(0, min(65535, hi))
        return (intensity >= lo) & (intensity <= hi)

    return run_clean(
        tile, ctx, NODE_ID, "clean_intensity", mask_fn,
        cls_value=int(ctx.get("target_class", 7)),
        protect_ground=bool(ctx.get("protect_ground", True)),
    )