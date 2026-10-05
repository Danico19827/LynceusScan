# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Node -- Scan Angle filter.

Keeps points whose scan angle (off-nadir degrees) falls inside the
configured range. High scan angles produce stretched, low-signal returns
with poor vertical accuracy at the tile edges and are prime terrain-noise
candidates when the flight was acquired with a wide field of view.
"""

from __future__ import annotations

from lynceus.nodes._point_source import MissingDimension, run_clean
from lynceus.nodes.ports import PortType

NODE_ID = "lynceus.nodes.lidar.clean.scan_angle"
NODE_NAME = "Scan Angle"
NODE_CATEGORY = "LiDAR"

NODE_SUBCATEGORY = "Cleaning"

NODE_DESCRIPTION = (
    "<b>Scan Angle</b> -- Off-Nadir Return Filter<br><br>"
    "Keeps points acquired within the configured scan angle range (degrees "
    "from nadir). Wide-angle returns have stretched footprints, weaker "
    "intensity and worse vertical precision, so trimming the extremes "
    "improves ground classification and surface quality.<br><br>"
    "<b>Tips:</b> Typical airborne LiDAR keeps angles within +/-25 to +/-40 "
    "degrees. The dimension is auto-detected as <i>scan_angle</i> (LAS 1.4) "
    "or <i>scan_angle_rank</i> (older formats). If the source has no scan "
    "angle dimension the node passes points through and warns."
)

INPUTS = (PortType.POINT_CLOUD,)
OUTPUTS = (PortType.POINT_CLOUD,)

PROCESSING_SPECS = {
    "tile_task": "tile_scan_angle",
    "input_port": "point_cloud",
    "output_port": "point_cloud",
    "passes_flags": ("classified",),
    "point_cloud_dir": "clean_scan",
    "output_globs": ("clean_scan/**/*.laz",),
    "config_schema": {
        "min_angle": {
            "type": "float",
            "default": -90.0,
            "minimum": -180.0,
            "maximum": 180.0,
            "description": "Lowest accepted scan angle (degrees from nadir).",
            "impact": "Raising towards 0 keeps only near-nadir returns, best vertical accuracy but sparser edges. Leave very negative for wide-angle acquisitions.",
            "group": "Range",
        },
        "max_angle": {
            "type": "float",
            "default": 90.0,
            "minimum": -180.0,
            "maximum": 180.0,
            "description": "Highest accepted scan angle (degrees from nadir).",
            "impact": "Tightening towards 0 removes stretched wide-angle returns at the swath edges, improving classification quality but reducing edge coverage.",
            "group": "Range",
        },
        "action": {
            "type": "str",
            "default": "classify",
            "options": ["classify", "remove"],
            "description": "How to handle out-of-range points.",
            "impact": "classify (default) tags them with the target class keeping the cloud complete and reversible. Ground (class 2) is never overwritten while protect ground is on. remove physically deletes the points.",
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
            "impact": "Class 7 (low point/noise) tags returns outside the angle window.",
            "group": "Output",
        },
    },
}


def get_config_defaults() -> dict:
    return {k: v["default"] for k, v in PROCESSING_SPECS["config_schema"].items()}


def tile_scan_angle(tile: dict, ctx: dict) -> dict:
    """Filter points by scan angle (off-nadir trimming or tagging)."""
    import numpy as np

    def mask_fn(record) -> np.ndarray:
        dims = set(record.point_format.dimension_names)
        if "scan_angle" in dims:
            raw = np.asarray(record.scan_angle, dtype=np.float32)
            if raw.size == 0:
                return np.zeros(0, dtype=bool)
            # Scale decision on a sample: a full-tile nanmax costs extra
            # passes, and a single wild value must not rescale the tile.
            # (A wild value outside the sample is treated as degrees and
            # may be tagged — conservative and documented.)
            sample = raw[:min(raw.size, 1 << 16)]
            if np.nanmax(np.abs(sample)) <= 90.0:
                angle = raw
            else:
                angle = raw * np.float32(0.006)
        elif "scan_angle_rank" in dims:
            angle = np.asarray(record.scan_angle_rank, dtype=np.float32)
        else:
            raise MissingDimension(
                "source has no scan_angle/scan_angle_rank dimension; "
                "points passed through unchanged"
            )
        lo = ctx.get("min_angle", -90.0)
        hi = ctx.get("max_angle", 90.0)
        return (angle >= lo) & (angle <= hi)

    return run_clean(
        tile, ctx, NODE_ID, "clean_scan", mask_fn,
        cls_value=int(ctx.get("target_class", 7)),
        protect_ground=bool(ctx.get("protect_ground", True)),
    )