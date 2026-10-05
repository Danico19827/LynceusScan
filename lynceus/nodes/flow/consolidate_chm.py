# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Variant CHM of the Consolidate strategy family (no NODE_ID)."""

from __future__ import annotations

from lynceus.nodes.flow._consolidate_base import consolidate_mosaic
from lynceus.nodes.ports import PortDef, PortType

VARIANT_OF = "lynceus.nodes.flow.consolidate"
VARIANT_KEY = "chm"
VARIANT_LABEL = "CHM"

INPUTS = (PortDef(PortType.CHM_MOSAIC),)
OUTPUTS = (PortType.CHM_MOSAIC,)

PROCESSING_SPECS = {
    "consolidator": True,
    "consolidate_task": "barrier_consolidate_chm",
    "output_files": {"chm_mosaic": "chm_mosaic.tif"},
    "output_globs": ("*/chm_mosaic.tif",),
    "qml": {"chm_mosaic.tif": "chm"},
    "config_schema": {
        "nodata_value": {
            "type": "float",
            "default": -9999.0,
            "minimum": -99999.0,
            "maximum": 99999.0,
            "description": "Value written to merged cells with no data.",
            "impact": "Cells without valid data in every segment keep this value, matching the convention of the provider mosaics.",
            "group": "Advanced",
            "advanced": True,
        },
    },
}


def barrier_consolidate_chm(ctx: dict) -> dict:
    """Merge per-segment CHM mosaics into one (aligned, NODATA preserved)."""
    payload = consolidate_mosaic(
        ctx, session_file="chm_mosaic.tif", input_type="chm_mosaic"
    )
    payload["display_name"] = "CHM"
    return payload
