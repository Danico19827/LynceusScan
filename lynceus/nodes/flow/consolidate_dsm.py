# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Variant DSM of the Consolidate strategy family (no NODE_ID)."""

from __future__ import annotations

from lynceus.nodes.flow._consolidate_base import consolidate_mosaic
from lynceus.nodes.ports import PortDef, PortType

VARIANT_OF = "lynceus.nodes.flow.consolidate"
VARIANT_KEY = "dsm"
VARIANT_LABEL = "DSM"

INPUTS = (PortDef(PortType.DSM_MOSAIC),)
OUTPUTS = (PortType.DSM_MOSAIC,)

PROCESSING_SPECS = {
    "consolidator": True,
    "consolidate_task": "barrier_consolidate_dsm",
    "output_files": {"dsm_mosaic": "dsm_mosaic.tif"},
    "output_globs": ("*/dsm_mosaic.tif",),
    "qml": {"dsm_mosaic.tif": "dsm"},
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


def barrier_consolidate_dsm(ctx: dict) -> dict:
    """Merge per-segment DSM mosaics into one (aligned, NODATA preserved)."""
    payload = consolidate_mosaic(
        ctx, session_file="dsm_mosaic.tif", input_type="dsm_mosaic"
    )
    payload["display_name"] = "DSM"
    return payload
