# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Variant DTM of the Consolidate strategy family (no NODE_ID)."""

from __future__ import annotations

from lynceus.nodes.flow._consolidate_base import consolidate_mosaic
from lynceus.nodes.ports import PortDef, PortType

VARIANT_OF = "lynceus.nodes.flow.consolidate"
VARIANT_KEY = "dtm"
VARIANT_LABEL = "DTM"

INPUTS = (PortDef(PortType.DTM_MOSAIC),)
OUTPUTS = (PortType.DTM_MOSAIC,)

PROCESSING_SPECS = {
    "consolidator": True,
    "consolidate_task": "barrier_consolidate_dtm",
    "output_files": {"dtm_mosaic": "dtm_mosaic.tif"},
    "output_globs": ("*/dtm_mosaic.tif",),
    "qml": {"dtm_mosaic.tif": "dtm"},
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


def barrier_consolidate_dtm(ctx: dict) -> dict:
    """Merge per-segment DTM mosaics into one (aligned, NODATA preserved)."""
    payload = consolidate_mosaic(
        ctx, session_file="dtm_mosaic.tif", input_type="dtm_mosaic"
    )
    payload["display_name"] = "DTM"
    return payload
