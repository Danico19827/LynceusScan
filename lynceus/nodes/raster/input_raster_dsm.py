# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Variant DSM of the Input Raster strategy family (no NODE_ID)."""

from __future__ import annotations

from lynceus.nodes._product_input import barrier_input_file
from lynceus.nodes.ports import PortType

VARIANT_OF = "lynceus.nodes.raster.input_raster"
VARIANT_KEY = "dsm"
VARIANT_LABEL = "DSM"

VARIANT_TARGET = {
    "node_id": "lynceus.nodes.raster.input_raster",
    "node_name": "Input DSM",
    "output_port": "dsm_mosaic",
    "session_file": "dsm_mosaic.tif",
    "kind_label": "DSM",
    "kind_group": "raster",
    "file_filters": "Raster files (*.tif *.tiff)",
}

INPUTS = ()
OUTPUTS = (PortType.DSM_MOSAIC,)

PROCESSING_SPECS = {
    "barrier_task": "barrier_input_raster_dsm",
    "output_port": "dsm_mosaic",
    "output_globs": ("*/dsm_mosaic.tif",),
    "input_file_key": "file_path",
    "session_file": "dsm_mosaic.tif",
    "qml": {"dsm_mosaic.tif": "dsm"},
}


def barrier_input_raster_dsm(ctx: dict) -> dict:
    """Import an existing DSM GeoTIFF as the session's canonical DSM mosaic."""
    return barrier_input_file(ctx, VARIANT_TARGET)