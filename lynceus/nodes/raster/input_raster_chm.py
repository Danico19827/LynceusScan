# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Variant CHM of the Input Raster strategy family (no NODE_ID)."""

from __future__ import annotations

from lynceus.nodes._product_input import barrier_input_file
from lynceus.nodes.ports import PortType

VARIANT_OF = "lynceus.nodes.raster.input_raster"
VARIANT_KEY = "chm"
VARIANT_LABEL = "CHM"

VARIANT_TARGET = {
    "node_id": "lynceus.nodes.raster.input_raster",
    "node_name": "Input CHM",
    "output_port": "chm_mosaic",
    "session_file": "chm_mosaic.tif",
    "kind_label": "CHM",
    "kind_group": "raster",
    "file_filters": "Raster files (*.tif *.tiff)",
}

INPUTS = ()
OUTPUTS = (PortType.CHM_MOSAIC,)

PROCESSING_SPECS = {
    "barrier_task": "barrier_input_raster_chm",
    "output_port": "chm_mosaic",
    "output_globs": ("*/chm_mosaic.tif",),
    "input_file_key": "file_path",
    "session_file": "chm_mosaic.tif",
    "qml": {"chm_mosaic.tif": "chm"},
}


def barrier_input_raster_chm(ctx: dict) -> dict:
    """Import an existing CHM GeoTIFF as the session's canonical CHM mosaic."""
    return barrier_input_file(ctx, VARIANT_TARGET)