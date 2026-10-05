# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Variant Multispectral of the Input Raster strategy family (no NODE_ID)."""

from __future__ import annotations

from lynceus.nodes._product_input import barrier_input_file
from lynceus.nodes.ports import PortType

VARIANT_OF = "lynceus.nodes.raster.input_raster"
VARIANT_KEY = "multispectral"
VARIANT_LABEL = "Multispectral"

VARIANT_TARGET = {
    "node_id": "lynceus.nodes.raster.input_raster",
    "node_name": "Input Multispectral",
    "output_port": "raster",
    "session_file": "multispectral.tif",
    "kind_label": "Multispectral",
    "kind_group": "multispectral",
    "file_filters": "Raster files (*.tif *.tiff)",
}

INPUTS = ()
OUTPUTS = (PortType.RASTER,)

PROCESSING_SPECS = {
    "barrier_task": "barrier_input_raster_multispectral",
    "output_port": "raster",
    "output_globs": ("*/multispectral.tif",),
    "input_file_key": "file_path",
    "session_file": "multispectral.tif",
    "qml": {"multispectral.tif": "generic"},
}


def barrier_input_raster_multispectral(ctx: dict) -> dict:
    """Import an existing multiband raster as the session's canonical copy."""
    return barrier_input_file(ctx, VARIANT_TARGET)
