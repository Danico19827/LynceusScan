# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Variant DTM of the Input Raster strategy family (no NODE_ID).

``VARIANT_OF`` attaches this module to the ``input_raster`` base node; the
strategy key ``dtm`` activates its OUTPUTS and PROCESSING_SPECS per instance.
"""

from __future__ import annotations

from lynceus.nodes._product_input import barrier_input_file
from lynceus.nodes.ports import PortType

VARIANT_OF = "lynceus.nodes.raster.input_raster"
VARIANT_KEY = "dtm"
VARIANT_LABEL = "DTM"

VARIANT_TARGET = {
    "node_id": "lynceus.nodes.raster.input_raster",
    "node_name": "Input DTM",
    "output_port": "dtm_mosaic",
    "session_file": "dtm_mosaic.tif",
    "kind_label": "DTM",
    "kind_group": "raster",
    "file_filters": "Raster files (*.tif *.tiff)",
}

INPUTS = ()
OUTPUTS = (PortType.DTM_MOSAIC,)

PROCESSING_SPECS = {
    "barrier_task": "barrier_input_raster_dtm",
    "output_port": "dtm_mosaic",
    "output_globs": ("*/dtm_mosaic.tif",),
    "input_file_key": "file_path",
    "session_file": "dtm_mosaic.tif",
    "qml": {"dtm_mosaic.tif": "dtm"},
}


def barrier_input_raster_dtm(ctx: dict) -> dict:
    """Import an existing DTM GeoTIFF as the session's canonical DTM mosaic."""
    return barrier_input_file(ctx, VARIANT_TARGET)