# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Variant CHM of the Merge Rasters strategy family (no NODE_ID)."""

from __future__ import annotations

from lynceus.nodes.ports import PortDef, PortType

VARIANT_OF = "lynceus.nodes.raster.ops.merge_rasters"
VARIANT_KEY = "chm"
VARIANT_LABEL = "CHM"

INPUTS = (
    PortDef(PortType.CHM_MOSAIC, "CHM 1"),
    PortDef(PortType.CHM_MOSAIC, "CHM 2", required=False),
    PortDef(PortType.CHM_MOSAIC, "CHM 3", required=False),
    PortDef(PortType.CHM_MOSAIC, "CHM 4", required=False),
    PortDef(PortType.CHM_MOSAIC, "CHM 5", required=False),
    PortDef(PortType.CHM_MOSAIC, "CHM 6", required=False),
    PortDef(PortType.CHM_MOSAIC, "CHM 7", required=False),
    PortDef(PortType.CHM_MOSAIC, "CHM 8", required=False),
)
OUTPUTS = (PortType.CHM_MOSAIC,)

PROCESSING_SPECS = {
    "barrier_task": "barrier_merge_rasters_chm",
    "input_ports": ("chm_mosaic",) * 8,
    "output_files": {"chm_mosaic": "merged_chm.tif"},
    "output_port": "chm_mosaic",
    "output_globs": ("merged_chm.tif",),
    "qml": {"merged_chm.tif": "chm"},
    # Dispatch working set in cell copies (plain merge): the gate budgets
    # the direct need when it fits, the window peak when it streams.
    "merge_copies": 3.0,
    "config_schema": {
        "method": {
            "type": "str",
            "default": "first",
            "options": ["first", "last", "min", "max"],
            "description": "Which input wins cells covered by several rasters.",
            "impact": "first keeps the earliest-wired raster on overlaps; last the latest; min/max take the extreme. Adjacent non-overlapping tiles never conflict.",
            "group": "Merge",
        },
        "nodata_value": {
            "type": "float",
            "default": -9999.0,
            "minimum": -99999.0,
            "maximum": 99999.0,
            "description": "Value written to cells with no data in the GeoTIFF.",
            "impact": "Should match the convention of your GIS software. Almost always -9999.0.",
            "group": "Advanced",
            "advanced": True,
        },
    },
}

_FAMILY_ID = "lynceus.nodes.raster.ops.merge_rasters"


def get_config_defaults() -> dict:
    return {k: v["default"] for k, v in PROCESSING_SPECS["config_schema"].items()}


def barrier_merge_rasters_chm(ctx: dict) -> dict:
    """Fuse wired CHM mosaics into one shared-scale merged_chm.tif."""
    from lynceus.nodes.raster.ops._merge_base import merge_rasters

    return merge_rasters(
        ctx, session_file="merged_chm.tif", kind_label="Merged CHM",
        display_label="CHM", module_id=_FAMILY_ID,
    )
