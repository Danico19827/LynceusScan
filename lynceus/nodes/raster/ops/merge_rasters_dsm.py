# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Variant DSM of the Merge Rasters strategy family (no NODE_ID)."""

from __future__ import annotations

from lynceus.nodes.ports import PortDef, PortType

VARIANT_OF = "lynceus.nodes.raster.ops.merge_rasters"
VARIANT_KEY = "dsm"
VARIANT_LABEL = "DSM"

INPUTS = (
    PortDef(PortType.DSM_MOSAIC, "DSM 1"),
    PortDef(PortType.DSM_MOSAIC, "DSM 2", required=False),
    PortDef(PortType.DSM_MOSAIC, "DSM 3", required=False),
    PortDef(PortType.DSM_MOSAIC, "DSM 4", required=False),
    PortDef(PortType.DSM_MOSAIC, "DSM 5", required=False),
    PortDef(PortType.DSM_MOSAIC, "DSM 6", required=False),
    PortDef(PortType.DSM_MOSAIC, "DSM 7", required=False),
    PortDef(PortType.DSM_MOSAIC, "DSM 8", required=False),
)
OUTPUTS = (PortType.DSM_MOSAIC,)

PROCESSING_SPECS = {
    "barrier_task": "barrier_merge_rasters_dsm",
    "input_ports": ("dsm_mosaic",) * 8,
    "output_files": {"dsm_mosaic": "merged_dsm.tif"},
    "output_port": "dsm_mosaic",
    "output_globs": ("merged_dsm.tif",),
    "qml": {"merged_dsm.tif": "dsm"},
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


def barrier_merge_rasters_dsm(ctx: dict) -> dict:
    """Fuse wired DSM mosaics into one shared-scale merged_dsm.tif."""
    from lynceus.nodes.raster.ops._merge_base import merge_rasters

    return merge_rasters(
        ctx, session_file="merged_dsm.tif", kind_label="Merged DSM",
        display_label="DSM", module_id=_FAMILY_ID,
    )
