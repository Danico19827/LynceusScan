# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Variant Raster of the Merge Rasters strategy family (no NODE_ID)."""

from __future__ import annotations

from lynceus.nodes.ports import PortDef, PortType

VARIANT_OF = "lynceus.nodes.raster.ops.merge_rasters"
VARIANT_KEY = "raster"
VARIANT_LABEL = "Raster"

INPUTS = (
    PortDef(PortType.RASTER, "Raster 1"),
    PortDef(PortType.RASTER, "Raster 2", required=False),
    PortDef(PortType.RASTER, "Raster 3", required=False),
    PortDef(PortType.RASTER, "Raster 4", required=False),
    PortDef(PortType.RASTER, "Raster 5", required=False),
    PortDef(PortType.RASTER, "Raster 6", required=False),
    PortDef(PortType.RASTER, "Raster 7", required=False),
    PortDef(PortType.RASTER, "Raster 8", required=False),
)
OUTPUTS = (PortType.RASTER,)

PROCESSING_SPECS = {
    "barrier_task": "barrier_merge_rasters_raster",
    "input_ports": ("raster",) * 8,
    "output_files": {"raster": "merged_raster.tif"},
    "output_port": "raster",
    "output_globs": ("merged_raster.tif",),
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


def barrier_merge_rasters_raster(ctx: dict) -> dict:
    """Fuse wired generic rasters into one shared-scale merged_raster.tif."""
    from lynceus.nodes.raster.ops._merge_base import merge_rasters

    return merge_rasters(
        ctx, session_file="merged_raster.tif", kind_label="Merged Raster",
        display_label="Raster", module_id=_FAMILY_ID,
    )
