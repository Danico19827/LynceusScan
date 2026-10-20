# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Variant Hillshade of the Terrain Derivatives family (no NODE_ID)."""

from __future__ import annotations

from lynceus.nodes.ports import PortType

VARIANT_OF = "lynceus.nodes.raster.ops.terrain_derivatives"
VARIANT_KEY = "hillshade"
VARIANT_LABEL = "Hillshade"

INPUTS = (PortType.DTM_MOSAIC,)
OUTPUTS = (PortType.RASTER,)

PROCESSING_SPECS = {
    "barrier_task": "barrier_terrain_hillshade",
    "input_ports": ("dtm_mosaic",),
    "output_files": {"raster": "hillshade.tif"},
    "output_port": "raster",
    "output_globs": ("hillshade.tif",),
    "qml": {"hillshade.tif": "hillshade"},
    "config_schema": {
        "azimuth_deg": {
            "type": "float",
            "default": 315.0,
            "minimum": 0.0,
            "maximum": 360.0,
            "description": "Sun azimuth in degrees clockwise from north.",
            "impact": "Standard cartography uses 315 (northwest) so relief reads naturally. Changing it relights the scene without touching data.",
            "group": "Terrain",
        },
        "altitude_deg": {
            "type": "float",
            "default": 45.0,
            "minimum": 0.0,
            "maximum": 90.0,
            "description": "Sun altitude in degrees above the horizon.",
            "impact": "Lower sun lengthens shadows and drama; 45 is the cartographic standard. At 90 the shading goes flat.",
            "group": "Terrain",
        },
        "z_factor": {
            "type": "float",
            "default": 1.0,
            "minimum": 0.1,
            "maximum": 10.0,
            "description": "Vertical exaggeration applied before derivatives.",
            "impact": "Values above 1 steepen relief (useful in flat terrain); below 1 flattens it. Slope values, aspect boundaries and hillshade all see the exaggerated surface.",
            "group": "Terrain",
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

_FAMILY_ID = "lynceus.nodes.raster.ops.terrain_derivatives"


def get_config_defaults() -> dict:
    return {
        "azimuth_deg": 315.0,
        "altitude_deg": 45.0,
        "z_factor": 1.0,
        "nodata_value": -9999.0,
    }


def barrier_terrain_hillshade(ctx: dict) -> dict:
    """Compute Horn-1981 hillshade 0-255 for the configured sun position."""
    from pathlib import Path

    from lynceus.nodes._paths import scoped_file
    from lynceus.nodes._point_source import clean_crs
    from lynceus.processing import provenance
    from lynceus.processing.raster import dem_mosaic, hillshade_from_dem

    prov_doc = provenance.build_provenance(ctx, "hillshade.tif")
    session = Path(ctx["session_dir"])
    in_path = Path(
        ctx.get("dtm_mosaic_path")
        or ctx.get("in0_path")
        or str(session / "dtm_mosaic.tif")
    )
    if not in_path.is_file():
        raise RuntimeError(
            "Terrain Hillshade needs a DTM mosaic wired to its input")
    out_path = scoped_file(ctx, "hillshade.tif")

    # The file's own NODATA tag (plus NaN, always) decides validity —
    # external DTMs tag voids their own way and must not read as ground.
    azimuth = float(ctx.get("azimuth_deg", 315.0))
    altitude = float(ctx.get("altitude_deg", 45.0))
    z_factor = float(ctx.get("z_factor", 1.0))
    _, empty = dem_mosaic(
        in_path, out_path,
        dem_fn=lambda arr, cx, cy: hillshade_from_dem(
            arr, cx, cy, nodata=float(ctx.get("nodata_value", -9999.0)),
            azimuth_deg=azimuth, altitude_deg=altitude, z_factor=z_factor),
        halo_cells=1,
        nodata=float(ctx.get("nodata_value", -9999.0)),
        crs_out=clean_crs(ctx.get("crs")) or None,
        metadata=prov_doc,
        memory_budget_bytes=ctx.get("_raster_grid_memory_budget_bytes"),
        notes=ctx.setdefault("_bounded_notes", []),
    )
    if empty:
        return {
            "file": out_path,
            "kind": "Hillshade",
            "node": _FAMILY_ID,
            "warnings": [
                "DTM input has no valid data; hillshade written as all-NODATA"
            ],
        }

    return {"file": out_path, "kind": "Hillshade", "node": _FAMILY_ID}
