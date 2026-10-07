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

    import numpy as np

    from lynceus.nodes._paths import scoped_file
    from lynceus.nodes._point_source import clean_crs
    from lynceus.processing import provenance
    from lynceus.processing.raster import (
        _valid_bounds,
        hillshade_from_dem,
        read_geotiff,
        write_geotiff,
    )

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
    nodata = float(ctx.get("nodata_value", -9999.0))
    out_path = scoped_file(ctx, "hillshade.tif")

    arr, transform, crs_in = read_geotiff(in_path)
    crs_out = clean_crs(ctx.get("crs")) or crs_in
    if _valid_bounds(arr, transform, nodata) is None:
        grid = np.full(arr.shape, nodata, dtype=np.float32)
        write_geotiff(out_path, grid, transform, crs=crs_out, nodata=nodata,
                      metadata=prov_doc)
        return {
            "file": out_path,
            "kind": "Hillshade",
            "node": _FAMILY_ID,
            "warnings": [
                "DTM input has no valid data; hillshade written as all-NODATA"
            ],
        }

    grid = hillshade_from_dem(
        arr.astype(np.float64), abs(transform.a), abs(transform.e),
        nodata=nodata, azimuth_deg=float(ctx.get("azimuth_deg", 315.0)),
        altitude_deg=float(ctx.get("altitude_deg", 45.0)),
        z_factor=float(ctx.get("z_factor", 1.0)),
    ).astype(np.float32)
    write_geotiff(out_path, grid, transform, crs=crs_out, nodata=nodata,
                  metadata=prov_doc)
    return {"file": out_path, "kind": "Hillshade", "node": _FAMILY_ID}
