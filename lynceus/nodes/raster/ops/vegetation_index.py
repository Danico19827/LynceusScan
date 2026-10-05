# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Node -- Vegetation Index ((A - B) / (A + B)) over a multispectral raster.

Barrier-only combination node: reads two bands of a single multiband
GeoTIFF (typically a drone orthomosaic) and writes the per-cell index as
a generic raster. The input arrives as one RASTER product (Input Raster
in Multispectral strategy), so no alignment is needed: both bands share
the file grid.
"""

from __future__ import annotations

from lynceus.nodes._point_source import clean_crs
from lynceus.nodes.ports import PortType

NODE_ID = "lynceus.nodes.raster.ops.vegetation_index"
NODE_NAME = "Vegetation Index"
NODE_CATEGORY = "Raster"
NODE_SUBCATEGORY = "Operations"

NODE_DESCRIPTION = (
    "<b>Vegetation Index</b> -- Crop Vigor from Multispectral Bands<br><br>"
    "Computes a normalized difference vegetation index cell by cell from "
    "two bands of a multispectral orthomosaic: "
    "<code>index = (A - B) / (A + B)</code> in [-1, 1].<br><br>"
    "<b>Process:</b> Pick the preset (NDVI, NDRE, GNDVI, SAVI) or Custom "
    "(which reuses the red/NIR band pair) and the 1-based band numbers of "
    "your sensor. Cells where either band is NODATA (or the denominator "
    "is zero) stay NODATA.<br><br>"
    "<b>Tips:</b> Feed this node with an <code>Input Raster</code> node set "
    "to the Multispectral strategy. Check your sensor band order first "
    "(defaults follow the common Blue/Green/Red/RedEdge/NIR layout). "
    "Check band count first: a single-band file is a finished product, "
    "not a multispectral source. RGB-only indices (VARI, GLI) are "
    "illumination-sensitive proxies, not a replacement for NIR-based ones."
)

INPUTS = (PortType.RASTER,)
OUTPUTS = (PortType.RASTER,)

PROCESSING_SPECS = {
    "barrier_task": "barrier_vegetation_index",
    "input_ports": ("raster",),
    "output_files": {"raster": "vegetation_index.tif"},
    "output_port": "raster",
    "output_globs": ("vegetation_index.tif",),
    "qml": {"vegetation_index.tif": "ndvi"},
    "config_schema": {
        "index": {
            "type": "str",
            "default": "ndvi",
            "options": [
                ("NDVI", "ndvi"),
                ("NDRE", "ndre"),
                ("GNDVI", "gndvi"),
                ("SAVI", "savi"),
                ("VARI", "vari"),
                ("GLI", "gli"),
                ("Custom", "custom"),
            ],
            "description": "Which vegetation index to compute.",
            "impact": "Presets fix the band pair and formula; Custom reuses the red/NIR bands with the plain normalized difference.",
            "group": "Index",
        },
        "red_band": {
            "type": "int",
            "default": 3,
            "minimum": 1,
            "maximum": 32,
            "description": "1-based band number for Red.",
            "impact": "Must match the sensor band order; a wrong band silently computes a meaningless index.",
            "group": "Index",
        },
        "nir_band": {
            "type": "int",
            "default": 5,
            "minimum": 1,
            "maximum": 32,
            "description": "1-based band number for near-infrared.",
            "impact": "Must match the sensor band order; a wrong band silently computes a meaningless index.",
            "group": "Index",
        },
        "rededge_band": {
            "type": "int",
            "default": 4,
            "minimum": 1,
            "maximum": 32,
            "description": "1-based band number for Red Edge (NDRE only).",
            "impact": "Only used by the NDRE preset; ignored otherwise.",
            "group": "Index",
        },
        "green_band": {
            "type": "int",
            "default": 2,
            "minimum": 1,
            "maximum": 32,
            "description": "1-based band number for Green (GNDVI, VARI, GLI).",
            "impact": "Used by GNDVI, VARI and GLI; ignored by the other presets.",
            "group": "Index",
        },
        "blue_band": {
            "type": "int",
            "default": 1,
            "minimum": 1,
            "maximum": 32,
            "description": "1-based band number for Blue (VARI, GLI).",
            "impact": "Only used by the RGB-only presets VARI and GLI; ignored otherwise.",
            "group": "Index",
        },
        "soil_brightness_L": {
            "type": "float",
            "default": 0.5,
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Soil brightness factor L (SAVI only).",
            "impact": "0.5 suits most canopies; raise toward 1.0 on sparse cover, lower toward 0.0 on dense cover. Ignored by other presets.",
            "group": "Index",
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

_INDEX_BANDS = {
    "ndvi": ("nir_band", "red_band"),
    "ndre": ("nir_band", "rededge_band"),
    "gndvi": ("nir_band", "green_band"),
    "savi": ("nir_band", "red_band"),
    "custom": ("nir_band", "red_band"),
    "vari": ("green_band", "red_band", "blue_band"),
    "gli": ("green_band", "red_band", "blue_band"),
}


def get_config_defaults() -> dict:
    return {
        "index": "ndvi",
        "red_band": 3,
        "nir_band": 5,
        "rededge_band": 4,
        "green_band": 2,
        "blue_band": 1,
        "soil_brightness_L": 0.5,
        "nodata_value": -9999.0,
    }


def barrier_vegetation_index(ctx: dict) -> dict:
    """Compute the selected index over two bands; NODATA elsewhere."""
    from pathlib import Path

    import numpy as np

    from lynceus.processing import provenance
    from lynceus.processing.raster import (
        _valid_bounds,
        read_geotiff_bands,
        write_geotiff,
    )

    prov_doc = provenance.build_provenance(ctx, "vegetation_index.tif")
    from lynceus.nodes._paths import scoped_file

    session = Path(ctx["session_dir"])
    src = Path(
        ctx.get("raster_path")
        or ctx.get("in0_path", "")
        or str(session / "multispectral.tif")
    )
    if not src.is_file():
        raise RuntimeError(
            "Vegetation Index requires a multispectral raster "
            "(Input Raster in Multispectral strategy)"
        )
    nodata = float(ctx.get("nodata_value", -9999.0))
    out_path = scoped_file(ctx, "vegetation_index.tif")

    preset = str(ctx.get("index", "ndvi") or "ndvi").lower()
    if preset not in _INDEX_BANDS:
        raise RuntimeError(
            f"Unknown vegetation index '{preset}' "
            "(expected ndvi, ndre, gndvi, savi, vari, gli or custom)"
        )
    band_keys = _INDEX_BANDS[preset]
    band_defaults = {
        "red_band": 3,
        "nir_band": 5,
        "rededge_band": 4,
        "green_band": 2,
        "blue_band": 1,
    }
    band_numbers = [
        int(ctx.get(key, band_defaults[key])) for key in band_keys
    ]
    cubes, transform, crs = read_geotiff_bands(src, band_numbers)
    valid_src = _valid_bounds(cubes[0], transform, nodata)
    crs_out = clean_crs(ctx.get("crs")) or crs
    if valid_src is None:
        grid = np.full(cubes[0].shape, nodata, dtype=np.float32)
        write_geotiff(
            out_path, grid, transform, crs=crs_out, nodata=nodata,
            metadata=prov_doc,
        )
        return {
            "file": out_path,
            "kind": "Vegetation Index",
            "node": NODE_ID,
            "warnings": [
                "Multispectral input has no valid data; "
                "index written as all-NODATA"
            ],
        }

    bands = cubes.astype(np.float64)
    valid = np.ones(bands.shape[1:], dtype=bool)
    for band in bands:
        valid &= band != nodata
    if preset in ("vari", "gli"):
        g, r, b = bands[0], bands[1], bands[2]
        if preset == "vari":
            top, bottom = g - r, g + r - b
        else:
            top, bottom = 2 * g - r - b, 2 * g + r + b
    else:
        a, b = bands[0], bands[1]
        top, bottom = a - b, a + b
        if preset == "savi":
            L = float(ctx.get("soil_brightness_L", 0.5))
            bottom = bottom + L
            top = top * (1.0 + L)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(valid & (bottom != 0), top / bottom, np.nan)
    grid = np.full(bands.shape[1:], nodata, dtype=np.float32)
    grid[~np.isnan(ratio)] = np.clip(ratio[~np.isnan(ratio)], -1.0, 1.0)

    write_geotiff(
        out_path, grid, transform, crs=crs_out, nodata=nodata, metadata=prov_doc
    )
    return {"file": out_path, "kind": "Vegetation Index", "node": NODE_ID}
