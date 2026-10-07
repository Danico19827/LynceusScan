# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Node -- CHM generation from DTM and DSM mosaics.

Barrier task: computes CHM = DSM - DTM from mosaics, applies smoothing
and height clipping, writes CHM mosaic.
"""

from __future__ import annotations

from lynceus.nodes._point_source import clean_crs
from lynceus.nodes.ports import PortType

NODE_ID = "lynceus.nodes.lidar.terrain.generate_chm"
NODE_NAME = "Generate CHM"
NODE_CATEGORY = "LiDAR"

NODE_SUBCATEGORY = "Terrain"

NODE_DESCRIPTION = (
    "<b>Generate CHM</b> -- Canopy Height Model<br><br>"
    "Computes vegetation height by subtracting DTM from DSM: "
    "<code>CHM = DSM - DTM</code>. Only cells where both models have valid "
    "data are computed; the rest remain NODATA.<br><br>"
    "<b>Process:</b> The DTM and DSM mosaics are aligned by their overlapping "
    "bounds, then subtracted. Heights are clamped to [min, max] range, "
    "and optional Gaussian smoothing reduces noise in the canopy surface. "
    "An optional pit-free composite fills small pits for tree work.<br><br>"
    "<b>Tips:</b> Increase <i>smooth sigma</i> for smoother canopy surfaces in "
    "forestry analysis. Lower it for individual tree crown detection. "
    "Adjust <i>max height</i> if processing very tall vegetation (e.g. "
    "tropical forests with trees over 60m)."
)

INPUTS = (PortType.DTM_MOSAIC, PortType.DSM_MOSAIC)
OUTPUTS = (PortType.CHM_MOSAIC,)

PROCESSING_SPECS = {
    "barrier_task": "barrier_generate_chm",
    "input_ports": ("dtm_mosaic", "dsm_mosaic"),
    "output_files": {"chm_mosaic": "chm_mosaic.tif"},
    "output_port": "chm_mosaic",
    "qml": {"chm_mosaic.tif": "chm"},
    "config_schema": {
        "smooth_enabled": {
            "type": "bool",
            "default": True,
            "description": "Enable Gaussian smoothing on the CHM surface.",
            "impact": "Smooths noise from DTM/DSM subtraction artifacts. Disable for maximum canopy detail.",
            "group": "Smoothing",
        },
        "smooth_sigma": {
            "type": "float",
            "default": 0.5,
            "minimum": 0.0,
            "maximum": 3.0,
            "description": "Sigma (radius) of the Gaussian smoothing filter in grid cells.",
            "impact": "Higher values produce a smoother canopy surface, useful for area-based forestry metrics. Lower values preserve individual tree crown shapes for crown delineation.",
            "group": "Smoothing",
        },
        "max_height_m": {
            "type": "float",
            "default": 60.0,
            "minimum": 5.0,
            "maximum": 200.0,
            "description": "Maximum vegetation height cap in meters.",
            "impact": "Heights above this value are clipped. Set higher for tall species (e.g. 80m for tropical canopy). Lower values act as an outlier filter for erroneous DSM peaks.",
            "group": "Clip",
        },
        "min_height_m": {
            "type": "float",
            "default": 0.0,
            "minimum": -5.0,
            "maximum": 5.0,
            "description": "Minimum vegetation height threshold in meters.",
            "impact": "Values below this are set to zero. A small negative value (e.g. -0.5) accounts for slight DTM/DSM misalignment. Set to 0 for strict positive-only heights.",
            "group": "Clip",
            "advanced": True,
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
        "fill_gaps": {
            "type": "bool",
            "default": False,
            "description": "Fill small NODATA gaps in the CHM using nearest-neighbor interpolation.",
            "impact": "When enabled, isolated NODATA cells surrounded by valid data are filled. Useful for smoothing DTM/DSM misalignment artifacts. Disable if gaps represent real no-data areas.",
            "group": "Post-Processing",
        },
        "pit_free": {
            "type": "bool",
            "default": False,
            "description": "Fill canopy pits with a multi-resolution maximum composite.",
            "impact": "Mosaic-level approximation of the Khosravipour pit-free CHM: removes small pits from penetrating returns for cleaner tree crowns. Slightly raises crown edges; keep off for precise height measurements. Only affects the DSM side.",
            "group": "Smoothing",
        },
        "output_resolution": {
            "type": "float",
            "default": 0.0,
            "minimum": 0.0,
            "maximum": 50.0,
            "description": "Override output resolution (0 = use input resolution).",
            "impact": "Resample the CHM to a different grid size. 0 keeps the native DTM/DSM resolution. Useful for standardizing outputs across projects.",
            "group": "Output",
        },
    },
}


def get_config_defaults() -> dict:
    return {k: v["default"] for k, v in PROCESSING_SPECS["config_schema"].items()}


def barrier_generate_chm(ctx: dict) -> dict:
    """Compute CHM = DSM - DTM from mosaics with smoothing and height clip."""
    from pathlib import Path

    import numpy as np

    from lynceus.processing import provenance
    from lynceus.processing.raster import (
        _valid_bounds,
        compute_chm,
        fill_holes,
        read_geotiff,
        write_geotiff,
    )

    prov_doc = provenance.build_provenance(ctx, "chm_mosaic.tif")
    from lynceus.nodes._paths import scoped_file

    session = Path(ctx["session_dir"])
    dtm = Path(ctx.get("dtm_mosaic_path", str(session / "dtm_mosaic.tif")))
    dsm = Path(ctx.get("dsm_mosaic_path", str(session / "dsm_mosaic.tif")))
    if not dtm.exists() or not dsm.exists():
        raise RuntimeError("CHM requires both DTM and DSM mosaics")
    out_path = scoped_file(ctx, "chm_mosaic.tif")
    nodata = ctx.get("nodata_value", -9999.0)

    dtm_arr, dtm_transform, crs_dtm = read_geotiff(dtm)
    dsm_arr, dsm_transform, crs_dsm = read_geotiff(dsm)
    if (
        _valid_bounds(dtm_arr, dtm_transform, nodata) is None
        or _valid_bounds(dsm_arr, dsm_transform, nodata) is None
    ):
        # Nothing to subtract: an all-NODATA CHM is faithful to the empty
        # stream instead of dying. The NODATA cell of the DTM canvas keeps
        # the raster registered with the session grid.
        grid = np.full(dtm_arr.shape, nodata, dtype=np.float32)
        write_geotiff(
            out_path, grid, dtm_transform, crs=ctx.get("crs"),
            nodata=nodata, metadata=prov_doc,
        )
        return {
            "file": out_path,
            "kind": "CHM",
            "node": NODE_ID,
            "warnings": ["DTM or DSM without valid data; CHM written as all-NODATA"],
        }

    if ctx.get("pit_free", False):
        from lynceus.processing.raster import pit_free_canopy

        dsm_arr = pit_free_canopy(dsm_arr, nodata)

    compute_chm(
        dtm, dsm, out_path, crs=clean_crs(ctx.get("crs")) or crs_dtm or crs_dsm,
        nodata=nodata,
        smooth_enabled=ctx.get("smooth_enabled", True),
        smooth_sigma=ctx.get("smooth_sigma", 0.5),
        max_height_m=ctx.get("max_height_m", 60.0),
        min_height_m=ctx.get("min_height_m", 0.0),
        metadata=prov_doc,
        dtm_arr=dtm_arr, dtm_transform=dtm_transform,
        dsm_arr=dsm_arr, dsm_transform=dsm_transform,
    )

    # Optional gap filling (native resolution: one-cell gaps by default).
    if ctx.get("fill_gaps", False):
        arr, transform, crs = read_geotiff(out_path)
        res = abs(float(transform.a))
        arr = fill_holes(arr, res, res, nodata)
        write_geotiff(out_path, arr, transform, crs=crs, nodata=nodata, metadata=prov_doc)

    # Optional resolution override (standard write path: deflate + tags).
    out_res = ctx.get("output_resolution", 0.0)
    if out_res > 0:
        import rasterio
        from rasterio.warp import reproject, calculate_default_transform

        with rasterio.open(out_path) as src:
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs, src.crs, src.width, src.height,
                *src.bounds, resolution=out_res,
            )
            src_crs = str(src.crs) if src.crs else None
            dst = np.full((dst_height, dst_width), nodata, dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=src.crs,
                dst_nodata=nodata,
            )
        write_geotiff(
            out_path, dst, dst_transform, crs=clean_crs(ctx.get("crs")) or src_crs,
            nodata=nodata, metadata=prov_doc,
        )

    return {"file": out_path, "kind": "CHM", "node": NODE_ID}
