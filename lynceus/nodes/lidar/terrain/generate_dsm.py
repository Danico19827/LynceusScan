# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Node -- DSM generation from point cloud.

Tile task: rasterizes max-Z of all points (core-masked) per tile.
Barrier task: merges all DSM tiles, applies hole-filling and smoothing.
"""

from __future__ import annotations

from pathlib import Path

from lynceus.nodes._point_source import point_src
from lynceus.nodes.ports import PortType

NODE_ID = "lynceus.nodes.lidar.terrain.generate_dsm"
NODE_NAME = "Generate DSM"
NODE_CATEGORY = "LiDAR"

NODE_SUBCATEGORY = "Terrain"

NODE_DESCRIPTION = (
    "<b>Generate DSM</b> -- Digital Surface Model<br><br>"
    "Creates a surface model capturing the highest return per cell, "
    "including buildings, vegetation canopy, and ground.<br><br>"
    "<b>Process:</b> For each tile, the maximum-Z of all points is rasterized. "
    "After merging, holes from sparse coverage are filled by nearest-neighbor "
    "interpolation, then optional Gaussian smoothing reduces noise.<br><br>"
    "<b>Tips:</b> Lower <i>hole fill distance</i> reduces extrapolation artifacts "
    "at scan edges. Increase <i>smooth sigma</i> for a cleaner visual surface; "
    "decrease for maximum detail in building edges."
)

INPUTS = (PortType.POINT_CLOUD,)
OUTPUTS = (PortType.DSM_TILE, PortType.DSM_MOSAIC)

PROCESSING_SPECS = {
    "tile_task": "tile_generate_dsm",
    "barrier_task": "barrier_generate_dsm",
    "input_port": "point_cloud",
    "output_ports": ("dsm_tile", "dsm_mosaic"),
    "output_files": {"dsm_mosaic": "dsm_mosaic.tif"},
    "qml": {"dsm_mosaic.tif": "dsm"},
    "config_schema": {
        "resolution": {
            "type": "float",
            "default": 1.0,
            "minimum": 0.1,
            "maximum": 5.0,
            "description": "Grid cell size in meters for the DSM raster.",
            "impact": "Smaller values capture building edges and canopy gaps better but increase memory and processing time.",
            "group": "Grid",
        },
        "hole_fill_enabled": {
            "type": "bool",
            "default": True,
            "description": "Enable hole-filling to close small gaps in the DSM.",
            "impact": "Fills isolated NODATA cells by copying the nearest valid neighbor. Disable if gaps represent real no-data zones.",
            "group": "Hole Filling",
        },
        "hole_fill_max_dist_m": {
            "type": "float",
            "default": 1.5,
            "minimum": 0.0,
            "maximum": 10.0,
            "description": "Maximum distance (meters) to fill holes from nearest valid cell.",
            "impact": "Larger values fill bigger gaps but may create false surface values at scan edges. Keep lower than DTM to avoid extrapolating building heights.",
            "group": "Hole Filling",
        },
        "smooth_enabled": {
            "type": "bool",
            "default": True,
            "description": "Enable Gaussian smoothing on the merged DSM.",
            "impact": "Reduces single-pixel noise from the raw max-Z surface. Disable for sharp building edges.",
            "group": "Smoothing",
        },
        "smooth_sigma": {
            "type": "float",
            "default": 0.5,
            "minimum": 0.0,
            "maximum": 3.0,
            "description": "Sigma (radius) of the Gaussian smoothing filter in grid cells.",
            "impact": "Higher values soften building edges and canopy detail. Lower values preserve sharp features. Use 0.5-1.0 for most urban/forest mixes.",
            "group": "Smoothing",
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
        "interpolation_method": {
            "type": "str",
            "default": "max",
            "options": ["min", "max", "mean", "median", "idw"],
            "description": "Aggregation method for points falling in each grid cell.",
            "impact": "max gives the standard canopy surface DSM. mean smooths the surface. median is robust to outliers. idw (Inverse Distance Weighted) interpolates between points for denser coverage.",
            "group": "Grid",
        },
        "crop_to_valid": {
            "type": "bool",
            "default": True,
            "description": "Crop the output raster to the bounding box of valid data.",
            "impact": "Removes NODATA borders around the survey area, producing tighter output files. Disable to keep the full rectangular extent of input tiles.",
            "group": "Output",
        },
    },
}


def get_config_defaults() -> dict:
    return {k: v["default"] for k, v in PROCESSING_SPECS["config_schema"].items()}


def tile_generate_dsm(tile: dict, ctx: dict) -> dict:
    """Rasterize max-Z of all points for a single tile (core-masked)."""
    import laspy
    import numpy as np

    from lynceus.processing import provenance
    from lynceus.processing.raster import (
        TileAccumulator,
        core_mask,
        write_geotiff,
    )
    from lynceus.processing.tiler import _laz_backend

    out_path = _raster_path(tile, ctx, "dsm")
    if Path(out_path).exists():
        return {"tile_id": tile["tile_id"], "node": NODE_ID, "output": out_path}

    prov_doc = provenance.build_provenance(ctx, f"{tile['tile_id']}.tif")
    src = point_src(tile, ctx)
    interp_method = ctx.get("interpolation_method", "max")
    nodata = ctx.get("nodata_value", -9999.0)
    acc = TileAccumulator(
        tile,
        ctx.get("resolution", 1.0),
        interp_method,
        memory_budget_bytes=ctx.get("_raster_grid_memory_budget_bytes"),
    )

    with laspy.open(src, laz_backend=_laz_backend()) as reader:
        has_withheld = "withheld" in reader.header.point_format.dimension_names
        for chunk in reader.chunk_iterator(2_000_000):
            x = np.asarray(chunk.x)
            y = np.asarray(chunk.y)
            z = np.asarray(chunk.z)
            mask = core_mask(x, y, tile)
            if has_withheld:
                # Vendor-withheld points are excluded from surfaces.
                mask = mask & ~np.asarray(chunk.withheld, dtype=bool)
            if mask.any():
                acc.add(x[mask], y[mask], z[mask])

    grid, transform = acc.result()
    write_geotiff(
        out_path, grid, transform, crs=ctx.get("crs"), nodata=nodata,
        metadata=prov_doc,
    )
    return {"tile_id": tile["tile_id"], "node": NODE_ID, "output": out_path}


def barrier_generate_dsm(ctx: dict) -> dict:
    """Merge all DSM tiles, then apply hole-filling, smoothing and crop."""
    from lynceus.processing import provenance
    from lynceus.processing.raster import (
        fill_holes,
        merge_geotiffs,
        smooth_nodata,
        write_geotiff,
    )

    prov_doc = provenance.build_provenance(ctx, "dsm_mosaic.tif")
    from lynceus.nodes._paths import scoped_dir, scoped_file

    paths = sorted(scoped_dir(ctx, "dsm").glob("*.tif"))
    if not paths:
        raise RuntimeError("No DSM tiles generated")
    nodata = ctx.get("nodata_value", -9999.0)
    out_path = scoped_file(ctx, "dsm_mosaic.tif")
    arr, transform, crs = merge_geotiffs(
        paths, ctx.get("crs"), out_path, nodata=nodata, metadata=prov_doc,
        return_array=True,
    )

    # Post-merge processing: hole-fill + smooth
    if ctx.get("hole_fill_enabled", True):
        max_dist = ctx.get("hole_fill_max_dist_m", 1.5)
        if max_dist > 0:
            resolution = ctx.get("resolution", 1.0)
            arr = fill_holes(arr, resolution, max_dist, nodata)

    if ctx.get("smooth_enabled", True):
        sigma = ctx.get("smooth_sigma", 0.5)
        if sigma > 0:
            arr = smooth_nodata(arr, sigma, nodata)

    if ctx.get("crop_to_valid", True):
        from lynceus.processing.raster import _crop_to_valid

        arr, transform = _crop_to_valid(arr, transform, nodata)

    write_geotiff(out_path, arr, transform, crs=crs, nodata=nodata, metadata=prov_doc)
    return {"file": out_path, "kind": "DSM", "node": NODE_ID}


def _raster_path(tile: dict, ctx: dict, kind: str) -> str:
    from lynceus.nodes._paths import scoped_dir

    return str(scoped_dir(ctx, kind) / f"{tile['tile_id']}.tif")
