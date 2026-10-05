# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Node -- DTM generation from classified point cloud.

Tile task: rasterizes min-Z of ground points (core-masked) per tile.
Barrier task: merges all DTM tiles, applies hole-filling and smoothing.
"""

from __future__ import annotations

from pathlib import Path

from lynceus.nodes._point_source import point_src
from lynceus.nodes.ports import PortType

NODE_ID = "lynceus.nodes.lidar.terrain.generate_dtm"
NODE_NAME = "Generate DTM"
NODE_CATEGORY = "LiDAR"

NODE_SUBCATEGORY = "Terrain"

NODE_DESCRIPTION = (
    "<b>Generate DTM</b> -- Digital Terrain Model<br><br>"
    "Creates a bare-earth elevation model by extracting minimum-Z values "
    "from classified ground points (class 2).<br><br>"
    "<b>Process:</b> For each tile, ground points are rasterized to a min-Z grid. "
    "After merging all tiles, optional hole-filling and Gaussian smoothing are "
    "applied to produce a clean surface.<br><br>"
    "<b>Tips:</b> Increase <i>hole fill distance</i> to close gaps in sparse areas. "
    "Higher <i>smooth sigma</i> removes micro-topography noise but may flatten "
    "real terrain features like small berms or ditches."
)

INPUTS = (PortType.POINT_CLOUD,)
OUTPUTS = (PortType.DTM_TILE, PortType.DTM_MOSAIC)

PROCESSING_SPECS = {
    "tile_task": "tile_generate_dtm",
    "barrier_task": "barrier_generate_dtm",
    "input_port": "point_cloud",
    "output_ports": ("dtm_tile", "dtm_mosaic"),
    "requires": {"classified": True},
    "qml": {"dtm_mosaic.tif": "dtm"},
    "config_schema": {
        "resolution": {
            "type": "float",
            "default": 1.0,
            "minimum": 0.1,
            "maximum": 5.0,
            "description": "Grid cell size in meters for the DTM raster.",
            "impact": "Smaller values capture finer detail but increase memory and processing time. Larger values speed up processing at the cost of terrain detail.",
            "group": "Grid",
        },
        "hole_fill_enabled": {
            "type": "bool",
            "default": True,
            "description": "Enable hole-filling to close small gaps in the DTM.",
            "impact": "When enabled, isolated NODATA cells surrounded by valid data are filled by copying the nearest valid neighbor. Disable if gaps represent real no-data zones (e.g. water bodies).",
            "group": "Hole Filling",
        },
        "hole_fill_max_dist_m": {
            "type": "float",
            "default": 3.0,
            "minimum": 0.0,
            "maximum": 20.0,
            "description": "Maximum distance (meters) to fill holes from nearest valid cell.",
            "impact": "Larger values fill bigger gaps but may extrapolate into areas without real ground data. Set to 0 to disable fill regardless of the toggle.",
            "group": "Hole Filling",
        },
        "smooth_enabled": {
            "type": "bool",
            "default": True,
            "description": "Enable Gaussian smoothing on the merged DTM.",
            "impact": "Smooths micro-topography noise from the raw min-Z surface. Disable for maximum raw detail.",
            "group": "Smoothing",
        },
        "smooth_sigma": {
            "type": "float",
            "default": 1.0,
            "minimum": 0.0,
            "maximum": 5.0,
            "description": "Sigma (radius) of the Gaussian smoothing filter in grid cells.",
            "impact": "Higher values produce a smoother surface, removing small terrain features like ditches and berms. Lower values preserve more detail. A value of 0 is equivalent to no smoothing.",
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
            "default": "min",
            "options": ["min", "mean", "median", "idw"],
            "description": "Aggregation method for points falling in each grid cell.",
            "impact": "min gives the standard bare-earth DTM. mean smooths the surface but may overestimate ground under sparse canopy. median is robust to outliers. idw (Inverse Distance Weighted) interpolates between points for denser coverage.",
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
    "output_files": {"dtm_mosaic": "dtm_mosaic.tif"},
}


def get_config_defaults() -> dict:
    return {k: v["default"] for k, v in PROCESSING_SPECS["config_schema"].items()}


def tile_generate_dtm(tile: dict, ctx: dict) -> dict:
    """Rasterize min-Z for a single tile (core-masked).

    Uses ground points (class 2) when available -- from the upstream
    Classify Ground node or auto-detected in the source cloud. Otherwise
    it degrades and computes from all points, warning the user.
    """
    import laspy
    import numpy as np

    from lynceus.processing import provenance
    from lynceus.processing.raster import (
        GROUND_CLASS,
        TileAccumulator,
        core_mask,
        write_geotiff,
    )
    from lynceus.processing.tiler import _laz_backend

    out_path = _raster_path(tile, ctx, "dtm")
    if Path(out_path).exists():
        return {"tile_id": tile["tile_id"], "node": NODE_ID, "output": out_path}

    prov_doc = provenance.build_provenance(ctx, f"{tile['tile_id']}.tif")
    src = point_src(tile, ctx)
    interp_method = ctx.get("interpolation_method", "min")
    resolution = ctx.get("resolution", 1.0)
    nodata = ctx.get("nodata_value", -9999.0)
    grid_budget = ctx.get("_raster_grid_memory_budget_bytes")
    acc = TileAccumulator(
        tile, resolution, interp_method, memory_budget_bytes=grid_budget
    )

    ground_found = False
    total_points = 0
    backend = _laz_backend()
    with laspy.open(src, laz_backend=backend) as reader:
        has_class = "classification" in reader.header.point_format.dimension_names
        has_withheld = "withheld" in reader.header.point_format.dimension_names
        for chunk in reader.chunk_iterator(2_000_000):
            x = np.asarray(chunk.x)
            y = np.asarray(chunk.y)
            z = np.asarray(chunk.z)
            mask = core_mask(x, y, tile)
            if has_withheld:
                # Vendor-withheld points are excluded from surfaces.
                mask = mask & ~np.asarray(chunk.withheld, dtype=bool)
            total_points += int(mask.sum())
            if has_class:
                cls = np.asarray(chunk.classification)
                gmask = mask & (cls == GROUND_CLASS)
                if gmask.any():
                    ground_found = True
                    acc.add(x[gmask], y[gmask], z[gmask])

    if not ground_found:
        # Degraded second pass over all points (rare): a single accumulator
        # stays resident in the common case instead of two always allocated.
        del acc
        acc_all = TileAccumulator(
            tile, resolution, interp_method, memory_budget_bytes=grid_budget
        )
        with laspy.open(src, laz_backend=backend) as reader:
            has_withheld = (
                "withheld" in reader.header.point_format.dimension_names
            )
            for chunk in reader.chunk_iterator(2_000_000):
                x = np.asarray(chunk.x)
                y = np.asarray(chunk.y)
                z = np.asarray(chunk.z)
                mask = core_mask(x, y, tile)
                if has_withheld:
                    mask = mask & ~np.asarray(chunk.withheld, dtype=bool)
                if mask.any():
                    acc_all.add(x[mask], y[mask], z[mask])
        acc = acc_all

    grid, transform = acc.result()
    write_geotiff(
        out_path, grid, transform, crs=ctx.get("crs"), nodata=nodata,
        metadata=prov_doc,
    )
    result = {"tile_id": tile["tile_id"], "node": NODE_ID, "output": out_path}
    if not ground_found:
        if total_points == 0:
            result["warnings"] = [
                "Tile has no points in its core; DTM written as all-NODATA."
            ]
        else:
            result["warnings"] = [
                "No ground (class 2) points found; DTM computed from all "
                "points. If ground was classified with a custom class, check "
                "the output_class of Classify Ground."
            ]
    return result


def barrier_generate_dtm(ctx: dict) -> dict:
    """Merge all DTM tiles, then apply hole-filling, smoothing and crop."""
    from lynceus.processing import provenance
    from lynceus.processing.raster import (
        fill_holes,
        merge_geotiffs,
        smooth_nodata,
        write_geotiff,
    )

    prov_doc = provenance.build_provenance(ctx, "dtm_mosaic.tif")
    from lynceus.nodes._paths import scoped_dir, scoped_file

    paths = sorted(scoped_dir(ctx, "dtm").glob("*.tif"))
    if not paths:
        raise RuntimeError("No DTM tiles generated")
    nodata = ctx.get("nodata_value", -9999.0)
    out_path = scoped_file(ctx, "dtm_mosaic.tif")
    arr, transform, crs = merge_geotiffs(
        paths, ctx.get("crs"), out_path, nodata=nodata, metadata=prov_doc,
        return_array=True,
    )

    # Post-merge processing: hole-fill + smooth
    if ctx.get("hole_fill_enabled", True):
        max_dist = ctx.get("hole_fill_max_dist_m", 3.0)
        if max_dist > 0:
            resolution = ctx.get("resolution", 1.0)
            arr = fill_holes(arr, resolution, max_dist, nodata)

    if ctx.get("smooth_enabled", True):
        sigma = ctx.get("smooth_sigma", 1.0)
        if sigma > 0:
            arr = smooth_nodata(arr, sigma, nodata)

    if ctx.get("crop_to_valid", True):
        from lynceus.processing.raster import _crop_to_valid

        arr, transform = _crop_to_valid(arr, transform, nodata)

    write_geotiff(out_path, arr, transform, crs=crs, nodata=nodata, metadata=prov_doc)
    return {"file": out_path, "kind": "DTM", "node": NODE_ID}


def _raster_path(tile: dict, ctx: dict, kind: str) -> str:
    from lynceus.nodes._paths import scoped_dir

    return str(scoped_dir(ctx, kind) / f"{tile['tile_id']}.tif")
