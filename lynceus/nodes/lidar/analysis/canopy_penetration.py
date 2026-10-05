# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Node -- canopy penetration map from classified point cloud.

Tile task: counts total and ground (class 2) points per cell and computes the
ground/total ratio. Barrier task: merges all tiles into a mosaic.
"""

from __future__ import annotations

from pathlib import Path

from lynceus.nodes._point_source import point_src
from lynceus.nodes.ports import PortType

NODE_ID = "lynceus.nodes.lidar.analysis.canopy_penetration"
NODE_NAME = "Canopy Penetration"
NODE_CATEGORY = "LiDAR"

NODE_SUBCATEGORY = "Analysis"

NODE_DESCRIPTION = (
    "<b>Canopy Penetration</b> -- Ground-to-Total Point Ratio<br><br>"
    "Maps how much LiDAR reaches the ground: for each cell the ratio of "
    "ground (class 2) points to all points. Values near 1 mean open ground; "
    "values near 0 mean dense canopy blocking the beam.<br><br>"
    "<b>Process:</b> Ground-classified points are preferred. If the source "
    "carries no classification (or no class 2), the map is set to NODATA with "
    "a warning instead of producing misleading zeros.<br><br>"
    "<b>Tips:</b> Raise <i>hole fill distance</i> to close sparse gaps in "
    "very dense stands. Lower <i>resolution</i> merges small canopy openings "
    "into the surrounding class."
)

INPUTS = (PortType.POINT_CLOUD,)
OUTPUTS = (PortType.CANOPY_PENETRATION,)

PROCESSING_SPECS = {
    "tile_task": "tile_canopy_penetration",
    "barrier_task": "barrier_canopy_penetration",
    "input_port": "point_cloud",
    "output_files": {"canopy_penetration": "canopy_penetration_mosaic.tif"},
    "output_port": "canopy_penetration",
    "output_globs": ("canopy_penetration/**/*.tif", "canopy_penetration_mosaic.tif"),
    "requires": {"classified": True},
    "qml": {"canopy_penetration_mosaic.tif": "penetration"},
    "config_schema": {
        "resolution": {
            "type": "float",
            "default": 1.0,
            "minimum": 0.1,
            "maximum": 10.0,
            "description": "Grid cell size in meters for the penetration raster.",
            "impact": "Smaller cells capture thin canopy gaps but increase noise and processing time. Larger cells smooth small openings away.",
            "group": "Grid",
        },
        "hole_fill_enabled": {
            "type": "bool",
            "default": False,
            "description": "Enable hole-filling to close small gaps in the map.",
            "impact": "Fills isolated NODATA cells by copying the nearest valid neighbor. Disable if gaps represent real no-data zones (e.g. water).",
            "group": "Hole Filling",
        },
        "hole_fill_max_dist_m": {
            "type": "float",
            "default": 3.0,
            "minimum": 0.0,
            "maximum": 30.0,
            "description": "Maximum distance (meters) to fill holes from nearest valid cell.",
            "impact": "Larger values fill bigger gaps but may extrapolate into areas without returns. Set to 0 to disable regardless of the toggle.",
            "group": "Hole Filling",
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


def get_config_defaults() -> dict:
    return {k: v["default"] for k, v in PROCESSING_SPECS["config_schema"].items()}


def tile_canopy_penetration(tile: dict, ctx: dict) -> dict:
    """Compute ground/total ratio per cell for one tile (core-masked)."""
    import laspy
    import numpy as np

    from lynceus.processing import provenance
    from lynceus.processing.raster import (
        GROUND_CLASS,
        CountAccumulator,
        core_mask,
        write_geotiff,
    )
    from lynceus.processing.tiler import _laz_backend

    out_path = _raster_path(tile, ctx)
    if Path(out_path).exists():
        return {"tile_id": tile["tile_id"], "node": NODE_ID, "output": out_path}

    prov_doc = provenance.build_provenance(ctx, f"{tile['tile_id']}.tif")
    src = point_src(tile, ctx)
    cell_size = float(ctx.get("resolution", 1.0))
    nodata = float(ctx.get("nodata_value", -9999.0))

    grid_budget = int(ctx.get("_raster_grid_memory_budget_bytes", 0) or 0)
    accumulator_budget = grid_budget // 2 if grid_budget else None
    acc_total = CountAccumulator(
        tile, cell_size, memory_budget_bytes=accumulator_budget
    )
    acc_ground = CountAccumulator(
        tile, cell_size, memory_budget_bytes=accumulator_budget
    )

    has_class = False
    ground_found = False
    with laspy.open(src, laz_backend=_laz_backend()) as reader:
        has_class = (
            "classification"
            in reader.header.point_format.dimension_names
        )
        has_withheld = "withheld" in reader.header.point_format.dimension_names
        for chunk in reader.chunk_iterator(2_000_000):
            x = np.asarray(chunk.x)
            y = np.asarray(chunk.y)
            mask = core_mask(x, y, tile)
            if has_withheld:
                # Vendor-withheld points are excluded from metrics.
                mask = mask & ~np.asarray(chunk.withheld, dtype=bool)
            if not mask.any():
                continue
            # One index computation per chunk feeds both accumulators; the
            # ground subset reuses the already-computed indices.
            xm, ym = x[mask], y[mask]
            flat = acc_total.flat(xm, ym)
            acc_total.add(xm, ym, flat=flat)
            if has_class:
                cls_m = np.asarray(chunk.classification)[mask]
                gpos = cls_m == GROUND_CLASS
                if gpos.any():
                    ground_found = True
                    acc_ground.add(xm[gpos], ym[gpos], flat=flat[gpos])

    grid_all, transform = acc_total.result("count", nodata=nodata)
    if has_class and ground_found:
        grid_ground, _ = acc_ground.result("count", nodata=nodata)
        ratio = np.full_like(grid_all, nodata, dtype=np.float32)
        valid = grid_all != nodata
        ratio[valid] = np.clip(
            grid_ground[valid] / grid_all[valid], 0.0, 1.0
        )
        grid = ratio
    else:
        grid = np.full_like(grid_all, nodata, dtype=np.float32)

    write_geotiff(
        out_path, grid, transform, crs=ctx.get("crs"), nodata=nodata,
        metadata=prov_doc,
    )
    result = {"tile_id": tile["tile_id"], "node": NODE_ID, "output": out_path}
    if not (has_class and ground_found):
        result["warnings"] = [
            "No ground (class 2) points found; canopy penetration map set "
            "to NODATA."
        ]
    return result


def barrier_canopy_penetration(ctx: dict) -> dict:
    """Merge all penetration tiles into a mosaic (optional hole-filling)."""
    from lynceus.processing import provenance
    from lynceus.processing.raster import (
        fill_holes,
        merge_geotiffs,
        write_geotiff,
    )

    prov_doc = provenance.build_provenance(ctx, "canopy_penetration_mosaic.tif")
    from lynceus.nodes._paths import scoped_dir, scoped_file

    paths = sorted(scoped_dir(ctx, "canopy_penetration").glob("*.tif"))
    if not paths:
        raise RuntimeError("No canopy penetration tiles generated")
    nodata = float(ctx.get("nodata_value", -9999.0))
    out_path = scoped_file(ctx, "canopy_penetration_mosaic.tif")
    arr, transform, crs = merge_geotiffs(
        paths, ctx.get("crs"), out_path, nodata=nodata, metadata=prov_doc,
        return_array=True,
    )

    if ctx.get("hole_fill_enabled", False):
        max_dist = float(ctx.get("hole_fill_max_dist_m", 3.0))
        if max_dist > 0:
            resolution = float(ctx.get("resolution", 1.0))
            arr = fill_holes(arr, resolution, max_dist, nodata)
    write_geotiff(
        out_path, arr, transform, crs=crs, nodata=nodata, metadata=prov_doc
    )

    return {"file": out_path, "kind": "canopy_penetration", "node": NODE_ID}


def _raster_path(tile: dict, ctx: dict) -> str:
    from lynceus.nodes._paths import scoped_dir

    return str(scoped_dir(ctx, "canopy_penetration") / f"{tile['tile_id']}.tif")