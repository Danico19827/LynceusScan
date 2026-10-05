# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Node -- flightline coverage map from point_source_id.

Tile task: counts unique point_source_id values per cell (coverage) or point
density per cell. Barrier task: merges all tiles into a mosaic.
"""

from __future__ import annotations

from pathlib import Path

from lynceus.nodes._point_source import point_src
from lynceus.nodes.ports import PortType

NODE_ID = "lynceus.nodes.lidar.analysis.flightline_coverage"
NODE_NAME = "Flightline Coverage"
NODE_CATEGORY = "LiDAR"

NODE_SUBCATEGORY = "Analysis"

NODE_DESCRIPTION = (
    "<b>Flightline Coverage</b> -- Unique Lines Per Cell<br><br>"
    "Quality-control map of how many flight lines cover each cell. Useful to "
    "spot swath edges, drop-outs or overlapping strips in the survey.<br><br>"
    "<b>Process:</b> Counts distinct <i>point_source_id</i> values per cell "
    "(coverage mode) or the point density per cell (density mode). If the "
    "source has no <i>point_source_id</i> dimension, the map is NODATA with a "
    "warning.<br><br>"
    "<b>Tips:</b> Cells with a single line are prone to gaps and edge "
    "effects; plan extra strips there. Use <i>density</i> mode to check where "
    "point density falls below requirements."
)

INPUTS = (PortType.POINT_CLOUD,)
OUTPUTS = (PortType.COVERAGE_RASTER,)

PROCESSING_SPECS = {
    "tile_task": "tile_flightline_coverage",
    "barrier_task": "barrier_flightline_coverage",
    "input_port": "point_cloud",
    "output_files": {"coverage_raster": "coverage_mosaic.tif"},
    "output_port": "coverage_raster",
    "output_globs": ("coverage/**/*.tif", "coverage_mosaic.tif"),
    "qml": {"coverage_mosaic.tif": "coverage"},
    "config_schema": {
        "resolution": {
            "type": "float",
            "default": 1.0,
            "minimum": 0.1,
            "maximum": 10.0,
            "description": "Grid cell size in meters for the coverage raster.",
            "impact": "Smaller cells reveal finer strip details but increase noise from point sparsity. Larger cells smooth coverage over the swath.",
            "group": "Grid",
        },
        "mode": {
            "type": "str",
            "default": "coverage",
            "options": ["coverage", "density"],
            "description": "Output metric: unique lines per cell or point density per cell.",
            "impact": "coverage counts distinct flight lines (0-65535). density counts points per cell, useful to check point distribution intensity.",
            "group": "Output",
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


def tile_flightline_coverage(tile: dict, ctx: dict) -> dict:
    """Count unique point_source_id values (or density) per cell for a tile."""
    import laspy
    import numpy as np

    from lynceus.processing import provenance
    from lynceus.processing.raster import (
        CountAccumulator,
        core_mask,
        write_geotiff,
    )
    from lynceus.processing.tiler import _laz_backend

    out_path = _raster_path(tile, ctx)
    if Path(out_path).exists():
        return {"tile_id": tile["tile_id"], "node": NODE_ID, "output": out_path}

    prov_doc = provenance.build_provenance(ctx, f"{tile['tile_id']}.tif")
    cell_size = float(ctx.get("resolution", 1.0))
    mode = str(ctx.get("mode", "coverage"))
    nodata = float(ctx.get("nodata_value", -9999.0))

    src = point_src(tile, ctx)
    acc = CountAccumulator(
        tile,
        cell_size,
        memory_budget_bytes=ctx.get("_raster_grid_memory_budget_bytes"),
    )
    needs_source = mode == "coverage"
    has_source = False
    with laspy.open(src, laz_backend=_laz_backend()) as reader:
        has_source = (
            "point_source_id"
            in list(reader.header.point_format.dimension_names)
        )
        if has_source or not needs_source:
            for chunk in reader.chunk_iterator(2_000_000):
                x = np.asarray(chunk.x)
                y = np.asarray(chunk.y)
                mask = core_mask(x, y, tile)
                if not mask.any():
                    continue
                if needs_source:
                    # Native dtype (uint16 per LAS): no upcast copy.
                    sid = np.asarray(chunk.point_source_id)
                    acc.add_unique(x[mask], y[mask], sid[mask])
                else:
                    acc.add(x[mask], y[mask])

    result_mode = "unique" if needs_source else "count"
    grid, transform = acc.result(result_mode, nodata=nodata)
    degrade = needs_source and not has_source
    if degrade:
        grid = np.full_like(grid, nodata, dtype=np.float32)

    write_geotiff(
        out_path, grid, transform, crs=ctx.get("crs"), nodata=nodata,
        metadata=prov_doc,
    )
    result = {"tile_id": tile["tile_id"], "node": NODE_ID, "output": out_path}
    if degrade:
        result["warnings"] = [
            "Source has no point_source_id dimension; coverage map set to "
            "NODATA."
        ]
    return result


def barrier_flightline_coverage(ctx: dict) -> dict:
    """Merge all coverage tiles into a mosaic."""
    from lynceus.processing import provenance
    from lynceus.processing.raster import merge_geotiffs

    prov_doc = provenance.build_provenance(ctx, "coverage_mosaic.tif")
    from lynceus.nodes._paths import scoped_dir, scoped_file

    paths = sorted(scoped_dir(ctx, "coverage").glob("*.tif"))
    if not paths:
        raise RuntimeError("No coverage tiles generated")
    nodata = float(ctx.get("nodata_value", -9999.0))
    out_path = scoped_file(ctx, "coverage_mosaic.tif")
    merge_geotiffs(paths, ctx.get("crs"), out_path, nodata=nodata, metadata=prov_doc)
    return {"file": out_path, "kind": "coverage_raster", "node": NODE_ID}


def _raster_path(tile: dict, ctx: dict) -> str:
    from lynceus.nodes._paths import scoped_dir

    return str(scoped_dir(ctx, "coverage") / f"{tile['tile_id']}.tif")