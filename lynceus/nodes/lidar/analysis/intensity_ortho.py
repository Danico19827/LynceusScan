# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Node -- LiDAR intensity orthophoto (mean intensity per cell).

Tile task: accumulates mean intensity per cell, optionally normalizing by the
cosine of the scan angle. Barrier task: merges tiles and applies a
NODATA-aware median filter.
"""

from __future__ import annotations

from pathlib import Path

from lynceus.nodes._point_source import point_src
from lynceus.nodes.ports import PortType

NODE_ID = "lynceus.nodes.lidar.analysis.intensity_ortho"
NODE_NAME = "Intensity Orthophoto"
NODE_CATEGORY = "LiDAR"

NODE_SUBCATEGORY = "Analysis"

NODE_DESCRIPTION = (
    "<b>Intensity Orthophoto</b> -- Mean LiDAR Intensity Per Cell<br><br>"
    "Renders the average return intensity as an ortho-like raster, useful as "
    "an optical proxy for timber volume, species contrast or surface "
    "differences across the survey.<br><br>"
    "<b>Process:</b> Per cell, the mean intensity of the returns is computed "
    "and optionally NODATA-aware median filtered. Scan-angle normalization "
    "(I / cos(theta)) compensates the darker edges of each swath; points "
    "beyond <i>max scan angle</i> are discarded.<br><br>"
    "<b>Tips:</b> Keep <i>normalize scan angle</i> on to avoid dark strip "
    "edges. Raise the <i>max scan angle</i> only if data is sparse near the "
    "swath borders."
)

INPUTS = (PortType.POINT_CLOUD,)
OUTPUTS = (PortType.INTENSITY_ORTHO,)

PROCESSING_SPECS = {
    "tile_task": "tile_intensity_ortho",
    "barrier_task": "barrier_intensity_ortho",
    "input_port": "point_cloud",
    "output_files": {"intensity_ortho": "intensity_mosaic.tif"},
    "output_port": "intensity_ortho",
    "output_globs": ("intensity/**/*.tif", "intensity_mosaic.tif"),
    "qml": {"intensity_mosaic.tif": "intensity"},
    "config_schema": {
        "resolution": {
            "type": "float",
            "default": 0.5,
            "minimum": 0.05,
            "maximum": 10.0,
            "description": "Grid cell size in meters for the intensity raster.",
            "impact": "Smaller cells give finer ortho detail but amplify point sparsity noise. For typical aerial LiDAR, 0.5-1.0 m balances detail and coverage.",
            "group": "Grid",
        },
        "normalize_scan_angle": {
            "type": "bool",
            "default": True,
            "description": "Normalize intensity by cos(scan_angle) to reduce edge-of-swath darkening.",
            "impact": "Compensates the cosine falloff at swath edges with I_corrected = I / cos(theta). Requires a scan_angle dimension; otherwise normalization is skipped.",
            "group": "Scan Angle",
        },
        "max_scan_angle_deg": {
            "type": "float",
            "default": 60.0,
            "minimum": 1.0,
            "maximum": 90.0,
            "description": "Maximum scan angle (degrees) for normalization; beyond this, points are excluded.",
            "impact": "Extreme angles carry the strongest falloff and most noise. Raising above 60 only helps if data is sparse near swath borders.",
            "group": "Scan Angle",
        },
        "median_filter_enabled": {
            "type": "bool",
            "default": True,
            "description": "Apply a NODATA-aware median filter to the merged mosaic.",
            "impact": "Removes single-pixel noise spikes while ignoring NODATA cells, so edges stay clean. Disable for the raw mean surface.",
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
    },
}


def get_config_defaults() -> dict:
    return {k: v["default"] for k, v in PROCESSING_SPECS["config_schema"].items()}


def tile_intensity_ortho(tile: dict, ctx: dict) -> dict:
    """Accumulate mean intensity per cell for one tile (core-masked)."""
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
    cell_size = float(ctx.get("resolution", 0.5))
    nodata = float(ctx.get("nodata_value", -9999.0))
    normalize = bool(ctx.get("normalize_scan_angle", True))
    max_angle_deg = float(ctx.get("max_scan_angle_deg", 60.0))

    acc = CountAccumulator(
        tile,
        cell_size,
        memory_budget_bytes=ctx.get("_raster_grid_memory_budget_bytes"),
    )
    warnings: list[str] = []
    has_intensity = False
    angle_dim: str | None = None
    min_cos = float(np.cos(np.deg2rad(max_angle_deg)))

    src = point_src(tile, ctx)

    with laspy.open(src, laz_backend=_laz_backend()) as reader:
        dims = list(reader.header.point_format.dimension_names)
        has_intensity = "intensity" in dims
        has_withheld = "withheld" in dims
        if "scan_angle" in dims:
            angle_dim = "scan_angle"
        elif "scan_angle_rank" in dims:
            angle_dim = "scan_angle_rank"
        if has_intensity:
            for chunk in reader.chunk_iterator(2_000_000):
                x = np.asarray(chunk.x)
                y = np.asarray(chunk.y)
                mask = core_mask(x, y, tile)
                if has_withheld:
                    # Vendor-withheld points are excluded from metrics.
                    mask = mask & ~np.asarray(chunk.withheld, dtype=bool)
                if not mask.any():
                    continue
                intensity = np.asarray(chunk.intensity).astype(np.float32)
                if normalize and angle_dim:
                    # Same convention as the Scan Angle cleaner: LAS 1.4
                    # scaled integers (|max| > 90) carry 0.006 units.
                    raw_angle = np.asarray(
                        getattr(chunk, angle_dim), dtype=np.float32
                    )
                    if angle_dim == "scan_angle":
                        sample = raw_angle[:min(raw_angle.size, 1 << 16)]
                        if sample.size and np.nanmax(np.abs(sample)) > 90.0:
                            raw_angle = raw_angle * np.float32(0.006)
                    keep = np.abs(raw_angle) <= max_angle_deg
                    keep &= mask
                    if not keep.any():
                        continue
                    # Exact per-angle correction via unique mapping (angles
                    # are low-cardinality integers): no per-point transcendentals.
                    uniq, inv = np.unique(raw_angle[keep], return_inverse=True)
                    cos_u = np.maximum(
                        np.cos(np.deg2rad(np.abs(uniq))).astype(np.float32),
                        np.float32(min_cos),
                    )
                    i_used = intensity[keep] / cos_u[inv]
                    acc.add_sum(x[keep], y[keep], i_used)
                else:
                    x_used = x[mask]
                    y_used = y[mask]
                    acc.add_sum(x_used, y_used, intensity[mask])

    if not has_intensity:
        warnings.append(
            "Source has no intensity dimension; intensity orthophoto set to "
            "NODATA."
        )
    elif normalize and angle_dim is None:
        warnings.append(
            "Source has no scan_angle/scan_angle_rank dimension; intensity "
            "normalization skipped."
        )

    grid, transform = acc.result("mean", nodata=nodata)
    if not has_intensity:
        grid = np.full_like(grid, nodata, dtype=np.float32)

    write_geotiff(
        out_path, grid, transform, crs=ctx.get("crs"), nodata=nodata,
        metadata=prov_doc,
    )
    result = {"tile_id": tile["tile_id"], "node": NODE_ID, "output": out_path}
    if warnings:
        result["warnings"] = warnings
    return result


def barrier_intensity_ortho(ctx: dict) -> dict:
    """Merge intensity tiles and apply a NODATA-aware median filter."""
    from lynceus.processing import provenance
    from lynceus.processing.raster import (
        median_nodata,
        merge_geotiffs,
        write_geotiff,
    )

    prov_doc = provenance.build_provenance(ctx, "intensity_mosaic.tif")
    from lynceus.nodes._paths import scoped_dir, scoped_file

    paths = sorted(scoped_dir(ctx, "intensity").glob("*.tif"))
    if not paths:
        raise RuntimeError("No intensity tiles generated")
    nodata = float(ctx.get("nodata_value", -9999.0))
    out_path = scoped_file(ctx, "intensity_mosaic.tif")
    arr, transform, crs = merge_geotiffs(
        paths, ctx.get("crs"), out_path, nodata=nodata, metadata=prov_doc,
        return_array=True,
    )

    if ctx.get("median_filter_enabled", True):
        arr = median_nodata(arr, size=3, nodata=nodata)
    write_geotiff(out_path, arr, transform, crs=crs, nodata=nodata, metadata=prov_doc)

    return {"file": out_path, "kind": "intensity_ortho", "node": NODE_ID}


def _raster_path(tile: dict, ctx: dict) -> str:
    from lynceus.nodes._paths import scoped_dir

    return str(scoped_dir(ctx, "intensity") / f"{tile['tile_id']}.tif")