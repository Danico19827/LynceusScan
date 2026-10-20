# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Node -- Grid Metrics (per-cell point cloud statistics).

Tile task: bins core-masked points into global grid cells (world-coordinate
keys, no global bounds needed) and accumulates partial aggregates per cell
in one streaming pass. Barrier task: merges tile tables exactly by cell key
(partial sums/min/max combine arithmetically, so tile-boundary cells that
straddle cores stay exact) into ``grid_metrics.gpkg``.

Metrics per cell: point_count, density, z_min/max/mean/std, ground_count,
ground_cover, intensity_mean/std, noise_count (classes 7+18). Intensity and
ground columns are NODATA when the source lacks those dimensions.
"""

from __future__ import annotations

from pathlib import Path

from lynceus.nodes._point_source import clean_crs, point_src
from lynceus.nodes.ports import PortType

NODE_ID = "lynceus.nodes.lidar.analysis.grid_metrics"
NODE_NAME = "Grid Metrics"
NODE_CATEGORY = "LiDAR"

NODE_SUBCATEGORY = "Analysis"

NODE_DESCRIPTION = (
    "<b>Grid Metrics</b> -- Per-Cell Point Cloud Statistics<br><br>"
    "Computes count, elevation, ground, intensity and noise statistics for "
    "every grid cell with points: density planning, quality control and "
    "features for downstream nodes.<br><br>"
    "<b>Process:</b> Core-masked points are binned into global grid cells "
    "in one streaming pass; per-tile tables merge exactly by cell key.<br><br>"
    "<b>Tips:</b> Cells without points produce no rows. Intensity and ground "
    "columns are NODATA when the source lacks those dimensions."
)

INPUTS = (PortType.POINT_CLOUD,)
OUTPUTS = (PortType.GRID_METRICS,)

METRICS_COMPUTED = (
    "point_count",
    "density",
    "z_min",
    "z_max",
    "z_mean",
    "z_std",
    "ground_count",
    "ground_cover",
    "intensity_mean",
    "intensity_std",
    "noise_count",
    "dominant_class",
    "intensity_min",
    "intensity_max",
    "z_range",
    "first_return_cover",
    "last_return_cover",
    "veg_cover",
    "building_cover",
    "water_cover",
)
"""Final metric columns, in order (gallery sidecar + viewer default)."""

BOOKKEEPING_COLS = ("cell_col", "cell_row", "gid")

PARTIAL_COLS = (
    "cell_col",
    "cell_row",
    "gid",
    "n",
    "z_sum",
    "z_sumsq",
    "z_min",
    "z_max",
    "ground",
    "noise",
    "i_sum",
    "i_sumsq",
    "i_n",
)

# ASPRS classes tracked per cell for dominant class and land covers.
# Counts travel as hidden partials (exact cross-segment merge); only the
# derived covers and dominant_class are visible finals.
CLASS_SET = (0, 1, 2, 3, 4, 5, 6, 7, 9, 17, 18)
CLASS_COLS = tuple(f"class_{v}" for v in CLASS_SET)

PROCESSING_SPECS = {
    "tile_task": "tile_grid_metrics",
    "barrier_task": "barrier_grid_metrics",
    "input_port": "point_cloud",
    "output_files": {"grid_metrics": "grid_metrics.gpkg"},
    "output_port": "grid_metrics",
    "input_globs": ("grid_metrics/**/*.gpkg",),
    "output_globs": (
        "grid_metrics/**/*.gpkg",
        "grid_metrics.gpkg",
        "grid_metrics.csv",
    ),
    "requires": {"classified": True},
    "qml": {"grid_metrics.gpkg": "density"},
    "config_schema": {
        "resolution": {
            "type": "float",
            "default": 1.0,
            "minimum": 0.1,
            "maximum": 10.0,
            "description": "Grid cell size in meters for the metrics grid.",
            "impact": "Smaller cells reveal finer density detail but increase row counts and file size.",
            "group": "Grid",
        },
        "nodata_value": {
            "type": "float",
            "default": -9999.0,
            "minimum": -99999.0,
            "maximum": 99999.0,
            "description": "Value used for missing statistics (empty dimensions).",
            "impact": "Should match the convention of your GIS software. Almost always -9999.0.",
            "group": "Advanced",
            "advanced": True,
        },
    },
}


def get_config_defaults() -> dict:
    return {k: v["default"] for k, v in PROCESSING_SPECS["config_schema"].items()}


def _cell_polygons(cols, rows, res: float) -> list:
    from shapely.geometry import box as shp_box

    return [
        shp_box(c * res, r * res, (c + 1) * res, (r + 1) * res)
        for c, r in zip(cols.tolist(), rows.tolist())
    ]


def _empty_frame(crs):
    import geopandas as gpd

    data = {
        "cell_col": gpd.pd.Series([], dtype="int64"),
        "cell_row": gpd.pd.Series([], dtype="int64"),
        "gid": gpd.pd.Series([], dtype="object"),
        "n": gpd.pd.Series([], dtype="int64"),
        "z_sum": gpd.pd.Series([], dtype="float64"),
        "z_sumsq": gpd.pd.Series([], dtype="float64"),
        "z_min": gpd.pd.Series([], dtype="float64"),
        "z_max": gpd.pd.Series([], dtype="float64"),
        "ground": gpd.pd.Series([], dtype="int64"),
        "noise": gpd.pd.Series([], dtype="int64"),
        "i_sum": gpd.pd.Series([], dtype="float64"),
        "i_sumsq": gpd.pd.Series([], dtype="float64"),
        "i_n": gpd.pd.Series([], dtype="int64"),
        "first_ret": gpd.pd.Series([], dtype="int64"),
        "last_ret": gpd.pd.Series([], dtype="int64"),
        "i_min": gpd.pd.Series([], dtype="float64"),
        "i_max": gpd.pd.Series([], dtype="float64"),
        **{f"class_{v}": gpd.pd.Series([], dtype="int64") for v in CLASS_SET},
    }
    return gpd.GeoDataFrame(data, geometry=[], crs=crs)


def tile_grid_metrics(tile: dict, ctx: dict) -> dict:
    """Accumulate per-cell partials for one tile into a GeoPackage."""
    import laspy
    import numpy as np

    import geopandas as gpd

    from lynceus.processing import provenance
    from lynceus.processing.raster import core_mask
    from lynceus.processing.tiler import _laz_backend
    from lynceus.nodes._paths import scoped_dir

    out_path = scoped_dir(ctx, "grid_metrics") / f"{tile['tile_id']}.gpkg"
    if out_path.exists():
        return {"tile_id": tile["tile_id"], "node": NODE_ID, "output": str(out_path)}

    res = float(ctx.get("resolution", 1.0))
    crs = clean_crs(ctx.get("crs"))
    src = point_src(tile, ctx)
    warnings: list[str] = []

    # Per-chunk cell aggregates; a single vectorized merge at the end
    # (no per-cell Python anywhere on the hot path).
    parts: list[tuple] = []
    has_class = False
    has_intensity = False
    has_returns = False
    with laspy.open(src, laz_backend=_laz_backend()) as reader:
        dims = set(reader.header.point_format.dimension_names)
        has_class = "classification" in dims
        has_withheld = "withheld" in dims
        has_intensity = "intensity" in dims
        has_returns = "return_number" in dims and "number_of_returns" in dims
        if not has_class:
            warnings.append(
                "Source has no classification dimension; ground and "
                "noise columns set to NODATA."
            )
        if not has_intensity:
            warnings.append(
                "Source has no intensity dimension; intensity columns "
                "set to NODATA."
            )
        for chunk in reader.chunk_iterator(2_000_000):
            x = np.asarray(chunk.x)
            y = np.asarray(chunk.y)
            mask = core_mask(x, y, tile)
            if has_withheld:
                # Vendor-withheld points are excluded from metrics.
                mask = mask & ~np.asarray(chunk.withheld, dtype=bool)
            if not mask.any():
                continue
            xm = x[mask]
            ym = y[mask]
            zm = np.asarray(chunk.z)[mask]
            cols = np.floor(xm / res).astype(np.int64)
            rows = np.floor(ym / res).astype(np.int64)
            pairs = np.stack((rows, cols), axis=1)
            ukeys, inv, _ = np.unique(
                pairs, axis=0, return_inverse=True, return_counts=True
            )
            n_cells = len(ukeys)
            zsum = np.bincount(inv, weights=zm)
            zsumsq = np.bincount(inv, weights=zm * zm)
            zmin = np.full(n_cells, np.inf)
            np.minimum.at(zmin, inv, zm)
            zmax = np.full(n_cells, -np.inf)
            np.maximum.at(zmax, inv, zm)
            if has_class:
                cls_m = np.asarray(chunk.classification)[mask]
                ground = np.bincount(inv, weights=(cls_m == 2)).astype(np.int64)
                noise = np.bincount(
                    inv, weights=((cls_m == 7) | (cls_m == 18))
                ).astype(np.int64)
                class_counts = [
                    np.bincount(inv, weights=(cls_m == v)).astype(np.int64)
                    for v in CLASS_SET
                ]
            else:
                ground = np.full(n_cells, -1, dtype=np.int64)
                noise = np.full(n_cells, -1, dtype=np.int64)
                class_counts = [
                    np.full(n_cells, -1, dtype=np.int64) for _ in CLASS_SET
                ]
            if has_intensity:
                im = np.asarray(chunk.intensity)[mask].astype(np.float64)
                isum = np.bincount(inv, weights=im)
                isumsq = np.bincount(inv, weights=im * im)
                inum = np.bincount(inv).astype(np.int64)
                imin = np.full(n_cells, np.inf)
                np.minimum.at(imin, inv, im)
                imax = np.full(n_cells, -np.inf)
                np.maximum.at(imax, inv, im)
            else:
                isum = np.zeros(n_cells)
                isumsq = np.zeros(n_cells)
                inum = np.full(n_cells, -1, dtype=np.int64)
                imin = np.full(n_cells, np.inf)
                imax = np.full(n_cells, -np.inf)
            if has_returns:
                ret = np.asarray(chunk.return_number)[mask].astype(np.int64)
                nret = np.asarray(chunk.number_of_returns)[mask].astype(np.int64)
                first = np.bincount(inv, weights=(ret == 1)).astype(np.int64)
                last = np.bincount(inv, weights=(ret == nret)).astype(np.int64)
            else:
                first = np.zeros(n_cells, dtype=np.int64)
                last = np.zeros(n_cells, dtype=np.int64)
            n = np.bincount(inv).astype(np.int64)
            parts.append(
                (ukeys, n, zsum, zsumsq, zmin, zmax, ground, noise,
                 isum, isumsq, inum, class_counts, first, last, imin, imax)
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not parts:
        _empty_frame(crs).to_file(str(out_path), driver="GPKG")
    else:
        all_keys = np.concatenate([p[0] for p in parts])
        cells, ginv, _ = np.unique(
            all_keys, axis=0, return_inverse=True, return_counts=True
        )

        def _merge_int(index: int) -> np.ndarray:
            # Sentinel -1 (dimension missing) is tile-uniform: all parts
            # carry it or none does. Summing blindly would turn -1 + -1
            # into -2 and break missing detection downstream.
            vals = np.concatenate([p[index] for p in parts])
            if bool((vals == -1).all()):
                return np.full(len(cells), -1, dtype=np.int64)
            return np.bincount(
                ginv, weights=np.where(vals >= 0, vals, 0)
            ).astype(np.int64)

        n = np.bincount(
            ginv, weights=np.concatenate([p[1] for p in parts])
        ).astype(np.int64)
        zsum = np.bincount(ginv, weights=np.concatenate([p[2] for p in parts]))
        zsumsq = np.bincount(ginv, weights=np.concatenate([p[3] for p in parts]))
        zmin = np.full(len(cells), np.inf)
        np.minimum.at(zmin, ginv, np.concatenate([p[4] for p in parts]))
        zmax = np.full(len(cells), -np.inf)
        np.maximum.at(zmax, ginv, np.concatenate([p[5] for p in parts]))
        ground = _merge_int(6)
        noise = _merge_int(7)
        isum = np.bincount(ginv, weights=np.concatenate([p[8] for p in parts]))
        isumsq = np.bincount(ginv, weights=np.concatenate([p[9] for p in parts]))
        inum = _merge_int(10)
        def _merge_class(k: int) -> np.ndarray:
            vals = np.concatenate([p[11][k] for p in parts])
            if bool((vals == -1).all()):
                return np.full(len(cells), -1, dtype=np.int64)
            return np.bincount(
                ginv, weights=np.where(vals >= 0, vals, 0)
            ).astype(np.int64)

        class_merged = [_merge_class(k) for k in range(len(CLASS_SET))]
        first = np.bincount(
            ginv, weights=np.concatenate([p[12] for p in parts])
        ).astype(np.int64)
        last = np.bincount(
            ginv, weights=np.concatenate([p[13] for p in parts])
        ).astype(np.int64)
        imin = np.full(len(cells), np.inf)
        np.minimum.at(imin, ginv, np.concatenate([p[14] for p in parts]))
        imax = np.full(len(cells), -np.inf)
        np.maximum.at(imax, ginv, np.concatenate([p[15] for p in parts]))
        cols = cells[:, 1]
        rows = cells[:, 0]
        col = {
            "cell_col": cols,
            "cell_row": rows,
            "gid": [f"{c}_{r}" for c, r in zip(cols.tolist(), rows.tolist())],
            "n": n,
            "z_sum": zsum,
            "z_sumsq": zsumsq,
            "z_min": zmin,
            "z_max": zmax,
            "ground": ground,
            "noise": noise,
            "i_sum": isum,
            "i_sumsq": isumsq,
            "i_n": inum,
            "first_ret": first,
            "last_ret": last,
            "i_min": imin,
            "i_max": imax,
        }
        for _v, _arr in zip(CLASS_SET, class_merged):
            col[f"class_{_v}"] = _arr
        gdf = gpd.GeoDataFrame(
            col,
            # Centroids, not boxes: tile partials are only ever read back
            # by the merge (never rendered), so points are cheaper to
            # write, read and concatenate than polygons.
            geometry=gpd.GeoSeries.from_xy(
                (cols + 0.5) * res, (rows + 0.5) * res, crs=crs
            ),
            crs=crs,
        )
        gdf.to_file(str(out_path), driver="GPKG")

    prov_doc = provenance.build_provenance(ctx, f"{tile['tile_id']}.gpkg")
    provenance.embed_gpkg_metadata(str(out_path), prov_doc)
    result = {"tile_id": tile["tile_id"], "node": NODE_ID, "output": str(out_path)}
    if warnings:
        result["warnings"] = warnings
    return result


GRID_AGG_COLUMNS = {
    "n": ("int64", "sum"),
    "z_sum": ("float64", "sum"),
    "z_sumsq": ("float64", "sum"),
    "z_min": ("float64", "min"),
    "z_max": ("float64", "max"),
    "ground": ("int64", "min"),
    "noise": ("int64", "min"),
    "i_sum": ("float64", "sum"),
    "i_sumsq": ("float64", "sum"),
    "i_n": ("int64", "min"),
    "first_ret": ("int64", "sum"),
    "last_ret": ("int64", "sum"),
    "i_min": ("float64", "min"),
    "i_max": ("float64", "max"),
    "__ok_ground": ("int64", "sum"),
    "__ok_noise": ("int64", "sum"),
    "__ok_i_n": ("int64", "sum"),
    **{f"class_{v}": ("int64", "sum") for v in CLASS_SET},
    **{f"__ok_class_{v}": ("int64", "sum") for v in CLASS_SET},
}
"""Keyed-aggregation spec for the streaming merge (sums/min/max only;
cell columns derive from the gid, geometry from the grid). Raw
``ground``/``noise``/``i_n`` travel with a min reducer for the -1
missing detection; ``__ok_`` clipped copies feed the sums."""


def read_grid_partials(path: str):
    """Read one tile partial table for the streaming merge.

    Returns the key plus aggregation columns (geometry dropped; ``__ok_``
    clipped copies precomputed exactly like the RAM path). Missing class
    columns default to zeros, matching the consolidate reader.
    """
    import geopandas as gpd

    frame = gpd.read_file(path)
    if frame is None or len(frame) == 0:
        return frame
    for col in ("ground", "noise", "i_n", *[f"class_{v}" for v in CLASS_SET]):
        if col not in frame.columns:
            frame[col] = 0
        frame["__ok_" + col] = frame[col].clip(lower=0)
    keep = ["gid", *[c for c in GRID_AGG_COLUMNS]]
    return frame[[c for c in keep if c in frame.columns]]


def finalize_grid_records(agg: dict, *, res: float, nodata: float) -> tuple:
    """Per-cell final metrics from merged aggregates (both merge paths).

    ``agg`` carries numpy arrays keyed by logical name (``gids``,
    ``cell_col``/``cell_row``, ``n``, ``z_sum``/``z_sumsq``/``z_min``/
    ``z_max``, ``ground_raw``/``ground_missing``, ``noise_raw``,
    ``i_sum``/``i_n``/``i_missing``/``i_sumsq``, ``first_raw``/
    ``last_raw``, ``classes`` {v: array}, ``i_min``/``i_max``). Returns
    ``(records, dims)`` with the product columns and
    ``{"ground_dim", "intensity_dim"}`` (any non-missing cell), so the
    caller builds geometry/warnings around it.
    """
    import numpy as np

    gids = agg["gids"]
    n = agg["n"].astype(np.float64)
    z_sum, z_sumsq = agg["z_sum"], agg["z_sumsq"]
    z_min, z_max = agg["z_min"], agg["z_max"]
    z_mean = z_sum / np.maximum(n, 1)
    z_var = np.maximum(z_sumsq / np.maximum(n, 1) - z_mean**2, 0.0)
    cell_area = res * res
    stacked = np.stack([agg["classes"][v] for v in CLASS_SET], axis=1)
    has_any = stacked.max(axis=1) > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        records = {
            "gid": gids,
            "cell_col": agg["cell_col"],
            "cell_row": agg["cell_row"],
            "point_count": n.astype(np.int64),
            "density": n / cell_area,
            "z_min": z_min,
            "z_max": z_max,
            "z_mean": z_mean,
            "z_std": np.sqrt(z_var),
            "z_range": z_max - z_min,
            "ground_count": np.where(
                agg["ground_missing"], nodata, agg["ground_raw"]),
            "ground_cover": np.where(
                agg["ground_missing"], nodata,
                agg["ground_raw"] / np.maximum(n, 1)),
            "intensity_mean": np.where(
                agg["i_missing"] | (agg["i_n"] <= 0), nodata,
                agg["i_sum"] / np.maximum(agg["i_n"], 1)),
            "intensity_std": np.where(
                agg["i_missing"] | (agg["i_n"] <= 0),
                nodata,
                np.sqrt(np.maximum(
                    agg["i_sumsq"] / np.maximum(agg["i_n"], 1)
                    - np.where(agg["i_missing"] | (agg["i_n"] <= 0),
                               0.0, agg["i_sum"] / np.maximum(agg["i_n"], 1)
                               ) ** 2,
                    0.0)),
            ),
            "intensity_min": np.where(
                agg["i_missing"], nodata, agg["i_min"]),
            "intensity_max": np.where(
                agg["i_missing"], nodata, agg["i_max"]),
            "noise_count": np.where(
                agg["ground_missing"], nodata, agg["noise_raw"]),
            "dominant_class": np.where(
                agg["ground_missing"] | ~has_any,
                nodata,
                np.array(CLASS_SET)[np.argmax(stacked, axis=1)],
            ).astype(np.float64),
            "first_return_cover": agg["first_raw"] / np.maximum(n, 1),
            "last_return_cover": agg["last_raw"] / np.maximum(n, 1),
            "veg_cover": np.where(
                agg["ground_missing"], nodata,
                (agg["classes"][3] + agg["classes"][4] + agg["classes"][5])
                / np.maximum(n, 1)),
            "building_cover": np.where(
                agg["ground_missing"], nodata,
                agg["classes"][6] / np.maximum(n, 1)),
            "water_cover": np.where(
                agg["ground_missing"], nodata,
                agg["classes"][9] / np.maximum(n, 1)),
            # Hidden partials for exact cross-segment merges (bookkeeping:
            # never styled, never in metrics_computed).
            "z_sumsq": z_sumsq,
            "i_sum": agg["i_sum"],
            "i_sumsq": agg["i_sumsq"],
            "i_n": agg["i_n"],
            "first_ret": agg["first_raw"].astype(np.int64),
            "last_ret": agg["last_raw"].astype(np.int64),
            "i_min": agg["i_min"],
            "i_max": agg["i_max"],
            **{f"class_{v}": agg["classes"][v] for v in CLASS_SET},
        }
    dims = {
        "ground_dim": bool((~agg["ground_missing"]).any()),
        "intensity_dim": bool((~agg["i_missing"]).any()),
    }
    return records, dims


def _write_grid_sidecar(ctx, out_path: str, nodata: float, res: float,
                        n_cells: int, metrics=None) -> None:
    """Write the product sidecar (metrics order, bookkeeping, defaults)."""
    import datetime

    from lynceus.processing import provenance

    metrics = list(metrics) if metrics is not None else list(METRICS_COMPUTED)
    prov_doc = provenance.build_provenance(ctx, "grid_metrics.gpkg")
    metadata = {
        "generated_at": datetime.datetime.now().isoformat(),
        "node_id": NODE_ID,
        "grid_size_m": res,
        "n_cells": n_cells,
        "metrics_computed": metrics,
        "default_metric": "density" if metrics else None,
        "bookkeeping_cols": list(BOOKKEEPING_COLS) + [
            "z_sumsq", "i_sum", "i_sumsq", "i_n", "first_ret", "last_ret",
            "i_min", "i_max",
            *[f"class_{v}" for v in CLASS_SET],
        ],
        "nodata": nodata,
    }
    provenance.write_sidecar(out_path, prov_doc, metadata)


def barrier_grid_metrics(ctx: dict) -> dict:
    """Merge tile tables exactly by cell key into grid_metrics.gpkg."""
    import geopandas as gpd

    from lynceus.nodes._paths import scoped_dir, scoped_file
    from lynceus.processing.chunked_merge import (
        append_gpkg,
        init_gpkg,
        note_streamed,
        stream_keyed_groups,
        tables_fit,
    )

    res = float(ctx.get("resolution", 1.0))
    nodata = float(ctx.get("nodata_value", -9999.0))
    paths = sorted(scoped_dir(ctx, "grid_metrics").glob("*.gpkg"))
    if not paths:
        raise RuntimeError("No grid metrics tiles generated")
    out_path = scoped_file(ctx, "grid_metrics.gpkg")
    budget = ctx.get("_raster_grid_memory_budget_bytes")

    def _empty_result():
        gdf = _empty_frame(clean_crs(ctx.get("crs")))
        _finalize(gdf, ctx, out_path, nodata, res, [],
                  ["No points in tile cores; grid metrics are empty."])
        return {"file": out_path, "kind": "grid_metrics", "node": NODE_ID,
                "warnings": ["No points in tile cores; grid metrics are empty."]}

    if tables_fit([str(p) for p in paths], 4.0, budget):
        frames = []
        for path in paths:
            frame = gpd.read_file(path)
            if frame is not None and len(frame) > 0:
                frames.append(frame)
        if not frames:
            return _empty_result()
        return _barrier_direct(ctx, paths, frames, out_path, res, nodata)

    # Streaming merge: same aggregates via sqlite, same finals per batch.
    note_streamed(ctx, "grid_metrics.gpkg")
    stream = stream_keyed_groups(
        [str(p) for p in paths], key="gid", columns=GRID_AGG_COLUMNS,
        reader=read_grid_partials)
    first_batch = next(stream, None)
    if first_batch is None:
        return _empty_result()
    crs = _peek_gpkg_crs([str(p) for p in paths])
    n_cells = 0
    ground_dim = False
    intensity_dim = False
    initialized = False
    batch = first_batch
    while True:
        records, dims = _batch_records(batch, res, nodata)
        gdf = gpd.GeoDataFrame(
            records, geometry=_cell_polygons(records["cell_col"],
                                             records["cell_row"], res),
            crs=crs,
        )
        if not initialized:
            init_gpkg(out_path, gdf)
            csv_path = str(Path(out_path).with_suffix(".csv"))
            gdf.drop(columns=["geometry"]).to_csv(csv_path, index=False)
            initialized = True
        else:
            gdf.drop(columns=["geometry"]).to_csv(
                csv_path, mode="a", header=False, index=False)
        append_gpkg(out_path, gdf)
        n_cells += len(gdf)
        ground_dim = ground_dim or dims["ground_dim"]
        intensity_dim = intensity_dim or dims["intensity_dim"]
        batch = next(stream, None)
        if batch is None:
            break
    warnings: list[str] = []
    if not ground_dim:
        warnings.append(
            "Source has no classification dimension; ground and noise "
            "columns set to NODATA."
        )
    if not intensity_dim:
        warnings.append(
            "Source has no intensity dimension; intensity columns set to "
            "NODATA."
        )
    _write_grid_sidecar(ctx, out_path, nodata, res, n_cells)
    result = {
        "file": out_path,
        "csv": csv_path,
        "kind": "grid_metrics",
        "node": NODE_ID,
    }
    if warnings:
        result["warnings"] = warnings
    return result


def _peek_gpkg_crs(paths: list) -> object:
    """CRS of the first readable input (cheap single-row peek)."""
    import geopandas as gpd

    for path in paths:
        try:
            return gpd.read_file(path, rows=1).crs
        except Exception:
            continue
    return None


def _batch_records(batch, res: float, nodata: float) -> tuple:
    """Finalize one streamed aggregate batch (columns parsed from gid)."""
    import numpy as np

    gids = batch["gid"].tolist()
    split = [g.rsplit("_", 1) for g in gids]
    agg = {
        "gids": gids,
        "cell_col": np.array([int(s[0]) for s in split], dtype=np.int64),
        "cell_row": np.array([int(s[1]) for s in split], dtype=np.int64),
        "n": batch["n"].to_numpy(dtype=np.float64),
        "z_sum": batch["z_sum"].to_numpy(dtype=np.float64),
        "z_sumsq": batch["z_sumsq"].to_numpy(dtype=np.float64),
        "z_min": batch["z_min"].to_numpy(dtype=np.float64),
        "z_max": batch["z_max"].to_numpy(dtype=np.float64),
        "ground_raw": batch["__ok_ground"].to_numpy(dtype=np.int64),
        "ground_missing": (batch["ground"].to_numpy() == -1),
        "noise_raw": batch["__ok_noise"].to_numpy(dtype=np.int64),
        "i_sum": batch["i_sum"].to_numpy(dtype=np.float64),
        "i_n": batch["__ok_i_n"].to_numpy(dtype=np.int64),
        "i_missing": (batch["i_n"].to_numpy() == -1),
        "i_sumsq": batch["i_sumsq"].to_numpy(dtype=np.float64),
        "first_raw": batch["first_ret"].to_numpy(dtype=np.float64),
        "last_raw": batch["last_ret"].to_numpy(dtype=np.float64),
        "classes": {
            v: batch[f"class_{v}"].to_numpy(dtype=np.int64)
            for v in CLASS_SET
        },
        "i_min": batch["i_min"].to_numpy(dtype=np.float64),
        "i_max": batch["i_max"].to_numpy(dtype=np.float64),
    }
    return finalize_grid_records(agg, res=res, nodata=nodata)


def _barrier_direct(ctx, paths, frames, out_path, res: float,
                    nodata: float) -> dict:
    """Legacy RAM merge, now sharing the finalize math (parity path)."""
    import numpy as np

    import geopandas as gpd

    merged = gpd.pd.concat(frames, ignore_index=True)
    # Sentinel -1 (dimension missing in that tile) must not pollute sums:
    # clipped copies feed the sums, min() detects all-missing groups.
    # Both are vectorized (the per-group apply lambdas were the hotspot).
    for _c in ("ground", "noise", "i_n",
               *[f"class_{v}" for v in CLASS_SET]):
        merged["__ok_" + _c] = merged[_c].clip(lower=0)
    grouped = merged.groupby("gid", sort=True)
    gids = list(grouped.groups.keys())
    first = grouped.first()

    def _col(name: str) -> np.ndarray:
        return grouped[name].sum().reindex(gids).to_numpy()

    def _min(name: str) -> np.ndarray:
        return grouped[name].min().reindex(gids).to_numpy()

    def _max(name: str) -> np.ndarray:
        return grouped[name].max().reindex(gids).to_numpy()

    def _int_col(name: str) -> np.ndarray:
        return grouped["__ok_" + name].sum().reindex(gids).to_numpy(dtype=np.int64)

    def _all_missing(name: str) -> np.ndarray:
        return (
            (grouped[name].min() == -1).reindex(gids).to_numpy(dtype=bool)
        )

    agg = {
        "gids": gids,
        "cell_col": first["cell_col"].reindex(gids).to_numpy(dtype=np.int64),
        "cell_row": first["cell_row"].reindex(gids).to_numpy(dtype=np.int64),
        "n": _col("n").astype(np.float64),
        "z_sum": _col("z_sum"),
        "z_sumsq": _col("z_sumsq"),
        "z_min": _min("z_min"),
        "z_max": _max("z_max"),
        "ground_raw": _int_col("ground"),
        "ground_missing": _all_missing("ground"),
        "noise_raw": _int_col("noise"),
        "i_sum": _col("i_sum"),
        "i_n": _int_col("i_n"),
        "i_missing": _all_missing("i_n"),
        "i_sumsq": _col("i_sumsq"),
        "first_raw": _col("first_ret"),
        "last_raw": _col("last_ret"),
        "classes": {v: _int_col(f"class_{v}") for v in CLASS_SET},
        "i_min": _min("i_min"),
        "i_max": _max("i_max"),
    }
    records, dims = finalize_grid_records(agg, res=res, nodata=nodata)
    cols = np.array(records["cell_col"])
    rows = np.array(records["cell_row"])
    gdf = gpd.GeoDataFrame(
        records, geometry=_cell_polygons(cols, rows, res),
        crs=frames[0].crs,
    )
    warnings: list[str] = []
    if not dims["ground_dim"]:
        warnings.append(
            "Source has no classification dimension; ground and noise "
            "columns set to NODATA."
        )
    if not dims["intensity_dim"]:
        warnings.append(
            "Source has no intensity dimension; intensity columns set to "
            "NODATA."
        )
    csv_path = _finalize(gdf, ctx, out_path, nodata, res, list(METRICS_COMPUTED), warnings)
    result = {
        "file": out_path,
        "csv": csv_path,
        "kind": "grid_metrics",
        "node": NODE_ID,
    }
    if warnings:
        result["warnings"] = warnings
    return result


def _finalize(gdf, ctx, out_path: str, nodata: float, res: float,
              metrics: list, warnings: list) -> str:
    """Write the final product: GPKG + CSV twin + metadata + sidecar.

    The CSV twin carries the same rows without geometry (spreadsheet/ML
    friendly); it shares the ``<basename>.meta.json`` sidecar, which
    ``write_sidecar`` merges instead of overwriting. Returns the CSV path.
    """
    from lynceus.processing import provenance

    prov_doc = provenance.build_provenance(ctx, "grid_metrics.gpkg")
    gdf.to_file(out_path, driver="GPKG")
    provenance.embed_gpkg_metadata(out_path, prov_doc)
    _write_grid_sidecar(ctx, out_path, nodata, res, len(gdf), metrics)
    csv_path = str(Path(out_path).with_suffix(".csv"))
    gdf.drop(columns=["geometry"]).to_csv(csv_path, index=False)
    return csv_path
