# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Node -- Grid Vegetation (per-cell vegetation statistics from points).

Tile task: normalizes heights against a wired DTM mosaic when the optional
DTM input is connected (per-point ground sampling, exact on any slope),
and falls back to the tile ground median otherwise (self-contained, no
DTM needed, approximate on slopes). Vegetation points are binned into
global grid cells with 1 m histograms. Tiles without ground points
produce no partials (there is no datum to normalize against). Barrier
task: merges partials exactly by cell key and derives distributional
metrics (percentiles, LAI, VCI, FHD, stratum densities, L-skewness).

Histogram-estimated metrics (percentiles and everything derived from the
distribution shape) are approximations at 1 m binning, consistent with the
histogram metrics of area-based forestry. Counts, sums, min/max and covers
are exact.
"""

from __future__ import annotations

from pathlib import Path

from lynceus.nodes._point_source import clean_crs, point_src
from lynceus.nodes.ports import PortDef, PortType

NODE_ID = "lynceus.nodes.lidar.analysis.grid_vegetation"
NODE_NAME = "Grid Vegetation"
NODE_CATEGORY = "LiDAR"

NODE_SUBCATEGORY = "Analysis"

NODE_DESCRIPTION = (
    "<b>Grid Vegetation</b> -- Per-Cell Vegetation Statistics<br><br>"
    "Computes vegetation structure per grid cell from normalized point "
    "heights: percentiles, canopy cover, LAI, VCI, FHD and stratum "
    "densities. No CHM needed: heights normalize by the tile ground "
    "median.<br><br>"
    "<b>Process:</b> Vegetation points are binned into global grid cells "
    "with 1 m histograms in one streaming pass; tables merge exactly by "
    "cell key.<br><br>"
    "<b>Tips:</b> Percentiles and diversity indices are histogram estimates "
    "(1 m bins). Cells without vegetation produce no rows."
)

INPUTS = (PortType.POINT_CLOUD, PortDef(PortType.DTM_MOSAIC, required=False))
OUTPUTS = (PortType.GRID_VEGETATION,)

HIST_BINS = 61
HIST_MAX_M = 60.0

METRICS_COMPUTED = (
    "point_count",
    "cover",
    "lai",
    "vci",
    "fhd",
    "shannon_h",
    "density_0_2m",
    "density_2_10m",
    "density_above_10m",
    "l_skewness",
    "p10",
    "p25",
    "p50",
    "p75",
    "p90",
    "p95",
    "p99",
    "ground_count",
)
"""Final metric columns, in order (gallery sidecar + viewer default)."""

BOOKKEEPING_COLS = ("cell_col", "cell_row", "gid")

PERCENTILES = (10, 25, 50, 75, 90, 95, 99)

# Fixed height strata (meters, normalized): same conventions as
# area-based forestry so values stay comparable between plots.
STRATA = ((0.0, 2.0), (2.0, 10.0), (10.0, float("inf")))

PROCESSING_SPECS = {
    "tile_task": "tile_grid_vegetation",
    "barrier_task": "barrier_grid_vegetation",
    "input_port": "point_cloud",
    "output_files": {"grid_vegetation": "grid_vegetation.gpkg"},
    "output_port": "grid_vegetation",
    "output_globs": (
        "grid_vegetation/**/*.gpkg",
        "grid_vegetation.gpkg",
        "grid_vegetation.csv",
    ),
    "requires": {"classified": True},
    "qml": {"grid_vegetation.gpkg": "cover"},
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
        "canopy_threshold_m": {
            "type": "float",
            "default": 2.0,
            "minimum": 0.0,
            "maximum": 50.0,
            "description": "Minimum height (meters) to consider a point as canopy.",
            "impact": "Used by canopy_cover and LAI calculations. Lower values include more of the understory as 'canopy', increasing cover estimates. Higher values restrict to dominant canopy only.",
            "group": "Vegetation",
        },
        "lai_k": {
            "type": "float",
            "default": 0.5,
            "minimum": 0.1,
            "maximum": 2.0,
            "description": "Beer-Lambert extinction coefficient (k) for LAI estimation.",
            "impact": "Controls the relationship between canopy cover and LAI. Lower k values produce higher LAI estimates for the same cover. Typical range for forests: 0.4-0.7.",
            "group": "Vegetation",
        },
        "vegetation_classes": {
            "type": "str",
            "default": "",
            "description": "LAS classes counted as vegetation (comma-separated).",
            "impact": "Empty (default) counts every class except ground (2) and noise (7, 18). List specific classes (e.g. 3,4,5) to restrict vegetation to them; class 2 is always excluded and counted as ground instead.",
            "group": "Vegetation",
        },
    },
}


def get_config_defaults() -> dict:
    return {k: v["default"] for k, v in PROCESSING_SPECS["config_schema"].items()}


def _parse_vegetation_classes(raw: str) -> set[int] | None:
    """None = default rule (everything except ground/noise)."""
    text = (raw or "").strip()
    if not text:
        return None
    out: set[int] = set()
    for token in text.split(","):
        token = token.strip()
        if token:
            out.add(int(token))
    out.discard(2)
    return out


def _vegetation_mask(cls, allowed):
    import numpy as np

    if allowed is None:
        return (cls != 2) & (cls != 7) & (cls != 18)
    keep = np.zeros(cls.shape, dtype=bool)
    for value in allowed:
        keep |= cls == value
    return keep


def _empty_frame(crs):
    import geopandas as gpd

    data = {
        "cell_col": gpd.pd.Series([], dtype="int64"),
        "cell_row": gpd.pd.Series([], dtype="int64"),
        "gid": gpd.pd.Series([], dtype="object"),
        "n": gpd.pd.Series([], dtype="int64"),
        "ground": gpd.pd.Series([], dtype="int64"),
        **{f"h{b}": gpd.pd.Series([], dtype="int64") for b in range(HIST_BINS)},
    }
    return gpd.GeoDataFrame(data, geometry=[], crs=crs)


def tile_grid_vegetation(tile: dict, ctx: dict) -> dict:
    """Accumulate normalized vegetation histograms per cell (partials)."""
    import laspy
    import numpy as np

    from lynceus.processing import provenance
    from lynceus.processing.raster import core_mask
    from lynceus.processing.tiler import _laz_backend
    from lynceus.nodes._paths import scoped_dir

    out_path = scoped_dir(ctx, "grid_vegetation") / f"{tile['tile_id']}.gpkg"
    if out_path.exists():
        return {"tile_id": tile["tile_id"], "node": NODE_ID, "output": str(out_path)}

    res = float(ctx.get("resolution", 1.0))
    crs = clean_crs(ctx.get("crs"))
    src = point_src(tile, ctx)
    warnings: list[str] = []
    try:
        allowed = _parse_vegetation_classes(str(ctx.get("vegetation_classes", "")))
    except ValueError:
        raise ValueError(
            "Grid Vegetation: invalid vegetation_classes "
            f"{ctx.get('vegetation_classes')!r} (comma-separated integers)"
        )

    parts: list[tuple] = []
    ground_z: list = []
    ground_total = 0
    with laspy.open(src, laz_backend=_laz_backend()) as reader:
        dims = set(reader.header.point_format.dimension_names)
        has_class = "classification" in dims
        has_withheld = "withheld" in dims
        if not has_class:
            warnings.append(
                "Source has no classification dimension; vegetation "
                "columns set to NODATA."
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
            # Compact retention (~25 B/pt): the tile ground median is only
            # known after the full pass, so binning happens below.
            xm = x[mask]
            ym = y[mask]
            zm = np.asarray(chunk.z)[mask]
            if has_class:
                cls_m = np.asarray(chunk.classification)[mask]
                gz = zm[cls_m == 2]
                if gz.size:
                    ground_z.append(gz)
                ground_total += int((cls_m == 2).sum())
            else:
                cls_m = np.zeros(zm.shape, dtype=np.uint8)
            parts.append((xm, ym, zm, cls_m))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dropped = 0
    if not parts or ground_total <= 0:
        if parts and ground_total <= 0:
            warnings.append(
                "No ground points in tile; no vegetation partials written."
            )
        _empty_frame(crs).to_file(str(out_path), driver="GPKG")
    else:
        median = float(np.median(np.concatenate(ground_z)))
        base_z, dropped, dtm_warning = _dtm_reference(parts, tile, ctx)
        if dtm_warning is not None:
            warnings.append(dtm_warning)
        frame, extra_dropped = _partials_to_frame(
            parts, median, res, crs, allowed, base_z=base_z,
        )
        frame.to_file(str(out_path), driver="GPKG")
        dropped += extra_dropped

    prov_doc = provenance.build_provenance(ctx, f"{tile['tile_id']}.gpkg")
    provenance.embed_gpkg_metadata(str(out_path), prov_doc)
    result = {"tile_id": tile["tile_id"], "node": NODE_ID, "output": str(out_path)}
    if dropped:
        result["unnormalized_dropped"] = dropped
    if warnings:
        result["warnings"] = warnings
    return result


def _dtm_reference(parts: list, tile: dict, ctx: dict):
    """Per-point ground heights from the wired DTM mosaic.

    Optional DTM input: sample the provider mosaic (injected as
    ``dtm_mosaic_path``) at every retained point with a tile-windowed
    read. Returns ``(base_arrays, dropped, warning)`` aligned with
    ``parts``; ``(None, 0, None)`` when unwired or unreadable, in which
    case the legacy tile-median reference applies. Points outside DTM
    coverage carry no valid datum and are dropped (counted, never warned
    per tile to avoid status-bar spam).
    """
    import numpy as np

    dtm_path = ctx.get("dtm_mosaic_path", "") or ""
    if not dtm_path:
        return None, 0, None
    from pathlib import Path

    if not Path(dtm_path).is_file():
        return None, 0, "DTM product missing; tile-median reference used"
    try:
        import rasterio
    except Exception:
        return None, 0, "DTM sampling unavailable; tile-median reference used"
    try:
        base = []
        dropped = 0
        with rasterio.open(str(dtm_path)) as src:
            tr = src.transform
            if tr.b != 0 or tr.d != 0:
                return None, 0, "Rotated DTM unsupported; tile-median used"
            nodata = src.nodata
            height, width = src.shape
            for xm, ym, _zm, _cls in parts:
                if xm.size == 0:
                    base.append(np.full(0, np.nan))
                    continue
                cols_g = np.floor(
                    (np.asarray(xm, dtype=np.float64) - tr.c) / tr.a
                ).astype(np.int64)
                if tr.e > 0:
                    rows_g = np.floor(
                        (np.asarray(ym, dtype=np.float64) - tr.f) / tr.e
                    ).astype(np.int64)
                else:
                    rows_g = np.floor(
                        (tr.f - np.asarray(ym, dtype=np.float64)) / (-tr.e)
                    ).astype(np.int64)
                # One integer window per chunk part (points plus one cell
                # of margin); global indices minus the window offsets.
                c0 = max(int(cols_g.min()) - 1, 0)
                r0 = max(int(rows_g.min()) - 1, 0)
                c1 = min(int(cols_g.max()) + 1, width - 1)
                r1 = min(int(rows_g.max()) + 1, height - 1)
                if c1 < c0 or r1 < r0:
                    base.append(np.full(xm.shape, np.nan))
                    dropped += int(xm.size)
                    continue
                from rasterio.windows import Window

                band = src.read(
                    1, window=Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1)
                ).astype(np.float64)
                rows = rows_g - r0
                cols = cols_g - c0
                ref = np.full(xm.shape, np.nan)
                ok = (
                    (rows >= 0) & (rows < band.shape[0])
                    & (cols >= 0) & (cols < band.shape[1])
                )
                ref[ok] = band[rows[ok], cols[ok]]
                if nodata is not None:
                    ref[ref == float(nodata)] = np.nan
                dropped += int(np.isnan(ref).sum())
                base.append(ref)
    except Exception:
        return None, 0, "DTM sampling failed; tile-median reference used"
    return base, dropped, None


def merge_vegetation_frames(frames):
    """Merge tile/segment partial tables into exact per-cell histograms.

    Shared by the barrier (tiles of one session) and the consolidate
    variant (segment finals): both carry ``h0..h60`` partial columns, so
    one code path merges either. Returns ``(gids, cols, rows, hists,
    n)`` with histograms summed exactly by cell key.
    """
    import geopandas as gpd
    import numpy as np

    merged = gpd.pd.concat(frames, ignore_index=True)
    grouped = merged.groupby("gid", sort=True)
    gids = list(grouped.groups.keys())
    first = grouped.first()
    hcols = [f"h{b}" for b in range(HIST_BINS)]
    hists = np.stack(
        [grouped[c].sum().reindex(gids).to_numpy(dtype=np.int64) for c in hcols],
        axis=1,
    )
    n = hists.sum(axis=1).astype(np.int64)
    cols = first["cell_col"].reindex(gids).to_numpy(dtype=np.int64)
    rows = first["cell_row"].reindex(gids).to_numpy(dtype=np.int64)
    return gids, cols, rows, hists, n


def finalize_vegetation_frame(gids, cols, rows, hists, n, res: float,
                              threshold_m: float, lai_k: float,
                              nodata: float):
    """Distributional finals from merged histograms (fully vectorized)."""
    import numpy as np

    totals = np.maximum(n.astype(np.float64), 1)
    canopy_bin = min(max(int(threshold_m), 0), HIST_BINS - 1)
    canopy = hists[:, canopy_bin:].sum(axis=1).astype(np.float64)
    cover = canopy / totals
    with np.errstate(invalid="ignore", divide="ignore"):
        lai = np.where(
            cover <= 0, 0.0,
            np.minimum(5.0, -np.log(np.maximum(1.0 - cover, 1e-9)) / lai_k),
        )
        vci = 1.0 - cover
        props = hists / totals[:, None]
        safe = np.where(hists > 0, props, 1.0)
        fhd = -np.sum(np.where(hists > 0, safe * np.log(safe), 0.0), axis=1)
        s3 = np.stack([
            hists[:, 0:2].sum(axis=1),
            hists[:, 2:10].sum(axis=1),
            hists[:, 10:].sum(axis=1),
        ], axis=1).astype(np.float64)
        s3 = s3 / totals[:, None]
        s3safe = np.where(s3 > 0, s3, 1.0)
        shannon_h = -np.sum(np.where(s3 > 0, s3safe * np.log(s3safe), 0.0), axis=1)
        density_0_2m = hists[:, 0:2].sum(axis=1) / totals
        density_2_10m = hists[:, 2:10].sum(axis=1) / totals
        density_above_10m = hists[:, 10:].sum(axis=1) / totals

        def _pct(p: float) -> np.ndarray:
            target = p / 100.0 * totals
            cum = np.cumsum(hists, axis=1)
            idx = np.argmax(cum >= target[:, None], axis=1)
            idx = np.clip(idx, 0, HIST_BINS - 1)
            prev = np.where(
                idx > 0, cum[np.arange(len(cum)), np.maximum(idx - 1, 0)], 0.0
            )
            width = hists[np.arange(len(cum)), idx].astype(np.float64)
            frac = np.where(width > 0, (target - prev) / np.maximum(width, 1e-300), 0.5)
            return idx + np.clip(frac, 0.0, 1.0)

        percentiles = {p: _pct(float(p)) for p in PERCENTILES}
        p10, p50, p90 = percentiles[10], percentiles[50], percentiles[90]
        span = p90 - p10
        l_skewness = np.where(span > 0, (p90 - 2 * p50 + p10) / np.maximum(span, 1e-300), 0.0)
    records = {
        "gid": gids,
        "cell_col": cols,
        "cell_row": rows,
        "point_count": n,
        "cover": cover,
        "lai": lai,
        "vci": vci,
        "fhd": fhd,
        "shannon_h": shannon_h,
        "density_0_2m": density_0_2m,
        "density_2_10m": density_2_10m,
        "density_above_10m": density_above_10m,
        "l_skewness": l_skewness,
        **{f"p{p}": percentiles[p] for p in PERCENTILES},
    }
    return records


def barrier_grid_vegetation(ctx: dict) -> dict:
    """Merge tile partials into vegetation finals + CSV twin."""
    import geopandas as gpd

    from lynceus.nodes._paths import scoped_dir, scoped_file

    res = float(ctx.get("resolution", 1.0))
    nodata = float(ctx.get("nodata_value", -9999.0))
    threshold_m = float(ctx.get("canopy_threshold_m", 2.0))
    lai_k = float(ctx.get("lai_k", 0.5))
    paths = sorted(scoped_dir(ctx, "grid_vegetation").glob("*.gpkg"))
    if not paths:
        raise RuntimeError("No grid vegetation tiles generated")
    frames = []
    for path in paths:
        frame = gpd.read_file(path)
        if frame is not None and len(frame) > 0:
            frames.append(frame)
    out_path = scoped_file(ctx, "grid_vegetation.gpkg")
    if not frames:
        gdf = _empty_frame(clean_crs(ctx.get("crs")))
        csv_path = _finalize_vegetation(
            gdf, ctx, out_path, nodata, res,
            ["No vegetation points in tile cores; grid vegetation is empty."],
        )
        return {"file": out_path, "csv": csv_path, "kind": "grid_vegetation",
                "node": NODE_ID,
                "warnings": ["No vegetation points in tile cores; grid vegetation is empty."]}

    gids, cols, rows, hists, n = merge_vegetation_frames(frames)
    finals = finalize_vegetation_frame(
        gids, cols, rows, hists, n, res, threshold_m, lai_k, nodata
    )
    gdf = gpd.GeoDataFrame(
        finals,
        geometry=_veg_polygons(cols, rows, res),
        crs=frames[0].crs,
    )
    csv_path = _finalize_vegetation(
        gdf, ctx, out_path, nodata, res, [], n_cells=len(gdf)
    )
    return {"file": out_path, "csv": csv_path, "kind": "grid_vegetation",
            "node": NODE_ID}


def _veg_polygons(cols, rows, res):
    import geopandas as gpd
    from shapely.geometry import box as shp_box

    return gpd.GeoSeries(
        [shp_box(c * res, r * res, (c + 1) * res, (r + 1) * res)
         for c, r in zip(cols.tolist(), rows.tolist())]
    )


def _finalize_vegetation(gdf, ctx, out_path: str, nodata: float, res: float,
                         warnings: list, n_cells: int | None = None) -> str:
    """Write the final product: GPKG + CSV twin + metadata + sidecar."""
    import datetime

    from lynceus.processing import provenance

    prov_doc = provenance.build_provenance(ctx, "grid_vegetation.gpkg")
    gdf.to_file(out_path, driver="GPKG")
    provenance.embed_gpkg_metadata(out_path, prov_doc)
    metadata = {
        "generated_at": datetime.datetime.now().isoformat(),
        "node_id": NODE_ID,
        "grid_size_m": res,
        "n_cells": len(gdf) if n_cells is None else n_cells,
        "metrics_computed": list(METRICS_COMPUTED),
        "default_metric": "cover",
        "bookkeeping_cols": list(BOOKKEEPING_COLS) + [
            f"h{b}" for b in range(HIST_BINS)
        ],
        "nodata": nodata,
        "canopy_threshold_m": float(ctx.get("canopy_threshold_m", 2.0)),
        "lai_k": float(ctx.get("lai_k", 0.5)),
    }
    provenance.write_sidecar(out_path, prov_doc, metadata)
    csv_path = str(Path(out_path).with_suffix(".csv"))
    gdf.drop(columns=["geometry"]).to_csv(csv_path, index=False)
    return csv_path


def _partials_to_frame(parts: list, median: float, res: float, crs,
                       allowed, base_z=None):
    """Bin retained chunk arrays into normalized per-cell histograms.

    Two vectorized levels, no per-cell Python on the hot path: per chunk,
    one bincount over (cell, bin) combined keys; at the end a single
    unique + one bincount per bin merges chunks. ``base_z`` (optional,
    aligned with ``parts``) carries per-point ground heights sampled from
    a wired DTM; without it heights normalize by the tile ground median.
    Points without a valid datum are dropped. Returns ``(frame,
    dropped)``.
    """
    import numpy as np

    import geopandas as gpd

    chunk_cells: list[tuple] = []
    dropped = 0
    refs = base_z if base_z is not None else [None] * len(parts)
    for (xm, ym, zm, cls_m), ref in zip(parts, refs):
        veg = _vegetation_mask(cls_m, allowed)
        if ref is None:
            hv = zm[veg] - median
        else:
            hv = zm[veg] - np.asarray(ref)[veg]
            keep_h = ~np.isnan(hv)
            dropped += int((~keep_h).sum())
            if not keep_h.all():
                veg_idx = np.flatnonzero(veg)
                veg = np.zeros(veg.shape, dtype=bool)
                veg[veg_idx[keep_h]] = True
                hv = hv[keep_h]
        xv, yv = xm[veg], ym[veg]
        if xv.size == 0:
            continue
        cols = np.floor(xv / res).astype(np.int64)
        rows = np.floor(yv / res).astype(np.int64)
        ukeys, inv, _ = np.unique(
            np.stack((rows, cols), axis=1),
            axis=0, return_inverse=True, return_counts=True,
        )
        hb = np.floor(np.clip(hv, 0.0, HIST_MAX_M)).astype(np.int64)
        hb = np.minimum(hb, HIST_BINS - 1)
        combo = inv.astype(np.int64) * HIST_BINS + hb
        bc = np.bincount(combo, minlength=len(ukeys) * HIST_BINS)
        chunk_cells.append((ukeys, bc.reshape(len(ukeys), HIST_BINS)))
    if not chunk_cells:
        return _empty_frame(crs), dropped
    all_keys = np.concatenate([u for u, _ in chunk_cells])
    cells, ginv, _ = np.unique(
        all_keys, axis=0, return_inverse=True, return_counts=True
    )
    n_cells = len(cells)
    hists = np.zeros((n_cells, HIST_BINS), dtype=np.int64)
    for b in range(HIST_BINS):
        hists[:, b] = np.bincount(
            ginv,
            weights=np.concatenate([bc[:, b] for _, bc in chunk_cells]),
            minlength=n_cells,
        ).astype(np.int64)
    cols = cells[:, 1]
    rows_idx = cells[:, 0]
    col = {
        "cell_col": cols,
        "cell_row": rows_idx,
        "gid": [f"{c}_{r}" for c, r in zip(cols.tolist(), rows_idx.tolist())],
        "n": hists.sum(axis=1).astype(np.int64),
        **{f"h{b}": hists[:, b].astype(np.int64) for b in range(HIST_BINS)},
    }
    gdf = gpd.GeoDataFrame(
        col,
        geometry=gpd.GeoSeries.from_xy(
            (cols + 0.5) * res, (rows_idx + 0.5) * res, crs=crs
        ),
        crs=crs,
    )
    return gdf, dropped
