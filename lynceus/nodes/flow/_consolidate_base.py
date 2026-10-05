# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Consolidation framework (no NODE_ID: the AST discovery ignores this module).

A Consolidate instance waits for every segment of a segmented run, merges
its provider's per-segment product into a single final one, and publishes
it so downstream nodes continue processing on the consolidated result. It
is deferred by the controller (`consolidator` capability): it never runs
inside a segment session; a final barrier-only session runs it against
every segment's output payload (injected via `ctx["segment_outputs"]`).

Each strategy variant merges exactly one product type through its own
thin wrapper (mosaic / keyed-grid / vector / table below). The controller
injects `ctx["segment_providers_by_port"]` (provider instance IDs per
output port type) so each merge resolves its per-segment files
independently.
"""

from __future__ import annotations

import os
from pathlib import Path

from lynceus.nodes._point_source import clean_crs
from lynceus.nodes.ports import can_connect, port_metadata
from lynceus.plugins.registry import manager


def _port_id(port) -> str:
    """Normalize a port value to its canonical string ID."""
    if hasattr(port, "value"):
        return port.value
    return str(port)


def _provider_types(task_ctx: dict, iid: str) -> dict[str, list[str]]:
    """Provider instance IDs per output port type for this instance."""
    by_port = task_ctx.get("segment_providers_by_port") or {}
    return dict(by_port.get(iid) or {})


def _input_connected(task_ctx: dict, input_type: str) -> bool:
    """Whether the given input type has a segment provider or a chained path."""
    iid = str(task_ctx.get("node_iid") or "")
    for port_id, providers in _provider_types(task_ctx, iid).items():
        if providers and can_connect(port_id, input_type):
            return True
    for key, value in task_ctx.items():
        if not isinstance(value, str) or not value:
            continue
        if not key.endswith("_path"):
            continue
        if key.startswith("in"):
            if _position_type(task_ctx, key) == input_type:
                return True
            continue
        if can_connect(key[:-5], input_type):
            return True
    return False


def _position_type(task_ctx: dict, key: str) -> str | None:
    """Resolve ``in<pos>_path`` to its input port type (module metadata)."""
    pos = key[2:-5]
    if not pos.isdigit():
        return None
    try:
        inputs, _ = manager.load_ports(task_ctx.get("module_id") or "")
    except Exception:
        return None
    items = list(inputs)
    idx = int(pos)
    if idx < 0 or idx >= len(items):
        return None
    return _port_id(port_metadata(items[idx]).port_type)


def _segment_files(task_ctx: dict, input_type: str) -> list[str]:
    """The provider product of every segment for the given input type.

    Provider edges are trimmed from the deferred subgraph, so the merge reads
    the per-segment output payloads: each payload's ``file`` is the session
    path of the provider product in that segment's directory (last segment
    product of a same-type multi-provider wins per segment). Deferred
    providers (other consolidators merged earlier in the final session, edge
    kept in the deferred subgraph) contribute their already-merged product via
    the injected ``*_path`` ctx keys.
    """
    iid = str(task_ctx.get("node_iid") or "")
    providers: list[str] = []
    for port_id, pids in _provider_types(task_ctx, iid).items():
        if pids and can_connect(port_id, input_type):
            providers.extend(pids)
    segments = task_ctx.get("segment_outputs") or []
    files: list[str] = []
    for seg in segments:
        found = ""
        for provider in providers:
            payload = seg.get(provider) or {}
            candidate = payload.get("file")
            if candidate:
                found = str(candidate)
                break
        if not found:
            if providers:
                raise RuntimeError(
                    f"Consolidation missing a segment product (node {iid}, "
                    f"providers {providers}, input {input_type})"
                )
            continue
        if not os.path.isfile(found):
            raise RuntimeError(f"Consolidation product not found: {found}")
        files.append(found)
    session_dir = task_ctx.get("session_dir") or ""
    for key in sorted(task_ctx):
        if not key.endswith("_path"):
            continue
        candidate = task_ctx[key]
        if not isinstance(candidate, str) or not candidate:
            continue
        if not os.path.isfile(candidate):
            continue
        if not session_dir or not candidate.startswith(str(session_dir)):
            continue
        port_id = key[:-5]
        if key.startswith("in"):
            port_id = _position_type(task_ctx, key) or port_id
        if not can_connect(port_id, input_type):
            continue
        if candidate not in files:
            files.append(candidate)
    if not files:
        raise RuntimeError(
            f"Consolidation has no provider products to merge (node {iid}, "
            f"input {input_type})"
        )
    return files


def _out_path(task_ctx: dict, session_file: str) -> Path:
    """iid-scoped artifact path so downstream path-by-port resolves it."""
    node_dir = Path(task_ctx["session_dir"]) / str(task_ctx.get("node_iid") or "")
    node_dir.mkdir(parents=True, exist_ok=True)
    return node_dir / session_file


def _consolidate_payload(out: Path, task_ctx: dict) -> dict:
    return {
        "file": str(out),
        "kind": "Consolidated",
        "node": task_ctx.get("module_id") or "",
    }


def _sidecar_meta(path) -> dict:
    """Parsed ``<file>.meta.json`` sidecar ({} when missing/unreadable)."""
    import json

    try:
        data = json.loads(Path(path).with_suffix(".meta.json").read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _first_segment_meta(task_ctx: dict, input_type: str) -> dict:
    """First parseable sidecar of the segment products ({} if none).

    Segment sidecars carry the producer's parameters (nodata, grid size,
    thresholds), so the merge honors them instead of re-asking the user
    on the consolidator instance.
    """
    for path in _segment_files(task_ctx, input_type):
        data = _sidecar_meta(path)
        stripped = {k: v for k, v in data.items() if k != "provenance"}
        if stripped:
            return data
    return {}


def _segment_node_meta(task_ctx: dict, input_type: str) -> dict:
    """Node-level sidecar keys carried from the first segment product.

    Grid/vector viewers read ``metrics_computed``/``bookkeeping_cols``/
    ``default_metric`` from ``<file>.meta.json``; without them the merged
    product loses metric order and default. Provenance itself is rebuilt
    (never copied from a segment).
    """
    data = _first_segment_meta(task_ctx, input_type)
    data = {k: v for k, v in data.items() if k != "provenance"}
    return data if data else {}


def _consolidate_doc(task_ctx: dict, session_file: str) -> dict:
    from lynceus.processing import provenance

    return provenance.build_provenance(task_ctx, session_file)


def consolidate_mosaic(
    task_ctx: dict,
    *,
    session_file: str,
    input_type: str,
    nodata_default: float = -9999.0,
) -> dict:
    """Merge per-segment GeoTIFF mosaics into one (aligned, NODATA preserved)."""
    from lynceus.processing.raster import merge_geotiffs

    files = _segment_files(task_ctx, input_type)
    if not files:
        raise RuntimeError(f"Consolidation has no segment mosaics for {input_type}")
    out = _out_path(task_ctx, session_file)
    nodata = float(task_ctx.get("nodata_value", nodata_default))
    merge_geotiffs(
        files, clean_crs(task_ctx.get("crs")), str(out), nodata=nodata,
        metadata=_consolidate_doc(task_ctx, session_file),
    )
    return _consolidate_payload(out, task_ctx)


def consolidate_vectors(
    task_ctx: dict,
    *,
    session_file: str,
    input_type: str,
) -> dict:
    """Concatenate per-segment vector/GPKG feature tables exactly by cell.

    Grid metrics use globally-keyed cells (cell_row/cell_col/gid), so the
    exact concatenation of segment tables is their union.
    """
    import geopandas as gpd

    files = _segment_files(task_ctx, input_type)
    if not files:
        raise RuntimeError(f"Consolidation has no segment tables for {input_type}")
    frames = []
    for path in files:
        frame = gpd.read_file(path)
        if frame is not None and len(frame) > 0:
            frames.append(frame)
    if not frames:
        raise RuntimeError("Segment tables are empty; nothing to consolidate")
    merged = gpd.pd.concat(frames, ignore_index=True)
    merged.crs = frames[0].crs
    out = _out_path(task_ctx, session_file)
    merged.to_file(str(out), driver="GPKG")
    from lynceus.processing import provenance

    prov_doc = _consolidate_doc(task_ctx, session_file)
    provenance.embed_gpkg_metadata(str(out), prov_doc)
    provenance.write_sidecar(
        str(out), prov_doc, _segment_node_meta(task_ctx, input_type)
    )
    return _consolidate_payload(out, task_ctx)


def consolidate_tables(
    task_ctx: dict,
    *,
    session_file: str,
    input_type: str,
    delimiter_default: str = ",",
) -> dict:
    """Concatenate per-segment CSV/GeoPackage tables to a single table."""
    import csv

    files = _segment_files(task_ctx, input_type)
    if not files:
        raise RuntimeError(f"Consolidation has no segment tables for {input_type}")
    delimiter = task_ctx.get("delimiter", delimiter_default)
    columns: list[str] = []
    seen: set[str] = set()
    rows: list[dict] = []
    for path in files:
        if str(path).lower().endswith(".gpkg"):
            import geopandas as gpd

            frame = gpd.read_file(path)
            if frame is None or len(frame) == 0:
                continue
            geom = frame.geometry.name
            if geom in frame.columns:
                frame = frame.drop(columns=[geom])
            for row in frame.to_dict("records"):
                for column in row.keys():
                    if column not in seen:
                        seen.add(column)
                        columns.append(column)
                rows.append(row)
            continue
        with open(path, newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle, delimiter=delimiter):
                for column in row.keys():
                    if column not in seen:
                        seen.add(column)
                        columns.append(column)
                rows.append(row)
    if not rows:
        raise RuntimeError("Segment tables are empty; nothing to consolidate")
    out = _out_path(task_ctx, session_file)
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter=delimiter)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    from lynceus.processing import provenance

    provenance.write_sidecar(
        str(out),
        _consolidate_doc(task_ctx, session_file),
        {"delimiter": delimiter},
    )
    return _consolidate_payload(out, task_ctx)


def consolidate_grid_metrics(
    task_ctx: dict,
    *,
    session_file: str,
    input_type: str,
) -> dict:
    """Merge per-segment grid-metric tables exactly by cell key.

    Grid cells are global: a cell straddling a segment border is emitted as
    partial rows by each segment, so blind concatenation would duplicate it
    (visible as a seam of off-valued cells). Segment tables carry hidden
    partial columns (``z_sumsq``, ``i_sum``, ``i_n``); only the finals
    (means, covers, densities) are recomputed here, arithmetically exact.
    """
    import numpy as np

    import geopandas as gpd

    files = _segment_files(task_ctx, input_type)
    if not files:
        raise RuntimeError("Consolidation has no segment tables for grid metrics")
    frames = []
    for path in files:
        frame = gpd.read_file(path)
        if frame is not None and len(frame) > 0:
            frames.append(frame)
    if not frames:
        raise RuntimeError("Segment tables are empty; nothing to consolidate")
    merged = gpd.pd.concat(frames, ignore_index=True)
    # Segment finals carry means, not sums: rebuild exact partials by
    # weighting with point counts (z_sum/z_sumsq/i_sum/i_n stay additive).
    merged["_z_w"] = merged["z_mean"] * merged["point_count"]
    # Sentinel columns (-1, or NODATA floats) must not pollute sums mixed
    # with real counts: clipped copies feed the sums, min() detects the
    # all-missing groups. All vectorized (no per-group lambdas).
    for _c in ("ground_count", "noise_count", "i_n"):
        merged["__ok_" + _c] = merged[_c].clip(lower=0)
    # Class counts may be absent in older segment files: default to zeros
    # (missing-dim detection still uses ground_count/i_n sentinels).
    for _v in (0, 1, 2, 3, 4, 5, 6, 7, 9, 17, 18):
        _col = f"class_{_v}"
        if _col not in merged.columns:
            merged[_col] = 0
        merged["__ok_" + _col] = merged[_col].clip(lower=0)
    for _c in ("first_ret", "last_ret", "i_min", "i_max"):
        if _c not in merged.columns:
            merged[_c] = 0
    grouped = merged.groupby("gid", sort=True)
    gids = list(grouped.groups.keys())
    first = grouped.first()

    def _sum(name: str) -> np.ndarray:
        return grouped[name].sum().reindex(gids).to_numpy()

    def _min(name: str) -> np.ndarray:
        return grouped[name].min().reindex(gids).to_numpy()

    def _max(name: str) -> np.ndarray:
        return grouped[name].max().reindex(gids).to_numpy()

    def _int_sum(name: str) -> np.ndarray:
        return grouped["__ok_" + name].sum().reindex(gids).to_numpy(dtype=np.int64)

    def _all_missing(name: str, sentinel) -> np.ndarray:
        return (
            (grouped[name].min() == sentinel).reindex(gids).to_numpy(dtype=bool)
        )

    nodata = _sidecar_meta(files[0]).get("nodata")
    nodata = float(nodata) if nodata is not None else float(
        task_ctx.get("nodata_value", -9999.0)
    )
    # Resolution comes from the cells themselves (square boxes); segment
    # finals carry no grid parameter.
    bounds = frames[0].geometry.iloc[0].bounds
    res = abs(bounds[2] - bounds[0]) or 1.0
    cell_area = res * res
    n = _sum("point_count").astype(np.float64)
    z_sum = _sum("_z_w")
    z_sumsq = _sum("z_sumsq")
    z_min = _min("z_min")
    z_max = _max("z_max")
    z_mean = z_sum / np.maximum(n, 1)
    z_var = np.maximum(z_sumsq / np.maximum(n, 1) - z_mean**2, 0.0)
    ground_raw = _int_sum("ground_count")
    # Segment finals store NODATA (not -1) for missing dims; a real count
    # is never negative so the equality is exact.
    ground_missing = _all_missing("ground_count", nodata)
    noise_raw = _int_sum("noise_count")
    i_sum = _sum("i_sum")
    i_n = _int_sum("i_n")
    i_missing = _all_missing("i_n", -1)
    i_sumsq = _sum("i_sumsq")
    i_min = _min("i_min")
    i_max = _max("i_max")
    first_raw = _sum("first_ret")
    last_raw = _sum("last_ret")
    class_merged = {v: _int_sum(f"class_{v}") for v in (0, 1, 2, 3, 4, 5, 6, 7, 9, 17, 18)}

    cols = first["cell_col"].reindex(gids).to_numpy(dtype=np.int64)
    rows = first["cell_row"].reindex(gids).to_numpy(dtype=np.int64)
    with np.errstate(invalid="ignore", divide="ignore"):
        ground_cover = np.where(
            ground_missing, nodata, ground_raw / np.maximum(n, 1)
        )
        intensity_mean = np.where(
            i_missing | (i_n <= 0), nodata, i_sum / np.maximum(i_n, 1)
        )
        i_mean = np.where(i_missing | (i_n <= 0), 0.0, i_sum / np.maximum(i_n, 1))
        intensity_std = np.where(
            i_missing | (i_n <= 0),
            nodata,
            np.sqrt(np.maximum(i_sumsq / np.maximum(i_n, 1) - i_mean**2, 0.0)),
        )
        first_cover = first_raw / np.maximum(n, 1)
        last_cover = last_raw / np.maximum(n, 1)
        veg = class_merged[3] + class_merged[4] + class_merged[5]
        veg_cover = np.where(ground_missing, nodata, veg / np.maximum(n, 1))
        building_cover = np.where(
            ground_missing, nodata, class_merged[6] / np.maximum(n, 1)
        )
        water_cover = np.where(
            ground_missing, nodata, class_merged[9] / np.maximum(n, 1)
        )
        stacked = np.stack(
            [class_merged[v] for v in (0, 1, 2, 3, 4, 5, 6, 7, 9, 17, 18)], axis=1
        )
        has_any = stacked.max(axis=1) > 0
        dominant = np.where(
            ground_missing | ~has_any,
            nodata,
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 17, 18])[
                np.argmax(stacked, axis=1)
            ],
        ).astype(np.float64)
    records = {
        "gid": gids,
        "cell_col": cols,
        "cell_row": rows,
        "point_count": n.astype(np.int64),
        "density": n / cell_area,
        "z_min": z_min,
        "z_max": z_max,
        "z_mean": z_mean,
        "z_std": np.sqrt(z_var),
        "z_range": z_max - z_min,
        "ground_count": np.where(ground_missing, nodata, ground_raw),
        "ground_cover": ground_cover,
        "intensity_mean": intensity_mean,
        "intensity_std": intensity_std,
        "intensity_min": np.where(i_missing, nodata, i_min),
        "intensity_max": np.where(i_missing, nodata, i_max),
        "noise_count": np.where(ground_missing, nodata, noise_raw),
        "dominant_class": dominant,
        "first_return_cover": first_cover,
        "last_return_cover": last_cover,
        "veg_cover": veg_cover,
        "building_cover": building_cover,
        "water_cover": water_cover,
        "z_sumsq": z_sumsq,
        "i_sum": i_sum,
        "i_sumsq": i_sumsq,
        "i_n": i_n,
        "first_ret": first_raw.astype(np.int64),
        "last_ret": last_raw.astype(np.int64),
        "i_min": i_min,
        "i_max": i_max,
        **{f"class_{v}": class_merged[v] for v in (0, 1, 2, 3, 4, 5, 6, 7, 9, 17, 18)},
    }
    # Cells are identical boxes across segments: reuse them instead of
    # rebuilding thousands of polygons.
    geoms = list(first.geometry.reindex(gids))
    merged_out = gpd.GeoDataFrame(records, geometry=geoms, crs=frames[0].crs)
    out = _out_path(task_ctx, session_file)
    merged_out.to_file(str(out), driver="GPKG")
    csv_path = str(Path(out).with_suffix(".csv"))
    merged_out.drop(columns=["geometry"]).to_csv(csv_path, index=False)
    from lynceus.processing import provenance

    prov_doc = _consolidate_doc(task_ctx, session_file)
    provenance.embed_gpkg_metadata(str(out), prov_doc)
    provenance.write_sidecar(
        str(out), prov_doc, _segment_node_meta(task_ctx, input_type)
    )
    payload = _consolidate_payload(out, task_ctx)
    payload["csv"] = csv_path
    return payload