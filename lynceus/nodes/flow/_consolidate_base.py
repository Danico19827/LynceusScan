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
import sqlite3
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
    from lynceus.processing.raster import finish_mosaic

    files = _segment_files(task_ctx, input_type)
    if not files:
        raise RuntimeError(f"Consolidation has no segment mosaics for {input_type}")
    out = _out_path(task_ctx, session_file)
    nodata = float(task_ctx.get("nodata_value", nodata_default))
    finish_mosaic(
        files, str(out), crs=clean_crs(task_ctx.get("crs")), nodata=nodata,
        metadata=_consolidate_doc(task_ctx, session_file),
        memory_budget_bytes=task_ctx.get("_raster_grid_memory_budget_bytes"),
        notes=task_ctx.setdefault("_bounded_notes", []),
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
    exact concatenation of segment tables is their union. Chunked append
    keeps peak at one slice, never the merge.
    """
    from lynceus.processing.chunked_merge import concat_gpkg

    files = _segment_files(task_ctx, input_type)
    if not files:
        raise RuntimeError(f"Consolidation has no segment tables for {input_type}")
    out = _out_path(task_ctx, session_file)
    try:
        res = concat_gpkg([str(p) for p in files], str(out))
    except RuntimeError:
        raise RuntimeError("Segment tables are empty; nothing to consolidate")
    if res["rows"] == 0:
        raise RuntimeError("Segment tables are empty; nothing to consolidate")
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
    """Concatenate per-segment CSV/GeoPackage tables to a single table.

    Two bounded passes (union headers, then rows): CSV rows stream via
    the stdlib reader and GPKG rows via a read-only sqlite cursor, so
    peak is one row batch, never the merge.
    """
    import csv

    files = _segment_files(task_ctx, input_type)
    if not files:
        raise RuntimeError(f"Consolidation has no segment tables for {input_type}")
    delimiter = task_ctx.get("delimiter", delimiter_default)
    columns: list[str] = []
    seen: set[str] = set()
    for path in files:
        for column in _table_headers(path, delimiter):
            if column not in seen:
                seen.add(column)
                columns.append(column)
    out = _out_path(task_ctx, session_file)
    total = 0
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns,
                                extrasaction="ignore", delimiter=delimiter)
        writer.writeheader()
        for path in files:
            for row in _table_rows(path, delimiter):
                writer.writerow({k: row.get(k, "") for k in columns})
                total += 1
    if total == 0:
        raise RuntimeError("Segment tables are empty; nothing to consolidate")
    from lynceus.processing import provenance

    provenance.write_sidecar(
        str(out),
        _consolidate_doc(task_ctx, session_file),
        {"delimiter": delimiter},
    )
    return _consolidate_payload(out, task_ctx)


def _gpkg_dropped_columns(path: str, table: str) -> set[str]:
    """Non-attribute columns of a GPKG feature table (geometry et al)."""
    dropped = {"geom", "geometry", "geometrie"}
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        try:
            for row in con.execute(
                "SELECT column_name FROM gpkg_geometry_columns "
                'WHERE table_name = ?', (table,)).fetchall():
                if row[0]:
                    dropped.add(row[0])
        except Exception:
            pass
    finally:
        con.close()
    return dropped


def _table_headers(path, delimiter: str = ",") -> list[str]:
    """Union-able headers of a CSV/GPKG table without reading its rows."""
    import csv

    from lynceus.processing.chunked_merge import gpkg_first_table

    if str(path).lower().endswith(".gpkg"):
        table = gpkg_first_table(str(path))
        if table is None:
            return []
        dropped = _gpkg_dropped_columns(str(path), table)
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            return [
                info[1]
                for info in con.execute(
                    f'PRAGMA table_info("{table}")').fetchall()
                if info[1] not in dropped
            ]
        finally:
            con.close()
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        fields = list(reader.fieldnames or [])
        for record in reader:
            for key in record:
                if key not in fields:
                    fields.append(key)
        return fields


def _table_rows(path, delimiter: str = ","):
    """Yield string row dicts (geometry dropped, None as empty)."""
    import csv

    from lynceus.processing.chunked_merge import gpkg_first_table

    if str(path).lower().endswith(".gpkg"):
        table = gpkg_first_table(str(path))
        if table is None:
            return
        dropped = _gpkg_dropped_columns(str(path), table)
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            con.row_factory = sqlite3.Row
            cursor = con.execute(f'SELECT * FROM "{table}"')
            cols = [d[0] for d in cursor.description
                    if d[0] not in dropped]
            for record in cursor:
                yield {k: ("" if record[k] is None else str(record[k]))
                       for k in cols}
        finally:
            con.close()
        return
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for record in reader:
            yield record


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
    Full fit goes direct (legacy code); otherwise the same aggregates
    stream through sqlite with the shared grid finalize.
    """
    from lynceus.processing.chunked_merge import note_streamed, tables_fit

    files = _segment_files(task_ctx, input_type)
    if not files:
        raise RuntimeError("Consolidation has no segment tables for grid metrics")
    budget = task_ctx.get("_raster_grid_memory_budget_bytes")
    if tables_fit([str(p) for p in files], 4.0, budget):
        import geopandas as gpd

        frames = []
        for path in files:
            frame = gpd.read_file(path)
            if frame is not None and len(frame) > 0:
                frames.append(frame)
        if not frames:
            raise RuntimeError("Segment tables are empty; nothing to consolidate")
        return _consolidate_grid_direct(
            task_ctx, files, frames, session_file, input_type)
    note_streamed(task_ctx, session_file)
    return _consolidate_grid_stream(
        task_ctx, files, session_file, input_type)


def read_segment_partials(path: str):
    """Read one segment final table for the streaming merge.

    Segment finals carry means, not sums: the exact partial ``_z_w`` is
    rebuilt by weighting with point counts (additive downstream), and
    ``__ok_`` clipped copies mirror the RAM path. Missing class columns
    default to zeros.
    """
    import numpy as np

    import geopandas as gpd

    frame = gpd.read_file(path)
    if frame is None or len(frame) == 0:
        return frame
    frame["_z_w"] = (
        frame["z_mean"].to_numpy(dtype=np.float64)
        * frame["point_count"].to_numpy(dtype=np.float64)
    )
    for _c in ("ground_count", "noise_count", "i_n"):
        if _c not in frame.columns:
            frame[_c] = 0
        frame["__ok_" + _c] = frame[_c].clip(lower=0)
    for _v in (0, 1, 2, 3, 4, 5, 6, 7, 9, 17, 18):
        _col = f"class_{_v}"
        if _col not in frame.columns:
            frame[_col] = 0
        frame["__ok_" + _col] = frame[_col].clip(lower=0)
    keep = ["gid", "point_count", "_z_w", "z_sumsq", "z_min", "z_max",
            "ground_count", "noise_count", "i_sum", "i_sumsq", "i_n",
            "first_ret", "last_ret", "i_min", "i_max",
            "__ok_ground_count", "__ok_noise_count", "__ok_i_n",
            *[f"__ok_class_{v}" for v in (0, 1, 2, 3, 4, 5, 6, 7, 9, 17, 18)]]
    return frame[[c for c in keep if c in frame.columns]]


SEG_AGG_COLUMNS = {
    "point_count": ("int64", "sum"),
    "_z_w": ("float64", "sum"),
    "z_sumsq": ("float64", "sum"),
    "z_min": ("float64", "min"),
    "z_max": ("float64", "max"),
    "ground_count": ("int64", "min"),
    "noise_count": ("int64", "sum"),
    "i_sum": ("float64", "sum"),
    "i_sumsq": ("float64", "sum"),
    "i_n": ("int64", "min"),
    "first_ret": ("int64", "sum"),
    "last_ret": ("int64", "sum"),
    "i_min": ("float64", "min"),
    "i_max": ("float64", "max"),
    "__ok_ground_count": ("int64", "sum"),
    "__ok_noise_count": ("int64", "sum"),
    "__ok_i_n": ("int64", "sum"),
    **{f"__ok_class_{v}": ("int64", "sum")
       for v in (0, 1, 2, 3, 4, 5, 6, 7, 9, 17, 18)},
}
"""Keyed-aggregation spec for segment finals (same reducers as the tile
spec; raw ``noise_count`` needs no missing detection, only its sum)."""


def _consolidate_grid_direct(task_ctx, files, frames, session_file,
                             input_type) -> dict:
    """Legacy RAM merge, sharing the grid finalize (parity path)."""
    import numpy as np

    import geopandas as gpd

    from lynceus.nodes.lidar.analysis.grid_metrics import finalize_grid_records

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
    agg = {
        "gids": gids,
        "cell_col": first["cell_col"].reindex(gids).to_numpy(dtype=np.int64),
        "cell_row": first["cell_row"].reindex(gids).to_numpy(dtype=np.int64),
        "n": _sum("point_count").astype(np.float64),
        "z_sum": _sum("_z_w"),
        "z_sumsq": _sum("z_sumsq"),
        "z_min": _min("z_min"),
        "z_max": _max("z_max"),
        "ground_raw": _int_sum("ground_count"),
        "ground_missing": _all_missing("ground_count", nodata),
        "noise_raw": _int_sum("noise_count"),
        "i_sum": _sum("i_sum"),
        "i_n": _int_sum("i_n"),
        "i_missing": _all_missing("i_n", -1),
        "i_sumsq": _sum("i_sumsq"),
        "first_raw": _sum("first_ret"),
        "last_raw": _sum("last_ret"),
        "classes": {v: _int_sum(f"class_{v}")
                    for v in (0, 1, 2, 3, 4, 5, 6, 7, 9, 17, 18)},
        "i_min": _min("i_min"),
        "i_max": _max("i_max"),
    }
    records, _ = finalize_grid_records(agg, res=res, nodata=nodata)
    # Cells are identical boxes across segments: reuse them instead of
    # rebuilding thousands of polygons.
    geoms = list(first.geometry.reindex(gids))
    merged_out = gpd.GeoDataFrame(records, geometry=geoms, crs=frames[0].crs)
    return _write_consolidate_grid(
        task_ctx, merged_out, session_file, input_type)


def _peek_feature(path: str):
    """First feature of a table (cheap CRS/bounds peek, one row)."""
    import geopandas as gpd

    try:
        one = gpd.read_file(path, rows=1)
    except Exception:
        return None
    if one is None or len(one) == 0:
        return None
    return one


def _consolidate_grid_stream(task_ctx, files, session_file,
                             input_type) -> dict:
    """Streaming keyed merge: same aggregates, same finalize, per batch."""
    import numpy as np

    import geopandas as gpd

    from lynceus.nodes.lidar.analysis.grid_metrics import (
        _cell_polygons,
        finalize_grid_records,
    )
    from lynceus.processing.chunked_merge import (
        append_gpkg,
        init_gpkg,
        stream_keyed_groups,
    )

    nodata = _sidecar_meta(files[0]).get("nodata")
    nodata = float(nodata) if nodata is not None else float(
        task_ctx.get("nodata_value", -9999.0)
    )
    peek = None
    for path in files:
        peek = _peek_feature(str(path))
        if peek is not None:
            break
    if peek is None:
        raise RuntimeError("Segment tables are empty; nothing to consolidate")
    bounds = peek.geometry.iloc[0].bounds
    res = abs(bounds[2] - bounds[0]) or 1.0
    crs = peek.crs
    stream = stream_keyed_groups(
        [str(p) for p in files], key="gid", columns=SEG_AGG_COLUMNS,
        reader=read_segment_partials)
    first_batch = next(stream, None)
    if first_batch is None:
        raise RuntimeError("Segment tables are empty; nothing to consolidate")
    out = _out_path(task_ctx, session_file)
    csv_path = str(Path(out).with_suffix(".csv"))
    initialized = False
    batch = first_batch
    while True:
        gids = batch["gid"].tolist()
        split = [g.rsplit("_", 1) for g in gids]
        cols = np.array([int(s[0]) for s in split], dtype=np.int64)
        rows = np.array([int(s[1]) for s in split], dtype=np.int64)
        agg = {
            "gids": gids,
            "cell_col": cols,
            "cell_row": rows,
            "n": batch["point_count"].to_numpy(dtype=np.float64),
            "z_sum": batch["_z_w"].to_numpy(dtype=np.float64),
            "z_sumsq": batch["z_sumsq"].to_numpy(dtype=np.float64),
            "z_min": batch["z_min"].to_numpy(dtype=np.float64),
            "z_max": batch["z_max"].to_numpy(dtype=np.float64),
            "ground_raw": batch["__ok_ground_count"].to_numpy(dtype=np.int64),
            "ground_missing": (batch["ground_count"].to_numpy() == nodata),
            "noise_raw": batch["__ok_noise_count"].to_numpy(dtype=np.int64),
            "i_sum": batch["i_sum"].to_numpy(dtype=np.float64),
            "i_n": batch["__ok_i_n"].to_numpy(dtype=np.int64),
            "i_missing": (batch["i_n"].to_numpy() == -1),
            "i_sumsq": batch["i_sumsq"].to_numpy(dtype=np.float64),
            "first_raw": batch["first_ret"].to_numpy(dtype=np.float64),
            "last_raw": batch["last_ret"].to_numpy(dtype=np.float64),
            "classes": {v: batch[f"__ok_class_{v}"].to_numpy(dtype=np.int64)
                        for v in (0, 1, 2, 3, 4, 5, 6, 7, 9, 17, 18)},
            "i_min": batch["i_min"].to_numpy(dtype=np.float64),
            "i_max": batch["i_max"].to_numpy(dtype=np.float64),
        }
        records, _ = finalize_grid_records(agg, res=res, nodata=nodata)
        gdf = gpd.GeoDataFrame(
            records, geometry=_cell_polygons(cols, rows, res), crs=crs)
        if not initialized:
            init_gpkg(out, gdf)
            gdf.drop(columns=["geometry"]).to_csv(csv_path, index=False)
            initialized = True
        else:
            gdf.drop(columns=["geometry"]).to_csv(
                csv_path, mode="a", header=False, index=False)
        append_gpkg(out, gdf)
        batch = next(stream, None)
        if batch is None:
            break
    from lynceus.processing import provenance

    prov_doc = _consolidate_doc(task_ctx, session_file)
    provenance.embed_gpkg_metadata(str(out), prov_doc)
    provenance.write_sidecar(
        str(out), prov_doc, _segment_node_meta(task_ctx, input_type)
    )
    payload = _consolidate_payload(out, task_ctx)
    payload["csv"] = csv_path
    return payload


def _write_consolidate_grid(task_ctx, merged_out, session_file,
                            input_type) -> dict:
    """Shared product tail: GPKG + CSV twin + provenance + payload."""
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