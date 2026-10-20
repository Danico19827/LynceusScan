# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Bounded-memory merges for tables and vectors (domain, no Qt).

The raster twin of this contract lives in ``raster.py`` (finish/stack/
dem windowed writers): one mechanism per data kind, the same rule — full
fit goes direct (legacy code, untouched), otherwise stream with identical
results. Heavy third-party imports (geopandas/pyogrio/pandas) stay inside
functions so importing this module never pulls them into the GUI process.
"""

from __future__ import annotations

import csv
import logging
import sqlite3
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_REDUCERS = ("sum", "min", "max", "count")
_COLUMN_KINDS = ("int64", "float64", "text")


def tables_fit(files, copies: float, budget_bytes) -> bool:
    """Whether ``copies`` temporaries over ``files`` fit a budget.

    Introspection only (sizes, no reads). Unset budget or any doubt fits
    (legacy direct behavior); the chunked writers decide by data.
    """
    try:
        budget = int(budget_bytes or 0)
    except (TypeError, ValueError):
        return True
    if budget <= 0:
        return True
    total = 0
    try:
        for path in files:
            total += Path(path).stat().st_size
    except OSError:
        return True
    return total * float(copies) <= budget


def note_streamed(ctx, product: str) -> None:
    """Record a chunked-merge notice for the barrier payload.

    Same contract as the raster writers' ``notes`` kwarg, for the table
    path where the fit check (not a writer) picks the mode: one line,
    same resolution, slower. ``ctx=None``-safe.
    """
    try:
        notes = ctx.setdefault("_bounded_notes", [])
    except AttributeError:
        return
    try:
        notes.append(
            f"Large table {product}: merged in chunks at full resolution; "
            "slower, identical rows.",
        )
    except AttributeError:
        pass


def concat_csv(files, out_path, *, delimiter: str = ",") -> dict:
    """Stream rows from CSV files into one CSV with union headers.

    Two bounded passes (headers, then rows); row dicts never accumulate.
    Returns ``{"file", "rows", "headers"}``.
    """
    files = [str(p) for p in files]
    headers: list[str] = []
    seen: set[str] = set()

    def _headers_of(path: str) -> None:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            for key in list(reader.fieldnames or []):
                if key not in seen:
                    seen.add(key)
                    headers.append(key)
            for record in reader:
                for key in record:
                    if key not in seen:
                        seen.add(key)
                        headers.append(key)

    for path in files:
        _headers_of(path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        if headers:
            writer = csv.DictWriter(handle, fieldnames=headers,
                                    extrasaction="ignore", delimiter=delimiter)
            writer.writeheader()
            for path in files:
                with open(path, newline="", encoding="utf-8-sig") as fh:
                    for record in csv.DictReader(fh, delimiter=delimiter):
                        writer.writerow(
                            {k: record.get(k, "") for k in headers})
                        total += 1
    return {"file": str(out_path), "rows": total, "headers": headers}


def _gpkg_first_table(path: str) -> str | None:
    return gpkg_first_table(path)


def gpkg_first_table(path: str) -> str | None:
    """First data table of a GPKG (validated identifier, None when none)."""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = [
            row[0]
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'gpkg_%' AND name NOT LIKE 'sqlite_%' "
                "AND name NOT LIKE 'rtree_%' ORDER BY name"
            ).fetchall()
        ]
    finally:
        con.close()
    import re

    for name in tables:
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            return name
    return None


def concat_gpkg(files, out_path, *, chunk_features: int = 200_000) -> dict:
    """Append GPKG feature tables into one GPKG with bounded RAM.

    Schema comes from the first non-empty file (tile-sized reads); every
    file then streams in ``chunk_features`` slices via pyogrio append, so
    peak is one chunk, never the merge. Returns ``{"file", "rows"}``.
    """
    import geopandas as gpd
    import pyogrio

    files = [str(p) for p in files]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    schema = None
    crs = None
    for path in files:
        try:
            info = pyogrio.read_info(path)
        except Exception:
            continue
        if int(info.get("features") or 0) <= 0:
            continue
        schema = path
        crs = info.get("crs")
        break
    if schema is None:
        raise RuntimeError("concat_gpkg: every input is empty")
    sample = gpd.read_file(schema, rows=1)
    sample.iloc[0:0].to_file(str(out_path), driver="GPKG")
    total = 0
    for path in files:
        try:
            info = pyogrio.read_info(path)
        except Exception:
            continue
        count = int(info.get("features") or 0)
        for start in range(0, count, chunk_features):
            frame = pyogrio.read_dataframe(
                path, skip_features=start, max_features=chunk_features)
            if frame is None or len(frame) == 0:
                continue
            frame = gpd.GeoDataFrame(frame, geometry="geometry", crs=crs)
            pyogrio.write_dataframe(frame, str(out_path), append=True)
            total += len(frame)
    logger.info("concat_gpkg: %d files -> %s (%d rows)",
                len(files), out_path.name, total)
    return {"file": str(out_path), "rows": total}


def stream_keyed_groups(files, *, key: str, columns: dict, reader,
                        batch_rows: int = 200_000, tmp_dir=None):
    """Yield per-key aggregates in batches with RAM bounded by batch size.

    ``columns`` maps ``name -> (kind, reducer)`` with kind in
    int64/float64/text and reducer in sum/min/max/count. ``reader(path)``
    returns a DataFrame carrying ``key`` plus clean-typed columns
    (geometry dropped by the reader; recompute it from the key
    downstream). Rows bulk-load into a temp sqlite database (pandas
    ``to_sql`` appends, then one index), and ``GROUP BY`` streams back
    out in ``batch_rows`` chunks — peak is one batch plus sqlite's page
    cache, never the merge.

    Float sums accumulate in deterministic file order but, like any
    reordered floating-point sum, may differ from a RAM concat in the
    last ulp; int/min/max aggregates are exact. Yields DataFrames with
    ``[key + agg columns]`` ordered by key.
    """
    import pandas as pd

    names = list(columns.keys())
    for name, (kind, reducer) in columns.items():
        if kind not in _COLUMN_KINDS:
            raise ValueError(f"stream_keyed_groups: bad kind {kind!r}")
        if reducer not in _REDUCERS:
            raise ValueError(f"stream_keyed_groups: bad reducer {reducer!r}")
    try:
        batch_rows = max(10_000, int(batch_rows))
    except (TypeError, ValueError):
        batch_rows = 200_000
    agg_sql = {
        "sum": "SUM",
        "min": "MIN",
        "max": "MAX",
        "count": "COUNT",
    }
    holder = None
    if tmp_dir is not None:
        group_dir = Path(tmp_dir)
        group_dir.mkdir(parents=True, exist_ok=True)
        db_path = group_dir / "keyed_groups.sqlite"
        if db_path.exists():
            db_path.unlink()
    else:
        holder = tempfile.TemporaryDirectory(prefix="keyed_groups_")
        db_path = Path(holder.name) / "groups.sqlite"
    con = sqlite3.connect(str(db_path))
    try:
        con.execute("PRAGMA journal_mode=OFF")
        con.execute("PRAGMA synchronous=OFF")
        con.execute("PRAGMA temp_store=FILE")
        first = True
        for path in files:
            frame = reader(str(path))
            if frame is None or len(frame) == 0:
                continue
            use = frame[[key, *names]]
            use.to_sql("rows", con, if_exists="append", index=False,
                       chunksize=20_000)
            first = False
        if first:
            return
        con.execute(f'CREATE INDEX key_idx ON rows ("{key}")')
        selects = ", ".join(
            f'{agg_sql[columns[name][1]]}("{name}") AS "{name}"'
            for name in names
        )
        query = (
            f'SELECT "{key}", {selects} FROM rows '
            f'GROUP BY "{key}" ORDER BY "{key}"'
        )
        for batch in pd.read_sql(query, con, chunksize=batch_rows):
            yield batch
    finally:
        try:
            con.close()
        except Exception:
            pass
        if holder is not None:
            holder.cleanup()


def init_gpkg(out_path, sample_frame) -> None:
    """Create an empty GPKG with the sample's schema/geometry/CRS."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    sample_frame.iloc[0:0].to_file(str(out_path), driver="GPKG")


def append_gpkg(out_path, frame) -> int:
    """Append a frame batch to a GPKG created by :func:`init_gpkg`."""
    import pyogrio

    if frame is None or len(frame) == 0:
        return 0
    pyogrio.write_dataframe(frame, str(out_path), append=True)
    return len(frame)
