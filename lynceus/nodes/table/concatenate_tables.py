# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Node -- Concatenate Tables (stack two tables into one CSV).

Barrier-only combination node (F2): merges two tables (CSV or GeoPackage)
by concatenating their rows. Columns are the union of both headers; missing
cells are empty. Geometry columns of a GeoPackage are dropped from the CSV
output. Both inputs share the generic `table` port type, so the two paths
arrive positionally as `ctx["in0_path"]` and `ctx["in1_path"]`.
"""

from __future__ import annotations

from lynceus.nodes.ports import PortDef, PortType

NODE_ID = "lynceus.nodes.table.concatenate_tables"
NODE_NAME = "Concatenate Tables"
NODE_CATEGORY = "Table"
NODE_SUBCATEGORY = "Concatenate"

NODE_DESCRIPTION = (
    "<b>Concatenate Tables</b> -- Stack Two Tables<br><br>"
    "Concatenates two tables (CSV or GeoPackage) into a single CSV with the "
    "union of their columns.<br><br>"
    "<b>Process:</b> Both inputs are read row by row and appended; columns "
    "that only exist in one table stay empty in the rows of the other. "
    "GeoPackage geometry columns are not exported to the CSV.<br><br>"
    "<b>Tips:</b> Use this to stack two survey tables into one dataset, or to "
    "combine field inventories with plot tables before a downstream analysis."
)

INPUTS = (
    PortDef(PortType.TABLE, name="Table A"),
    PortDef(PortType.TABLE, name="Table B"),
)
OUTPUTS = (PortType.TABLE_CSV,)

PROCESSING_SPECS = {
    "barrier_task": "barrier_concatenate_tables",
    "input_ports": ("table", "table"),
    "output_files": {"table_csv": "concatenated.csv"},
    "output_port": "table_csv",
    "output_globs": ("concatenated.csv",),
    "config_schema": {
        "delimiter": {
            "type": "str",
            "default": ",",
            "description": "Column delimiter of the concatenated CSV output.",
            "impact": "Use ',' for standard CSV, ';' for locales where comma is the decimal separator, or '\\t' for tab-separated output compatible with spreadsheets.",
            "group": "Output",
        },
    },
}


def get_config_defaults() -> dict:
    return {"delimiter": ","}


def _read_csv_rows(path, delimiter: str = ",") -> tuple[list[dict], list[str]]:
    import csv

    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        headers = list(reader.fieldnames or [])
        for record in reader:
            rows.append({h: record.get(h, "") for h in headers})
    return rows, headers


def _read_gpkg_rows(path) -> tuple[list[dict], list[str]]:
    import re
    import sqlite3

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
        if not tables:
            return [], []
        # The names come from the file's own SQLite; only simple identifiers
        # are accepted when building the SELECT (no arbitrary string
        # interpolation).
        name = tables[0]
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            return [], []
        dropped = {"geom", "geometry", "geometrie"}
        columns = [
            info[1]
            for info in con.execute(f'PRAGMA table_info("{name}")').fetchall()
            if info[1] not in dropped
        ]
        con.row_factory = sqlite3.Row
        records = con.execute(f'SELECT * FROM "{name}"').fetchall()
    finally:
        con.close()
    return [
        {k: ("" if v is None else str(v)) for k, v in dict(record).items()
         if k not in dropped}
        for record in records
    ], columns


def _read_table_rows(path, delimiter: str = ",") -> tuple[list[dict], list[str]]:
    suffix = path.suffix.lower()
    if suffix == ".gpkg":
        return _read_gpkg_rows(path)
    return _read_csv_rows(path, delimiter)


def barrier_concatenate_tables(ctx: dict) -> dict:
    """Concatenate two tables into `concatenated.csv` under the session artifacts."""
    import csv
    from pathlib import Path

    from lynceus.processing import provenance

    session = Path(ctx["session_dir"])
    from lynceus.nodes._paths import scoped_file

    delimiter = ctx.get("delimiter", ",")
    prov_doc = provenance.build_provenance(ctx, "concatenated.csv")
    a_path = Path(ctx.get("in0_path", str(session / "table.csv")))
    b_path = Path(ctx.get("in1_path", str(session / "rows.gpkg")))
    if not a_path.is_file() or not b_path.is_file():
        raise RuntimeError(
            "Concatenate Tables requires two tables (Table A and Table B)"
        )
    out_path = Path(scoped_file(ctx, "concatenated.csv"))

    rows_a, headers_a = _read_table_rows(a_path, delimiter)
    rows_b, headers_b = _read_table_rows(b_path, delimiter)
    # Concatenation stacks schemas: the header is the union of both file
    # headers (A order, then B-only columns), so header-only inputs still
    # produce a valid header row. Row-only keys append defensively.
    headers: list[str] = []
    seen: set[str] = set()
    for key in list(headers_a) + list(headers_b):
        if key not in seen:
            seen.add(key)
            headers.append(key)
    for row in rows_a + rows_b:
        for key in row:
            if key not in seen:
                seen.add(key)
                headers.append(key)

    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        if headers:
            writer = csv.DictWriter(handle, fieldnames=headers,
                                    extrasaction="ignore", delimiter=delimiter)
            writer.writeheader()
            for row in rows_a + rows_b:
                writer.writerow({k: row.get(k, "") for k in headers})
        else:
            handle.write("")

    provenance.write_sidecar(out_path, prov_doc, {"delimiter": delimiter})
    total = len(rows_a) + len(rows_b)
    payload = {
        "file": str(out_path),
        "kind": "Table",
        "node": NODE_ID,
        "rows": total,
    }
    if total == 0:
        payload.setdefault("warnings", []).append(
            "Both input tables have no data rows; concatenated table is empty."
        )
    return payload