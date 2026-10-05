# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Vector product reading without geopandas/pyproj (domain, no Qt).

The GUI process must load a single geospatial CRS stack: rasterio bundles
one PROJ build while pyproj/geopandas/pyogrio bundle another, and mixing
both in one process is a known source of native crashes (access
violation inside ``proj_9`` while previewing vectors after a run).

GeoPackages are read here with ``sqlite3`` (stdlib) plus shapely for
geometries and numpy for columns; the CRS label comes straight from
``gpkg_spatial_ref_sys``, so no PROJ is involved. GeoJSON is plain JSON
plus shapely shapes. SHP is intentionally not read: the core never
produces it and reading it would pull the pyogrio stack back in.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

VECTOR_EXT = {".gpkg", ".geojson"}

NODATA = -9999.0

_GPKG_META_PREFIXES = ("gpkg_", "rtree_", "sqlite_")
_NUMERIC_TYPES = ("INT", "REAL", "FLOA", "DOUB", "DEC", "NUM")
_TEXT_TYPES = ("TEXT", "CHAR", "CLOB", "VARCHAR")


@dataclass
class VectorFeatures:
    """Tabular vector content plus parsed geometries (no CRS objects)."""

    fields: list[str]
    columns: dict[str, np.ndarray]
    numeric: set[str]
    geometries: np.ndarray | None
    bounds: tuple[float, float, float, float] | None
    crs_label: str
    feature_count: int

    def __len__(self) -> int:
        return self.feature_count


def _connect(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _first_feature_table(con: sqlite3.Connection) -> str | None:
    try:
        row = con.execute(
            "SELECT table_name FROM gpkg_contents "
            "WHERE data_type = 'features' ORDER BY table_name LIMIT 1"
        ).fetchone()
        if row:
            return str(row[0])
    except sqlite3.Error:
        pass
    try:
        for (name,) in con.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        ):
            if not str(name).lower().startswith(_GPKG_META_PREFIXES):
                return str(name)
    except sqlite3.Error:
        pass
    return None


def _geometry_column(
    con: sqlite3.Connection, table: str
) -> tuple[str | None, int | None]:
    try:
        row = con.execute(
            "SELECT column_name, srs_id FROM gpkg_geometry_columns "
            "WHERE table_name = ?",
            (table,),
        ).fetchone()
        if row:
            srs = row[1]
            return str(row[0]), int(srs) if srs is not None else None
    except sqlite3.Error:
        pass
    return None, None


def _crs_label(con: sqlite3.Connection, srs_id: int | None) -> str:
    if srs_id is None:
        return ""
    try:
        row = con.execute(
            "SELECT organization, organization_coordsys_id "
            "FROM gpkg_spatial_ref_sys WHERE srs_id = ?",
            (srs_id,),
        ).fetchone()
    except sqlite3.Error:
        return ""
    if not row:
        return ""
    org, code = row[0], row[1]
    if not org or str(org).strip().upper() in ("NONE", "UNDEFINED"):
        return ""
    if code is None:
        return str(org)
    return f"{str(org).strip()}:{int(code)}"


def _declared_fields(
    con: sqlite3.Connection, table: str
) -> list[tuple[str, str]]:
    """Non-key attribute columns in table order (the fid is bookkeeping)."""
    rows = con.execute(f'PRAGMA table_info("{table}")').fetchall()
    return [
        (str(row[1]), str(row[2] or ""))
        for row in rows
        if not int(row[5] or 0)  # pk flag: the feature id never shows
    ]


def _declared_numeric(declared_type: str) -> bool | None:
    """True/False from the declared SQL type; None when it is unknown."""
    upper = declared_type.strip().upper()
    if not upper:
        return None
    if any(token in upper for token in _NUMERIC_TYPES):
        return True
    if any(token in upper for token in _TEXT_TYPES):
        return False
    return None


def _build_column(values: list, declared: bool | None) -> tuple[np.ndarray, bool]:
    """(array, is_numeric) for one column of raw sqlite values."""
    if declared is not False:
        try:
            return (
                np.array(
                    [np.nan if v is None else float(v) for v in values],
                    dtype=np.float64,
                ),
                True,
            )
        except (TypeError, ValueError):
            if declared is True:
                # Declared numeric but holding text: degrade to strings
                # instead of failing the whole read.
                pass
    return (
        np.array(["" if v is None else str(v) for v in values], dtype=object),
        False,
    )


_GPKG_ENVELOPE_BYTES = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}


def _wkb_of(blob) -> bytes | None:
    """Raw WKB from a GeoPackage geometry BLOB.

    GeoPackage stores geometries as ``GP`` magic + flags + srs_id +
    optional envelope + WKB; shapely only understands the WKB tail.
    Plain WKB blobs (other producers) pass through untouched.
    """
    if blob is None:
        return None
    try:
        data = bytes(blob)
    except (TypeError, ValueError):
        return None
    if len(data) < 8 or data[:2] != b"GP":
        return data
    flags = data[3]
    envelope = _GPKG_ENVELOPE_BYTES.get((flags >> 1) & 0x07, 0)
    offset = 8 + envelope
    return data[offset:] if len(data) > offset else None


def _parse_geometries(blobs: list) -> np.ndarray:
    from shapely import from_wkb

    array = np.array([_wkb_of(blob) for blob in blobs], dtype=object)
    try:
        return np.asarray(from_wkb(array), dtype=object)
    except Exception:
        # Corrupt/foreign WKB: parse row by row, None on failure so a bad
        # feature never takes the preview down.
        parsed = []
        for blob in array:
            if blob is None:
                parsed.append(None)
                continue
            try:
                parsed.append(from_wkb(bytes(blob)))
            except Exception:
                parsed.append(None)
        return np.array(parsed, dtype=object)


def _bounds_of(geometries: np.ndarray | None) -> tuple | None:
    if geometries is None or len(geometries) == 0:
        return None
    from shapely import bounds

    try:
        box = bounds(geometries)
    except Exception:
        return None
    if box.size == 0:
        return None
    valid = box[~np.isnan(box).any(axis=1)]
    if valid.size == 0:
        return None
    minx = float(valid[:, 0].min())
    miny = float(valid[:, 1].min())
    maxx = float(valid[:, 2].max())
    maxy = float(valid[:, 3].max())
    return (minx, miny, maxx, maxy)


def read_gpkg_features(
    path: str | Path,
    fields: list[str] | None = None,
    with_geometry: bool = True,
) -> VectorFeatures:
    """Read the first feature table of a GeoPackage without PROJ.

    ``fields`` restricts and orders nothing — the table order is kept and
    unknown names are ignored; callers reorder as needed. With
    ``with_geometry=False`` the WKB column is never touched (statistics
    only), which is noticeably faster on dense grids.
    """
    source = Path(path)
    con = _connect(source)
    try:
        table = _first_feature_table(con)
        if table is None:
            return VectorFeatures([], {}, set(), None, None, "", 0)
        geometry_col, srs_id = _geometry_column(con, table)
        declared = _declared_fields(con, table)
        wanted = [
            name
            for name, _type in declared
            if name != geometry_col and (fields is None or name in fields)
        ]
        selected = list(wanted)
        if with_geometry and geometry_col:
            selected = [geometry_col] + selected
        if not selected:
            return VectorFeatures([], {}, set(), None, None,
                                  _crs_label(con, srs_id), 0)
        column_sql = ", ".join(f'"{name}"' for name in selected)
        rows = con.execute(f'SELECT {column_sql} FROM "{table}"').fetchall()
        crs_label = _crs_label(con, srs_id)
    finally:
        con.close()

    geometries = None
    if with_geometry and geometry_col:
        geometries = _parse_geometries([row[0] for row in rows])
        offset = 1
    else:
        offset = 0

    declared_of = dict(declared)
    columns: dict[str, np.ndarray] = {}
    numeric: set[str] = set()
    for index, name in enumerate(wanted):
        values = [row[offset + index] for row in rows]
        array, is_numeric = _build_column(values, _declared_numeric(declared_of.get(name, "")))
        columns[name] = array
        if is_numeric:
            numeric.add(name)
    return VectorFeatures(
        fields=wanted,
        columns=columns,
        numeric=numeric,
        geometries=geometries,
        bounds=_bounds_of(geometries),
        crs_label=crs_label,
        feature_count=len(rows),
    )


def _geojson_features(data: dict) -> list[dict]:
    kind = data.get("type")
    if kind == "FeatureCollection":
        features = data.get("features")
        return [f for f in features if isinstance(f, dict)] if isinstance(features, list) else []
    if kind == "Feature":
        return [data]
    return []


def read_geojson_features(
    path: str | Path,
    fields: list[str] | None = None,
    with_geometry: bool = True,
) -> VectorFeatures:
    """Read a GeoJSON FeatureCollection without PROJ."""
    from shapely.geometry import shape

    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    features = _geojson_features(data)

    order: list[str] = []
    for feature in features:
        properties = feature.get("properties")
        if not isinstance(properties, dict):
            continue
        for key in properties:
            if key not in order and (fields is None or key in fields):
                order.append(str(key))

    rows: dict[str, list] = {name: [] for name in order}
    geometries = [] if with_geometry else None
    for feature in features:
        properties = feature.get("properties")
        properties = properties if isinstance(properties, dict) else {}
        for name in order:
            rows[name].append(properties.get(name))
        if geometries is not None:
            geometry = feature.get("geometry")
            try:
                geometries.append(shape(geometry) if geometry else None)
            except Exception:
                geometries.append(None)

    columns: dict[str, np.ndarray] = {}
    numeric: set[str] = set()
    for name in order:
        array, is_numeric = _build_column(rows[name], None)
        columns[name] = array
        if is_numeric:
            numeric.add(name)

    crs_label = ""
    crs = data.get("crs")
    if isinstance(crs, dict):
        name = (crs.get("properties") or {}).get("name")
        if isinstance(name, str) and name:
            crs_label = name

    geometry_array = (
        np.array(geometries, dtype=object) if geometries is not None else None
    )
    return VectorFeatures(
        fields=order,
        columns=columns,
        numeric=numeric,
        geometries=geometry_array,
        bounds=_bounds_of(geometry_array),
        crs_label=crs_label,
        feature_count=len(features),
    )


def read_vector_features(
    path: str | Path,
    fields: list[str] | None = None,
    with_geometry: bool = True,
) -> VectorFeatures:
    """Read vector features from a GeoPackage or GeoJSON (no PROJ)."""
    suffix = Path(path).suffix.lower()
    if suffix == ".gpkg":
        return read_gpkg_features(path, fields=fields, with_geometry=with_geometry)
    if suffix == ".geojson":
        return read_geojson_features(path, fields=fields, with_geometry=with_geometry)
    raise ValueError(f"Unsupported vector format: {suffix or Path(path).name}")


def read_vector_rows(
    path: str | Path,
    limit: int | None = None,
    offset: int = 0,
) -> tuple[list[str], list[list[str]]]:
    """Attribute rows as display strings (table previews).

    Returns ``(headers, rows)``; geometry is never read and values render
    as ``""`` when null. ``limit``/``offset`` page the underlying table.
    """
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".geojson":
        features = read_geojson_features(source, with_geometry=False)
        headers = list(features.fields)
        start = max(0, int(offset))
        stop = features.feature_count if limit is None else start + max(0, int(limit))
        rows = []
        for index in range(min(start, features.feature_count), min(stop, features.feature_count)):
            display = []
            for name in headers:
                value = features.columns[name][index]
                if isinstance(value, float) and np.isnan(value):
                    display.append("")
                else:
                    display.append(str(value))
            rows.append(display)
        return headers, rows
    if suffix != ".gpkg":
        raise ValueError(f"Unsupported vector format: {suffix or source.name}")

    con = _connect(source)
    try:
        table = _first_feature_table(con)
        if table is None:
            return [], []
        geometry_col, _srs = _geometry_column(con, table)
        headers = [
            name
            for name, _type in _declared_fields(con, table)
            if name != geometry_col
        ]
        if not headers:
            return [], []
        column_sql = ", ".join(f'"{name}"' for name in headers)
        sql = f'SELECT {column_sql} FROM "{table}"'
        parameters: list[int] = []
        if limit is not None:
            sql += " LIMIT ?"
            parameters.append(max(0, int(limit)))
        if offset:
            if limit is None:
                sql += " LIMIT -1"
            sql += " OFFSET ?"
            parameters.append(max(0, int(offset)))
        rows = [
            ["" if value is None else str(value) for value in row]
            for row in con.execute(sql, parameters)
        ]
        return headers, rows
    finally:
        con.close()
