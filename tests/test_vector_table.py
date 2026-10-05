# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""PROJ-free vector reader (sqlite3 + shapely + numpy).

Builds synthetic GeoPackages/GeoJSON with stdlib tools only: the reader
must never need geopandas/pyproj/pyogrio to expose fields, values, CRS
labels, geometries or paged rows.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
from shapely import wkb
from shapely.geometry import box

from lynceus.processing.vector_table import (
    VectorFeatures,
    read_gpkg_features,
    read_geojson_features,
    read_vector_features,
    read_vector_rows,
)


def _write_gpkg(path: Path, rows: list[tuple]) -> Path:
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE gpkg_spatial_ref_sys (
            srs_name TEXT, srs_id INTEGER PRIMARY KEY, organization TEXT,
            organization_coordsys_id INTEGER, definition TEXT, description TEXT);
        CREATE TABLE gpkg_contents (
            table_name TEXT PRIMARY KEY, data_type TEXT, identifier TEXT,
            description TEXT, last_change TEXT, min_x REAL, min_y REAL,
            max_x REAL, max_y REAL, srs_id INTEGER);
        CREATE TABLE gpkg_geometry_columns (
            table_name TEXT, column_name TEXT, geometry_type_name TEXT,
            srs_id INTEGER, z TINYINT, m TINYINT);
        CREATE TABLE cells (
            fid INTEGER PRIMARY KEY, geom BLOB,
            cover REAL, point_count INTEGER, label TEXT);
        INSERT INTO gpkg_spatial_ref_sys VALUES
            ('POSGAR 2007 / Argentina 2', 5346, 'EPSG', 5346, '', '');
        INSERT INTO gpkg_contents VALUES
            ('cells', 'features', 'cells', '', '', 0, 0, 2, 1, 5346);
        INSERT INTO gpkg_geometry_columns VALUES
            ('cells', 'geom', 'POLYGON', 5346, 0, 0);
        """
    )
    con.executemany(
        "INSERT INTO cells (fid, geom, cover, point_count, label) "
        "VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    con.commit()
    con.close()
    return path


def _row(fid, geom, cover, count, label):
    blob = None if geom is None else wkb.dumps(geom)
    return (fid, blob, cover, count, label)


def _gpb(geom, envelope: bool = False) -> bytes:
    """GeoPackage geometry BLOB: GP header + srs_id (+envelope) + WKB."""
    import struct

    flags = 0x01 | ((0x01 << 1) if envelope else 0)
    header = b"GP" + bytes([0, flags]) + struct.pack("<i", 5346)
    if envelope:
        header += struct.pack("<4d", *geom.bounds)
    return header + wkb.dumps(geom)


class GpkgFeaturesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = _write_gpkg(
            self.root / "cells.gpkg",
            [
                _row(1, box(0, 0, 1, 1), 0.5, 10, "a"),
                _row(2, box(1, 0, 2, 1), None, 20, None),
                _row(3, None, 1.5, 30, "c"),
            ],
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_fields_values_and_numeric(self) -> None:
        features = read_gpkg_features(self.path)
        self.assertIsInstance(features, VectorFeatures)
        self.assertEqual(features.fields, ["cover", "point_count", "label"])
        self.assertEqual(features.numeric, {"cover", "point_count"})
        self.assertEqual(len(features), 3)
        self.assertTrue(np.isnan(features.columns["cover"][1]))
        self.assertEqual(features.columns["point_count"].tolist(), [10, 20, 30])
        self.assertEqual(features.columns["label"].tolist(), ["a", "", "c"])

    def test_crs_label_from_srs_table(self) -> None:
        features = read_gpkg_features(self.path)
        self.assertEqual(features.crs_label, "EPSG:5346")

    def test_geometries_and_bounds(self) -> None:
        features = read_gpkg_features(self.path)
        self.assertIsNotNone(features.geometries)
        self.assertTrue(features.geometries[0].equals(box(0, 0, 1, 1)))
        self.assertIsNone(features.geometries[2])
        self.assertEqual(features.bounds, (0.0, 0.0, 2.0, 1.0))

    def test_field_filter_keeps_table_order(self) -> None:
        features = read_gpkg_features(
            self.path, fields=["label", "missing", "point_count"]
        )
        self.assertEqual(features.fields, ["point_count", "label"])
        self.assertEqual(features.numeric, {"point_count"})

    def test_without_geometry_skips_wkb(self) -> None:
        features = read_gpkg_features(self.path, with_geometry=False)
        self.assertIsNone(features.geometries)
        self.assertIsNone(features.bounds)
        self.assertEqual(len(features), 3)

    def test_corrupt_wkb_degrades_to_none(self) -> None:
        path = _write_gpkg(
            self.root / "bad.gpkg",
            [
                _row(1, box(0, 0, 1, 1), 1.0, 1, "a"),
                (2, b"\x00\x01broken", 2.0, 2, "b"),
            ],
        )
        features = read_gpkg_features(path)
        self.assertIsNotNone(features.geometries[0])
        self.assertIsNone(features.geometries[1])

    def test_gpkg_binary_header_is_stripped(self) -> None:
        # Real GeoPackages store GP header + optional envelope + WKB;
        # a raw-WKB parser would return None for every feature.
        path = _write_gpkg(
            self.root / "gpb.gpkg",
            [
                (1, _gpb(box(0, 0, 1, 1)), 1.0, 1, "a"),
                (2, _gpb(box(1, 0, 2, 1), envelope=True), 2.0, 2, "b"),
            ],
        )
        features = read_gpkg_features(path)
        self.assertTrue(features.geometries[0].equals(box(0, 0, 1, 1)))
        self.assertTrue(features.geometries[1].equals(box(1, 0, 2, 1)))
        self.assertEqual(features.bounds, (0.0, 0.0, 2.0, 1.0))


class VectorRowsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = _write_gpkg(
            self.root / "cells.gpkg",
            [
                _row(1, box(0, 0, 1, 1), 0.5, 10, "a"),
                _row(2, box(1, 0, 2, 1), 1.5, 20, None),
                _row(3, box(2, 0, 3, 1), 2.5, 30, "c"),
            ],
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_limit_offset_and_null_rendering(self) -> None:
        headers, rows = read_vector_rows(self.path, limit=2)
        self.assertEqual(headers, ["cover", "point_count", "label"])
        self.assertEqual(
            rows,
            [["0.5", "10", "a"], ["1.5", "20", ""]],
        )
        headers, rows = read_vector_rows(self.path, limit=1, offset=1)
        self.assertEqual(rows, [["1.5", "20", ""]])


class GeojsonFeaturesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = self.root / "areas.geojson"
        self.path.write_text(
            json.dumps(
                {
                    "type": "FeatureCollection",
                    "crs": {
                        "type": "name",
                        "properties": {"name": "EPSG:4326"},
                    },
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": box(0, 0, 1, 1).__geo_interface__,
                            "properties": {"cover": 0.5, "label": "a"},
                        },
                        {
                            "type": "Feature",
                            "geometry": box(1, 0, 2, 1).__geo_interface__,
                            "properties": {"cover": 1.5, "label": "b"},
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_features_and_label(self) -> None:
        features = read_geojson_features(self.path)
        self.assertEqual(features.fields, ["cover", "label"])
        self.assertEqual(features.numeric, {"cover"})
        self.assertEqual(len(features), 2)
        self.assertEqual(features.crs_label, "EPSG:4326")
        self.assertAlmostEqual(features.bounds[2], 2.0)

    def test_rows(self) -> None:
        headers, rows = read_vector_rows(self.path, limit=1)
        self.assertEqual(headers, ["cover", "label"])
        self.assertEqual(rows, [["0.5", "a"]])


class DispatchTests(unittest.TestCase):
    def test_unsupported_extension_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.shp"
            path.write_bytes(b"x")
            with self.assertRaises(ValueError):
                read_vector_features(path)


if __name__ == "__main__":
    unittest.main()
