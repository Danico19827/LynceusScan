# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Vector QML: symbol types follow the layer geometry (treetops fix).

Point layers styled with fill symbols render nothing in QGIS (the user
had to restyle treetops by hand). The writer now reads the declared
GPKG geometry type (metadata only, no feature scan) and emits marker /
line / fill symbols; unknown kinds keep the legacy fills.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

CLASSES = [
    (0.0, 5.0, "#ffffb2", "0 - 5 m"),
    (5.0, 10.0, "#fecc5c", "5 - 10 m"),
    (10.0, 20.0, "#fd8d3c", "10 - 20 m"),
]


def _write_gpkg(path: Path, geoms: list, field: str = "height_m") -> Path:
    import geopandas as gpd

    records = [{"gid": i + 1, field: float(2 + i * 4)}
               for i in range(len(geoms))]
    gdf = gpd.GeoDataFrame(records, geometry=list(geoms), crs="EPSG:32721")
    gdf.to_file(path, driver="GPKG")
    return path


def _style(path: Path) -> str:
    from lynceus.processing.qml_style import style_vector_file

    assert style_vector_file(str(path), field="height_m",
                             classes=CLASSES, name="T")
    return path.with_suffix(".qml").read_text(encoding="utf-8")


class VectorQmlGeometryTests(unittest.TestCase):
    def test_points_get_marker_symbols(self) -> None:
        from shapely.geometry import Point

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_gpkg(Path(tmp) / "tops.gpkg",
                               [Point(0, 0), Point(10, 10), Point(20, 5)])
            xml = _style(path)
            self.assertIn('type="marker"', xml)
            self.assertIn('class="SimpleMarker"', xml)
            self.assertNotIn('type="fill"', xml)

    def test_lines_get_line_symbols(self) -> None:
        from shapely.geometry import LineString

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_gpkg(
                Path(tmp) / "lines.gpkg",
                [LineString([(0, 0), (10, 10)]),
                 LineString([(20, 20), (30, 5)])])
            xml = _style(path)
            self.assertIn('type="line"', xml)
            self.assertIn('class="SimpleLine"', xml)
            self.assertNotIn('type="fill"', xml)

    def test_polygons_keep_fill_symbols(self) -> None:
        from shapely.geometry import Polygon

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_gpkg(
                Path(tmp) / "crowns.gpkg",
                [Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
                 Polygon([(20, 20), (30, 20), (30, 30), (20, 30)])])
            xml = _style(path)
            self.assertIn('type="fill"', xml)
            self.assertIn('class="SimpleFill"', xml)

    def test_treetops_spec_renders_markers(self) -> None:
        import importlib.util

        from shapely.geometry import Point

        root = Path(__file__).resolve().parent.parent
        node = root / "extensions" / "nodes" / "forestry" / "treetops.py"
        if not node.is_file():
            raise unittest.SkipTest("treetops node not present")
        from lynceus.processing.qml_style import style_vector_file

        spec = importlib.util.spec_from_file_location("treetops_qml", node)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        qml = module.PROCESSING_SPECS["qml"]["treetops.gpkg"]
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_gpkg(Path(tmp) / "treetops.gpkg",
                               [Point(0, 0), Point(5, 5)])
            self.assertTrue(style_vector_file(
                str(path), field=qml["field"], classes=qml["classes"],
                name=qml["name"]))
            xml = path.with_suffix(".qml").read_text(encoding="utf-8")
            self.assertIn('type="marker"', xml)

    def test_missing_file_returns_false(self) -> None:
        from lynceus.processing.qml_style import style_vector_file

        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(style_vector_file(
                str(Path(tmp) / "nope.gpkg"), field="height_m",
                classes=CLASSES, name="T"))


if __name__ == "__main__":
    unittest.main()
