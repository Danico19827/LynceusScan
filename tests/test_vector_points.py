# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Vector 2D viewer: point geometries render (treetops regression).

The viewer was polygon-only: POINT layers (treetops) fell through the
paint loop with zero drawn features and showed the "no polygonal
features" placeholder. Points must render as choropleth discs with the
same per-feature colors; polygons keep working through both paths.
"""

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication


def _wait_for(predicate, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.05)
    QApplication.processEvents()
    return predicate()


def _write_points_gpkg(path: Path) -> Path:
    import geopandas as gpd
    from shapely.geometry import Point

    records = [
        {"tree_id": 1, "height_m": 15.0},
        {"tree_id": 2, "height_m": 10.0},
        {"tree_id": 3, "height_m": 5.0},
    ]
    points = [Point(10.0, 40.0), Point(30.0, 30.0), Point(50.0, 50.0)]
    gdf = gpd.GeoDataFrame(records, geometry=points, crs="EPSG:32721")
    gdf.to_file(path, driver="GPKG")
    meta = {
        "metrics_computed": ["height_m"],
        "default_metric": "height_m",
        "bookkeeping_cols": ["tree_id"],
    }
    Path(str(path).replace(".gpkg", ".meta.json")).write_text(
        json.dumps(meta), encoding="utf-8")
    return path


def _write_polygons_gpkg(path: Path) -> Path:
    import geopandas as gpd
    from shapely.geometry import Polygon

    records = [
        {"gid": 1, "h_mean": 12.0},
        {"gid": 2, "h_mean": 6.0},
    ]
    polys = [Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
             Polygon([(20, 20), (30, 20), (30, 30), (20, 30)])]
    gdf = gpd.GeoDataFrame(records, geometry=polys, crs="EPSG:32721")
    gdf.to_file(path, driver="GPKG")
    return path


def _drawn_pixels(viewer) -> int:
    from lynceus.ui.branding import THEME_COLORS
    from PySide6.QtGui import QColor

    item = viewer._view._item
    if item is None:
        return 0
    img = item.pixmap().toImage()
    bg = QColor(THEME_COLORS["canvas_bg"])
    count = 0
    for y in range(img.height()):
        for x in range(img.width()):
            if QColor(img.pixel(x, y)) != bg:
                count += 1
    return count


class VectorPointsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._app = QApplication.instance() or QApplication([])

    def _viewer_for(self, path: Path):
        from lynceus.ui.viewers.vector_2d import Vector2DViewer

        viewer = Vector2DViewer("vector")
        viewer.set_payload({"file": str(path)})
        return viewer

    def test_points_render_discs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_points_gpkg(Path(tmp) / "tops.gpkg")
            viewer = self._viewer_for(path)
            try:
                shown = _wait_for(lambda: viewer._view._item is not None)
                self.assertTrue(shown)
                self.assertGreater(_drawn_pixels(viewer), 0)
            finally:
                viewer.deleteLater()

    def test_polygons_still_render(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_polygons_gpkg(Path(tmp) / "crowns.gpkg")
            viewer = self._viewer_for(path)
            try:
                shown = _wait_for(lambda: viewer._view._item is not None)
                self.assertTrue(shown)
                self.assertGreater(_drawn_pixels(viewer), 0)
            finally:
                viewer.deleteLater()


if __name__ == "__main__":
    unittest.main()
