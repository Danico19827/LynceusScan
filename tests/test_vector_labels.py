# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Tests for compact CRS labels in the vector viewer info bar."""

from __future__ import annotations

import unittest

from lynceus.ui.viewers.vector_2d import CRS_LABEL_LEN, short_crs_label


class ShortCrsLabelTests(unittest.TestCase):
    def test_none_is_dash(self) -> None:
        self.assertEqual(short_crs_label(None), "—")

    def test_epsg_resolves(self) -> None:
        from pyproj import CRS

        self.assertEqual(
            short_crs_label(CRS.from_epsg(25831)), "EPSG:25831"
        )

    def test_long_name_truncated(self) -> None:
        class _Custom:
            def to_epsg(self):
                return None

            @property
            def name(self):
                return "X" * (CRS_LABEL_LEN + 50)

        label = short_crs_label(_Custom())
        self.assertTrue(len(label) <= CRS_LABEL_LEN + 1)
        self.assertTrue(label.endswith("…"))

    def test_plain_string_passthrough(self) -> None:
        self.assertEqual(short_crs_label("EPSG:25831"), "EPSG:25831")


class VectorRenderCancellationTests(unittest.TestCase):
    def test_grid_fast_path_avoids_geopandas_crs_area_properties(self) -> None:
        from unittest.mock import PropertyMock, patch

        import geopandas as gpd
        from shapely.geometry import box

        from lynceus.ui.viewers.vector_2d import Vector2DViewer

        frame = gpd.GeoDataFrame(
            {"metric": [1.0, 2.0]},
            geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
            crs="EPSG:25831",
        )
        with (
            patch.object(
                gpd.GeoSeries, "area", new_callable=PropertyMock
            ) as area,
            patch.object(
                gpd.GeoSeries, "length", new_callable=PropertyMock
            ) as length,
        ):
            result = Vector2DViewer._render_field(frame, "metric")

        self.assertIsNotNone(result)
        self.assertEqual(result[0].size().width(), 2000)
        area.assert_not_called()
        length.assert_not_called()

    def test_cached_metric_selection_cancels_inflight_render(self) -> None:
        import os
        from threading import Event

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtCore import QEventLoop, QTimer
        from PySide6.QtGui import QImage, QColor
        from PySide6.QtWidgets import QApplication
        import geopandas as gpd
        from shapely.geometry import box

        from lynceus.ui.viewers.vector_2d import Vector2DViewer

        QApplication.instance() or QApplication([])
        viewer = Vector2DViewer("vector")
        viewer._gdf = gpd.GeoDataFrame(
            {"metric": [1.0]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:25831"
        )
        viewer._fields = ["metric"]
        viewer._metric_combo.addItem("metric", "metric")
        image = QImage(2, 2, QImage.Format.Format_RGB32)
        image.fill(QColor("red"))
        viewer._img_cache[("metric", False)] = (image, (1.0, 1.0, 1))

        started = Event()
        release = Event()

        def slow_render(cancelled):
            started.set()
            release.wait(5)
            return None

        viewer.start_async_load(
            slow_render, lambda _result: None, lambda _error: None,
            show_loading=False,
        )
        cancelled = next(iter(viewer._async_jobs.values()))[0]

        loop = QEventLoop()
        poll = QTimer()
        poll.timeout.connect(lambda: loop.quit() if started.is_set() else None)
        deadline = QTimer()
        deadline.setSingleShot(True)
        deadline.timeout.connect(loop.quit)
        poll.start(5)
        deadline.start(2000)
        loop.exec()
        poll.stop()
        self.assertTrue(started.is_set())

        viewer._refresh()
        self.assertTrue(cancelled.is_set())
        self.assertFalse(viewer._async_jobs)
        self.assertFalse(viewer._view.isHidden())

        release.set()
        viewer.close()


if __name__ == "__main__":
    unittest.main()
