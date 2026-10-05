# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Tests for compact CRS labels and vector rendering (PROJ-free).

The viewer reads vector products through ``lynceus.processing.vector_table``
(sqlite3 + shapely): these tests use the same container so the GUI path
never touches geopandas/pyproj/pyogrio.
"""

from __future__ import annotations

import unittest

import numpy as np
from shapely.geometry import box, Polygon

from lynceus.processing.vector_table import VectorFeatures
from lynceus.ui.viewers.vector_2d import CRS_LABEL_LEN, short_crs_label


def _features(geometries, metric) -> VectorFeatures:
    from shapely import bounds as sh_bounds

    array = np.array(geometries, dtype=object)
    b = sh_bounds(array)
    return VectorFeatures(
        fields=["metric"],
        columns={"metric": np.array(metric, dtype=np.float64)},
        numeric={"metric"},
        geometries=array,
        bounds=(
            float(b[:, 0].min()),
            float(b[:, 1].min()),
            float(b[:, 2].max()),
            float(b[:, 3].max()),
        ),
        crs_label="EPSG:25831",
        feature_count=len(geometries),
    )


class ShortCrsLabelTests(unittest.TestCase):
    def test_none_or_empty_is_dash(self) -> None:
        self.assertEqual(short_crs_label(None), "—")
        self.assertEqual(short_crs_label(""), "—")

    def test_crs_object_epsg_resolves(self) -> None:
        class _Crs:
            def to_epsg(self):
                return 25831

            @property
            def name(self):
                return "ETRS89 / UTM zone 31N"

        self.assertEqual(short_crs_label(_Crs()), "EPSG:25831")

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
        long = "Y" * (CRS_LABEL_LEN + 10)
        self.assertEqual(len(short_crs_label(long)), CRS_LABEL_LEN + 1)
        self.assertTrue(short_crs_label(long).endswith("…"))


class VectorRenderTests(unittest.TestCase):
    def test_fast_path_renders_axis_aligned_boxes(self) -> None:
        from lynceus.ui.viewers.vector_2d import Vector2DViewer

        features = _features(
            [box(0, 0, 1, 1), box(1, 0, 2, 1)], [1.0, 2.0]
        )
        result = Vector2DViewer._render_field(features, "metric")
        self.assertIsNotNone(result)
        self.assertEqual(result[0].size().width(), 2000)
        self.assertEqual(result[1][2], 2)  # 2 valid cells

    def test_non_box_geometry_uses_painter_fallback(self) -> None:
        from lynceus.ui.viewers.vector_2d import Vector2DViewer

        triangle = Polygon([(0, 0), (1, 0), (0, 1)])
        features = _features([triangle], [5.0])
        result = Vector2DViewer._render_field(features, "metric")
        self.assertIsNotNone(result)

    def test_all_nodata_keeps_mosaic(self) -> None:
        from lynceus.ui.viewers.vector_2d import VECTOR_NODATA, Vector2DViewer

        features = _features([box(0, 0, 1, 1), box(1, 0, 2, 1)],
                             [VECTOR_NODATA, VECTOR_NODATA])
        result = Vector2DViewer._render_field(features, "metric")
        self.assertIsNotNone(result)
        self.assertEqual(result[1][2], 0)  # no valid cells


class VectorRenderCancellationTests(unittest.TestCase):
    def test_cached_metric_selection_cancels_inflight_render(self) -> None:
        import os
        from threading import Event

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtCore import QEventLoop, QTimer
        from PySide6.QtGui import QColor, QImage
        from PySide6.QtWidgets import QApplication

        from lynceus.ui.viewers.vector_2d import Vector2DViewer

        QApplication.instance() or QApplication([])
        viewer = Vector2DViewer("vector")
        viewer._features = _features([box(0, 0, 1, 1)], [1.0])
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
