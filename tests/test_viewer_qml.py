# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Viewer/QGIS parity: the 2D raster viewer shares the .qml ramp base.

Continuous products must render the same mapping QGIS does (ramp stops +
min/max from the product sidecar): same colors, same stretch, gradient
legend. Foreign files without a sidecar keep the legacy elevation ramp.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _write_tif(path: Path, array: np.ndarray, nodata: float) -> Path:
    import rasterio
    from affine import Affine

    with rasterio.open(
        str(path), "w", driver="GTiff", height=array.shape[0],
        width=array.shape[1], count=1, dtype="float32",
        crs="EPSG:32612", transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
        nodata=nodata,
    ) as ds:
        ds.write(array.astype(np.float32), 1)
    return path


class ReadRampStopsTests(unittest.TestCase):
    def test_roundtrip_continuous_qml(self) -> None:
        from lynceus.processing.qml_style import (
            RAMPS,
            generate_qml,
            read_ramp_stops,
        )

        with tempfile.TemporaryDirectory() as tmp:
            arr = np.arange(64, dtype=np.float32).reshape(8, 8) + 10.0
            path = _write_tif(Path(tmp) / "slope.tif", arr, -9999.0)
            self.assertTrue(generate_qml(str(path), ramp_key="slope"))
            parsed = read_ramp_stops(str(Path(tmp) / "slope.qml"))
            self.assertIsNotNone(parsed)
            minimum, maximum, stops = parsed
            self.assertEqual((minimum, maximum), (10.0, 73.0))
            self.assertEqual(len(stops), len(RAMPS["slope"]["stops"]))
            for (value, color, _label), (pos, want, _wlabel) in zip(
                stops, RAMPS["slope"]["stops"]
            ):
                self.assertAlmostEqual(
                    value, 10.0 + pos * 63.0, places=4)
                self.assertEqual(color.lower(), want.lower())

    def test_paletted_qml_yields_no_ramp(self) -> None:
        from lynceus.processing.qml_style import read_ramp_stops

        with tempfile.TemporaryDirectory() as tmp:
            qml = Path(tmp) / "x.qml"
            qml.write_text(
                "<?xml version=\"1.0\"?><qgis>"
                "<pipe><rasterrenderer type=\"paletted\">"
                "</rasterrenderer></pipe></qgis>",
                encoding="utf-8",
            )
            self.assertIsNone(read_ramp_stops(str(qml)))

    def test_missing_qml_yields_no_ramp(self) -> None:
        from lynceus.processing.qml_style import read_ramp_stops

        self.assertIsNone(read_ramp_stops("/nonexistent/x.qml"))


class RampColorsTests(unittest.TestCase):
    def test_exact_qgis_mapping(self) -> None:
        from lynceus.ui.viewers.raster_2d import _hex_rgb, _ramp_colors

        stops = [
            (10.0, _hex_rgb("#ffffcc")),
            (41.5, _hex_rgb("#fd8d3c")),
            (73.0, _hex_rgb("#800026")),
        ]
        # Stops hit exactly; outside clips to edge colors (QGIS clip="0").
        probe = np.array([[10.0, 41.5, 73.0, -5.0, 200.0]])
        rgb = _ramp_colors(probe, 10.0, 73.0, stops)
        self.assertEqual(tuple(rgb[0, 0]), _hex_rgb("#ffffcc"))
        self.assertEqual(tuple(rgb[0, 1]), _hex_rgb("#fd8d3c"))
        self.assertEqual(tuple(rgb[0, 2]), _hex_rgb("#800026"))
        self.assertEqual(tuple(rgb[0, 3]), _hex_rgb("#ffffcc"))
        self.assertEqual(tuple(rgb[0, 4]), _hex_rgb("#800026"))
        # Midpoint interpolates linearly between stops.
        mid = _ramp_colors(np.array([[25.75]]), 10.0, 73.0, stops)[0, 0]
        lo = np.array(_hex_rgb("#ffffcc"), dtype=float)
        hi = np.array(_hex_rgb("#fd8d3c"), dtype=float)
        self.assertTrue(np.allclose(np.array(mid, dtype=float),
                                    (lo + hi) / 2, atol=1))

    def test_image_uses_qml_range(self) -> None:
        from PySide6.QtWidgets import QApplication

        from lynceus.ui.viewers.raster_2d import (
            _hex_rgb,
            _raster_to_image,
        )

        QApplication.instance() or QApplication([])
        arr = np.arange(16, dtype=np.float32).reshape(4, 4)
        ramp = (0.0, 100.0, [(0.0, _hex_rgb("#000000")),
                             (100.0, _hex_rgb("#ffffff"))])
        image, vrange = _raster_to_image(arr, -9999.0, ramp=ramp)
        self.assertIsNotNone(image)
        self.assertEqual(vrange, (0.0, 100.0))

    def test_foreign_file_keeps_legacy_stretch(self) -> None:
        from PySide6.QtWidgets import QApplication

        from lynceus.ui.viewers.raster_2d import _raster_to_image

        QApplication.instance() or QApplication([])
        arr = np.arange(100, dtype=np.float32).reshape(10, 10)
        image, vrange = _raster_to_image(arr, -9999.0)
        self.assertIsNotNone(image)
        # Legacy 2-98 percentile stretch, not full range.
        self.assertGreater(vrange[0], 0.0)
        self.assertLess(vrange[1], 99.0)


class DiscreteCastGuardTests(unittest.TestCase):
    def test_hostile_sample_returns_none_without_warning(self) -> None:
        import warnings

        from lynceus.ui.viewers.raster_2d import _discrete_style

        # Untagged foreign tag + NaN + inf soup: nothing castable, no noise.
        arr = np.array([[np.nan, np.inf],
                        [-3.4028235e38, -9999.0]], dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertIsNone(_discrete_style(arr, -9999.0))

    def test_genuine_classes_still_detected(self) -> None:
        from lynceus.ui.viewers.raster_2d import _discrete_style

        arr = np.array([[0.0, 1.0], [2.0, -9999.0]], dtype=np.float32)
        lut, labels = _discrete_style(arr, -9999.0)
        # Generic palette covers the detected classes (full 0-8 set).
        self.assertTrue({0, 1, 2} <= set(lut))
        self.assertTrue(len(labels) >= 3)


class RefineWindowTests(unittest.TestCase):
    def test_window_covers_rect_and_clamps(self) -> None:
        from lynceus.ui.viewers.raster_2d import Raster2DViewer

        # Pixel rect partly outside a 50x40 raster: clamps, native readout.
        windowed = Raster2DViewer._window_for_pixels(
            -5.0, -3.0, 60.0, 38.0, 50, 40, 10_000_000)
        self.assertIsNotNone(windowed)
        window, out_shape = windowed
        self.assertEqual((window.col_off, window.row_off), (0, 0))
        self.assertEqual((window.width, window.height), (50, 38))
        self.assertIsNone(out_shape)

    def test_empty_returns_none_and_oversize_decimates(self) -> None:
        from lynceus.ui.viewers.raster_2d import Raster2DViewer

        window_for = Raster2DViewer._window_for_pixels
        # Empty after clamping (fully outside) refuses.
        self.assertIsNone(window_for(60.0, 60.0, 50.0, 40.0, 50, 40, 10**9))
        # Reversed but overlapping normalizes instead of refusing.
        windowed = window_for(20.0, 20.0, 10.0, 10.0, 50, 40, 10**9)
        self.assertIsNotNone(windowed)
        self.assertIsNone(
            window_for(200.0, 200.0, 300.0, 300.0, 50, 40, 10**9))
        # Over budget: same window, decimated straight to it (never refused).
        windowed = window_for(0.0, 0.0, 50.0, 40.0, 50, 40, 100)
        self.assertIsNotNone(windowed)
        window, out_shape = windowed
        self.assertEqual((window.width, window.height), (50, 40))
        self.assertIsNotNone(out_shape)
        self.assertLessEqual(out_shape[0] * out_shape[1], 100)
        # Small window: native.
        windowed = window_for(10.0, 10.0, 20.0, 20.0, 50, 40, 10**9)
        self.assertIsNotNone(windowed)
        window, out_shape = windowed
        self.assertEqual((window.width, window.height), (10, 10))
        self.assertIsNone(out_shape)

    def test_render_products_shared_path(self) -> None:
        from PySide6.QtWidgets import QApplication

        from lynceus.ui.viewers.raster_2d import _render_products

        QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as tmp:
            arr = np.arange(64, dtype=np.float32).reshape(8, 8) + 10.0
            path = _write_tif(Path(tmp) / "slope.tif", arr, -9999.0)
            from lynceus.processing.qml_style import generate_qml

            self.assertTrue(generate_qml(str(path), ramp_key="slope"))
            from affine import Affine

            (image, vrange, discrete, ramp, origin_full, out_step,
             _array, _nodata) = _render_products(
                arr, Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0), -9999.0,
                str(path))
            self.assertIsNotNone(image)
            self.assertEqual(vrange, (10.0, 73.0))
            self.assertIsNone(discrete)
            self.assertIsNotNone(ramp)
            self.assertEqual(origin_full, (0, 0))
            self.assertEqual(out_step, (1.0, 1.0))


class ZoomFreedomTests(unittest.TestCase):
    """Zooming out never snaps back: no auto-fit, no reload, user is free."""

    def test_zoom_out_stays_put(self) -> None:
        import tempfile
        import time

        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtGui import QWheelEvent
        from PySide6.QtWidgets import QApplication

        from lynceus.ui.viewers.raster_2d import Raster2DViewer

        QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as tmp:
            arr = np.arange(2500, dtype=np.float32).reshape(50, 50)
            path = _write_tif(Path(tmp) / "small.tif", arr, -9999.0)
            viewer = Raster2DViewer("raster")
            viewer.resize(800, 600)
            viewer.show()
            try:
                overs = []
                orig_over = viewer._start_overview
                viewer._start_overview = lambda: (
                    overs.append(1), orig_over())[1]
                viewer.set_payload({"file": str(path)})

                def pump_until(condition, timeout_s=15.0):
                    deadline = time.time() + timeout_s
                    while time.time() < deadline:
                        QApplication.instance().processEvents()
                        time.sleep(0.02)
                        if condition():
                            return True
                    return False

                view = viewer._image_view
                shown = pump_until(lambda: view.isVisible()
                                   and view._item is not None)
                self.assertTrue(shown)
                fit_scale = view.transform().m11()
                for _ in range(8):
                    event = QWheelEvent(
                        view.viewport().rect().center(),
                        view.mapToGlobal(view.viewport().rect().center()),
                        QPoint(0, -120), QPoint(0, -120),
                        Qt.MouseButton.NoButton,
                        Qt.KeyboardModifier.NoModifier,
                        Qt.ScrollPhase.NoScrollPhase, False,
                        Qt.MouseEventSource.MouseEventNotSynthesized,
                    )
                    view.wheelEvent(event)
                settled = pump_until(
                    lambda: not view._zoom_timer.isActive()
                    and not view._refine_timer.isActive(),
                    timeout_s=15.0,
                )
                self.assertTrue(settled)
                # Zoomed out well below fit, never yanked back, no reload.
                self.assertLess(view.transform().m11(), fit_scale * 0.5)
                self.assertEqual(len(overs), 1)
            finally:
                viewer.close()


class LegendStabilityTests(unittest.TestCase):
    def test_identical_legend_rebuild_is_skipped(self) -> None:
        from PySide6.QtWidgets import QApplication

        from lynceus.ui.viewers.raster_2d import (
            Raster2DViewer,
            _hex_rgb,
        )

        QApplication.instance() or QApplication([])
        viewer = Raster2DViewer("raster")
        try:
            ramp = (10.0, 73.0, [(10.0, _hex_rgb("#ffffcc")),
                                 (73.0, _hex_rgb("#800026"))])
            viewer._set_gradient_legend(ramp)
            before = [
                viewer._legend_layout.itemAt(i).widget()
                for i in range(viewer._legend_layout.count())
            ]
            viewer._set_gradient_legend(ramp)
            after = [
                viewer._legend_layout.itemAt(i).widget()
                for i in range(viewer._legend_layout.count())
            ]
            self.assertEqual(before, after)
        finally:
            viewer.close()


class OverviewStabilityTests(unittest.TestCase):
    """Opening a raster must settle: one overview, never a reload loop."""

    def _show_viewer(self, path):
        from PySide6.QtWidgets import QApplication

        from lynceus.ui.viewers.raster_2d import Raster2DViewer

        QApplication.instance() or QApplication([])
        viewer = Raster2DViewer("raster")
        viewer.resize(800, 600)
        viewer.show()
        starts = []
        orig = viewer._start_overview
        viewer._start_overview = lambda: (
            starts.append(1), orig())[1]
        viewer.set_payload({"file": str(path)})
        return viewer, starts

    @staticmethod
    def _pump_while(condition, timeout_s=15.0):
        import time

        from PySide6.QtWidgets import QApplication

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            QApplication.instance().processEvents()
            time.sleep(0.02)
            if condition():
                return True
        return False

    def test_open_settles_with_single_overview(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            arr = np.arange(64, dtype=np.float32).reshape(8, 8) + 10.0
            path = _write_tif(Path(tmp) / "slope.tif", arr, -9999.0)
            viewer, starts = self._show_viewer(path)
            try:
                shown = self._pump_while(
                    lambda: viewer._image_view.isVisible())
                self.assertTrue(shown, "overview never displayed")
                self.assertEqual(len(starts), 1)
                # Settle: no further loads without input.
                import time

                time.sleep(2.0)
                self._pump_while(lambda: False, timeout_s=1.0)
                self.assertEqual(len(starts), 1)
                self.assertTrue(viewer._placeholder.isHidden())
            finally:
                viewer.close()

    def test_reload_over_image_stays_silent(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            arr = np.arange(64, dtype=np.float32).reshape(8, 8) + 10.0
            path = _write_tif(Path(tmp) / "slope.tif", arr, -9999.0)
            viewer, starts = self._show_viewer(path)
            try:
                shown = self._pump_while(
                    lambda: viewer._image_view.isVisible())
                self.assertTrue(shown, "overview never displayed")
                viewer._start_overview()
                done = self._pump_while(
                    lambda: len(starts) >= 2 and viewer._image is not None)
                self.assertTrue(done, "reload never finished")
                import time

                time.sleep(1.0)
                self._pump_while(lambda: False, timeout_s=0.5)
                # Silent reload: no placeholder split over the image.
                self.assertTrue(viewer._placeholder.isHidden())
            finally:
                viewer.close()


class SmoothZoomTests(unittest.TestCase):
    def test_factor_is_proportional(self) -> None:
        from lynceus.ui.viewers.raster_2d import _zoom_factor

        self.assertEqual(_zoom_factor(0), 1.0)
        self.assertAlmostEqual(_zoom_factor(120), 1.197, places=3)
        self.assertAlmostEqual(_zoom_factor(-120), 1 / 1.197, places=3)
        # Smooth trackpad deltas stay gradual (no 1.25 jumps).
        self.assertLess(_zoom_factor(15), 1.03)
        self.assertGreater(_zoom_factor(-15), 1 / 1.03)

    def test_grid_placement_exact_on_rotated_read(self) -> None:
        """Scene placement needs no world math: origin/scale are grid-exact.

        A rotated raster still crops axis-aligned in grid space (rotation
        lives in the world transform), so (origin, scale) lands pixel-true
        where the old overlay needed a full inverse composition.
        Regression cover for the zoom-misalignment class.
        """
        import numpy as np
        from affine import Affine

        from lynceus.ui.viewers.raster_2d import _render_products

        nodata = -9999.0
        arr = np.full((20, 20), nodata, dtype=np.float32)
        arr[5:15, 7:17] = 42.0
        transform = Affine(2.0, 0.5, 100.0, -0.25, -2.0, 200.0)
        (image, _vrange, _discrete, _ramp, origin_full, scale,
         _array, _nodata) = _render_products(
            arr, transform, nodata, "memory.tif",
            origin=(100, 200), step=(2.0, 2.0))
        self.assertIsNotNone(image)
        self.assertEqual(origin_full, (100 + 7 * 2.0, 200 + 5 * 2.0))
        self.assertEqual(scale, (2.0, 2.0))

    def test_crop_valid_keeps_rotation_terms(self) -> None:
        """Crop shifts the origin through the full affine (b/d survive)."""
        import numpy as np
        from affine import Affine

        from lynceus.ui.viewers.raster_2d import Raster2DViewer

        nodata = -9999.0
        array = np.full((10, 12), nodata, dtype=np.float32)
        array[2:6, 3:9] = 5.0
        transform = Affine(2.0, 0.5, 100.0, -0.25, -2.0, 200.0)
        cropped, out_t = Raster2DViewer._crop_valid(array, transform, nodata)
        self.assertEqual(cropped.shape, (4, 6))
        self.assertAlmostEqual(out_t.a, 2.0)
        self.assertAlmostEqual(out_t.b, 0.5)
        self.assertAlmostEqual(out_t.c, 100.0 + 2.0 * 3 + 0.5 * 2)
        self.assertAlmostEqual(out_t.f, 200.0 + -0.25 * 3 + -2.0 * 2)

    def test_decimate_average_never_dilutes_valid(self) -> None:
        """NODATA never waters down valid neighbours; voids stay voids."""
        import numpy as np
        from affine import Affine

        from lynceus.ui.viewers.raster_2d import Raster2DViewer

        nodata = -9999.0
        array = np.full((4, 4), nodata, dtype=np.float32)
        array[0, 0] = 8.0
        transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 4.0)
        out, out_t = Raster2DViewer._decimate_average(
            array, transform, 2, nodata)
        self.assertEqual(out.shape, (2, 2))
        self.assertAlmostEqual(float(out[0, 0]), 8.0)
        self.assertEqual(float(out[1, 1]), nodata)
        self.assertAlmostEqual(out_t.a, 2.0)
        self.assertAlmostEqual(out_t.e, -2.0)
        self.assertAlmostEqual(out_t.c, 0.0)
        self.assertAlmostEqual(out_t.f, 4.0)


class OverlayStabilityTests(unittest.TestCase):
    """Refine overlays detail: the view transform never moves for a load."""

    def setUp(self) -> None:
        # The display cell budget is a user preference (QSettings): pin it
        # for these tests so the fixtures decimate at load and the refine
        # has something sharper to swap in. On a large preset the overview
        # is already native and the (correct) viewer skip makes the swap
        # assertions unreachable — an environment-dependent false red.
        from PySide6.QtCore import QSettings

        from lynceus.ui.settings_keys import (
            DISPLAY_RASTER_CELLS_DEFAULT,
            DISPLAY_RASTER_CELLS_KEY,
            SETTINGS_APP,
            SETTINGS_ORG,
        )

        self._settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
        self._had_cells = self._settings.contains(DISPLAY_RASTER_CELLS_KEY)
        self._saved_cells = self._settings.value(DISPLAY_RASTER_CELLS_KEY)
        self._settings.setValue(
            DISPLAY_RASTER_CELLS_KEY, DISPLAY_RASTER_CELLS_DEFAULT)
        self._settings.sync()

        def _restore():
            if self._had_cells:
                self._settings.setValue(
                    DISPLAY_RASTER_CELLS_KEY, self._saved_cells)
            else:
                self._settings.remove(DISPLAY_RASTER_CELLS_KEY)
            self._settings.sync()

        self.addCleanup(_restore)

    def test_zoom_keeps_view_still(self) -> None:
        import tempfile
        import time

        from PySide6.QtWidgets import QApplication

        from lynceus.ui.viewers.raster_2d import Raster2DViewer

        QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as tmp:
            arr = np.arange(2500, dtype=np.float32).reshape(50, 50)
            path = _write_tif(Path(tmp) / "big.tif", arr, -9999.0)
            viewer = Raster2DViewer("raster")
            viewer.resize(800, 600)
            viewer.show()
            try:
                viewer.set_payload({"file": str(path)})

                def pump_until(condition, timeout_s=15.0):
                    deadline = time.time() + timeout_s
                    while time.time() < deadline:
                        QApplication.instance().processEvents()
                        time.sleep(0.02)
                        if condition():
                            return True
                    return False

                shown = pump_until(
                    lambda: viewer._image_view.isVisible())
                self.assertTrue(shown)
                view = viewer._image_view
                # Deep zoom via the wheel path (proportional + eased).
                from PySide6.QtCore import QPoint, Qt
                from PySide6.QtGui import QWheelEvent

                for _ in range(4):
                    event = QWheelEvent(
                        view.viewport().rect().center(),
                        view.mapToGlobal(view.viewport().rect().center()),
                        QPoint(0, 120), QPoint(0, 120),
                        Qt.MouseButton.NoButton,
                        Qt.KeyboardModifier.NoModifier,
                        Qt.ScrollPhase.NoScrollPhase, False,
                        Qt.MouseEventSource.MouseEventNotSynthesized,
                    )
                    view.wheelEvent(event)
                settled = pump_until(
                    lambda: (not view._zoom_timer.isActive())
                    and (not view._refine_timer.isActive()),
                    timeout_s=15.0,
                )
                self.assertTrue(settled, "zoom never settled")
                # The view moved only by easing (monotonic, bounded steps).
                self.assertIsNone(view._zoom_target)
            finally:
                viewer.close()

    def test_sig_moved_tolerates_noise_and_catches_moves(self) -> None:
        from lynceus.ui.viewers.raster_2d import _sig_moved

        # Identical rects never count as moved (float noise must not drop
        # a fresh single image); any real pan/zoom does.
        self.assertFalse(_sig_moved((0.0, 0.0, 100.0, 50.0),
                                    (0.0, 0.0, 100.0, 50.0)))
        self.assertFalse(_sig_moved((500000.0, 6999900.0, 500100.0,
                                      7000000.0),
                                     (500000.0 + 1e-9, 6999900.0,
                                      500100.0, 7000000.0)))
        self.assertTrue(_sig_moved((0.0, 0.0, 100.0, 50.0),
                                   (5.0, 0.0, 105.0, 50.0)))

    def test_second_refine_stays_world_anchored(self) -> None:
        """Panning past the image must reload the new ground, not jump.

        Single-image invariant: exactly one pixmap in the scene, always
        covering the viewport center. No overlay exists to misalign.
        """
        import tempfile
        import time

        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QTransform
        from PySide6.QtWidgets import (
            QApplication,
            QGraphicsPixmapItem,
        )

        from lynceus.ui.viewers.raster_2d import Raster2DViewer

        QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as tmp:
            rows, cols = np.mgrid[0:2500, 0:2500]
            arr = (rows * 0.05 + cols * 0.03).astype(np.float32)
            path = _write_tif(Path(tmp) / "big.tif", arr, -9999.0)
            viewer = Raster2DViewer("raster")
            viewer.resize(800, 600)
            viewer.show()
            try:
                viewer.set_payload({"file": str(path)})

                def pump_until(condition, timeout_s=20.0):
                    deadline = time.time() + timeout_s
                    while time.time() < deadline:
                        QApplication.instance().processEvents()
                        time.sleep(0.02)
                        if condition():
                            return True
                    return False

                def assert_single_and_covering(step: str) -> None:
                    view = viewer._image_view
                    items = [it for it in view.scene().items()
                             if isinstance(it, QGraphicsPixmapItem)]
                    self.assertEqual(len(items), 1, step)
                    current = view.mapToScene(
                        view.viewport().rect().center())
                    x0, y0, x1, y1 = viewer._image["extent"]
                    self.assertGreaterEqual(current.x(), x0 - 1.0, step)
                    self.assertLessEqual(current.x(), x1 + 1.0, step)
                    self.assertGreaterEqual(current.y(), y0 - 1.0, step)
                    self.assertLessEqual(current.y(), y1 + 1.0, step)

                view = viewer._image_view
                shown = pump_until(lambda: view.isVisible()
                                    and view._item is not None)
                self.assertTrue(shown)
                # First deep zoom: native window swaps in.
                view.resetTransform()
                view.setTransform(QTransform().scale(8.0, 8.0))
                view.centerOn(view.sceneRect().center())
                first_item = view._item
                view._refine_timer.start()
                first = pump_until(
                    lambda: (view._item is not None
                             and view._item is not first_item),
                    timeout_s=25.0)
                self.assertTrue(first, "first window never swapped in")
                assert_single_and_covering("zoom8-center")
                # Pan well past the image coverage (same zoom, absolute
                # offset: the scene rect shrinks with every swap): the next
                # window must cover the new viewport center.
                current = view.mapToScene(view.viewport().rect().center())
                view.centerOn(QPointF(current.x() + 600.0,
                                      current.y() + 200.0))
                before = view._item
                view._refine_timer.start()
                second = pump_until(
                    lambda: (view._item is not None
                             and view._item is not before),
                    timeout_s=25.0,
                )
                self.assertTrue(second, "second window never swapped in")
                assert_single_and_covering("pan-far")
            finally:
                viewer.close()

    def test_sweep_stays_world_anchored_on_mosaic_scale(self) -> None:
        """Zoom/pan sweep over a mosaic-scale raster: every swap covers.

        Fairbanks-like fixture (6000x4000 with NODATA borders, decimated
        overview): zoom in steps with pans between, asserting single-image
        viewport coverage at every landing. Any misplacement class (crop
        offset, stale state) fails here.
        """
        import tempfile
        import time

        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QTransform
        from PySide6.QtWidgets import QApplication, QGraphicsPixmapItem

        from lynceus.ui.viewers.raster_2d import Raster2DViewer

        QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as tmp:
            w, h, border = 6000, 4000, 200
            rows, cols = np.mgrid[0:h, 0:w]
            arr = np.full((h, w), -9999.0, dtype=np.float32)
            valid = np.zeros((h, w), dtype=bool)
            valid[border:h - border, border:w - border] = True
            arr[valid] = (rows[valid] * 0.05 + cols[valid] * 0.03 + 10.0
                          ).astype(np.float32)
            path = _write_tif(Path(tmp) / "mosaic.tif", arr, -9999.0)
            viewer = Raster2DViewer("raster")
            viewer.resize(800, 600)
            viewer.show()
            try:
                viewer.set_payload({"file": str(path)})

                def pump_until(condition, timeout_s=25.0):
                    deadline = time.time() + timeout_s
                    while time.time() < deadline:
                        QApplication.instance().processEvents()
                        time.sleep(0.02)
                        if condition():
                            return True
                    return False

                def assert_covered(step: str) -> None:
                    view = viewer._image_view
                    items = [it for it in view.scene().items()
                             if isinstance(it, QGraphicsPixmapItem)]
                    self.assertEqual(len(items), 1, step)
                    current = view.mapToScene(
                        view.viewport().rect().center())
                    x0, y0, x1, y1 = viewer._image["extent"]
                    self.assertGreaterEqual(current.x(), x0 - 1.0, step)
                    self.assertLessEqual(current.x(), x1 + 1.0, step)
                    self.assertGreaterEqual(current.y(), y0 - 1.0, step)
                    self.assertLessEqual(current.y(), y1 + 1.0, step)

                def goto(scale: float, dx: float, dy: float,
                         step: str) -> None:
                    view = viewer._image_view
                    before = view._item
                    view.resetTransform()
                    view.setTransform(QTransform().scale(scale, scale))
                    view.centerOn(QPointF(3000.0 + dx, 2000.0 + dy))
                    view._refine_timer.start()
                    landed = pump_until(
                        lambda: (view._item is not None
                                 and view._item is not before))
                    self.assertTrue(landed, f"no swap at {step}")
                    assert_covered(step)

                view = viewer._image_view
                shown = pump_until(lambda: view.isVisible()
                                    and view._item is not None)
                self.assertTrue(shown, "overview never displayed")
                goto(4.0, 0.0, 0.0, "zoom4-center")
                goto(4.0, -1500.0, -800.0, "pan-nw")
                goto(4.0, 1500.0, -600.0, "pan-ne")
                goto(4.0, 0.0, 1200.0, "pan-s")
                goto(8.0, 600.0, 600.0, "zoom8-se")
                goto(8.0, -500.0, 500.0, "pan-sw")
            finally:
                viewer.close()

    def test_refine_overlays_without_moving_or_reloading(self) -> None:
        import tempfile
        import time

        from PySide6.QtGui import QTransform
        from PySide6.QtWidgets import QApplication

        from lynceus.ui.viewers.raster_2d import Raster2DViewer

        QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as tmp:
            # Over the decimation budget: zooming refines for real.
            rows, cols = np.mgrid[0:2500, 0:2500]
            arr = (rows * 0.05 + cols * 0.03).astype(np.float32)
            path = _write_tif(Path(tmp) / "big.tif", arr, -9999.0)
            viewer = Raster2DViewer("raster")
            viewer.resize(800, 600)
            viewer.show()
            try:
                overs = []
                orig_over = viewer._start_overview
                viewer._start_overview = lambda: (
                    overs.append(1), orig_over())[1]
                viewer.set_payload({"file": str(path)})

                def pump_until(condition, timeout_s=20.0):
                    deadline = time.time() + timeout_s
                    while time.time() < deadline:
                        QApplication.instance().processEvents()
                        time.sleep(0.02)
                        if condition():
                            return True
                    return False

                view = viewer._image_view
                shown = pump_until(lambda: view.isVisible()
                                   and view._item is not None)
                self.assertTrue(shown)
                self.assertEqual(len(overs), 1)
                # Deterministic deep zoom (absolute scale, no gesture math).
                view.resetTransform()
                view.setTransform(QTransform().scale(8.0, 8.0))
                view.centerOn(view.sceneRect().center())
                first_item = view._item
                view._refine_timer.start()
                swapped = pump_until(
                    lambda: (view._item is not None
                             and view._item is not first_item),
                    timeout_s=25.0)
                self.assertTrue(swapped, "no single-image swap happened")
                # The view must not jump when the swap lands, and no
                # overview reload may fire afterwards. Center tolerance is
                # sub-pixel-screen (late layout reflows); a snap would move
                # hundreds of scene units.
                before = (view.transform().m11(),
                          view.mapToScene(view.viewport().rect().center()))
                time.sleep(2.0)
                pump_until(lambda: False, timeout_s=2.0)
                after = (view.transform().m11(),
                         view.mapToScene(view.viewport().rect().center()))
                self.assertEqual(before[0], after[0])
                self.assertLess(abs(before[1].x() - after[1].x()), 2.0)
                self.assertLess(abs(before[1].y() - after[1].y()), 2.0)
                self.assertEqual(len(overs), 1)
                self.assertTrue(viewer._placeholder.isHidden())
            finally:
                viewer.close()


class GradientLegendTests(unittest.TestCase):
    def test_legend_shows_bar_with_min_max(self) -> None:
        from PySide6.QtWidgets import QApplication

        from lynceus.ui.viewers.raster_2d import (
            Raster2DViewer,
            _hex_rgb,
        )

        QApplication.instance() or QApplication([])
        viewer = Raster2DViewer("raster")
        try:
            viewer._set_gradient_legend(
                (10.0, 73.0, [(10.0, _hex_rgb("#ffffcc")),
                              (73.0, _hex_rgb("#800026"))])
            )
            self.assertTrue(viewer._legend.isVisibleTo(viewer))
            labels = [
                viewer._legend_layout.itemAt(i).widget().text()
                for i in range(viewer._legend_layout.count())
                if hasattr(viewer._legend_layout.itemAt(i).widget(), "text")
            ]
            self.assertIn("10.00", labels)
            self.assertIn("73.00", labels)
        finally:
            viewer.deleteLater()


class RasterStatsStreamTests(unittest.TestCase):
    """Validity stats without full-array reads (28 GB mosaics styling)."""

    def test_min_max_exact_across_strips(self) -> None:
        from lynceus.processing.qml_style import _read_raster_stats

        with tempfile.TemporaryDirectory() as tmp:
            rng = np.random.default_rng(3)
            arr = rng.uniform(0, 500, (2500, 1800)).astype(np.float32)
            arr[0, 0] = -9999.0
            arr[100, 100] = np.nan
            arr[2000, 1500] = 600.0
            arr[2499, 1799] = -5.0
            path = _write_tif(Path(tmp) / "big.tif", arr, -9999.0)
            stats = _read_raster_stats(path)
            assert stats is not None
            self.assertAlmostEqual(stats[0], -5.0, places=3)
            self.assertAlmostEqual(stats[1], 600.0, places=3)
            self.assertEqual(stats[2], -9999.0)

    def test_all_nodata_yields_none(self) -> None:
        from lynceus.processing.qml_style import _read_raster_stats

        with tempfile.TemporaryDirectory() as tmp:
            arr = np.full((1300, 1300), -9999.0, dtype=np.float32)
            path = _write_tif(Path(tmp) / "void.tif", arr, -9999.0)
            self.assertIsNone(_read_raster_stats(path))

    def test_discrete_verdict_across_strips(self) -> None:
        from lynceus.processing.qml_style import _looks_discrete

        with tempfile.TemporaryDirectory() as tmp:
            rng = np.random.default_rng(5)
            mask = np.zeros((2500, 1800), dtype=np.float32)
            mask[rng.uniform(0, 1, mask.shape) < 0.5] = 1.0
            mask[rng.uniform(0, 1, mask.shape) < 0.1] = 4.0
            path = _write_tif(Path(tmp) / "mask.tif", mask, -9999.0)
            self.assertTrue(_looks_discrete(path))
            cont = rng.uniform(0, 50, (1300, 1300)).astype(np.float32)
            path = _write_tif(Path(tmp) / "cont.tif", cont, -9999.0)
            self.assertFalse(_looks_discrete(path))
            many = rng.integers(0, 12, (100, 100)).astype(np.float32)
            path = _write_tif(Path(tmp) / "many.tif", many, -9999.0)
            self.assertFalse(_looks_discrete(path))

    def test_generate_qml_on_large_raster(self) -> None:
        from lynceus.processing.qml_style import (
            generate_qml,
            read_ramp_stops,
        )

        with tempfile.TemporaryDirectory() as tmp:
            rng = np.random.default_rng(9)
            arr = rng.uniform(100, 400, (1500, 1500)).astype(np.float32)
            path = _write_tif(Path(tmp) / "dtm_mosaic.tif", arr, -9999.0)
            self.assertTrue(generate_qml(str(path), ramp_key="dtm"))
            parsed = read_ramp_stops(str(path.with_suffix(".qml")))
            assert parsed is not None
            self.assertAlmostEqual(parsed[0], float(arr.min()), places=2)
            self.assertAlmostEqual(parsed[1], float(arr.max()), places=2)


if __name__ == "__main__":
    unittest.main()
