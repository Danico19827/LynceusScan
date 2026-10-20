# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Point-cloud load performance (block C): dense clouds must load fast.

The loader and the GPU packing are Qt-free, so the optimizations are
covered without GL: the exact fast path keeps every point, the reservoir
still caps, the single-alloc interleave is value-identical to the old
hstack, and the shuffle order is built lazily (pure helper tested).
"""

from __future__ import annotations

import os
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

# NOTE: no QT_QPA_PLATFORM default here on purpose: the striped-upload
# test needs a real GL context and skips itself under offscreen.


def _write_laz(path: Path, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import laspy

    header = laspy.LasHeader(point_format=2, version="1.2")
    las = laspy.LasData(header)
    x = 500000.0 + np.arange(n) * 0.5
    y = 7000000.0 + np.arange(n) * 0.25
    z = 100.0 + (np.arange(n) % 50)
    las.x = x
    las.y = y
    las.z = z
    red = (np.arange(n) * 7 % 256).astype(np.uint16)
    las.red = red
    las.green = (255 - red).astype(np.uint16)
    las.blue = (red // 2).astype(np.uint16)
    las.classification = (np.arange(n) % 19).astype(np.uint8)
    las.write(path)
    return x, red / 255.0, (np.arange(n) % 19).astype(np.uint8)


class InterleaveTests(unittest.TestCase):
    def test_single_alloc_matches_hstack(self) -> None:
        from lynceus.ui.viewers.point_cloud_3d import _interleaved_points

        rng = np.random.default_rng(7)
        positions = (rng.random((500, 3)) * 100).astype(np.float32)
        rgb = rng.random((500, 3)).astype(np.float32)
        classes = (np.arange(500) % 19).astype(np.uint8)
        for mode, extra in (
            ("rgb", {"rgb": rgb}),
            ("classification", {"classes": classes}),
            ("elevation", {}),
            ("single", {}),
        ):
            with self.subTest(mode=mode):
                got = _interleaved_points(
                    positions,
                    extra.get("rgb"),
                    extra.get("classes"),
                    mode,
                )
                self.assertEqual(got.shape, (500, 6))
                self.assertEqual(got.dtype, np.float32)
                np.testing.assert_array_equal(got[:, :3], positions)
                self.assertTrue(np.all(np.isfinite(got[:, 3:])))

    def test_classification_colors_match_lut(self) -> None:
        from lynceus.ui.viewers.point_cloud_3d import (
            CLASS_COLORS,
            _interleaved_points,
        )

        positions = np.zeros((19, 3), dtype=np.float32)
        classes = np.arange(19, dtype=np.uint8)
        got = _interleaved_points(positions, None, classes, "classification")
        np.testing.assert_array_equal(
            got[:, 3:], np.asarray(CLASS_COLORS, dtype=np.float32))


class ShuffledOrderTests(unittest.TestCase):
    def test_permutation_and_deterministic(self) -> None:
        from lynceus.ui.viewers.point_cloud_3d import _shuffled_order

        first = _shuffled_order(1000)
        second = _shuffled_order(1000)
        self.assertEqual(first.dtype, np.int32)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(np.sort(first), np.arange(1000))


class SplitPiecesTests(unittest.TestCase):
    def test_split_covers_rows_under_ceiling(self) -> None:
        from lynceus.ui.viewers.point_cloud_3d import _GLView

        data = np.zeros((2500, 6), dtype=np.float32)
        pieces = _GLView._split_pieces(data, 1000)
        self.assertEqual([p.shape[0] for p in pieces], [1000, 1000, 500])
        self.assertLessEqual(max(p.nbytes for p in pieces), 1000 * 24)
        single = _GLView._split_pieces(data[:500], 1000)
        self.assertEqual(len(single), 1)
        # Stripes stay under common single-allocation caps (~256 MB).
        self.assertEqual(_GLView.vbo_split_points * 24, 240_000_000)

    def test_viewer_first_show_uploads_and_paints(self) -> None:
        """Full viewer path: decode → GL init on show → upload → paint.

        Regression: hiding the GL viewport until the first upload meant a
        hidden QOpenGLWidget never initialized, so points waited forever.
        """
        import tempfile
        import time

        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            self.skipTest("needs a real GL context")
        from PySide6.QtWidgets import QApplication

        from lynceus.ui.viewers.point_cloud_3d import PointCloud3DViewer

        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cloud.laz"
            _write_laz(path, 5000)
            viewer = PointCloud3DViewer("point_cloud")
            uploaded: list[int] = []
            viewer._gl.points_uploaded.connect(uploaded.append)
            try:
                viewer.show()
                viewer.set_payload({"file": str(path)})
                deadline = time.time() + 60.0
                while time.time() < deadline and not uploaded:
                    app.processEvents()
                    time.sleep(0.02)
                self.assertEqual(uploaded, [5000])
                self.assertTrue(viewer._gl.isVisible())
                self.assertTrue(viewer._placeholder.isHidden())
                rows = sum(
                    count for _vbo, count in viewer._gl._points_vbos)
                self.assertEqual(rows, 5000)
                app.processEvents()
                time.sleep(0.2)
            finally:
                viewer.close()
                viewer.deleteLater()

    def test_striped_upload_completes_on_real_gl(self) -> None:
        """End-to-end striped VBO upload (never offscreen: needs GL)."""
        import time

        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            self.skipTest("needs a real GL context")
        from PySide6.QtWidgets import QApplication

        from lynceus.ui.viewers.point_cloud_3d import (
            _GLView,
            _interleaved_points,
        )

        app = QApplication.instance() or QApplication([])
        old_split = _GLView.vbo_split_points
        _GLView.vbo_split_points = 1000
        self.addCleanup(setattr, _GLView, "vbo_split_points", old_split)
        rng = np.random.default_rng(11)
        positions = (rng.random((2500, 3)) * 50).astype(np.float32)
        data = _interleaved_points(positions, None, None, "elevation")
        view = _GLView()
        uploaded: list[int] = []
        view.points_uploaded.connect(uploaded.append)
        try:
            view.show()
            view.queue_points_upload(data, 2500)
            deadline = time.time() + 30.0
            while time.time() < deadline and not uploaded:
                app.processEvents()
                time.sleep(0.02)
            self.assertEqual(uploaded, [2500])
            rows = sum(count for _vbo, count in view._points_vbos)
            self.assertEqual(rows, 2500)
            self.assertGreaterEqual(len(view._points_vbos), 2)
            app.processEvents()
            time.sleep(0.2)
        finally:
            view.close()
            view.deleteLater()

    def test_uploaded_points_reach_pixels_on_real_gl(self) -> None:
        """Painted points (never offscreen: needs GL).

        Regression: allocate() without a prior bind() is a silent no-op,
        so uploads "completed" into empty buffers and points never painted
        anywhere (only axes/grid ever showed). Pixel readback is the only
        net that catches it: completion signals lie.
        """
        import time

        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            self.skipTest("needs a real GL context")
        from PySide6.QtWidgets import QApplication

        from lynceus.ui.viewers.point_cloud_3d import (
            _GLView,
            _interleaved_points,
        )

        app = QApplication.instance() or QApplication([])
        rng = np.random.default_rng(11)
        positions = (rng.random((2500, 3)) * 50).astype(np.float32)
        data = _interleaved_points(positions, None, None, "elevation")
        view = _GLView()
        uploaded: list[int] = []
        view.points_uploaded.connect(uploaded.append)
        try:
            view.resize(800, 600)
            view.show()
            view.set_points(positions, None, None, None, "elevation")
            view.queue_points_upload(data, 2500)
            deadline = time.time() + 30.0
            while time.time() < deadline and not uploaded:
                app.processEvents()
                time.sleep(0.02)
            self.assertEqual(uploaded, [2500])
            view._lines_count = 0  # points only: lines would mask them
            for _ in range(5):
                app.processEvents()
                time.sleep(0.1)
            img = view.grabFramebuffer()
            self.assertFalse(img.isNull())
            ptr = img.constBits()
            arr = np.frombuffer(ptr, dtype=np.uint8).reshape(
                img.height(), img.bytesPerLine() // 4, 4)
            bright = int((arr[:, :, :3].astype(int).sum(axis=2)
                          .ravel() > 150).sum())
            self.assertGreater(bright, 15000)
        finally:
            view.close()
            view.deleteLater()


class LoaderPathTests(unittest.TestCase):
    def test_exact_path_keeps_everything(self) -> None:
        from lynceus.ui.viewers.point_cloud_3d import _PointCloudLoader

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cloud.laz"
            x, red, classes = _write_laz(path, 3000)
            loader = _PointCloudLoader([{"file": str(path)}])
            result = loader.run(threading.Event())
            self.assertIsNotNone(result)
            positions, rgb, out_classes, _order = result
            # No order is built eagerly anymore (density 100 never reads it).
            self.assertIsNone(_order)
            self.assertEqual(positions.shape[0], 3000)
            # Localized around the file center: spacing survives, magnitude
            # is local (not projected), and everything stays finite.
            np.testing.assert_allclose(
                np.diff(positions[:, 0]), 0.5, rtol=0, atol=0.02)
            self.assertLess(float(np.abs(positions).max()), 5000.0)
            self.assertTrue(np.all(np.isfinite(positions)))
            self.assertIsNotNone(rgb)
            np.testing.assert_allclose(rgb[:, 0], red, rtol=1e-5, atol=1e-6)
            np.testing.assert_array_equal(out_classes, classes)

    def test_reservoir_path_still_caps(self) -> None:
        from lynceus.ui.viewers.point_cloud_3d import _PointCloudLoader

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cloud.laz"
            _write_laz(path, 3000)
            loader = _PointCloudLoader([])
            out = loader._load_tile(
                {"file": str(path)}, 100, np.zeros(3), threading.Event()
            )
            positions, _rgb, _classes, has_rgb, has_classes = out
            self.assertTrue(has_rgb)
            self.assertTrue(has_classes)
            self.assertLessEqual(positions.shape[0], 100)
            self.assertGreater(positions.shape[0], 0)

    def test_reservoir_sample_uniform_and_exact(self) -> None:
        """Over budget: full scan + reservoir (uniform, exact count)."""
        import laspy

        from lynceus.ui.viewers.point_cloud_3d import _PointCloudLoader

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "big.laz"
            n = 20_000
            _write_laz(path, n)
            seen: list[tuple[int, int]] = []
            loader = _PointCloudLoader([], on_progress=lambda d, t: seen.append((d, t)))
            loader._progress_total = n
            with laspy.open(path) as reader:
                header = reader.header
                origin = (
                    np.asarray(header.mins, dtype=np.float64)
                    + np.asarray(header.maxs, dtype=np.float64)
                ) / 2.0
            out = loader._load_tile(
                {"file": str(path)}, 1000, origin, threading.Event()
            )
            positions, _rgb, _classes, _hr, _hc = out
            self.assertEqual(positions.shape[0], 1000)
            self.assertTrue(seen)
            self.assertEqual(seen[-1][0], n)
            span = float(positions[:, 0].max() - positions[:, 0].min())
            self.assertGreater(span, (n * 0.5) * 0.8)


class PreviewCacheTests(unittest.TestCase):
    def _arrays(self, n: int = 1000):
        rng = np.random.default_rng(4)
        positions = (rng.random((n, 3)) * 100).astype(np.float32)
        rgb = (rng.random((n, 3))).astype(np.float32)
        classes = (np.arange(n) % 19).astype(np.uint8)
        origin = np.array([1.0, 2.0, 3.0])
        return positions, rgb, classes, origin

    def test_roundtrip(self) -> None:
        from lynceus.ui.viewers.point_cloud_3d import (
            load_preview_cache,
            save_preview_cache,
        )

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            positions, rgb, classes, origin = self._arrays()
            target = save_preview_cache(
                directory, "abc123", 999, positions, rgb, classes, origin)
            self.assertIsNotNone(target)
            stored = load_preview_cache(target)
            self.assertIsNotNone(stored)
            out_pos, out_rgb, out_cls, out_origin, out_budget = stored
            np.testing.assert_array_equal(out_pos, positions)
            np.testing.assert_array_equal(out_rgb, rgb)
            np.testing.assert_array_equal(out_cls, classes)
            np.testing.assert_array_equal(out_origin, origin)
            self.assertEqual(out_budget, 999)

    def test_missing_and_corrupt_are_none(self) -> None:
        from lynceus.ui.viewers.point_cloud_3d import load_preview_cache

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.assertIsNone(load_preview_cache(directory / "nope.npz"))
            bad = directory / "bad.npz"
            bad.write_bytes(b"not a zip")
            self.assertIsNone(load_preview_cache(bad))

    def test_find_exact_and_larger(self) -> None:
        from lynceus.ui.viewers.point_cloud_3d import (
            find_cached_preview,
            preview_cache_name,
            save_preview_cache,
        )

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            positions, rgb, classes, origin = self._arrays()
            save_preview_cache(
                directory, "fp1", 10_000, positions, rgb, classes, origin)
            save_preview_cache(
                directory, "fp1", 50_000, positions, rgb, classes, origin)
            (directory / "preview_fp1_garbage.npz").touch()
            found = find_cached_preview(directory, "fp1", 10_000)
            self.assertEqual(found[1], 10_000)
            found = find_cached_preview(directory, "fp1", 20_000)
            self.assertEqual(found[1], 50_000)
            self.assertIsNone(find_cached_preview(directory, "fp1", 99_000))
            self.assertIsNone(find_cached_preview(directory, "other", 10))
            self.assertEqual(preview_cache_name("fp1", 7), "preview_fp1_7.npz")

    def test_downsample_exact_and_deterministic(self) -> None:
        from lynceus.ui.viewers.point_cloud_3d import reservoir_downsample

        positions, rgb, classes, _origin = self._arrays(2000)
        first = reservoir_downsample(positions, rgb, classes, 500, 123)
        second = reservoir_downsample(positions, rgb, classes, 500, 123)
        for got in (first, second):
            self.assertEqual(got[0].shape[0], 500)
        np.testing.assert_array_equal(first[0], second[0])
        kept = set(float(v) for v in first[0][:, 0])
        source = set(float(v) for v in positions[:, 0])
        self.assertTrue(kept <= source)
        # No-op at or under the cap.
        same = reservoir_downsample(positions, rgb, classes, 5000, 123)
        self.assertIs(same[0], positions)

    def test_prune_keeps_newest(self) -> None:
        import time

        from lynceus.ui.viewers.point_cloud_3d import save_preview_cache

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            positions, rgb, classes, origin = self._arrays(10)
            base = time.time() - 100
            for index in range(6):
                target = save_preview_cache(
                    directory, f"old{index}", 10, positions, rgb, classes,
                    origin)
                stamp = base + index
                os.utime(target, (stamp, stamp))
            save_preview_cache(
                directory, "new", 10, positions, rgb, classes, origin)
            names = sorted(p.name for p in directory.glob("preview_*.npz"))
            self.assertEqual(len(names), 5)
            self.assertIn("preview_new_10.npz", names)

    def test_run_uses_and_reuses_cache(self) -> None:
        from lynceus.ui.viewers.point_cloud_3d import (
            _PointCloudLoader,
            preview_cache_dir,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cloud.laz"
            _write_laz(path, 5000)
            cache = Path(tmp) / "cache"
            os.environ["LYNCEUS_PREVIEW_CACHE_DIR"] = str(cache)
            self.addCleanup(os.environ.pop, "LYNCEUS_PREVIEW_CACHE_DIR", None)
            first = _PointCloudLoader(
                [{"file": str(path)}]).run(threading.Event())
            self.assertIsNotNone(first)
            self.assertEqual(first[0].shape[0], 5000)
            self.assertEqual(len(list(preview_cache_dir().glob("*.npz"))), 1)
            second = _PointCloudLoader(
                [{"file": str(path)}]).run(threading.Event())
            np.testing.assert_array_equal(second[0], first[0])
            np.testing.assert_array_equal(second[1], first[1])
            np.testing.assert_array_equal(second[2], first[2])


class GpuHonestyTests(unittest.TestCase):
    def test_classify_gpu(self) -> None:
        from lynceus.ui.viewers.point_cloud_3d import classify_gpu

        self.assertEqual(
            classify_gpu("NVIDIA Corporation", "NVIDIA GeForce RTX 4070"),
            "discrete")
        self.assertEqual(
            classify_gpu("Intel", "Intel(R) Iris(R) Xe Graphics"), "integrated")
        self.assertEqual(classify_gpu("Intel", "Intel Arc A770"), "discrete")
        self.assertEqual(classify_gpu("", "llvmpipe (LLVM 15)"), "software")
        self.assertEqual(classify_gpu("MysteryCo", "Mystery 9000"), "unknown")

    def test_auto_budget_gpu_aware(self) -> None:
        from lynceus.ui.viewers.point_cloud_3d import (
            INTEGRATED_AUTO_POINT_CAP,
            MAX_DISPLAY_POINT_BUDGET,
            MIN_DISPLAY_POINT_BUDGET,
            _auto_point_budget,
        )

        rich = 64 * 1024 ** 3
        self.assertEqual(
            _auto_point_budget(rich, "discrete"),
            min(MAX_DISPLAY_POINT_BUDGET, int(rich * 0.10 / 88)))
        self.assertEqual(
            _auto_point_budget(rich, "integrated"), INTEGRATED_AUTO_POINT_CAP)
        self.assertEqual(
            _auto_point_budget(rich, "software"), INTEGRATED_AUTO_POINT_CAP)
        self.assertEqual(
            _auto_point_budget(rich, None),
            min(MAX_DISPLAY_POINT_BUDGET, int(rich * 0.10 / 88)))
        self.assertEqual(
            _auto_point_budget(1024, "discrete"), MIN_DISPLAY_POINT_BUDGET)

    def test_available_backends_hides_phantoms(self) -> None:
        from unittest import mock

        from lynceus.processing import acceleration as accel

        accel._cached_device_probe.cache_clear()
        self.addCleanup(accel._cached_device_probe.cache_clear)
        with mock.patch.object(
                accel, "_cuda_devices", return_value=[]), mock.patch.object(
                accel, "_opencl_devices", return_value=[]), mock.patch.object(
                accel, "_directml_devices", return_value=[]):
            self.assertEqual(accel.available_backends(), ["auto", "cpu"])
        accel._cached_device_probe.cache_clear()
        with mock.patch.object(
                accel, "_cuda_devices",
                return_value=[accel.AccelerationDevice(
                    "cuda:0", "GPU", "cuda", True, True)]):
            backends = accel.available_backends()
            self.assertIn("cuda", backends)
            self.assertEqual(backends[:2], ["auto", "cpu"])


if __name__ == "__main__":
    unittest.main()
