# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""O2 parity: vectorized replacements match the reference algorithms.

Each test compares the optimized path against an obviously-correct brute
force (or analytic) reference on small fixtures — never implementation
against itself.
"""

import statistics
import tempfile
import unittest
from pathlib import Path

import laspy
import numpy as np

from lynceus.processing.raster import (
    CountAccumulator,
    TileAccumulator,
    cloth_simulation_filter,
    median_nodata,
    progressive_morphological_filter,
    simple_morphological_filter,
    tile_transform,
)


def _tile_dict():
    return {
        "tile_id": "t",
        "core_x_min": 0.0,
        "core_x_max": 40.0,
        "core_y_min": 0.0,
        "core_y_max": 40.0,
    }


def _reference_median(arr: np.ndarray, size: int, nodata: float) -> np.ndarray:
    pad = size // 2
    padded = np.pad(arr, pad, mode="reflect")
    out = arr.copy()
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            window = padded[r : r + size, c : c + size]
            vals = window[window != nodata]
            if vals.size:
                out[r, c] = float(np.median(vals)) if arr[r, c] != nodata else nodata
    return out


class MedianNodataParityTests(unittest.TestCase):
    def test_matches_reference_with_holes(self) -> None:
        rng = np.random.default_rng(11)
        arr = (rng.uniform(0, 50, (37, 41))).astype(np.float32)
        arr[rng.uniform(0, 1, arr.shape) < 0.15] = -9999.0
        got = median_nodata(arr, size=3, nodata=-9999.0)
        want = _reference_median(arr, 3, -9999.0)
        # NODATA pattern identical; valid cells equal within float tolerance.
        self.assertTrue(np.array_equal(got == -9999.0, want == -9999.0))
        np.testing.assert_allclose(
            got[got != -9999.0], want[want != -9999.0], rtol=1e-4
        )

    def test_all_nodata_returns_input(self) -> None:
        arr = np.full((8, 8), -9999.0, dtype=np.float32)
        np.testing.assert_array_equal(median_nodata(arr, nodata=-9999.0), arr)


class RasterGridBudgetTests(unittest.TestCase):
    def test_normal_tile_grid_keeps_dimensions(self) -> None:
        _, cols, rows = tile_transform(
            _tile_dict(), 1.0, memory_budget_bytes=1024 * 1024
        )

        self.assertEqual((cols, rows), (40, 40))

    def test_tiny_resolution_is_rejected_before_accumulator_allocation(self) -> None:
        budget = 1024 * 1024

        with self.assertRaisesRegex(ValueError, "per-worker grid budget"):
            TileAccumulator(
                _tile_dict(), 1e-12, "min", memory_budget_bytes=budget
            )
        with self.assertRaisesRegex(ValueError, "per-worker grid budget"):
            CountAccumulator(
                _tile_dict(), 1e-12, memory_budget_bytes=budget
            )

    def test_non_positive_or_non_finite_resolution_is_rejected(self) -> None:
        for resolution in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(resolution=resolution):
                with self.assertRaisesRegex(ValueError, "cell size"):
                    tile_transform(_tile_dict(), resolution)

    def test_ground_filters_reject_unbounded_point_grids(self) -> None:
        x = np.array([0.0, 40.0])
        y = np.array([0.0, 40.0])
        z = np.array([1.0, 2.0])
        filters = (
            progressive_morphological_filter,
            cloth_simulation_filter,
            simple_morphological_filter,
        )

        for ground_filter in filters:
            with self.subTest(filter=ground_filter.__name__):
                with self.assertRaisesRegex(
                    ValueError, "tile probe.*per-worker grid budget"
                ):
                    ground_filter(
                        x,
                        y,
                        z,
                        cell_size=1e-12,
                        memory_budget_bytes=1024 * 1024,
                        tile_id="probe",
                    )


class ReducerParityTests(unittest.TestCase):
    def _points(self, n=20000, seed=5):
        rng = np.random.default_rng(seed)
        x = rng.uniform(0, 40, n)
        y = rng.uniform(0, 40, n)
        z = rng.uniform(390, 410, n)
        return x, y, z

    def test_median_matches_statistics(self) -> None:
        x, y, z = self._points()
        acc = TileAccumulator(_tile_dict(), 2.0, "median")
        acc.add(x, y, z)
        grid, _ = acc.result()
        # Brute force per cell.
        c = np.floor(x / 2.0).astype(int)
        r = np.floor((40.0 - y) / 2.0).astype(int)
        for cell in range(20 * 20):
            sel = z[(r * 20 + c) == cell]
            want = float(statistics.median(sel)) if sel.size else -9999.0
            got = float(grid.flat[cell])
            if want == -9999.0:
                self.assertEqual(got, -9999.0)
            else:
                self.assertAlmostEqual(got, want, places=4)

    def test_idw_matches_direct_formula(self) -> None:
        x, y, z = self._points(n=5000)
        acc = TileAccumulator(_tile_dict(), 4.0, "idw")
        acc.add(x, y, z)
        grid, _ = acc.result()
        c = np.floor(x / 4.0).astype(int)
        r = np.floor((40.0 - y) / 4.0).astype(int)
        flat = np.clip(r, 0, 9) * 10 + np.clip(c, 0, 9)
        for cell in (0, 17, 55, 99):
            sel = z[flat == cell]
            if not sel.size:
                self.assertEqual(float(grid.flat[cell]), -9999.0)
                continue
            if sel.size == 1:
                want = float(sel[0])
            else:
                w = np.ones(sel.size) / np.maximum((sel - sel.mean()) ** 2, 1e-6)
                want = float(np.sum(w * sel) / np.sum(w))
            self.assertAlmostEqual(float(grid.flat[cell]), want, places=4)


class ClassifyStreamParityTests(unittest.TestCase):
    def _write_cloud(self, root: Path, xs, ys, zs) -> Path:
        n = len(xs)
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.scales = np.array([0.01, 0.01, 0.01])
        header.offsets = np.zeros(3)
        rec = laspy.ScaleAwarePointRecord.zeros(
            n, point_format=header.point_format,
            scales=header.scales, offsets=header.offsets,
        )
        rec.x = np.asarray(xs)
        rec.y = np.asarray(ys)
        rec.z = np.asarray(zs)
        rec.classification = np.zeros(n, dtype=np.uint8)
        path = root / "tile.laz"
        with laspy.open(str(path), mode="w", header=header) as writer:
            writer.write_points(rec)
        return path

    def test_flat_plane_all_ground(self) -> None:
        from lynceus.nodes.lidar.terrain.classify_ground_pmf import (
            tile_classify_ground_pmf as tile_classify_ground,
        )

        rng = np.random.default_rng(2)
        n = 20000
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = self._write_cloud(
                root,
                rng.uniform(0, 40, n),
                rng.uniform(0, 40, n),
                400.0 + rng.uniform(-0.2, 0.2, n),
            )
            ctx = {"session_dir": str(root / "s"), "node_iid": "c",
                   "tile_id": "t"}
            tile = {"tile_id": "t", "file": str(src)}
            out = tile_classify_ground(tile, ctx)
            with laspy.open(out["output"]) as reader:
                cls = np.concatenate(
                    [np.asarray(ch.classification) for ch in reader.chunk_iterator(2_000_000)]
                )
            self.assertEqual(len(cls), n)
            self.assertTrue(bool((cls == 2).all()))

    def test_spike_not_ground_and_deterministic(self) -> None:
        from lynceus.nodes.lidar.terrain.classify_ground_pmf import (
            tile_classify_ground_pmf as tile_classify_ground,
        )

        rng = np.random.default_rng(4)
        n = 20000
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            xs = rng.uniform(0, 40, n)
            ys = rng.uniform(0, 40, n)
            zs = np.full(n, 400.0) + rng.uniform(-0.2, 0.2, n)
            zs[:30] = 460.0  # narrow tall spike
            src = self._write_cloud(root, xs, ys, zs)
            ctx = {"session_dir": str(root / "s"), "node_iid": "c",
                   "tile_id": "t"}
            tile = {"tile_id": "t", "file": str(src)}
            first = tile_classify_ground(tile, ctx)
            with laspy.open(first["output"]) as reader:
                cls = np.concatenate(
                    [np.asarray(ch.classification) for ch in reader.chunk_iterator(2_000_000)]
                )
            self.assertGreater(int((cls == 2).sum()), n // 2)
            self.assertGreater(int((cls != 2).sum()), 0)
            # Deterministic re-run from a fresh session dir.
            ctx2 = {"session_dir": str(root / "s2"), "node_iid": "c",
                    "tile_id": "t"}
            second = tile_classify_ground(tile, ctx2)
            with laspy.open(second["output"]) as reader:
                cls2 = np.concatenate(
                    [np.asarray(ch.classification) for ch in reader.chunk_iterator(2_000_000)]
                )
            np.testing.assert_array_equal(cls, cls2)


class PMFEarlyExitTests(unittest.TestCase):
    def test_flat_surface_keeps_everything(self) -> None:
        rng = np.random.default_rng(9)
        n = 50000
        x = rng.uniform(0, 100, n)
        y = rng.uniform(0, 100, n)
        z = np.full(n, 400.0)
        mask = progressive_morphological_filter(
            x, y, z, cell_size=1.0, slope=1.0, intercept=0.5,
            initial_window_m=2.0, max_window_m=8.0,
        )
        self.assertTrue(bool(mask.all()))


if __name__ == "__main__":
    unittest.main()
