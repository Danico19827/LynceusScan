# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Heterogeneous CHM: overlap-windowed reads, refusal sized past the union.

Different extents must not materialize full arrays: the plan streams
valid boxes only, reads cover the overlap (cell-identical to full reads),
and the refusal counts live temporaries by active option. Same message
shape as the legacy refusal (no catalog churn).
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _write_grid(path: Path, array: np.ndarray, origin: tuple[float, float],
                nodata: float = -9999.0) -> Path:
    import rasterio
    from affine import Affine

    with rasterio.open(
        str(path), "w", driver="GTiff", height=array.shape[0],
        width=array.shape[1], count=1, dtype="float32",
        crs="EPSG:32612",
        transform=Affine(1.0, 0.0, origin[0], 0.0, -1.0, origin[1]),
        nodata=nodata,
    ) as ds:
        ds.write(array.astype(np.float32), 1)
    return path


def _ramps(w: int, h: int):
    rows, cols = np.mgrid[0:h, 0:w]
    base = 100.0 + rows * 0.01 + cols * 0.01
    return base.astype(np.float32)


def _offset_pair(root: Path):
    """Two same-resolution grids sharing one field, shifted 20x10 cells."""
    field_rows, field_cols = np.mgrid[0:90, 0:120]
    field = (100.0 + field_rows * 0.01 + field_cols * 0.01).astype(np.float32)
    dtm = _write_grid(root / "dtm.tif", field[0:80, 0:100],
                      (500000.0, 7000000.0))
    dsm = _write_grid(root / "dsm.tif", field[10:90, 20:120] + 5.0,
                      (500020.0, 6999990.0))
    return dtm, dsm


class OverlapPlanTests(unittest.TestCase):
    def test_windows_cover_overlap(self) -> None:
        from lynceus.processing.raster import _chm_overlap_plan

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dtm = _write_grid(root / "dtm.tif", _ramps(100, 80),
                              (500000.0, 7000000.0))
            dsm = _write_grid(root / "dsm.tif", _ramps(100, 80),
                              (500020.0, 6999990.0))
            plan = _chm_overlap_plan(dtm, dsm, -9999.0)
            self.assertFalse(plan["empty_dtm"])
            self.assertFalse(plan["empty_dsm"])
            self.assertFalse(plan["disjoint"])
            self.assertEqual(plan["union_cells"], 2 * 100 * 80)
            # Overlap in DTM pixels: cols 20..99, rows 10..79 (+1px margin,
            # clamped at the file edge).
            win = plan["dtm_win"]
            self.assertEqual(
                (win.col_off, win.row_off), (19, 9))
            self.assertEqual(
                (win.col_off + win.width, win.row_off + win.height),
                (100, 80))
            self.assertGreater(plan["overlap_cells"], 0)
            self.assertLess(plan["overlap_cells"], plan["union_cells"])

    def test_empty_and_disjoint(self) -> None:
        from lynceus.processing.raster import _chm_overlap_plan

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dtm = _write_grid(root / "dtm.tif", _ramps(50, 40),
                              (500000.0, 7000000.0))
            void = _write_grid(
                root / "void.tif",
                np.full((50, 40), -9999.0, dtype=np.float32),
                (500000.0, 7000000.0))
            plan = _chm_overlap_plan(dtm, void, -9999.0)
            self.assertTrue(plan["empty_dsm"])
            far = _write_grid(root / "far.tif", _ramps(50, 40),
                              (600000.0, 7000000.0))
            plan = _chm_overlap_plan(dtm, far, -9999.0)
            self.assertTrue(plan["disjoint"])


class WindowedComputeTests(unittest.TestCase):
    def test_offset_grids_match_manual(self) -> None:
        """Overlap cells equal DSM-DTM (smooth off): exact values."""
        import rasterio

        from lynceus.processing.raster import compute_chm

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dtm, dsm = _offset_pair(root)
            out = root / "chm.tif"
            compute_chm(dtm, dsm, out, crs="EPSG:32612",
                        smooth_enabled=False,
                        memory_budget_bytes=10 * 1024 ** 3)
            with rasterio.open(out) as src:
                arr = src.read(1)
                transform = src.transform
            # Fully-valid overlap crops to itself: constant 5.0 everywhere.
            self.assertEqual(arr.shape, (70, 80))
            np.testing.assert_allclose(arr, 5.0, rtol=0, atol=1e-4)
            self.assertAlmostEqual(transform.c, 500020.0)
            self.assertAlmostEqual(transform.f, 6999990.0)

    def test_smooth_and_pit_complete(self) -> None:
        from lynceus.processing.raster import compute_chm, pit_free_canopy

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dtm, dsm = _offset_pair(root)
            out = root / "chm_smooth.tif"
            compute_chm(dtm, dsm, out, crs="EPSG:32612",
                        smooth_enabled=True, smooth_sigma=0.5,
                        memory_budget_bytes=10 * 1024 ** 3)
            import rasterio

            with rasterio.open(out) as src:
                arr = src.read(1)
            # Constant field survives NODATA-aware smoothing untouched.
            np.testing.assert_allclose(arr, 5.0, rtol=0, atol=1e-4)
            pit = pit_free_canopy(
                (_ramps(100, 80) + 5.0), -9999.0)
            self.assertEqual(pit.shape, (80, 100))
            self.assertTrue(np.all(np.isfinite(pit)))

    def test_refusal_fires_and_passes(self) -> None:
        from lynceus.processing.raster import compute_chm

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dtm = _write_grid(root / "dtm.tif", _ramps(100, 80),
                              (500000.0, 7000000.0))
            dsm = _write_grid(root / "dsm.tif", _ramps(100, 80) + 5.0,
                              (500020.0, 6999990.0))
            with self.assertRaisesRegex(ValueError, "CHM mosaic would need"):
                compute_chm(dtm, dsm, root / "nope.tif", crs="EPSG:32612",
                            smooth_enabled=False, memory_budget_bytes=1024)
            out = root / "fits.tif"
            compute_chm(dtm, dsm, out, crs="EPSG:32612",
                        smooth_enabled=False,
                        memory_budget_bytes=10 * 1024 ** 3)
            self.assertTrue(out.exists())

    def test_stack_matches_legacy_with_smooth(self) -> None:
        """Oversized stackable grids stream (halo) instead of refusing."""
        import rasterio

        from lynceus.nodes.lidar.terrain.generate_chm import (
            barrier_generate_chm,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Non-constant difference (single spike) so smoothing mixes
            # neighbours and the halo path is actually exercised.
            field = _ramps(100, 80)
            spike = field + 5.0
            spike[40, 50] += 20.0
            dtm = _write_grid(root / "dtm.tif", field,
                              (500000.0, 7000000.0))
            dsm = _write_grid(root / "dsm.tif", spike,
                              (500020.0, 6999990.0))
            sessions = [root / "s_direct", root / "s_stack"]
            for session in sessions:
                session.mkdir()

            def _run(session: Path, budget: int):
                ctx = {
                    "session_dir": str(session),
                    "dtm_mosaic_path": str(dtm),
                    "dsm_mosaic_path": str(dsm),
                    "crs": "EPSG:32612",
                    "_raster_grid_memory_budget_bytes": budget,
                    "_bounded_notes": [],
                }
                out = barrier_generate_chm(ctx)
                with rasterio.open(out["file"]) as src:
                    return (src.read(1), src.transform,
                            list(ctx["_bounded_notes"]))

            direct, direct_t, direct_notes = _run(sessions[0], 10 * 1024 ** 3)
            stacked, stacked_t, stacked_notes = _run(sessions[1], 1)
            self.assertEqual(direct_notes, [])
            self.assertEqual(len(stacked_notes), 1)
            self.assertIn("full resolution", stacked_notes[0])
            self.assertEqual(direct_t, stacked_t)
            np.testing.assert_array_equal(direct, stacked)

    def test_disjoint_and_empty_errors(self) -> None:
        from lynceus.processing.raster import compute_chm

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dtm = _write_grid(root / "dtm.tif", _ramps(50, 40),
                              (500000.0, 7000000.0))
            far = _write_grid(root / "far.tif", _ramps(50, 40),
                              (600000.0, 7000000.0))
            with self.assertRaisesRegex(ValueError, "do not overlap"):
                compute_chm(dtm, far, root / "nope.tif", crs="EPSG:32612",
                            memory_budget_bytes=10 * 1024 ** 3)
            void = _write_grid(
                root / "void.tif",
                np.full((50, 40), -9999.0, dtype=np.float32),
                (500000.0, 7000000.0))
            with self.assertRaisesRegex(ValueError, "without valid data"):
                compute_chm(dtm, void, root / "nope.tif", crs="EPSG:32612",
                            memory_budget_bytes=10 * 1024 ** 3)


if __name__ == "__main__":
    unittest.main()
