# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Unit tests for bounded raster mosaics: direct-vs-windowed byte
equivalence (merge + finishing ops), all-NODATA parity, heterogeneous
fallback, stacked CHM/derivative equivalence, fit checks and the
dispatch-time input estimate (globs + segment outputs).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from affine import Affine

from lynceus.processing.raster import (
    _mosaic_plan,
    compute_chm,
    finish_mosaic,
    mosaic_fits,
    slope_from_dem,
    stack_mosaic,
    write_geotiff,
)

CRS = "EPSG:32621"


def _tile(path: Path, value: float, west: float, north: float = 40.0,
          width: int = 30, height: int = 30) -> Path:
    arr = np.full((height, width), value, np.float32)
    arr[0, 0] = -9999.0
    write_geotiff(path, arr, Affine(1.0, 0.0, west, 0.0, -1.0, north),
                  crs=CRS)
    return path


def _read(path: Path):
    import rasterio

    with rasterio.open(str(path)) as src:
        return src.read(1), src.transform, src.crs, src.nodata


class FinishEquivalenceTests(unittest.TestCase):
    def setUp(self) -> None:
        import shutil

        self.tmp = Path(tempfile.mkdtemp(prefix="winmos_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.a = _tile(self.tmp / "a.tif", 5.0, 0.0)
        self.b = _tile(self.tmp / "b.tif", 9.0, 24.0)
        self.srcs = [self.a, self.b]

    def test_windowed_matches_direct_with_ops(self) -> None:
        direct = self.tmp / "direct.tif"
        finish_mosaic(
            self.srcs, direct, crs=CRS, fill_dist_m=3.0, smooth_sigma=1.0,
            median_size=3, resolution=1.0, memory_budget_bytes=None)
        for budget in (1, 1 << 20, 1 << 30):
            out = self.tmp / f"win_{budget}.tif"
            finish_mosaic(
                self.srcs, out, crs=CRS, fill_dist_m=3.0, smooth_sigma=1.0,
                median_size=3, resolution=1.0, memory_budget_bytes=budget)
            da, dt, _, _ = _read(direct)
            wa, wt, _, _ = _read(out)
            self.assertEqual(da.shape, wa.shape)
            self.assertEqual(tuple(dt), tuple(wt))
            self.assertTrue(np.array_equal(da, wa, equal_nan=True),
                            f"budget {budget}")

    def test_no_ops_matches_merge(self) -> None:
        from lynceus.processing.raster import merge_geotiffs

        ref = self.tmp / "ref.tif"
        merge_geotiffs([str(s) for s in self.srcs], CRS, ref)
        out = self.tmp / "win.tif"
        finish_mosaic(self.srcs, out, crs=CRS, memory_budget_bytes=1)
        ra, _, _, _ = _read(ref)
        wa, _, _, _ = _read(out)
        self.assertTrue(np.array_equal(ra, wa, equal_nan=True))

    def test_all_nodata_parity(self) -> None:
        n1 = _tile(self.tmp / "n1.tif", -9999.0, 0.0)
        n2 = _tile(self.tmp / "n2.tif", -9999.0, 30.0)
        for i, budget in enumerate((None, 1)):
            out = self.tmp / f"nodata{i}.tif"
            finish_mosaic([n1, n2], out, crs=CRS,
                          memory_budget_bytes=budget)
            arr, _, _, _ = _read(out)
            self.assertTrue((arr == -9999.0).all())

    def test_empty_raises_and_heterogeneous_stays_direct(self) -> None:
        with self.assertRaises(ValueError):
            finish_mosaic([], self.tmp / "x.tif", memory_budget_bytes=1)
        other = _tile(self.tmp / "o.tif", 1.0, 0.0)
        import rasterio

        with rasterio.open(other, "r+") as dst:
            dst.crs = rasterio.crs.CRS.from_epsg(32620)
        out = self.tmp / "hetero.tif"
        # Multi-CRS cannot window: legacy direct behavior (rio raises).
        with self.assertRaises(Exception):
            finish_mosaic([self.a, other], out, crs=CRS,
                          memory_budget_bytes=1)

    def test_fits_check(self) -> None:
        self.assertTrue(mosaic_fits(self.srcs, 3.0, None))
        self.assertTrue(mosaic_fits(self.srcs, 3.0, 1 << 40))
        self.assertFalse(mosaic_fits(self.srcs, 3.0, 1))
        self.assertTrue(mosaic_fits(self.srcs, 3.0, 0))


class StackEquivalenceTests(unittest.TestCase):
    def setUp(self) -> None:
        import shutil

        self.tmp = Path(tempfile.mkdtemp(prefix="stackmos_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        base = np.arange(900, dtype=np.float32).reshape(30, 30) + 100.0
        write_geotiff(self.tmp / "dtm.tif", base,
                      Affine(1.0, 0.0, 0.0, 0.0, -1.0, 30.0), crs=CRS)
        write_geotiff(self.tmp / "dsm.tif", base + 12.0,
                      Affine(1.0, 0.0, 0.0, 0.0, -1.0, 30.0), crs=CRS)
        self.dtm = self.tmp / "dtm.tif"
        self.dsm = self.tmp / "dsm.tif"

    def test_chm_windowed_matches_direct(self) -> None:
        ref = self.tmp / "chm_ref.tif"
        compute_chm(self.dtm, self.dsm, ref, crs=CRS, smooth_sigma=0.5,
                    max_height_m=60.0, min_height_m=0.0)

        def chm_op(wins, info):
            from lynceus.processing.raster import smooth_nodata

            nd = info["nodata"]
            dtm_w = np.asarray(wins[0], dtype=np.float32)
            dsm_w = np.asarray(wins[1], dtype=np.float32)
            both = (dtm_w != nd) & (dsm_w != nd)
            chm = np.full(dtm_w.shape, nd, dtype=np.float32)
            chm[both] = np.clip(dsm_w[both] - dtm_w[both], 0.0, 60.0)
            return smooth_nodata(chm, 0.5, nd)

        out = self.tmp / "chm_win.tif"
        stack_mosaic([self.dtm, self.dsm], out, op=chm_op, halo_cells=3,
                     extent="intersection", nodata=-9999.0,
                     memory_budget_bytes=1)
        ra, rt, _, _ = _read(ref)
        wa, wt, _, _ = _read(out)
        self.assertEqual(ra.shape, wa.shape)
        self.assertEqual(tuple(rt), tuple(wt))
        self.assertTrue(np.allclose(ra, wa, equal_nan=True))

    def test_heterogeneous_stack_raises(self) -> None:
        import rasterio

        with rasterio.open(self.dsm, "r+") as dst:
            dst.crs = rasterio.crs.CRS.from_epsg(32620)
        with self.assertRaises(ValueError):
            stack_mosaic([self.dtm, self.dsm], self.tmp / "x.tif",
                         op=lambda w, i: w[0], memory_budget_bytes=1)

    def test_dem_mosaic_matches_direct(self) -> None:
        from lynceus.processing.raster import dem_mosaic, load_dem_masked

        arr, transform, _, nd = load_dem_masked(self.dtm, -9999.0)
        ref = slope_from_dem(arr, 1.0, 1.0, nodata=nd).astype(np.float32)
        out = self.tmp / "slope_win.tif"
        _, empty = dem_mosaic(
            self.dtm, out,
            dem_fn=lambda a, cx, cy: slope_from_dem(a, cx, cy, nodata=nd),
            halo_cells=1, nodata=nd, memory_budget_bytes=1)
        self.assertFalse(empty)
        wa, _, _, _ = _read(out)
        self.assertEqual(ref.shape, wa.shape)
        self.assertTrue(np.allclose(ref, wa, equal_nan=True))

    def test_dem_mosaic_empty_branch(self) -> None:
        from lynceus.processing.raster import dem_mosaic

        void = _tile(self.tmp / "void.tif", -9999.0, 0.0)
        out = self.tmp / "void_out.tif"
        _, empty = dem_mosaic(
            void, out, dem_fn=lambda a, cx, cy: a, memory_budget_bytes=1)
        self.assertTrue(empty)
        arr, _, _, _ = _read(out)
        self.assertTrue((arr == -9999.0).all())


class EstimateTests(unittest.TestCase):
    def setUp(self) -> None:
        import shutil

        self.tmp = Path(tempfile.mkdtemp(prefix="estbytes_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_globs_and_nested_outputs_count(self) -> None:

        from lynceus.processing.executor import _barrier_input_bytes
        from lynceus.processing.steps import Task

        arts = self.tmp / "arts"
        (arts / "dtm").mkdir(parents=True)
        (arts / "dtm" / "t0.tif").write_bytes(b"x" * 1000)
        (arts / "dtm" / "sub").mkdir()
        (arts / "dtm" / "sub" / "t1.tif").write_bytes(b"y" * 500)
        ctx = {"session_dir": str(arts)}
        task = Task("d|merge", "d", "m", "barrier", None, (ctx,), set(),
                    input_globs=("dtm/**/*.tif",))
        self.assertEqual(_barrier_input_bytes(task), 1500)

    def test_segment_outputs_count(self) -> None:

        from lynceus.processing.executor import _barrier_input_bytes
        from lynceus.processing.steps import Task

        seg = self.tmp / "seg"
        seg.mkdir()
        (seg / "m.gpkg").write_bytes(b"z" * 700)
        ctx = {"session_dir": str(self.tmp),
               "segment_outputs": [{"g": {"file": str(seg / "m.gpkg")}}]}
        task = Task("c|merge", "c", "m", "barrier", None, (ctx,), set())
        self.assertEqual(_barrier_input_bytes(task), 700)

    def test_plan_shared_with_vrt(self) -> None:
        a = _tile(self.tmp / "a.tif", 1.0, 0.0)
        plan = _mosaic_plan([a])
        self.assertNotIn("skipped", plan)
        self.assertEqual((plan["width"], plan["height"]), (30, 30))


class NodeWiringTests(unittest.TestCase):
    """Wired barriers complete on a forced tiny budget (windowed path)."""

    def setUp(self) -> None:
        import shutil

        self.tmp = Path(tempfile.mkdtemp(prefix="winwire_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        arts = self.tmp / "arts"
        (arts / "dtm").mkdir(parents=True)
        for i, value in enumerate((10.0, 20.0)):
            arr = np.full((40, 40), value, np.float32)
            write_geotiff(arts / "dtm" / f"t{i}.tif", arr,
                          Affine(1.0, 0.0, i * 40.0, 0.0, -1.0, 40.0),
                          crs=CRS)
        self.ctx = {
            "session_dir": str(arts),
            "node_iid": "d1",
            "crs": CRS,
            "resolution": 1.0,
            "_raster_grid_memory_budget_bytes": 1 << 20,
        }

    def test_dtm_barrier_windowed(self) -> None:
        from lynceus.nodes.lidar.terrain.generate_dtm import (
            barrier_generate_dtm,
        )

        out = barrier_generate_dtm(dict(self.ctx))
        arr, _, _, _ = _read(Path(out["file"]))
        self.assertEqual(arr.shape, (40, 80))
        self.assertEqual(float(arr.min()), 10.0)
        self.assertEqual(float(arr.max()), 20.0)

    def test_slope_barrier_windowed(self) -> None:
        from lynceus.nodes.lidar.terrain.generate_dtm import (
            barrier_generate_dtm,
        )
        from lynceus.nodes.raster.ops.terrain_derivative_slope import (
            barrier_terrain_slope,
        )

        dtm_out = barrier_generate_dtm(dict(self.ctx))
        ctx = dict(self.ctx)
        ctx["dtm_mosaic_path"] = dtm_out["file"]
        out = barrier_terrain_slope(ctx)
        arr, _, _, _ = _read(Path(out["file"]))
        self.assertEqual(arr.shape, (40, 80))
        self.assertTrue(np.isfinite(arr).any())


    def test_chm_barrier_routes_by_fit(self) -> None:
        from lynceus.nodes.lidar.terrain.generate_chm import (
            barrier_generate_chm,
        )

        arts = Path(self.ctx["session_dir"])
        base = np.arange(1600, dtype=np.float32).reshape(40, 40) + 100.0
        write_geotiff(arts / "dtm_mosaic.tif", base,
                      Affine(1.0, 0.0, 0.0, 0.0, -1.0, 40.0), crs=CRS)
        write_geotiff(arts / "dsm_mosaic.tif", base + 12.0,
                      Affine(1.0, 0.0, 0.0, 0.0, -1.0, 40.0), crs=CRS)

        def _run(budget):
            ctx = dict(self.ctx)
            ctx["_raster_grid_memory_budget_bytes"] = budget
            ctx["_bounded_notes"] = []
            out = barrier_generate_chm(ctx)
            arr, _, _, _ = _read(Path(out["file"]))
            return arr, ctx["_bounded_notes"]

        # Fitting union goes direct (fast path, no bounded notice).
        direct, direct_notes = _run(1 << 30)
        self.assertEqual(direct_notes, [])
        # Oversized same-grid union streams window by window (notice on).
        windowed, windowed_notes = _run(1)
        self.assertEqual(len(windowed_notes), 1)
        self.assertIn("full resolution", windowed_notes[0])
        for arr in (direct, windowed):
            self.assertEqual(arr.shape, (40, 40))
            valid = arr[arr != -9999.0]
            self.assertTrue(valid.size > 0)
            self.assertAlmostEqual(float(valid.min()), 12.0, places=4)
            self.assertAlmostEqual(float(valid.max()), 12.0, places=3)


class BoundedNotesTests(unittest.TestCase):
    """Bounded-mode notices reach the payload warnings (one mechanism)."""

    def setUp(self) -> None:
        import shutil

        self.tmp = Path(tempfile.mkdtemp(prefix="bnotes_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.a = _tile(self.tmp / "a.tif", 5.0, 0.0)
        self.b = _tile(self.tmp / "b.tif", 9.0, 24.0)

    def test_windowed_appends_note_direct_does_not(self) -> None:
        notes: list = []
        finish_mosaic([self.a, self.b], self.tmp / "d.tif", crs=CRS,
                      memory_budget_bytes=10 ** 15, notes=notes)
        self.assertEqual(notes, [])
        notes = []
        finish_mosaic([self.a, self.b], self.tmp / "w.tif", crs=CRS,
                      memory_budget_bytes=1, notes=notes)
        self.assertEqual(len(notes), 1)
        self.assertIn("full resolution", notes[0])

    def test_wrapper_forwards_and_drains(self) -> None:
        from lynceus.processing.steps import _run_barrier_task

        ctx = {"_bounded_notes": ["n1"]}
        result = _run_barrier_task(
            lambda c: {"file": "x", "warnings": ["w0"]}, ctx)
        self.assertEqual(result["warnings"], ["w0", "n1"])
        self.assertNotIn("_bounded_notes", ctx)

    def test_wrapper_dedupes_and_ignores_non_dicts(self) -> None:
        from lynceus.processing.steps import _forward_bounded_notes

        result = _forward_bounded_notes(
            {"file": "x", "warnings": ["n1"]}, {"_bounded_notes": ["n1"]})
        self.assertEqual(result["warnings"], ["n1"])
        self.assertEqual(
            _forward_bounded_notes("x", {"_bounded_notes": ["n1"]}), "x")

    def test_streamed_tables_note(self) -> None:
        from lynceus.processing.chunked_merge import note_streamed

        ctx: dict = {}
        note_streamed(ctx, "grid_metrics.gpkg")
        self.assertEqual(len(ctx["_bounded_notes"]), 1)
        self.assertIn("grid_metrics.gpkg", ctx["_bounded_notes"][0])
        note_streamed(None, "x")  # ctx-safe, never raises


class GridSnapTests(unittest.TestCase):
    """Tile raster origins snap to the tiling grid (exact merges).

    Fairbanks 20261014: DTM/DSM at 0.3 m on 100 m tiles (100/0.3 =
    333.33 cells) left neighbors on different sub-pixel phases, the
    merger rejected them, and the direct fallback OOMed on a 27.9 GB
    dest. Snapping outward to res multiples from the tiling origin puts
    every fragment on one exact grid at any resolution.
    """

    OX, OY = -16471674.18, 9596234.55

    def _tile(self, col: int, row: int = 0, res: float = 1.0,
              span: int = 1):
        w = 100.0 * span
        return {
            "tile_id": "t",
            "core_x_min": self.OX + col * 100.0,
            "core_x_max": self.OX + col * 100.0 + w,
            "core_y_min": self.OY + row * 100.0,
            "core_y_max": self.OY + row * 100.0 + w,
            "grid_ox": self.OX,
            "grid_oy": self.OY,
        }

    def test_dividing_resolutions_snap_identity(self) -> None:
        from lynceus.processing.raster import tile_transform

        for res in (1.0, 0.5, 2.0, 5.0):
            with self.subTest(res=res):
                tile = self._tile(464, 80, res)
                transform, cols, rows = tile_transform(tile, res)
                self.assertEqual(transform.c, tile["core_x_min"])
                self.assertEqual(transform.f, tile["core_y_max"])
                side = int(100.0 / res)
                self.assertEqual((cols, rows), (side, side))

    def test_fractional_resolution_aligns_neighbors(self) -> None:
        from lynceus.processing.raster import tile_transform

        left, _, _ = tile_transform(self._tile(464, 80, 0.3), 0.3)
        right, _, _ = tile_transform(self._tile(465, 80, 0.3), 0.3)
        for edge in (left.c, right.c):
            quotient = (edge - self.OX) / 0.3
            self.assertLess(abs(quotient - round(quotient)), 1e-6)
        self.assertGreater(right.c, left.c)
        # Coverage never shrinks: snapped bounds contain the core.
        self.assertLessEqual(left.c, self.OX + 46400.0)

    def test_missing_anchor_keeps_legacy(self) -> None:
        from lynceus.processing.raster import tile_transform

        tile = {"tile_id": "t", "core_x_min": 0.5, "core_x_max": 100.5,
                "core_y_min": 0.5, "core_y_max": 100.5}
        transform, cols, rows = tile_transform(tile, 0.3)
        self.assertEqual((transform.c, transform.f), (0.5, 100.5))

    def test_misaligned_fragments_merge_at_fractional_res(self) -> None:
        import shutil

        from lynceus.processing.raster import (
            _mosaic_plan,
            finish_mosaic,
            tile_transform,
            write_geotiff,
        )

        tmp = Path(tempfile.mkdtemp(prefix="snapmerge_"))
        self.addCleanup(shutil.rmtree, tmp, True)
        paths = []
        for col, value in ((464, 10.0), (465, 20.0)):
            transform, cols, rows = tile_transform(
                self._tile(col, 80, 0.3), 0.3)
            arr = np.full((rows, cols), value, np.float32)
            path = tmp / f"t{col}.tif"
            write_geotiff(path, arr, transform, crs=CRS)
            paths.append(path)
        plan = _mosaic_plan(paths)
        self.assertNotIn("skipped", plan)
        out = tmp / "merged.tif"
        finish_mosaic(paths, out, crs=CRS, memory_budget_bytes=1)
        arr, out_transform, _, _ = _read(out)
        self.assertEqual(arr.shape[1], plan["width"])
        uniques = set(np.unique(arr[arr != -9999.0]).tolist())
        self.assertEqual(uniques, {10.0, 20.0})
        self.assertAlmostEqual(float(out_transform.c), float(plan["west"]))


    def test_legacy_dicts_backfill_then_merge(self) -> None:
        import shutil

        from lynceus.processing.raster import (
            _mosaic_plan,
            finish_mosaic,
            tile_transform,
            write_geotiff,
        )
        from lynceus.processing.tiler import backfill_grid_anchor

        # Legacy cached manifests predate grid_ox/grid_oy: the exact
        # Fairbanks shape (0.3 m fragments, .18-decimal tiling origin).
        tiles = []
        for col in (464, 465):
            tiles.append({
                "tile_id": f"t{col}",
                "col": col,
                "row": 80,
                "core_x_min": self.OX + col * 100.0,
                "core_x_max": self.OX + col * 100.0 + 100.0,
                "core_y_min": self.OY + 80 * 100.0,
                "core_y_max": self.OY + 80 * 100.0 + 100.0,
            })
        backfill_grid_anchor(tiles, 100.0)
        tmp = Path(tempfile.mkdtemp(prefix="snaplegacy_"))
        self.addCleanup(shutil.rmtree, tmp, True)
        paths = []
        for tile, value in zip(tiles, (10.0, 20.0)):
            transform, cols, rows = tile_transform(tile, 0.3)
            arr = np.full((rows, cols), value, np.float32)
            path = tmp / f"{tile['tile_id']}.tif"
            write_geotiff(path, arr, transform, crs=CRS)
            paths.append(path)
        plan = _mosaic_plan(paths)
        self.assertNotIn("skipped", plan)
        out = tmp / "merged.tif"
        finish_mosaic(paths, out, crs=CRS, memory_budget_bytes=1)
        arr, _, _, _ = _read(out)
        uniques = set(np.unique(arr[arr != -9999.0]).tolist())
        self.assertEqual(uniques, {10.0, 20.0})


class RefusalTests(unittest.TestCase):
    """Heterogeneous giants refuse honestly instead of OOMing opaquely."""

    def setUp(self) -> None:
        import shutil

        self.tmp = Path(tempfile.mkdtemp(prefix="refuse_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_refuse_reports_numbers(self) -> None:
        from lynceus.processing.raster import refuse_if_unfit

        path = self.tmp / "a.tif"
        path.write_bytes(b"x" * 1000)
        with self.assertRaisesRegex(ValueError, "MiB.*budget"):
            refuse_if_unfit([path], 3.0, 1, "Mosaic test.tif")
        refuse_if_unfit([path], 3.0, 10 ** 12, "Mosaic test.tif")
        refuse_if_unfit([path], 3.0, None, "Mosaic test.tif")
        refuse_if_unfit([self.tmp / "gone.tif"], 3.0, 1, "x")

    def test_heterogeneous_merge_refuses_on_tiny_budget(self) -> None:
        import rasterio

        from lynceus.processing.raster import finish_mosaic

        a = _tile(self.tmp / "a.tif", 1.0, 0.0)
        other = _tile(self.tmp / "o.tif", 1.0, 0.0)
        with rasterio.open(other, "r+") as dst:
            dst.crs = rasterio.crs.CRS.from_epsg(32620)
        with self.assertRaisesRegex(ValueError, "MiB.*budget"):
            finish_mosaic([a, other], self.tmp / "out.tif", crs=CRS,
                          memory_budget_bytes=1)

    def test_chm_heterogeneous_refuses_on_tiny_budget(self) -> None:
        from lynceus.processing.raster import compute_chm

        base = np.arange(100, dtype=np.float32).reshape(10, 10) + 100.0
        write_geotiff(self.tmp / "dtm.tif", base,
                      Affine(1.0, 0.0, 0.0, 0.0, -1.0, 10.0), crs=CRS)
        write_geotiff(self.tmp / "dsm.tif", base + 12.0,
                      Affine(1.0, 0.0, 0.0, 0.0, -1.0, 10.0), crs=CRS)
        with self.assertRaisesRegex(ValueError, "MiB.*budget"):
            compute_chm(self.tmp / "dtm.tif", self.tmp / "dsm.tif",
                        self.tmp / "chm.tif", crs=CRS,
                        memory_budget_bytes=1)


class VoidSkipTests(unittest.TestCase):
    """Void windows skip reads+compute with byte-identical results."""

    def setUp(self) -> None:
        import shutil

        self.tmp = Path(tempfile.mkdtemp(prefix="voidskip_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _frag(self, name: str, value: float, west: float, north: float = 40.0,
              size: int = 30):
        from lynceus.processing.raster import write_geotiff

        arr = np.full((size, size), value, np.float32)
        arr[0, 0] = -9999.0
        path = self.tmp / name
        write_geotiff(path, arr, Affine(1.0, 0.0, west, 0.0, -1.0, north),
                      crs=CRS)
        return path

    def test_finish_sparse_union_matches_direct(self) -> None:
        from lynceus.processing.raster import finish_mosaic

        a = self._frag("a.tif", 5.0, 0.0)
        b = self._frag("b.tif", 9.0, 5000.0)
        ref = self.tmp / "ref.tif"
        finish_mosaic([a, b], ref, crs=CRS, fill_dist_m=3.0,
                      smooth_sigma=1.0, memory_budget_bytes=None)
        out = self.tmp / "win.tif"
        finish_mosaic([a, b], out, crs=CRS, fill_dist_m=3.0,
                      smooth_sigma=1.0, memory_budget_bytes=1)
        ra, rt, _, _ = _read(ref)
        wa, wt, _, _ = _read(out)
        self.assertEqual(ra.shape, wa.shape)
        self.assertEqual(tuple(rt), tuple(wt))
        self.assertTrue(np.array_equal(ra, wa, equal_nan=True))

    def test_finish_halo_reaches_into_void(self) -> None:
        # Fill must act across the valid/void boundary: windows whose
        # core is void but whose halo holds valid cells still compute.
        from lynceus.processing.raster import finish_mosaic

        arr = np.full((30, 30), -9999.0, np.float32)
        arr[:, :10] = 7.0
        path = self.tmp / "edge.tif"
        write_geotiff(path, arr, Affine(1.0, 0.0, 0.0, 0.0, -1.0, 30.0),
                      crs=CRS)
        ref = self.tmp / "ref.tif"
        finish_mosaic([path], ref, crs=CRS, fill_dist_m=5.0,
                      memory_budget_bytes=None)
        out = self.tmp / "win.tif"
        finish_mosaic([path], out, crs=CRS, fill_dist_m=5.0,
                      memory_budget_bytes=1)
        ra, _, _, _ = _read(ref)
        wa, _, _, _ = _read(out)
        self.assertTrue(np.array_equal(ra, wa, equal_nan=True))
        # Fill actually crossed into the void (the test is not vacuous).
        self.assertTrue(bool((ra[:, 10:15] != -9999.0).any()))

    def test_dem_sparse_matches_direct(self) -> None:
        from lynceus.processing.raster import dem_mosaic, load_dem_masked

        arr = np.full((600, 600), -9999.0, np.float32)
        arr[100:200, 100:200] = np.arange(10000, dtype=np.float32).reshape(
            100, 100) + 100.0
        path = self.tmp / "dem.tif"
        write_geotiff(path, arr, Affine(1.0, 0.0, 0.0, 0.0, -1.0, 600.0),
                      crs=CRS)
        full, _, _, nd = load_dem_masked(path, -9999.0)
        ref = slope_from_dem(full, 1.0, 1.0, nodata=nd).astype(np.float32)
        out = self.tmp / "slope.tif"
        _, empty = dem_mosaic(
            path, out,
            dem_fn=lambda a, cx, cy: slope_from_dem(a, cx, cy, nodata=nd),
            halo_cells=1, nodata=nd, memory_budget_bytes=1)
        self.assertFalse(empty)
        wa, _, _, _ = _read(out)
        self.assertEqual(ref.shape, wa.shape)
        self.assertTrue(np.allclose(ref, wa, equal_nan=True))


class BigTiffTests(unittest.TestCase):
    """Products write BigTIFF (no 4 GiB classic ceiling)."""

    def test_writes_carry_bigtiff_magic(self) -> None:
        import shutil

        from lynceus.processing.raster import (
            _open_tiled_output,
            write_geotiff,
        )

        tmp = Path(tempfile.mkdtemp(prefix="bigtiff_"))
        self.addCleanup(shutil.rmtree, tmp, True)
        plain = tmp / "plain.tif"
        write_geotiff(plain, np.ones((4, 4), np.float32),
                      Affine(1.0, 0.0, 0.0, 0.0, -1.0, 4.0), crs=CRS)
        tiled = tmp / "tiled.tif"
        with _open_tiled_output(
            tiled, 4, 4, Affine(1.0, 0.0, 0.0, 0.0, -1.0, 4.0),
            CRS, -9999.0, 1, "float32", None,
        ) as dst:
            from rasterio.windows import Window

            dst.write(np.ones((4, 4), np.float32), 1,
                      window=Window(0, 0, 4, 4))
        for path in (plain, tiled):
            magic = int.from_bytes(path.read_bytes()[2:4], "little")
            self.assertEqual(magic, 43, path.name)
            arr, _, _, _ = _read(path)
            self.assertTrue(bool((arr == 1.0).all()))


if __name__ == "__main__":
    unittest.main()
