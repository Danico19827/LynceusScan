# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Tests for the forestry tree nodes (treetops + crowns) and pit-free CHM.

Dual checkout: loads extension modules from dev (extensions/nodes) or
the extensions repo root (nodes/), skipping when absent. Fixtures are
analytic gaussian cones with known apexes; assertions pin count,
position, height and crown areas plus the EULA/metadata contract.
"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _find_node(folder: str, stem: str) -> Path | None:
    for base in (ROOT / "nodes", ROOT / "extensions" / "nodes"):
        candidate = base / folder / f"{stem}.py"
        if candidate.is_file():
            return candidate
    return None


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TREETOPS_PATH = _find_node("forestry", "treetops")
CROWNS_PATH = _find_node("forestry", "crown_delineation")


def _cones_chm(path: Path, pits: bool = False) -> Path:
    """60x60 m CHM, 1 m cells, 3 gaussian cones (apex 15/10/8 m)."""
    from affine import Affine

    from lynceus.processing.raster import write_geotiff

    n = 60
    xs = np.arange(n) + 0.5
    ys = (n - 1 - np.arange(n)) + 0.5
    xx, yy = np.meshgrid(xs, ys)
    grid = np.zeros((n, n), dtype=np.float32)
    for cx, cy, h in ((15.0, 45.0, 15.0), (40.0, 40.0, 10.0),
                      (30.0, 15.0, 8.0)):
        grid += (h * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2)
                            / (2 * 4.0 ** 2))).astype(np.float32)
    if pits:
        # Low cells inside the first cone (apex ~15 m): true pits.
        grid[14, 14] = 1.0
        grid[14, 15] = 1.0
    write_geotiff(str(path), grid, Affine(1.0, 0.0, 0.0, 0.0, -1.0, 60.0),
                  crs="EPSG:32721", nodata=-9999.0)
    return path


def _ctx(session: Path, extra: dict | None = None) -> dict:
    session.mkdir(parents=True, exist_ok=True)
    ctx = {"session_dir": str(session)}
    if extra:
        ctx.update(extra)
    return ctx


class TreetopsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if TREETOPS_PATH is None:
            raise unittest.SkipTest("treetops node not present")
        cls.mod = _load(TREETOPS_PATH, "treetops_under_test")

    def test_three_cones_three_treetops(self) -> None:
        import geopandas as gpd

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chm = _cones_chm(root / "chm.tif")
            out = self.mod.barrier_treetops(
                _ctx(root / "s", {"chm_mosaic_path": str(chm)}))
            frame = gpd.read_file(out["file"])
            self.assertEqual(len(frame), 3)
            got = sorted(frame["height_m"].tolist())
            self.assertTrue(
                abs(got[0] - 8.0) < 0.5
                and abs(got[1] - 10.0) < 0.5
                and abs(got[2] - 15.0) < 0.5)
            apexes = [(15.0, 45.0), (40.0, 40.0), (30.0, 15.0)]
            for pt in frame.geometry:
                dist = min(abs(pt.x - ax) + abs(pt.y - ay) for ax, ay in apexes)
                self.assertLess(dist, 2.0)

    def test_contract_metadata(self) -> None:
        mod = self.mod
        self.assertEqual(mod.NODE_ID, "forestry.treetops")
        self.assertEqual(mod.NODE_FOLDER, "forestry")
        self.assertTrue(mod.NODE_ID.startswith(mod.NODE_FOLDER + "."))
        self.assertTrue(mod.NODE_EULA)
        self.assertTrue(mod.NODE_DISCLAIMER)
        self.assertEqual(mod.NODE_LICENSE, "GPL-3.0-or-later")


class CrownsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if TREETOPS_PATH is None or CROWNS_PATH is None:
            raise unittest.SkipTest("tree nodes not present")
        cls.treetops = _load(TREETOPS_PATH, "treetops_under_test")
        cls.crowns = _load(CROWNS_PATH, "crowns_under_test")

    def test_three_cones_three_crowns(self) -> None:
        import geopandas as gpd

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chm = _cones_chm(root / "chm.tif")
            tops = self.treetops.barrier_treetops(
                _ctx(root / "s", {"chm_mosaic_path": str(chm)}))
            out = self.crowns.barrier_crown_delineation(
                _ctx(root / "s2", {"chm_mosaic_path": str(chm),
                                   "vector_path": tops["file"]}))
            frame = gpd.read_file(out["file"])
            self.assertEqual(len(frame), 3)
            self.assertTrue(bool((frame["crown_area_m2"] > 10.0).all()))
            self.assertTrue(bool((frame["height_m"] > 5.0).all()))

    def test_requires_treetops_vector(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chm = _cones_chm(root / "chm.tif")
            with self.assertRaisesRegex(RuntimeError, "Treetops vector"):
                self.crowns.barrier_crown_delineation(
                    _ctx(root / "s", {"chm_mosaic_path": str(chm)}))

    def test_contract_metadata(self) -> None:
        mod = self.crowns
        self.assertEqual(mod.NODE_ID, "forestry.crown_delineation")
        self.assertTrue(mod.NODE_ID.startswith(mod.NODE_FOLDER + "."))
        self.assertTrue(mod.NODE_EULA)
        self.assertTrue(mod.NODE_DISCLAIMER)


class PitFreeTests(unittest.TestCase):
    def test_pit_filled_only_when_enabled(self) -> None:
        from lynceus.nodes.lidar.terrain.generate_chm import (
            barrier_generate_chm,
        )
        from lynceus.processing.raster import read_geotiff, write_geotiff

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chm_src = _cones_chm(root / "src.tif", pits=True)
            arr, transform, crs = read_geotiff(chm_src)
            dtm = np.zeros_like(arr)
            write_geotiff(str(root / "dtm.tif"), dtm, transform,
                          crs="EPSG:32721", nodata=-9999.0)
            write_geotiff(str(root / "dsm.tif"), arr, transform,
                          crs="EPSG:32721", nodata=-9999.0)
            # Separate sessions: both barriers write chm_mosaic.tif and
            # the second run must not overwrite the first.
            base = {"dtm_mosaic_path": str(root / "dtm.tif"),
                    "dsm_mosaic_path": str(root / "dsm.tif"),
                    "smooth_enabled": False}
            plain = barrier_generate_chm(
                dict(base, session_dir=str(root / "s_plain")))
            pit = barrier_generate_chm(
                dict(base, session_dir=str(root / "s_pit"), pit_free=True))
            a0, _t, _c = read_geotiff(plain["file"])
            a1, _t, _c = read_geotiff(pit["file"])
            # The pitted cells read low without pit_free, filled with it.
            self.assertLess(float(a0[14, 14]), 5.0)
            self.assertGreater(float(a1[14, 14]), 5.0)
            # Untouched canopy identical elsewhere.
            same = (a0 != -9999.0) & (a1 != -9999.0)
            diff = np.abs(a0[same] - a1[same])
            self.assertGreater(float(diff.max()), 0.0)
            self.assertLess(float(np.median(diff)), 1.0)


class PitFreeNodataTests(unittest.TestCase):
    def test_voids_stay_nodata_without_warnings(self) -> None:
        import warnings

        from lynceus.processing.raster import pit_free_canopy

        rng = np.random.default_rng(4)
        dsm = np.full((50, 50), -9999.0)
        dsm[10:40, 10:40] = 400.0 + rng.uniform(0, 12, (30, 30))
        dsm[24:26, 24:26] = 401.0
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = pit_free_canopy(dsm, -9999.0)
        self.assertGreater(float(out[24, 24]), 405.0)
        self.assertTrue(bool((out[:5, :] == -9999.0).all()))


if __name__ == "__main__":
    unittest.main()
