# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Tests for the Terrain Derivatives strategy family (Horn 1981).

Math fixtures with exact answers on analytic planes (slope/aspect per
orientation, flat convention, NODATA propagation, hand-computed
hillshade), family discovery (base inert, one instance per derivative),
effective capabilities per strategy, and one barrier end-to-end per
derivative over a written DTM. Stdlib + engine modules only.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from affine import Affine

from lynceus.nodes.ports import PortType
from lynceus.plugins.registry import manager
from lynceus.processing.raster import (
    aspect_from_dem,
    hillshade_from_dem,
    slope_from_dem,
)

FAMILY = "lynceus.nodes.raster.ops.terrain_derivatives"
DTM = PortType.DTM_MOSAIC
RASTER = PortType.RASTER


def _north_up_plane(gx: float, gy: float, n: int = 10, cell: float = 1.0):
    """North-up grid (row 0 = max y) of z = gx*x + gy*y."""
    xs = np.arange(n) * cell + cell / 2
    ys = (n - 1 - np.arange(n)) * cell + cell / 2
    xx, yy = np.meshgrid(xs, ys)
    return gx * xx + gy * yy, cell, cell


class HornMathTests(unittest.TestCase):
    def test_slope_exact_on_tilted_plane(self) -> None:
        dem, dx, dy = _north_up_plane(2.0, 3.0)
        got = slope_from_dem(dem, dx, dy)
        want = float(np.degrees(np.arctan(np.hypot(2.0, 3.0))))
        self.assertAlmostEqual(float(got[5, 5]), want, places=6)
        self.assertTrue(bool((got[1:-1, 1:-1] == got[5, 5]).all()))

    def test_aspect_cardinal_directions(self) -> None:
        # Aspect is the DESCENT direction: a plane rising eastward
        # descends westward (270), one rising westward descends
        # eastward (90); rising southward descends northward (0),
        # rising northward descends southward (180).
        cases = ((1.0, 0.0, 270.0), (0.0, -1.0, 0.0),
                 (-1.0, 0.0, 90.0), (0.0, 1.0, 180.0))
        for gx, gy, want in cases:
            with self.subTest(gx=gx, gy=gy):
                dem, dx, dy = _north_up_plane(gx, gy)
                got = aspect_from_dem(dem, dx, dy)
                self.assertAlmostEqual(float(got[5, 5]), want, places=6)

    def test_aspect_diagonal(self) -> None:
        dem, dx, dy = _north_up_plane(2.0, 3.0)
        got = aspect_from_dem(dem, dx, dy)
        want = float(np.degrees(np.arctan2(-2.0, -3.0)) % 360.0)
        self.assertAlmostEqual(float(got[5, 5]), want, places=6)

    def test_flat_reads_zero_slope_minus_one_aspect(self) -> None:
        dem = np.full((8, 8), 100.0)
        self.assertTrue(
            bool((slope_from_dem(dem, 1.0, 1.0)[1:-1, 1:-1] == 0.0).all()))
        self.assertTrue(
            bool((aspect_from_dem(dem, 1.0, 1.0)[1:-1, 1:-1] == -1.0).all()))

    def test_nodata_window_and_rim(self) -> None:
        dem, dx, dy = _north_up_plane(1.0, 1.0, n=12)
        dem[6, 6] = -9999.0
        for fn in (slope_from_dem,
                   lambda a, x, y: aspect_from_dem(a, x, y),
                   lambda a, x, y: hillshade_from_dem(a, x, y)):
            with self.subTest(fn=fn.__name__):
                got = fn(dem, dx, dy)
                self.assertTrue(bool((got[5:8, 5:8] == -9999.0).all()))
                self.assertTrue(bool((got[0, :] == -9999.0).all()))
                self.assertTrue(bool((got[2, 2] != -9999.0)))

    def test_hillshade_matches_hand_formula(self) -> None:
        dem, dx, dy = _north_up_plane(2.0, 3.0)
        got = hillshade_from_dem(dem, dx, dy, azimuth_deg=315.0,
                                 altitude_deg=45.0)
        sl = np.arctan(np.hypot(2.0, 3.0))
        asp = np.radians(float(np.degrees(np.arctan2(-2.0, -3.0)) % 360.0))
        zen = np.radians(45.0)
        want = 255.0 * (np.cos(zen) * np.cos(sl)
                        + np.sin(zen) * np.sin(sl)
                        * np.cos(np.radians(315.0) - asp))
        self.assertAlmostEqual(float(got[5, 5]), float(want), places=4)

    def test_hillshade_flat_is_uniform(self) -> None:
        dem = np.full((8, 8), 100.0)
        got = hillshade_from_dem(dem, 1.0, 1.0)
        want = 255.0 * np.cos(np.radians(45.0))
        self.assertTrue(
            bool(np.allclose(got[1:-1, 1:-1], want, atol=1e-9)))

    def test_z_factor_steepens(self) -> None:
        dem, dx, dy = _north_up_plane(0.5, 0.0)
        plain = float(slope_from_dem(dem, dx, dy)[5, 5])
        steep = float(slope_from_dem(dem, dx, dy, z_factor=2.0)[5, 5])
        self.assertGreater(steep, plain)
        self.assertAlmostEqual(
            steep, float(np.degrees(np.arctan(1.0))), places=6)


class FamilyDiscoveryTests(unittest.TestCase):
    def test_three_derivative_variants(self) -> None:
        manager.discover()
        keys = [v["key"] for v in manager.list_variants(FAMILY)]
        self.assertEqual(keys, ["aspect", "hillshade", "slope"])

    def test_base_inert_without_strategy(self) -> None:
        manager.discover()
        info = manager.node_info(FAMILY)
        self.assertIsNotNone(info)
        inputs, outputs = manager.load_ports(FAMILY)
        self.assertEqual(inputs, ())
        self.assertEqual(outputs, ())

    def test_effective_caps_per_derivative(self) -> None:
        from lynceus.processing.steps import effective_node_caps

        for key, fn in (
            ("slope", "barrier_terrain_slope"),
            ("aspect", "barrier_terrain_aspect"),
            ("hillshade", "barrier_terrain_hillshade"),
        ):
            caps = effective_node_caps(FAMILY, {"strategy": key})
            self.assertEqual(caps.get("barrier_task").__name__, fn)
            self.assertEqual(caps.get("output_port"), "raster")

    def test_resolved_ports_identical_across_derivatives(self) -> None:
        from lynceus.nodes._variants import resolved_ports

        for key in ("slope", "aspect", "hillshade"):
            self.assertEqual(
                resolved_ports(FAMILY, {"strategy": key}), ((DTM,), (RASTER,))
            )


def _write_dtm(path: Path) -> Path:
    from lynceus.processing.raster import write_geotiff

    xs = np.arange(0.5, 20.0, 1.0)
    grid = np.tile(100.0 + 0.5 * xs, (10, 1)).astype(np.float32)
    write_geotiff(str(path), grid, Affine(1.0, 0.0, 0.0, 0.0, -1.0, 10.0),
                  crs="EPSG:32721", nodata=-9999.0)
    return path


class BarrierEndToEndTests(unittest.TestCase):
    def test_each_barrier_writes_its_product(self) -> None:
        from lynceus.processing.raster import read_geotiff

        jobs = (
            ("slope", "slope.tif",
             float(np.degrees(np.arctan(0.5)))),
            ("aspect", "aspect.tif", 270.0),
            ("hillshade", "hillshade.tif", None),
        )
        for key, product, want in jobs:
            with self.subTest(derivative=key):
                if key == "slope":
                    from lynceus.nodes.raster.ops.terrain_derivative_slope import (
                        barrier_terrain_slope as fn,
                    )
                elif key == "aspect":
                    from lynceus.nodes.raster.ops.terrain_derivative_aspect import (
                        barrier_terrain_aspect as fn,
                    )
                else:
                    from lynceus.nodes.raster.ops.terrain_derivative_hillshade import (
                        barrier_terrain_hillshade as fn,
                    )
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    src = _write_dtm(root / "dtm.tif")
                    session = root / "s"
                    session.mkdir(parents=True, exist_ok=True)
                    out = fn({"session_dir": str(session),
                              "dtm_mosaic_path": str(src)})
                    self.assertEqual(out["node"], FAMILY)
                    self.assertTrue(Path(out["file"]).is_file())
                    self.assertEqual(Path(out["file"]).name, product)
                    arr, _t, _c = read_geotiff(out["file"])
                    core = arr[1:-1, 1:-1]
                    core = core[core != -9999.0]
                    if want is not None:
                        self.assertTrue(
                            bool(np.allclose(core, want, atol=1e-4)), key)
                    else:
                        self.assertTrue(bool((core >= 0.0).all()
                                            and (core <= 255.0).all()))

    def test_missing_input_raises(self) -> None:
        from lynceus.nodes.raster.ops.terrain_derivative_slope import (
            barrier_terrain_slope,
        )

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "needs a DTM"):
                barrier_terrain_slope({"session_dir": tmp})


if __name__ == "__main__":
    unittest.main()
