# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Paper-fidelity tests for the ground filters (T0 rigor harness).

Each test pins observable behavior derived from the reference paper on an
analytic synthetic cloud with a known-correct answer (no laspy, no runs):

- PMF (Zhang et al., IEEE TGRS 41(4), 2003, DOI 10.1109/TGRS.2003.814625):
  differential threshold dh_T(k) = s*(w_k - w_{k-1})*c + dh_0 over odd
  windows. A 1.0 m x 5 m bump must be removed at the 7 m window
  (threshold 0.9 m) while an absolute schedule would keep it.
- CSF (Zhang et al., Remote Sens. 8(4):283, 2016): cloth follows a steep
  plane with rigidness=3 and buries a box.
- SMRF (Pingel et al., IEEE GRSL 10(5), 2013): fixed threshold at every
  scale — a sub-threshold bump is kept, a wide tall one is removed.
"""

from __future__ import annotations

import unittest

import numpy as np

from lynceus.processing.raster import (
    adaptive_tin_filter,
    cloth_simulation_filter,
    multiscale_curvature_filter,
    progressive_morphological_filter,
    simple_morphological_filter,
)


def _plane(n: int = 20000, seed: int = 21, slope: float = 0.0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 40, n)
    y = rng.uniform(0, 40, n)
    z = 400.0 + slope * x + rng.uniform(-0.1, 0.1, n)
    return x, y, z


def _with_bump(height: float, half: float, n_bump: int = 400, seed: int = 23):
    """Flat plane with a compact square bump centered at (20, 20).

    Background points inside the square are displaced first: otherwise
    every cell minimum stays on the plane and the bump is invisible to
    the min-surface (it would only trip the final gate, testing
    nothing about the opening schedule).
    """
    x, y, z = _plane()
    keep = (np.abs(x - 20) > half) | (np.abs(y - 20) > half)
    x, y, z = x[keep], y[keep], z[keep]
    rng = np.random.default_rng(seed)
    bx = rng.uniform(20 - half, 20 + half, n_bump)
    by = rng.uniform(20 - half, 20 + half, n_bump)
    bz = np.full(n_bump, 400.0 + height)
    return (
        np.concatenate([x, bx]),
        np.concatenate([y, by]),
        np.concatenate([z, bz]),
    )


class PmfScheduleTests(unittest.TestCase):
    def test_bump_removed_by_differential_threshold(self) -> None:
        # 1.0 m high, 5 m wide bump. Paper schedule (s=0.3, dh_0=0.3,
        # cell=1 m, windows 3/5/7/9): at the 7 m window the opening erases
        # the bump and diff (1.0) > dh_T = 0.3*(7-5)*1+0.3 = 0.9.
        # An absolute schedule (0.3*w+0.3) would keep it (2.7 at 8 m).
        x, y, z = _with_bump(height=1.0, half=2.5)
        ground = progressive_morphological_filter(
            x, y, z, cell_size=1.0, slope=0.3, intercept=0.3,
            initial_window_m=3.0, max_window_m=9.0,
        )
        is_bump = (np.abs(x - 20) <= 2.5) & (np.abs(y - 20) <= 2.5) & (z > 400.5)
        self.assertTrue(bool(is_bump.any()))
        self.assertFalse(bool(ground[is_bump].any()))
        plane = ~is_bump
        self.assertGreater(int(ground[plane].sum()), int(plane.sum() * 0.99))

    def test_tilted_plane_survives_opening(self) -> None:
        # Morphological opening is exact on affine functions: a clean
        # tilted plane must be (almost) fully kept.
        x, y, z = _plane(slope=0.05)
        ground = progressive_morphological_filter(
            x, y, z, cell_size=1.0, slope=0.3, intercept=0.3,
            initial_window_m=2.0, max_window_m=8.0,
        )
        self.assertGreater(int(ground.sum()), int(len(z) * 0.99))


class CsfSemanticsTests(unittest.TestCase):
    def test_steep_plane_followed_with_rigidness_3(self) -> None:
        # Per the paper rigidness=3 is the steep-terrain setting: a 0.3
        # slope plane must be kept as ground.
        x, y, z = _plane(slope=0.3, seed=31)
        ground = cloth_simulation_filter(
            x, y, z, cell_size=0.5, class_threshold=0.5,
            rigidness=3, iterations=300,
        )
        self.assertGreater(int(ground.sum()), int(len(z) * 0.95))

    def test_box_buried_under_cloth(self) -> None:
        x, y, z = _with_bump(height=60.0, half=2.5, n_bump=30, seed=33)
        ground = cloth_simulation_filter(
            x, y, z, cell_size=1.0, class_threshold=0.5,
            rigidness=3, iterations=100,
        )
        is_bump = z > 430.0
        self.assertTrue(bool(is_bump.any()))
        self.assertFalse(bool(ground[is_bump].any()))
        self.assertGreater(int(ground[~is_bump].sum()), int((~is_bump).sum() * 0.95))


class SmrfThresholdTests(unittest.TestCase):
    def test_sub_threshold_bump_kept(self) -> None:
        # Fixed threshold 0.4 m at every scale: a 0.2 m bump never exceeds
        # it, whatever the window.
        x, y, z = _with_bump(height=0.2, half=2.0)
        ground = simple_morphological_filter(
            x, y, z, cell_size=1.0, slope=0.2, max_window_m=18.0,
            threshold=0.4,
        )
        is_bump = (np.abs(x - 20) <= 2.0) & (np.abs(y - 20) <= 2.0) & (z > 400.1)
        self.assertTrue(bool(is_bump.any()))
        self.assertGreater(int(ground[is_bump].sum()), int(is_bump.sum() * 0.95))

    def test_wide_tall_bump_removed(self) -> None:
        # A 2 m high, 10 m wide bump: steep edge cells wait for large
        # windows, where the opening erases it and diff (2.0) > 0.4.
        x, y, z = _with_bump(height=2.0, half=5.0, n_bump=800)
        ground = simple_morphological_filter(
            x, y, z, cell_size=1.0, slope=0.2, max_window_m=18.0,
            threshold=0.4,
        )
        is_bump = (np.abs(x - 20) <= 5.0) & (np.abs(y - 20) <= 5.0) & (z > 400.5)
        self.assertTrue(bool(is_bump.any()))
        self.assertFalse(bool(ground[is_bump].any()))


class MccForestTests(unittest.TestCase):
    @staticmethod
    def _forest_patch():
        # 25x25 m fully vegetated patch with NO ground returns inside:
        # the paper's raison d'etre (local-minimum filters cannot work
        # here). A 25 m patch also exceeds the PMF default max window.
        x, y, z = _plane(seed=41)
        keep = (np.abs(x - 20) > 12.5) | (np.abs(y - 20) > 12.5)
        x, y, z = x[keep], y[keep], z[keep]
        rng = np.random.default_rng(42)
        bx = rng.uniform(7.5, 32.5, 6000)
        by = rng.uniform(7.5, 32.5, 6000)
        bz = 400.0 + rng.uniform(2.0, 15.0, 6000)
        return (
            np.concatenate([x, bx]),
            np.concatenate([y, by]),
            np.concatenate([z, bz]),
        )

    def _mask(self, x, y, z):
        return multiscale_curvature_filter(
            x, y, z, cell_size=1.0, scale=1.5, threshold=0.3,
            domains=3, max_iterations=10,
        )

    def test_vegetated_patch_removed(self) -> None:
        x, y, z = self._forest_patch()
        ground = self._mask(x, y, z)
        in_patch = (np.abs(x - 20) <= 12.5) & (np.abs(y - 20) <= 12.5)
        self.assertTrue(bool(in_patch.any()))
        self.assertLess(int(ground[in_patch].sum()), int(in_patch.sum() * 0.05))
        plane = ~in_patch
        self.assertGreater(int(ground[plane].sum()), int(plane.sum() * 0.95))

    def test_deterministic(self) -> None:
        x, y, z = self._forest_patch()
        np.testing.assert_array_equal(self._mask(x, y, z), self._mask(x, y, z))

    def test_empty_cloud(self) -> None:
        empty = np.zeros(0)
        self.assertEqual(self._mask(empty, empty, empty).size, 0)


class AtinTinTests(unittest.TestCase):
    def _mask(self, x, y, z):
        return adaptive_tin_filter(
            x, y, z, cell_size=1.0, seed_m=10.0, max_angle_deg=6.0,
            max_dist_m=1.0, max_iterations=10,
        )

    def test_box_removed_plane_kept(self) -> None:
        rng = np.random.default_rng(51)
        n = 20000
        x = rng.uniform(0, 40, n)
        y = rng.uniform(0, 40, n)
        z = np.full(n, 400.0) + rng.uniform(-0.1, 0.1, n)
        keep = (np.abs(x - 20) > 4) | (np.abs(y - 20) > 4)
        x, y, z = x[keep], y[keep], z[keep]
        rb = np.random.default_rng(52)
        bx = rb.uniform(16, 24, 1500)
        by = rb.uniform(16, 24, 1500)
        bz = np.full(1500, 406.0)
        x = np.concatenate([x, bx])
        y = np.concatenate([y, by])
        z = np.concatenate([z, bz])
        ground = self._mask(x, y, z)
        is_roof = (np.abs(x - 20) <= 4) & (np.abs(y - 20) <= 4) & (z > 403)
        self.assertTrue(bool(is_roof.any()))
        self.assertFalse(bool(ground[is_roof].any()))
        self.assertGreater(int(ground[~is_roof].sum()),
                           int((~is_roof).sum() * 0.95))

    def test_terrace_both_levels_kept(self) -> None:
        # The TIN follows abrupt relief facet by facet: both sides of a
        # 1.5 m step must survive (the morphology filters round it off).
        rng = np.random.default_rng(53)
        n = 20000
        x = rng.uniform(0, 40, n)
        y = rng.uniform(0, 40, n)
        z = np.full(n, 400.0) + rng.uniform(-0.1, 0.1, n)
        z[x >= 20.0] += 1.5
        ground = self._mask(x, y, z)
        for side in (x < 18.0, x > 22.0):
            self.assertGreater(int(ground[side].sum()),
                               int(side.sum() * 0.95))

    def test_deterministic(self) -> None:
        rng = np.random.default_rng(55)
        n = 5000
        x = rng.uniform(0, 40, n)
        y = rng.uniform(0, 40, n)
        z = np.full(n, 400.0) + rng.uniform(-0.1, 0.1, n)
        np.testing.assert_array_equal(self._mask(x, y, z),
                                      self._mask(x, y, z))

    def test_empty_cloud(self) -> None:
        empty = np.zeros(0)
        self.assertEqual(self._mask(empty, empty, empty).size, 0)


if __name__ == "__main__":
    unittest.main()
