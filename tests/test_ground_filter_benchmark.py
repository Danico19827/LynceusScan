# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Benchmark: Type I/II errors of the ground filters on a realistic scene.

Scene (100x100 m tile, ~12 pts/m2, Gaussian noise sigma = 5 cm, ground
truth by construction): base plane, 1.5 m terrace step (must keep both
levels), 0.3 m road strip (must keep), two buildings, a 20x20 m fully
vegetated patch with NO ground returns, isolated cone trees, and a bridge
deck with visible ground beneath (deck removed, ground kept).

Metrics follow the ISPRS filter-test convention (Sithole & Vosselman):
Type I = ground lost (omission), Type II = object kept (commission).
Bounds below are regression pins with headroom, not accuracy claims;
each documents the method's known scope on that zone.
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

BASE = 400.0
SEED = 101


def build_scene():
    """Return (x, y, z, truth_ground, zones) for the benchmark tile."""
    rng = np.random.default_rng(SEED)
    extent, step = 100.0, 0.29  # ~12 pts/m2 uniform
    gx, gy = np.meshgrid(
        np.arange(0, extent, step), np.arange(0, extent, step)
    )
    x = gx.reshape(-1) + rng.uniform(-0.1, 0.1, gx.size)
    y = gy.reshape(-1) + rng.uniform(-0.1, 0.1, gy.size)
    z = np.full_like(x, BASE) + rng.normal(0, 0.05, x.size)
    truth = np.ones_like(x, dtype=bool)
    zones = {}

    # Terrace: +1.5 m step east of x = 60 (both levels are ground).
    terrace = x >= 60.0
    z[terrace] += 1.5
    zones["terrace"] = terrace & (np.abs(x - 60.0) > 2.0)

    # Road: +0.3 m strip, y in [45, 50] (must keep).
    road = (y >= 45.0) & (y < 50.0) & (x < 60.0)
    z[road] += 0.3
    zones["road"] = road

    # Buildings: solid boxes (non-ground). The small one fits the PMF
    # default window, the big one exceeds it (documents PMF's scope).
    buildings = np.zeros_like(x, dtype=bool)
    for cx, cy, hw, hd, h in ((20.0, 75.0, 4.0, 4.0, 6.0),
                              (72.0, 72.0, 7.5, 5.0, 6.0)):
        in_box = (np.abs(x - cx) <= hw) & (np.abs(y - cy) <= hd)
        buildings |= in_box
    # Box volume points replace the ground points inside.
    bx, by, bz = [], [], []
    for cx, cy, hw, hd, h in ((20.0, 75.0, 4.0, 4.0, 6.0),
                              (72.0, 72.0, 7.5, 5.0, 6.0)):
        n = 1500
        px = rng.uniform(cx - hw, cx + hw, n)
        py = rng.uniform(cy - hd, cy + hd, n)
        base = BASE + (1.5 if cx >= 60.0 else 0.0)
        pz = rng.uniform(base + h - 0.1, base + h + 0.1, n)
        bx.append(px)
        by.append(py)
        bz.append(pz)
    keep = ~buildings
    x, y, z = x[keep], y[keep], z[keep]
    truth = truth[keep]
    zones = {k: v[keep] for k, v in zones.items()}
    n_prev = len(x)
    x = np.concatenate([x] + bx)
    y = np.concatenate([y] + by)
    z = np.concatenate([z] + bz)
    n_box = len(x) - n_prev
    truth = np.concatenate([truth, np.zeros(n_box, dtype=bool)])
    zones = {k: np.concatenate([v, np.zeros(len(x) - len(v), dtype=bool)])
             for k, v in zones.items()}
    box_idx = np.arange(len(x))
    zones["buildings"] = box_idx >= n_prev
    # First box in the loop is the small one (8x8 m), second is big.
    zones["building_small"] = (box_idx >= n_prev) & (box_idx < n_prev + 1500)
    zones["building_big"] = box_idx >= n_prev + 1500

    # Forest patch 20x20 m at (20, 20): no ground returns inside.
    patch = (np.abs(x - 20.0) <= 10.0) & (np.abs(y - 20.0) <= 10.0)
    keep = ~patch
    x, y, z = x[keep], y[keep], z[keep]
    truth = truth[keep]
    zones = {k: v[keep] for k, v in zones.items()}
    n_veg = 8000
    vx = rng.uniform(10.0, 30.0, n_veg)
    vy = rng.uniform(10.0, 30.0, n_veg)
    vz = BASE + rng.uniform(2.0, 12.0, n_veg)
    n_prev = len(x)
    x = np.concatenate([x, vx])
    y = np.concatenate([y, vy])
    z = np.concatenate([z, vz])
    truth = np.concatenate([truth, np.zeros(n_veg, dtype=bool)])
    zones = {k: np.concatenate([v, np.zeros(n_veg, dtype=bool)])
             for k, v in zones.items()}
    zones["forest"] = np.arange(len(x)) >= n_prev

    # Isolated cone trees on the plane (non-ground).
    tree_pts = []
    for cx, cy in ((45.0, 20.0), (55.0, 30.0), (35.0, 60.0), (50.0, 80.0),
                   (80.0, 40.0), (10.0, 40.0), (85.0, 85.0), (5.0, 90.0)):
        n = 120
        rr = np.sqrt(rng.uniform(0, 1, n)) * 2.0
        th = rng.uniform(0, 2 * np.pi, n)
        tree_pts.append((cx + rr * np.cos(th), cy + rr * np.sin(th),
                         BASE + 8.0 * (1 - rr / 2.0)
                         + rng.normal(0, 0.1, n)))
    tx = np.concatenate([t[0] for t in tree_pts])
    ty = np.concatenate([t[1] for t in tree_pts])
    tz = np.concatenate([t[2] for t in tree_pts])
    n_prev = len(x)
    x = np.concatenate([x, tx])
    y = np.concatenate([y, ty])
    z = np.concatenate([z, tz])
    truth = np.concatenate([truth, np.zeros(len(tx), dtype=bool)])
    zones = {k: np.concatenate([v, np.zeros(len(tx), dtype=bool)])
             for k, v in zones.items()}
    zones["trees"] = np.arange(len(x)) >= n_prev

    # Bridge deck at +4 m over y in [75, 80], x in [40, 60]: deck points
    # added ON TOP of kept ground (deck removed, ground beneath kept).
    dx, dy = np.meshgrid(np.arange(40.0, 60.0, 0.5),
                         np.arange(75.0, 80.0, 0.5))
    dx = dx.reshape(-1) + rng.uniform(-0.1, 0.1, dx.size)
    dy = dy.reshape(-1) + rng.uniform(-0.1, 0.1, dy.size)
    dz = np.full_like(dx, BASE + 4.0) + rng.normal(0, 0.05, dx.size)
    n_prev = len(x)
    x = np.concatenate([x, dx])
    y = np.concatenate([y, dy])
    z = np.concatenate([z, dz])
    truth = np.concatenate([truth, np.zeros(len(dx), dtype=bool)])
    zones = {k: np.concatenate([v, np.zeros(len(dx), dtype=bool)])
             for k, v in zones.items()}
    zones["deck"] = np.arange(len(x)) >= n_prev

    return x, y, z, truth, zones


METHODS = {
    "pmf": lambda x, y, z: progressive_morphological_filter(
        x, y, z, cell_size=1.0, slope=0.3, intercept=0.3,
        initial_window_m=2.0, max_window_m=8.0,
    ),
    "csf": lambda x, y, z: cloth_simulation_filter(
        x, y, z, cell_size=0.5, class_threshold=0.5,
        rigidness=3, iterations=300,
    ),
    "smrf": lambda x, y, z: simple_morphological_filter(
        x, y, z, cell_size=1.0, slope=0.2, max_window_m=18.0,
        threshold=0.4,
    ),
    "mcc": lambda x, y, z: multiscale_curvature_filter(
        x, y, z, cell_size=1.0, scale=1.5, threshold=0.3,
        domains=3, max_iterations=10,
    ),
    "atin": lambda x, y, z: adaptive_tin_filter(
        x, y, z, cell_size=1.0, seed_m=20.0, max_angle_deg=6.0,
        max_dist_m=1.0, max_iterations=10,
    ),
}


def confusion(ground_mask: np.ndarray, truth: np.ndarray) -> dict:
    """Type I (omission) / Type II (commission) rates on a zone."""
    truth = np.asarray(truth, dtype=bool)
    total = max(1, int(truth.size))
    type1 = float((truth & ~ground_mask).sum() / max(1, int(truth.sum())))
    type2 = float((~truth & ground_mask).sum()
                  / max(1, int((~truth).sum())))
    total_err = float((ground_mask != truth).sum() / total)
    return {"type1": type1, "type2": type2, "total": total_err}


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.x, cls.y, cls.z, cls.truth, cls.zones = build_scene()
        cls.masks = {name: fn(cls.x, cls.y, cls.z)
                     for name, fn in METHODS.items()}

    def test_scene_sane(self) -> None:
        self.assertGreater(len(self.x), 100000)
        self.assertGreater(int(self.truth.sum()), int(len(self.x) * 0.8))
        for name in ("terrace", "road", "buildings", "building_small",
                     "building_big", "forest", "trees", "deck"):
            self.assertTrue(bool(self.zones[name].any()), name)

    def test_buildings_removed(self) -> None:
        for name, mask in self.masks.items():
            with self.subTest(method=name):
                small = self.zones["building_small"]
                kept = float(mask[small].sum() / max(1, int(small.sum())))
                self.assertLess(kept, 0.05, name)
        for name in ("csf", "smrf", "mcc", "atin"):
            with self.subTest(method=name):
                big = self.zones["building_big"]
                mask = self.masks[name]
                kept = float(mask[big].sum() / max(1, int(big.sum())))
                self.assertLess(kept, 0.05, name)

    def test_pmf_window_scope_documented(self) -> None:
        # The 15x10 m building exceeds the PMF default 8 m window: it is
        # kept by design. Enlarging the default window must update this.
        big = self.zones["building_big"]
        mask = self.masks["pmf"]
        kept = float(mask[big].sum() / max(1, int(big.sum())))
        self.assertGreater(kept, 0.5)

    def test_deck_removed_ground_beneath_kept(self) -> None:
        for name, mask in self.masks.items():
            with self.subTest(method=name):
                deck = self.zones["deck"]
                kept = float(mask[deck].sum() / max(1, int(deck.sum())))
                self.assertLess(kept, 0.05, name)

    def test_road_and_terrace_kept(self) -> None:
        # Breaklines lose a 1-2 cell strip by design (opening rounds the
        # step); the bound pins the strip, not perfection.
        for name, mask in self.masks.items():
            with self.subTest(method=name):
                for zone_name in ("road", "terrace"):
                    zone = self.zones[zone_name]
                    kept = float(mask[zone].sum()
                                 / max(1, int(zone.sum())))
                    self.assertGreater(kept, 0.80, (name, zone_name))

    def test_trees_mostly_removed(self) -> None:
        # Cone skirts within the labeling band of the plane read as
        # ground across the family; the bound pins the skirt, not zero.
        for name, mask in self.masks.items():
            with self.subTest(method=name):
                zone = self.zones["trees"]
                kept = float(mask[zone].sum() / max(1, int(zone.sum())))
                self.assertLess(kept, 0.30, name)

    def test_forest_patch_by_method(self) -> None:
        # MCC and ATIN clear patches without ground returns best; cloth
        # drapes the canopy without them; window-capped morphology keeps
        # the low layer. Bounds pin the measured split with headroom.
        zone = self.zones["forest"]
        kept = {n: float(m[zone].sum() / max(1, int(zone.sum())))
                for n, m in self.masks.items()}
        self.assertLess(kept["mcc"], 0.05)
        self.assertLess(kept["atin"], 0.05)
        self.assertLess(kept["mcc"], kept["pmf"])
        self.assertLess(kept["mcc"], kept["csf"])
        self.assertLess(kept["mcc"], kept["smrf"])
        self.assertLess(kept["csf"], 0.15)
        self.assertLess(kept["smrf"], 0.10)
        self.assertLess(kept["pmf"], 0.15)

    def test_total_error_bounded(self) -> None:
        for name, mask in self.masks.items():
            with self.subTest(method=name):
                err = confusion(mask, self.truth)
                self.assertLess(err["total"], 0.12, (name, err))


if __name__ == "__main__":
    unittest.main()
