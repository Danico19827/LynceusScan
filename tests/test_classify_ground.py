# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Tests for the Classify Ground strategy family (U-4: PMF/CSF/SMRF).

Mask-level invariants on synthetic clouds (no laspy), family discovery
(base inert, one instance per method), effective capabilities per
strategy, and one end-to-end PMF tile. Stdlib + engine modules only
(no Qt, no real runs).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from lynceus.nodes.ports import PortType
from lynceus.plugins.registry import manager
from lynceus.processing.raster import (
    cloth_simulation_filter,
    progressive_morphological_filter,
    simple_morphological_filter,
)

FAMILY = "lynceus.nodes.lidar.terrain.classify_ground"
PC = PortType.POINT_CLOUD


def _flat(n: int = 20000, seed: int = 2):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 40, n)
    y = rng.uniform(0, 40, n)
    z = np.full(n, 400.0) + rng.uniform(-0.2, 0.2, n)
    return x, y, z


def _with_box(n: int = 20000, seed: int = 4):
    x, y, z = _flat(n, seed)
    z[:30] = 460.0
    return x, y, z


MASKS = {
    "pmf": lambda x, y, z: progressive_morphological_filter(
        x, y, z, cell_size=1.0, slope=1.0, intercept=0.5,
        initial_window_m=2.0, max_window_m=8.0,
    ),
    "csf": lambda x, y, z: cloth_simulation_filter(
        x, y, z, cell_size=1.0, class_threshold=0.5,
        rigidness=3, iterations=100,
    ),
    "smrf": lambda x, y, z: simple_morphological_filter(
        x, y, z, cell_size=1.0, slope=0.2, max_window_m=18.0,
        threshold=0.4,
    ),
}


class MaskInvariantTests(unittest.TestCase):
    def test_flat_plane_all_ground(self) -> None:
        x, y, z = _flat()
        for name, mask in MASKS.items():
            with self.subTest(method=name):
                self.assertTrue(bool(mask(x, y, z).all()))

    def test_box_not_ground(self) -> None:
        x, y, z = _with_box()
        for name, mask in MASKS.items():
            with self.subTest(method=name):
                ground = mask(x, y, z)
                self.assertGreater(int(ground.sum()), len(z) // 2)
                self.assertFalse(bool(ground[:30].all()))

    def test_empty_cloud(self) -> None:
        empty = np.zeros(0)
        for name, mask in MASKS.items():
            with self.subTest(method=name):
                self.assertEqual(mask(empty, empty, empty).size, 0)

    def test_deterministic(self) -> None:
        x, y, z = _with_box()
        for name, mask in MASKS.items():
            with self.subTest(method=name):
                np.testing.assert_array_equal(mask(x, y, z), mask(x, y, z))

    def test_gentle_slope_kept(self) -> None:
        rng = np.random.default_rng(7)
        n = 20000
        x = rng.uniform(0, 40, n)
        y = rng.uniform(0, 40, n)
        z = 400.0 + 0.05 * x + rng.uniform(-0.1, 0.1, n)
        for name in ("pmf", "csf"):
            with self.subTest(method=name):
                self.assertGreater(int(MASKS[name](x, y, z).sum()), int(n * 0.95))

    def test_mild_relief_kept_by_smrf(self) -> None:
        # SMRF is strict on relief by design (fixed threshold at every
        # scale): its scope is flat-to-mild terrain, slopes go to PMF/CSF.
        rng = np.random.default_rng(7)
        n = 20000
        x = rng.uniform(0, 40, n)
        y = rng.uniform(0, 40, n)
        z = 400.0 + 0.02 * x + rng.uniform(-0.1, 0.1, n)
        self.assertGreater(int(MASKS["smrf"](x, y, z).sum()), int(n * 0.95))


class FamilyDiscoveryTests(unittest.TestCase):
    def test_three_method_variants(self) -> None:
        manager.discover()
        keys = [v["key"] for v in manager.list_variants(FAMILY)]
        self.assertEqual(keys, ["csf", "pmf", "smrf"])

    def test_base_inert_without_strategy(self) -> None:
        manager.discover()
        info = manager.node_info(FAMILY)
        self.assertIsNotNone(info)
        inputs, outputs = manager.load_ports(FAMILY)
        self.assertEqual(inputs, ())
        self.assertEqual(outputs, ())

    def test_effective_caps_per_method(self) -> None:
        from lynceus.processing.steps import effective_node_caps

        for key, fn in (
            ("pmf", "tile_classify_ground_pmf"),
            ("csf", "tile_classify_ground_csf"),
            ("smrf", "tile_classify_ground_smrf"),
        ):
            caps = effective_node_caps(FAMILY, {"strategy": key})
            self.assertEqual(caps.get("tile_task").__name__, fn)
            self.assertEqual(caps.get("point_cloud_dir"), "classified")
            self.assertEqual(caps.get("provides"), {"classified": True})

    def test_resolved_ports_identical_across_methods(self) -> None:
        from lynceus.nodes._variants import resolved_ports

        for key in ("pmf", "csf", "smrf"):
            self.assertEqual(
                resolved_ports(FAMILY, {"strategy": key}), ((PC,), (PC,))
            )


class PmfTileTests(unittest.TestCase):
    @staticmethod
    def _write_cloud(root: Path, xs, ys, zs) -> Path:
        import laspy

        n = len(xs)
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.scales = np.array([0.01, 0.01, 0.01])
        header.offsets = np.array([0.0, 0.0, 0.0])
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

    def test_pmf_tile_marks_flat_plane(self) -> None:
        import laspy

        from lynceus.nodes.lidar.terrain.classify_ground_pmf import (
            tile_classify_ground_pmf,
        )

        x, y, z = _flat(n=3000, seed=11)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = self._write_cloud(root, x, y, z)
            ctx = {"session_dir": str(root / "s")}
            tile = {"tile_id": "t", "file": str(src)}
            out = tile_classify_ground_pmf(tile, ctx)
            with laspy.open(out["output"]) as reader:
                cls = np.concatenate([
                    np.asarray(ch.classification)
                    for ch in reader.chunk_iterator(2_000_000)
                ])
            self.assertEqual(len(cls), 3000)
            self.assertTrue(bool((cls == 2).all()))


if __name__ == "__main__":
    unittest.main()
