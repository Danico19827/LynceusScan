# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""ROR semantics and precision in Classify Noise.

The radius test must be exactly equivalent to counting neighbors inside
the radius, but computed as an early-exit kNN threshold (radius counting
explodes at modern densities). Coordinates are rebased to the local
centroid before the float32 cast so sub-meter spacing survives at UTM
magnitudes. Stdlib + engine/domain modules only (no Qt).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import laspy
import numpy as np
from scipy.spatial import cKDTree

from lynceus.nodes.lidar.clean.class_noise import tile_class_noise

FLAGGED = 7
KEPT = 1


def _write_cloud(path: Path, points: np.ndarray) -> Path:
    header = laspy.LasHeader(point_format=1, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = [
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 2].min()),
    ]
    record = laspy.ScaleAwarePointRecord.zeros(len(points), header=header)
    record.x = points[:, 0]
    record.y = points[:, 1]
    record.z = points[:, 2]
    record.classification[:] = KEPT
    with laspy.open(str(path), mode="w", header=header) as writer:
        writer.write_points(record)
    return path


def _load_points(path: Path) -> np.ndarray:
    with laspy.open(str(path)) as reader:
        cloud = reader.read()
    return np.column_stack(
        (
            np.asarray(cloud.x, dtype=np.float64),
            np.asarray(cloud.y, dtype=np.float64),
            np.asarray(cloud.z, dtype=np.float64),
        )
    )


def _flagged_mask(session: Path, src: Path, **config) -> np.ndarray:
    tile = {"tile_id": "t_c0000_r0000", "file": str(src)}
    ctx = {
        "session_dir": str(session),
        "intermediate_ext": "laz",
        "method": "ror",
        "action": "classify",
        "noise_class": FLAGGED,
        "protect_ground": True,
        **config,
    }
    result = tile_class_noise(tile, ctx)
    with laspy.open(result["output"]) as reader:
        classes = np.asarray(reader.read().classification)
    return classes == FLAGGED


def _reference_flagged(points: np.ndarray, radius: float, min_k: int) -> np.ndarray:
    """Counting semantics in the same rebased float32 frame the node uses."""
    n = len(points)
    if min_k <= 1:
        return np.zeros(n, dtype=bool)
    if min_k > n:
        return np.ones(n, dtype=bool)
    rebased = (points - points.mean(axis=0)).astype(np.float32)
    counts = cKDTree(rebased).query_ball_point(
        rebased, radius, return_length=True
    )
    return counts < min_k


class RorEquivalenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.rng = np.random.default_rng(11)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _check(self, points: np.ndarray, radius: float, min_k: int) -> None:
        tag = f"{len(points)}_{radius}_{min_k}"
        src = _write_cloud(self.root / f"pt_{tag}.laz", points)
        flagged = _flagged_mask(
            self.root / f"s_{tag}",
            src,
            ror_radius_m=radius,
            ror_min_points=min_k,
        )
        # Reference on the LAS-quantized coordinates the node actually reads
        # (1 mm scales), so the only variable under test is kNN-vs-counting.
        expected = _reference_flagged(_load_points(src), radius, min_k)
        self.assertTrue(
            np.array_equal(flagged, expected),
            msg=f"radius={radius} min_k={min_k}: "
            f"{int(flagged.sum())} vs {int(expected.sum())} flagged",
        )

    def test_matches_radius_count_reference(self) -> None:
        for count in (500, 5_000):
            points = np.column_stack(
                [
                    self.rng.uniform(0, 30, count) + 4_346_018.0,
                    self.rng.uniform(0, 30, count) + 7_391_246.0,
                    self.rng.uniform(0, 4, count) + 100.0,
                ]
            )
            for radius, min_k in ((2.0, 3), (5.0, 4), (0.5, 2), (10.0, 6)):
                with self.subTest(count=count, radius=radius, min_k=min_k):
                    self._check(points, radius, min_k)

    def test_sparse_outliers_flagged(self) -> None:
        dense = self.rng.uniform(0, 10, (2_000, 3))
        outliers = np.full((7, 3), -1.0)
        # 12 m apart: beyond 2x the 5 m radius, so every outlier has only
        # itself in range and must be flagged.
        outliers[:, 0] = np.linspace(30, 102, 7)
        points = np.vstack([dense, outliers])
        src = _write_cloud(self.root / "sparse.laz", points)
        flagged = _flagged_mask(
            self.root / "s_sparse", src, ror_radius_m=5.0, ror_min_points=3
        )
        self.assertEqual(int(flagged.sum()), 7)
        self.assertTrue(np.all(flagged[-7:]))

    def test_min_k_one_keeps_everything(self) -> None:
        points = self.rng.uniform(0, 10, (300, 3))
        src = _write_cloud(self.root / "k1.laz", points)
        flagged = _flagged_mask(
            self.root / "s_k1", src, ror_radius_m=0.1, ror_min_points=1
        )
        self.assertFalse(flagged.any())

    def test_min_k_above_point_count_flags_all(self) -> None:
        points = self.rng.uniform(0, 10, (20, 3))
        src = _write_cloud(self.root / "kbig.laz", points)
        flagged = _flagged_mask(
            self.root / "s_kbig", src, ror_radius_m=1.0, ror_min_points=50
        )
        self.assertTrue(flagged.all())

    def test_single_point_is_kept(self) -> None:
        src = _write_cloud(self.root / "one.laz", np.zeros((1, 3)))
        flagged = _flagged_mask(
            self.root / "s_one", src, ror_radius_m=5.0, ror_min_points=3
        )
        self.assertFalse(flagged.any())

    def test_duplicates_count_toward_neighborhood(self) -> None:
        points = np.repeat(self.rng.uniform(0, 5, (50, 3)), 5, axis=0)
        points = np.vstack([points, [[50.0, 50.0, 0.0]]])
        src = _write_cloud(self.root / "dups.laz", points)
        flagged = _flagged_mask(
            self.root / "s_dups", src, ror_radius_m=1.0, ror_min_points=4
        )
        self.assertEqual(int(flagged.sum()), 1)
        self.assertTrue(flagged[-1])

    def test_sub_ulp_spacing_uses_rebased_precision(self) -> None:
        # 5 cm apart at 4.3M/7.4M easting-northing: raw float32 quantizes
        # to ~0.5 m and would collapse them; rebased, they keep their real
        # separation. Radius 4 cm must flag both (no neighbor in range).
        points = np.array(
            [
                [4_346_018.010, 7_391_246.000, 100.0],
                [4_346_018.060, 7_391_246.000, 100.0],
            ]
        )
        src = _write_cloud(self.root / "fine.laz", points)
        flagged = _flagged_mask(
            self.root / "s_fine", src, ror_radius_m=0.04, ror_min_points=2
        )
        self.assertTrue(flagged.all())
        # Control: 3 cm apart does fit inside 4 cm -> kept.
        points[1, 0] = 4_346_018.040
        src = _write_cloud(self.root / "fine2.laz", points)
        flagged = _flagged_mask(
            self.root / "s_fine2", src, ror_radius_m=0.04, ror_min_points=2
        )
        self.assertFalse(flagged.any())


if __name__ == "__main__":
    unittest.main()
