# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Tests for the protect_ground opt-out in run_clean.

Ground (class 2) is never retagged by default; with protect_ground=False
the operator explicitly allows retagging mislabeled ground (e.g. vendor
noise classified as ground at altitude) so a later remove can delete it.
The remove path never had the protection and stays untouched. Stdlib +
engine/domain modules only (no Qt).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import laspy
import numpy as np

from lynceus.nodes._point_source import GROUND_CLASS, run_clean

MODULE = "test.protect_ground"


def _write_cloud(path: Path) -> Path:
    """200 pts: 100 ground out-of-range (z=-5), 50 veg in-range (z=10),
    50 noise out-of-range (z=-5)."""
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.01, 0.01, 0.01])
    header.offsets = np.zeros(3)
    n = 200
    record = laspy.ScaleAwarePointRecord.zeros(
        n,
        point_format=header.point_format,
        scales=header.scales,
        offsets=header.offsets,
    )
    record.x = np.arange(n, dtype=float)
    record.y = np.zeros(n)
    record.z = np.concatenate([
        np.full(100, -5.0), np.full(50, 10.0), np.full(50, -5.0),
    ])
    record.classification = np.concatenate([
        np.full(100, 2, dtype=np.uint8),
        np.full(50, 5, dtype=np.uint8),
        np.full(50, 1, dtype=np.uint8),
    ])
    with laspy.open(str(path), mode="w", header=header) as writer:
        writer.write_points(record)
    return path


def _run(session: Path, src: Path, name: str, **config) -> dict:
    tile = {"tile_id": "t_c0000_r0000", "file": str(src)}
    protect = bool(config.get("protect_ground", True))
    ctx = {"session_dir": str(session), "intermediate_ext": "laz", **config}
    mask = lambda record: np.asarray(record.z) >= 0.0
    return run_clean(tile, ctx, MODULE, name, mask,
                     protect_ground=protect)


def _classes(path: str) -> dict[int, int]:
    from lynceus.processing.tiler import _laz_backend

    with laspy.open(path, laz_backend=_laz_backend()) as reader:
        hist = np.zeros(256, dtype=np.int64)
        for chunk in reader.chunk_iterator(2_000_000):
            hist += np.bincount(
                np.asarray(chunk.classification), minlength=256
            )
    return {int(k): int(v) for k, v in enumerate(hist) if v}


class ProtectGroundTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.src = _write_cloud(self.root / "src.laz")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_default_protects_ground(self) -> None:
        res = _run(self.root / "s1", self.src, "clean_a")
        self.assertEqual(res["tagged_noise"], 50)
        self.assertEqual(res["ground_retagged"], 0)
        self.assertNotIn("warnings", res)
        self.assertEqual(
            _classes(res["output"]),
            {2: 100, 5: 50, 7: 50},
        )

    def test_opt_out_retags_ground(self) -> None:
        res = _run(
            self.root / "s2", self.src, "clean_b", protect_ground=False
        )
        self.assertEqual(res["tagged_noise"], 150)
        self.assertEqual(res["ground_retagged"], 100)
        self.assertTrue(
            any("ground" in w for w in res.get("warnings", []))
        )
        self.assertEqual(
            _classes(res["output"]),
            {5: 50, 7: 150},
        )

    def test_remove_ignores_the_flag(self) -> None:
        for protect in (True, False):
            res = _run(
                self.root / f"s3_{protect}", self.src, "clean_c",
                action="remove", protect_ground=protect,
            )
            self.assertEqual(res["removed"], 150)
            self.assertEqual(_classes(res["output"]), {5: 50})

    def test_target_ground_still_forbidden(self) -> None:
        tile = {"tile_id": "t_c0000_r0000", "file": str(self.src)}
        ctx = {"session_dir": str(self.root / "s4"),
               "intermediate_ext": "laz", "protect_ground": False}
        mask = lambda record: np.asarray(record.z) >= 0.0
        with self.assertRaises(ValueError):
            run_clean(tile, ctx, MODULE, "clean_d", mask, cls_value=2)

    def test_ground_class_constant(self) -> None:
        self.assertEqual(GROUND_CLASS, 2)


class ElevationWiringTests(unittest.TestCase):
    """The flag travels from node config (ctx) into run_clean."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.src = _write_cloud(self.root / "src.laz")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _run_elevation(self, session: Path, **config) -> dict:
        from lynceus.nodes.lidar.clean.elevation_range import (
            tile_elevation_range,
        )

        tile = {"tile_id": "t_c0000_r0000", "file": str(self.src)}
        ctx = {
            "session_dir": str(session),
            "intermediate_ext": "laz",
            "min_elevation": 0.0,
            "max_elevation": 100000.0,
            **config,
        }
        return tile_elevation_range(tile, ctx)

    def test_default_keeps_ground_below_min(self) -> None:
        res = self._run_elevation(self.root / "e1")
        self.assertEqual(res["ground_retagged"], 0)
        self.assertEqual(
            _classes(res["output"]),
            {2: 100, 5: 50, 7: 50},
        )

    def test_opt_out_retags_ground_below_min(self) -> None:
        res = self._run_elevation(
            self.root / "e2", protect_ground=False
        )
        self.assertEqual(res["ground_retagged"], 100)
        self.assertEqual(
            _classes(res["output"]),
            {5: 50, 7: 150},
        )


class ProtectGroundSchemaTests(unittest.TestCase):
    """The four classify-capable clean nodes expose the opt-in (on by
    default); the always-remove finishers never limited ground."""

    def test_schema_default_on(self) -> None:
        from lynceus.nodes.lidar.clean import (
            class_noise,
            elevation_range,
            intensity_range,
            scan_angle,
        )

        for module in (
            elevation_range, intensity_range, scan_angle, class_noise,
        ):
            with self.subTest(module=module.NODE_ID):
                spec = module.PROCESSING_SPECS["config_schema"][
                    "protect_ground"
                ]
                self.assertIs(spec["type"], "bool")
                self.assertTrue(spec["default"])
                self.assertEqual(spec["group"], "Behavior")


if __name__ == "__main__":
    unittest.main()
