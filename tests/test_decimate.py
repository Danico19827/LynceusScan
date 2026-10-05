"""Unit tests for the Decimate node (density reduction).

Exercises the tile-task directly over synthetic LAZ: reaches are kept in an
exact, deterministic way (nth vs seeded random) and the pass-through cases
(ratio >= 100, adaptive with the tile already under target) lose nothing.
"""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import laspy
import numpy as np

from lynceus.nodes.lidar.clean.decimate import _keep_mask, tile_decimate
from lynceus.nodes.lidar.clean.decimate import NODE_ID


def _write_cloud(path: Path, count: int) -> Path:
    rng = np.random.default_rng(11)
    header = laspy.LasHeader(point_format=1, version="1.2")
    header.scales = np.array([0.01, 0.01, 0.01])
    header.offsets = np.zeros(3)
    record = laspy.ScaleAwarePointRecord.zeros(
        count,
        point_format=header.point_format,
        scales=header.scales,
        offsets=header.offsets,
    )
    record.x = rng.uniform(0, 200, count)
    record.y = rng.uniform(0, 200, count)
    record.z = np.zeros(count)
    record.classification = rng.integers(0, 25, count).astype(np.uint8)
    with laspy.open(str(path), mode="w", header=header) as writer:
        writer.write_points(record)
    return path


def _run_decimate(session: Path, src: Path, config: dict, tile_id: str = "dst_c0000_r0000") -> dict:
    tile = {"tile_id": tile_id, "file": str(src)}
    ctx = {"session_dir": str(session), "intermediate_ext": "laz", **config}
    return tile_decimate(tile, ctx)


def _count(path: str) -> int:
    with laspy.open(path, laz_backend="lazrs") as reader:
        return int(reader.header.point_count)


class DecimateNodeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.src = _write_cloud(self.root / "src.laz", 4_000)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_ratio_keeps_one_in_step(self) -> None:
        out = _run_decimate(self.root / "s1", self.src, {"mode": "ratio", "sampling_ratio": 25.0, "strategy": "nth"})
        self.assertEqual(_count(out["output"]), 1_000)  # 4000 / 4

    def test_ratio_full_is_passthrough_without_loss(self) -> None:
        out = _run_decimate(self.root / "s1", self.src, {"mode": "ratio", "sampling_ratio": 100.0})
        self.assertEqual(_count(out["output"]), 4_000)
        self.assertEqual(out.get("removed"), 0)

    def test_adaptive_under_target_is_passthrough(self) -> None:
        out = _run_decimate(self.root / "s1", self.src, {"mode": "adaptive", "target_points": 50_000})
        self.assertEqual(_count(out["output"]), 4_000)
        self.assertEqual(out.get("removed"), 0)

    def test_adaptive_thins_just_enough(self) -> None:
        out = _run_decimate(self.root / "s1", self.src, {"mode": "adaptive", "target_points": 1_000})
        self.assertEqual(_count(out["output"]), 1_000)

    def test_nth_and_random_keep_same_count_but_differ(self) -> None:
        a = _run_decimate(self.root / "a", self.src, {"strategy": "nth", "sampling_ratio": 25.0})
        b = _run_decimate(self.root / "b", self.src, {"strategy": "random", "sampling_ratio": 25.0})
        self.assertEqual(_count(a["output"]), _count(b["output"]))
        xs = np.array([r["X"] for r in laspy.read(a["output"]).points])
        ys = np.array([r["X"] for r in laspy.read(b["output"]).points])
        self.assertFalse(np.array_equal(np.sort(xs), np.sort(ys)))

    def test_output_is_deterministic_across_runs(self) -> None:
        a = _run_decimate(self.root / "a", self.src, {"strategy": "random", "sampling_ratio": 25.0})
        b = _run_decimate(self.root / "b", self.src, {"strategy": "random", "sampling_ratio": 25.0})
        ha = hashlib.sha1(Path(a["output"]).read_bytes()).hexdigest()
        hb = hashlib.sha1(Path(b["output"]).read_bytes()).hexdigest()
        self.assertEqual(ha, hb)

    def test_keep_mask_bounds(self) -> None:
        header = laspy.LasHeader(point_format=1, version="1.2")
        rec = laspy.ScaleAwarePointRecord.zeros(
            5, point_format=header.point_format, scales=header.scales, offsets=header.offsets
        )
        mask = _keep_mask(rec, {"mode": "ratio", "sampling_ratio": 100.0})
        self.assertEqual(mask.tolist(), [True, True, True, True, True])
        mask2 = _keep_mask(rec, {})  # defaults: ratio 25 -> step 4 -> ceil(5/4)=2
        mask3 = _keep_mask(rec, {"strategy": "random"})
        self.assertEqual(int(mask2.sum()), 2)
        self.assertEqual(int(mask3.sum()), 2)
        self.assertEqual(mask2.tolist(), [True, False, False, False, True])

    def test_node_is_registered_and_specs(self) -> None:
        from lynceus.plugins.registry import manager
        from lynceus.processing.steps import discover_node_capabilities

        manager.discover()
        info = manager.node_info(NODE_ID)
        self.assertIsNotNone(info)
        caps = discover_node_capabilities(NODE_ID)
        self.assertEqual(caps["tile_task"].__name__, "tile_decimate")
        self.assertEqual(caps["point_cloud_dir"], "decimated")
        self.assertIn("classified", caps["passes_flags"])


if __name__ == "__main__":
    unittest.main()