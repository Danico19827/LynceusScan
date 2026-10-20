# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Unit tests for sparse supertile planning (P0: pure algorithm, no tiler).

The merge contract: every nonempty base lands in exactly one final block;
blocks are full aligned span x span squares; merged blocks obey the floor;
subdivided bases (and their ancestors) never merge; traversal is sorted so
the map is deterministic across machines.
"""

from __future__ import annotations

import unittest

from lynceus.processing.sparse_merge import MAX_MERGE_SPAN, plan_sparse_merge


def _grid(counts: list[list[int]]) -> dict[int, int]:
    """Row-major base counts from a row list (nonzero entries only)."""
    out = {}
    for r, row in enumerate(counts):
        for c, n in enumerate(row):
            if n:
                out[r * len(row) + c] = n
    return out


class AnalyticFixtureTests(unittest.TestCase):
    """Hand-computed 4x4 grid (floor 10): exact expected map."""

    COUNTS = _grid([
        [3, 3, 1, 1],
        [3, 3, 1, 1],
        [8, 8, 2, 2],
        [8, 8, 2, 2],
    ])

    def test_exact_map(self) -> None:
        super_of, blocks = plan_sparse_merge(
            self.COUNTS, cols=4, rows=4, floor=10)
        # TL (12 pts) and BL (32 pts) stay single; TR (4) and BR (8) merge.
        self.assertEqual(blocks, {
            2: {"span": 2, "members": [2, 3, 6, 7], "points": 4},
            10: {"span": 2, "members": [10, 11, 14, 15], "points": 8},
        })
        self.assertEqual(super_of, {
            0: 0, 1: 1, 2: 2, 3: 2, 4: 4, 5: 5, 6: 2, 7: 2,
            8: 8, 9: 9, 10: 10, 11: 10, 12: 12, 13: 13, 14: 10,
            15: 10,
        })


class PartitionInvariantTests(unittest.TestCase):
    def _check_partition(self, counts, cols, rows, floor, subdivided=frozenset()):
        super_of, blocks = plan_sparse_merge(
            counts, set(subdivided), cols=cols, rows=rows, floor=floor)
        # Every nonempty base lands in exactly one final block.
        self.assertEqual(sorted(super_of), sorted(counts))
        # Block members are disjoint and cover their claimed points.
        seen: list[int] = []
        for origin, block in blocks.items():
            for base in block["members"]:
                self.assertIn(base, counts)
                self.assertEqual(super_of[base], origin)
            seen.extend(block["members"])
        self.assertEqual(len(set(seen)), len(seen))
        # Rectangular aligned squares holding exactly their members' points.
        for origin, block in blocks.items():
            span = block["span"]
            ocol, orow = origin % cols, origin // cols
            square = {
                (orow + dr) * cols + (ocol + dc)
                for dr in range(span) for dc in range(span)
            }
            self.assertLessEqual(set(block["members"]), square)
            self.assertEqual(
                block["points"],
                sum(counts[b] for b in block["members"]),
            )
            self.assertLessEqual(block["points"], floor)
        return super_of, blocks

    def test_random_layouts_partition(self) -> None:
        import random

        rng = random.Random(20261012)
        for trial in range(25):
            cols = rng.randint(1, 9)
            rows = rng.randint(1, 9)
            counts = {}
            for base in range(cols * rows):
                if rng.random() < 0.45:
                    counts[base] = rng.randint(1, 30)
            floor = rng.choice([1, 5, 10, 50, 200])
            subdivided = frozenset(
                b for b in counts if rng.random() < 0.1)
            self._check_partition(counts, cols, rows, floor, subdivided)

    def test_oversized_singles_stay_single(self) -> None:
        counts = {0: 500, 1: 1, 4: 1, 5: 1}
        super_of, blocks = plan_sparse_merge(
            counts, cols=4, rows=4, floor=10)
        # Base 0 alone exceeds the floor: it (and its aligned siblings,
        # whose 2x2 also exceeds it) stay single.
        self.assertEqual(super_of[0], 0)
        self.assertEqual(blocks, {})

    def test_span_cap_binds(self) -> None:
        counts = {b: 1 for b in range(16 * 16)}
        super_of, blocks = plan_sparse_merge(
            counts, cols=16, rows=16, floor=10 ** 9, max_span=4)
        for block in blocks.values():
            self.assertLessEqual(block["span"], 4)
        # Full 16x16 of ones under a huge floor still tiles the plane.
        self._check_partition(counts, 16, 16, 10 ** 9)

    def test_locked_quadrants_never_merge(self) -> None:
        # A subdivided center poisons every aligned parent touching it: the
        # sparse ring around it must not bridge across (L-shapes forbidden).
        counts = {b: 1 for b in range(16)}
        super_of, blocks = plan_sparse_merge(
            counts, {5, 6, 9, 10}, cols=4, rows=4, floor=100)
        # Every 2x2 quadrant holds a locked cell: nothing merges anywhere.
        self.assertEqual(blocks, {})
        self.assertEqual(super_of, {b: b for b in range(16)})

    def test_partial_lock_merges_around_it(self) -> None:
        # One locked cell (5): its own quadrant stays single, the other
        # three sparse quadrants still merge.
        counts = {b: 1 for b in range(16)}
        super_of, blocks = plan_sparse_merge(
            counts, {5}, cols=4, rows=4, floor=100)
        self.assertEqual(blocks, {
            2: {"span": 2, "members": [2, 3, 6, 7], "points": 4},
            8: {"span": 2, "members": [8, 9, 12, 13], "points": 4},
            10: {"span": 2, "members": [10, 11, 14, 15], "points": 4},
        })
        self.assertEqual(super_of, {
            0: 0, 1: 1, 2: 2, 3: 2, 4: 4, 5: 5, 6: 2, 7: 2,
            8: 8, 9: 8, 10: 10, 11: 10, 12: 8, 13: 8, 14: 10,
            15: 10,
        })


class DeterminismTests(unittest.TestCase):
    def test_insertion_order_irrelevant(self) -> None:
        import random

        rng = random.Random(7)
        items = [(b, rng.randint(1, 50)) for b in range(200)]
        forward = dict(items)
        backward = dict(reversed(items))
        cols = rows = 15
        first = plan_sparse_merge(
            forward, cols=cols, rows=rows, floor=500)
        second = plan_sparse_merge(
            backward, cols=cols, rows=rows, floor=500)
        self.assertEqual(first, second)

    def test_identity_cases(self) -> None:
        counts = {0: 5, 1: 5, 4: 5, 5: 5}
        for kwargs in (dict(floor=0), dict(floor=10, max_span=1)):
            super_of, blocks = plan_sparse_merge(
                counts, cols=4, rows=4, **kwargs)
            self.assertEqual(super_of, {b: b for b in counts})
            self.assertEqual(blocks, {})
        self.assertEqual(
            plan_sparse_merge({}, cols=4, rows=4, floor=10), ({}, {}))

    def test_max_merge_span_constant(self) -> None:
        self.assertEqual(MAX_MERGE_SPAN, 8)


class MergeWiringTests(unittest.TestCase):
    """P1: the tiler routes merged buckets into super files end to end."""

    @staticmethod
    def _write_cloud(path, sparse: int = 60_000, dense: int = 40_000):
        import laspy
        import numpy as np

        rng = np.random.default_rng(5)
        header = laspy.LasHeader(point_format=0, version="1.2")
        header.scales = np.array([0.01, 0.01, 0.01])
        header.offsets = np.zeros(3)
        total = sparse + dense
        record = laspy.ScaleAwarePointRecord.zeros(
            total,
            point_format=header.point_format,
            scales=header.scales,
            offsets=header.offsets,
        )
        record.x = np.concatenate([
            rng.uniform(0, 2000, sparse), rng.uniform(900, 1100, dense)])
        record.y = np.concatenate([
            rng.uniform(0, 2000, sparse), rng.uniform(900, 1100, dense)])
        record.z = np.zeros(total)
        with laspy.open(str(path), mode="w", header=header) as writer:
            writer.write_points(record)
        return path

    def _tile(self, **kwargs):
        import tempfile
        from pathlib import Path

        from lynceus.processing.tiler import LiDARTiler

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        source = self._write_cloud(root / "src.laz")
        params = dict(tile_size=100.0, buffer_m=10.0, chunk_size=50_000)
        params.update(kwargs)
        result = LiDARTiler(source, root / "tiles", **params).run(
            cancel_flag=None)
        return result

    def test_merge_reduces_files(self) -> None:
        plain = self._tile()
        merged = self._tile(merge_sparse_below=20_000)
        self.assertGreater(len(plain.tiles_generated), 100)
        self.assertLess(len(merged.tiles_generated), 40)
        supers = [t for t in merged.tiles_generated if t.get("span", 0) > 1]
        self.assertGreater(len(supers), 0)
        self.assertEqual(
            merged.tiling_telemetry["merged_supertiles"], len(supers))
        for tile in merged.tiles_generated:
            self.assertIn("span", tile)

    def test_core_partition_exact(self) -> None:
        import numpy as np

        merged = self._tile(merge_sparse_below=20_000)
        cores = [
            (t["core_x_min"], t["core_y_min"], t["core_x_max"],
             t["core_y_max"])
            for t in merged.tiles_generated
        ]
        source = np.zeros((100_000, 2))
        rng = np.random.default_rng(5)
        source[:, 0] = np.concatenate([
            rng.uniform(0, 2000, 60_000), rng.uniform(900, 1100, 40_000)])
        source[:, 1] = np.concatenate([
            rng.uniform(0, 2000, 60_000), rng.uniform(900, 1100, 40_000)])
        hits = np.zeros(len(source), dtype=np.int64)
        for x0, y0, x1, y1 in cores:
            hits += (
                (source[:, 0] >= x0) & (source[:, 0] < x1)
                & (source[:, 1] >= y0) & (source[:, 1] < y1)
            )
        # Every source point lies in exactly one tile core (cores are a
        # half-open partition of the plane; buffer copies live outside).
        self.assertTrue(bool((hits == 1).all()))

    def test_subdivide_and_merge_stay_disjoint(self) -> None:
        merged = self._tile(
            merge_sparse_below=20_000, max_points_per_tile=5_000)
        subs = [t for t in merged.tiles_generated
                if t.get("sub_level", 0) > 0]
        supers = [t for t in merged.tiles_generated if t.get("span", 0) > 1]
        self.assertGreater(len(subs), 0)
        self.assertGreater(len(supers), 0)
        for tile in merged.tiles_generated:
            self.assertFalse(
                tile.get("sub_level", 0) > 0 and tile.get("span", 0) > 1)

    def test_manifest_freshness_keys(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        from lynceus.processing.tiler import LiDARTiler

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        source = self._write_cloud(root / "src.laz")
        out = root / "tiles"
        LiDARTiler(
            source, out, tile_size=100.0, buffer_m=10.0,
            chunk_size=50_000, merge_sparse_below=20_000).run(
            cancel_flag=None)
        manifest = json.loads(
            (out / "tiling_manifest.json").read_text())
        self.assertEqual(manifest.get("merge_sparse_below"), 20000)
        # Legacy manifests (no merge key) stay fresh under defaults...
        tiler = LiDARTiler(
            source, out, tile_size=100.0, buffer_m=10.0,
            chunk_size=50_000)
        manifest.pop("merge_sparse_below", None)
        (out / "tiling_manifest.json").write_text(json.dumps(manifest))
        self.assertTrue(tiler._cache_is_fresh())
        # ...and go stale once the floor is set.
        tiler_on = LiDARTiler(
            source, out, tile_size=100.0, buffer_m=10.0,
            chunk_size=50_000, merge_sparse_below=20_000)
        self.assertFalse(tiler_on._cache_is_fresh())

    def test_determinism_across_runs(self) -> None:
        import hashlib
        import tempfile
        from pathlib import Path

        from lynceus.processing.tiler import LiDARTiler

        def _hashes(result) -> dict:
            out = {}
            for tile in result.tiles_generated:
                data = Path(tile["file"]).read_bytes()
                out[tile["tile_id"]] = (
                    hashlib.sha1(data).hexdigest(), tile["point_count"])
            return out

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        source = self._write_cloud(root / "src.laz")
        kwargs = dict(tile_size=100.0, buffer_m=10.0, chunk_size=50_000,
                      merge_sparse_below=20_000)
        first = LiDARTiler(source, root / "a", **kwargs).run(
            cancel_flag=None)
        second = LiDARTiler(source, root / "b", **kwargs).run(
            cancel_flag=None)
        self.assertEqual(_hashes(first), _hashes(second))


class MergeFingerprintTests(unittest.TestCase):
    """P2: the merge floor travels in node fingerprints (reuse keys)."""

    LOAD = "lynceus.nodes.lidar.source.load_las_laz"
    DTM = "lynceus.nodes.lidar.terrain.generate_dtm"

    def _ctrl(self):
        from lynceus.processing.controller import PipelineController

        ctrl = PipelineController()
        ctrl._modules = {"l1": self.LOAD, "c1": self.DTM}
        return ctrl

    def _edges(self):
        return [("l1", "c1", "point_cloud", "point_cloud")]

    def test_floor_travels_in_fp(self) -> None:
        ctrl = self._ctrl()
        base = {"file_path": "/d/a.laz", "merge_sparse_below": 0}
        merged = {"file_path": "/d/a.laz", "merge_sparse_below": 500000}
        plain = ctrl._node_fingerprints(["l1"], [], {"l1": base}, {})
        fused = ctrl._node_fingerprints(["l1"], [], {"l1": merged}, {})
        self.assertNotEqual(plain["l1"], fused["l1"])

    def test_identical_config_stable_fp(self) -> None:
        ctrl = self._ctrl()
        cfg = {"file_path": "/d/a.laz", "merge_sparse_below": 500000}
        first = ctrl._node_fingerprints(
            ["l1"], [], {"l1": dict(cfg)}, {})
        second = ctrl._node_fingerprints(
            ["l1"], [], {"l1": dict(cfg)}, {})
        self.assertEqual(first["l1"], second["l1"])

    def test_downstream_inherits_floor_change(self) -> None:
        ctrl = self._ctrl()
        iids = ["l1", "c1"]
        plain = ctrl._node_fingerprints(
            iids, self._edges(),
            {"l1": {"file_path": "/d/a.laz", "merge_sparse_below": 0}}, {},
        )
        fused = ctrl._node_fingerprints(
            iids, self._edges(),
            {"l1": {"file_path": "/d/a.laz", "merge_sparse_below": 500000}},
            {},
        )
        self.assertNotEqual(plain["l1"], fused["l1"])
        self.assertNotEqual(plain["c1"], fused["c1"])


class DownstreamSeamTests(unittest.TestCase):
    """P2: downstream file resolution works on super layouts unchanged."""

    def test_point_src_and_clean_paths_on_supers(self) -> None:
        import tempfile
        from pathlib import Path

        import laspy

        from lynceus.nodes._point_source import (
            clean_out_path,
            point_src,
        )
        from lynceus.processing.tiler import LiDARTiler

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        source = MergeWiringTests._write_cloud(root / "src.laz")
        result = LiDARTiler(
            source, root / "tiles", tile_size=100.0, buffer_m=10.0,
            chunk_size=50_000, merge_sparse_below=20_000).run(
            cancel_flag=None)
        supers = [t for t in result.tiles_generated if t.get("span", 0) > 1]
        self.assertGreater(len(supers), 0)
        session = root / "session"
        session.mkdir()
        ctx = {"session_dir": str(session), "intermediate_ext": "laz"}
        for tile in result.tiles_generated:
            # Raw tile files resolve (fallback) and stay readable with
            # matching counts; clean outputs land under the super id.
            self.assertEqual(point_src(tile, ctx), tile["file"])
            with laspy.open(tile["file"]) as reader:
                self.assertEqual(
                    reader.header.point_count, tile["point_count"])
            out = clean_out_path(tile, ctx, "clean_elevation")
            self.assertTrue(out.endswith(f"{tile['tile_id']}.laz"))
        super_ids = {t["tile_id"] for t in supers}
        self.assertTrue(any("_x" in tid for tid in super_ids))


if __name__ == "__main__":
    unittest.main()
