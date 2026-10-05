# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Tests for branch isolation (F4.1 detection, F4.2a scoping).

Two instances of the same node over overlapping tiles used to share every
output path: concurrent workers corrupted the files and sequential runs
cross-contaminated. F4.1 failed fast with a clear error; F4.2a scopes
those instances apart instead (``tile_scope`` per task, instance dirs on
disk), so both run and keep separate products. Multi-source aggregation,
chained instances and per-instance writers stay on the shared layout.
Stdlib + engine modules only (no Qt, no real runs).
"""

from __future__ import annotations

import unittest

from lynceus.nodes.ports import PortType
from lynceus.processing.steps import build_dag

LOAD = "lynceus.nodes.lidar.source.load_las_laz"
CLASSIFY = "lynceus.nodes.lidar.terrain.classify_ground"
DTM = "lynceus.nodes.lidar.terrain.generate_dtm"
CONS = "lynceus.nodes.flow.consolidate"
INPUT_RASTER = "lynceus.nodes.raster.input_raster"
EXPORT = "lynceus.nodes.lidar.output.export_point_cloud"
PC = PortType.POINT_CLOUD


def _tiles(*ids: str, src: str = "loader") -> list[dict]:
    return [
        {"tile_id": tile_id, "file": "x.laz", "src_iid": src}
        for tile_id in ids
    ]


def _task_ctxs(tasks) -> dict[str, dict]:
    """Map task_id -> task context (last arg of exec args)."""
    return {t.task_id: t.args[-1] for t in tasks}


class TileCollisionTests(unittest.TestCase):
    def test_same_module_same_tile_scopes(self) -> None:
        tiles = _tiles("tag_c0000_r0000")
        tasks = build_dag(
            ["c1", "c2"],
            [],
            tiles,
            {"session_dir": "unused"},
            modules={"c1": CLASSIFY, "c2": CLASSIFY},
            configs={
                "c1": {"strategy": "pmf"},
                "c2": {"strategy": "pmf"},
            },
        )
        ctxs = _task_ctxs(tasks)
        # Both instances run, each in its own scope.
        self.assertEqual(
            ctxs["c1|tag_c0000_r0000"]["tile_scope"], "c1"
        )
        self.assertEqual(
            ctxs["c2|tag_c0000_r0000"]["tile_scope"], "c2"
        )

    def test_same_module_mosaics_same_domain_scope(self) -> None:
        tiles = _tiles("tag_c0000_r0000")
        tasks = build_dag(
            ["d1", "d2"],
            [],
            tiles,
            {"session_dir": "unused"},
            modules={"d1": DTM, "d2": DTM},
            configs={},
        )
        ctxs = _task_ctxs(tasks)
        self.assertEqual(ctxs["d1|merge"]["tile_scope"], "d1")
        self.assertEqual(ctxs["d2|merge"]["tile_scope"], "d2")


class LegalShapesTests(unittest.TestCase):
    def test_chained_same_module_barriers_ok(self) -> None:
        tasks = build_dag(
            ["c1", "c2"],
            [("c1", "c2", PortType.DTM_MOSAIC, PortType.DTM_MOSAIC)],
            [],
            {"session_dir": "unused"},
            modules={"c1": CONS, "c2": CONS},
            configs={
                "c1": {"strategy": "dtm"},
                "c2": {"strategy": "dtm"},
            },
        )
        self.assertTrue(
            any(t.task_id == "c2|merge" for t in tasks)
        )

    def test_multi_source_disjoint_tiles_ok(self) -> None:
        tasks = build_dag(
            ["loader1", "loader2", "d1", "d2"],
            [
                ("loader1", "d1", PC, PC),
                ("loader2", "d2", PC, PC),
            ],
            _tiles("tag1_c0000_r0000", src="loader1")
            + _tiles("tag2_c0000_r0000", src="loader2"),
            {"session_dir": "unused"},
            modules={
                "loader1": LOAD,
                "loader2": LOAD,
                "d1": DTM,
                "d2": DTM,
            },
            configs={},
        )
        ids = sorted(t.task_id for t in tasks)
        self.assertIn("d1|tag1_c0000_r0000", ids)
        self.assertIn("d2|tag2_c0000_r0000", ids)

    def test_different_products_same_module_ok(self) -> None:
        tasks = build_dag(
            ["c1", "c2"],
            [],
            [],
            {"session_dir": "unused"},
            modules={"c1": CONS, "c2": CONS},
            configs={
                "c1": {"strategy": "dtm"},
                "c2": {"strategy": "chm"},
            },
        )
        self.assertEqual(
            sorted(t.task_id for t in tasks), ["c1|merge", "c2|merge"]
        )

    def test_product_inputs_never_collide(self) -> None:
        tasks = build_dag(
            ["a", "b"],
            [],
            [],
            {"session_dir": "unused"},
            modules={"a": INPUT_RASTER, "b": INPUT_RASTER},
            configs={
                "a": {"strategy": "chm"},
                "b": {"strategy": "chm"},
            },
        )
        self.assertEqual(
            sorted(t.task_id for t in tasks), ["a|merge", "b|merge"]
        )

    def test_exports_never_collide(self) -> None:
        tiles = _tiles("tag_c0000_r0000")
        tasks = build_dag(
            ["c", "e1", "e2"],
            [
                ("c", "e1", PC, PC),
                ("c", "e2", PC, PC),
            ],
            tiles,
            {"session_dir": "unused"},
            modules={
                "c": CLASSIFY,
                "e1": EXPORT,
                "e2": EXPORT,
            },
            configs={"c": {"strategy": "pmf"}},
        )
        self.assertIn("e1|merge", [t.task_id for t in tasks])
        self.assertIn("e2|merge", [t.task_id for t in tasks])


if __name__ == "__main__":
    unittest.main()
