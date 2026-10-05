# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Unit tests for pre-tiling run gates (no Qt, stdlib + engine only)."""

import tempfile
import unittest
from pathlib import Path

import laspy
import numpy as np

from lynceus.processing.controller import (
    PipelineCallbacks,
    PipelineController,
)

LOAD = "lynceus.nodes.lidar.source.load_las_laz"


def _tiny_cloud(path: Path, n: int = 500) -> None:
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = np.array([0.01, 0.01, 0.01])
    header.offsets = np.array([643000.0, 6851000.0, 0.0])
    record = laspy.ScaleAwarePointRecord.zeros(
        n, point_format=header.point_format,
        scales=header.scales, offsets=header.offsets,
    )
    record.x = np.linspace(643000.0, 643099.0, n)
    record.y = np.linspace(6851000.0, 6851099.0, n)
    record.z = np.linspace(100.0, 110.0, n)
    with laspy.open(str(path), mode="w", header=header) as writer:
        writer.write_points(record)


class MissingSourceGateTests(unittest.TestCase):
    def _run(self, nodes, source_paths, root: Path):
        ctrl = PipelineController()
        ctrl._begin_run()
        messages: list = []
        finished: list = []
        cb = PipelineCallbacks(
            on_message=lambda text, kind="info": messages.append((kind, text)),
            on_node_status=lambda iid, st: messages.append(("status", iid, st)),
            on_finished=lambda *args: finished.append(args),
        )
        ctrl._run_worker(
            nodes, [], source_paths, cb, str(root), {}, 2, None, False,
        )
        return messages, finished

    def test_lone_loader_without_file_is_blocked_not_crash(self) -> None:
        # Regression: an in-scope loader with no file used to die with
        # KeyError(iid) in _session_root_name ("Pipeline failed: '...'").
        with tempfile.TemporaryDirectory() as tmp:
            messages, finished = self._run(
                [("L1", LOAD)], {}, Path(tmp) / "s"
            )
        self.assertTrue(finished, "run must finish through the gate")
        _outputs, _tiles, failed, _t, metrics = finished[0]
        self.assertEqual(failed, 0)
        self.assertTrue(metrics.get("blocked_by_source"))
        errors = [text for kind, text, *_ in messages if kind == "error"]
        self.assertTrue(errors, "a readable message must name the node")
        self.assertIn("Load LAS/LAZ", errors[0])
        self.assertNotIn("'L1'", errors[0])

    def test_lone_loader_with_file_tiles_its_session(self) -> None:
        # Documented behavior: an isolated loader creates its tile session.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _tiny_cloud(root / "tiny.laz")
            messages, finished = self._run(
                [("L1", LOAD)], {"L1": str(root / "tiny.laz")}, root / "s"
            )
        self.assertTrue(finished)
        _outputs, _tiles, failed, _t, _metrics = finished[0]
        self.assertEqual(failed, 0)
        kinds = [kind for kind, *_ in messages]
        self.assertNotIn("error", kinds)


if __name__ == "__main__":
    unittest.main()
