# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Robust project opening (block A): corrupt documents never crash nor
wipe the live canvas.

Covers the open choke point at two levels: Qt-free document validation
(``lynceus.project``) and canvas restore (offscreen) — per-node state
failures degrade to warnings with defaults, garbage view/edges degrade
to clamps/skips, and full ``MainWindow._open_path`` atomicity (snapshot +
rollback) is structural on top of these units.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from lynceus import project as project_io
from lynceus.ui.nodes.canvas import NodeCanvasView
from lynceus.ui.nodes.node_item import NodeItem

LOAD_ID = "lynceus.nodes.lidar.source.load_las_laz"
DTM_ID = "lynceus.nodes.lidar.terrain.generate_dtm"


def _doc(nodes, edges=None, **extra):
    doc = {
        "format": project_io.PROJECT_FORMAT,
        "version": project_io.PROJECT_VERSION,
        "nodes": nodes,
        "edges": edges or [],
    }
    doc.update(extra)
    return doc


def _node(node_id, iid="n1", x=10.0, y=20.0, config=None, label=""):
    entry = {"id": node_id, "iid": iid, "x": x, "y": y}
    if label:
        entry["label"] = label
    if config is not None:
        entry["config"] = config
    return entry


class LoadValidationTests(unittest.TestCase):
    def _write(self, content: str) -> str:
        tmp = tempfile.NamedTemporaryFile(
            suffix=".lynx", delete=False, mode="w", encoding="utf-8"
        )
        tmp.write(content)
        tmp.close()
        self.addCleanup(os.unlink, tmp.name)
        return tmp.name

    def test_missing_file_rejected(self) -> None:
        with self.assertRaises(project_io.ProjectError):
            project_io.load_project("/nonexistent/path/file.lynx")

    def test_bad_json_rejected(self) -> None:
        path = self._write("{not json")
        with self.assertRaises(project_io.ProjectError):
            project_io.load_project(path)

    def test_wrong_format_rejected(self) -> None:
        path = self._write(json.dumps({"format": "other", "version": 1}))
        with self.assertRaises(project_io.ProjectError):
            project_io.load_project(path)

    def test_duplicate_iids_rejected(self) -> None:
        path = self._write(json.dumps(_doc([
            _node(LOAD_ID, iid="dup"), _node(DTM_ID, iid="dup"),
        ])))
        with self.assertRaises(project_io.ProjectError):
            project_io.load_project(path)

    def test_edge_to_unknown_node_rejected(self) -> None:
        path = self._write(json.dumps(_doc(
            [_node(LOAD_ID, iid="a")],
            [{"src": "a", "dst": "ghost"}],
        )))
        with self.assertRaises(project_io.ProjectError):
            project_io.load_project(path)


class RestoreRobustnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self._app = QApplication.instance() or QApplication([])
        self.view = NodeCanvasView()

    def tearDown(self) -> None:
        self.view.deleteLater()

    def test_garbage_view_never_crashes(self) -> None:
        warnings = self.view.restore_graph(_doc(
            [_node(LOAD_ID)], view={"scale": "huge", "cx": None},
        ))
        self.assertIsInstance(warnings, list)
        self.assertEqual(len(self.view._node_items), 1)

    def test_garbage_edges_warn_without_raise(self) -> None:
        warnings = self.view.restore_graph(_doc(
            [_node(LOAD_ID, iid="a")],
            [{"src": "a", "dst": "a", "src_out": 99, "dst_in": -1},
             {"src": "a", "dst": "a"}],
        ))
        self.assertTrue(warnings)
        self.assertEqual(len(self.view._node_items), 1)

    def test_bad_config_keeps_node_with_warning(self) -> None:
        with mock.patch.object(
            NodeItem, "apply_config", side_effect=RuntimeError("boom")
        ):
            warnings = self.view.restore_graph(_doc(
                [_node(LOAD_ID, iid="a", config={"pipeline": {"x": 1}})],
            ))
        self.assertEqual(len(self.view._node_items), 1)
        self.assertTrue(
            any("defaults" in w for w in warnings),
            f"expected kept-with-defaults warning, got {warnings}",
        )

    def test_failed_restore_leaves_prior_state_restorable(self) -> None:
        """Snapshot-then-restore round-trips: the unit _open_path
        rollback relies on (canvas-level half of atomicity)."""
        self.view.add_node(LOAD_ID)
        snapshot = self.view.serialize_graph()
        self.view.clear_graph()
        self.assertEqual(self.view._node_items, [])
        warnings = self.view.restore_graph(snapshot)
        self.assertIsInstance(warnings, list)
        self.assertEqual(len(self.view._node_items), 1)


if __name__ == "__main__":
    unittest.main()
