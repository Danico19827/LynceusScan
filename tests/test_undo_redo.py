# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Undo/redo history (item 9): snapshot-based document history with gesture
merging, the 100-step cap, dirty-by-comparison and run suppression."""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF

from PySide6.QtWidgets import QApplication

from lynceus.plugins.registry import manager
from lynceus.ui.nodes import canvas as canvas_module
from lynceus.ui.nodes.canvas import NodeCanvasView

LOAD_ID = "lynceus.nodes.lidar.source.load_las_laz"
DTM_ID = "lynceus.nodes.lidar.terrain.generate_dtm"
RASTER_ID = "lynceus.nodes.raster.input_raster"

OLD_WINDOW = canvas_module.UNDO_MERGE_WINDOW
OLD_LIMIT = canvas_module.UNDO_LIMIT


class UndoRedoTests(unittest.TestCase):
    def setUp(self) -> None:
        self._app = QApplication.instance() or QApplication([])
        self.assertTrue(manager.contains(LOAD_ID))
        self.assertTrue(manager.contains(DTM_ID))
        self.assertTrue(manager.contains(RASTER_ID))
        canvas_module.UNDO_MERGE_WINDOW = OLD_WINDOW
        canvas_module.UNDO_LIMIT = OLD_LIMIT
        self.view = NodeCanvasView()
        self.view.clear_history()
        self.history: list[tuple[bool, bool]] = []
        self.applied = 0
        self.view.history_changed.connect(
            lambda a, b: self.history.append((a, b))
        )
        self.view.history_applied.connect(lambda: setattr(self, "applied", self.applied + 1))

    def tearDown(self) -> None:
        self.view.deleteLater()

    def _snapshot(self) -> dict:
        return self.view.document_snapshot()

    def test_add_node_undo_redo(self) -> None:
        before = self._snapshot()
        item = self.view.add_node(LOAD_ID, QPointF(100, 100))
        after = self._snapshot()
        self.assertNotEqual(before, after)
        self.assertTrue(self.view.can_undo())
        self.view.undo()
        self.assertEqual(self.view._node_items, [])
        self.assertEqual(self._snapshot(), before)
        self.assertTrue(self.view.can_redo())
        self.view.redo()
        self.assertEqual(len(self.view._node_items), 1)
        self.assertEqual(self.view._node_items[0].iid, item.iid)
        self.assertEqual(self._snapshot(), after)
        self.assertEqual(self.applied, 2)

    def test_connection_undo_redo(self) -> None:
        a = self.view.add_node(LOAD_ID, QPointF(0, 0))
        b = self.view.add_node(DTM_ID, QPointF(300, 0))
        wires = len(self.view._connections)
        self.view._create_connection(a.outputs()[0], b.inputs()[0])
        self.assertEqual(len(self.view._connections), wires + 1)
        self.view.undo()
        self.assertEqual(len(self.view._connections), wires)
        self.view.redo()
        self.assertEqual(len(self.view._connections), wires + 1)

    def test_disconnect_undo_redo(self) -> None:
        a = self.view.add_node(LOAD_ID, QPointF(0, 0))
        b = self.view.add_node(DTM_ID, QPointF(300, 0))
        self.view._create_connection(a.outputs()[0], b.inputs()[0])
        self.view.undo()
        self.assertEqual(self.view._connections, [])
        self.view.redo()
        self.assertEqual(len(self.view._connections), 1)
        self.view._disconnect(self.view._connections[0])
        self.assertEqual(self.view._connections, [])
        self.view.undo()
        self.assertEqual(len(self.view._connections), 1)

    def test_delete_selected_is_one_step(self) -> None:
        a = self.view.add_node(LOAD_ID, QPointF(0, 0))
        b = self.view.add_node(DTM_ID, QPointF(300, 0))
        self.view._create_connection(a.outputs()[0], b.inputs()[0])
        self.view.clear_history()
        a.setSelected(True)
        b.setSelected(True)
        self.view.delete_selected()
        self.assertEqual(self.view._node_items, [])
        self.view.undo()
        self.assertEqual(len(self.view._node_items), 2)
        self.assertEqual(len(self.view._connections), 1)

    def test_config_edit_undo_redo(self) -> None:
        # Config edits round-trip as whole-document snapshots: compare the
        # documents instead of key shapes (capture_config wraps "pipeline").
        item = self.view.add_node(LOAD_ID, QPointF(0, 0))
        iid = item.iid
        before = self._snapshot()
        config = item.pipeline_config()
        config["tile_size_m"] = (config.get("tile_size_m") or 0.0) + 42.5
        item.set_pipeline_config(config)
        self.view.on_config_changed(iid)
        after = self._snapshot()
        self.assertNotEqual(before, after)
        self.view.undo()
        self.assertEqual(self._snapshot(), before)
        self.view.redo()
        self.assertEqual(self._snapshot(), after)

    def test_label_edit_undo_redo(self) -> None:
        item = self.view.add_node(LOAD_ID, QPointF(0, 0))
        iid = item.iid
        item.set_label("My Label")
        self.view.on_node_renamed(iid)
        self.assertEqual(self.view._node_items[0].label(), "My Label")
        self.view.undo()
        self.assertNotEqual(self.view._node_items[0].label(), "My Label")
        self.view.redo()
        self.assertEqual(self.view._node_items[0].label(), "My Label")

    def test_strategy_change_undo_redo(self) -> None:
        item = self.view.add_node(RASTER_ID, QPointF(0, 0))
        self.assertEqual(item.strategy(), "")
        self.assertFalse(item.outputs())  # inert base until a strategy is set
        self.view.set_strategy(item, "dtm")
        self.assertEqual(self.view._node_items[0].strategy(), "dtm")
        self.assertTrue(self.view._node_items[0].outputs())
        self.view.undo()
        self.assertEqual(self.view._node_items[0].strategy(), "")
        self.assertFalse(self.view._node_items[0].outputs())  # back to inert
        self.view.redo()
        self.assertEqual(self.view._node_items[0].strategy(), "dtm")
        self.assertTrue(self.view._node_items[0].outputs())

    def test_move_coalesces_into_one_step(self) -> None:
        item = self.view.add_node(LOAD_ID, QPointF(0, 0))
        self.view.clear_history()
        canvas_module.UNDO_MERGE_WINDOW = 10.0
        start = self._snapshot()
        item.setPos(QPointF(100, 100))
        self.view._graph_mutated("move")
        item.setPos(QPointF(200, 200))
        self.view._graph_mutated("move")
        item.setPos(QPointF(300, 300))
        self.view._graph_mutated("move")
        # The whole drag collapses into a single undo step.
        self.view.undo()
        self.assertEqual(self._snapshot(), start)
        self.assertFalse(self.view.can_undo())

    def test_window_boundary_splits_steps(self) -> None:
        canvas_module.UNDO_MERGE_WINDOW = 0.0
        self.view.add_node(LOAD_ID, QPointF(0, 0))
        self.view.add_node(DTM_ID, QPointF(300, 0))
        self.assertEqual(len(self.view._history), 3)  # seed + 2 steps
        self.view.undo()
        self.assertEqual(len(self.view._node_items), 1)
        self.view.undo()
        self.assertEqual(self.view._node_items, [])

    def test_redo_invalidated_by_new_mutation(self) -> None:
        self.view.add_node(LOAD_ID, QPointF(0, 0))
        self.view.undo()
        self.assertTrue(self.view.can_redo())
        self.view.add_node(DTM_ID, QPointF(300, 0))
        self.assertFalse(self.view.can_redo())
        self.view.undo()
        self.view.undo()
        self.assertEqual(self.view._node_items, [])

    def test_limit_caps_history(self) -> None:
        canvas_module.UNDO_MERGE_WINDOW = 0.0
        canvas_module.UNDO_LIMIT = 3
        for i in range(10):
            self.view.add_node(LOAD_ID, QPointF(i * 10, 0))
        self.assertLessEqual(len(self.view._history), 4)  # 3 + seed
        for _ in range(10):
            self.view.undo()
        # Only the 3 most recent adds are undoable.
        self.assertEqual(len(self.view._node_items), 7)

    def test_clear_history_resets(self) -> None:
        self.view.add_node(LOAD_ID, QPointF(0, 0))
        self.assertTrue(self.view.can_undo())
        self.view.clear_history()
        self.assertFalse(self.view.can_undo())
        self.assertFalse(self.view.can_redo())
        self.assertEqual(len(self.view._history), 1)
        self.view.undo()  # no-op and harmless
        self.assertEqual(len(self.view._node_items), 1)

    def test_run_active_suppresses_recording(self) -> None:
        self.view.add_node(LOAD_ID, QPointF(0, 0))
        self.view.undo()  # clear to empty baseline
        self.view._run_active = True
        self.view.add_node(DTM_ID, QPointF(300, 0))
        self.assertFalse(self.view.can_undo())
        self.view.undo()  # blocked while running
        self.assertEqual(len(self.view._node_items), 1)
        self.view._run_active = False

    def test_apply_run_staging_is_one_step(self) -> None:
        # Edits made during a run record as a single step when applied.
        self.view.add_node(LOAD_ID, QPointF(0, 0))
        pre_run = self._snapshot()
        self.view._run_active = True
        self.view._graph_snapshot = self.view.serialize_graph()
        self.view.add_node(DTM_ID, QPointF(300, 0))
        self.assertFalse(self.view.can_undo())
        self.assertNotEqual(self._snapshot(), pre_run)
        self.view._clear_run_pending()
        self.view._record_current_state("run")
        self.assertTrue(self.view.can_undo())
        self.view.undo()
        self.assertEqual(self._snapshot(), pre_run)
        self.view.redo()
        self.assertEqual(len(self.view._node_items), 2)

    def test_discard_run_staging_restores_pre_run_state(self) -> None:
        self.view.add_node(LOAD_ID, QPointF(0, 0))
        pre_run = self._snapshot()
        self.view._run_active = True
        self.view._graph_snapshot = self.view.serialize_graph()
        self.view.add_node(DTM_ID, QPointF(300, 0))
        self.view._discard_pending_changes()
        self.assertEqual(self._snapshot(), pre_run)
        self.assertTrue(self.view.can_undo())  # history intact

    def test_document_snapshot_excludes_view(self) -> None:
        node = self.view.add_node(LOAD_ID, QPointF(0, 0))
        node.setPos(QPointF(50, 60))
        doc = self.view.document_snapshot()
        self.assertNotIn("view", doc)
        self.assertIn("nodes", doc)
        self.assertIn("edges", doc)

    def test_undo_restore_keeps_viewport(self) -> None:
        self.view.add_node(LOAD_ID, QPointF(0, 0))
        self.view.scale(2.0, 2.0)
        before = self.view.transform().m11()
        self.view.undo()  # restore path must not reset zoom
        self.view.redo()
        self.assertEqual(self.view.transform().m11(), before)


if __name__ == "__main__":
    unittest.main()