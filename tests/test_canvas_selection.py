# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Canvas selection funnel: single shows config, several show the multi panel,
zero shows the placeholder."""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

from lynceus.plugins.registry import manager
from lynceus.ui.nodes.canvas import NodeCanvasView

LOAD_ID = "lynceus.nodes.lidar.source.load_las_laz"
DTM_ID = "lynceus.nodes.lidar.terrain.generate_dtm"


class SelectionFunnelTests(unittest.TestCase):
    def setUp(self) -> None:
        self._app = QApplication.instance() or QApplication([])
        self.assertTrue(manager.contains(LOAD_ID))
        self.assertTrue(manager.contains(DTM_ID))
        self.view = NodeCanvasView()
        self.a = self.view.add_node(LOAD_ID, QPointF(0, 0))
        self.b = self.view.add_node(DTM_ID, QPointF(500, 0))
        self.emitted: list = []
        self.multi: list = []
        self.view.node_selected.connect(self.emitted.append)
        self.view.nodes_selected.connect(self.multi.append)

    def tearDown(self) -> None:
        self.view.deleteLater()

    def test_native_multi_select_emits_multi_panel(self) -> None:
        # Native Qt multi-selection (Ctrl+click path): no collapse, the
        # inspector shows the multi panel instead of stale content.
        self.emitted.clear()
        self.multi.clear()
        self.a.setSelected(True)
        self.b.setSelected(True)
        self.assertEqual(self.view.selected_nodes(), [self.a, self.b])
        self.assertEqual(self.multi, [[self.a, self.b]])
        self.assertEqual(self.emitted, [])
        self.assertIsNone(self.view.selected_item())

    def test_back_to_single_restores_config(self) -> None:
        self.a.setSelected(True)
        self.b.setSelected(True)
        self.emitted.clear()
        self.b.setSelected(False)
        self.assertEqual(self.emitted, [self.a])
        self.assertIs(self.view.selected_item(), self.a)

    def test_bare_click_collapses_and_shows(self) -> None:
        self.a.setSelected(True)
        self.b.setSelected(True)
        self.emitted.clear()
        self.view._select_node(self.b)
        self.assertEqual(self.view.selected_nodes(), [self.b])
        self.assertEqual(self.emitted, [self.b])

    def test_delete_all_blanks_after_multi(self) -> None:
        # The inspector is showing the multi panel: deleting every selected
        # node must switch it back to the placeholder (one emission).
        self.a.setSelected(True)
        self.b.setSelected(True)
        self.emitted.clear()
        self.view.delete_selected()
        self.assertEqual(self.view.selected_nodes(), [])
        self.assertEqual(self.emitted, [None])

    def test_deselect_all_after_multi_blanks(self) -> None:
        self.a.setSelected(True)
        self.b.setSelected(True)
        self.emitted.clear()
        self.b.setSelected(False)
        self.a.setSelected(False)
        self.assertEqual(self.emitted, [self.a, None])
        self.assertIsNone(self.view.selected_item())

    def test_select_all_emits_multi_once(self) -> None:
        self.emitted.clear()
        self.multi.clear()
        self.view.select_all()
        self.assertEqual(self.view.selected_nodes(), [self.a, self.b])
        self.assertEqual(len(self.multi), 1)
        self.assertEqual(len(self.multi[0]), 2)


class ClipboardTests(unittest.TestCase):
    def setUp(self) -> None:
        self._app = QApplication.instance() or QApplication([])
        self.view = NodeCanvasView()
        self.a = self.view.add_node(LOAD_ID, QPointF(0, 0))
        self.b = self.view.add_node(DTM_ID, QPointF(500, 0))
        self.view._create_connection(self.a.outputs()[0], self.b.inputs()[0])

    def tearDown(self) -> None:
        QApplication.clipboard().clear()
        self.view.deleteLater()

    def _select_both(self) -> None:
        self.view._sync_selection_to([self.a, self.b])
        self.view._sync_inspector_to_scene()

    def test_copy_paste_roundtrip_keeps_internal_edge(self) -> None:
        self._select_both()
        self.view.copy_selection()
        before = len(self.view._node_items)
        new_items = self.view.paste()
        self.assertIsNotNone(new_items)
        self.assertEqual(len(self.view._node_items), before + 2)
        # The pasted pair keeps its single internal wire.
        self.assertEqual(len(self.view._connections), 2)
        # Fresh instance ids: no aliasing with the source pair.
        new_iids = {item.iid for item in new_items}
        self.assertFalse(new_iids & {self.a.iid, self.b.iid})

    def test_duplicate_clones_selection_without_external_wires(self) -> None:
        self._select_both()
        before = len(self.view._node_items)
        new_items = self.view.duplicate()
        self.assertEqual(len(new_items), 2)
        self.assertEqual(len(self.view._node_items), before + 2)
        self.assertEqual(len(self.view._connections), 2)

    def test_paste_without_payload_warns(self) -> None:
        QApplication.clipboard().setText("not a graph")
        messages: list = []
        self.view.message_requested.connect(lambda text, kind: messages.append(text))
        self.assertIsNone(self.view.paste())
        self.assertTrue(messages)


class LoaderFocusTests(unittest.TestCase):
    """Preflight merge suggestion: detect an enabled floor and jump to it."""

    def setUp(self) -> None:
        self._app = QApplication.instance() or QApplication([])
        self.view = NodeCanvasView()
        self.a = self.view.add_node(LOAD_ID, QPointF(0, 0))
        self.b = self.view.add_node(DTM_ID, QPointF(500, 0))
        # Helpers run post-snapshot in production (run_pipeline rebuilds
        # the iid map before preflight); mirror that state here.
        self.view._item_by_iid = {
            item.iid: item for item in self.view._node_items}

    def tearDown(self) -> None:
        self.view.deleteLater()

    def _nodes(self) -> list:
        return [(item.iid, None) for item in self.view._node_items]

    def test_merge_disabled_by_default(self) -> None:
        self.assertFalse(self.view._merge_enabled(self._nodes()))

    def test_merge_enabled_with_floor(self) -> None:
        config = self.a.pipeline_config()
        config["merge_sparse_below"] = 500000
        self.a.set_pipeline_config(config)
        self.assertTrue(self.view._merge_enabled(self._nodes()))

    def test_focus_loader_selects_it(self) -> None:
        self.b.setSelected(True)
        self.view._focus_loader(self._nodes())
        self.assertEqual(self.view.selected_nodes(), [self.a])

    def test_apply_merge_floor_sets_loaders_only(self) -> None:
        changes: list = []
        self.view.graph_changed.connect(lambda: changes.append(1))
        self.view._apply_merge_floor(self._nodes(), 500000)
        self.assertEqual(
            self.a.pipeline_config()["merge_sparse_below"], 500000)
        self.assertNotIn("merge_sparse_below", self.b.pipeline_config())
        self.assertTrue(changes)  # dirty via the config funnel

    def test_apply_merge_floor_skips_current_and_junk(self) -> None:
        config = self.a.pipeline_config()
        config["merge_sparse_below"] = 500000
        self.a.set_pipeline_config(config)
        changes: list = []
        self.view.graph_changed.connect(lambda: changes.append(1))
        self.view._apply_merge_floor(self._nodes(), 500000)
        self.assertEqual(changes, [])
        bad = self.a.pipeline_config()
        bad["merge_sparse_below"] = "junk"
        self.a.set_pipeline_config(bad)
        self.view._apply_merge_floor(self._nodes(), 250000)
        self.assertEqual(
            self.a.pipeline_config()["merge_sparse_below"], 250000)

    def test_apply_merge_floor_rejects_nonpositive(self) -> None:
        changes: list = []
        self.view.graph_changed.connect(lambda: changes.append(1))
        self.view._apply_merge_floor(self._nodes(), 0)
        self.view._apply_merge_floor(self._nodes(), -5)
        self.assertEqual(changes, [])
        self.assertNotIn(
            "merge_sparse_below", self.a.pipeline_config())

    def test_apply_node_config_sets_raster_resolution(self) -> None:
        changes: list = []
        self.view.graph_changed.connect(lambda: changes.append(1))
        applied = self.view._apply_node_config(
            self._nodes(),
            "resolution", 4.0,
            lambda caps: isinstance(caps.get("config_schema"), dict)
            and "resolution" in caps["config_schema"],
        )
        self.assertEqual(applied, 1)
        self.assertEqual(self.b.pipeline_config()["resolution"], 4.0)
        self.assertNotIn("resolution", self.a.pipeline_config())
        self.assertTrue(changes)

    def test_apply_segments_sets_cap_and_flag(self) -> None:
        changes: list = []
        self.view.graph_changed.connect(lambda: changes.append(1))
        self.view._apply_segments(self._nodes(), 25_000_000)
        self.assertEqual(
            self.a.pipeline_config()["max_points_per_part"], 25_000_000)
        self.assertEqual(
            self.a.pipeline_config()["merge_segments"], False)
        self.assertNotIn("max_points_per_part", self.b.pipeline_config())
        self.assertNotIn("merge_segments", self.b.pipeline_config())
        self.assertTrue(changes)

    def test_apply_segments_rejects_nonpositive(self) -> None:
        changes: list = []
        self.view.graph_changed.connect(lambda: changes.append(1))
        self.view._apply_segments(self._nodes(), 0)
        self.assertEqual(changes, [])
        self.assertNotIn(
            "max_points_per_part", self.a.pipeline_config())


if __name__ == "__main__":
    unittest.main()
