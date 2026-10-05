# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Canvas selection funnel: single shows config, zero/several show blank."""

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
        self.view.node_selected.connect(self.emitted.append)

    def tearDown(self) -> None:
        self.view.deleteLater()

    def test_native_multi_select_blanks_inspector(self) -> None:
        # Native Qt multi-selection (Ctrl+click path): no collapse, the
        # inspector must go blank instead of showing stale content.
        self.emitted.clear()
        self.a.setSelected(True)
        self.b.setSelected(True)
        self.assertEqual(self.view.selected_nodes(), [self.a, self.b])
        self.assertEqual(self.emitted, [None])
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

    def test_delete_all_blanks_once(self) -> None:
        self.a.setSelected(True)
        self.b.setSelected(True)
        # Inspector already blank from the multi-select: deleting changes
        # nothing it shows, so no redundant emission follows.
        self.emitted.clear()
        self.view.delete_selected()
        self.assertEqual(self.view.selected_nodes(), [])
        self.assertEqual(self.emitted, [])

    def test_deselect_all_after_multi_blanks(self) -> None:
        self.a.setSelected(True)
        self.b.setSelected(True)
        self.emitted.clear()
        self.b.setSelected(False)
        self.a.setSelected(False)
        self.assertEqual(self.emitted, [self.a, None])
        self.assertIsNone(self.view.selected_item())


if __name__ == "__main__":
    unittest.main()
