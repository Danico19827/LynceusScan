# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Canvas floating toolbar: states, collapse and control names."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication


class CanvasToolbarTests(unittest.TestCase):
    def setUp(self) -> None:
        from lynceus.ui.nodes.canvas_toolbar import CanvasToolbar

        self._app = QApplication.instance() or QApplication([])
        self.bar = CanvasToolbar()
        self.bar.show()

    def tearDown(self) -> None:
        self.bar.deleteLater()

    def test_state_machine_enables(self) -> None:
        self.bar.set_state("idle")
        self.assertTrue(self.bar.btn_run.isEnabled())
        self.assertFalse(self.bar.btn_cancel.isEnabled())
        self.assertFalse(self.bar.btn_pause.isEnabled())
        self.bar.set_state("running")
        self.assertFalse(self.bar.btn_run.isEnabled())
        self.assertTrue(self.bar.btn_cancel.isEnabled())
        self.assertTrue(self.bar.btn_pause.isEnabled())
        self.assertEqual(
            self.bar.btn_pause.property("origText_en"), "Pause")
        self.bar.set_state("paused")
        self.assertEqual(
            self.bar.btn_pause.property("origText_en"), "Resume")

    def test_collapse_toggles_views(self) -> None:
        self.assertTrue(self.bar._bar.isVisible())
        self.bar.set_collapsed(True)
        self.assertFalse(self.bar._bar.isVisible())
        self.assertTrue(self.bar._tab.isVisible())
        self.bar.toggle()
        self.assertTrue(self.bar._bar.isVisible())

    def test_named_controls(self) -> None:
        names = {b.objectName() for b in
                 self.bar.findChildren(type(self.bar.btn_run))}
        for expected in ("btnRun", "btnPause", "btnCancel", "btnCollapse",
                         "btnExpand"):
            self.assertIn(expected, names)


if __name__ == "__main__":
    unittest.main()
