# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Launcher structure (phase 1 polish): lists populate, names resolve."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication


def _template(name: str, valid: bool = True):
    return SimpleNamespace(
        name=name,
        source="bundled",
        valid=valid,
        path=f"/tmp/{name}.lynx",
        description=f"{name} description",
        author="Author",
        version="1.0",
        tags=["demo"],
        node_count=3,
        error="" if valid else "broken",
    )


class StartDialogStructureTests(unittest.TestCase):
    def setUp(self) -> None:
        self._app = QApplication.instance() or QApplication([])

    def test_lists_populate_and_names_resolve(self) -> None:
        import tempfile
        from pathlib import Path

        from lynceus.ui.start_dialog import StartDialog

        with tempfile.TemporaryDirectory() as tmp:
            real = str(Path(tmp) / "a.lynx")
            Path(real).touch()
            dialog = StartDialog(
                [real, str(Path(tmp) / "missing.lynx")],
                [_template("Alpha"), _template("Broken", valid=False)],
                parent=None,
            )
            try:
                self._assert_structure(dialog, real)
            finally:
                dialog.deleteLater()

    def _assert_structure(self, dialog, real: str) -> None:
            self.assertEqual(
                dialog._recent_list.objectName(), "recentList")
            self.assertEqual(
                dialog._template_list.list.objectName(), "templateList")
            # Only existing files enter the recents list.
            self.assertEqual(dialog._recent_list.count(), 1)
            self.assertGreaterEqual(
                dialog._template_list.list.count(), 2)
            selected = dialog._template_list.selected()
            self.assertIsNotNone(selected)
            self.assertEqual(selected.name, "Alpha")
            titles = {
                w.objectName() for w in dialog.findChildren(object)
                if w.objectName().startswith("start")}
            # Brand header: lockup art when assets ship, text fallback.
            self.assertTrue("startLockup" in titles or "startTitle" in titles)
            self.assertIn("startSubtitle", titles)
            self.assertIn("startSection", titles)
            cards = [
                w for w in dialog.findChildren(object)
                if w.objectName() == "extName"]
            self.assertGreaterEqual(len(cards), 3)


if __name__ == "__main__":
    unittest.main()
