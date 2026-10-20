# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Stock-button localization: no English Qt built-in reaches the user.

Every QMessageBox/QDialogButtonBox stock button goes through the dialogs
facade, which pins catalog texts. Offscreen Qt throughout (construct, no
paint); modal flows close themselves via a zero-shot timer.
"""

from __future__ import annotations

import ast
import os
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QApplication,
    QDialogButtonBox,
    QMessageBox,
)

from lynceus.plugins.locale import locale_manager, t
from lynceus.ui.dialogs import (
    _BUTTON_KEYS,
    ask_save_discard_cancel,
    ask_yes_no,
    localize_buttons,
    show_critical,
    show_info,
    show_warning,
)

ROOT = Path(__file__).resolve().parent.parent
_PACKS = ROOT / "extensions" / "locales"
FACADE_FILES = (
    ROOT / "lynceus" / "ui" / "extensions_dialog.py",
    ROOT / "lynceus" / "ui" / "main_window.py",
    ROOT / "lynceus" / "ui" / "runs_panel.py",
)
BUTTONBOX_FILES = (
    ROOT / "lynceus" / "ui" / "preferences_dialog.py",
    ROOT / "lynceus" / "ui" / "nodes" / "widgets" / "file_queue_dialog.py",
    ROOT / "lynceus" / "ui" / "start_dialog.py",
)


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


class HelperMappingTests(unittest.TestCase):
    def setUp(self) -> None:
        _app()
        self._old = locale_manager.language
        locale_manager.set_language("es")
        self.addCleanup(locale_manager.set_language, self._old)

    @unittest.skipUnless(
        _PACKS.is_dir(), "locale packs not present in this checkout")
    def test_message_box_buttons_follow_catalog(self) -> None:
        box = QMessageBox(
            QMessageBox.Icon.Question, "T", "B",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        localize_buttons(box, "Yes", "No")
        self.assertEqual(
            box.button(QMessageBox.StandardButton.Yes).text(), "Sí")
        self.assertEqual(
            box.button(QMessageBox.StandardButton.No).text(), "No")
        self.assertEqual(
            box.button(QMessageBox.StandardButton.Yes).property(
                "origText_en"), "Yes")

    @unittest.skipUnless(
        _PACKS.is_dir(), "locale packs not present in this checkout")
    def test_button_box_buttons_follow_catalog(self) -> None:
        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Apply
            | QDialogButtonBox.StandardButton.Cancel
        )
        localize_buttons(box, "Ok", "Apply", "Cancel")
        texts = {
            role: box.button(getattr(QDialogButtonBox.StandardButton, role)).text()
            for role in ("Ok", "Apply", "Cancel")
        }
        self.assertEqual(texts, {"Ok": "Aceptar", "Apply": "Aplicar",
                                 "Cancel": "Cancelar"})

    def test_english_active_keeps_english(self) -> None:
        locale_manager.set_language("en")
        box = QMessageBox(
            QMessageBox.Icon.Warning, "T", "B",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
        )
        localize_buttons(box, "Save", "Discard", "Cancel")
        self.assertEqual(
            box.button(QMessageBox.StandardButton.Discard).text(), "Discard")


class HelperErrorTests(unittest.TestCase):
    def setUp(self) -> None:
        _app()

    def test_unknown_role_fails_loud(self) -> None:
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        with self.assertRaises(ValueError):
            localize_buttons(box, "OK")
        with self.assertRaises(ValueError):
            localize_buttons(box, "yes")

    def test_wrong_box_type_fails_loud(self) -> None:
        with self.assertRaises(TypeError):
            localize_buttons(object(), "Ok")

    def test_absent_button_is_skipped(self) -> None:
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        localize_buttons(box, "Ok", "Cancel")
        self.assertEqual(
            box.button(QDialogButtonBox.StandardButton.Ok).text(),
            t("OK"))


class AskFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        _app()
        self._old = locale_manager.language
        locale_manager.set_language("es")
        self.addCleanup(locale_manager.set_language, self._old)

    def _close_active(self, result) -> None:
        widget = QApplication.activeModalWidget()
        if widget is not None:
            widget.done(result)

    def test_ask_yes_no_maps_result(self) -> None:
        QTimer.singleShot(
            0, lambda: self._close_active(QMessageBox.StandardButton.Yes))
        self.assertTrue(ask_yes_no(None, "T", "B"))
        QTimer.singleShot(
            0, lambda: self._close_active(QMessageBox.StandardButton.No))
        self.assertFalse(ask_yes_no(None, "T", "B"))

    def test_ask_save_discard_cancel_maps_result(self) -> None:
        QTimer.singleShot(
            0, lambda: self._close_active(
                QMessageBox.StandardButton.Discard))
        self.assertEqual(
            ask_save_discard_cancel(None, "T", "B"),
            QMessageBox.StandardButton.Discard)

    def test_show_helpers_do_not_raise(self) -> None:
        QTimer.singleShot(0, lambda: self._close_active(1))
        show_info(None, "T", "B")
        QTimer.singleShot(0, lambda: self._close_active(1))
        show_warning(None, "T", "B")
        QTimer.singleShot(0, lambda: self._close_active(1))
        show_critical(None, "T", "B")


class FacadeEnforcementTests(unittest.TestCase):
    """No raw stock-button callables outside the dialogs facade."""

    def _calls(self, path: Path):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "QMessageBox"
        ]

    def test_no_static_convenience_calls(self) -> None:
        banned = {"information", "warning", "critical", "question"}
        for path in FACADE_FILES:
            with self.subTest(file=path.name):
                bad = [c for c in self._calls(path)
                       if c.func.attr in banned]
                self.assertEqual(bad, [])

    def test_localize_roles_are_known(self) -> None:
        known = {role for role, _key in _BUTTON_KEYS}
        paths = list(FACADE_FILES) + list(BUTTONBOX_FILES) + [
            ROOT / "lynceus" / "ui" / "dialogs.py"]
        for path in paths:
            with self.subTest(file=path.name):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "localize_buttons"
                    ):
                        for arg in node.args[1:]:
                            self.assertIsInstance(arg, ast.Constant)
                            self.assertIn(arg.value, known)


if __name__ == "__main__":
    unittest.main()
