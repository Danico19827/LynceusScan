# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Retranslation pins for persistent widgets (offscreen Qt + AST).

Regression cover for the untranslated-UI report: labels built with ``t()``
at construction strand in the construction language unless the English
source is pinned (``origText_en``), because the translate pass captures the
current (possibly already translated) text as the "original" on its first
run. Persistent widgets must therefore pin every display string up front;
transient per-open menus are exempt (rebuilt in the current language).
"""

from __future__ import annotations

import ast
import os
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel, QMenu

from lynceus.plugins.locale import locale_manager, t
from lynceus.ui.translate import translate_widget

ROOT = Path(__file__).resolve().parents[1]

_KEEP = "Keep intermediate files and session cache"
_HINT = "Performance settings apply to the next run."
_OUTPUT = "Output folder"
_OUTPUT_DESC = (
    "Where runs store their sessions (artifacts, quality report, "
    "cache). Leave empty for the default output folder. Changing it "
    "redirects future runs only; existing sessions stay where they "
    "are. The CLI --output flag, when given, overrides this value."
)


def _catalog(lang: str) -> dict:
    import json

    path = ROOT / "extensions" / "locales" / lang / f"{lang}.json"
    return json.loads(path.read_text(encoding="utf-8"))[lang]


class PersistentWidgetRoundtripTests(unittest.TestCase):
    """es -> ru -> es over real persistent widgets (nothing strands)."""

    @classmethod
    def setUpClass(cls) -> None:
        if not (ROOT / "extensions" / "locales" / "es" / "es.json").is_file():
            raise unittest.SkipTest("locale packs not present in this checkout")
        cls._app = QApplication.instance() or QApplication([])
        cls._previous = locale_manager.language

    @classmethod
    def tearDownClass(cls) -> None:
        locale_manager.set_language(cls._previous)

    def _to(self, lang: str) -> None:
        locale_manager.set_language(lang)
        self.assertEqual(locale_manager.language, lang)

    def test_keep_intermediates_roundtrip(self) -> None:
        from lynceus.ui.preferences_dialog import _PerformancePage

        page = _PerformancePage()
        try:
            self.assertEqual(
                page.keep_intermediates.property("origText_en"), _KEEP
            )
            self._to("es")
            translate_widget(page)
            self.assertEqual(
                page.keep_intermediates.text(), _catalog("es")[_KEEP]
            )
            self._to("ru")
            translate_widget(page)
            self.assertEqual(
                page.keep_intermediates.text(), _catalog("ru")[_KEEP]
            )
            self.assertNotEqual(page.keep_intermediates.text(), _KEEP)
            self._to("en")
            translate_widget(page)
            self.assertEqual(page.keep_intermediates.text(), _KEEP)
        finally:
            page.deleteLater()

    def test_performance_hint_roundtrip(self) -> None:
        from lynceus.ui.preferences_dialog import _PerformancePage

        page = _PerformancePage()
        try:
            pinned = [
                label for label in page.findChildren(QLabel)
                if label.property("origText_en") == _HINT
            ]
            self.assertEqual(len(pinned), 1)
            self._to("ru")
            translate_widget(page)
            self.assertEqual(pinned[0].text(), _catalog("ru")[_HINT])
            self._to("en")
            translate_widget(page)
            self.assertEqual(pinned[0].text(), _HINT)
        finally:
            page.deleteLater()

    def test_output_folder_row_roundtrip(self) -> None:
        from lynceus.ui.preferences_dialog import _GeneralPage

        page = _GeneralPage()
        try:
            self._to("ru")
            translate_widget(page)
            ru = _catalog("ru")
            texts = [label.text() for label in page.findChildren(QLabel)]
            self.assertIn(ru[_OUTPUT], texts)
            self.assertIn(ru[_OUTPUT_DESC], texts)
            self.assertEqual(
                page.output_root.placeholderText(), ru["Default"]
            )
            self._to("en")
            translate_widget(page)
            texts = [label.text() for label in page.findChildren(QLabel)]
            self.assertIn(_OUTPUT, texts)
        finally:
            page.deleteLater()

    def test_preferences_stock_buttons_roundtrip(self) -> None:
        from PySide6.QtWidgets import QDialogButtonBox

        from lynceus.ui.preferences_dialog import PreferencesDialog

        dialog = PreferencesDialog()
        try:
            box = dialog.findChild(QDialogButtonBox)
            self.assertIsNotNone(box)
            ok_btn = box.button(QDialogButtonBox.StandardButton.Ok)
            self._to("ru")
            translate_widget(dialog)
            self.assertEqual(ok_btn.text(), _catalog("ru")["OK"])
            self._to("en")
            translate_widget(dialog)
            self.assertEqual(ok_btn.text(), "OK")
        finally:
            dialog.deleteLater()

    def test_pinned_menu_action_roundtrip(self) -> None:
        """The pin contract itself: pinned actions follow the language."""
        from PySide6.QtWidgets import QWidget

        host = QWidget()
        menu = QMenu(host)
        try:
            action = menu.addAction(t("Edit"))
            action.setProperty("origText_en", "Edit")
            self._to("ru")
            translate_widget(host)
            self.assertEqual(action.text(), _catalog("ru")["Edit"])
            self._to("en")
            translate_widget(host)
            self.assertEqual(action.text(), "Edit")
        finally:
            host.deleteLater()


class EditMenuPinsTests(unittest.TestCase):
    """Every persistent Edit-menu action pins its English source (AST).

    The main window cannot be instantiated offscreen (QUiLoader), so the
    pin contract for its long-lived menu is guarded statically: an
    ``addAction(t("<lit>"))`` without a nearby ``origText_en`` pin strands
    the entry in the construction language after a switch.
    """

    def test_edit_actions_pinned(self) -> None:
        path = ROOT / "lynceus" / "ui" / "main_window.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        missing: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and func.attr == "addAction"
            ):
                continue
            if not node.args:
                continue
            arg = node.args[0]
            if not (
                isinstance(arg, ast.Call)
                and isinstance(arg.func, ast.Name)
                and arg.func.id == "t"
                and arg.args
                and isinstance(arg.args[0], ast.Constant)
            ):
                continue
            literal = arg.args[0].value
            # The pin must be a setProperty("origText_en", "<same>") call
            # shortly after within the same statement block.
            found = False
            for other in ast.walk(tree):
                if not isinstance(other, ast.Call):
                    continue
                ofunc = other.func
                if not (
                    isinstance(ofunc, ast.Attribute)
                    and ofunc.attr == "setProperty"
                    and len(other.args) >= 2
                    and isinstance(other.args[0], ast.Constant)
                    and other.args[0].value == "origText_en"
                    and isinstance(other.args[1], ast.Constant)
                    and other.args[1].value == literal
                ):
                    continue
                found = True
                break
            if not found:
                missing.append(literal)
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
