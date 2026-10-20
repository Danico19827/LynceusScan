# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Preferences dialog keeps translated texts across language switches.

Regression test: combo entries and labels built with t() while a
non-English language is active used to strand on the old language,
because the generic translate pass captured the displayed text as the
English original. Population now goes through set_combo_items (English
pinned per index) and dynamic combos rebuild from English sources.
"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt, qInstallMessageHandler
from PySide6.QtWidgets import QApplication, QComboBox

from lynceus.plugins.locale import locale_manager, t
from lynceus.ui.translate import ORIG_ITEM_ROLE, language_changed


def _find_combos(widget):
    found = []

    def collect(node):
        if isinstance(node, QComboBox):
            found.append(node)
        for child in node.children():
            if hasattr(child, "children"):
                try:
                    collect(child)
                except RuntimeError:
                    pass  # already destroyed during rebuilds

    collect(widget)
    return found


class PreferencesRetranslateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def tearDown(self) -> None:
        locale_manager.set_language("en")

    def _open(self):
        from lynceus.ui.preferences_dialog import PreferencesDialog

        dialog = PreferencesDialog()
        dialog.show()
        self._app.processEvents()
        self.addCleanup(dialog.deleteLater)
        return dialog

    def _switch(self, dialog, code: str) -> None:
        locale_manager.set_language(code)
        self._app.setLayoutDirection(
            Qt.LayoutDirection.RightToLeft
            if code == "ar"
            else Qt.LayoutDirection.LeftToRight
        )
        language_changed.emit(code)
        self._app.processEvents()
        dialog.repaint()
        self._app.processEvents()

    def _assert_matches_catalog(self, dialog, code: str) -> None:
        locale_manager.set_language(code)
        combos = _find_combos(dialog)
        self.assertTrue(combos, "no combos found")
        for combo in combos:
            for i in range(combo.count()):
                pinned = combo.itemData(i, ORIG_ITEM_ROLE)
                if pinned:
                    self.assertEqual(combo.itemText(i), t(pinned))

    def test_built_in_arabic_retranslates_to_french(self) -> None:
        # The reported scenario: dialog first opened while Arabic was
        # active, then switched to French.
        locale_manager.set_language("ar")
        dialog = self._open()
        self._switch(dialog, "fr")
        self._assert_matches_catalog(dialog, "fr")
        # Spot-check the reported items by their English source.
        texts = []
        for combo in _find_combos(dialog):
            texts.extend(combo.itemText(i) for i in range(combo.count()))
        for english in ("Auto", "System default"):
            with self.subTest(english=english):
                self.assertIn(t(english), texts)

    def test_theme_combo_data_holds_ids_not_labels(self) -> None:
        from lynceus.ui.theming import available_themes

        dialog = self._open()
        general = None
        for _label, page in dialog._pages:
            if type(page).__name__ == "_GeneralPage":
                general = page
                break
        self.assertIsNotNone(general)
        expected = [tid for tid, _label in available_themes()]
        datas = [
            general.theme.itemData(i)
            for i in range(general.theme.count())
        ]
        self.assertEqual(datas, expected)
        # A non-default selection must survive rebuilds and save as its id.
        if len(expected) > 1:
            from PySide6.QtCore import QSettings

            from lynceus.ui.settings_keys import (
                SETTINGS_APP,
                SETTINGS_ORG,
                THEME_KEY,
            )

            settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
            backup = {
                key: settings.value(key) for key in settings.allKeys()
            }

            def _restore():
                current = set(settings.allKeys())
                for key in current - set(backup):
                    settings.remove(key)
                for key, value in backup.items():
                    settings.setValue(key, value)
                settings.sync()

            self.addCleanup(_restore)
            general.theme.setCurrentIndex(1)
            wanted = general.theme.currentData()
            self._switch(dialog, "ar")
            self._switch(dialog, "fr")
            self.assertEqual(general.theme.currentData(), wanted)
            for _label, page in dialog._pages:
                if hasattr(page, "save"):
                    page.save()
            self.assertEqual(
                str(settings.value(THEME_KEY)), wanted
            )

    def test_built_in_arabic_retranslates_to_english(self) -> None:
        locale_manager.set_language("ar")
        dialog = self._open()
        self._switch(dialog, "en")
        self._assert_matches_catalog(dialog, "en")

    def test_language_cycles_emit_no_geometry_warning(self) -> None:
        captured = []
        previous = qInstallMessageHandler(lambda *a: captured.append(a[-1]))
        try:
            dialog = self._open()
            for _ in range(3):
                for code in ("ar", "fr", "en"):
                    self._switch(dialog, code)
        finally:
            qInstallMessageHandler(previous)
        bad = [m for m in captured if "setGeometry" in m]
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
