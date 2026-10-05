# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Extensions dialog: disable/enable packs + live retranslation.

Regression: disabling the active language left the app stranded (dialog
never retranslated itself) and the pack must stay listed with an Enable
action so it can be turned back on.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QPushButton

from lynceus.plugins.locale import locale_manager
from lynceus.plugins.registry import manager
from lynceus.plugins.store import ConsentStore
from lynceus.ui.extensions_dialog import ExtensionsDialog
from lynceus.ui.translate import language_changed


def _write_locale_pack(root: Path) -> None:
    (root).mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "id": "xx",
                "version": "1.0.0",
                "kind": "locale",
                "display_name": "LangX",
                "author": "Test",
                "payload": {"catalog": "xx.json"},
            }
        ),
        encoding="utf-8",
    )
    (root / "xx.json").write_text(
        json.dumps({"xx": {"Language": "LangX"}}), encoding="utf-8"
    )


def _button_texts(dialog: ExtensionsDialog) -> list[str]:
    return [
        b.text()
        for b in dialog.findChildren(QPushButton)
        if b.text()
    ]


class ExtensionsDialogLanguageTests(unittest.TestCase):
    def setUp(self) -> None:
        self._app = QApplication.instance() or QApplication([])
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self._ext = tmp / "extensions"
        _write_locale_pack(self._ext / "locales" / "xx")
        self._store = ConsentStore(root=str(tmp / "appdata"))
        self._old_dir = manager._extensions_dir
        self._old_provider = manager.disabled_provider
        self._old_lang = locale_manager.language
        manager._extensions_dir = self._ext.resolve()
        manager.disabled_provider = lambda: self._store.disabled_ids()
        manager.discover()
        locale_manager.mark_dirty()
        locale_manager.set_language("xx")
        self._dialog = ExtensionsDialog(self._store)

    def tearDown(self) -> None:
        self._dialog.deleteLater()
        manager._extensions_dir = self._old_dir
        manager.disabled_provider = self._old_provider
        manager.discover()
        locale_manager.mark_dirty()
        locale_manager.set_language(self._old_lang)
        language_changed.emit(locale_manager.language)
        self._tmp.cleanup()

    def _entry(self, node_id: str) -> dict:
        for row in self._dialog._extension_entries():
            if row["node_id"] == node_id:
                return row
        self.fail(f"entry {node_id} missing from dialog")

    def _apply_main_fallback(self) -> None:
        """Mirror MainWindow._on_extensions_changed for locales."""
        locale_manager.mark_dirty()
        if locale_manager.language not in locale_manager.available():
            locale_manager.set_language("en")
            language_changed.emit(locale_manager.language)

    def test_disable_active_language_falls_back_and_row_persists(self) -> None:
        self.assertEqual(locale_manager.language, "xx")
        self._dialog._toggle_enabled(self._entry("xx"))
        self._apply_main_fallback()
        # App is back to English...
        self.assertEqual(locale_manager.language, "en")
        # ...the open dialog followed (title, actions, group header)...
        self.assertEqual(self._dialog.windowTitle(), "Extensions")
        texts = _button_texts(self._dialog)
        self.assertIn("Import extension...", texts)
        self.assertIn("Close", texts)
        # ...and the disabled pack stays listed with an Enable action.
        self._entry("xx")
        self._dialog._rebuild()
        self.assertIn("Enable", _button_texts(self._dialog))

    def test_reenable_restores_language(self) -> None:
        self._dialog._toggle_enabled(self._entry("xx"))
        self._apply_main_fallback()
        self.assertEqual(locale_manager.language, "en")
        self._dialog._toggle_enabled(self._entry("xx"))
        manager.discover()
        locale_manager.mark_dirty()
        self.assertIn("xx", locale_manager.available())
        locale_manager.set_language("xx")
        self.assertEqual(locale_manager.language, "xx")


if __name__ == "__main__":
    unittest.main()
