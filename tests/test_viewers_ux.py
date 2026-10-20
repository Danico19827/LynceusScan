# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Viewer UX (block C): class legend, help affordance and raised limits.

GL widgets cannot be instantiated offscreen, so widget behavior is covered
through pure helpers (legend rows) and a help contract; limits are plain
constants/presets. Catalog coverage for the legend names and help sheets
is asserted here (the static scanner only sees t() literals).
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
LANGS = ["es", "ru", "it", "pt-BR", "zh", "hi", "ar", "fr", "bn"]
_PACKS = ROOT / "extensions" / "locales"


def _catalog(lang: str) -> dict:
    path = ROOT / "extensions" / "locales" / lang / f"{lang}.json"
    return json.loads(path.read_text(encoding="utf-8"))[lang]


class LegendTests(unittest.TestCase):
    def test_names_cover_palette(self) -> None:
        from lynceus.ui.viewers.point_cloud_3d import CLASS_COLORS, CLASS_NAMES

        self.assertEqual(len(CLASS_NAMES), 19)
        self.assertGreaterEqual(len(CLASS_COLORS), 19)

    def test_legend_entries(self) -> None:
        import numpy as np

        from lynceus.ui.viewers.point_cloud_3d import legend_entries

        self.assertEqual(legend_entries(None), [])
        rows = legend_entries(
            np.array([2, 2, 5, 18, 255], dtype=np.uint8))
        self.assertEqual([code for code, _, _ in rows], [2, 5, 18, 255])
        by_code = {code: (name, rgb) for code, name, rgb in rows}
        self.assertEqual(by_code[2][0], "Ground")
        self.assertEqual(by_code[255][0], "Class 255")
        self.assertEqual(by_code[255][1], (0.65, 0.65, 0.65))

    @unittest.skipUnless(
        _PACKS.is_dir(), "locale packs not present in this checkout")
    def test_legend_names_translated_everywhere(self) -> None:
        from lynceus.ui.viewers.point_cloud_3d import CLASS_NAMES

        catalogs = {lang: _catalog(lang) for lang in LANGS}
        for lang in LANGS:
            for name in CLASS_NAMES:
                with self.subTest(lang=lang, key=name):
                    self.assertIn(name, catalogs[lang])
                    self.assertNotEqual(catalogs[lang][name], name)


class HelpContractTests(unittest.TestCase):
    def _texts(self) -> list[str]:
        from lynceus.ui.viewers.point_cloud_3d import PointCloud3DViewer
        from lynceus.ui.viewers.raster_2d import Raster2DViewer

        return [
            PointCloud3DViewer.help_text(None),
            Raster2DViewer.help_text(None),
        ]

    def test_help_sheets_present(self) -> None:
        for text in self._texts():
            self.assertTrue(text)
            self.assertIn("<br>", text)

    @unittest.skipUnless(
        _PACKS.is_dir(), "locale packs not present in this checkout")
    def test_help_sheets_translated_everywhere(self) -> None:
        catalogs = {lang: _catalog(lang) for lang in LANGS}
        for lang in LANGS:
            for text in self._texts():
                with self.subTest(lang=lang):
                    self.assertIn(text, catalogs[lang])
                    self.assertNotEqual(catalogs[lang][text], text)

    def test_help_button_installed_only_with_help(self) -> None:
        from PySide6.QtWidgets import QApplication, QHBoxLayout, QWidget

        from lynceus.ui.viewers.base import BaseViewer

        QApplication.instance() or QApplication([])

        class Helped(BaseViewer):
            def help_text(self):
                return "x"

        host = QWidget()
        layout = QHBoxLayout(host)
        plain = BaseViewer("raster")
        plain.install_help_button(layout)
        self.assertEqual(layout.count(), 0)
        helped = Helped("raster")
        helped.install_help_button(layout)
        self.assertEqual(layout.count(), 1)
        button = layout.itemAt(0).widget()
        self.assertEqual(button.text(), "?")
        self.assertEqual(button.property("origText_en"), "?")
        # The global QPushButton padding (12px sides) leaves zero content
        # width on a 24px square: the objectName carries the tight rule.
        self.assertEqual(button.objectName(), "viewerHelpButton")
        self.assertEqual((button.width(), button.height()), (24, 24))
        host.deleteLater()
        plain.deleteLater()
        helped.deleteLater()

    def test_legend_position_top_right(self) -> None:
        from PySide6.QtCore import QPoint

        from lynceus.ui.viewers.point_cloud_3d import _legend_position

        self.assertEqual(
            _legend_position(800, 600, 120, 200), QPoint(672, 8))
        # Never negative on tiny viewports.
        self.assertEqual(
            _legend_position(50, 50, 120, 200), QPoint(0, 8))

    def test_progress_signals_survive_huge_counts(self) -> None:
        """Byte/point counts past INT_MAX must not overflow the binding."""
        from PySide6.QtWidgets import QApplication

        from lynceus.ui.viewers.point_cloud_3d import _GLView

        QApplication.instance() or QApplication([])
        view = _GLView()
        try:
            got: list[tuple[float, float]] = []
            view.upload_progress.connect(lambda d, t: got.append((d, t)))
            view.upload_progress.emit(2_200_000_000, 2_400_000_000)
            self.assertEqual(got, [(2_200_000_000, 2_400_000_000)])
        finally:
            view.deleteLater()


class LimitTests(unittest.TestCase):
    def test_point_budget_ceiling_raised(self) -> None:
        from lynceus.ui.viewers.point_cloud_3d import (
            MAX_DISPLAY_POINT_BUDGET,
            MIN_DISPLAY_POINT_BUDGET,
        )

        self.assertEqual(MAX_DISPLAY_POINT_BUDGET, 100_000_000)
        self.assertEqual(MIN_DISPLAY_POINT_BUDGET, 750_000)

    def test_display_presets_raised(self) -> None:
        from lynceus.ui.preferences_dialog import (
            BUDGET_PRESETS,
            RASTER_CELL_PRESETS,
            TABLE_ROW_PRESETS,
        )

        self.assertEqual(max(v for _, v in BUDGET_PRESETS if v != "auto"),
                         100_000_000)
        self.assertEqual(max(v for _, v in RASTER_CELL_PRESETS),
                         32_000_000)
        self.assertEqual(max(v for _, v in TABLE_ROW_PRESETS), 100_000)


if __name__ == "__main__":
    unittest.main()
