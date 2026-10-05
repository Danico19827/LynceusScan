# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Interface fonts: families, live apply, pack hook."""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from lynceus.ui.fonts import (
    apply_font,
    available_families,
    font_changed,
    register_font_file,
    saved_family,
)


class FontTests(unittest.TestCase):
    def setUp(self) -> None:
        self._app = QApplication.instance() or QApplication([])
        self._previous = self._app.font()

    def tearDown(self) -> None:
        self._app.setFont(self._previous)

    def test_apply_default_never_crashes(self) -> None:
        emitted: list[str] = []
        font_changed.connect(emitted.append)
        try:
            self.assertEqual(apply_font(""), "")
        finally:
            font_changed.disconnect(emitted.append)
        self.assertEqual(emitted, [""])

    def test_apply_listed_family_sticks(self) -> None:
        families = available_families()
        if not families:
            self.skipTest("offscreen platform ships no fonts")
        emitted: list[str] = []
        font_changed.connect(emitted.append)
        try:
            target = families[0]
            self.assertEqual(apply_font(target), target)
            self.assertEqual(self._app.font().family(), target)
        finally:
            font_changed.disconnect(emitted.append)
        self.assertEqual(emitted, [target])

    def test_register_invalid_file_fails_cleanly(self) -> None:
        self.assertEqual(register_font_file("/nonexistent/font.ttf"), "")

    def test_restore_returns_startup_font(self) -> None:
        from lynceus.ui import fonts as fonts_mod

        fonts_mod._original_font = None
        before = self._app.font().family()
        apply_font("DefinitelyNotAFamily123")
        self.assertEqual(apply_font(""), "")
        self.assertEqual(self._app.font().family(), before)

    def test_unusable_families_rejected(self) -> None:
        from lynceus.ui.fonts import _is_usable

        self.assertFalse(_is_usable("NoSuchFamily123"))
        self.assertFalse(_is_usable("Terminal"))
        self.assertFalse(_is_usable("Wingdings"))

    def test_saved_family_defaults_empty(self) -> None:
        # No write: only meaningful when the user never picked a font.
        # Just assert the accessor is a string (real value is user state).
        self.assertIsInstance(saved_family(), str)

    def test_scaled_point_size_math(self) -> None:
        from lynceus.ui.fonts import font_scale, scaled_point_size

        self.assertEqual(scaled_point_size(12, scale=1.0), 12)
        self.assertEqual(scaled_point_size(12, scale=1.25), 15)
        self.assertEqual(scaled_point_size(8, scale=0.7), 6)
        self.assertEqual(scaled_point_size(10, scale=0.0), 5)
        self.assertGreaterEqual(font_scale(scale=99.0), 0.5)

    def test_stylesheet_scales_font_sizes(self) -> None:
        import re

        from lynceus.plugins.theme import THEME_ROLES
        from lynceus.ui.theming import build_stylesheet, qss_template

        template = qss_template()
        scaled = build_stylesheet(template, dict(THEME_ROLES), 1.25)
        # 13px title rule scales (whitespace-tolerant: template wraps).
        self.assertIsNotNone(re.search(r"font-size:\s*16px", scaled))
        self.assertIsNone(re.search(r"font-size:\s*13px", scaled))
        # Scale applies to palette output too, not just the template.
        self.assertIn("#eceef4", build_stylesheet(template, {
            **THEME_ROLES, "background": "#eceef4",
        }, 1.5))


if __name__ == "__main__":
    unittest.main()
