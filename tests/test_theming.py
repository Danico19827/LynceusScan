# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Live theme application: QSS regeneration and painted constants."""

import os
import re
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from lynceus.plugins.theme import THEME_ROLES, theme_manager
from lynceus.ui.theming import (
    _ROLE_FOR_HEX,
    apply_theme,
    available_themes,
    build_stylesheet,
    qss_template,
    theme_changed,
)

ROOT = Path(__file__).resolve().parent.parent
_PACKS = ROOT / "extensions" / "locales"

# Semantic one-offs that must NOT follow the theme (status chips, notice
# blocks, viewer ramps): documented in theming.py, frozen here so a newly
# fixed hex is a deliberate decision, never an accident.
FIXED_HEXES = frozenset({
    "#0c241a", "#151c2a", "#1f2740", "#241c0c", "#2e6a4a", "#3a2a2a",
    "#6a541e", "#7ec4a1", "#e0b45a", "#7aa8d4", "#d4a94e", "#6ec87a",
    "#d46a6a",
})


class StylesheetTests(unittest.TestCase):
    def test_default_palette_reproduces_template_byte_for_byte(self) -> None:
        template = qss_template()
        self.assertEqual(build_stylesheet(template, dict(THEME_ROLES)), template)

    def test_light_stylesheet_replaces_dark_surfaces(self) -> None:
        light = theme_manager.palette("light")
        css = build_stylesheet(qss_template(), light)
        # Dark-only hexes (light overrides every one of these roles).
        for dark_hex in ("#0b0f1a", "#2f3a4f", "#e6e9f0", "#8a93a6"):
            self.assertNotIn(dark_hex, css)
        # Light roles land (note: light text IS #1c2333, shared by design).
        for light_hex in ("#eceef4", "#ffffff", "#1c2333", "#3f6fb4"):
            self.assertIn(light_hex, css)

    def test_selection_color_and_semantics_stay_fixed(self) -> None:
        light = theme_manager.palette("light")
        css = build_stylesheet(qss_template(), light)
        self.assertIn("selection-color: #ffffff", css)
        # Semantic chips (success/error blocks) never follow the theme.
        self.assertIn("#0c241a", css)
        self.assertIn("#3a2a2a", css)

    def test_no_fixed_alpha_backgrounds(self) -> None:
        # A fixed rgba() can never follow a theme (notice tray regression).
        self.assertNotIn("rgba(", qss_template())

    def test_every_hex_is_a_theme_role_or_documented_fixed(self) -> None:
        # A hex outside _ROLE_FOR_HEX freezes across themes: light packs
        # would inherit a dark color (white-button class of bug). New fixed
        # hexes must join FIXED_HEXES deliberately.
        found = {
            match.group(0).lower()
            for match in re.finditer(r"#[0-9a-fA-F]{6}", qss_template())
        }
        self.assertEqual(found - set(_ROLE_FOR_HEX) - FIXED_HEXES, set())

    def test_native_faced_widgets_have_explicit_colors(self) -> None:
        # Buttons, checkboxes, lists and sliders render their face natively:
        # without explicit rules the face follows the OS theme while the
        # text follows the palette (white-on-white on light Windows).
        template = qss_template()
        for selector in ("QPushButton", "QCheckBox", "QListWidget",
                         "QSlider::groove:horizontal"):
            match = re.search(
                r"^" + re.escape(selector) + r"\s*\{([^}]*)\}",
                template,
                re.MULTILINE,
            )
            self.assertIsNotNone(match, selector)
            block = match.group(1)
            self.assertIn("color", block, selector)

    @staticmethod
    def _contrast_ratio(back: str, fore: str) -> float:
        def luminance(hex_color: str) -> float:
            rgb = [int(hex_color[i : i + 2], 16) / 255.0 for i in (1, 3, 5)]

            def channel(value: float) -> float:
                if value <= 0.03928:
                    return value / 12.92
                return ((value + 0.055) / 1.055) ** 2.4

            channels = [channel(v) for v in rgb]
            return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]

        lighter = luminance(back)
        darker = luminance(fore)
        if darker > lighter:
            lighter, darker = darker, lighter
        return (lighter + 0.05) / (darker + 0.05)

    def _rule_colors(self, css: str, selector: str) -> tuple[str, str]:
        match = re.search(
            r"^" + re.escape(selector) + r"\s*\{([^}]*)\}",
            css,
            re.MULTILINE,
        )
        self.assertIsNotNone(match, selector)
        block = match.group(1)
        back = re.search(r"background-color:\s*(#[0-9a-fA-F]{6})", block)
        fore = re.search(r"^\s*color:\s*(#[0-9a-fA-F]{6})", block,
                         re.MULTILINE)
        self.assertIsNotNone(back, selector)
        self.assertIsNotNone(fore, selector)
        return back.group(1), fore.group(1)

    def test_button_and_list_contrast_in_dark_and_light(self) -> None:
        # The reported bug: light text on a light native face. Explicit
        # faces must contrast AA (>= 4.5) in every shipped palette.
        for palette in (dict(THEME_ROLES),
                        theme_manager.palette("light")):
            css = build_stylesheet(qss_template(), palette)
            for selector in ("QPushButton", "QListWidget"):
                back, fore = self._rule_colors(css, selector)
                ratio = self._contrast_ratio(back, fore)
                self.assertGreaterEqual(
                    ratio, 4.5, f"{selector} {back} vs {fore}")

    def test_menu_logo_tint_matches_theme_text(self) -> None:
        import io

        import numpy as np
        from PIL import Image
        from PySide6.QtCore import QBuffer, QIODevice
        from PySide6.QtGui import QColor

        from lynceus.ui.branding import (
            ICON_SQUARE_SVG,
            render_svg,
            tint_pixmap,
        )

        for color in ("#e6e9f0", "#1c2333"):
            pixmap = tint_pixmap(
                render_svg(ICON_SQUARE_SVG, 64), QColor(color)
            )
            self.assertFalse(pixmap.isNull())
            buf = QBuffer()
            buf.open(QIODevice.OpenModeFlag.WriteOnly)
            pixmap.toImage().save(buf, "PNG")
            data = np.asarray(
                Image.open(io.BytesIO(bytes(buf.data()))).convert("RGBA"),
                dtype=np.int32,
            )
            target = np.array(
                [int(color[i : i + 2], 16) for i in (1, 3, 5)],
                dtype=np.int32,
            )
            opaque = data[..., 3] > 128
            self.assertGreater(int(opaque.sum()), 100)
            dist = np.abs(
                data[..., :3].astype(np.int32) - target
            ).sum(axis=2)
            self.assertGreater(int(((dist <= 30) & opaque).sum()), 100)


class LiveApplyTests(unittest.TestCase):
    def setUp(self) -> None:
        from PySide6.QtWidgets import QApplication

        self._app = QApplication.instance() or QApplication([])
        self._previous_theme = theme_manager.theme
        self._emitted: list[str] = []
        theme_changed.connect(self._emitted.append)

    def tearDown(self) -> None:
        theme_changed.disconnect(self._emitted.append)
        apply_theme("default")
        theme_manager.set_theme(self._previous_theme)

    def test_apply_light_repaints_constants_and_chrome(self) -> None:
        from lynceus.ui import branding
        from lynceus.ui.nodes import canvas as canvas_mod
        from lynceus.ui.nodes import node_item as node_mod

        palette = apply_theme("light")
        self.assertEqual(palette["background"], "#eceef4")
        self.assertEqual(node_mod.FILL.name(), "#ffffff")
        self.assertEqual(node_mod.TITLE_COLOR.name(), "#1c2333")
        self.assertEqual(node_mod.BORDER_SELECTED.name(), "#3f6fb4")
        self.assertEqual(canvas_mod.CANVAS_BG.name(), "#dfe3ec")
        self.assertEqual(canvas_mod.WIRE_COLOR.name(), "#5d6a82")
        self.assertEqual(branding.THEME_COLORS["splash_bg"], "#eceef4")
        self.assertEqual(branding.THEME_COLORS["accent"], "#3f6fb4")
        self.assertIn("#eceef4", self._app.styleSheet())
        self.assertEqual(self._emitted, ["light"])

    def test_available_themes_lists_official_packs(self) -> None:
        ids = dict(available_themes())
        self.assertEqual(ids["default"], "Default")
        self.assertEqual(ids["dark"], "Dark")
        self.assertEqual(ids["light"], "Light")

    def test_app_icon_follows_theme_text(self) -> None:
        import io

        import numpy as np
        from PIL import Image
        from PySide6.QtCore import QBuffer, QIODevice
        from PySide6.QtGui import QColor

        from lynceus.ui.branding import app_icon

        for color in ("#e6e9f0", "#1c2333"):
            icon = app_icon(QColor(color))
            pixmap = icon.pixmap(64, 64)
            self.assertFalse(pixmap.isNull())
            target = np.array(
                [int(color[i : i + 2], 16) for i in (1, 3, 5)],
                dtype=np.int32,
            )
            buf = QBuffer()
            buf.open(QIODevice.OpenModeFlag.WriteOnly)
            pixmap.toImage().save(buf, "PNG")
            data = np.asarray(
                Image.open(io.BytesIO(bytes(buf.data()))).convert("RGBA"),
                dtype=np.int32,
            )
            opaque = data[..., 3] > 128
            self.assertGreater(int(opaque.sum()), 100)
            dist = np.abs(
                data[..., :3].astype(np.int32) - target
            ).sum(axis=2)
            self.assertGreater(int(((dist <= 30) & opaque).sum()), 100)

    @unittest.skipUnless(
        _PACKS.is_dir(), "locale packs not present in this checkout")
    def test_prefs_dialog_retranslates_while_open(self) -> None:
        from lynceus.plugins.locale import locale_manager
        from lynceus.ui.preferences_dialog import PreferencesDialog
        from lynceus.ui.translate import language_changed

        dialog = PreferencesDialog()
        self.assertEqual(dialog.windowTitle(), "Preferences")
        previous = locale_manager.language
        try:
            locale_manager.set_language("es")
            language_changed.emit("es")
            self.assertEqual(dialog.windowTitle(), "Preferencias")
            self.assertEqual(dialog.nav.item(1).text(), "Rendimiento")
            locale_manager.set_language("en")
            language_changed.emit("en")
            self.assertEqual(dialog.windowTitle(), "Preferences")
            self.assertEqual(dialog.nav.item(1).text(), "Performance")
        finally:
            locale_manager.set_language(previous)
            language_changed.emit(locale_manager.language)
            dialog.deleteLater()

    def test_font_combo_items_carry_explicit_sizes(self) -> None:
        from PySide6.QtCore import Qt

        from lynceus.ui.preferences_dialog import PreferencesDialog

        dialog = PreferencesDialog()
        try:
            page = dialog._pages[0][1]
            combo = page.font
            self.assertGreaterEqual(combo.count(), 1)
            for row in range(combo.count()):
                role = combo.itemData(row, Qt.ItemDataRole.FontRole)
                if role is None:
                    continue  # "System default" entry
                self.assertGreater(role.pointSize(), 0)
        finally:
            dialog.deleteLater()

    def test_apply_button_saves_pages_without_closing(self) -> None:
        from PySide6.QtWidgets import QDialog, QDialogButtonBox

        from lynceus.ui.preferences_dialog import PreferencesDialog

        calls: list[str] = []
        dialog = PreferencesDialog(
            on_apply=lambda: calls.append("applied")
        )
        saved: list[str] = []
        for _label, page in dialog._pages:
            name = type(page).__name__
            page.save = lambda n=name: saved.append(n)  # type: ignore[method-assign]
        apply_btn = dialog._buttons.button(QDialogButtonBox.StandardButton.Apply)
        self.assertEqual(apply_btn.property("origText_en"), "Apply")
        apply_btn.click()
        self.assertEqual(
            sorted(saved),
            sorted(type(page).__name__ for _label, page in dialog._pages),
        )
        self.assertEqual(calls, ["applied"])
        # Apply saves + notifies but never closes the dialog.
        self.assertEqual(dialog.result(), 0)
        dialog.reject()
        self.assertEqual(
            dialog.result(), QDialog.DialogCode.Rejected.value
        )


if __name__ == "__main__":
    unittest.main()
