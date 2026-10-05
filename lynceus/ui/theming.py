# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Live theme application (Qt).

A theme pack only overrides color roles (see lynceus/plugins/theme.py);
this module turns the active palette into the running interface:

- QSS: ``app.qss`` is the template. Every hex it uses maps to exactly one
  role (``_ROLE_FOR_HEX``); semantic one-offs (status greens/ambers/reds,
  notice chips) stay fixed by design, like the node STATUS colors.
  Regenerating the default palette reproduces the shipped QSS byte for
  byte (guarded by tests).
- QPalette: fully derived from roles (covers what QSS does not style).
- Painted constants: module-level QColors in node_item/canvas are mutated
  in place (``setNamedColor``), so existing bindings (including
  ``from node_item import LABEL_COLOR`` in widgets) follow without
  re-imports; scenes repaint via the ``theme_changed`` signal.

``THEME_COLORS`` in branding.py is the live view the splash/chrome read.
"""

from __future__ import annotations

import re

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from lynceus.plugins.theme import DEFAULT_THEME_ID, theme_manager
from lynceus.resources import resource_path
from lynceus.ui.settings_keys import (
    SETTINGS_APP,
    SETTINGS_ORG,
    THEME_DEFAULT,
    THEME_KEY,
)

QSS_TEMPLATE_PATH = str(resource_path("lynceus/ui/styles/app.qss"))

# Every hex used by app.qss, mapped to its role. Semantic one-offs that
# must NOT follow the theme (status chips, notice blocks, viewer ramps)
# are absent here on purpose and stay fixed.
_ROLE_FOR_HEX = {
    # Deep backgrounds.
    "#0b0f1a": "background",
    "#12151f": "background",
    "#161b28": "background",
    # Surfaces.
    "#1c2333": "surface",
    "#1b2233": "surface",
    # Borders.
    "#2f3a4f": "border",
    "#3a465c": "border_strong",
    # Primary text (incl. bright variants and tree selection text).
    "#e6e9f0": "text",
    "#ffffff": "text",
    "#c9d1e3": "text",
    "#a3bfe0": "text",
    "#cbd3e0": "text",
    # Body text.
    "#aeb7c6": "text_body",
    # Secondary text (incl. dim variants).
    "#8a93a6": "muted",
    "#5d6a82": "muted",
    "#48546a": "muted",
    # Accent (incl. hover fills).
    "#7d9fd4": "accent",
    "#9db8e3": "accent",
    # Warm accent / amber.
    "#e08a8a": "danger",
    "#c9a86a": "amber",
}

# `selection-color: #ffffff` must stay white on the accent fill in every
# theme (dark text on mid-blue reads poorly), so it is stashed before the
# generic #ffffff -> text substitution and restored after.
_SELECTION_SENTINEL = "\ue000SELECTION\ue001"


class _ThemeBridge(QObject):
    changed = Signal(str)


_bridge = _ThemeBridge()
theme_changed = _bridge.changed
"""Bound signal emitted with the new theme id; scenes must repaint."""


def saved_theme_id() -> str:
    """Theme id stored in QSettings (default when unset)."""
    from PySide6.QtCore import QSettings

    return str(
        QSettings(SETTINGS_ORG, SETTINGS_APP).value(THEME_KEY, THEME_DEFAULT)
        or THEME_DEFAULT
    ).lower()


def active_palette(theme_id: str | None = None) -> dict[str, str]:
    """Palette for `theme_id` (or the saved one), defaults-merged."""
    tid = (theme_id or saved_theme_id()).lower()
    theme_manager.set_theme(tid)
    return theme_manager.palette()


def build_stylesheet(
    template: str, palette: dict[str, str], font_scale: float = 1.0
) -> str:
    """Regenerate the QSS template with palette roles + font scale.

    `font_scale` multiplies every integer `font-size: Npx` rule
    (`max(1, round(...))`); 1.0 reproduces the template byte for byte.
    """
    stashed = template.replace(
        "selection-color: #ffffff", f"selection-color: {_SELECTION_SENTINEL}"
    ).replace(
        "selection-color: #FFFFFF", f"selection-color: {_SELECTION_SENTINEL}"
    )

    def _replace(match: re.Match) -> str:
        role = _ROLE_FOR_HEX.get(match.group(0).lower())
        if role is None:
            return match.group(0)
        return palette.get(role, match.group(0))

    out = re.sub(r"#[0-9a-fA-F]{6}", _replace, stashed)
    if font_scale != 1.0:
        out = _scale_font_sizes(out, font_scale)
    return out.replace(_SELECTION_SENTINEL, "#ffffff")


def _scale_font_sizes(css: str, scale: float) -> str:
    """Scale integer `font-size: Npx` rules, flooring at 1px."""

    def _size(match: re.Match) -> str:
        try:
            base = int(match.group(1))
        except ValueError:
            return match.group(0)
        return f"font-size: {max(1, round(base * scale))}px"

    return re.sub(r"font-size:\s*(\d+)px", _size, css)


def qss_template() -> str:
    """Raw app.qss template text."""
    with open(QSS_TEMPLATE_PATH, encoding="utf-8") as qss_file:
        return qss_file.read()


def apply_qpalette(app: QApplication, palette: dict[str, str]) -> None:
    """QPalette fully derived from roles (what QSS does not style)."""
    qpal = QPalette()
    get = palette.get
    qpal.setColor(QPalette.ColorRole.Window, QColor(get("surface")))
    qpal.setColor(QPalette.ColorRole.Base, QColor(get("background")))
    base = QColor(get("background"))
    alternate = (
        base.lighter(108) if _is_dark(base) else base.darker(106)
    )
    qpal.setColor(QPalette.ColorRole.AlternateBase, alternate)
    qpal.setColor(QPalette.ColorRole.ToolTipBase, QColor(get("surface")))
    qpal.setColor(QPalette.ColorRole.ToolTipText, QColor(get("text")))
    qpal.setColor(QPalette.ColorRole.Text, QColor(get("text")))
    qpal.setColor(QPalette.ColorRole.WindowText, QColor(get("text")))
    qpal.setColor(QPalette.ColorRole.ButtonText, QColor(get("text")))
    qpal.setColor(QPalette.ColorRole.Button, QColor(get("surface")))
    qpal.setColor(QPalette.ColorRole.Highlight, QColor(get("border")))
    qpal.setColor(
        QPalette.ColorRole.HighlightedText, QColor(get("text"))
    )
    qpal.setColor(QPalette.ColorRole.Link, QColor(get("accent")))
    qpal.setColor(
        QPalette.ColorRole.PlaceholderText, QColor(get("muted"))
    )
    for role in (
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
        QPalette.ColorRole.WindowText,
    ):
        qpal.setColor(
            QPalette.ColorGroup.Disabled, role, QColor(get("muted"))
        )
    app.setPalette(qpal)


def _is_dark(color: QColor) -> bool:
    luminance = (
        0.2126 * color.red() + 0.7152 * color.green() + 0.0722 * color.blue()
    )
    return luminance < 128


def refresh_painted(palette: dict[str, str]) -> None:
    """Mutate painted QColor constants in place (bindings follow)."""
    from lynceus.ui import branding
    from lynceus.ui.nodes import canvas as canvas_mod
    from lynceus.ui.nodes import node_item as node_mod

    get = palette.get
    node_map = {
        "FILL": "surface",
        "BORDER": "border",
        "BORDER_SELECTED": "accent",
        "TITLE_COLOR": "text",
        "CATEGORY_COLOR": "muted",
        "LABEL_COLOR": "text_body",
        "DOT_FILL": "muted",
        "DOT_BORDER": "border",
        "DOT_HOVER": "accent",
        "CLOSE_COLOR": "muted",
        "CLOSE_HOVER": "danger",
        "BUTTON_FILL": "border",
        "BUTTON_FILL_HOVER": "border_strong",
        "BUTTON_TEXT": "text",
        "PULSE_BORDER": "accent",
    }
    for name, role in node_map.items():
        color = getattr(node_mod, name, None)
        if isinstance(color, QColor):
            color.setNamedColor(get(role))
    canvas_map = {
        "WIRE_COLOR": "wire",
        "WIRE_GHOST_COLOR": "border",
        "CANVAS_BG": "canvas_bg",
    }
    for name, role in canvas_map.items():
        color = getattr(canvas_mod, name, None)
        if isinstance(color, QColor):
            color.setNamedColor(get(role))

    branding.THEME_COLORS.update(
        {
            "splash_bg": get("background"),
            "surface": get("surface"),
            "border": get("border"),
            "text": get("text"),
            "muted": get("muted"),
            "accent": get("accent"),
            "canvas_bg": get("canvas_bg"),
        }
    )


def apply_theme(theme_id: str | None = None) -> dict[str, str]:
    """Apply a theme live: QSS + QPalette + painted constants + signal.

    Returns the active palette. No-op safely before QApplication exists
    (stylesheet/palette steps are skipped; constants still refresh).
    """
    palette = active_palette(theme_id)
    app = QApplication.instance()
    if app is not None:
        from lynceus.ui.branding import app_icon
        from lynceus.ui.fonts import font_scale

        app.setStyleSheet(
            build_stylesheet(qss_template(), palette, font_scale())
        )
        apply_qpalette(app, palette)
        # All windows inherit the app icon: retint it with the theme text
        # color (white artwork is invisible on light title bars).
        app.setWindowIcon(app_icon(QColor(palette["text"])))
        # Re-polish every live widget: without this, QSS with object-name
        # selectors does not re-apply to already-styled widgets and whole
        # areas (prefs nav, gallery rows, notice tray) lag one theme behind.
        style = app.style()
        for widget in app.allWidgets():
            style.unpolish(widget)
            style.polish(widget)
            widget.update()
    refresh_painted(palette)
    theme_changed.emit(theme_manager.theme)
    return palette


def available_themes() -> list[tuple[str, str]]:
    """(id, display label) pairs for the theme selector, default first."""
    theme_manager.mark_dirty()
    ids = theme_manager.available()
    if DEFAULT_THEME_ID in ids:
        ids.remove(DEFAULT_THEME_ID)
    return [(DEFAULT_THEME_ID, "Default")] + [
        (tid, theme_manager.display_name(tid)) for tid in ids
    ]
