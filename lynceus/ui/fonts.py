# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Interface font selection (Qt).

The app never hardcodes a font family: every ``QFont()`` (widgets and
painted canvas items alike) follows the application font, and no QSS rule
sets ``font-family``. So one ``QApplication.setFont`` call re-skins all
text at once.

Families come from ``QFontDatabase`` (system fonts today, filtered to
smoothly-scalable Latin-covering faces: bitmap/symbol/display faces break
layouts or fail under DirectWrite). Future font packs (``kind=font`` with
TTF files) plug in through ``register_font_file``, which needs no other
change: the selector lists whatever families are registered.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication

from lynceus.ui.settings_keys import (
    FONT_FAMILY_DEFAULT,
    FONT_FAMILY_KEY,
    FONT_SIZE_DEFAULT,
    FONT_SIZE_KEY,
    SETTINGS_APP,
    SETTINGS_ORG,
)


class _FontBridge(QObject):
    changed = Signal(str)


_bridge = _FontBridge()
font_changed = _bridge.changed
"""Bound signal emitted with the new family ("" = startup font)."""

_original_font: QFont | None = None
"""Application font snapshotted before the first apply (restore target)."""


def available_families() -> list[str]:
    """Sorted usable font families (plus any registered pack fonts).

    Filters out dangerous families: legacy bitmap/raster fonts (Terminal,
    System, Modern, 8514oem, ...) fail under DirectWrite, symbol-only
    faces (Wingdings, ...) render no text, and oversized display faces
    blow up every dialog's minimum size. A family must be smoothly
    scalable AND cover Latin to be offered.
    """
    return sorted(f for f in QFontDatabase.families() if _is_usable(f))


def _is_usable(family: str) -> bool:
    if not QFontDatabase.isSmoothlyScalable(family):
        return False
    try:
        systems = QFontDatabase.writingSystems(family)
    except Exception:
        return False
    return QFontDatabase.WritingSystem.Latin in systems


def register_font_file(path: str) -> str:
    """Register a TTF/OTF file; returns its family name ("" on failure).

    The future ``kind=font`` packs call this at discovery; the family
    then appears in :func:`available_families` like a system font.
    """
    family_id = QFontDatabase.addApplicationFont(path)
    if family_id < 0:
        return ""
    families = QFontDatabase.applicationFontFamilies(family_id)
    return families[0] if families else ""


def saved_family() -> str:
    """Font family stored in QSettings ("" = system default)."""
    from PySide6.QtCore import QSettings

    return str(
        QSettings(SETTINGS_ORG, SETTINGS_APP).value(
            FONT_FAMILY_KEY, FONT_FAMILY_DEFAULT
        )
        or FONT_FAMILY_DEFAULT
    )


def saved_size() -> int:
    """Font size percent stored in QSettings (clamped 70-150)."""
    from PySide6.QtCore import QSettings

    try:
        saved = int(
            QSettings(SETTINGS_ORG, SETTINGS_APP).value(
                FONT_SIZE_KEY, FONT_SIZE_DEFAULT
            )
            or FONT_SIZE_DEFAULT
        )
    except (TypeError, ValueError):
        saved = int(FONT_SIZE_DEFAULT)
    return min(150, max(70, saved))


def font_scale(scale: float | None = None) -> float:
    """Interface font scale factor (1.0 = theme size).

    Read live from QSettings unless overridden (tests). Painted canvas
    text and QSS `font-size` rules multiply by this; world-anchored canvas
    geometry does not move, only glyphs grow.
    """
    if scale is not None:
        return max(0.5, min(2.0, scale))
    return max(0.5, min(2.0, saved_size() / 100.0))


def scaled_point_size(base: int | float, scale: float | None = None) -> int:
    """Integer point size for `setPointSize` (Qt aborts on float-for-int)."""
    return max(1, round(float(base) * font_scale(scale)))


def scaled_px(base: int | float, scale: float | None = None) -> int:
    """Layout pixels that grow with the interface font (buttons, rows)."""
    return max(1, round(float(base) * font_scale(scale)))


def apply_font(family: str | None = None) -> str:
    """Apply a font family live ("" restores the startup font).

    Returns the active family. The original application font is snapshotted
    on the first call: restoring means putting that object back, because
    ``setFont(QFont())`` does NOT reliably reset to the system default.
    Repolishes live widgets like themes do.
    """
    global _original_font
    active = family if family is not None else saved_family()
    app = QApplication.instance()
    if app is not None:
        if _original_font is None:
            _original_font = QFont(app.font())
        app.setFont(QFont(active) if active else QFont(_original_font))
        style = app.style()
        for widget in app.allWidgets():
            style.unpolish(widget)
            style.polish(widget)
            widget.update()
    font_changed.emit(active)
    return active
