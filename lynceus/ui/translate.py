# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Qt-side translation pass (exact-match catalogs, no tr() retrofit).

Walks a widget tree and translates static texts (labels, buttons, group
titles, tooltips, menus/actions, tabs, placeholders, combo items) using the
LocaleManager exact-match catalog. Strings absent from the catalog stay
untouched, so dynamic text (paths, status) is never broken.

The module-level `language_changed` signal is emitted after the active
language changes; connected widgets must re-run the pass and re-render
dynamic surfaces (node library, inspector, canvas items).
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractButton,
    QComboBox,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMenu,
    QTabWidget,
    QWidget,
)

from lynceus.plugins.locale import t


class _LanguageBridge(QObject):
    changed = Signal(str)


_bridge = _LanguageBridge()
language_changed = _bridge.changed
"""Bound signal emitted with the new language code; UI must re-render."""


ORIG_ITEM_ROLE = Qt.ItemDataRole.UserRole + 100
"""Item-data role holding a combo entry's English source text.

QComboBox user data lives in UserRole; this separate role pins the
English source next to it, so the translate pass never has to guess the
original from the currently displayed (possibly already translated) text.
"""


def set_combo_items(combo: QComboBox, items: list) -> None:
    """Populate a combo from English ``(text, data)`` sources.

    Stores the English text per index, shows the translated text, preserves
    the current selection by data and blocks signals while rebuilding so no
    side effect (device reloads, config writes) fires mid-rebuild.
    """
    previous = combo.currentData()
    combo.blockSignals(True)
    try:
        combo.clear()
        for english, data in items:
            combo.addItem(t(english), data)
            combo.setItemData(
                combo.count() - 1, english, ORIG_ITEM_ROLE
            )
        restored = -1
        if previous is not None:
            restored = combo.findData(previous)
        combo.setCurrentIndex(restored if restored >= 0 else 0)
    finally:
        combo.blockSignals(False)


def _translate_action(action) -> None:
    text = action.text()
    if text:
        bare = text.replace("&", "")
        orig = action.property("origText_en")
        if orig is None:
            target = action.menu()
            if target is not None:
                # Representative action of a submenu: derive the source English
                # title from the menu title, never from a possibly-translated
                # current text (would strand the item in the old language).
                orig = _orig(target, "origTitle_en", target.title())
                action.setProperty("origText_en", orig)
            else:
                action.setProperty("origText_en", bare)
                orig = bare
        action.setText(t(orig))
    tip = action.toolTip()
    if tip:
        orig_tip = action.property("origTip_en")
        if orig_tip is None:
            action.setProperty("origTip_en", tip)
            orig_tip = tip
        action.setToolTip(t(orig_tip))
    status = action.statusTip()
    if status:
        orig_st = action.property("origStatus_en")
        if orig_st is None:
            action.setProperty("origStatus_en", status)
            orig_st = status
        action.setStatusTip(t(orig_st))


def _orig(widget: QWidget, prop: str, current: str) -> str:
    """Return and persist a widget's original English text.

    The value is captured once so repeated language passes never translate an
    already translated string.
    """
    stored = widget.property(prop)
    if stored is None:
        widget.setProperty(prop, current)
        return current
    return stored


def _translate_one(widget: QWidget) -> None:
    if isinstance(widget, QLabel):
        if widget.text():
            text = t(_orig(widget, "origText_en", widget.text()))
            if widget.objectName() == "inspectorGroupHeader":
                text = text.upper()
            widget.setText(text)
    elif isinstance(widget, QAbstractButton):
        if widget.text():
            widget.setText(t(_orig(widget, "origText_en", widget.text())))
        tip = widget.toolTip()
        if tip:
            widget.setToolTip(t(_orig(widget, "origTip_en", tip)))
    elif isinstance(widget, QGroupBox):
        if widget.title():
            widget.setTitle(t(_orig(widget, "origTitle_en", widget.title())))
    elif isinstance(widget, QLineEdit):
        if widget.placeholderText():
            ph = widget.placeholderText()
            widget.setPlaceholderText(t(_orig(widget, "origPh_en", ph)))
    elif isinstance(widget, QTabWidget):
        for i in range(widget.count()):
            tab_text = widget.tabText(i)
            prop = f"origTab{i}_en"
            widget.setTabText(i, t(_orig(widget, prop, tab_text)))
    elif isinstance(widget, QComboBox):
        for i in range(widget.count()):
            pinned = widget.itemData(i, ORIG_ITEM_ROLE)
            if isinstance(pinned, str) and pinned:
                widget.setItemText(i, t(pinned))
                continue
            item = widget.itemText(i)
            prop = f"origItem{i}_en"
            widget.setItemText(i, t(_orig(widget, prop, item)))
    elif isinstance(widget, QMenu):
        if widget.title():
            # The menu title is rendered by the menu's representative action
            # (menuAction). Prefer its origText_en, which dynamic callers set
            # to the real English source at construction, so we never capture a
            # possibly-translated title as the "original" and strand the item.
            rep = widget.menuAction()
            orig = rep.property("origText_en")
            if orig is None:
                orig = _orig(widget, "origTitle_en", widget.title())
                rep.setProperty("origText_en", orig)
            widget.setTitle(t(orig))

    window_title = widget.windowTitle()
    if window_title:
        widget.setWindowTitle(t(_orig(widget, "origWinTitle_en", window_title)))


def translate_widget(widget: QWidget) -> None:
    """Recursively translate static texts of a widget subtree."""
    _translate_one(widget)
    for menu in widget.findChildren(QMenu):
        for action in menu.actions():
            _translate_action(action)
    for child in widget.children():
        if isinstance(child, QWidget):
            translate_widget(child)
