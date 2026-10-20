# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Localized message dialogs (single choke point for stock buttons).

Qt renders stock button texts (OK/Cancel/Yes/No/...) from its own locale,
ignoring the app catalog: every stock button in the app goes through here
so no English built-in ever reaches the user. Dialogs with custom labels
(preflight Continue, terms Agree/Decline, extensions Accept/Decline, ...)
keep their own setText calls and never touch this module.
"""

from __future__ import annotations

from PySide6.QtWidgets import QDialogButtonBox, QMessageBox

from lynceus.plugins.locale import t

_BUTTON_KEYS = (
    ("Ok", "OK"),
    ("Cancel", "Cancel"),
    ("Yes", "Yes"),
    ("No", "No"),
    ("Save", "Save"),
    ("Discard", "Discard"),
    ("Apply", "Apply"),
    ("Close", "Close"),
    ("Open", "Open"),
)
"""Stock role name (Qt spelling) to catalog key, in a fixed order."""


def localize_buttons(box, *roles: str) -> None:
    """Override the named stock buttons with catalog texts.

    Roles use the Qt spelling (``"Ok"``, not ``"OK"``); the catalog keys
    are fixed by :data:`_BUTTON_KEYS`. Each button also pins
    ``origText_en`` so the retranslate pass survives language switches.
    Unknown roles raise ValueError (a typo must fail loud, never render
    half-translated). Boxes keeping custom labels must NOT call this:
    role lookup would overwrite their text with the stock one.
    """
    if isinstance(box, QMessageBox):
        enum = QMessageBox.StandardButton
    elif isinstance(box, QDialogButtonBox):
        enum = QDialogButtonBox.StandardButton
    else:
        raise TypeError(
            "localize_buttons expects QMessageBox or QDialogButtonBox, "
            f"got {type(box).__name__}"
        )
    known = dict(_BUTTON_KEYS)
    for role in roles:
        if role not in known:
            raise ValueError(f"Unknown stock role: {role!r}")
        button = box.button(getattr(enum, role))
        if button is not None:
            button.setText(t(known[role]))
            button.setProperty("origText_en", known[role])


def _stock_box(parent, icon, title: str, text: str,
               buttons) -> QMessageBox:
    """QMessageBox instance mirroring the static convenience calls."""
    return QMessageBox(icon, title, text, buttons, parent)


def show_info(parent, title: str, text: str) -> None:
    """Information box with a localized OK (return discarded, as usual)."""
    box = _stock_box(parent, QMessageBox.Icon.Information, title, text,
                     QMessageBox.StandardButton.Ok)
    localize_buttons(box, "Ok")
    box.exec()


def show_warning(parent, title: str, text: str) -> None:
    """Warning box with a localized OK (return discarded, as usual)."""
    box = _stock_box(parent, QMessageBox.Icon.Warning, title, text,
                     QMessageBox.StandardButton.Ok)
    localize_buttons(box, "Ok")
    box.exec()


def show_critical(parent, title: str, text: str) -> None:
    """Critical box with a localized OK (return discarded, as usual)."""
    box = _stock_box(parent, QMessageBox.Icon.Critical, title, text,
                     QMessageBox.StandardButton.Ok)
    localize_buttons(box, "Ok")
    box.exec()


def ask_yes_no(parent, title: str, text: str) -> bool:
    """Yes/No question; True only when Yes is pressed (both localized)."""
    box = _stock_box(
        parent, QMessageBox.Icon.Question, title, text,
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
    )
    localize_buttons(box, "Yes", "No")
    return box.exec() == QMessageBox.StandardButton.Yes


def ask_save_discard_cancel(parent, title: str, text: str):
    """Save/Discard/Cancel question; returns the pressed StandardButton."""
    box = _stock_box(
        parent, QMessageBox.Icon.Warning, title, text,
        QMessageBox.StandardButton.Save
        | QMessageBox.StandardButton.Discard
        | QMessageBox.StandardButton.Cancel,
    )
    localize_buttons(box, "Save", "Discard", "Cancel")
    return box.exec()
