# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""App-level Terms & Conditions dialog (Qt UI, first-launch gate).

Shown from main.py before the main window appears. The user must agree to
the app-level Terms (TERMS_TEXT / TERMS_VERSION) to use the software;
Decline exits the application. Acceptance is persisted by
ConsentStore.terms_accept in terms.json.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QTextEdit,
    QVBoxLayout,
)

from lynceus.plugins.locale import t


class TermsDialog(QDialog):
    """Modal first-launch gate for the app-level Terms & Conditions."""

    def __init__(self, version: str, text: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("Terms & Conditions"))
        self.setModal(True)
        self.resize(640, 540)

        layout = QVBoxLayout(self)
        title = QLabel(
            "<b>LynceusScan - Terms &amp; Conditions</b> "
            f'<span style="color:#8a93a6">v{version}</span>'
        )
        layout.addWidget(title)

        body = QTextEdit()
        body.setReadOnly(True)
        body.setPlainText(text)
        layout.addWidget(body, 1)

        hint = QLabel(
            t(
                "Installing or using this software means you accept these "
                "terms. Acceptance is recorded locally with the version and "
                "the date."
            )
        )
        hint.setWordWrap(True)
        hint.setObjectName("hintLabel")
        layout.addWidget(hint)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(t("Agree"))
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(t("Decline"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)