# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Reusable Qt widgets shared across UI modules (no domain logic)."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from lynceus.plugins.locale import t

_ARROW_COLLAPSED = "\u25b8"  # ▸
_ARROW_EXPANDED = "\u25be"  # ▾


class CollapsibleSection(QWidget):
    """A widget with a clickable header that toggles content visibility.

    The header label carries the ``origTitle_en`` property so the
    translation pass (lynceus/ui/translate.py) retranslates it on language
    switch. Content is laid out through :meth:`content_layout`.
    """

    def __init__(
        self,
        title: str,
        parent: QWidget | None = None,
        *,
        initially_expanded: bool = False,
    ):
        super().__init__(parent)
        self._title_en = title
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(0)

        self._header = QHBoxLayout()
        self._header.setContentsMargins(0, 4, 0, 4)
        self._toggle_label = QLabel("")
        self._toggle_label.setObjectName("collapsibleHeader")
        self._toggle_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle_label.mousePressEvent = self._toggle
        self._header.addWidget(self._toggle_label)
        self._header.addStretch()
        layout.addLayout(self._header)

        self._content = QWidget()
        self._content.setObjectName("collapsibleContent")
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(12, 2, 0, 8)
        self._content_layout.setSpacing(4)
        layout.addWidget(self._content)

        self._collapsed = not initially_expanded
        self._update_state()

    # -- public API ---------------------------------------------------------

    def content_layout(self) -> QVBoxLayout:
        return self._content_layout

    def set_title(self, title_en: str) -> None:
        """Set the section title (English source; translated on render)."""
        self._title_en = title_en
        self._update_state()

    def set_collapsed(self, collapsed: bool) -> None:
        self._collapsed = bool(collapsed)
        self._update_state()

    def expand(self) -> None:
        self.set_collapsed(False)

    @property
    def is_expanded(self) -> bool:
        return not self._collapsed

    # -- internals ----------------------------------------------------------

    def _toggle(self, event=None) -> None:
        """Toggles content visibility; used as the header mouse handler."""
        self.set_collapsed(not self._collapsed)

    def _update_state(self) -> None:
        self._content.setVisible(not self._collapsed)
        self._toggle_label.setProperty("origTitle_en", self._title_en)
        arrow = _ARROW_COLLAPSED if self._collapsed else _ARROW_EXPANDED
        self._toggle_label.setText(f"{arrow} {t(self._title_en)}")