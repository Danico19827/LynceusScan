# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""File-queue manager dialog (Qt): reorderable per-node file list."""

import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

from lynceus.nodes import _file_queue as fq
from lynceus.plugins.locale import t
from lynceus.ui.dialogs import localize_buttons


class FileQueueDialog(QDialog):
    """Add/remove/reorder the files one input node processes in sequence."""

    def __init__(self, parent, queue: list[str], file_filters: str) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("Manage queue..."))
        self._filters = file_filters or "All files (*)"

        layout = QVBoxLayout(self)
        self._list = QListWidget(self)
        self._list.setObjectName("queueList")
        self._list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        layout.addWidget(self._list)

        buttons = QHBoxLayout()
        self._add_btn = QPushButton(t("Add files..."), self)
        self._folder_btn = QPushButton(t("Add from folder..."), self)
        self._remove_btn = QPushButton(t("Remove"), self)
        self._up_btn = QPushButton(t("Move up"), self)
        self._down_btn = QPushButton(t("Move down"), self)
        self._add_btn.clicked.connect(self._add_files)
        self._folder_btn.clicked.connect(self._add_folder)
        self._remove_btn.clicked.connect(self._remove_selected)
        self._up_btn.clicked.connect(lambda: self._move_selected(-1))
        self._down_btn.clicked.connect(lambda: self._move_selected(1))
        for btn in (
            self._add_btn, self._folder_btn, self._remove_btn, self._up_btn,
            self._down_btn,
        ):
            buttons.addWidget(btn)
        layout.addLayout(buttons)

        close_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Close, self)
        close_box.rejected.connect(self.reject)
        localize_buttons(close_box, "Close")
        layout.addWidget(close_box)

        for path in fq.normalize_queue(queue):
            self._add_row(path)
        self._refresh_buttons()
        self._list.itemSelectionChanged.connect(self._refresh_buttons)

    # ---------- rows ----------

    def _add_row(self, path: str) -> None:
        name = os.path.basename(path) or path
        if os.path.isfile(path):
            text = name
        else:
            text = f"\u26a0 {name} {t('(file not found)')}"
        row = QListWidgetItem(text, self._list)
        row.setData(Qt.ItemDataRole.UserRole, path)
        row.setToolTip(path)

    def queue(self) -> list[str]:
        """Full paths in list order."""
        return [
            str(self._list.item(i).data(Qt.ItemDataRole.UserRole))
            for i in range(self._list.count())
        ]

    # ---------- actions ----------

    def _add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, t("Manage queue..."), "", self._filters)
        if not paths:
            return
        current = self.queue()
        updated, _added = fq.add_to_queue(current, paths)
        if len(updated) != len(current):
            self._list.clear()
            for path in updated:
                self._add_row(path)
        self._refresh_buttons()

    def _add_folder(self) -> None:
        """Append every matching file below a picked folder (recursive)."""
        from lynceus.graph_clipboard import collect_files, parse_filters_extensions

        directory = QFileDialog.getExistingDirectory(
            self, t("Add from folder..."), "")
        if not directory:
            return
        extensions = parse_filters_extensions(self._filters)
        paths = collect_files(directory, extensions or None, recursive=True)
        if not paths:
            self._refresh_buttons()
            return
        current = self.queue()
        updated, _added = fq.add_to_queue(current, paths)
        if len(updated) != len(current):
            self._list.clear()
            for path in updated:
                self._add_row(path)
        self._refresh_buttons()

    def _remove_selected(self) -> None:
        row = self._list.currentRow()
        if row < 0:
            return
        self._list.takeItem(row)
        self._refresh_buttons()

    def _move_selected(self, delta: int) -> None:
        row = self._list.currentRow()
        target = row + delta
        if row < 0 or not 0 <= target < self._list.count():
            return
        row_item = self._list.takeItem(row)
        self._list.insertItem(target, row_item)
        self._list.setCurrentRow(target)
        self._refresh_buttons()

    def _refresh_buttons(self) -> None:
        row = self._list.currentRow()
        count = self._list.count()
        has_row = 0 <= row < count
        self._remove_btn.setEnabled(has_row)
        self._up_btn.setEnabled(has_row and row > 0)
        self._down_btn.setEnabled(has_row and row < count - 1)
