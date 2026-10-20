# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Assign Files dialog (item 8 Fase A): folder -> node file assignment.

Pairs N file-backed nodes (in selection order) with the files found in a
folder (recursive, optional extension filter), top-down. The no-partial
rule is enforced: N != M shows a warning and disables Assign, so a run
never starts "half-assigned". It never creates nodes nor imposes flows.
"""

import os

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from lynceus.graph_clipboard import collect_files, parse_filters_extensions
from lynceus.plugins.locale import t


class AssignFilesDialog(QDialog):
    """Assign one file per selected input node, seeded from a folder scan."""

    def __init__(self, parent, items) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("Assign Files"))
        self._items = list(items)

        layout = QVBoxLayout(self)

        # Folder + filter row
        folder_row = QHBoxLayout()
        self._folder = QLineEdit(self)
        self._folder.setPlaceholderText(t("Choose a folder with the files"))
        self._browse_btn = QPushButton(t("Browse..."), self)
        self._browse_btn.clicked.connect(self._pick_folder)
        folder_row.addWidget(self._folder, 1)
        folder_row.addWidget(self._browse_btn)
        layout.addLayout(folder_row)

        filter_row = QHBoxLayout()
        self._recursive = QCheckBox(t("Recursive"), self)
        self._recursive.setChecked(True)
        self._extension_combo = QComboBox(self)
        self._extensions: list[str] = []
        union: list[str] = []
        for item in self._items:
            filters = self._item_filters(item)
            for ext in parse_filters_extensions(filters):
                if ext not in union:
                    union.append(ext)
        if union:
            spec = " ".join(f"*.{e}" for e in union)
            self._extension_combo.addItem(t("Compatible files") + f"  ({spec})", union)
            self._extensions = list(union)
        self._extension_combo.addItem(t("All files") + "  (*)", None)
        self._extension_combo.setCurrentIndex(0)
        filter_row.addWidget(self._recursive)
        filter_row.addWidget(self._extension_combo, 1)
        layout.addLayout(filter_row)

        # Preview: node (fixed order) -> assigned file
        self._table = QTableWidget(len(self._items), 2, self)
        self._table.setObjectName("assignTable")
        self._table.setHorizontalHeaderLabels([t("Node"), t("File")])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._table.setColumnWidth(0, 220)
        layout.addWidget(self._table, 1)
        for row, item in enumerate(self._items):
            cell = QTableWidgetItem(item.display_name())
            cell.setToolTip(item.node_id())
            self._table.setItem(row, 0, cell)

        self._warning = QLabel("", self)
        self._warning.setObjectName("assignWarning")
        self._warning.setWordWrap(True)
        self._warning.hide()
        layout.addWidget(self._warning)

        # Apply / Cancel
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self._assign_btn = QPushButton(t("Assign"), self)
        self._assign_btn.clicked.connect(self._assign)
        cancel_btn = QPushButton(t("Cancel"), self)
        cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(self._assign_btn)
        buttons.addWidget(cancel_btn)
        layout.addLayout(buttons)

        self._folder.textChanged.connect(self._rescan)
        self._recursive.toggled.connect(self._rescan)
        self._extension_combo.currentIndexChanged.connect(self._rescan)
        self._rescan()

    # ---------- helpers ----------

    @staticmethod
    def _item_filters(item) -> str:
        try:
            filters = item._queue_filters()
        except Exception:
            return ""
        return filters or ""

    def _pick_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, t("Assign Files"), self._folder.text() or ""
        )
        if path:
            self._folder.setText(path)

    def _current_extensions(self):
        data = self._extension_combo.currentData()
        return list(data) if isinstance(data, (list, tuple)) else None

    def _rescan(self, *_args) -> None:
        folder = self._folder.text().strip()
        files = collect_files(
            folder, self._current_extensions(), self._recursive.isChecked()
        )
        count = len(self._items)
        for row, item in enumerate(self._items):
            path = files[row] if row < len(files) else ""
            cell = QTableWidgetItem(os.path.basename(path) if path else "")
            cell.setToolTip(path)
            self._table.setItem(row, 1, cell)

        matched = count == len(files)
        self._assign_btn.setEnabled(matched and bool(files))
        if matched:
            self._warning.hide()
        else:
            self._warning.setText(
                t(
                    "{nodes} node(s) selected but {files} file(s) found — "
                    "no files will be assigned unless counts match"
                ).format(nodes=count, files=len(files))
            )
            self._warning.show()

    def _assign(self) -> None:
        folder = self._folder.text().strip()
        files = collect_files(
            folder, self._current_extensions(), self._recursive.isChecked()
        )
        if len(files) != len(self._items):
            return
        for item, path in zip(self._items, files):
            item.set_file_queue([path])
        self.accept()