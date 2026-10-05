# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Tabular viewer for CSV statistics and GeoPackage attribute tables.

The payload prefers ``stats_csv`` and falls back to ``file``. CSV uses the
standard library; GeoPackage attributes use geopandas. Registered for
``table_csv``.
"""

import csv
from pathlib import Path

from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex
from PySide6.QtWidgets import QTableView

from lynceus.plugins.locale import t
from lynceus.ui.viewers.base import BaseViewer
from lynceus.ui.viewers.registry import register

MAX_PREVIEW_ROWS = 5000


def max_preview_rows() -> int:
    """Table rows shown in previews (Preferences > Display)."""
    try:
        from PySide6.QtCore import QSettings

        from lynceus.ui.settings_keys import (
            DISPLAY_TABLE_ROWS_KEY,
            SETTINGS_APP,
            SETTINGS_ORG,
        )

        value = int(
            QSettings(SETTINGS_ORG, SETTINGS_APP).value(
                DISPLAY_TABLE_ROWS_KEY, MAX_PREVIEW_ROWS
            )
        )
        return value if value > 0 else MAX_PREVIEW_ROWS
    except Exception:
        return MAX_PREVIEW_ROWS


class _TableModel(QAbstractTableModel):
    def __init__(self, headers: list[str], rows: list[list[str]]):
        super().__init__()
        self._headers = headers
        self._rows = rows

    def rowCount(self, parent=QModelIndex()):
        return len(self._rows)

    def columnCount(self, parent=QModelIndex()):
        return len(self._headers)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        return self._rows[index.row()][index.column()]

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return self._headers[section]
        return str(section + 1)


class TableCsvViewer(BaseViewer):
    """Read-only table with selection and frozen headers."""

    def __init__(self, kind: str, parent=None):
        super().__init__(kind, parent)
        self._table = QTableView(self)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._layout.addWidget(self._table)
        self._table.hide()

    def set_payload(self, payload) -> None:
        path = None
        if isinstance(payload, dict):
            candidate = payload.get("stats_csv") or payload.get("file")
            path = str(candidate) if candidate else ""
        if not path:
            self.show_placeholder(t("No table data available"))
            return

        suffix = Path(path).suffix.lower()
        limit = max_preview_rows()

        def load(cancelled):
            if cancelled.is_set():
                return None
            if suffix == ".gpkg":
                return self._read_gpkg(path, limit)
            return self._read_csv(path, limit, cancelled)

        def loaded(result) -> None:
            if result is None:
                return
            headers, rows = result
            if not rows:
                self.show_placeholder(t("Table has no data rows"))
                return
            self._table.setModel(_TableModel(headers, rows))
            self._table.show()
            self._placeholder.hide()
            self.setWindowTitle(self._compose_title())

        def failed(exc: str) -> None:
            self.show_placeholder(
                t("Cannot read table: {exc}").format(exc=exc)
                + (f"\n{path}" if path else "")
            )

        self._table.hide()
        self.start_async_load(load, loaded, failed)

    @staticmethod
    def _read_csv(path: str, limit: int, cancelled) -> tuple[list[str], list[list[str]]]:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = []
            for _, row in zip(range(limit + 1), reader):
                if cancelled.is_set():
                    return [], []
                rows.append(row)
        if not rows:
            return [], []
        return rows[0], rows[1 : limit + 1]

    @staticmethod
    def _read_gpkg(path: str, limit: int) -> tuple[list[str], list[list[str]]]:
        import pyogrio

        frame = pyogrio.read_dataframe(
            path, max_features=limit, read_geometry=False
        )
        headers = [str(column) for column in frame.columns]
        rows = [
            ["" if v is None else str(v) for v in rec]
            for rec in frame.itertuples(index=False, name=None)
        ]
        return headers, rows


register("table_csv", TableCsvViewer)
