# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter
from PySide6.QtWidgets import QFileDialog

from lynceus.plugins.locale import t
from lynceus.ui.fonts import scaled_point_size, scaled_px
from lynceus.ui.nodes.node_item import (
    LABEL_COLOR,
    NodeItem,
)
from lynceus.ui.nodes.widgets.file_queue import FileQueueMixin

BUTTON_HEIGHT = 24
LABEL_LINE_HEIGHT = 16
MISSING_COLOR = QColor("#d97a7a")  # Error state: inaccessible source.
LIDAR_FILTERS = "LiDAR files (*.laz *.las)"


class LoadLasLazNodeItem(FileQueueMixin, NodeItem):
    """Node 'Load LAS/LAZ': painted file selector for .laz / .las (UI only)."""

    file_changed = Signal()

    def __init__(
        self,
        node_id: str,
        node_name: str,
        category: str,
        inputs: tuple[str, ...] = (),
        outputs: tuple[str, ...] = (),
        parent=None,
    ):
        super().__init__(node_id, node_name, category, inputs, outputs, parent)
        self._queue_init()
        self._file_path: str | None = None
        self._file_name: str | None = None
        self._file_missing = False
        self._tile_count: int | None = None

    def content_height(self) -> float:
        bh, ll = scaled_px(BUTTON_HEIGHT), scaled_px(LABEL_LINE_HEIGHT)
        return bh + 8 + ll * 2 + 4

    def _button_rect(self) -> QRectF:
        rect = self._content_rect()
        return QRectF(rect.x(), rect.y(), rect.width(), scaled_px(BUTTON_HEIGHT))

    def _label_rect(self) -> QRectF:
        rect = self._content_rect()
        bh, ll = scaled_px(BUTTON_HEIGHT), scaled_px(LABEL_LINE_HEIGHT)
        return QRectF(rect.x(), rect.y() + bh + 8, rect.width(), ll)

    def _tiles_rect(self) -> QRectF:
        rect = self._content_rect()
        bh, ll = scaled_px(BUTTON_HEIGHT), scaled_px(LABEL_LINE_HEIGHT)
        return QRectF(
            rect.x(),
            rect.y() + bh + 8 + ll + 4,
            rect.width(),
            ll,
        )

    def set_tile_count(self, count: int) -> None:
        self._tile_count = count
        self.update()

    def _hit_region(self, pos):
        if self._button_rect().contains(pos):
            return "button"
        return super()._hit_region(pos)

    def on_content_click(self, region) -> None:
        if region == "button":
            self._browse()

    def paint_content(self, painter: QPainter, rect: QRectF) -> None:
        hovered = self._hover_region == "button"
        self.paint_button(
            painter, self._button_rect(), t("Browse..."), hovered
        )

        text, missing = self._queue_label_text(t("No file selected"))
        painter.setPen(MISSING_COLOR if missing else LABEL_COLOR)
        label_rect = self._label_rect()
        font = QFont()
        font.setPointSize(scaled_point_size(9))
        elided = QFontMetrics(font).elidedText(
            text, Qt.TextElideMode.ElideRight, int(label_rect.width())
        )
        painter.drawText(
            label_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            elided,
        )

        painter.setFont(self._font(8))
        if self._tile_count is not None:
            painter.drawText(
                self._tiles_rect(),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                t("Tiles: {n}").format(n=self._tile_count),
            )

    def _browse_files(self) -> list[str]:
        paths, _ = QFileDialog.getOpenFileNames(
            None, t("Select LiDAR file"), "", LIDAR_FILTERS
        )
        return [str(path) for path in paths or []]

    def _queue_filters(self) -> str:
        return LIDAR_FILTERS

    def _persist_queue(self, queue: list[str]) -> None:
        # Flat config is rebuilt from state on capture: nothing to store.
        return

    def file_path(self) -> str | None:
        return self._file_path

    def file_missing(self) -> bool:
        return self._file_missing

    def set_file_path(self, path: str) -> None:
        """Single-file pick (kept for callers; delegates to the queue)."""
        self.set_file_queue([path] if path else [])

    def capture_config(self) -> dict:
        config = super().capture_config()
        if self._file_path:
            config["file_path"] = self._file_path
        if len(self.file_queue()) > 1:
            config["files"] = self.file_queue()
        return config

    def apply_config(self, config: dict) -> None:
        super().apply_config(config)
        path = config.get("file_path")
        # Re-derive the visible file state from the restored flat config
        # (no signal: restore must not mark the project modified).
        self._restore_queue(
            config.get("files"),
            path if isinstance(path, str) else None,
        )
        self.update()
