# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
import os

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter
from PySide6.QtWidgets import QFileDialog

from lynceus.plugins.locale import t
from lynceus.ui.fonts import scaled_point_size, scaled_px
from lynceus.ui.nodes.node_item import (
    LABEL_COLOR,
    NodeItem,
)

BUTTON_HEIGHT = 24
LABEL_LINE_HEIGHT = 16
MISSING_COLOR = QColor("#d97a7a")  # Error state: inaccessible source.


class LoadLasLazNodeItem(NodeItem):
    """Node 'Load LAS/LAZ': painted file selector for .laz / .las (UI only)."""

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

        if self._file_name:
            if self._file_missing:
                text = f"\u26a0 {self._file_name} {t('(file not found)')}"
                painter.setPen(MISSING_COLOR)
            else:
                text = f"\u2713 {self._file_name}"
                painter.setPen(LABEL_COLOR)
        else:
            text = t("No file selected")
            painter.setPen(LABEL_COLOR)
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

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            None, t("Select LiDAR file"), "", "LiDAR files (*.laz *.las)"
        )
        if path:
            self.set_file_path(path)

    def file_path(self) -> str | None:
        return self._file_path

    def file_missing(self) -> bool:
        return self._file_missing

    def set_file_path(self, path: str) -> None:
        self._file_path = path
        self._file_name = os.path.basename(path)
        self._file_missing = not os.path.isfile(path)
        self.update()

    def capture_config(self) -> dict:
        config = super().capture_config()
        if self._file_path:
            config["file_path"] = self._file_path
        return config

    def apply_config(self, config: dict) -> None:
        super().apply_config(config)
        path = config.get("file_path")
        if isinstance(path, str) and path:
            self.set_file_path(path)
