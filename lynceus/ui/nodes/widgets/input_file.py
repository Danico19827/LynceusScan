# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Generic UI item for product-input nodes (Browse + state label)."""

import os

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter
from PySide6.QtWidgets import QFileDialog

from lynceus.nodes.table._input_targets import input_target
from lynceus.plugins.locale import t
from lynceus.ui.nodes.node_item import LABEL_COLOR, NodeItem

BUTTON_HEIGHT = 24
LABEL_LINE_HEIGHT = 16
MISSING_COLOR = QColor("#d97a7a")  # error red: inaccessible file


class InputFileNodeItem(NodeItem):
    """Product-input item: 'Browse...' button + picked file name (UI only).

    The picked path lives in the node config (`_pipeline_config["file_path"]`),
    so it travels via `_pipeline_configs()` to `ctx` in the worker and
    persists in projects (capture_config/apply_config) without entering
    config_schema or the Inspector.
    """

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
        self._file_path: str | None = None
        self._file_name: str | None = None
        self._file_missing = False
        self._file_filters: str | None = None

    def set_file_filters(self, filters: str | None) -> None:
        """File-dialog filters from the node specs (third-party inputs)."""
        self._file_filters = filters or None

    def content_height(self) -> float:
        from lynceus.ui.fonts import scaled_px

        return scaled_px(BUTTON_HEIGHT) + 8 + scaled_px(LABEL_LINE_HEIGHT) + 4

    def _button_rect(self) -> QRectF:
        from lynceus.ui.fonts import scaled_px

        rect = self._content_rect()
        return QRectF(rect.x(), rect.y(), rect.width(), scaled_px(BUTTON_HEIGHT))

    def _label_rect(self) -> QRectF:
        from lynceus.ui.fonts import scaled_px

        rect = self._content_rect()
        return QRectF(
            rect.x(),
            rect.y() + scaled_px(BUTTON_HEIGHT) + 8,
            rect.width(),
            scaled_px(LABEL_LINE_HEIGHT),
        )

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
        from lynceus.ui.fonts import scaled_point_size

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

    def _browse(self) -> None:
        filters = self._file_filters
        if not filters:
            target = input_target(self.node_id())
            filters = target.get("file_filters") or "All files (*)"
        path, _ = QFileDialog.getOpenFileName(
            None, t("Select product file"), "", filters
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
        config = self.pipeline_config()
        config["file_path"] = path
        self.set_pipeline_config(config)
        self.file_changed.emit()
        self.update()

    def apply_config(self, config: dict) -> None:
        super().apply_config(config)
        # Re-derive the visible file state from the restored pipeline config
        # (no signal: restore must not mark the project modified).
        path = self._pipeline_config.get("file_path")
        if isinstance(path, str) and path:
            self._file_path = path
            self._file_name = os.path.basename(path)
            self._file_missing = not os.path.isfile(path)
            self.update()