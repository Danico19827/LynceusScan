# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""UI item for strategy nodes (variant selector + product input fields)."""

import os

from PySide6.QtCore import QPoint, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QCursor, QFont, QFontMetrics, QPainter
from PySide6.QtWidgets import QFileDialog, QMenu

from lynceus.nodes._variants import (
    STRATEGY_FIELD,
    resolved_ports,
    strategy_options,
    variant_meta,
)
from lynceus.plugins.locale import t
from lynceus.ui.fonts import scaled_point_size, scaled_px
from lynceus.ui.nodes.node_item import LABEL_COLOR, NodeItem

ROW_HEIGHT = 24
ROW_SPACING = 6
LABEL_LINE_HEIGHT = 16
SELECTOR_MAX_WIDTH = 180
MISSING_COLOR = QColor("#d97a7a")  # error red: inaccessible file


def _none_label() -> str:
    return t("(none) — no strategy")


def _cursor_pos() -> QPoint:
    return QCursor.pos()


class StrategyNodeItem(NodeItem):
    """Strategy family item: product selector + Browse + file label.

    The selected variant lives in ``_pipeline_config[STRATEGY_FIELD]`` and the
    file path in ``_pipeline_config["file_path"]``, so both travel to ``ctx``
    in the worker and persist in projects. Ports are rebuilt from the selected
    variant (``resolved_ports``), matching the engine's per-instance contract.

    The on-node menu only *requests* a change
    (``strategy_change_requested``); the canvas is the single mutator
    (``set_strategy``), so every path (canvas menu, Inspector) cuts cables,
    rebuilds ports and refreshes the other view consistently.
    """

    strategy_change_requested = Signal(str)
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

    # ---------- strategy state ----------

    def strategy(self) -> str:
        return str(self._pipeline_config.get(STRATEGY_FIELD, "") or "")

    def _strategy_label(self) -> str:
        key = self.strategy()
        for item_key, item_label in strategy_options(self.node_id()):
            if item_key == key and item_label:
                return item_label
        return key.upper()

    def set_strategy(self, key: str) -> bool:
        key = (key or "").strip()
        if key == self.strategy():
            return False
        config = self.pipeline_config()
        config[STRATEGY_FIELD] = key
        self.set_pipeline_config(config)
        self._sync_ports()
        self.update()
        return True

    def _sync_ports(self) -> None:
        inputs, outputs = resolved_ports(self.node_id(), self.pipeline_config())
        self.rebuild_ports(inputs, outputs)

    # ---------- file (variant-specific filters) ----------

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

    def _file_filters(self) -> str:
        meta = variant_meta(self.node_id(), self.strategy())
        return meta.get("file_filters") or "Raster files (*.tif *.tiff)"

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            None, t("Select product file"), "", self._file_filters()
        )
        if path:
            self.set_file_path(path)

    # ---------- config persistence ----------

    def capture_config(self) -> dict:
        config = dict(self._pipeline_config)
        if not config:
            return {}
        return {"pipeline": config}

    def apply_config(self, config: dict) -> None:
        super().apply_config(config)
        cfg = config.get("pipeline")
        if isinstance(cfg, dict):
            self._file_path = cfg.get("file_path") or self._file_path
            self._file_name = (
                os.path.basename(self._file_path) if self._file_path else None
            )
            self._file_missing = bool(
                self._file_path and not os.path.isfile(self._file_path)
            )
        self._sync_ports()

    # ---------- layout ----------

    def content_height(self) -> float:
        rh, rs, ll = scaled_px(ROW_HEIGHT), scaled_px(ROW_SPACING), scaled_px(LABEL_LINE_HEIGHT)
        return rh + rs + rh + rs + ll + 4

    def _selector_rect(self) -> QRectF:
        rect = self._content_rect()
        return QRectF(rect.x(), rect.y(), rect.width(), scaled_px(ROW_HEIGHT))

    def _button_rect(self) -> QRectF:
        rect = self._content_rect()
        rh, rs = scaled_px(ROW_HEIGHT), scaled_px(ROW_SPACING)
        return QRectF(
            rect.x(),
            rect.y() + rh + rs,
            rect.width(),
            rh,
        )

    def _label_rect(self) -> QRectF:
        rect = self._content_rect()
        rh, rs = scaled_px(ROW_HEIGHT), scaled_px(ROW_SPACING)
        y = rect.y() + (rh + rs) * 2
        return QRectF(rect.x(), y, rect.width(), scaled_px(LABEL_LINE_HEIGHT))

    def _hit_region(self, pos):
        if self._selector_rect().contains(pos):
            return "strategy"
        if self._button_rect().contains(pos):
            return "button"
        return super()._hit_region(pos)

    def mousePressEvent(self, event) -> None:
        region = self._hit_region(event.pos())
        if region in ("strategy", "button"):
            self.on_content_click(region)
            event.accept()
            return
        super().mousePressEvent(event)

    def on_content_click(self, region) -> None:
        if region == "strategy":
            self._choose_strategy()
        elif region == "button":
            self._browse()

    def _choose_strategy(self) -> None:
        options = strategy_options(self.node_id())
        if not options:
            return
        menu = QMenu()
        for item_key, item_label in options:
            action = menu.addAction(t(item_label))
            action.setData(item_key)
        menu.addSeparator()
        none_action = menu.addAction(_none_label())
        none_action.setData("")
        chosen = menu.exec(_cursor_pos())
        if chosen is not None:
            self.strategy_change_requested.emit(str(chosen.data() or ""))

    # ---------- painting ----------

    def paint_content(self, painter: QPainter, rect: QRectF) -> None:
        selector_hovered = self._hover_region == "strategy"
        button_hovered = self._hover_region == "button"

        key = self.strategy()
        label = t(self._strategy_label()) if key else _none_label()
        selector_text = f"{t('Product:')} {label} \u25be"
        selector_rect = self._selector_rect()
        self.paint_button(painter, selector_rect, selector_text, selector_hovered)

        self.paint_button(
            painter, self._button_rect(), t("Browse..."), button_hovered
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
        label = QFontMetrics(font).elidedText(
            text, Qt.TextElideMode.ElideRight, int(label_rect.width())
        )
        painter.setFont(font)
        painter.drawText(
            label_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            label,
        )


class SelectorNodeItem(StrategyNodeItem):
    """Strategy item with a selector row only (no file rows).

    Shared by families whose strategy needs no picked file (merges pick
    a product, methods pick an algorithm): subclasses only set
    ``selector_title``. Non-selector regions delegate to the plain item
    so a Browse button can never fire where none is painted.
    """

    selector_title = "Product:"

    def content_height(self) -> float:
        return scaled_px(ROW_HEIGHT) + 4

    def _selector_rect(self) -> QRectF:
        rect = self._content_rect()
        return QRectF(rect.x(), rect.y(), rect.width(), scaled_px(ROW_HEIGHT))

    def _hit_region(self, pos):
        if self._selector_rect().contains(pos):
            return "strategy"
        return NodeItem._hit_region(self, pos)

    def on_content_click(self, region) -> None:
        if region == "strategy":
            self._choose_strategy()
        else:
            NodeItem.on_content_click(self, region)

    def paint_content(self, painter: QPainter, rect) -> None:
        selector_hovered = self._hover_region == "strategy"
        key = self.strategy()
        label = t(self._strategy_label()) if key else _none_label()
        selector_text = f"{t(self.selector_title)} {label} \u25be"
        selector_rect = self._selector_rect()
        self.paint_button(painter, selector_rect, selector_text, selector_hovered)


class MethodNodeItem(SelectorNodeItem):
    """Classify-style item: method selector, no file rows."""

    selector_title = "Method:"