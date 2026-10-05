# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Floating preview windows with the core frameless chrome.

Viewers (``BaseViewer`` subclasses) are content widgets: they get embedded
in a ``PreviewWindow`` that owns the titlebar (brand icon top-left, title,
min/max/close), window drag and edge resize — the same language as the
main window. The viewer keeps composing its own ``windowTitle``; the
wrapper mirrors it into the titlebar label via ``windowTitleChanged``.
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from lynceus.ui.branding import ICON_SQUARE_SVG, render_svg

TITLEBAR_HEIGHT = 32
RESIZE_MARGIN = 8
MIN_WINDOW_SIZE = (400, 300)

_EDGE_CURSORS = {
    "left": Qt.CursorShape.SizeHorCursor,
    "right": Qt.CursorShape.SizeHorCursor,
    "top": Qt.CursorShape.SizeVerCursor,
    "bottom": Qt.CursorShape.SizeVerCursor,
    "topleft": Qt.CursorShape.SizeFDiagCursor,
    "bottomright": Qt.CursorShape.SizeFDiagCursor,
    "topright": Qt.CursorShape.SizeBDiagCursor,
    "bottomleft": Qt.CursorShape.SizeBDiagCursor,
}


class PreviewWindow(QWidget):
    """Frameless host for a viewer widget."""

    def __init__(self, viewer: QWidget):
        super().__init__(None)
        self._viewer = viewer
        self._window_maximized = False
        self._normal_geometry: QRect | None = None
        self._dragging = False
        self._drag_offset = QPoint()
        self._resize_edge: str | None = None
        self._resize_origin = QPoint()
        self._resize_geometry = QRect()
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setObjectName("previewWindow")
        self.setMinimumSize(*MIN_WINDOW_SIZE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(0)
        layout.addWidget(self._build_titlebar())
        layout.addWidget(viewer)
        viewer.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        self.setWindowTitle(viewer.windowTitle())
        viewer.windowTitleChanged.connect(self._on_viewer_title)
        self._title_label.setText(viewer.windowTitle())

    # ---------- chrome ----------

    def _build_titlebar(self) -> QWidget:
        bar = QWidget(self)
        bar.setObjectName("previewTitleBar")
        bar.setFixedHeight(TITLEBAR_HEIGHT)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(8, 0, 4, 0)
        layout.setSpacing(6)

        icon_label = QLabel(bar)
        icon_label.setObjectName("previewTitleIcon")
        pixmap = render_svg(ICON_SQUARE_SVG, 32)
        if not pixmap.isNull():
            icon_label.setPixmap(
                pixmap.scaled(
                    16,
                    16,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        icon_label.setFixedSize(16, 16)
        layout.addWidget(icon_label)

        self._title_label = QLabel(bar)
        self._title_label.setObjectName("previewTitleLabel")
        layout.addWidget(self._title_label, 1)

        self._btn_minimize = self._chrome_button(bar, "\u2013")
        self._btn_minimize.clicked.connect(self.showMinimized)
        self._btn_maximize = self._chrome_button(bar, "\u25a1")
        self._btn_maximize.clicked.connect(self._toggle_maximize)
        self._btn_close = self._chrome_button(bar, "\u2715", "previewClose")
        self._btn_close.clicked.connect(self.close)
        layout.addWidget(self._btn_minimize)
        layout.addWidget(self._btn_maximize)
        layout.addWidget(self._btn_close)
        return bar

    @staticmethod
    def _chrome_button(parent: QWidget, text: str, name: str = "") -> QToolButton:
        button = QToolButton(parent)
        if name:
            button.setObjectName(name)
        button.setText(text)
        button.setFixedSize(32, 24)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        return button

    # ---------- title ----------

    def _on_viewer_title(self, title: str) -> None:
        self.setWindowTitle(title)
        self._title_label.setText(title)

    def setWindowTitle(self, title: str) -> None:
        super().setWindowTitle(title)
        if hasattr(self, "_title_label"):
            self._title_label.setText(title)

    # ---------- maximize ----------

    def _toggle_maximize(self) -> None:
        if self._window_maximized:
            if self._normal_geometry is not None and self._normal_geometry.isValid():
                self.setGeometry(self._normal_geometry)
            self._window_maximized = False
        else:
            self._normal_geometry = self.geometry()
            screen = self.screen()
            if screen is not None:
                self.setGeometry(screen.availableGeometry())
            self._window_maximized = True
        self._btn_maximize.setText(
            "\u2750" if self._window_maximized else "\u25a1"
        )

    # ---------- drag + resize ----------

    def _titlebar_at(self, pos: QPoint) -> bool:
        bar = self.findChild(QWidget, "previewTitleBar")
        if bar is None:
            return False
        return bar.geometry().contains(pos)

    def _edge_at(self, pos: QPoint) -> str | None:
        rect = self.rect()
        left = pos.x() <= RESIZE_MARGIN
        right = pos.x() >= rect.width() - RESIZE_MARGIN
        top = pos.y() <= RESIZE_MARGIN
        bottom = pos.y() >= rect.height() - RESIZE_MARGIN
        if top and left:
            return "topleft"
        if top and right:
            return "topright"
        if bottom and left:
            return "bottomleft"
        if bottom and right:
            return "bottomright"
        if left:
            return "left"
        if right:
            return "right"
        if top:
            return "top"
        if bottom:
            return "bottom"
        return None

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position().toPoint()
            if not self._window_maximized:
                edge = self._edge_at(pos)
                if edge:
                    self._resize_edge = edge
                    self._resize_origin = event.globalPosition().toPoint()
                    self._resize_geometry = self.geometry()
                    event.accept()
                    return
            if self._titlebar_at(pos):
                self._dragging = True
                self._drag_offset = (
                    event.globalPosition().toPoint()
                    - self.frameGeometry().topLeft()
                )
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position().toPoint()
        global_pos = event.globalPosition().toPoint()
        if self._resize_edge and event.buttons() & Qt.MouseButton.LeftButton:
            self._apply_resize(global_pos)
            event.accept()
            return
        if self._dragging and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(global_pos - self._drag_offset)
            event.accept()
            return
        if not self._window_maximized:
            cursor = _EDGE_CURSORS.get(self._edge_at(pos) or "")
            if cursor is not None:
                self.setCursor(cursor)
            else:
                self.unsetCursor()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._dragging = False
        self._resize_edge = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if self._titlebar_at(event.position().toPoint()):
            self._toggle_maximize()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def _apply_resize(self, global_pos: QPoint) -> None:
        rect = self._resize_geometry
        edge = self._resize_edge or ""
        dx = global_pos.x() - self._resize_origin.x()
        dy = global_pos.y() - self._resize_origin.y()
        new = QRect(rect)
        if "left" in edge:
            new.setLeft(rect.left() + dx)
        if "right" in edge:
            new.setRight(rect.right() + dx)
        if "top" in edge:
            new.setTop(rect.top() + dy)
        if "bottom" in edge:
            new.setBottom(rect.bottom() + dy)
        if new.width() >= MIN_WINDOW_SIZE[0] and new.height() >= MIN_WINDOW_SIZE[1]:
            self.setGeometry(new)

    # ---------- teardown ----------

    def closeEvent(self, event) -> None:
        viewer = self._viewer
        self._viewer = None  # type: ignore[assignment]
        if viewer is not None:
            viewer.close()
        super().closeEvent(event)


def show_preview(viewer: QWidget):
    """Show a viewer inside the core-style frameless chrome."""
    from lynceus.ui.viewers.base import default_preview_size

    window = PreviewWindow(viewer)
    window.resize(default_preview_size())
    window.show()
    window.raise_()
    window.activateWindow()
    return window
