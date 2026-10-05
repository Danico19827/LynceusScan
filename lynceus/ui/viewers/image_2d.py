# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Plain image viewer for non-georeferenced pictures (PNG/JPG/...).

Unlike the raster viewer (GeoTIFFs with CRS/transform via rasterio), this
reads through Pillow and shows pixels as-is: no georeferencing, no QML,
no NODATA semantics. Registered for ``image``; custom ports opt in with
``viewer_kind: "image"``.
"""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

from lynceus.plugins.locale import t
from lynceus.ui.viewers.base import BaseViewer
from lynceus.ui.viewers.registry import register

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")

MAX_PREVIEW_PX = 4096


class _ZoomView(QGraphicsView):
    """Pan with drag, zoom with the wheel, fit on demand."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )

    def wheelEvent(self, event) -> None:  # Qt override
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)

    def fit_scene(self) -> None:
        scene = self.scene()
        if scene is not None:
            self.fitInView(scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)


class ImageViewer(BaseViewer):
    """Zoomable plain-image view with fit on load."""

    def __init__(self, kind: str, parent=None):
        super().__init__(kind, parent)
        self._view = _ZoomView(self)
        self._scene = QGraphicsScene(self)
        self._view.setScene(self._scene)
        self._layout.addWidget(self._view)
        self._view.hide()

    def set_payload(self, payload) -> None:
        path = None
        if isinstance(payload, dict):
            candidate = payload.get("file")
            path = str(candidate) if candidate else ""
        if not path or Path(path).suffix.lower() not in IMAGE_EXTENSIONS:
            self.show_placeholder(t("No image data available"))
            return

        def load(cancelled):
            from PIL import Image

            if cancelled.is_set():
                return None
            with Image.open(path) as opened:
                opened.load()
                frame = opened.convert("RGB")
                frame.thumbnail((MAX_PREVIEW_PX, MAX_PREVIEW_PX))
                return frame.tobytes(), frame.size, frame.mode

        def loaded(result) -> None:
            if result is None:
                return
            data, (width, height), mode = result
            image = QImage(data, width, height, 3 * width, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(image.copy())
            if pixmap.isNull():
                self.show_placeholder(t("Cannot read image"))
                return
            self._scene.clear()
            self._scene.addItem(QGraphicsPixmapItem(pixmap))
            self._view.show()
            self._placeholder.hide()
            self._view.fit_scene()
            self.setWindowTitle(self._compose_title())

        def failed(exc: str) -> None:
            self.show_placeholder(
                t("Cannot read image: {exc}").format(exc=exc)
                + (f"\n{path}" if path else "")
            )

        self._view.hide()
        self.start_async_load(load, loaded, failed)


register("image", ImageViewer)
