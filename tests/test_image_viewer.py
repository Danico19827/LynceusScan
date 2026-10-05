# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Plain image viewer: routing, registration and async render."""

import os
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication


def _wait_for(predicate, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.05)
    QApplication.processEvents()
    return predicate()


class ImageRoutingTests(unittest.TestCase):
    def test_extensions_route_to_image(self) -> None:
        from lynceus.ui.outputs_model import viewer_for_path

        for ext in (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"):
            self.assertEqual(viewer_for_path(f"a{ext}"), "image")
        self.assertEqual(viewer_for_path("a.tif"), "raster")
        self.assertIsNone(viewer_for_path("a.xyz"))

    def test_image_viewer_registered_with_label(self) -> None:
        from lynceus.ui.outputs_model import viewer_kind_label
        from lynceus.ui.viewers.registry import viewers_for

        self.assertTrue(viewers_for("image"))
        self.assertEqual(viewer_kind_label("image"), "Image")


class ImageRenderTests(unittest.TestCase):
    def setUp(self) -> None:
        self._app = QApplication.instance() or QApplication([])

    def _viewer(self, product):
        from lynceus.ui.viewers.image_2d import ImageViewer

        viewer = ImageViewer("image")
        viewer.set_product_title("Image Filter", product.name)
        viewer.set_payload({"file": str(product)})
        return viewer

    def test_png_renders_pixmap_item(self) -> None:
        from PIL import Image as PILImage

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "photo.png"
            PILImage.new("RGB", (32, 16), (10, 20, 30)).save(path)
            viewer = self._viewer(path)
            try:
                shown = _wait_for(
                    lambda: len(viewer._scene.items()) == 1
                )
                self.assertTrue(shown)
            finally:
                viewer.deleteLater()

    def test_missing_file_shows_placeholder(self) -> None:
        from lynceus.ui.viewers.image_2d import ImageViewer

        viewer = ImageViewer("image")
        try:
            viewer.set_payload({"file": str(Path("nope.png").absolute())})
            shown = _wait_for(
                lambda: "Cannot read" in viewer._placeholder.text()
            )
            self.assertTrue(shown)
        finally:
            viewer.deleteLater()

    def test_empty_payload_shows_placeholder(self) -> None:
        from lynceus.ui.viewers.image_2d import ImageViewer

        viewer = ImageViewer("image")
        try:
            viewer.set_payload({})
            self.assertIn("No image", viewer._placeholder.text())
        finally:
            viewer.deleteLater()

    def test_corrupt_file_reports_read_error(self) -> None:
        from lynceus.ui.viewers.image_2d import ImageViewer

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "junk.png"
            path.write_bytes(b"not an image at all")
            viewer = ImageViewer("image")
            try:
                viewer.set_payload({"file": str(path)})
                shown = _wait_for(
                    lambda: "Cannot read" in viewer._placeholder.text()
                )
                self.assertTrue(shown)
            finally:
                viewer.deleteLater()


if __name__ == "__main__":
    unittest.main()
