# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Node painting: offscreen render smoke + style constants.

Painting overrides abort the process on any Python exception, so every
visual tweak to NodeItem earns a render pass here (normal, hovered port,
selected).
"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication

from lynceus.ui.nodes.canvas import NodeCanvasView
from lynceus.ui.nodes import node_item as node_mod

LOAD_ID = "lynceus.nodes.lidar.source.load_las_laz"
DTM_ID = "lynceus.nodes.lidar.terrain.generate_dtm"


class NodeStyleTests(unittest.TestCase):
    def setUp(self) -> None:
        self._app = QApplication.instance() or QApplication([])

    def test_style_constants(self) -> None:
        from PySide6.QtGui import QColor

        from lynceus.nodes.ports import get_port_color

        self.assertEqual(node_mod.CORNER_RADIUS, 8)
        view = NodeCanvasView()
        try:
            item = view.add_node(DTM_ID, QPointF(0, 0))
            expected = QColor(get_port_color(
                item.outputs()[0].port_type_id)).name()
            self.assertEqual(item._spine_color().name(), expected)
        finally:
            view.deleteLater()

    def test_paint_smoke(self) -> None:
        view = NodeCanvasView()
        try:
            first = view.add_node(LOAD_ID, QPointF(0, 0))
            second = view.add_node(DTM_ID, QPointF(300, 0))
            view._create_connection(
                first.outputs()[0], second.inputs()[0])
            for item in (first, second):
                rect = item.boundingRect()
                image = QImage(int(rect.width()) + 40,
                               int(rect.height()) + 40,
                               QImage.Format.Format_ARGB32)
                painter = QPainter(image)
                try:
                    item.paint(painter, None, None)
                finally:
                    painter.end()
                self.assertFalse(image.isNull())
            # Hovered port + selection paths paint too.
            second._hovered_port = second.inputs()[0]
            rect = second.boundingRect()
            image = QImage(int(rect.width()) + 40,
                           int(rect.height()) + 40,
                           QImage.Format.Format_ARGB32)
            painter = QPainter(image)
            try:
                second.paint(painter, None, None)
            finally:
                painter.end()
                second._hovered_port = None
            self.assertFalse(image.isNull())
        finally:
            view.deleteLater()

    def test_canvas_render_smoke(self) -> None:
        """Full scene render: wires, vignette grid and notice styling."""
        from lynceus.ui.nodes.canvas import NOTICE_KIND_COLORS

        view = NodeCanvasView()
        try:
            first = view.add_node(LOAD_ID, QPointF(0, 0))
            second = view.add_node(DTM_ID, QPointF(300, 0))
            view._create_connection(
                first.outputs()[0], second.inputs()[0])
            view.resize(800, 600)
            view.show()
            image = QImage(800, 600, QImage.Format.Format_ARGB32)
            painter = QPainter(image)
            try:
                view.render(painter)
            finally:
                painter.end()
            self.assertFalse(image.isNull())
            view.post_notice("smoke", "warning")
            rows = view._notice_tray.findChildren(object, "noticeRow") \
                if hasattr(view, "_notice_tray") else []
            if rows:
                row = rows[0]
                self.assertEqual(row.property("noticeKind"), "warning")
                self.assertEqual(row.styleSheet(), "")
                dots = row.findChildren(object, "noticeDot")
                self.assertEqual(len(dots), 1)
            self.assertIn("warning", NOTICE_KIND_COLORS)
        finally:
            view.deleteLater()

    def test_background_stable_across_partial_repaints(self) -> None:
        """A partial repaint must shade exactly like the full frame.

        Regression: the vignette was centered on the exposed rect, so the
        rubber-band repaint left a dirty rectangle behind (light themes).
        """
        from PySide6.QtCore import QRectF

        view = NodeCanvasView()
        try:
            view.resize(700, 420)
            view.show()
            full = QImage(700, 420, QImage.Format.Format_ARGB32)
            painter = QPainter(full)
            try:
                view.drawBackground(
                    painter, QRectF(0, 0, 700, 420))
            finally:
                painter.end()
            part = QImage(700, 420, QImage.Format.Format_ARGB32)
            part.fill(0xFFFF00FF)
            painter = QPainter(part)
            try:
                view.drawBackground(painter, QRectF(100, 100, 200, 150))
            finally:
                painter.end()
            for x, y in ((100, 100), (150, 175), (299, 249), (200, 150)):
                self.assertEqual(full.pixel(x, y), part.pixel(x, y))
        finally:
            view.deleteLater()


if __name__ == "__main__":
    unittest.main()
