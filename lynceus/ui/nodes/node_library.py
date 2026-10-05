# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
from PySide6.QtCore import QEvent, QMimeData, Qt, Signal
from PySide6.QtGui import QDrag
from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem

NODE_MIME_TYPE = "application/x-lynceus-node"


class NodeLibrary(QTreeWidget):
    """Node tree with drag-and-drop, double-click insertion, and tree toggling."""

    node_add_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setDragDropMode(QTreeWidget.DragDropMode.DragOnly)
        self.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        self.setAnimated(True)

        self.itemActivated.connect(self._on_item_activated)

    def _on_item_activated(self, item: QTreeWidgetItem, column: int) -> None:
        node_id = item.data(0, Qt.ItemDataRole.UserRole)
        if node_id:
            self.node_add_requested.emit(node_id)

    def viewportEvent(self, event) -> bool:
        if (
            event.type() == QEvent.Type.MouseButtonPress
            and event.button() == Qt.MouseButton.LeftButton
        ):
            pos = event.position().toPoint()
            index = self.indexAt(pos)
            if index.isValid() and index.parent().isValid() is False:
                item = self.itemFromIndex(index)
                if item is not None and item.childCount() > 0:
                    item.setExpanded(not item.isExpanded())
                    return True
        return super().viewportEvent(event)

    def startDrag(self, supported_actions) -> None:
        item = self.currentItem()
        if item is None:
            return
        node_id = item.data(0, Qt.ItemDataRole.UserRole)
        if not node_id:
            return

        mime_data = QMimeData()
        mime_data.setData(NODE_MIME_TYPE, node_id.encode("utf-8"))

        drag = QDrag(self)
        drag.setMimeData(mime_data)
        drag.exec(supported_actions)
