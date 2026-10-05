# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Outputs panel: catalog of products with an available viewer.

Products are grouped by node. Double-click opens a floating preview and the
context menu opens the containing folder. The catalog is populated after a run
and cleared when a new run or session starts.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from PySide6.QtCore import QUrl, Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QLabel,
    QMenu,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lynceus.plugins.locale import t
from lynceus.ui.outputs_model import (
    OutputProduct,
    display_labels,
    product_for_path,
    render_flow,
    viewer_kind_label,
)
from lynceus.ui.translate import language_changed

def _kind_label(viewer: str) -> str:
    return t(viewer_kind_label(viewer))


def _human_size(size: int) -> str:
    if size <= 0:
        return ""
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} {unit}"
        size /= 1024
    return ""


def _segment_index(label: str) -> int:
    """Numeric batch index from a segment label, language-agnostic.

    Segment tags are translated at creation ("Segment 2/5" in the active
    language), so only the digits are parsed, never the word.
    """
    match = re.search(r"(\d+)\s*/", label)
    return int(match.group(1)) if match else 2**31


_SEGMENT_RE = re.compile(r"^(.*?)\s*(\d+)\s*/\s*(\d+)\s*$")


def _segment_block_label(label: str) -> str:
    """Render a segment block label through the current catalog.

    ``label`` is the canonical English form ("Segment 2/5") stored at publish
    time; rebuilding it from the live template keeps it in sync with language
    switches without requiring a new run.
    """
    match = _SEGMENT_RE.match(label)
    if match:
        return t("Segment {k}/{n}").format(k=match.group(2), n=match.group(3))
    return label


class OutputsPanel(QWidget):
    """Gallery of visualizable products from the latest run."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("outputsPanel")
        self._products: list[OutputProduct] = []
        self._open_viewers: list[QWidget] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.tree = QTreeWidget(self)
        self.tree.setObjectName("outputsTree")
        self.tree.setAlternatingRowColors(True)
        self.tree.setRootIsDecorated(True)
        self.tree.setColumnWidth(0, 200)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.itemDoubleClicked.connect(self._on_double_clicked)
        self.tree.customContextMenuRequested.connect(self._on_context_menu)
        layout.addWidget(self.tree)

        self._empty_hint = QLabel(t("No previewable outputs."))
        self._empty_hint.setProperty("origText_en", "No previewable outputs.")
        self._empty_hint.setObjectName("panelHint")
        self._empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._empty_hint)

        self.set_products([])
        language_changed.connect(self._rebuild)

    # ---------------- data input ----------------

    def set_products(self, products: list[OutputProduct]) -> None:
        self._products = list(products)
        self._rebuild()

    def clear(self) -> None:
        self.set_products([])

    def open_external_file(self, path: str, node_name: str) -> bool:
        """Open a floating preview for an arbitrary file (bridge mode).

        No pipeline, session, or copy is involved; the gallery is untouched.
        Returns False when the file has no known viewer (the caller reports it).
        """
        product = product_for_path(path, node_name)
        if product is None:
            return False
        self._open_preview(product)
        return True

    def _labels(self) -> list[str]:
        """Per-product gallery label (relative path when basenames collide)."""
        bases = [self._product_base(p) for p in self._products]
        return display_labels(
            bases,
            [p.path for p in self._products],
            [p.iid for p in self._products],
        )

    def _full_label(self, product: OutputProduct, base: str) -> str:
        flow = _flow_text(product)
        return f"{base} · {flow}" if flow else base

    @staticmethod
    def _product_base(product: OutputProduct) -> str:
        if product.viewer == "point_cloud" and not product.path:
            return f"{t('Point Cloud')} ({len(product.tiles)} tiles)"
        return Path(product.name).name

    def _rebuild(self) -> None:
        self.tree.clear()
        self.tree.setHeaderLabels(
            [t("Product"), t("Viewer"), t("Size")]
        )
        if not self._products:
            self.tree.setVisible(False)
            self._empty_hint.setText(t("No previewable outputs."))
            self._empty_hint.setVisible(True)
            return

        segment_labels = sorted(
            {p.segment for p in self._products if p.segment},
            key=_segment_index,
        )
        labels = self._labels()
        label_of = {
            id(product): label
            for product, label in zip(self._products, labels)
        }
        if not segment_labels:
            # Plain node grouping (no segmented run): one top-level per node.
            node_map: dict[str, list[OutputProduct]] = {}
            for product in self._products:
                node_map.setdefault(product.node_name, []).append(product)
            for node_name, prods in node_map.items():
                parent = QTreeWidgetItem([t(node_name)])
                for p in prods:
                    parent.addChild(
                        _item_for_product(
                            p, self._full_label(p, label_of[id(p)])
                        )
                    )
                self.tree.addTopLevelItem(parent)
                parent.setExpanded(True)
            self.tree.setVisible(True)
            self._empty_hint.setVisible(False)
            return

        # Segmented run: one expandable block per finished segment; products
        # without a segment (consolidated outputs) form a final block.
        blocks: dict[str | None, dict[str, list[OutputProduct]]] = {
            label: {} for label in segment_labels
        }
        if any(p.segment is None for p in self._products):
            blocks[None] = {}
        for product in self._products:
            seg = product.segment if product.segment else None
            node_map = blocks.setdefault(seg, {})
            node_map.setdefault(product.node_name, []).append(product)

        def add_block(label: str, node_map) -> None:
            parent = QTreeWidgetItem([label])
            for node_name, prods in node_map.items():
                node_item = QTreeWidgetItem([t(node_name)])
                for p in prods:
                    node_item.addChild(
                        _item_for_product(
                            p, self._full_label(p, label_of[id(p)])
                        )
                    )
                parent.addChild(node_item)
            self.tree.addTopLevelItem(parent)
            parent.setExpanded(True)

        for label in segment_labels:
            add_block(_segment_block_label(label), blocks[label])
        if blocks.get(None):
            add_block(t("Consolidated"), blocks[None])
        self.tree.setVisible(True)
        self._empty_hint.setVisible(False)

    # ---------------- actions ----------------

    def _selected_product(self) -> OutputProduct | None:
        item = self.tree.currentItem()
        if item is None:
            return None
        product = item.data(0, Qt.ItemDataRole.UserRole)
        return product if isinstance(product, OutputProduct) else None

    def _on_double_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        product = item.data(0, Qt.ItemDataRole.UserRole)
        if isinstance(product, OutputProduct):
            self._open_preview(product)

    def _on_context_menu(self, pos) -> None:
        product = self._selected_product()
        if product is None:
            return
        menu = QMenu(self)
        menu.addAction(t("Open preview"), lambda: self._open_preview(product))
        menu.addAction(
            t("Open containing folder"),
            lambda: self._open_folder(product),
        )
        menu.exec(self.tree.viewport().mapToGlobal(pos))

    def _open_preview(self, product: OutputProduct) -> None:
        from lynceus.ui.preview_window import show_preview
        from lynceus.ui.viewers.registry import viewers_for

        if product.viewer == "point_cloud":
            if product.path:
                payload = {"file": product.path}
            elif product.tiles:
                payload = {"tiles": list(product.tiles)}
            else:
                return
        elif product.path:
            payload = {"file": product.path}
        else:
            return
        viewers = viewers_for(product.viewer)
        if not viewers:
            return
        viewer = viewers[0](product.viewer)
        try:
            index = self._products.index(product)
            viewer.set_product_title(
                t(product.node_name), self._labels()[index]
            )
        except ValueError:
            pass
        viewer.set_payload(payload)
        window = show_preview(viewer)
        self._prune_closed_viewers()
        self._open_viewers.append(window)

    def _open_folder(self, product: OutputProduct) -> None:
        first_tile = product.tiles[0] if product.tiles else ""
        if isinstance(first_tile, dict):
            first_tile = first_tile.get("file", "")
        target = product.path or first_tile
        if not target:
            return
        folder = os.path.dirname(target)
        if os.path.isdir(folder):
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    def _prune_closed_viewers(self) -> None:
        self._open_viewers = [v for v in self._open_viewers if v.isVisible()]


def _flow_text(product: OutputProduct) -> str:
    """Upstream → self → downstream lineage, empty for lone nodes."""
    if len(product.flow) < 2:
        return ""
    return render_flow(product.flow, t)


def _item_for_product(product: OutputProduct, label: str | None = None) -> QTreeWidgetItem:
    if product.viewer == "point_cloud" and not product.path:
        name = f"{t('Point Cloud')} ({len(product.tiles)} tiles)"
        tip = "; ".join(
            tile.get("file", "") if isinstance(tile, dict) else str(tile)
            for tile in product.tiles[:4]
        )
    else:
        name = Path(product.name).name
        tip = product.path or ""
    flow = _flow_text(product)
    if flow:
        tip = f"{tip}\n{flow}" if tip else flow
    item = QTreeWidgetItem(
        [label or name, _kind_label(product.viewer), _human_size(product.size)]
    )
    item.setData(0, Qt.ItemDataRole.UserRole, product)
    if tip:
        item.setToolTip(0, tip)
    return item