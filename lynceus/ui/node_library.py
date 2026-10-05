# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Populate the node library dynamically using a QTreeWidget.

The catalog comes from PluginManager: built-in and installed extension nodes.
Metadata is read through the AST without executing code; a module is imported
only when the user drops a node onto the canvas.

The tree is organized in three levels: category -> subcategory (optional,
``NODE_SUBCATEGORY``) -> node. Nodes without a subcategory hang directly
under their category.
"""

import logging

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem

from lynceus.plugins.locale import t
from lynceus.plugins.registry import manager

logger = logging.getLogger(__name__)

SUBGROUP_ORDER = (
    "Input",
    "Operations",
    "Concatenate",
    "Source",
    "Cleaning",
    "Terrain",
    "Analysis",
    "Export",
)
"""Canonical subcategory order across categories (LiDAR Source..Export,
Raster Input/Operations/Analysis, Table Input/Concatenate); unknown
subcategories sort alphabetically afterwards. Keys are English metadata,
not translated labels."""


def _discover_nodes() -> list[tuple[str, str, str, str]]:
    """Return raw (category, subcategory, name, id) registry entries."""
    found = []
    for info in manager.list_nodes():
        if info.name and info.category:
            found.append(
                (info.category, info.subcategory or "", info.name, info.node_id)
            )
    return found


def _subcategory_sort_key(subcategory: str) -> tuple:
    if subcategory in SUBGROUP_ORDER:
        return (0, SUBGROUP_ORDER.index(subcategory))
    if subcategory:
        return (1, subcategory)
    return (2, "")


def _node_item(name: str, node_id: str) -> QTreeWidgetItem:
    item = QTreeWidgetItem([name])
    item.setData(0, Qt.ItemDataRole.UserRole, node_id)
    item.setToolTip(0, node_id)
    return item


def populate_library(tree: QTreeWidget) -> None:
    """Build the tree hierarchy from the available nodes.

    Categories and subcategories are created dynamically; labels are
    translated, grouping/order is decided on the raw English metadata. Node
    ids are stored in UserRole for drag-and-drop onto the canvas.
    """
    tree.clear()

    grouped: dict[str, dict[str, list[tuple[str, str]]]] = {}
    for category, subcategory, name, node_id in _discover_nodes():
        grouped.setdefault(category, {}).setdefault(subcategory, []).append(
            (name, node_id)
        )

    for category in sorted(grouped):
        category_item = QTreeWidgetItem([t(category)])
        subgroups = grouped[category]
        for subcategory in sorted(subgroups, key=_subcategory_sort_key):
            if subcategory:
                sub_item = QTreeWidgetItem([t(subcategory)])
                for name, node_id in sorted(subgroups[subcategory]):
                    sub_item.addChild(_node_item(t(name), node_id))
                category_item.addChild(sub_item)
            else:
                for name, node_id in sorted(subgroups[subcategory]):
                    category_item.addChild(_node_item(t(name), node_id))
        tree.addTopLevelItem(category_item)
        category_item.setExpanded(False)