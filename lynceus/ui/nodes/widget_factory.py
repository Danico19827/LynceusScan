# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
from lynceus.ui.nodes.node_item import NodeItem
from lynceus.ui.nodes.widgets.input_file import InputFileNodeItem
from lynceus.ui.nodes.widgets.load_las_laz import LoadLasLazNodeItem
from lynceus.ui.nodes.widgets.merge import MergeNodeItem
from lynceus.ui.nodes.widgets.strategy import MethodNodeItem, StrategyNodeItem

_ITEM_FACTORIES = {
    "lynceus.nodes.lidar.source.load_las_laz": LoadLasLazNodeItem,
    # File inputs need NO registration: any node declaring input_file_key
    # in specs gets InputFileNodeItem by convention (filters from specs,
    # then the legacy per-kind target table).
    "lynceus.nodes.raster.input_raster": StrategyNodeItem,
    # Method strategies (one Classify Ground instance per algorithm).
    "lynceus.nodes.lidar.terrain.classify_ground": MethodNodeItem,
    # Merge products per strategy (one Consolidate instance per product).
    "lynceus.nodes.flow.consolidate": MergeNodeItem,
}


def create_node_item(
    node_id: str,
    node_name: str,
    category: str,
    inputs: tuple[str, ...] = (),
    outputs: tuple[str, ...] = (),
    parent=None,
    specs: dict | None = None,
) -> NodeItem:
    """Factory: maps a node id to its item class.

    Explicit registrations win. Otherwise the file-input convention
    applies: any node declaring ``input_file_key`` in its specs gets the
    Browse + file-label item (with ``file_filters`` from the same specs),
    so third-party inputs need no core registration. Anything else uses
    the generic base NodeItem. Strategy/variant families keep explicit
    registration (their variants carry the file config, not the base).
    """
    item_class = _ITEM_FACTORIES.get(node_id, NodeItem)
    file_filters = None
    if item_class is NodeItem and specs and specs.get("input_file_key"):
        item_class = InputFileNodeItem
        file_filters = specs.get("file_filters")
    item = item_class(node_id, node_name, category, inputs, outputs, parent)
    if file_filters and isinstance(item, InputFileNodeItem):
        item.set_file_filters(file_filters)
    return item
