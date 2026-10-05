# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Table product-input registry (no NODE_ID: the AST discovery ignores it).

Declarative data keyed by NODE_ID: adding a new table input is one row here +
a trivial module (`input_<x>.py` in `lynceus/nodes/table/`). `file_filters`
feeds the UI item's file dialog; `kind_group` selects the validation branch in
`lynceus.nodes._product_input.barrier_input_file`.
"""

from __future__ import annotations

_input_targets: dict[str, dict] = {
    "lynceus.nodes.table.input_table": {
        "node_id": "lynceus.nodes.table.input_table",
        "node_name": "Input Table",
        "output_port": "table_csv",
        "session_file": "table.csv",
        "kind_label": "Table",
        "kind_group": "table_csv",
        "file_filters": "CSV files (*.csv)",
    },
    "lynceus.nodes.table.input_gpkg": {
        "node_id": "lynceus.nodes.table.input_gpkg",
        "node_name": "Input GeoPackage",
        "output_port": "table_gpkg",
        "session_file": "rows.gpkg",
        "kind_label": "GeoPackage",
        "kind_group": "table_gpkg",
        "file_filters": "GeoPackage files (*.gpkg)",
    },
}

INPUT_TARGETS = _input_targets


def input_target(node_id: str) -> dict:
    """Declarative target row for a table input ({} if the id is not registered)."""
    return _input_targets.get(node_id, {})