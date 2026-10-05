# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Variant table of the Consolidate strategy family (no NODE_ID)."""

from __future__ import annotations

from lynceus.nodes.flow._consolidate_base import consolidate_tables
from lynceus.nodes.ports import PortDef, PortType

VARIANT_OF = "lynceus.nodes.flow.consolidate"
VARIANT_KEY = "table"
VARIANT_LABEL = "Table"

INPUTS = (PortDef(PortType.TABLE),)
OUTPUTS = (PortType.TABLE_CSV,)

PROCESSING_SPECS = {
    "consolidator": True,
    "consolidate_task": "barrier_consolidate_table",
    "output_files": {"table_csv": "consolidated.csv"},
    "output_globs": ("*/consolidated.csv",),
    "config_schema": {
        "delimiter": {
            "type": "str",
            "default": ",",
            "description": "Delimiter used when merging CSV tables.",
            "impact": "The merged table is written with this separator; provider CSV files are read with it too.",
            "group": "Advanced",
            "advanced": True,
        },
    },
}


def barrier_consolidate_table(ctx: dict) -> dict:
    """Concatenate per-segment tables into one CSV (column union)."""
    payload = consolidate_tables(
        ctx, session_file="consolidated.csv", input_type="table"
    )
    payload["display_name"] = "Table"
    return payload
