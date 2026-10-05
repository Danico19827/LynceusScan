# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Variant grid metrics of the Consolidate strategy family (no NODE_ID)."""

from __future__ import annotations

from lynceus.nodes.flow._consolidate_base import consolidate_grid_metrics
from lynceus.nodes.ports import PortDef, PortType

VARIANT_OF = "lynceus.nodes.flow.consolidate"
VARIANT_KEY = "grid_metrics"
VARIANT_LABEL = "Grid Metrics"

INPUTS = (PortDef(PortType.GRID_METRICS),)
OUTPUTS = (PortType.GRID_METRICS,)

PROCESSING_SPECS = {
    "consolidator": True,
    "consolidate_task": "barrier_consolidate_grid_metrics",
    "output_files": {"grid_metrics": "grid_metrics.gpkg"},
    "output_globs": ("*/grid_metrics.gpkg", "*/grid_metrics.csv"),
}


def barrier_consolidate_grid_metrics(ctx: dict) -> dict:
    """Merge per-segment grid tables exactly by cell key (no seams)."""
    payload = consolidate_grid_metrics(
        ctx, session_file="grid_metrics.gpkg", input_type="grid_metrics"
    )
    payload["display_name"] = "Grid Metrics"
    return payload
