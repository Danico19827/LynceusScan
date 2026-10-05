# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Variant vector of the Consolidate strategy family (no NODE_ID)."""

from __future__ import annotations

from lynceus.nodes.flow._consolidate_base import consolidate_vectors
from lynceus.nodes.ports import PortDef, PortType

VARIANT_OF = "lynceus.nodes.flow.consolidate"
VARIANT_KEY = "vector"
VARIANT_LABEL = "Vector"

INPUTS = (PortDef(PortType.VECTOR),)
OUTPUTS = (PortType.VECTOR,)

PROCESSING_SPECS = {
    "consolidator": True,
    "consolidate_task": "barrier_consolidate_vector",
    "output_files": {"vector": "vector.gpkg"},
    "output_globs": ("*/vector.gpkg",),
}


def barrier_consolidate_vector(ctx: dict) -> dict:
    """Concatenate per-segment vector feature tables exactly."""
    payload = consolidate_vectors(
        ctx, session_file="vector.gpkg", input_type="vector"
    )
    payload["display_name"] = "Vector"
    return payload
