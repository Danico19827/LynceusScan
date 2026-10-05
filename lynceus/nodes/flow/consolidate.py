# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Node - Consolidate (strategy family: merge one product across segments).

This is a *strategy node*: its ports/capabilities come from a variant
selected in its config (`strategy` key). With no strategy the node is
inert (no ports). Each variant merges exactly one product type (CHM, DSM,
DTM, grid metrics, vector, table); consolidate several products with one
instance per strategy. Shared merge machinery lives in
``_consolidate_base``.
"""

from __future__ import annotations

from lynceus.nodes._variants import STRATEGY_FIELD

NODE_ID = "lynceus.nodes.flow.consolidate"
NODE_NAME = "Consolidate"
NODE_CATEGORY = "Flow"

NODE_DESCRIPTION = (
    "<b>Consolidate</b> -- Merge One Product Across Segments<br><br>"
    "Merges the per-segment products of one product type into a single "
    "final result, so downstream nodes and the gallery see the consolidated "
    "output instead of per-segment ones. Add one instance per product and "
    "choose its type below.<br><br>"
    "<b>Process:</b> Choose the product with the button on the node (or "
    "the Strategy dropdown in the Inspector), then connect its provider. "
    "The merge waits for all segments and runs in a final session.<br><br>"
    "<b>Tips:</b> Changing the product rebuilds the node's ports and cuts "
    "existing connections."
)

INPUTS = ()
OUTPUTS = ()

PROCESSING_SPECS = {
    "config_schema": {
        STRATEGY_FIELD: {
            "type": "str",
            "default": "",
            "description": (
                "Which product this instance merges. Selecting one "
                "activates its input and output ports."
            ),
            "impact": (
                "Changing the product rebuilds the node's ports and cuts all "
                "existing connections to it."
            ),
            "group": "Product",
        }
    },
}


def get_config_defaults() -> dict:
    return {STRATEGY_FIELD: ""}
