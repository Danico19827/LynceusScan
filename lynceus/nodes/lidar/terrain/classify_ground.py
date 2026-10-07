# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Node -- Classify Ground (strategy family: ground filtering algorithms).

This is a *strategy node*: its ports/capabilities come from the method
variant selected in the config (`strategy` key). With no method the node
is inert (no ports). Variants are the `classify_ground_*` modules whose
``VARIANT_OF`` points back to this node id; each declares the same
POINT_CLOUD in/out contract with its own algorithm, tile task and
parameters. All variants write the shared `classified/` stream, so
downstream nodes read it identically; fingerprints (strategy included)
keep methods from ever sharing reuse.
"""

from __future__ import annotations

from lynceus.nodes._variants import STRATEGY_FIELD

NODE_ID = "lynceus.nodes.lidar.terrain.classify_ground"
NODE_NAME = "Classify Ground"
NODE_CATEGORY = "LiDAR"
NODE_SUBCATEGORY = "Terrain"

NODE_DESCRIPTION = (
    "<b>Classify Ground</b> -- Ground Point Classification<br><br>"
    "Identifies bare-earth points and assigns them to class 2 (ground), "
    "writing a classified point-cloud stream for terrain products.<br><br>"
    "<b>Process:</b> Choose the filtering method with the button on the "
    "node (or the Strategy dropdown in the Inspector): PMF (progressive "
    "morphology, general purpose), CSF (cloth simulation, noisy or "
    "photogrammetric clouds), SMRF (fixed-threshold morphology, flat "
    "and agricultural terrain), MCC (curvature plate, dense forest "
    "without ground returns) or ATIN (adaptive TIN, stepwise "
    "densification from seed minima). Each tile is classified "
    "independently "
    "and downstream nodes read the shared stream identically.<br><br>"
    "<b>Tips:</b> Changing the method rebuilds the node's ports and cuts "
    "existing connections. Re-runs with another method recompute from "
    "scratch (methods never share cached products)."
)

INPUTS = ()
OUTPUTS = ()

PROCESSING_SPECS = {
    "config_schema": {
        STRATEGY_FIELD: {
            "type": "str",
            "default": "",
            "description": (
                "Which ground filtering method this node runs. Selecting "
                "one activates the point-cloud ports with that method's "
                "parameters."
            ),
            "impact": (
                "Changing the method rebuilds the node's ports and cuts "
                "all existing connections to it."
            ),
            "group": "Method",
        }
    },
}


def get_config_defaults() -> dict:
    return {STRATEGY_FIELD: ""}
