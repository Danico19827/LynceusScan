# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Node -- Terrain Derivatives (strategy family: Horn 1981 operators).

This is a *strategy node*: its ports/capabilities come from the derivative
variant selected in the config (`strategy` key). With no derivative the
node is inert (no ports). Variants are the `terrain_derivative_*` modules
whose ``VARIANT_OF`` points back to this node id; each reads one wired DTM
mosaic and writes one raster product (slope/aspect/hillshade) as a
barrier task. Fingerprints (strategy included) keep derivatives from ever
sharing reuse.
"""

from __future__ import annotations

from lynceus.nodes._variants import STRATEGY_FIELD

NODE_ID = "lynceus.nodes.raster.ops.terrain_derivatives"
NODE_NAME = "Terrain Derivatives"
NODE_CATEGORY = "Raster"
NODE_SUBCATEGORY = "Operations"

NODE_DESCRIPTION = (
    "<b>Terrain Derivatives</b> -- Slope, Aspect and Hillshade from a DTM"
    "<br><br>Derives terrain surfaces from a wired DTM mosaic with Horn "
    "(1981) 3x3 operators: slope in degrees, aspect clockwise from north "
    "(flat cells read -1), and hillshade 0-255 for the configured sun."
    "<br><br><b>Process:</b> Choose the derivative with the button on the "
    "node (or the Strategy dropdown in the Inspector). Cells whose 3x3 "
    "window touches NODATA stay NODATA, and a 1-cell rim stays NODATA by "
    "design.<br><br><b>Tips:</b> Feed it a DTM (bare earth). Slope honors "
    "z_factor for vertical exaggeration; the hillshade sun defaults to "
    "the 315/45 cartographic standard."
)

INPUTS = ()
OUTPUTS = ()

PROCESSING_SPECS = {
    "config_schema": {
        STRATEGY_FIELD: {
            "type": "str",
            "default": "",
            "description": (
                "Which terrain derivative this node computes. Selecting "
                "one activates the raster ports with that product's "
                "parameters."
            ),
            "impact": (
                "Changing the derivative rebuilds the node's ports and cuts "
                "all existing connections to it."
            ),
            "group": "Method",
        }
    },
}


def get_config_defaults() -> dict:
    return {STRATEGY_FIELD: ""}
