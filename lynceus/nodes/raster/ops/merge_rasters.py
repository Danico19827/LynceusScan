# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Node -- Merge Rasters (strategy family: fuse N rasters, one scale).

This is a *strategy node*: its ports/capabilities come from the product
variant selected in its config (`strategy` key). With no product the node
is inert (no ports). Each variant fuses up to 8 wired rasters of one
product type (DTM, DSM, CHM, generic raster) into a single mosaic with one
shared-scale `.qml`, so separately processed tiles of the same zone render
without ramp seams. Chain merge instances for more than 8 inputs. Shared
merge machinery lives in ``_merge_base``.
"""

from __future__ import annotations

from lynceus.nodes._variants import STRATEGY_FIELD

NODE_ID = "lynceus.nodes.raster.ops.merge_rasters"
NODE_NAME = "Merge Rasters"
NODE_CATEGORY = "Raster"
NODE_SUBCATEGORY = "Operations"

NODE_DESCRIPTION = (
    "<b>Merge Rasters</b> -- Fuse Rasters With One Shared Scale<br><br>"
    "Merges up to 8 wired rasters of one product into a single mosaic whose "
    "`.qml` ramp spans the merged range, so tiles processed apart (separate "
    "runs, separate sessions) display with one scale instead of per-file "
    "seams. Chain merge instances for more than 8 inputs.<br><br>"
    "<b>Process:</b> Choose the product with the button on the node (or "
    "the Strategy dropdown in the Inspector), then connect at least two "
    "providers. Inputs must share CRS, dtype and band count; differing "
    "resolutions resample to the first input's grid with a warning.<br><br>"
    "<b>Tips:</b> Feed it with <code>Input Raster</code> nodes (one per "
    "file) to fuse products from other sessions. Changing the product "
    "rebuilds the node's ports and cuts existing connections."
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
