# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Node - Load LAS/LAZ file.

Source node: no inputs, produces a POINT_CLOUD stream (the file is split
into tiles internally for parallel processing). The tiling and segmentation
parameters live in this node's configuration: tiling shapes the single
per-source pass, segmentation runs the pipeline as sequential batches.
"""

from __future__ import annotations

from lynceus.nodes.ports import PortType

NODE_ID = "lynceus.nodes.lidar.source.load_las_laz"
NODE_NAME = "Load LAS/LAZ"
NODE_CATEGORY = "LiDAR"

NODE_SUBCATEGORY = "Source"

NODE_DESCRIPTION = (
    "<b>Load LAS/LAZ</b> -- Point Cloud Source<br><br>"
    "Loads a LAS/LAZ file and advertises the point cloud to the pipeline. "
    "The file is tiled automatically so downstream nodes can process it in "
    "parallel parts.<br><br>"
    "<b>Process:</b> The file is read in chunks and divided into spatial tiles. "
    "A buffer margin is added around each tile so that features crossing tile "
    "boundaries are fully contained in at least one tile. Downstream nodes "
    "consume the point cloud in these per-tile parts.<br><br>"
    "<b>Tiling:</b> The <i>Tiling</i> group controls the tile size, buffer and "
    "per-tile point cap of the source pass. The <i>Segmentation</i> group runs "
    "the pipeline in sequential batches over groups of whole tiles, bounding "
    "the memory envelope of each pass (0 = one single pass)."
)

INPUTS = ()
OUTPUTS = (PortType.POINT_CLOUD,)

PROCESSING_SPECS = {
    "triggers_tiling": True,
    "output_port": "point_cloud",
    "config_schema": {
        "tile_size_m": {
            "type": "float",
            "default": 100.0,
            "minimum": 10.0,
            "maximum": 10000.0,
            "description": "Target size of each tile in meters.",
            "impact": "Smaller tiles are processed faster in parallel but increase overhead. Larger tiles reduce overhead but may cause memory issues on dense files.",
            "group": "Tiling",
        },
        "buffer_m": {
            "type": "float",
            "default": 10.0,
            "minimum": 0.0,
            "maximum": 100.0,
            "description": "Margin (meters) added around each tile edge.",
            "impact": "Ensures features crossing tile boundaries are fully included in at least one tile. Increase for tall vegetation (e.g. 5m for 30m trees). Higher values increase overlap and processing redundancy.",
            "group": "Tiling",
        },
        "max_points_per_tile": {
            "type": "int",
            "default": 0,
            "minimum": 0,
            "maximum": 100_000_000,
            "description": "Maximum points per tile (0 = no limit).",
            "impact": "Dense cells are subdivided into sub-tiles so none exceeds this cap, bounding the memory of each parallel worker. Set to 0 to disable the cap and use only tile size.",
            "group": "Tiling",
        },
        "max_points_per_part": {
            "type": "int",
            "default": 0,
            "minimum": 0,
            "maximum": 1_000_000_000,
            "description": "Maximum points per batch (0 = single pass).",
            "impact": "Tiles are grouped in row-column order into sequential batches that do not exceed this cap. Each batch runs the full pipeline in its own mini-session, so lower values shrink the memory envelope at the cost of more passes. A single oversized tile still forms its own batch.",
            "group": "Segmentation",
        },
    },
}


def get_config_defaults() -> dict:
    return {k: v["default"] for k, v in PROCESSING_SPECS["config_schema"].items()}