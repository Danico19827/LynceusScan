# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Node -- Filter by Classification.

Finisher of the cleaning chain: keeps (or removes) only the points whose
LAS class is listed in ``classes``. Placed at the end of a tagging chain it
turns the tags produced by the range/noise nodes into actual deletion.
"""

from __future__ import annotations

from lynceus.nodes._point_source import run_clean
from lynceus.nodes.ports import PortType

NODE_ID = "lynceus.nodes.lidar.clean.filter_classification"
NODE_NAME = "Filter by Classification"
NODE_CATEGORY = "LiDAR"

NODE_SUBCATEGORY = "Cleaning"

NODE_DESCRIPTION = (
    "<b>Filter by Classification</b> -- Class-Based Finisher<br><br>"
    "Keeps or removes only the points whose LAS class appears in the "
    "comma-separated list. Placed at the end of a cleaning chain it turns "
    "the tags produced by Elevation Range / Classify Noise / Scan Angle "
    "into actual deletion.<br><br>"
    "<b>Examples:</b> <i>keep</i> \"2\" produces a ground-only cloud (class 2); "
    "<i>remove</i> \"7,18\" deletes all tagged noise while preserving every "
    "other class. As the tag-to-deletion finisher, this is the step that "
    "physically culls points."
)

INPUTS = (PortType.POINT_CLOUD,)
OUTPUTS = (PortType.POINT_CLOUD,)

PROCESSING_SPECS = {
    "tile_task": "tile_filter_classification",
    "input_port": "point_cloud",
    "output_port": "point_cloud",
    "passes_flags": ("classified",),
    "point_cloud_dir": "clean_class",
    "output_globs": ("clean_class/**/*.laz",),
    "config_schema": {
        "classes": {
            "type": "str",
            "default": "2",
            "description": "Comma-separated LAS classes to keep or remove.",
            "impact": "\"2\" keeps only ground; \"2,9\" adds water. To drop tagged noise use mode=remove with \"7,18\". Classes outside 0-31 are ignored.",
            "group": "Selection",
        },
        "mode": {
            "type": "str",
            "default": "keep",
            "options": ["keep", "remove"],
            "description": "Whether the listed classes are kept or removed.",
            "impact": "keep drops every class NOT listed (classic ground isolation). remove deletes exactly the listed classes and preserves everything else.",
            "group": "Selection",
        },
    },
}


def get_config_defaults() -> dict:
    return {k: v["default"] for k, v in PROCESSING_SPECS["config_schema"].items()}


def tile_filter_classification(tile: dict, ctx: dict) -> dict:
    """Cull points by their LAS class (this node never just tags)."""
    import numpy as np

    def mask_fn(record) -> np.ndarray:
        raw = str(ctx.get("classes", "2"))
        try:
            selected = {int(c) for c in raw.split(",") if c.strip()}
        except ValueError:
            raise ValueError(
                f"Filter by Classification: invalid classes {raw!r} "
                "(expected comma-separated integers, e.g. '2' or '7,18')"
            )
        # 256-entry LUT: classification is uint8, so one fancy-index lookup
        # replaces np.isin (sort/search) entirely.
        lut = np.zeros(256, dtype=bool)
        lut[[c for c in selected if 0 <= c <= 255]] = True
        cls = np.asarray(record.classification)
        keep = lut[cls]
        if ctx.get("mode", "keep") == "remove":
            return ~keep
        return keep

    return run_clean(
        tile, ctx, NODE_ID, "clean_class", mask_fn, force_remove=True
    )