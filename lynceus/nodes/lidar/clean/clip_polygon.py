# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Node -- Clip by Polygon (area-of-interest mask).

Keeps only points inside a WKT polygon (plus an optional outward buffer)
and physically deletes the rest, tile by tile. The polygon is read as
planar XY in the cloud CRS: no reprojection is applied, so WKT
coordinates must already match the survey CRS.
"""

from __future__ import annotations

from lynceus.nodes._point_source import run_clean
from lynceus.nodes.ports import PortType

NODE_ID = "lynceus.nodes.lidar.clean.clip_polygon"
NODE_NAME = "Clip by Polygon"
NODE_CATEGORY = "LiDAR"

NODE_SUBCATEGORY = "Cleaning"

NODE_DESCRIPTION = (
    "<b>Clip by Polygon</b> -- Area of Interest Mask<br><br>"
    "Keeps only points inside a WKT polygon (plus an optional outward "
    "buffer) and deletes the rest permanently, tile by tile. Boundary "
    "points are kept. The polygon is planar XY in the cloud CRS: no "
    "reprojection is applied.<br><br>"
    "<b>Tips:</b> Paste the polygon WKT in the Inspector. An empty "
    "polygon blocks the run instead of passing everything (fail-loud "
    "beats silent)."
)

INPUTS = (PortType.POINT_CLOUD,)
OUTPUTS = (PortType.POINT_CLOUD,)

PROCESSING_SPECS = {
    "tile_task": "tile_clip_polygon",
    "input_port": "point_cloud",
    "output_port": "point_cloud",
    "passes_flags": ("classified",),
    "point_cloud_dir": "clip_polygon",
    "output_globs": ("clip_polygon/**/*.laz",),
    "config_schema": {
        "polygon_wkt": {
            "type": "str",
            "default": "",
            "description": "WKT polygon selecting the area of interest.",
            "impact": "Coordinates must be in the cloud CRS (no reprojection is applied). Points outside (plus the buffer) are deleted permanently. Empty text blocks the run: an empty polygon is almost always a mistake.",
            "group": "Area",
        },
        "buffer_m": {
            "type": "float",
            "default": 0.0,
            "minimum": 0.0,
            "maximum": 1000.0,
            "description": "Outward buffer around the polygon (meters).",
            "impact": "Positive values keep a margin (useful against edge artifacts); 0 clips exactly. Applied in cloud CRS units.",
            "group": "Area",
        },
    },
}


def get_config_defaults() -> dict:
    return {k: v["default"] for k, v in PROCESSING_SPECS["config_schema"].items()}


def _load_polygon(wkt: str, buffer_m: float):
    """Parse and buffer the AOI polygon, failing loud on bad input."""
    import shapely

    text = (wkt or "").strip()
    if not text:
        raise RuntimeError(
            f"{NODE_ID}: polygon_wkt is empty; paste a WKT polygon "
            "instead of clipping nothing"
        )
    try:
        geom = shapely.from_wkt(text)
    except Exception as exc:
        raise RuntimeError(
            f"{NODE_ID}: polygon_wkt is not valid WKT ({exc})"
        ) from exc
    if geom.is_empty:
        raise RuntimeError(f"{NODE_ID}: polygon_wkt is an empty geometry")
    if buffer_m and float(buffer_m) > 0:
        geom = geom.buffer(float(buffer_m))
    return geom


def tile_clip_polygon(tile: dict, ctx: dict) -> dict:
    """Keep points inside the WKT polygon; drop the rest unconditionally."""
    import numpy as np
    import shapely

    geom = _load_polygon(str(ctx.get("polygon_wkt", "")),
                         float(ctx.get("buffer_m", 0.0)))

    def mask_fn(record) -> np.ndarray:
        x = np.asarray(record.x, dtype=np.float64)
        y = np.asarray(record.y, dtype=np.float64)
        if x.size == 0:
            return np.zeros(0, dtype=bool)
        # The geometry parsed cleanly above; a failure here is a real
        # bug and must fail the tile, never pass silently.
        return np.asarray(shapely.covers(geom, shapely.points(x, y)),
                         dtype=bool)

    return run_clean(tile, ctx, NODE_ID, "clip_polygon", mask_fn,
                     force_remove=True)
