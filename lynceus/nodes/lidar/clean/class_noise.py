# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Node -- Classify Noise (SOR / ROR outlier detection).

Statistical outlier detection on the local neighborhood:

- ``sor`` (Statistical Outlier Removal): points whose mean distance to their
  k nearest neighbors exceeds ``mean + multiplier * std`` are flagged.
  Isolated clouds far from the surface (birds, sensor artifacts) are the
  classic case -> tag class 18 (high noise).
- ``ror`` (Radius Outlier Removal): points with fewer than ``min_points``
  neighbors within ``radius`` meters are flagged -> isolated low points
  under the terrain (drop artifacts) -> tag class 7 (low point).
"""

from __future__ import annotations

from lynceus.nodes._point_source import NOISE_HIGH_CLASS, NOISE_LOW_CLASS, run_clean
from lynceus.nodes.ports import PortType

NODE_ID = "lynceus.nodes.lidar.clean.class_noise"
NODE_NAME = "Classify Noise"
NODE_CATEGORY = "LiDAR"

NODE_SUBCATEGORY = "Cleaning"

NODE_DESCRIPTION = (
    "<b>Classify Noise</b> -- Outlier Detection (SOR / ROR)<br><br>"
    "Flags isolated points that do not belong to a surface:<br><br>"
    "&bull; <b>Statistical Outlier Removal (SOR)</b> -- points whose mean "
    "distance to their k nearest neighbors exceeds the global mean + "
    "multiplier x std. Catches isolated high noise (birds, sensor "
    "artifacts, multi-return ghosts).<br>"
    "&bull; <b>Radius Outlier Removal (ROR)</b> -- points with fewer than "
    "min points inside a radius. Catches sparse low points under the "
    "terrain (drop artifacts).<br><br>"
    "<b>Tips:</b> Run SOR early in the chain (default tag class 18). For "
    "low isolated points use ROR and tag class 7. Verify the tagged classes "
    "with a color-by-class viewer before removing."
)

INPUTS = (PortType.POINT_CLOUD,)
OUTPUTS = (PortType.POINT_CLOUD,)

PROCESSING_SPECS = {
    "tile_task": "tile_class_noise",
    "input_port": "point_cloud",
    "output_port": "point_cloud",
    "passes_flags": ("classified",),
    "point_cloud_dir": "clean_noise",
    "output_globs": ("clean_noise/**/*.laz",),
    "config_schema": {
        "method": {
            "type": "str",
            "default": "sor",
            "options": ["sor", "ror"],
            "description": "Outlier detection algorithm.",
            "impact": "sor (Statistical Outlier Removal) flags isolated high noise using neighborhood distances. ror (Radius Outlier Removal) flags isolated low points too sparse to belong to a surface.",
            "group": "Algorithm",
        },
        "sor_points": {
            "type": "int",
            "default": 6,
            "minimum": 2,
            "maximum": 100,
            "description": "Number of nearest neighbors (k) used by SOR.",
            "impact": "Lower k is more sensitive to local gaps; higher k is more stable but may miss thin clusters of noise. 6-12 is the common range.",
            "group": "SOR",
        },
        "sor_multiplier": {
            "type": "float",
            "default": 3.0,
            "minimum": 0.1,
            "maximum": 20.0,
            "description": "Standard-deviation multiplier for the SOR threshold.",
            "impact": "Lower values flag more points as noise (aggressive). Higher values only flag extreme outliers. 2.0-4.0 is the common range.",
            "group": "SOR",
        },
        "ror_radius_m": {
            "type": "float",
            "default": 5.0,
            "minimum": 0.1,
            "maximum": 100.0,
            "description": "Search radius (meters) used by ROR.",
            "impact": "Points isolated from any neighbor inside this radius are suspect. Smaller radii require denser neighborhoods. Scale it with the point spacing of the survey.",
            "group": "ROR",
        },
        "ror_min_points": {
            "type": "int",
            "default": 3,
            "minimum": 1,
            "maximum": 100,
            "description": "Minimum neighbors inside the radius for a point to be kept.",
            "impact": "Higher values reject sparser patches, useful when the cloud has thin low artifacts below the surface.",
            "group": "ROR",
        },
        "action": {
            "type": "str",
            "default": "classify",
            "options": ["classify", "remove"],
            "description": "How to handle detected outliers.",
            "impact": "classify (default) tags them without touching ground points (unless protect ground is off) and keeps the cloud complete; use remove only after reviewing the tagged classes.",
            "group": "Behavior",
        },
        "protect_ground": {
            "type": "bool",
            "default": True,
            "description": "Keep ground points (class 2) untouched when tagging.",
            "impact": "On (default) preserves current behavior: ground is never retagged, even outside the range. Off lets the node retag ground as noise so a later remove can delete it — bare-earth products lose those points and degrade with a warning.",
            "group": "Behavior",
        },
        "noise_class": {
            "type": "int",
            "default": 18,
            "minimum": 0,
            "maximum": 31,
            "description": "LAS class assigned to detected outliers when action=classify.",
            "impact": "ASPRS: 18 for high noise (SOR candidates), 7 for low points (ROR candidates). Use two Classify Noise nodes with different tags to separate both families.",
            "group": "Output",
        },
    },
}


def get_config_defaults() -> dict:
    return {k: v["default"] for k, v in PROCESSING_SPECS["config_schema"].items()}


def tile_class_noise(tile: dict, ctx: dict) -> dict:
    """Flag statistical/radius outliers on the local neighborhood."""
    import numpy as np
    from scipy.spatial import cKDTree

    def mask_fn(record) -> np.ndarray:
        # Rebase to the local centroid before the float32 cast: raw UTM
        # coordinates in the millions quantize to ~0.5 m in float32, which
        # collapses real point spacing. The offset keeps sub-millimeter
        # precision at half the tree bandwidth. Workers stay
        # single-threaded on purpose (the pool already sizes processes
        # to CPUs; inner threads would oversubscribe).
        coords = np.column_stack(
            (
                np.asarray(record.x, dtype=np.float64),
                np.asarray(record.y, dtype=np.float64),
                np.asarray(record.z, dtype=np.float64),
            )
        )
        n = len(coords)
        if n <= 1:
            # Neighborhood statistics are undefined for a lone point;
            # keep it conservatively instead of flagging on self-count.
            return np.ones(n, dtype=bool)
        points = (coords - coords.mean(axis=0)).astype(np.float32)
        method = ctx.get("method", "sor")
        if method == "sor":
            k = min(int(ctx.get("sor_points", 6)), max(n - 1, 1))
            neighbors = cKDTree(points, leafsize=32).query(points, k=k + 1)[0]
            mean = neighbors[:, 1:].mean(axis=1)
            multiplier = float(ctx.get("sor_multiplier", 3.0))
            threshold = mean.mean() + multiplier * mean.std()
            return mean <= threshold
        radius = float(ctx.get("ror_radius_m", 5.0))
        min_k = int(ctx.get("ror_min_points", 3))
        # A point has >= min_k neighbors within the radius iff its
        # (min_k-1)-th nearest distance (self included at 0) is within
        # it: an early-exit kNN threshold, equivalent to counting every
        # neighbor in the radius. Radius counting explodes at modern
        # densities (~21k neighbors per point at 278 pts/m2 with r=5 m).
        if min_k <= 1:
            return np.ones(n, dtype=bool)
        if min_k > n:
            return np.zeros(n, dtype=bool)
        dists = cKDTree(points, leafsize=32).query(points, k=min_k)[0]
        return dists[:, min_k - 1] <= radius

    default_class = (
        NOISE_HIGH_CLASS if ctx.get("method", "sor") == "sor" else NOISE_LOW_CLASS
    )
    return run_clean(
        tile, ctx, NODE_ID, "clean_noise", mask_fn,
        cls_value=int(ctx.get("noise_class", default_class)),
        protect_ground=bool(ctx.get("protect_ground", True)),
    )