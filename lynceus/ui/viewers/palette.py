# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Shared elevation and metric colormaps for the viewers."""

import numpy as np

ELEV_STOPS = np.array(
    [
        [0.02, 0.16, 0.45],  # Deep blue.
        [0.05, 0.35, 0.25],  # Green.
        [0.55, 0.75, 0.30],  # Yellow-green.
        [0.95, 0.85, 0.25],  # Yellow.
        [0.55, 0.35, 0.15],  # Brown.
    ],
    dtype=np.float32,
)
ELEV_POSITIONS = np.array([0.0, 0.4, 0.7, 0.9, 1.0], dtype=np.float32)

# Metric ramp (RdYlGn): low green -> yellow -> high red.
METRIC_STOPS = np.array(
    [
        [0.28, 0.55, 0.22],  # Green.
        [0.88, 0.82, 0.28],  # Yellow.
        [0.83, 0.32, 0.30],  # Red.
    ],
    dtype=np.float32,
)
METRIC_POSITIONS = np.array([0.0, 0.5, 1.0], dtype=np.float32)


def elevation_colors(z: np.ndarray) -> np.ndarray:
    """Map elevations to RGB (..., 3) float32 values in the 0..1 range."""
    z_min = float(z.min())
    z_range = float(z.max() - z_min) or 1.0
    t = np.clip((z - z_min) / z_range, 0.0, 1.0).ravel()
    colors = np.empty((t.size, 3), dtype=np.float32)
    for channel in range(3):
        colors[:, channel] = np.interp(t, ELEV_POSITIONS, ELEV_STOPS[:, channel])
    return colors.reshape(z.shape + (3,))


def metric_colors(t: np.ndarray) -> np.ndarray:
    """Map normalized 0..1 values to RGB using the metric ramp."""
    t = np.clip(t, 0.0, 1.0).ravel()
    colors = np.empty((t.size, 3), dtype=np.float32)
    for channel in range(3):
        colors[:, channel] = np.interp(t, METRIC_POSITIONS, METRIC_STOPS[:, channel])
    return colors.reshape(colors.size // 3, 3)