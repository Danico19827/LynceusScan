# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Raster processing utilities for point-cloud tasks without Qt.

The module provides progressive morphological filtering, core-masked tile
rasterization, GeoTIFF I/O, and aligned mosaic merging.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
from affine import Affine
from rasterio.merge import merge as rio_merge
from scipy.ndimage import minimum_filter, maximum_filter, distance_transform_edt, gaussian_filter

logger = logging.getLogger(__name__)

RASTER_CELL_SIZE = 1.0
RASTER_GRID_BYTES_PER_CELL = 32
DEFAULT_RASTER_GRID_BUDGET_BYTES = 512 * 1024 * 1024
POINT_GRID_BYTES_PER_CELL = 96
NODATA = -9999.0
GROUND_CLASS = 2

GTIFF_CREATION = {"bigtiff": "YES"}
"""Shared GeoTIFF creation options: BigTIFF removes the 4 GiB classic
limit that killed 0.3 m mosaics mid-write (Fairbanks 20261014: three
derivatives dead at exactly 4096 MB). One place so no future writer
repeats it; universally readable (GDAL/QGIS/viewer)."""

_GRID_BUDGET_FLOOR_BYTES = 64 * 1024 * 1024
"""Floor for per-worker grid budgets: instantaneous free RAM collapses
under page-cache pressure (dozens of GB streaming through workers) and
must never refuse grids that fit the configured share. Genuinely huge
grids still refuse against the configured cap."""

_GRID_BUDGET_RETRIES = 3
_GRID_BUDGET_RETRY_S = 10.0
"""Transient-pressure backoff: a refused grid waits for siblings to drain
before failing, so legitimate work survives a hot minute (slower, never
dead) while real exhaustion still reports with numbers."""


def _free_bytes_now() -> int | None:
    """Free RAM right now, or None when unreadable."""
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except (ImportError, OSError, AttributeError, ValueError):
        return None


def _stable_grid_budget(memory_budget_bytes: int | None) -> tuple[int, str]:
    """Per-worker grid budget plus a forensics note (both numbers).

    The configured share is stable across the run; the free-RAM term only
    ever lowers it to the floor, so a hot page cache cannot zero it.
    Returns ``(budget, detail)`` with ``configured=XMiB/free=YMiB``.
    """
    configured = (
        int(memory_budget_bytes)
        if memory_budget_bytes is not None and int(memory_budget_bytes) > 0
        else DEFAULT_RASTER_GRID_BUDGET_BYTES
    )
    free = _free_bytes_now()
    if free is None:
        return max(1, configured), f"configured={configured // (1024 ** 2)}MiB"
    headroom = max(free // 4, _GRID_BUDGET_FLOOR_BYTES)
    budget = max(1, min(configured, headroom))
    detail = (f"configured={configured // (1024 ** 2)}MiB/"
              f"free={free // (1024 ** 2)}MiB")
    return budget, detail


def _admit_grid(estimated_bytes: int, memory_budget_bytes: int | None,
                describe: str) -> int:
    """Return a budget covering ``estimated_bytes`` or raise with numbers.

    Transient pressure waits (bounded retries holding a pool slot:
    backpressure, not idleness); genuine exhaustion raises a ValueError
    carrying required/configured/free so F0 persists the real cause.
    """
    import time

    budget, detail = _stable_grid_budget(memory_budget_bytes)
    if estimated_bytes <= budget:
        return budget
    for _ in range(_GRID_BUDGET_RETRIES):
        logger.warning(
            "Grid budget exceeded (%s needs %s; %s): waiting for RAM",
            describe, f"{estimated_bytes // (1024 ** 2)}MiB", detail,
        )
        time.sleep(_GRID_BUDGET_RETRY_S)
        budget, detail = _stable_grid_budget(memory_budget_bytes)
        if estimated_bytes <= budget:
            return budget
    raise ValueError(
        f"{describe} would require {estimated_bytes // (1024 ** 2):,} MiB, "
        f"above the per-worker grid budget ({budget // (1024 ** 2):,} MiB; "
        f"{detail}). Increase cell size or reduce tile size / worker count."
    )


def _point_grid_shape(
    x: np.ndarray,
    y: np.ndarray,
    cell_size: float,
    memory_budget_bytes: int | None,
    method: str,
    tile_id: str | None,
) -> tuple[float, float, int, int]:
    """Validate point-derived raster dimensions before allocating grid arrays."""
    try:
        resolution = float(cell_size)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{method} cell size must be numeric") from exc
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError(
            f"{method} cell size must be finite and greater than 0; got {cell_size!r}"
        )
    if x.size == 0 or y.size == 0:
        return 0.0, 0.0, 1, 1

    x0, y0 = float(np.min(x)), float(np.min(y))
    x_max, y_max = float(np.max(x)), float(np.max(y))
    if not all(math.isfinite(value) for value in (x0, y0, x_max, y_max)):
        raise ValueError(f"{method} received non-finite point coordinates")
    col_ratio = (x_max - x0) / resolution
    row_ratio = (y_max - y0) / resolution
    if not math.isfinite(col_ratio) or not math.isfinite(row_ratio):
        raise ValueError(f"{method} grid dimensions are not finite")
    cols = max(1, math.ceil(col_ratio) + 1)
    rows = max(1, math.ceil(row_ratio) + 1)
    estimated_bytes = rows * cols * POINT_GRID_BYTES_PER_CELL
    label = f" for tile {tile_id}" if tile_id else ""
    _admit_grid(
        estimated_bytes, memory_budget_bytes,
        f"{method} grid{label} at {resolution:g} m "
        f"({rows:,} x {cols:,} cells)",
    )
    return x0, y0, cols, rows

# PMF defaults; node configuration can override them.
PMF_SLOPE = 0.3
PMF_INTERCEPT = 0.3
PMF_INITIAL_WINDOW_M = 2.0
PMF_MAX_WINDOW_M = 8.0

# MCC defaults; node configuration can override them.
MCC_CELL_SIZE = 1.0
MCC_SCALE = 1.5
MCC_THRESHOLD = 0.3
MCC_DOMAINS = 3
MCC_ITERATIONS = 10
MCC_MAX_FIT_CENTERS = 1500
MCC_CONVERGENCE = 0.005

# ATIN defaults; node configuration can override them.
ATIN_CELL_SIZE = 1.0
ATIN_SEED_M = 20.0
ATIN_MAX_ANGLE = 6.0
ATIN_MAX_DIST = 1.0
ATIN_ITERATIONS = 10


def fill_holes(
    array: np.ndarray,
    resolution: float,
    max_dist_m: float,
    nodata: float = NODATA,
) -> np.ndarray:
    """Fill small gaps by copying the nearest valid cell.

    Uses scipy distance_transform_edt to find, for every NODATA cell, the
    closest valid cell and its value. Cells farther than *max_dist_m* from
    any valid data remain NODATA (avoids extrapolation artifacts at edges).
    """
    valid = array != nodata
    if valid.all() or not valid.any():
        return array
    dist_pixels, indices = distance_transform_edt(
        ~valid, return_distances=True, return_indices=True
    )
    filled = array[indices[0], indices[1]]
    dist_meters = dist_pixels * resolution
    filled[dist_meters > max_dist_m] = nodata
    return filled


def smooth_nodata(
    arr: np.ndarray,
    sigma: float,
    nodata: float = NODATA,
) -> np.ndarray:
    """Gaussian smoothing that ignores NODATA cells.

    Instead of smoothing the raw array (which contaminates valid cells
    near NODATA boundaries with -9999.0), this computes a weighted average
    using only valid contributions.
    """
    mask = (arr != nodata).astype(np.float32)
    if mask.sum() == 0:
        return arr
    data = np.where(mask, arr, 0.0)
    smoothed_data = gaussian_filter(data, sigma=sigma)
    smoothed_weights = gaussian_filter(mask, sigma=sigma)
    result = arr.copy()
    # Only update originally-valid cells; NODATA cells stay NODATA.
    valid = (arr != nodata) & (smoothed_weights > 0)
    result[valid] = (smoothed_data[valid] / smoothed_weights[valid]).astype(arr.dtype)
    return result


def median_nodata(
    arr: np.ndarray,
    size: int = 3,
    nodata: float = NODATA,
) -> np.ndarray:
    """Median filter where only valid cells vote inside each window.

    Each valid cell is replaced by the median of the valid values in its
    neighborhood; NODATA cells are never used as votes, so output edges stay
    clean (no -9999.0 bleeding into valid data like a plain scipy filter).
    NODATA cells remain NODATA.

    Banded NaN-median (reflect-padded like ``scipy.ndimage`` defaults):
    vectorized like ``generic_filter`` output, without the per-window Python
    callback, and with bounded memory on large mosaics.
    """
    import warnings

    from numpy.lib.stride_tricks import sliding_window_view

    valid = arr != nodata
    if valid.sum() == 0:
        return arr
    if size <= 1:
        return arr.copy()
    pad = size // 2
    data = np.where(valid, arr, np.nan).astype(np.float32)
    padded = np.pad(data, pad, mode="reflect")
    result = arr.copy()
    # Row bands with halo: peak ~= band * width * size^2 floats.
    band = max(1, min(arr.shape[0], 512))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for start in range(0, arr.shape[0], band):
            stop = min(arr.shape[0], start + band)
            view = sliding_window_view(
                padded[start : stop + 2 * pad, :], (size, size)
            )
            med = np.nanmedian(view.reshape(stop - start, -1, size * size), axis=2)
            chunk_valid = valid[start:stop]
            chunk_out = result[start:stop]
            chunk_out[chunk_valid] = med[chunk_valid].astype(arr.dtype)
    return result


def _crs_value(crs: str | None) -> str | None:
    """Normalize tiler CRS values for rasterio."""
    if not crs or crs == "Unknown":
        return None
    return crs


# ---------------------------------------------------------------------------
# Progressive morphological filter (PMF).
# ---------------------------------------------------------------------------

def progressive_morphological_filter(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cell_size: float = RASTER_CELL_SIZE,
    slope: float = PMF_SLOPE,
    intercept: float = PMF_INTERCEPT,
    initial_window_m: float = PMF_INITIAL_WINDOW_M,
    max_window_m: float = PMF_MAX_WINDOW_M,
    memory_budget_bytes: int | None = None,
    tile_id: str | None = None,
) -> np.ndarray:
    """Return a boolean ground mask using the Zhang et al. (2003) PMF.

    Reference: Zhang, Chen, Whitman, Shyu, Yan, Zhang, "A Progressive
    Morphological Filter for Removing Nonground Measurements From Airborne
    LIDAR Data", IEEE Trans. Geosci. Remote Sens. 41(4):872-878, 2003
    (DOI 10.1109/TGRS.2003.814625).

    Math implemented: a minimum-elevation surface is opened with a growing
    series of odd windows ``w_k`` (cells; ``+2`` per iteration from the
    initial window up to the maximum). At iteration ``k`` cells are removed
    when ``surface - opened > dh_T(k)`` with the paper's differential
    threshold ``dh_T(k) = s * (w_k - w_{k-1}) * c + dh_0`` (``w_0 = 0``),
    where ``s`` = ``slope``, ``c`` = ``cell_size`` and ``dh_0`` =
    ``intercept``. Removed cells take the opened value for the next
    iteration. Units: meters throughout.

    Deliberate refinement vs the paper: the paper labels every point inside
    a surviving cell as ground; we additionally require ``z <= surface +
    s*c + dh_0`` to suppress low-vegetation commission inside ground cells.
    On slopes steeper than ``s`` this gate may omit true ground — use
    CSF/ATIN there (see their fichas). Standards: output codes travel via
    ``output_class`` (default 2 = ground per ASPRS LAS 1.4); ``withheld``
    points are excluded upstream and never labeled (see _classify_base).
    """
    if x.size == 0:
        return np.zeros(0, dtype=bool)

    x0, y0, cols, rows = _point_grid_shape(
        x, y, cell_size, memory_budget_bytes, "PMF", tile_id
    )

    c = np.floor((x - x0) / cell_size).astype(np.int64)
    r = np.floor((y - y0) / cell_size).astype(np.int64)
    np.clip(c, 0, cols - 1, out=c)
    np.clip(r, 0, rows - 1, out=r)
    flat = r * cols + c

    # np.minimum.at is substantially faster than sort/reduceat for this
    # aggregation on the supported NumPy versions.
    grid = np.full(rows * cols, np.inf, dtype=np.float64)
    np.minimum.at(grid, flat, z)
    grid = grid.reshape(rows, cols)
    valid = np.isfinite(grid)

    if not valid.any():
        return np.zeros(x.size, dtype=bool)

    fill = float(grid[valid].min())
    surface = np.where(valid, grid, fill)
    ground = valid.copy()

    def _odd_cells(meters: float) -> int:
        n = max(3, int(round(meters / cell_size)))
        return n if n % 2 else n + 1

    window_cells = _odd_cells(initial_window_m)
    max_cells = max(window_cells, _odd_cells(max_window_m))
    last_cells = 0
    while True:
        opened = maximum_filter(minimum_filter(surface, size=window_cells), size=window_cells)
        diff = surface - opened
        threshold = slope * (window_cells - last_cells) * cell_size + intercept
        removed = ground & (diff > threshold)
        if removed.any():
            ground &= ~removed
            surface = np.where(removed, opened, surface)
        last_cells = window_cells
        if window_cells >= max_cells:
            break
        window_cells = min(max_cells, window_cells + 2)

    cell_idx = np.flatnonzero(ground)
    surface_flat = surface.reshape(-1)
    cell_lookup = np.full(rows * cols, np.nan, dtype=np.float64)
    cell_lookup[cell_idx] = surface_flat[cell_idx]

    band = slope * cell_size + intercept
    is_ground = cell_lookup[flat]
    return np.isfinite(is_ground) & (z <= is_ground + band)


def cloth_simulation_filter(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cell_size: float = 0.5,
    class_threshold: float = 0.5,
    rigidness: int = 3,
    iterations: int = 1000,
    gravity: float = 0.05,
    memory_budget_bytes: int | None = None,
    tile_id: str | None = None,
) -> np.ndarray:
    """Return a boolean ground mask using a simplified Cloth Simulation
    Filter (Zhang et al., 2016).

    Reference: Zhang, Qi, Hu, Zhong, Wu, Zhang, "An Easy-to-Use Airborne
    LiDAR Data Filtering Method Based on Cloth Simulation", Remote Sens.
    8(4):283, 2016 (DOI 10.3390/rs8040283).

    Math implemented: heights are flipped (``zinv = max - z``) so bare
    earth becomes the ceiling; a cloth grid starts above everything and
    settles under uniform gravity (``gravity`` meters per iteration),
    spring relaxation toward the 4-neighbor average (``rigidness`` passes
    per step) and collision with the per-cell maximum. Points within
    ``class_threshold`` meters under the settled cloth are ground; small
    objects stay buried because neighbor tension holds the cloth above
    them. Units: meters throughout. Gravity is deliberately small
    (0.05 m/iteration): tension then dominates and the cloth bridges
    roof pits instead of sagging into them (flat roofs would otherwise
    read as ground); ``iterations`` covers 50 m of relief at the default.

    Simplified vs the paper: uniform gravity, fixed stiffness from
    ``rigidness`` (1 = flat/loosely draped, 3 = tight/steep-following),
    nearest-cell collision, no slope post-processing. ``rigidness``
    accepts 1..3. Standards: output codes travel via ``output_class``
    (default 2 = ground per ASPRS LAS 1.4); ``withheld`` points are
    excluded upstream and never labeled (see _classify_base).
    """
    if x.size == 0:
        return np.zeros(0, dtype=bool)

    x0, y0, cols, rows = _point_grid_shape(
        x, y, cell_size, memory_budget_bytes, "CSF", tile_id
    )
    c = np.clip(np.floor((x - x0) / cell_size).astype(np.int64), 0, cols - 1)
    r = np.clip(np.floor((y - y0) / cell_size).astype(np.int64), 0, rows - 1)
    flat = r * cols + c

    zinv = float(z.max()) - z
    surf = np.full(rows * cols, -np.inf, dtype=np.float64)
    np.maximum.at(surf, flat, zinv)
    surf = surf.reshape(rows, cols)
    has = np.isfinite(surf)

    rigidness = int(min(3, max(1, rigidness)))
    iterations = int(max(1, iterations))
    cloth = np.full((rows, cols), float(zinv.max()), dtype=np.float64)
    padded = np.empty((rows + 2, cols + 2), dtype=np.float64)
    for _ in range(iterations):
        cloth -= gravity
        for _ in range(rigidness):
            padded[1:-1, 1:-1] = cloth
            padded[0, 1:-1] = cloth[0, :]
            padded[-1, 1:-1] = cloth[-1, :]
            padded[1:-1, 0] = cloth[:, 0]
            padded[1:-1, -1] = cloth[:, -1]
            padded[0, 0] = cloth[0, 0]
            padded[0, -1] = cloth[0, -1]
            padded[-1, 0] = cloth[-1, 0]
            padded[-1, -1] = cloth[-1, -1]
            avg = (
                padded[:-2, 1:-1]
                + padded[2:, 1:-1]
                + padded[1:-1, :-2]
                + padded[1:-1, 2:]
            ) * 0.25
            cloth += 0.5 * (avg - cloth)
        cloth = np.where(has, np.maximum(cloth, surf), cloth)

    cloth_c = cloth[r, c]
    return (cloth_c - zinv) < class_threshold


def simple_morphological_filter(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cell_size: float = 1.0,
    slope: float = 0.2,
    max_window_m: float = 18.0,
    threshold: float = 0.3,
    memory_budget_bytes: int | None = None,
    tile_id: str | None = None,
) -> np.ndarray:
    """Return a boolean ground mask using a simplified SMRF (Pingel 2013).

    Reference: Pingel, Clarke, McBride, "An Improved Simple Morphological
    Filter for Binary LiDAR Data", IEEE Geosci. Remote Sens. Lett.
    10(5):1105-1109, 2013.

    Math implemented: like PMF this opens a minimum-elevation surface
    with growing windows, but the elevation threshold stays FIXED at
    every scale (PMF grows it with the window) while the window itself
    adapts to local slope: steeper cells get larger windows, so
    flat/agricultural ground is cleaned aggressively without stripping
    real relief. Units: meters throughout. Simplified vs the paper:
    quantized window levels instead of truly per-cell windows, no final
    slope-based relabeling. Scope by design: flat-to-mild terrain;
    slopes go to PMF/CSF. Standards: output codes travel via
    ``output_class`` (default 2 = ground per ASPRS LAS 1.4); ``withheld``
    points are excluded upstream and never labeled (see _classify_base).
    """
    if x.size == 0:
        return np.zeros(0, dtype=bool)

    x0, y0, cols, rows = _point_grid_shape(
        x, y, cell_size, memory_budget_bytes, "SMRF", tile_id
    )
    c = np.clip(np.floor((x - x0) / cell_size).astype(np.int64), 0, cols - 1)
    r = np.clip(np.floor((y - y0) / cell_size).astype(np.int64), 0, rows - 1)
    flat = r * cols + c

    grid = np.full(rows * cols, np.inf, dtype=np.float64)
    np.minimum.at(grid, flat, z)
    grid = grid.reshape(rows, cols)
    valid = np.isfinite(grid)
    if not valid.any():
        return np.zeros(x.size, dtype=bool)

    fill = float(grid[valid].min())
    surface = np.where(valid, grid, fill)
    gy, gx = np.gradient(surface, cell_size, cell_size)
    steep = np.hypot(gx, gy)

    ground = valid.copy()
    max_cells = max(3, int(round(max_window_m / cell_size)))
    level = 3
    while True:
        opened = maximum_filter(
            minimum_filter(surface, size=level), size=level
        )
        diff = surface - opened
        # Slope-adaptive gating: gentle cells are tested from the first
        # levels (small features caught early); steeper cells wait for
        # larger windows. The threshold itself stays fixed at every
        # scale, unlike PMF's growing one.
        need = 3.0 + (steep / max(slope, 1e-9)) * max(max_cells - 3, 0)
        due = ground & (np.minimum(need, max_cells) <= level)
        removed = due & (diff > threshold)
        ground &= ~removed
        surface = np.where(removed, opened, surface)
        if level >= max_cells or not ground.any():
            break
        level = min(max_cells, level * 2)

    cell_idx = np.flatnonzero(ground)
    surface_flat = surface.reshape(-1)
    cell_lookup = np.full(rows * cols, np.nan, dtype=np.float64)
    cell_lookup[cell_idx] = surface_flat[cell_idx]
    is_ground = cell_lookup[flat]
    return np.isfinite(is_ground) & (z <= is_ground + threshold)


# ---------------------------------------------------------------------------
# Multiscale Curvature Classification (MCC).
# ---------------------------------------------------------------------------

def multiscale_curvature_filter(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cell_size: float = MCC_CELL_SIZE,
    scale: float = MCC_SCALE,
    threshold: float = MCC_THRESHOLD,
    domains: int = MCC_DOMAINS,
    max_iterations: int = MCC_ITERATIONS,
    memory_budget_bytes: int | None = None,
    tile_id: str | None = None,
) -> np.ndarray:
    """Return a boolean ground mask using Multiscale Curvature
    Classification (Evans & Hudak, 2007).

    Reference: Evans, Hudak, "A Multiscale Curvature Algorithm for
    Classifying Discrete Return LiDAR in Forested Environments", IEEE
    Trans. Geosci. Remote Sens. 45(4):1029-1038, 2007
    (DOI 10.1109/TGRS.2006.890412).

    Math implemented: a minimum-elevation surface is fit with a thin-plate
    spline (``scipy.interpolate.RBFInterpolator``, thin-plate kernel);
    cells deviating more than ``threshold`` meters above the plate are
    removed as non-ground and the plate refits until fewer than 0.5% of
    cells change (``max_iterations`` cap). This repeats over ``domains``
    scale domains with plate slack ``(scale + d) ** 2`` for domain ``d``,
    so coarser domains bridge larger objects. Only positive deviations
    are removed (pits below the plate stay ground, per the paper's
    default). Units: meters throughout.

    Mappings vs the paper (declared): the paper's scale parameter is
    carried as ridge smoothing ``(scale + d) ** 2`` of the RBF fit
    (heuristic validated on fixtures, see tests/test_ground_filter_math);
    the plate fits a deterministic strided subsample of at most 1500 cell
    centers (dense solve, exact on the subsample). Final labeling matches
    the PMF/SMRF convention (``z <= surface + threshold`` inside surviving
    cells). Standards: output codes travel via ``output_class`` (default 2
    = ground per ASPRS LAS 1.4); ``withheld`` points are excluded upstream
    and never labeled (see _classify_base).
    """
    from scipy.interpolate import RBFInterpolator

    if x.size == 0:
        return np.zeros(0, dtype=bool)

    x0, y0, cols, rows = _point_grid_shape(
        x, y, cell_size, memory_budget_bytes, "MCC", tile_id
    )
    c = np.clip(np.floor((x - x0) / cell_size).astype(np.int64), 0, cols - 1)
    r = np.clip(np.floor((y - y0) / cell_size).astype(np.int64), 0, rows - 1)
    flat = r * cols + c

    grid = np.full(rows * cols, np.inf, dtype=np.float64)
    np.minimum.at(grid, flat, z)
    grid = grid.reshape(rows, cols)
    valid = np.isfinite(grid)
    if not valid.any():
        return np.zeros(x.size, dtype=bool)

    ground = valid.copy()
    ys, ex = np.mgrid[0:rows, 0:cols]
    eval_pts = np.column_stack(
        (
            x0 + (ex.reshape(-1) + 0.5) * cell_size,
            y0 + (ys.reshape(-1) + 0.5) * cell_size,
        )
    )
    surface = grid.reshape(-1)
    for domain in range(max(1, int(domains))):
        smoothing = (float(scale) + domain) ** 2
        for _ in range(max(1, int(max_iterations))):
            idx = np.flatnonzero(ground)
            if idx.size == 0:
                break
            step = max(1, int(np.ceil(idx.size / MCC_MAX_FIT_CENTERS)))
            fit_idx = idx[::step]
            fit_pts = np.column_stack(
                (
                    x0 + (fit_idx % cols + 0.5) * cell_size,
                    y0 + (fit_idx // cols + 0.5) * cell_size,
                )
            )
            plate = RBFInterpolator(
                fit_pts,
                surface[fit_idx],
                kernel="thin_plate_spline",
                smoothing=smoothing,
            )(eval_pts).reshape(rows, cols)
            curvature = np.where(ground, grid - plate, -np.inf)
            removed = ground & (curvature > threshold)
            n_removed = int(removed.sum())
            if n_removed == 0:
                break
            ground &= ~removed
            if n_removed < max(1, MCC_CONVERGENCE * idx.size):
                break

    cell_lookup = np.where(ground, grid, np.nan).reshape(-1)
    is_ground = cell_lookup[flat]
    return np.isfinite(is_ground) & (z <= is_ground + threshold)


# ---------------------------------------------------------------------------
# Adaptive TIN ground filter (ATIN).
# ---------------------------------------------------------------------------

def adaptive_tin_filter(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cell_size: float = ATIN_CELL_SIZE,
    seed_m: float = ATIN_SEED_M,
    max_angle_deg: float = ATIN_MAX_ANGLE,
    max_dist_m: float = ATIN_MAX_DIST,
    max_iterations: int = ATIN_ITERATIONS,
    memory_budget_bytes: int | None = None,
    tile_id: str | None = None,
) -> np.ndarray:
    """Return a boolean ground mask using an Adaptive TIN filter
    (Axelsson, 2000).

    Reference: Axelsson, "DEM from laser scanner data using adaptive TIN
    models", Int. Arch. Photogramm. Remote Sens. 33(B4/1):110-117, 2000.

    Math implemented: a minimum-elevation surface is seeded with the
    lowest cell of each ``seed_m`` block; a Delaunay TIN over the seed
    centers densifies iteratively: an unclassified cell inside a facet
    joins the ground set when its absolute height above the facet plane
    is within ``max_dist_m`` AND its maximum angle to the three facet
    vertices (angle between vertex-to-point segment and facet plane) is
    within ``max_angle_deg``. Iteration stops when no cell joins or
    ``max_iterations`` is reached. Units: meters and degrees throughout.

    Mappings vs the paper (declared): the paper triangulates raw points;
    here cells stand in for points (grid-based adaptation, consistent
    with the PMF/CSF/SMRF/MCC implementations); the paper's maximum-edge
    guard is not implemented, so triangles may span unseeded gaps and the
    seed spacing (``seed_m``) must exceed the largest building. A
    hull-closing pass grows the ground set across 8-adjacency with the
    same distance criterion (the TIN alone never expands past the seed
    convex hull and would orphan a rim); steps taller than ``max_dist_m``
    still stop the growth. Final labeling matches the family convention
    (``z <= surface + max_dist_m`` inside surviving cells). Standards:
    output codes travel via ``output_class`` (default 2 = ground per
    ASPRS LAS 1.4); ``withheld`` points are excluded upstream and never
    labeled (see _classify_base).
    """
    from scipy.spatial import Delaunay, QhullError

    if x.size == 0:
        return np.zeros(0, dtype=bool)

    x0, y0, cols, rows = _point_grid_shape(
        x, y, cell_size, memory_budget_bytes, "ATIN", tile_id
    )
    c = np.clip(np.floor((x - x0) / cell_size).astype(np.int64), 0, cols - 1)
    r = np.clip(np.floor((y - y0) / cell_size).astype(np.int64), 0, rows - 1)
    flat = r * cols + c

    grid = np.full(rows * cols, np.inf, dtype=np.float64)
    np.minimum.at(grid, flat, z)
    grid = grid.reshape(rows, cols)
    valid = np.isfinite(grid)
    if not valid.any():
        return np.zeros(x.size, dtype=bool)

    surface = grid.reshape(-1)
    valid_flat = valid.reshape(-1)
    centers_x = x0 + (np.tile(np.arange(cols), rows) + 0.5) * cell_size
    centers_y = y0 + (np.repeat(np.arange(rows), cols) + 0.5) * cell_size

    # Seed: lowest valid cell of each seed_m block.
    block = max(1, int(round(float(seed_m) / cell_size)))
    block_r = np.repeat(np.arange(rows), cols) // block
    block_c = np.tile(np.arange(cols), rows) // block
    seed_idx = []
    for key in np.unique(np.stack((block_r[valid_flat],
                                   block_c[valid_flat]), axis=1), axis=0):
        in_block = ((block_r == key[0]) & (block_c == key[1])
                    & valid_flat)
        members = np.flatnonzero(in_block)
        if members.size:
            seed_idx.append(members[int(np.argmin(surface[members]))])
    ground = np.zeros(rows * cols, dtype=bool)
    if seed_idx:
        ground[np.asarray(seed_idx)] = True

    max_angle = float(np.deg2rad(max_angle_deg))
    for _ in range(max(1, int(max_iterations))):
        pts = np.column_stack((centers_x[ground], centers_y[ground]))
        if pts.shape[0] < 4:
            break
        try:
            tri = Delaunay(pts)
        except QhullError:
            break
        cand = np.flatnonzero(valid_flat & ~ground)
        if cand.size == 0:
            break
        simplex = tri.find_simplex(
            np.column_stack((centers_x[cand], centers_y[cand])))
        inside = simplex >= 0
        if not inside.any():
            break
        in_cand = cand[inside]
        in_simp = simplex[inside]
        verts = tri.simplices[in_simp]  # (M, 3) ground-set positions
        vert_flat = np.asarray(
            [np.flatnonzero(ground)[v] for v in verts.reshape(-1)]
        ).reshape(verts.shape)
        vx = centers_x[vert_flat]
        vy = centers_y[vert_flat]
        vz = surface[vert_flat]
        trans = tri.transform[in_simp]  # (M, 3, 2)
        rel = np.stack((centers_x[in_cand], centers_y[in_cand]),
                       axis=1) - trans[:, 2, :]
        bary12 = np.einsum("mij,mj->mi", trans[:, :2, :], rel)
        bary0 = 1.0 - bary12.sum(axis=1)
        plane_z = (bary0 * vz[:, 0] + bary12[:, 0] * vz[:, 1]
                   + bary12[:, 1] * vz[:, 2])
        dz = np.abs(surface[in_cand] - plane_z)
        # Max angle vertex-to-point vs facet plane.
        px = centers_x[in_cand]
        py = centers_y[in_cand]
        pz = surface[in_cand]
        e1x, e1y, e1z = vx[:, 1] - vx[:, 0], vy[:, 1] - vy[:, 0], vz[:, 1] - vz[:, 0]
        e2x, e2y, e2z = vx[:, 2] - vx[:, 0], vy[:, 2] - vy[:, 0], vz[:, 2] - vz[:, 0]
        nx = e1y * e2z - e1z * e2y
        ny = e1z * e2x - e1x * e2z
        nz = e1x * e2y - e1y * e2x
        norm_n = np.sqrt(nx * nx + ny * ny + nz * nz)
        worst = np.zeros(in_cand.size)
        for k in range(3):
            dx = px - vx[:, k]
            dy = py - vy[:, k]
            dzv = pz - vz[:, k]
            norm_d = np.sqrt(dx * dx + dy * dy + dzv * dzv)
            denom = norm_d * norm_n
            sin_a = np.where(denom > 0,
                             np.abs(dx * nx + dy * ny + dzv * nz) / np.maximum(
                                 denom, 1e-300), np.inf)
            worst = np.maximum(worst, np.arcsin(np.clip(sin_a, 0.0, 1.0)))
        add = np.zeros(rows * cols, dtype=bool)
        join = (dz <= float(max_dist_m)) & (worst <= max_angle)
        add[in_cand[join]] = True
        if not add.any():
            break
        ground |= add

    # Hull-closing pass: the TIN never expands past the seed convex hull,
    # which would orphan a rim of valid cells. Grow the ground set across
    # 8-adjacency with the same distance criterion (documented extension):
    # a cell joins when it stands within ``max_dist_m`` of the lowest
    # neighboring ground surface. Steps taller than the threshold (walls,
    # terrace faces, canopy) still stop the growth.
    from scipy.ndimage import binary_dilation, minimum_filter

    ground_2d = ground.reshape(rows, cols)
    for _ in range(2 * max(rows, cols)):
        adjacent = (
            binary_dilation(ground_2d, structure=np.ones((3, 3)))
            & valid
            & ~ground_2d
        )
        if not adjacent.any():
            break
        neighbor = minimum_filter(
            np.where(ground_2d, grid, np.inf), size=3
        )
        with np.errstate(invalid="ignore"):
            close_enough = np.abs(grid - neighbor) <= float(max_dist_m)
        grown = adjacent & close_enough
        if not grown.any():
            break
        ground_2d = ground_2d | grown
    ground = ground_2d.reshape(-1)

    cell_lookup = np.where(ground.reshape(rows, cols), grid,
                           np.nan).reshape(-1)
    is_ground = cell_lookup[flat]
    return np.isfinite(is_ground) & (z <= is_ground + float(max_dist_m))


# ---------------------------------------------------------------------------
# Terrain derivatives (Horn 1981).
# ---------------------------------------------------------------------------

def _horn_gradients(arr: np.ndarray, cell_x: float, cell_y: float,
                    nodata: float):
    """Horn 3x3 weighted east/north gradients with a validity mask.

    Only cells whose full 3x3 window is valid are computed; anything
    else (including the 1-cell rim) is left to the caller as NODATA.
    """
    from scipy.ndimage import minimum_filter

    valid = arr != nodata
    window_ok = minimum_filter(valid, size=3, mode="constant", cval=False)
    filled = np.where(valid, arr, 0.0).astype(np.float64)
    n = filled[:-2, 1:-1]
    s = filled[2:, 1:-1]
    e = filled[1:-1, 2:]
    w = filled[1:-1, :-2]
    ne = filled[:-2, 2:]
    nw = filled[:-2, :-2]
    se = filled[2:, 2:]
    sw = filled[2:, :-2]
    dzdx = ((ne + 2.0 * e + se) - (nw + 2.0 * w + sw)) / (8.0 * cell_x)
    dzdy = ((nw + 2.0 * n + ne) - (sw + 2.0 * s + se)) / (8.0 * cell_y)
    return dzdx, dzdy, window_ok[1:-1, 1:-1]


def slope_from_dem(arr: np.ndarray, cell_x: float, cell_y: float | None = None,
                   nodata: float = NODATA, z_factor: float = 1.0) -> np.ndarray:
    """Return slope in degrees using Horn (1981) 3x3 operators.

    Reference: Horn, "Hill shading and the reflectance map", Proc. IEEE
    69(1):14-47, 1981 (DOI 10.1109/PROC.1981.11918).

    Math implemented: ``slope = arctan(hypot(dzdx, dzdy))`` with Horn's
    weighted differences (opposite neighbor 1x, orthogonal 2x) over the
    cell size; ``z_factor`` exaggerates elevations first (ESRI-style).
    Output degrees in [0, 90]. Cells whose 3x3 window touches NODATA
    (including the 1-cell rim) read NODATA by design. Units: input meters,
    output degrees.
    """
    if cell_y is None:
        cell_y = cell_x
    out = np.full(arr.shape, nodata, dtype=np.float64)
    dzdx, dzdy, ok = _horn_gradients(arr, cell_x, cell_y, nodata)
    with np.errstate(invalid="ignore"):
        out[1:-1, 1:-1][ok] = np.degrees(
            np.arctan(np.hypot(dzdx[ok], dzdy[ok]) * z_factor))
    return out


def aspect_from_dem(arr: np.ndarray, cell_x: float, cell_y: float | None = None,
                    nodata: float = NODATA, z_factor: float = 1.0) -> np.ndarray:
    """Return aspect in degrees clockwise from north (Horn 1981).

    Same reference and operators as :func:`slope_from_dem`. Aspect is the
    direction of steepest DESCENT: ``atan2(-dzdx, -dzdy)`` folded to
    [0, 360), so a slope descending eastward reads 90 (East) and one
    descending northward reads 0 (North). Flat cells (zero gradient)
    read -1
    (ArcGIS-style "no aspect", declared): 0 always means north, never
    flat. Same NODATA window rule as slope. Units: input meters, output
    degrees.
    """
    if cell_y is None:
        cell_y = cell_x
    out = np.full(arr.shape, nodata, dtype=np.float64)
    dzdx, dzdy, ok = _horn_gradients(arr, cell_x, cell_y, nodata)
    with np.errstate(invalid="ignore"):
        raw = np.degrees(np.arctan2(-dzdx[ok] * z_factor,
                                     -dzdy[ok] * z_factor)) % 360.0
        flat = np.hypot(dzdx[ok], dzdy[ok]) == 0.0
        raw[flat] = -1.0
        out[1:-1, 1:-1][ok] = raw
    return out


def hillshade_from_dem(arr: np.ndarray, cell_x: float, cell_y: float | None = None,
                        nodata: float = NODATA, azimuth_deg: float = 315.0,
                        altitude_deg: float = 45.0,
                        z_factor: float = 1.0) -> np.ndarray:
    """Return hillshade 0-255 for a sun position (Horn 1981 + ESRI model).

    Same reference and operators as :func:`slope_from_dem`; shading
    follows the standard ``255 * (cos(zen)*cos(slope) + sin(zen)*
    sin(slope)*cos(az - aspect))`` with azimuth clockwise from north and
    ``zen = 90 - altitude``. Flat cells shade uniformly (the aspect term
    vanishes with slope) regardless of the -1 aspect convention. Same
    NODATA window rule as slope. Defaults 315/45 are the cartographic
    standard (northwest light). Units: input meters, output 0-255.
    """
    if cell_y is None:
        cell_y = cell_x
    out = np.full(arr.shape, nodata, dtype=np.float64)
    dzdx, dzdy, ok = _horn_gradients(arr, cell_x, cell_y, nodata)
    with np.errstate(invalid="ignore"):
        slope = np.arctan(np.hypot(dzdx[ok], dzdy[ok]) * z_factor)
        aspect = np.radians(np.degrees(
            np.arctan2(-dzdx[ok] * z_factor, -dzdy[ok] * z_factor)) % 360.0)
        zen = np.radians(90.0 - altitude_deg)
        az = np.radians(azimuth_deg)
        out[1:-1, 1:-1][ok] = 255.0 * (
            np.cos(zen) * np.cos(slope)
            + np.sin(zen) * np.sin(slope) * np.cos(az - aspect))
    return out


# ---------------------------------------------------------------------------
# Pit-free canopy compositing (Khosravipour et al. 2014, mosaic-level).
# ---------------------------------------------------------------------------

def pit_free_canopy(dsm: np.ndarray, nodata: float,
                    factors: tuple[int, ...] = (1, 2, 4)) -> np.ndarray:
    """Fill canopy pits with a multi-resolution maximum composite.

    Reference: Khosravipour, Skidmore, Isenburg, Larrañaga, "Generating
    Pit-free Canopy Height Models from Airborne Lidar",
    Photogramm. Eng. Remote Sens. 80(9):863-872, 2014
    (DOI 10.14358/PERS.80.9.863).

    Math implemented: the DSM is block-max downsampled at each factor,
    nearest-upsampled back, and composited per-pixel with a NODATA-aware
    maximum. A pit cell (low between high neighbors) takes the coarser
    maximum and fills. Units: input meters, output meters.

    Mappings vs the paper (declared): the paper assigns LiDAR points to
    multi-resolution rasters; here the already-gridded DSM mosaic is
    composited, so sub-cell pit geometry is lost and crown edges bias
    slightly upward (coarse maxima bleed outward). Use for tree
    segmentation, not for precise height measurement. All-NODATA stacks
    stay NODATA. Deterministic (no randomness, no ties to break).
    """
    grid = np.asarray(dsm, dtype=np.float64)
    rows, cols = grid.shape
    composite = np.full((rows, cols), -np.inf)
    any_valid = np.zeros((rows, cols), dtype=bool)
    for factor in dict.fromkeys(max(1, int(f)) for f in factors):
        if factor <= 1:
            coarse = grid
        else:
            import warnings

            pad_r = (-rows) % factor
            pad_c = (-cols) % factor
            padded = np.pad(grid, ((0, pad_r), (0, pad_c)), mode="edge")
            blocked = padded.reshape((rows + pad_r) // factor, factor,
                                     (cols + pad_c) // factor, factor)
            # All-NODATA blocks are legitimate (voids): nanmax warns
            # with a plain RuntimeWarning that errstate cannot catch.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                coarse_small = np.nanmax(
                    np.where(blocked == nodata, np.nan, blocked), axis=(1, 3))
            coarse = np.repeat(np.repeat(coarse_small, factor, axis=0),
                               factor, axis=1)[:rows, :cols]
        valid = np.isfinite(coarse) & (coarse != nodata)
        composite = np.where(valid & (coarse > composite), coarse, composite)
        any_valid |= valid
    return np.where(any_valid, composite, nodata)


# ---------------------------------------------------------------------------
# Per-tile rasterization masks.
# ---------------------------------------------------------------------------

def core_mask(x: np.ndarray, y: np.ndarray, tile: dict) -> np.ndarray:
    """Mask of points inside a tile core (half-open), excluding the buffer.

    The strict core is the right mask for writers whose per-tile partials
    merge back by cell KEY (a frontier cell split between two tiles is
    completed additively): every point lands in exactly one tile. Writers
    whose partials merge as PIXELS (first-wins mosaics) must use
    :func:`extent_mask` instead: the snapped grid can stick out of the
    core, and the strict mask would cut the shared frontier cell in half
    (a one-cell under-dense seam at resolutions that do not divide the
    tile size).
    """
    return (
        (x >= tile["core_x_min"])
        & (x < tile["core_x_max"])
        & (y >= tile["core_y_min"])
        & (y < tile["core_y_max"])
    )


def extent_mask(
    x: np.ndarray, y: np.ndarray, transform, cols: int, rows: int
) -> np.ndarray:
    """Mask of points inside a snapped raster grid (see ``tile_transform``).

    The grid bounds can stick out of the tile core by up to one cell
    (outward snapping keeps neighbors on ONE lattice, so mosaics merge
    exactly). Keeping the strict core would leave the frontier cell shared
    with the neighbor with only this tile's sliver; masking by the grid
    keeps it whole (the buffer holds the neighbor-side points).

    Edges inherit the core's semi-open convention: west/south inclusive,
    east/north exclusive (a point on a far grid limit belongs to the next
    grid and first-wins resolves the ≤1-cell overlap deterministically).
    Both limits compare with the snapping epsilon: a core point lying
    exactly on the south/west line divides to ``rows`` (or ``-0.0``) with
    float dust and would otherwise fall out of its own grid. The
    accumulator clips that index into the edge cell.
    """
    tol = 1e-7  # cells; same epsilon as the snapping, far below LAS scale
    c = (x - transform.c) / transform.a
    r = (transform.f - y) / (-transform.e)
    return (
        (c >= -tol)
        & (c < cols - tol)
        & (r > tol)
        & (r < rows + tol)
    )


def _snap_tile_bounds(tile: dict, x_min: float, x_max: float,
                      y_min: float, y_max: float,
                      resolution: float) -> tuple[float, float, float, float]:
    """Snap tile raster bounds outward to the tiling grid (exact merges).

    A tile size that is not a whole number of cells (100 m tiles at 0.3 m
    = 333.33 cells) leaves neighboring fragments on different sub-pixel
    phases: genuinely unmergeable without resampling, so the merger
    rejects them and the legacy fallback OOMs on big unions. Snapping
    west/south down and east/north up to res multiples from the tiling
    origin (``grid_ox``/``grid_oy``) puts every fragment on ONE exact
    grid at any resolution. Outward (never inward): no point drops out
    of rasterization; the ≤1-cell seam overlap resolves deterministically
    through first-wins merging. Dividing resolutions (1 m, 0.5 m, 2 m)
    snap to identity, bit for bit — the epsilon only absorbs float dust
    on mathematically-integer quotients. Missing anchor keys keep the
    legacy behavior (old manifests, synthetic tiles).
    """
    try:
        ox = float(tile.get("grid_ox"))
        oy = float(tile.get("grid_oy"))
    except (TypeError, ValueError):
        return x_min, x_max, y_min, y_max
    if not (math.isfinite(ox) and math.isfinite(oy)):
        return x_min, x_max, y_min, y_max

    def _snap(value: float, origin: float, down: bool) -> float:
        quotient = (value - origin) / resolution
        # Fixed epsilon: float dust at these magnitudes is ~1e-11 cells
        # (coordinates ~1e7 m in EPSG:3857); 1e-7 cells (~3e-8 m at 0.3 m)
        # sits far below any physically meaningful fraction (LAS stores
        # centimeters) while absorbing inexact-binary resolutions such
        # as 0.1 m. Near-grid origins snap exact.
        eps = 1e-7
        steps = math.floor(quotient + eps) if down else math.ceil(quotient - eps)
        return origin + steps * resolution

    return (
        _snap(x_min, ox, True),
        _snap(x_max, ox, False),
        _snap(y_min, oy, True),
        _snap(y_max, oy, False),
    )


def tile_transform(
    tile: dict,
    cell_size: float = RASTER_CELL_SIZE,
    memory_budget_bytes: int | None = None,
) -> tuple[Affine, int, int]:
    """Return bounded north-up grid dimensions for a tile core."""
    try:
        resolution = float(cell_size)
        bounds = tuple(
            float(tile[key])
            for key in ("core_x_min", "core_x_max", "core_y_min", "core_y_max")
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Tile raster bounds and cell size must be numeric") from exc
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError(
            f"Raster cell size must be finite and greater than 0; got {cell_size!r}"
        )
    if not all(math.isfinite(value) for value in bounds):
        raise ValueError("Tile raster bounds must be finite")

    x_min, x_max, y_min, y_max = bounds
    width = x_max - x_min
    height = y_max - y_min
    if width < 0 or height < 0:
        raise ValueError("Tile raster bounds have a negative extent")
    x_min, x_max, y_min, y_max = _snap_tile_bounds(
        tile, x_min, x_max, y_min, y_max, resolution)
    width = x_max - x_min
    height = y_max - y_min
    col_ratio = width / resolution
    row_ratio = height / resolution
    if not math.isfinite(col_ratio) or not math.isfinite(row_ratio):
        raise ValueError("Raster grid dimensions are not finite")
    # Snapped bounds sit on the lattice, so the ratio is an integer up to
    # float dust (334 * 0.3 re-added to a ~4.3e6 origin lands one ulp
    # high). Without the epsilon that dust adds a WHOLE extra cell: the
    # grid would overshoot the core by 1.5 cells and the extent mask would
    # fill it from the buffer, stealing neighbor cells in first-wins
    # mosaics.
    tolerance = 1e-7
    cols = max(1, math.ceil(col_ratio - tolerance))
    rows = max(1, math.ceil(row_ratio - tolerance))
    cell_count = rows * cols
    estimated_bytes = cell_count * RASTER_GRID_BYTES_PER_CELL
    tile_id = tile.get("tile_id", "unknown")
    _admit_grid(
        estimated_bytes, memory_budget_bytes,
        f"Raster grid for tile {tile_id} at {resolution:g} m "
        f"({rows:,} x {cols:,} cells)",
    )
    return Affine(
        resolution, 0.0, x_min, 0.0, -resolution, y_max
    ), cols, rows


class TileAccumulator:
    """Accumulate chunked points into a tile-core grid.

    ``min`` and ``max`` support DTM/DSM, ``mean`` and ``median`` produce
    average surfaces, and ``idw`` performs inverse-distance interpolation.
    Buffers are bounded per cell rather than retaining the complete cloud.
    """

    _MEAN_LIKE = {"min", "max", "mean", "median"}
    _IDW_NEEDS_VALUES = {"idw"}

    def __init__(
        self,
        tile: dict,
        cell_size: float = RASTER_CELL_SIZE,
        reducer: str = "min",
        memory_budget_bytes: int | None = None,
    ):
        self._transform, self._cols, self._rows = tile_transform(
            tile, cell_size, memory_budget_bytes
        )
        n = self._rows * self._cols
        # Keep the grid and input values at the same dtype so NumPy can use
        # its fast ufunc.at path.
        if reducer in self._MEAN_LIKE:
            init = np.inf if reducer == "min" else -np.inf
            # min/max select (no arithmetic): float32 is exact. mean sums
            # stay float64; the final cast happens in result() either way.
            dtype = np.float32 if reducer in ("min", "max") else np.float64
            self._grid = np.full(n, init, dtype=dtype)
            if reducer in ("mean", "median"):
                self._sum = np.zeros(n, dtype=np.float64)
                self._count = np.zeros(n, dtype=np.int64)
            else:
                self._sum = self._count = None
        elif reducer in self._IDW_NEEDS_VALUES:
            self._grid = np.zeros(n, dtype=np.float64)
            self._sum = np.zeros(n, dtype=np.float64)
            self._count = np.zeros(n, dtype=np.int64)
        else:
            raise ValueError(f"Unknown reducer: {reducer}")
        # median/idw staging: sorted (flat, value) runs per chunk instead of
        # per-cell lists (a million empty lists cost more than the points).
        self._pending: list[tuple[np.ndarray, np.ndarray]] = []
        self._reducer = reducer

    def extent_mask(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Mask of points inside this accumulator's own raster extent.

        Use it instead of ``core_mask`` whenever the partials merge as
        pixels: the frontier cell shared with a neighbor stays whole.
        """
        return extent_mask(x, y, self._transform, self._cols, self._rows)

    def add(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
        if x.size == 0:
            return
        c = np.floor((x - self._transform.c) / self._transform.a).astype(np.int64)
        r = np.floor((self._transform.f - y) / (-self._transform.e)).astype(np.int64)
        np.clip(c, 0, self._cols - 1, out=c)
        np.clip(r, 0, self._rows - 1, out=r)
        flat = r * self._cols + c
        # Matching dtypes keep ufunc.at on NumPy's fast path.
        if self._reducer == "min":
            np.minimum.at(self._grid, flat, z)
        elif self._reducer == "max":
            np.maximum.at(self._grid, flat, z)
        elif self._reducer == "mean":
            np.add.at(self._sum, flat, z)
            np.add.at(self._count, flat, 1)
        elif self._reducer == "median":
            np.add.at(self._sum, flat, z)
            np.add.at(self._count, flat, 1)
            order = np.argsort(flat, kind="stable")
            self._pending.append((flat[order], np.asanyarray(z)[order]))
        elif self._reducer == "idw":
            order = np.argsort(flat, kind="stable")
            self._pending.append((flat[order], np.asanyarray(z)[order]))

    def result(self) -> tuple[np.ndarray, Affine]:
        if self._reducer == "min" or self._reducer == "max":
            grid = self._grid
        elif self._reducer == "mean":
            valid = self._count > 0
            grid = np.where(valid, self._sum / np.maximum(self._count, 1), np.inf)
        elif self._reducer == "median":
            grid = np.full(self._rows * self._cols, np.inf, dtype=np.float64)
            if self._pending:
                flat_all = np.concatenate([f for f, _ in self._pending])
                vals_all = np.concatenate([v for _, v in self._pending])
                order = np.argsort(flat_all, kind="stable")
                f_sorted = flat_all[order]
                v_sorted = vals_all[order]
                cells, starts, counts = np.unique(
                    f_sorted, return_index=True, return_counts=True
                )
                ends = starts + counts
                for cell, s, e in zip(cells.tolist(), starts.tolist(), ends.tolist()):
                    grid[cell] = float(np.median(v_sorted[s:e]))
        elif self._reducer == "idw":
            # Per-cell IDW using the points assigned to that cell. Empty cells
            # remain NODATA. Fully vectorized through group ids + bincount.
            grid = np.full(self._rows * self._cols, np.inf, dtype=np.float64)
            if self._pending:
                flat_all = np.concatenate([f for f, _ in self._pending])
                vals_all = np.concatenate([v for _, v in self._pending])
                order = np.argsort(flat_all, kind="stable")
                f_sorted = flat_all[order]
                v_sorted = vals_all[order].astype(np.float64)
                is_start = np.ones(f_sorted.shape[0], dtype=bool)
                is_start[1:] = f_sorted[1:] != f_sorted[:-1]
                gid = np.cumsum(is_start, dtype=np.int64) - 1
                counts = np.bincount(gid)
                # Weight = 1/distance^2; equal-cell points use an
                # effective distance of one to avoid bias.
                means = np.bincount(gid, weights=v_sorted) / np.maximum(counts, 1)
                weights = np.ones(v_sorted.shape[0]) / np.maximum(
                    (v_sorted - means[gid]) ** 2, 1e-6
                )
                wsum = np.bincount(gid, weights=weights)
                wval = np.bincount(gid, weights=weights * v_sorted)
                cells = f_sorted[is_start]
                grid[cells] = wval / np.maximum(wsum, 1e-300)
        else:
            grid = self._grid
        grid = np.where(np.isfinite(grid), grid, NODATA)
        return grid.reshape(self._rows, self._cols).astype(np.float32), self._transform


class CountAccumulator:
    """Accumulates per-cell counts over the tile core (chunk-aware).

    - add(x, y)            -> counts points per cell (result mode "count")
    - add_sum(x, y, v)     -> counts points and accumulates values (mode "mean")
    - add_unique(x, y, k)  -> counts DISTINCT keys per cell (mode "unique")

    ``add_unique`` sorts (cell, key) per chunk and keeps only first occurrences,
    avoiding per-cell sets. ``result`` returns float32 with NODATA for empty
    cells without requiring pandas.
    """

    def __init__(
        self,
        tile: dict,
        cell_size: float = RASTER_CELL_SIZE,
        memory_budget_bytes: int | None = None,
    ):
        self._transform, self._cols, self._rows = tile_transform(
            tile, cell_size, memory_budget_bytes
        )
        n = self._rows * self._cols
        # int32 counts (2B points per cell is unreachable) and float32 sums
        # (exact for integer values far beyond per-cell counts).
        self._count = np.zeros(n, dtype=np.int32)
        self._sum = np.zeros(n, dtype=np.float32)

    def _flat(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        c = np.floor((x - self._transform.c) / self._transform.a).astype(np.int64)
        r = np.floor((self._transform.f - y) / (-self._transform.e)).astype(np.int64)
        np.clip(c, 0, self._cols - 1, out=c)
        np.clip(r, 0, self._rows - 1, out=r)
        return r * self._cols + c

    def flat(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Cell indices for points (shared across accumulators on one chunk)."""
        return self._flat(x, y)

    def extent_mask(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Mask of points inside this accumulator's own raster extent.

        Use it instead of ``core_mask`` whenever the partials merge as
        pixels: the frontier cell shared with a neighbor stays whole.
        """
        return extent_mask(x, y, self._transform, self._cols, self._rows)

    def add(
        self, x: np.ndarray, y: np.ndarray, flat: np.ndarray | None = None
    ) -> None:
        if x.size == 0:
            return
        np.add.at(self._count, self._flat(x, y) if flat is None else flat, 1)

    def add_sum(
        self,
        x: np.ndarray,
        y: np.ndarray,
        values: np.ndarray,
        flat: np.ndarray | None = None,
    ) -> None:
        if x.size == 0:
            return
        flat = self._flat(x, y) if flat is None else flat
        np.add.at(self._count, flat, 1)
        np.add.at(self._sum, flat, values.astype(np.float32))

    def add_unique(
        self,
        x: np.ndarray,
        y: np.ndarray,
        keys: np.ndarray,
        flat: np.ndarray | None = None,
    ) -> None:
        """Counts unique ``(cell, key)`` pairs in one call.

        Writers call this once per chunk (2M points): a tile larger than a
        chunk can count the same pair twice when its copies land in
        different chunks. Known approximation; tiles under one chunk are
        exact (the usual case: a 100 m tile at 0.3 m is ~1M points).
        """
        if x.size == 0:
            return
        flat = self._flat(x, y) if flat is None else flat
        order = np.lexsort((keys, flat))
        flat_s = flat[order]
        keys_s = keys[order]
        first = np.concatenate(
            (
                np.ones(1, dtype=bool),
                (flat_s[1:] != flat_s[:-1])
                | (keys_s[1:] != keys_s[:-1]),
            )
        )
        np.add.at(self._count, flat_s[first], 1)

    def result(self, mode: str = "count", nodata: float = NODATA):
        count = self._count
        if mode == "mean":
            grid = self._sum / np.maximum(count, 1)
        elif mode in ("count", "unique"):
            grid = count.astype(np.float64)
        else:
            raise ValueError(f"Unknown CountAccumulator mode: {mode}")
        grid = np.where(count > 0, grid, nodata)
        return grid.reshape(self._rows, self._cols).astype(np.float32), self._transform


# ---------------------------------------------------------------------------
# GeoTIFF (rasterio)
# ---------------------------------------------------------------------------

def write_geotiff(
    path: str | Path,
    array: np.ndarray,
    transform: Affine,
    crs: str | None = None,
    nodata: float = NODATA,
    metadata: dict | None = None,
) -> None:
    """Writes a single-band float32 GeoTIFF.

    Deflate + floating point predictor: much smaller files and faster I/O
    (write, merge, re-read on re-runs) at the cost of cheap CPU. When
    ``metadata`` (a provenance document) is given it is embedded as a
    dataset tag, best-effort.
    """
    import json

    import rasterio
    from rasterio.crs import CRS

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    crs_obj = None
    if crs:
        try:
            crs_obj = CRS.from_user_input(crs)
        except Exception:
            crs_obj = None
    with rasterio.open(
        str(path),
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype="float32",
        crs=crs_obj,
        transform=transform,
        nodata=nodata,
        compress="deflate",
        predictor="3",
        **GTIFF_CREATION,
    ) as dst:
        dst.write(array.astype(np.float32), 1)
        if metadata:
            try:
                dst.update_tags(
                    **{"lynceus_provenance": json.dumps(metadata, ensure_ascii=True)}
                )
            except Exception:
                pass


def _guard_raster_alloc(width: int, height: int, bands: int, path) -> None:
    """Refuse single-array raster reads that cannot fit RAM (legibly).

    Tiling covers point clouds; raster barriers read whole bands. A
    Sentinel-2 scale array would swap the machine instead of failing the
    node: fail fast with the shortfall named. Windowed reads arrive
    post-MVP.
    """
    need = max(0, int(width)) * max(0, int(height)) * max(1, int(bands)) * 4
    try:
        import psutil

        available = int(psutil.virtual_memory().available)
    except Exception:
        return
    if available > 0 and need > available // 2:
        raise RuntimeError(
            f"{Path(path).name} needs ~{need // (1024 ** 2)} MiB for one "
            f"read with ~{available // (1024 ** 2)} MiB RAM available: "
            "too large for a single read (windowed processing arrives "
            "post-MVP)"
        )


def load_dem_masked(
    path: str | Path,
    fallback_nodata: float = NODATA,
) -> tuple:
    """Read a DEM honoring the file's own NODATA tag (plus NaN, always).

    External rasters tag voids their own way (``-3.4e38``, untagged zeros,
    NaN...): comparing against a fixed default treats voids as valid ground
    and Horn happily computes ~90-degree slopes over them, so products show
    data outside the scanned zone. The returned array is float64 with every
    invalid cell remapped to ``fallback_nodata``; the output nodata is always
    the fallback, keeping downstream math and file tags on the house
    convention. Returns ``(array, transform, crs_str, nodata)``.
    """
    import rasterio

    with rasterio.open(str(path)) as src:
        tag = src.nodata
        _guard_raster_alloc(src.width, src.height, 1, path)
        raw = src.read(1).astype(np.float64)
        transform, crs_str = src.transform, (
            str(src.crs) if src.crs else None
        )
    valid = np.isfinite(raw)
    if tag is not None:
        try:
            valid &= raw != float(tag)
        except (TypeError, ValueError):
            pass
    work = np.where(valid, raw, float(fallback_nodata))
    return work, transform, crs_str, float(fallback_nodata)


def read_geotiff(
    path: str | Path,
) -> tuple[np.ndarray, Affine, str | None]:
    """Read band 1 and return (array, transform, crs_str)."""
    import rasterio

    with rasterio.open(str(path)) as src:
        _guard_raster_alloc(src.width, src.height, 1, path)
        array = src.read(1).astype(np.float32)
        return array, src.transform, (str(src.crs) if src.crs else None)


def read_geotiff_bands(
    path: str | Path, indices: list[int] | tuple[int, ...],
) -> tuple[np.ndarray, Affine, str | None]:
    """Read selected 1-based bands and return ((bands, h, w), transform, crs).

    Raises RuntimeError on missing files, out-of-range indices or unreadable
    data (fail loud: a vegetation index on the wrong bands is worse than
    no index).
    """
    import rasterio

    indices = [int(i) for i in indices]
    if not indices or any(i < 1 for i in indices):
        raise RuntimeError(
            f"Band indices must be 1-based positives: {list(indices)}"
        )
    try:
        with rasterio.open(str(path)) as src:
            count = src.count
            bad = [i for i in indices if i > count]
            if bad:
                raise RuntimeError(
                    f"{Path(path).name} has {count} band(s); "
                    f"no such band(s): {bad}"
                )
            _guard_raster_alloc(src.width, src.height, len(indices), path)
            array = src.read(indices).astype(np.float32)
            return array, src.transform, (str(src.crs) if src.crs else None)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Cannot read bands {list(indices)} from {Path(path).name} ({exc})"
        ) from exc


def _crop_to_valid(
    array: np.ndarray, transform: Affine, nodata: float
) -> tuple[np.ndarray, Affine]:
    """Crops to the bbox of valid cells: there is nothing outside the scan.

    Returns (cropped_array, offset_transform); with no valid data returns
    the input as-is.
    """
    valid = array != nodata
    if not valid.any():
        return array, transform
    rows = np.flatnonzero(valid.any(axis=1))
    cols = np.flatnonzero(valid.any(axis=0))
    r0, c0 = int(rows[0]), int(cols[0])
    r1, c1 = int(rows[-1]) + 1, int(cols[-1]) + 1
    return array[r0:r1, c0:c1], transform * Affine.translation(c0, r0)


def _valid_bounds(
    array: np.ndarray, transform: Affine, nodata: float
) -> tuple[float, float, float, float] | None:
    """World bounds (left, bottom, right, top) of the valid area, or None."""
    valid = array != nodata
    if not valid.any():
        return None
    rows = np.flatnonzero(valid.any(axis=1))
    cols = np.flatnonzero(valid.any(axis=0))
    t_win = transform * Affine.translation(int(cols[0]), int(rows[0]))
    n_cols = int(cols[-1]) - int(cols[0]) + 1
    n_rows = int(rows[-1]) - int(rows[0]) + 1
    return (
        t_win.c,
        t_win.f + n_rows * t_win.e,
        t_win.c + n_cols * t_win.a,
        t_win.f,
    )


def align_to_grid(
    array: np.ndarray,
    transform: Affine,
    shape: tuple[int, int],
    target_transform: Affine,
    fill: float,
) -> np.ndarray:
    """Remaps an array onto the target grid (same cell and alignment).

    The mosaics derive from the same tiler grid: only the crop changes,
    so an integer cell offset suffices; uncovered areas keep the fill.
    """
    out = np.full(shape, fill, dtype=array.dtype)
    inv = ~target_transform
    c_off, r_off = inv * (transform.c, transform.f)
    c0, r0 = int(round(float(c_off))), int(round(float(r_off)))
    h, w = array.shape
    dc0, dr0 = max(0, c0), max(0, r0)
    dc1, dr1 = min(shape[1], c0 + w), min(shape[0], r0 + h)
    if dc1 <= dc0 or dr1 <= dr0:
        return out
    out[dr0:dr1, dc0:dc1] = array[dr0 - r0 : dr1 - r0, dc0 - c0 : dc1 - c0]
    return out


def merge_geotiffs(
    paths: list[str | Path],
    crs: str | None,
    out_path: str | Path,
    nodata: float = NODATA,
    metadata: dict | None = None,
    return_array: bool = False,
    method: str = "first",
):
    """Merges aligned GeoTIFFs (exact mosaic, no resampling) into a single one.

    The mosaic is cropped to the data bbox: tile-grid margins without points
    are left outside the file. ``metadata`` is forwarded to ``write_geotiff``.
    With ``return_array=True`` the merged (cropped) array is returned instead
    of written, so barriers can post-process before the single write:
    ``(array, transform, crs)``. ``method`` selects the overlap winner
    (``first``/``last``/``min``/``max``, passed to ``rasterio.merge``).
    """
    datasets = [str(p) for p in paths]
    if not datasets:
        raise ValueError("No tile rasters to merge")
    crs = crs or "EPSG:4326"
    dest, transform = rio_merge(datasets, nodata=nodata, dtype="float32",
                                method=method)
    cropped, crop_transform = _crop_to_valid(dest[0], transform, nodata)
    if return_array:
        return cropped, crop_transform, crs
    write_geotiff(
        out_path, cropped, crop_transform, crs=crs, nodata=nodata, metadata=metadata
    )
    return str(out_path)


_GDAL_DTYPE = {
    "uint8": "Byte",
    "uint16": "UInt16",
    "int16": "Int16",
    "uint32": "UInt32",
    "int32": "Int32",
    "float32": "Float32",
    "float64": "Float64",
}
"""rasterio dtype name to GDAL VRT ``dataType`` (the VRT mosaic only spans
the plain numeric types the pipeline writes; anything else is skipped)."""


def _mosaic_plan(sources: list[str | Path]) -> dict:
    """Shared mosaic introspection: homogeneity, union grid, placements.

    Returns a plan dict (sorted sources, union extent, per-source integer
    pixel offsets, WKT, dtypes) or ``{"skipped": reason}`` on any mismatch —
    the multi-CRS edge included. One planner feeds the VRT writer, the
    windowed merger and the fit checks, so all three agree on what "same
    grid" means. Pixel data is never read here.
    """
    import rasterio

    ordered = sorted({str(Path(s)) for s in sources})
    if not ordered:
        return {"skipped": "no sources"}
    try:
        infos = []
        for src_path in ordered:
            with rasterio.open(src_path) as src:
                infos.append({
                    "path": src_path,
                    "width": int(src.width),
                    "height": int(src.height),
                    "transform": src.transform,
                    "crs": src.crs,
                    "dtypes": tuple(src.dtypes),
                    "count": int(src.count),
                    "nodata": src.nodata,
                    "blocks": [tuple(b) for b in src.block_shapes],
                })
    except Exception as exc:
        return {"skipped": f"unreadable source ({exc})"}
    first = infos[0]
    if first["crs"] is None:
        return {"skipped": "sources carry no CRS"}
    try:
        if not first["crs"].is_projected:
            return {"skipped": "geographic CRS has no meter grid"}
    except Exception:
        return {"skipped": "unusable CRS"}
    for info in infos[1:]:
        if info["crs"] != first["crs"]:
            return {"skipped": "CRS differs between sources"}
    res_x = abs(float(first["transform"].a))
    res_y = abs(float(first["transform"].e))
    if not (res_x > 0 and res_y > 0):
        return {"skipped": "degenerate resolution"}
    for info in infos[1:]:
        t = info["transform"]
        if not (
            math.isclose(abs(float(t.a)), res_x, rel_tol=1e-9)
            and math.isclose(abs(float(t.e)), res_y, rel_tol=1e-9)
        ):
            return {"skipped": "resolution differs between sources"}
        if info["dtypes"] != first["dtypes"]:
            return {"skipped": "dtype differs between sources"}
        if info["count"] != first["count"]:
            return {"skipped": "band count differs between sources"}
        if (info["nodata"] is None) != (first["nodata"] is None) or (
            info["nodata"] is not None
            and info["nodata"] != first["nodata"]
        ):
            return {"skipped": "nodata differs between sources"}
    try:
        gdal_dtype = [_GDAL_DTYPE[d] for d in first["dtypes"]]
    except KeyError:
        return {"skipped": f"unsupported dtype {first['dtypes']}"}
    west = min(i["transform"].c for i in infos)
    north = max(i["transform"].f for i in infos)
    east = max(i["transform"].c + i["transform"].a * i["width"]
               for i in infos)
    south = min(i["transform"].f + i["transform"].e * i["height"]
                for i in infos)
    width = int(round((east - west) / res_x))
    height = int(round((north - south) / res_y))
    if width <= 0 or height <= 0:
        return {"skipped": "empty union extent"}
    placements = []
    for info in infos:
        t = info["transform"]
        x_off = (t.c - west) / res_x
        y_off = (north - t.f) / res_y
        if abs(x_off - round(x_off)) > 1e-6 or abs(y_off - round(y_off)) > 1e-6:
            return {"skipped": "sources are not grid-aligned"}
        placements.append((int(round(x_off)), int(round(y_off))))
    try:
        wkt = first["crs"].to_wkt()
    except Exception as exc:
        return {"skipped": f"CRS has no WKT ({exc})"}
    return {
        "sources": ordered,
        "infos": infos,
        "width": width,
        "height": height,
        "west": west,
        "north": north,
        "res_x": res_x,
        "res_y": res_y,
        "crs": first["crs"],
        "wkt": wkt,
        "nodata": first["nodata"],
        "dtypes": first["dtypes"],
        "gdal_dtype": gdal_dtype,
        "count": first["count"],
        "placements": placements,
    }


def build_vrt_mosaic(
    sources: list[str | Path], out_path: str | Path
) -> dict:
    """Assemble a mosaic VRT over same-grid rasters without copying bytes.

    The VRT references every source with a relative path, so the file stays
    valid wherever the session folder moves and rebuilding it is
    byte-identical (sorted sources, fixed formatting). Reading through the
    VRT matches :func:`merge_geotiffs` with ``method="first"``: sources are
    listed bottom-to-top in reverse-sorted order so the first-sorted source
    wins overlaps, and void pixels fall through to the source beneath
    (``VRTComplexSource`` + NODATA) exactly like the physical merge. The
    merge crops to valid bounds while the VRT keeps the union, so the two
    agree on the common region, not on the canvas size.

    Homogeneity is enforced (projected CRS, resolution, dtype, band count,
    nodata tag, integer grid alignment); any mismatch — the multi-CRS edge
    included — returns ``{"skipped": reason}`` instead of writing a wrong
    mosaic. Success returns ``{"file": str(out_path)}``.
    """
    import os
    import xml.etree.ElementTree as ET

    plan = _mosaic_plan(sources)
    if "skipped" in plan:
        return plan
    ordered = plan["sources"]
    infos = plan["infos"]
    width, height = plan["width"], plan["height"]
    west, north = plan["west"], plan["north"]
    res_x, res_y = plan["res_x"], plan["res_y"]
    wkt = plan["wkt"]
    gdal_dtype = plan["gdal_dtype"]
    placements = plan["placements"]
    out_path = Path(out_path)
    try:
        rels = [
            Path(os.path.relpath(p, out_path.parent)).as_posix()
            for p in ordered
        ]
    except ValueError as exc:
        return {"skipped": f"sources outside the session ({exc})"}

    root = ET.Element(
        "VRTDataset", {"rasterXSize": str(width), "rasterYSize": str(height)})
    srs = ET.SubElement(root, "SRS")
    srs.text = wkt
    gt = ET.SubElement(root, "GeoTransform")
    gt.text = (
        f"{west:.16g}, {res_x:.16g}, 0, {north:.16g}, 0, {-res_y:.16g}")
    nodata = plan["nodata"]
    for band in range(1, plan["count"] + 1):
        band_el = ET.SubElement(root, "VRTRasterBand", {
            "dataType": gdal_dtype[band - 1], "band": str(band)})
        if nodata is not None:
            nd = ET.SubElement(band_el, "NoDataValue")
            nd.text = repr(float(nodata))
        # Reverse-sorted: the VRT renders later sources on top, so the
        # first-sorted source wins overlaps (merge_geotiffs method="first").
        # ComplexSource + NODATA makes the winner's void pixels transparent
        # so the source beneath shows through; SimpleSource would stamp its
        # NODATA over valid data (stitch gaps along segment overlaps).
        for info, rel, (x_off, y_off) in reversed(
            list(zip(infos, rels, placements))
        ):
            src_el = ET.SubElement(
                band_el,
                "ComplexSource" if nodata is not None else "SimpleSource",
            )
            fn = ET.SubElement(
                src_el, "SourceFilename", {"relativeToVRT": "1"})
            fn.text = rel
            sb = ET.SubElement(src_el, "SourceBand")
            sb.text = str(band)
            bw, bh = info["blocks"][band - 1]
            ET.SubElement(src_el, "SourceProperties", {
                "RasterXSize": str(info["width"]),
                "RasterYSize": str(info["height"]),
                "DataType": gdal_dtype[band - 1],
                "BlockXSize": str(min(bw, info["width"])),
                "BlockYSize": str(min(bh, info["height"])),
            })
            ET.SubElement(src_el, "SrcRect", {
                "xOff": "0", "yOff": "0",
                "xSize": str(info["width"]), "ySize": str(info["height"])})
            ET.SubElement(src_el, "DstRect", {
                "xOff": str(x_off), "yOff": str(y_off),
                "xSize": str(info["width"]), "ySize": str(info["height"])})
            if nodata is not None:
                nd_src = ET.SubElement(src_el, "NODATA")
                nd_src.text = repr(float(nodata))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    ET.indent(tree)
    tree.write(str(out_path), encoding="utf-8", xml_declaration=True)
    return {"file": str(out_path)}


_MERGE_WORKING_COPIES = 3
"""Full-mosaic temporaries of the direct merge (rio result + crop + write)."""

#: Full-mosaic temporaries of the finishing ops (per float32 cell):
#: hole fill ~9 (distance float64 + indices int64x2 + filled + meters),
#: Gaussian smooth ~8 (mask + data + two smoothed + result),
#: median ~2 (banded full copies).
_FINISH_OP_COPIES = {"fill": 9.0, "smooth": 8.0, "median": 2.0}


def _finish_copies(fill_dist_m: float = 0.0, smooth_sigma: float = 0.0,
                   median_size: int = 0) -> float:
    """Working-set model of finish_mosaic's direct path (float32 cells).

    Shared by the full-fit check and the honest refusal so both agree on
    what "fits" means; the windowed writer needs none of this (bounded
    windows by construction).
    """
    copies = float(_MERGE_WORKING_COPIES)
    try:
        if float(fill_dist_m) > 0:
            copies += _FINISH_OP_COPIES["fill"]
        if float(smooth_sigma) > 0:
            copies += _FINISH_OP_COPIES["smooth"]
        if int(median_size) > 1:
            copies += _FINISH_OP_COPIES["median"]
    except (TypeError, ValueError):
        pass
    return copies

_WINDOW_WORKING_COPIES = 6
"""Per-window temporaries (composite + op internals + output buffer)."""

_WINDOW_MIN_SIDE = 256
_WINDOW_MAX_SIDE = 4096
"""Window side clamp: the floor guarantees progress on tiny budgets (256 KB
per band); the ceiling keeps file-open churn and per-window overhead sane."""

_VOID_BLOCK = 1024
"""Coarse validity block (px) for window skipping: windows whose expanded
region touches no valid block skip reads and compute exactly. One block
row matches the streaming strip, so occupancy rides the validity scan
for free; 1024 px catches flight gaps, water and tile margins without
drowning uniform-sparse data (which legitimately keeps every window)."""


def mosaic_fits(sources: list[str | Path], copies: float,
                budget_bytes: int | None) -> bool:
    """Whether a full-mosaic array of ``copies`` temporaries fits a budget.

    Introspection only (no pixel reads): union cells × 4 B × bands × copies
    against ``budget_bytes``. Unset budget, heterogeneous sources or any
    doubt fit (legacy direct behavior); the windowed writers decide by data.
    """
    try:
        budget = int(budget_bytes or 0)
    except (TypeError, ValueError):
        return True
    if budget <= 0:
        return True
    plan = _mosaic_plan(sources)
    if "skipped" in plan:
        return True
    need = plan["width"] * plan["height"] * 4 * max(1, plan["count"])
    return need * float(copies) <= budget


def mosaic_need(sources: list[str | Path], copies: float,
                cell_bytes: int = 4) -> int | None:
    """Direct-mode working set in bytes (None on any doubt).

    Union cells × cell bytes × bands × copies from the shared planner
    (headers only, no pixels): the peak the writer WILL hold when the
    full fit goes direct. None on heterogeneous sources, unreadable
    files or junk parameters — the dispatcher keeps its legacy
    input-bytes estimate instead. This is the other half of the
    estimate/mode contract: the gate budgets the peak of the mode that
    actually runs, direct or windowed.
    """
    try:
        copies_f = float(copies)
        cell = int(cell_bytes)
    except (TypeError, ValueError):
        return None
    if copies_f <= 0.0 or cell <= 0:
        return None
    plan = _mosaic_plan(sources)
    if "skipped" in plan:
        return None
    return int(plan["width"] * plan["height"] * cell
               * max(1, plan["count"]) * copies_f)


def window_peak(budget_bytes: int | None, cell_bytes: int = 4) -> int:
    """Windowed-mode peak in bytes for a per-worker budget (0 if unknown).

    Inverts the writer's window sizing: the side holds ``copies`` working
    temporaries inside the budget, so the peak is side² × cell × copies —
    just under the budget by construction. 0 when the budget is unset
    (the writer then goes direct and this model does not apply).
    """
    try:
        budget = int(budget_bytes or 0)
        cell = int(cell_bytes or 4)
    except (TypeError, ValueError):
        return 0
    if budget <= 0 or cell <= 0:
        return 0
    side = _window_side(budget, cell, _WINDOW_WORKING_COPIES)
    return side * side * cell * _WINDOW_WORKING_COPIES


#: Assumed raw-bytes-per-compressed-byte for GeoTIFF inputs (deflate on
#: float32 noise lands ~2-4x; 3.0 leans to refusal, the safe direction).
DECOMPRESS_GUESS = 3.0


def refuse_if_unfit(sources: list[str | Path], copies: float,
                    budget_bytes: int | None, describe: str) -> None:
    """Raise an honest ValueError when a full-array computation cannot fit.

    For heterogeneous inputs there is no union plan, so the need is sized
    from compressed file sizes (``DECOMPRESS_GUESS`` × ``copies`` working
    temporaries). Unset budget or fitting needs pass silently (legacy
    behavior); the alternative to this refusal is an opaque worker
    ``MemoryError`` mid-computation, which F0 then reports without numbers.
    """
    try:
        budget = int(budget_bytes or 0)
    except (TypeError, ValueError):
        return
    if budget <= 0:
        return
    total = 0
    try:
        for path in sources:
            total += Path(path).stat().st_size
    except OSError:
        return
    need = int(total * DECOMPRESS_GUESS * float(copies))
    if need <= budget:
        return
    raise ValueError(
        f"{describe} would need ~{need // (1024 ** 2):,} MiB "
        f"(inputs total ~{total // (1024 ** 2):,} MiB on disk, working "
        "copies included), "
        f"above the ~{budget // (1024 ** 2):,} MiB budget. Wire same-grid "
        "inputs, raise the memory budget, or run fewer products at once."
    )


def _valid_box_streaming(path: str, nodata, strip: int = 1024):
    """Pixel box (c0, r0, c1, r1, exclusive) of valid band-1 cells, or None.

    Row strips only: peak is one strip, never the mosaic. Early exit is the
    caller's job (this returns the full box).
    """
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(path) as src:
        width, height = int(src.width), int(src.height)
        r0, c0 = height, width
        r1, c1 = -1, -1
        for start in range(0, height, strip):
            rows = min(strip, height - start)
            arr = src.read(1, window=Window(0, start, width, rows))
            mask = arr != nodata
            if not mask.any():
                continue
            hit_rows = np.flatnonzero(mask.any(axis=1)) + start
            hit_cols = np.flatnonzero(mask.any(axis=0))
            r0 = min(r0, int(hit_rows[0]))
            r1 = max(r1, int(hit_rows[-1]))
            c0 = min(c0, int(hit_cols[0]))
            c1 = max(c1, int(hit_cols[-1]))
        if r1 < 0:
            return None
        return (c0, r0, c1 + 1, r1 + 1)


def _mosaic_source_boxes(plan: dict, nodata) -> list:
    """Per-source valid boxes in union pixels (void sources omitted).

    One streaming scan per source; callers reuse the list both for the
    union box and for skipping provably-void windows (a window outside
    every box holds no valid cell from any source, so compositing it is
    pure overhead).
    """
    boxes = []
    for info, (ox, oy) in zip(plan["infos"], plan["placements"]):
        box = _valid_box_streaming(info["path"], nodata)
        if box is None:
            continue
        boxes.append((box[0] + ox, box[1] + oy, box[2] + ox, box[3] + oy))
    return boxes


def _box_hits(box, c0: int, r0: int, c1: int, r1: int) -> bool:
    """Whether a half-open pixel box intersects ``(c0, r0, c1, r1)``."""
    return box[0] < c1 and box[2] > c0 and box[1] < r1 and box[3] > r0


def _mosaic_valid_box(plan: dict, nodata) -> tuple | None:
    """Union of the sources' valid boxes in union pixels, or None.

    Merged validity is the union of source validity (first-wins keeps any
    valid cell), so this box is exactly what the direct merge crops to.
    """
    boxes = _mosaic_source_boxes(plan, nodata)
    if not boxes:
        return None
    return (
        min(b[0] for b in boxes), min(b[1] for b in boxes),
        max(b[2] for b in boxes), max(b[3] for b in boxes),
    )


def _box_hits(box, c0: int, r0: int, c1: int, r1: int) -> bool:
    """Whether a half-open pixel box intersects ``(c0, r0, c1, r1)``."""
    return box[0] < c1 and box[2] > c0 and box[1] < r1 and box[3] > r0


def _window_side(budget_bytes: int, cell_bytes: int, copies: int) -> int:
    """Square window side holding ``copies`` temporaries inside a budget."""
    try:
        budget = int(budget_bytes)
    except (TypeError, ValueError):
        budget = 0
    cells = max(1, budget // max(1, cell_bytes * copies))
    side = int(math.isqrt(cells))
    return max(_WINDOW_MIN_SIDE, min(_WINDOW_MAX_SIDE, side))


def _note_bounded(notes, text: str) -> None:
    """Record a bounded-mode notice for the barrier payload (or ignore).

    Writers never touch the UI: they append to the barrier's note list
    and the engine wrapper forwards it. ``None`` keeps writers usable
    standalone (tests, tools).
    """
    if notes is not None:
        try:
            notes.append(text)
        except AttributeError:
            pass


def _open_tiled_output(path: Path, width: int, height: int, transform,
                       crs, nodata, count: int, dtype: str,
                       metadata: dict | None = None):
    """Create a tiled, compressed GeoTIFF for windowed writes."""
    import rasterio
    from rasterio.crs import CRS

    crs_obj = None
    if crs is not None and not hasattr(crs, "to_wkt"):
        try:
            crs_obj = CRS.from_user_input(crs)
        except Exception:
            crs_obj = None
    else:
        crs_obj = crs
    predictor = 3 if np.dtype(dtype).kind == "f" else 2
    return rasterio.open(
        str(path), "w", driver="GTiff", height=height, width=width,
        count=count, dtype=dtype, crs=crs_obj, transform=transform,
        nodata=nodata, tiled=True, blockxsize=256, blockysize=256,
        compress="deflate", predictor=predictor,
        **GTIFF_CREATION,
    )


def _stamp_provenance_tag(dst, metadata: dict | None) -> None:
    import json

    if not metadata:
        return
    try:
        dst.update_tags(
            **{"lynceus_provenance": json.dumps(metadata, ensure_ascii=True)})
    except Exception:
        pass


def _finish_ops(arr: np.ndarray, resolution: float, fill_dist_m: float,
                smooth_sigma: float, median_size: int, nodata: float):
    """Post-merge finishing shared by the direct and windowed writers."""
    if fill_dist_m > 0:
        arr = fill_holes(arr, resolution, fill_dist_m, nodata)
    if smooth_sigma > 0:
        arr = smooth_nodata(arr, smooth_sigma, nodata)
    if median_size > 1:
        arr = median_nodata(arr, size=median_size, nodata=nodata)
    return arr


def _finish_halo(fill_dist_m: float, smooth_sigma: float, median_size: int,
                 resolution: float) -> int:
    """Window halo covering the composed finishing ops (additive rule).

    The ops run in sequence (fill, then smooth, then median), so supports
    add up: the fill reaches max_dist past valid data and the smooth needs
    its 4σ neighborhood beyond that. Core cells see identical neighborhoods
    either way. Gaussian truncation is 4σ, the median pad is size//2.
    """
    halo = 0
    if fill_dist_m > 0 and resolution > 0:
        halo += int(math.ceil(fill_dist_m / resolution))
    if smooth_sigma > 0:
        halo += int(4.0 * smooth_sigma) + 1
    if median_size > 1:
        halo += int(median_size) // 2
    return halo


def finish_mosaic(
    sources: list[str | Path],
    out_path: str | Path,
    *,
    crs: str | None = None,
    nodata: float = NODATA,
    metadata: dict | None = None,
    fill_dist_m: float = 0.0,
    smooth_sigma: float = 0.0,
    median_size: int = 0,
    resolution: float = 1.0,
    crop_to_valid: bool = True,
    method: str = "first",
    memory_budget_bytes: int | None = None,
    notes: list | None = None,
) -> str:
    """Merge tile fragments into one finished mosaic, direct or windowed.

    The finishing ops (hole fill, Gaussian smooth, median, valid crop) run
    in both modes with byte-identical results: the windowed writer sizes a
    halo from each op's support, so core cells see the same neighborhoods.
    Mode selects by data — full fit inside ``memory_budget_bytes`` goes
    direct (the legacy ``merge_geotiffs`` path, heterogeneous inputs and
    non-first methods included); otherwise windows of bounded RAM stream
    the same mosaic. Unset budget keeps the legacy behavior bit for bit.
    The windowed mode appends one notice to ``notes`` (same resolution,
    slower) for the engine to forward.
    """
    import rasterio
    from rasterio.windows import Window

    ordered = sorted({str(Path(s)) for s in sources})
    if not ordered:
        raise ValueError("No tile rasters to merge")
    crs = crs or "EPSG:4326"
    out_path = Path(out_path)
    try:
        budget = int(memory_budget_bytes or 0)
    except (TypeError, ValueError):
        budget = 0
    plan = None
    union_fits = False
    skip_reason = ""
    if budget > 0 and method == "first":
        plan = _mosaic_plan(ordered)
        if "skipped" in plan:
            skip_reason = str(plan["skipped"])
            plan = None
        elif (
            plan["width"] * plan["height"] * 4 * max(1, plan["count"])
            * _finish_copies(fill_dist_m, smooth_sigma, median_size)
            <= budget
        ):
            # Full fit: the legacy direct path (windowed mode would only
            # add file-open churn and tiled-output overhead here).
            plan = None
            union_fits = True
    if plan is None and budget > 0 and not union_fits:
        # No union plan (heterogeneous inputs or non-first method): size
        # the need from file sizes and refuse honestly instead of OOMing
        # opaquely inside the merge.
        refuse_if_unfit(
            ordered,
            _finish_copies(fill_dist_m, smooth_sigma, median_size),
            budget,
            f"Mosaic {out_path.name}"
            + (f" ({skip_reason})" if skip_reason else ""),
        )
    if plan is None:
        dest, transform = rio_merge(ordered, nodata=nodata, dtype="float32",
                                    method=method)
        arr = dest[0]
        # Legacy order (fill/smooth/median BEFORE crop): the ops run on
        # the full union so the fillable fringe survives, and the crop
        # then keeps it (it is valid data now). Crop-first would silently
        # drop the fringe and diverge from the windowed writer.
        arr = _finish_ops(arr, resolution, fill_dist_m, smooth_sigma,
                          median_size, nodata)
        if crop_to_valid:
            arr, transform = _crop_to_valid(arr, transform, nodata)
        write_geotiff(out_path, arr, transform, crs=crs, nodata=nodata,
                      metadata=metadata)
        return str(out_path)

    count = plan["count"]
    west, north = plan["west"], plan["north"]
    res_x, res_y = plan["res_x"], plan["res_y"]
    src_boxes = _mosaic_source_boxes(plan, nodata)
    if src_boxes:
        box = (
            min(b[0] for b in src_boxes), min(b[1] for b in src_boxes),
            max(b[2] for b in src_boxes), max(b[3] for b in src_boxes),
        )
    else:
        box = (0, 0, plan["width"], plan["height"])
    if fill_dist_m > 0 and resolution > 0:
        fill_cells = int(math.ceil(fill_dist_m / resolution))
        box = (
            max(0, box[0] - fill_cells), max(0, box[1] - fill_cells),
            min(plan["width"], box[2] + fill_cells),
            min(plan["height"], box[3] + fill_cells),
        )
    if not crop_to_valid:
        box = (0, 0, plan["width"], plan["height"])
    out_w, out_h = box[2] - box[0], box[3] - box[1]
    out_transform = (
        Affine(res_x, 0.0, west + box[0] * res_x,
               0.0, -res_y, north - box[1] * res_y)
    )
    halo = _finish_halo(fill_dist_m, smooth_sigma, median_size, resolution)
    side = _window_side(budget, 4 * max(1, count), _WINDOW_WORKING_COPIES)
    logger.info(
        "finish_mosaic: %d sources -> %s in windows (%d px, halo %d)",
        len(ordered), out_path.name, side, halo,
    )
    _note_bounded(
        notes,
        f"Large mosaic {out_path.name}: merged in windows "
        f"({side} px) at full resolution; slower, identical pixels.",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    skipped = 0
    total_windows = 0
    with _open_tiled_output(
        out_path, out_w, out_h, out_transform, plan["crs"], nodata, count,
        "float32", metadata,
    ) as dst:
        for r0 in range(box[1], box[3], side):
            for c0 in range(box[0], box[2], side):
                c1, r1 = min(box[2], c0 + side), min(box[3], r0 + side)
                ec0, er0 = max(0, c0 - halo), max(0, r0 - halo)
                ec1 = min(plan["width"], c1 + halo)
                er1 = min(plan["height"], r1 + halo)
                total_windows += 1
                window = Window(c0 - box[0], r0 - box[1], c1 - c0, r1 - r0)
                if not any(_box_hits(b, ec0, er0, ec1, er1)
                           for b in src_boxes):
                    # Provably void: no source holds valid data in the
                    # expanded window (halo included, so fill cannot pull
                    # anything in), and every finishing op preserves
                    # voids — the core writes NODATA exactly as the full
                    # run, skipping all reads and compute.
                    if count == 1:
                        dst.write(
                            np.full((r1 - r0, c1 - c0), nodata,
                                    dtype=np.float32),
                            1, window=window)
                    else:
                        dst.write(
                            np.full((count, r1 - r0, c1 - c0), nodata,
                                    dtype=np.float32),
                            list(range(1, count + 1)), window=window)
                    skipped += 1
                    continue
                ew, eh = ec1 - ec0, er1 - er0
                comp = np.full((count, eh, ew), nodata, dtype=np.float32)
                filled = np.zeros((count, eh, ew), dtype=bool)
                for info, (ox, oy) in zip(plan["infos"], plan["placements"]):
                    ix0, iy0 = max(ec0, ox), max(er0, oy)
                    ix1 = min(ec1, ox + info["width"])
                    iy1 = min(er1, oy + info["height"])
                    if ix1 <= ix0 or iy1 <= iy0:
                        continue
                    with rasterio.open(info["path"]) as src:
                        frag = src.read(
                            window=Window(ix0 - ox, iy0 - oy,
                                          ix1 - ix0, iy1 - iy0))
                    if frag.ndim == 2:
                        frag = frag[np.newaxis, :, :]
                    w = frag != nodata
                    into = (slice(None), slice(iy0 - er0, iy1 - er0),
                            slice(ix0 - ec0, ix1 - ec0))
                    fresh = w & ~filled[into]
                    if fresh.any():
                        target = comp[into]
                        target[fresh] = frag[fresh]
                        filled[into] |= fresh
                    if filled.all():
                        break
                core = (slice(None), slice(r0 - er0, r1 - er0),
                        slice(c0 - ec0, c1 - ec0))
                if count == 1:
                    done = _finish_ops(
                        comp[0], resolution, fill_dist_m, smooth_sigma,
                        median_size, nodata)
                    dst.write(done[core[1], core[2]].astype(np.float32), 1,
                              window=Window(c0 - box[0], r0 - box[1],
                                            c1 - c0, r1 - r0))
                else:
                    bands = np.stack([
                        _finish_ops(comp[b], resolution, fill_dist_m,
                                    smooth_sigma, median_size, nodata)
                        for b in range(count)
                    ]).astype(np.float32)
                    dst.write(bands[(slice(None),) + core[1:]],
                              window=Window(c0 - box[0], r0 - box[1],
                                            c1 - c0, r1 - r0))
    if skipped:
        logger.info(
            "finish_mosaic: %d/%d void windows skipped for %s",
            skipped, total_windows, out_path.name,
        )
    return str(out_path)


def _same_grid(paths: list[str | Path]):
    """Plan check across layer files (stack inputs): plan or skipped reason."""
    return _mosaic_plan(paths)


def stack_mosaic(
    layers: list[str | Path],
    out_path: str | Path,
    *,
    op,
    halo_cells: int = 0,
    extent: str = "union",
    crs=None,
    nodata: float = NODATA,
    metadata: dict | None = None,
    dtype: str = "float32",
    memory_budget_bytes: int | None = None,
    notes: list | None = None,
) -> str:
    """Process aligned raster layers window by window into one product.

    ``op(expanded_layers, info)`` receives each layer's halo-expanded
    window and returns the FULL expanded array; the writer trims ``halo``
    cells per side, so core cells match a whole-array run exactly whenever
    the op's support fits the halo (document the support at the call site;
    shift-variant ops such as multi-resolution composites stay direct).
    ``extent`` is ``"union"`` or ``"intersection"`` of the layers' valid
    boxes on the first layer's grid. The expanded reads clamp to the
    output extent (border behavior identical to the cropped legacy arrays)
    and to each layer's footprint (outside reads as NODATA).
    Heterogeneous grids raise ValueError (the caller keeps its legacy
    full-array path for those). One notice goes to ``notes`` (same
    resolution, slower) for the engine to forward. Void windows are NOT
    skipped here (unlike finish/dem_mosaic): ``op`` is arbitrary and only
    the caller knows whether it preserves voids.
    """
    import rasterio
    from rasterio.windows import Window

    ordered = [str(Path(s)) for s in layers]
    if not ordered:
        raise ValueError("No layers to stack")
    out_path = Path(out_path)
    plan = _mosaic_plan(ordered)
    if "skipped" in plan:
        raise ValueError(f"stack_mosaic needs one grid: {plan['skipped']}")
    try:
        budget = int(memory_budget_bytes or 0)
    except (TypeError, ValueError):
        budget = 0
    halo = max(0, int(halo_cells))
    res_x, res_y = plan["res_x"], plan["res_y"]
    boxes = []
    for info, (ox, oy) in zip(plan["infos"], plan["placements"]):
        box = _valid_box_streaming(info["path"], nodata)
        if box is None:
            if extent == "intersection":
                raise ValueError("stack_mosaic: a layer has no valid data")
            continue
        boxes.append((box[0] + ox, box[1] + oy, box[2] + ox, box[3] + oy))
    if extent == "intersection":
        if len(boxes) != len(ordered):
            raise ValueError("stack_mosaic: a layer has no valid data")
        box = (
            max(b[0] for b in boxes), max(b[1] for b in boxes),
            min(b[2] for b in boxes), min(b[3] for b in boxes),
        )
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError("stack_mosaic: layers do not overlap")
    else:
        if not boxes:
            box = (0, 0, plan["width"], plan["height"])
        else:
            box = (
                min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes),
            )
    out_w, out_h = box[2] - box[0], box[3] - box[1]
    out_transform = (
        Affine(res_x, 0.0, plan["west"] + box[0] * res_x,
               0.0, -res_y, plan["north"] - box[1] * res_y)
    )
    itemsize = int(np.dtype(dtype).itemsize)
    side = _window_side(max(budget, 1), itemsize * len(ordered),
                        _WINDOW_WORKING_COPIES)
    logger.info(
        "stack_mosaic: %d layers -> %s in windows (%d px, halo %d)",
        len(ordered), out_path.name, side, halo,
    )
    _note_bounded(
        notes,
        f"Large stack {out_path.name}: processed in windows "
        f"({side} px) at full resolution; slower, identical pixels.",
    )
    # The plan sorts for determinism, but layers are positional for the
    # op (DTM first, DSM second): resolve each input in caller order.
    by_path = {info["path"]: (info, place)
               for info, place in zip(plan["infos"], plan["placements"])}
    layers = [by_path[p] for p in ordered]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    info = {"nodata": nodata, "res_x": res_x, "res_y": res_y,
            "transform": out_transform}
    with _open_tiled_output(
        out_path, out_w, out_h, out_transform, plan["crs"], nodata, 1,
        dtype, metadata,
    ) as dst:
        for r0 in range(box[1], box[3], side):
            for c0 in range(box[0], box[2], side):
                c1, r1 = min(box[2], c0 + side), min(box[3], r0 + side)
                ec0 = max(box[0], max(0, c0 - halo))
                er0 = max(box[1], max(0, r0 - halo))
                ec1 = min(box[2], min(plan["width"], c1 + halo))
                er1 = min(box[3], min(plan["height"], r1 + halo))
                wins = []
                for info_l, (ox, oy) in layers:
                    win = np.full((er1 - er0, ec1 - ec0), nodata,
                                  dtype=np.float32)
                    ix0, iy0 = max(ec0, ox), max(er0, oy)
                    ix1 = min(ec1, ox + info_l["width"])
                    iy1 = min(er1, oy + info_l["height"])
                    if ix1 > ix0 and iy1 > iy0:
                        with rasterio.open(info_l["path"]) as src:
                            win[iy0 - er0:iy1 - er0, ix0 - ec0:ix1 - ec0] = (
                                src.read(1, window=Window(
                                    ix0 - ox, iy0 - oy, ix1 - ix0, iy1 - iy0)))
                    wins.append(win)
                done = np.asarray(
                    op(wins, dict(info)), dtype=dtype)
                core = done[r0 - er0:r1 - er0, c0 - ec0:c1 - ec0]
                dst.write(core.astype(dtype), 1,
                          window=Window(c0 - box[0], r0 - box[1],
                                        c1 - c0, r1 - r0))
    return str(out_path)


def dem_mosaic(
    in_path: str | Path,
    out_path: str | Path,
    *,
    dem_fn,
    halo_cells: int = 1,
    copies: float = 6.0,
    nodata: float = NODATA,
    crs_out: str | None = None,
    metadata: dict | None = None,
    memory_budget_bytes: int | None = None,
    notes: list | None = None,
) -> tuple[str, bool]:
    """Apply a local DEM op over a mosaic, direct or windowed.

    ``dem_fn(masked_f64, cell_x, cell_y)`` runs on the load_dem_masked view
    (file tag + NaN void, float64). Halo 1 covers 3x3 stencils (Horn); the
    writer trims it, so core cells match a whole-array run exactly. Mode
    selects by data like :func:`finish_mosaic`. Returns ``(path, empty)``:
    an all-NODATA input writes a same-shape NODATA grid (legacy behavior)
    with ``empty=True`` so the caller keeps its warning. The windowed mode
    appends one notice to ``notes`` (same resolution, slower).
    ``dem_fn`` must be void-preserving (all-NODATA in, all-NODATA out —
    true of the Horn operators): windows provably holding no valid data
    skip the read and the compute with byte-identical results.
    """
    import rasterio
    from rasterio.windows import Window

    in_path, out_path = Path(in_path), Path(out_path)
    try:
        budget = int(memory_budget_bytes or 0)
    except (TypeError, ValueError):
        budget = 0
    with rasterio.open(str(in_path)) as src:
        width, height = int(src.width), int(src.height)
        transform = src.transform
        tag = src.nodata
        file_crs = src.crs
        src_crs = str(file_crs) if file_crs else None
    try:
        tag_value = None if tag is None else float(tag)
    except (TypeError, ValueError):
        tag_value = None

    def masked(window_arr: np.ndarray) -> np.ndarray:
        work = np.asarray(window_arr, dtype=np.float64)
        valid = np.isfinite(work)
        if tag_value is not None:
            valid &= work != tag_value
        return np.where(valid, work, float(nodata))

    box = None
    # Coarse validity occupancy (1024 px blocks holding >= 1 valid cell),
    # recorded during the same streaming scan: windows whose expanded
    # region touches no valid block skip the read and the compute exactly.
    occ: set[tuple[int, int]] = set()
    with rasterio.open(str(in_path)) as src:
        strip = _VOID_BLOCK
        ncb = (width + _VOID_BLOCK - 1) // _VOID_BLOCK
        r0, c0 = height, width
        r1, c1 = -1, -1
        for start in range(0, height, strip):
            rows = min(strip, height - start)
            raw = src.read(
                1, window=Window(0, start, width, rows)).astype(np.float64)
            valid = np.isfinite(raw)
            if tag_value is not None:
                valid &= raw != tag_value
            if not valid.any():
                continue
            for bc in range(ncb):
                if valid[:, bc * _VOID_BLOCK:min(
                        width, (bc + 1) * _VOID_BLOCK)].any():
                    occ.add((bc, start // _VOID_BLOCK))
            hit_rows = np.flatnonzero(valid.any(axis=1)) + start
            hit_cols = np.flatnonzero(valid.any(axis=0))
            r0 = min(r0, int(hit_rows[0]))
            r1 = max(r1, int(hit_rows[-1]))
            c0 = min(c0, int(hit_cols[0]))
            c1 = max(c1, int(hit_cols[-1]))
        if r1 >= 0:
            box = (c0, r0, c1 + 1, r1 + 1)
    if box is None:
        grid = np.full((height, width), nodata, dtype=np.float32)
        write_geotiff(out_path, grid, transform,
                      crs=crs_out or src_crs, nodata=nodata, metadata=metadata)
        return str(out_path), True

    cell_x, cell_y = abs(float(transform.a)), abs(float(transform.e))
    need = width * height * 8 * float(copies)
    if budget <= 0 or need <= budget:
        arr, _, _, _ = load_dem_masked(in_path, float(nodata))
        grid = np.asarray(
            dem_fn(arr, cell_x, cell_y), dtype=np.float32)
        write_geotiff(out_path, grid, transform,
                      crs=crs_out or src_crs, nodata=nodata, metadata=metadata)
        return str(out_path), False

    halo = max(0, int(halo_cells))
    side = _window_side(budget, 8, int(math.ceil(copies)))
    logger.info(
        "dem_mosaic: %s -> %s in windows (%d px, halo %d)",
        in_path.name, out_path.name, side, halo,
    )
    _note_bounded(
        notes,
        f"Large raster {out_path.name}: processed in windows "
        f"({side} px) at full resolution; slower, identical pixels.",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    skipped = 0
    total_windows = 0
    with _open_tiled_output(
        out_path, width, height, transform, file_crs, nodata, 1, "float32",
        metadata,
    ) as dst:
        for r0w in range(0, height, side):
            for c0w in range(0, width, side):
                c1w, r1w = min(width, c0w + side), min(height, r0w + side)
                ec0, er0 = max(0, c0w - halo), max(0, r0w - halo)
                ec1, er1 = min(width, c1w + halo), min(height, r1w + halo)
                total_windows += 1
                window = Window(c0w, r0w, c1w - c0w, r1w - r0w)
                if not any(
                    (bc, br) in occ
                    for br in range(er0 // _VOID_BLOCK,
                                    (er1 - 1) // _VOID_BLOCK + 1)
                    for bc in range(ec0 // _VOID_BLOCK,
                                    (ec1 - 1) // _VOID_BLOCK + 1)
                ):
                    # Provably void at block granularity (dem_fn is
                    # void-preserving by contract): identical NODATA
                    # without the read or the compute.
                    dst.write(
                        np.full((r1w - r0w, c1w - c0w), nodata,
                                dtype=np.float32),
                        1, window=window)
                    skipped += 1
                    continue
                with rasterio.open(str(in_path)) as src:
                    raw = src.read(
                        1, window=Window(ec0, er0, ec1 - ec0, er1 - er0))
                done = np.asarray(
                    dem_fn(masked(raw), cell_x, cell_y), dtype=np.float32)
                core = done[r0w - er0:r1w - er0, c0w - ec0:c1w - ec0]
                dst.write(core.astype(np.float32), 1, window=window)
    if skipped:
        logger.info(
            "dem_mosaic: %d/%d void windows skipped for %s",
            skipped, total_windows, out_path.name,
        )
    return str(out_path), False


def _raster_shape(path: str | Path):
    """Cheap (width, height, transform, crs, nodata) without pixel reads."""
    import rasterio

    with rasterio.open(str(path)) as src:
        return (int(src.width), int(src.height), src.transform, src.crs,
                src.nodata)


#: CHM working copies (float32-equivalents) by phase, documented upper
#: estimates: reads(2, freed after align) + aligned(2) + mask/out(2) peek
#: at 6; the NODATA-aware smooth holds ~9 more (float64 data/weights);
#: the optional resample holds ~3; the pit-free composite on the union
#: holds ~14 (float64 grids + maxima). Phases never overlap past these.
def _chm_overlap_plan(dtm_path, dsm_path, nodata) -> dict:
    """Overlap sizing for the heterogeneous CHM path (streaming only).

    No full reads: per-file valid boxes stream strip by strip, shapes come
    from headers. Returns pixel windows per input (1px safety margin,
    clamped), overlap/union cell counts and emptiness flags. Reads through
    these windows are cell-identical to full reads, so every downstream
    computation (bounds, align, subtract, smooth, crop) is byte-identical
    while the peak tracks the overlap instead of the union.
    """
    from rasterio.windows import Window

    plan: dict = {
        "empty_dtm": False, "empty_dsm": False, "disjoint": False,
        "dtm_win": None, "dsm_win": None,
        "overlap_cells": 0, "union_cells": 0,
    }
    infos = []
    for path in (dtm_path, dsm_path):
        width, height, transform, _crs, _nd = _raster_shape(path)
        box = _valid_box_streaming(str(path), nodata)
        infos.append((width, height, transform, box))
        plan["union_cells"] += width * height
    (w1, h1, t1, box1), (w2, h2, t2, box2) = infos
    if box1 is None:
        plan["empty_dtm"] = True
        return plan
    if box2 is None:
        plan["empty_dsm"] = True
        return plan

    def world_box(transform, box):
        c0, r0, c1, r1 = box
        corners = [transform * (c, r) for c, r in
                   ((c0, r0), (c1, r0), (c0, r1), (c1, r1))]
        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]
        return (min(xs), min(ys), max(xs), max(ys))

    wb1 = world_box(t1, box1)
    wb2 = world_box(t2, box2)
    ox0, oy0 = max(wb1[0], wb2[0]), max(wb1[1], wb2[1])
    ox1, oy1 = min(wb1[2], wb2[2]), min(wb1[3], wb2[3])
    if ox0 >= ox1 or oy0 >= oy1:
        plan["disjoint"] = True
        return plan
    for key, (w, h, transform) in (("dtm_win", infos[0][:3]),
                                   ("dsm_win", infos[1][:3])):
        inv = ~transform
        cols, rows = [], []
        for wx, wy in ((ox0, oy0), (ox1, oy0), (ox0, oy1), (ox1, oy1)):
            c, r = inv * (wx, wy)
            cols.append(c)
            rows.append(r)
        c0 = max(0, int(np.floor(min(cols))) - 1)
        r0 = max(0, int(np.floor(min(rows))) - 1)
        c1 = min(int(w), int(np.ceil(max(cols))) + 1)
        r1 = min(int(h), int(np.ceil(max(rows))) + 1)
        if c1 <= c0 or r1 <= r0:
            plan["disjoint"] = True
            return plan
        plan[key] = Window(c0, r0, c1 - c0, r1 - r0)
    plan["overlap_cells"] = max(
        plan["dtm_win"].width * plan["dtm_win"].height,
        plan["dsm_win"].width * plan["dsm_win"].height)
    return plan


def _chm_peak_need(plan: dict, *, smooth: bool, outres: bool,
                  pit: bool) -> int:
    """Peak working bytes for the CHM legacy path (documented estimate).

    Single formula behind routing and refusal alike (they can never
    disagree again): windowed phases scale by overlap, full-read phases
    (pit) by the union. Copies count live temporaries by active option —
    reads freed after align, then the smooth phase dominates.
    """
    union = max(int(plan.get("union_cells") or 0), 1)
    overlap = max(int(plan.get("overlap_cells") or 0), 1)
    if pit:
        return int(14.0 * union * 4)
    copies = 6.0 + (5.0 if smooth else 0.0) + (3.0 if outres else 0.0)
    return int(copies * overlap * 4)


def _chm_refuse(plan: dict, sources: list, budget_bytes,
                smooth: bool, outres: bool, pit: bool) -> None:
    """Honest refusal for the CHM legacy path, sized from cell counts.

    The need is the SAME peak the router measures (overlap cells × live
    temporaries, union cells for the pit full-read path) — never a
    file-size guess, so routing and refusal cannot disagree: a legacy
    run that fits never refuses, and an oversized unstackable grid
    always refuses instead of OOMing opaquely. File sizes appear only
    as context in the message, whose shape (and catalog key) never
    changes.
    """
    try:
        budget = int(budget_bytes or 0)
    except (TypeError, ValueError):
        return
    if budget <= 0:
        return
    need = _chm_peak_need(plan, smooth=smooth, outres=outres, pit=pit)
    if need <= budget:
        return
    total = 0
    try:
        for path in sources:
            total += Path(path).stat().st_size
    except OSError:
        return
    raise ValueError(
        f"CHM mosaic would need ~{need // (1024 ** 2):,} MiB "
        f"(inputs total ~{total // (1024 ** 2):,} MiB on disk, working "
        "copies included), "
        f"above the ~{budget // (1024 ** 2):,} MiB budget. Wire same-grid "
        "inputs, raise the memory budget, or run fewer products at once."
    )


def compute_chm(
    dtm_path: str | Path,
    dsm_path: str | Path,
    out_path: str | Path,
    crs: str | None = None,
    nodata: float = NODATA,
    smooth_enabled: bool = True,
    smooth_sigma: float = 0.5,
    max_height_m: float = 60.0,
    min_height_m: float = 0.0,
    metadata: dict | None = None,
    dtm_arr: np.ndarray | None = None,
    dtm_transform=None,
    dsm_arr: np.ndarray | None = None,
    dsm_transform=None,
    memory_budget_bytes: int | None = None,
    _overlap_plan: dict | None = None,
) -> str:
    """CHM = DSM - DTM where both have valid data; NODATA elsewhere.

    The two mosaics may differ in extent: they are aligned by the
    intersection of their valid bounds before subtraction. Heights are
    clamped to [min_height_m, max_height_m] and optional Gaussian
    smoothing is applied. Pre-loaded ``dtm_arr``/``dsm_arr`` (+ transforms)
    skip the file reads when the caller already holds them. Heterogeneous
    extents read only the overlap windows (cell-identical to full reads,
    so every step below is byte-identical) while the peak tracks the
    overlap instead of the union; refusal is sized the same way, with
    temporaries counted by active option. Windowed same-grid work goes
    through ``stack_mosaic`` instead, never here.
    """
    if memory_budget_bytes:
        # Temporaries by live option (reads freed after align, then the
        # smooth phase dominates): base 6, smooth +5, pit path sizes on
        # the union below via the preloaded branch.
        if dtm_arr is None and dsm_arr is None:
            if _overlap_plan is None:
                _overlap_plan = _chm_overlap_plan(
                    dtm_path, dsm_path, nodata)
            if (_overlap_plan["empty_dtm"] or _overlap_plan["empty_dsm"]
                    or _overlap_plan["disjoint"]):
                pass  # same ValueErrors as the full-read path below
            else:
                _chm_refuse(_overlap_plan, [dtm_path, dsm_path],
                            memory_budget_bytes, smooth_enabled, False,
                            False)
        else:
            copies = 8.0 + (9.0 if smooth_enabled else 0.0)
            refuse_if_unfit(
                [dtm_path, dsm_path], copies, memory_budget_bytes,
                "CHM mosaic",
            )
    if dtm_arr is None and dsm_arr is None:
        if _overlap_plan is None:
            _overlap_plan = _chm_overlap_plan(dtm_path, dsm_path, nodata)
        if _overlap_plan["empty_dtm"] or _overlap_plan["empty_dsm"]:
            raise ValueError("compute_chm: DTM/DSM without valid data")
        if _overlap_plan["disjoint"]:
            raise ValueError("compute_chm: DTM and DSM do not overlap")
        import rasterio

        with rasterio.open(str(dtm_path)) as src:
            dtm_arr = src.read(
                1, window=_overlap_plan["dtm_win"]).astype(np.float32)
            dtm_transform = src.window_transform(_overlap_plan["dtm_win"])
            crs_a = src.crs
        with rasterio.open(str(dsm_path)) as src:
            dsm_arr = src.read(
                1, window=_overlap_plan["dsm_win"]).astype(np.float32)
            dsm_transform = src.window_transform(_overlap_plan["dsm_win"])
            crs_b = src.crs
    else:
        if dtm_arr is None:
            dtm_arr, dtm_transform, crs_a = read_geotiff(dtm_path)
        else:
            crs_a = None
        if dsm_arr is None:
            dsm_arr, dsm_transform, crs_b = read_geotiff(dsm_path)
        else:
            crs_b = None
    crs_out = crs or crs_a or crs_b

    bounds_dtm = _valid_bounds(dtm_arr, dtm_transform, nodata)
    bounds_dsm = _valid_bounds(dsm_arr, dsm_transform, nodata)
    if bounds_dtm is None or bounds_dsm is None:
        raise ValueError("compute_chm: DTM/DSM without valid data")
    common = (
        max(bounds_dtm[0], bounds_dsm[0]),
        max(bounds_dtm[1], bounds_dsm[1]),
        min(bounds_dtm[2], bounds_dsm[2]),
        min(bounds_dtm[3], bounds_dsm[3]),
    )
    if common[0] >= common[2] or common[1] >= common[3]:
        raise ValueError("compute_chm: DTM and DSM do not overlap")

    inv = ~dtm_transform
    c0f, r0f = inv * (common[0], common[3])
    c1f, r1f = inv * (common[2], common[1])
    c0, r0 = int(round(float(c0f))), int(round(float(r0f)))
    shape = (
        int(round(float(r1f))) - r0,
        int(round(float(c1f))) - c0,
    )
    if shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("compute_chm: empty overlap window")
    target_transform = dtm_transform * Affine.translation(c0, r0)

    dtm_w = align_to_grid(dtm_arr, dtm_transform, shape, target_transform, nodata)
    dsm_w = align_to_grid(dsm_arr, dsm_transform, shape, target_transform, nodata)
    del dtm_arr, dsm_arr  # the windowed reads served their purpose

    both = (dtm_w != nodata) & (dsm_w != nodata)
    chm = np.full(shape, nodata, dtype=np.float32)
    if both.any():
        vals = (dsm_w[both] - dtm_w[both]).astype(np.float32)
        vals = np.clip(vals, min_height_m, max_height_m)
        chm[both] = vals
        del vals
    del dtm_w, dsm_w, both

    # NODATA-aware smoothing
    if smooth_enabled and smooth_sigma > 0:
        chm = smooth_nodata(chm, smooth_sigma, nodata)

    cropped, crop_transform = _crop_to_valid(chm, target_transform, nodata)
    write_geotiff(
        out_path, cropped, crop_transform, crs=crs_out, nodata=nodata, metadata=metadata
    )
    return str(out_path)