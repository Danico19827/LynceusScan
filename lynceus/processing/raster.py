# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Raster processing utilities for point-cloud tasks without Qt.

The module provides progressive morphological filtering, core-masked tile
rasterization, GeoTIFF I/O, and aligned mosaic merging.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from affine import Affine
from rasterio.merge import merge as rio_merge
from scipy.ndimage import minimum_filter, maximum_filter, distance_transform_edt, gaussian_filter

RASTER_CELL_SIZE = 1.0
RASTER_GRID_BYTES_PER_CELL = 32
DEFAULT_RASTER_GRID_BUDGET_BYTES = 512 * 1024 * 1024
POINT_GRID_BYTES_PER_CELL = 96
NODATA = -9999.0
GROUND_CLASS = 2


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
    budget = _raster_grid_memory_budget(memory_budget_bytes)
    if estimated_bytes > budget:
        label = f" for tile {tile_id}" if tile_id else ""
        raise ValueError(
            f"{method} grid{label} at {resolution:g} m would require "
            f"{rows:,} x {cols:,} cells (~{estimated_bytes // (1024 ** 3):,} GiB), "
            f"above the per-worker grid budget ({budget // (1024 ** 2):,} MiB). "
            "Increase cell size or reduce tile size / worker count."
        )
    return x0, y0, cols, rows

# PMF defaults; node configuration can override them.
PMF_SLOPE = 0.3
PMF_INTERCEPT = 0.3
PMF_INITIAL_WINDOW_M = 2.0
PMF_MAX_WINDOW_M = 8.0


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

    A minimum-elevation surface is built per cell and processed with growing
    morphological windows. Cells whose elevation change stays below the
    window-dependent threshold are retained as ground.
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

    window_m = initial_window_m
    while window_m <= max_window_m:
        window_cells = max(2, int(round(window_m / cell_size)))
        opened = maximum_filter(minimum_filter(surface, size=window_cells), size=window_cells)
        diff = surface - opened
        threshold = slope * window_m + intercept
        removed = ground & (diff > threshold)
        if not removed.any():
            break
        ground &= ~removed
        surface = np.where(removed, opened, surface)
        window_m *= 2.0

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
    iterations: int = 300,
    gravity: float = 0.2,
    memory_budget_bytes: int | None = None,
    tile_id: str | None = None,
) -> np.ndarray:
    """Return a boolean ground mask using a simplified Cloth Simulation
    Filter (Zhang et al., 2016).

    Heights are flipped so bare earth becomes the ceiling; a cloth grid
    starts above everything and settles under gravity, spring tension to
    its 4-neighbors (``rigidness`` passes per step) and collision with
    the per-cell maximum. Points close under the settled cloth are
    ground; small objects stay buried because neighbor tension holds
    the cloth above them.

    Simplified vs the paper: uniform gravity, fixed stiffness from
    ``rigidness`` (1 = flat/loosely draped, 3 = tight/steep-following),
    nearest-cell collision, no slope post-processing. ``rigidness``
    accepts 1..3.
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

    Like PMF this opens a minimum-elevation surface with growing windows,
    but the elevation threshold stays FIXED at every scale (PMF grows it
    with the window) while the window itself adapts to local slope:
    steeper cells get larger windows, so flat/agricultural ground is
    cleaned aggressively without stripping real relief. Simplified vs the
    paper: quantized window levels instead of truly per-cell windows, no
    final slope-based relabeling.
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
# Core-masked per-tile rasterization.
# ---------------------------------------------------------------------------

def core_mask(x: np.ndarray, y: np.ndarray, tile: dict) -> np.ndarray:
    """Return a mask for points inside a tile core, excluding its buffer."""
    return (
        (x >= tile["core_x_min"])
        & (x < tile["core_x_max"])
        & (y >= tile["core_y_min"])
        & (y < tile["core_y_max"])
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
    col_ratio = width / resolution
    row_ratio = height / resolution
    if not math.isfinite(col_ratio) or not math.isfinite(row_ratio):
        raise ValueError("Raster grid dimensions are not finite")
    cols = max(1, math.ceil(col_ratio))
    rows = max(1, math.ceil(row_ratio))
    cell_count = rows * cols
    budget = _raster_grid_memory_budget(memory_budget_bytes)
    estimated_bytes = cell_count * RASTER_GRID_BYTES_PER_CELL
    if estimated_bytes > budget:
        tile_id = tile.get("tile_id", "unknown")
        estimated_gib = estimated_bytes // (1024 ** 3)
        raise ValueError(
            f"Raster grid for tile {tile_id} at {resolution:g} m would require "
            f"{rows:,} x {cols:,} cells (~{estimated_gib:,} GiB), "
            f"above the per-worker grid budget ({budget / (1024 ** 2):.0f} MiB). "
            "Increase the cell size, reduce tile size, or lower the worker count."
        )
    return Affine(
        resolution, 0.0, x_min, 0.0, -resolution, y_max
    ), cols, rows


def _raster_grid_memory_budget(memory_budget_bytes: int | None) -> int:
    """Limit one tile's grid allocations to available and configured RAM."""
    configured = (
        int(memory_budget_bytes)
        if memory_budget_bytes is not None and int(memory_budget_bytes) > 0
        else DEFAULT_RASTER_GRID_BUDGET_BYTES
    )
    try:
        import psutil

        available_limit = int(psutil.virtual_memory().available // 4)
    except (ImportError, OSError, AttributeError):
        available_limit = DEFAULT_RASTER_GRID_BUDGET_BYTES
    return max(1, min(configured, available_limit))


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
):
    """Merges aligned GeoTIFFs (exact mosaic, no resampling) into a single one.

    The mosaic is cropped to the data bbox: tile-grid margins without points
    are left outside the file. ``metadata`` is forwarded to ``write_geotiff``.
    With ``return_array=True`` the merged (cropped) array is returned instead
    of written, so barriers can post-process before the single write:
    ``(array, transform, crs)``.
    """
    datasets = [str(p) for p in paths]
    if not datasets:
        raise ValueError("No tile rasters to merge")
    crs = crs or "EPSG:4326"
    dest, transform = rio_merge(datasets, nodata=nodata, dtype="float32")
    cropped, crop_transform = _crop_to_valid(dest[0], transform, nodata)
    if return_array:
        return cropped, crop_transform, crs
    write_geotiff(
        out_path, cropped, crop_transform, crs=crs, nodata=nodata, metadata=metadata
    )
    return str(out_path)


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
) -> str:
    """CHM = DSM - DTM where both have valid data; NODATA elsewhere.

    The two mosaics may differ in extent: they are aligned by the
    intersection of their valid bounds before subtraction. Heights are
    clamped to [min_height_m, max_height_m] and optional Gaussian
    smoothing is applied. Pre-loaded ``dtm_arr``/``dsm_arr`` (+ transforms)
    skip the file reads when the caller already holds them.
    """
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

    both = (dtm_w != nodata) & (dsm_w != nodata)
    chm = np.full(shape, nodata, dtype=np.float32)
    if both.any():
        vals = (dsm_w[both] - dtm_w[both]).astype(np.float32)
        vals = np.clip(vals, min_height_m, max_height_m)
        chm[both] = vals

    # NODATA-aware smoothing
    if smooth_enabled and smooth_sigma > 0:
        chm = smooth_nodata(chm, smooth_sigma, nodata)

    cropped, crop_transform = _crop_to_valid(chm, target_transform, nodata)
    write_geotiff(
        out_path, cropped, crop_transform, crs=crs_out, nodata=nodata, metadata=metadata
    )
    return str(out_path)