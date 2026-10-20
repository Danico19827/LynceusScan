# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Native Qt/OpenGL 3D viewer for point-cloud products.

Loads data in a worker thread, applies a bounded spatial budget, and renders
with a single OpenGL draw call. It provides an orbital camera, axes, grid, and
RGB/classification/elevation color modes.

Tiles are shown with their buffers to avoid visual gaps; the export node is
responsible for removing overlap from final products.
"""

import hashlib
import math
import os
import sys
from pathlib import Path

import laspy
import numpy as np
from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_FLOAT,
    GL_LEQUAL,
    GL_LINES,
    GL_POINTS,
    GL_RENDERER,
    GL_VENDOR,
    glClear,
    glClearColor,
    glDepthFunc,
    glDrawArrays,
    glEnable,
    glGetError,
    glGetString,
    glPointSize,
    glUniform1f,
)

from PySide6.QtCore import QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QMatrix4x4, QPixmap, QSurfaceFormat, QVector3D
from PySide6.QtOpenGL import (
    QOpenGLBuffer,
    QOpenGLShader,
    QOpenGLShaderProgram,
    QOpenGLVertexArrayObject,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from lynceus.plugins.locale import t
from lynceus.ui.viewers.base import BaseViewer, viewer_title
from lynceus.ui.viewers.palette import elevation_colors
from lynceus.ui.viewers.registry import register

# ---------------------------------------------------------------------------
# OpenGL constants (avoid relying on constant bindings)
# ---------------------------------------------------------------------------
DISPLAY_POINT_BUDGET = 1_500_000
MIN_DISPLAY_POINT_BUDGET = 750_000
MAX_DISPLAY_POINT_BUDGET = 100_000_000
POINT_MEMORY_ESTIMATE = 88  # retained arrays, upload preparation, and VBO allowance
VIEWER_CHUNK = 1_000_000

def legend_entries(classes: np.ndarray | None) -> list[tuple[int, str, tuple]]:
    """Legend rows for the ASPRS classes present: ``(code, name, rgb)``.

    Pure helper (unit-tested): unknown codes fall back to ``Class {n}``
    grey so future LAS classes never break the legend.
    """
    if classes is None:
        return []
    rows = []
    for code in sorted({int(c) for c in np.unique(classes)}):
        if 0 <= code < len(CLASS_NAMES):
            name = CLASS_NAMES[code]
        else:
            name = f"Class {code}"
        if 0 <= code < len(CLASS_COLORS):
            rgb = tuple(float(v) for v in CLASS_COLORS[code])
        else:
            rgb = (0.65, 0.65, 0.65)
        rows.append((code, name, rgb))
    return rows


def _legend_position(view_w: int, view_h: int, legend_w: int,
                     legend_h: int, margin: int = 8):
    """Top-right corner for the floating legend (pure helper, tested)."""
    from PySide6.QtCore import QPoint

    return QPoint(max(0, view_w - legend_w - margin), margin)


# ASPRS LAS class names for the classification legend (catalog keys).
CLASS_NAMES = (
    "Created",             # 0 never classified
    "Unclassified",        # 1
    "Ground",              # 2
    "Low vegetation",      # 3
    "Medium vegetation",   # 4
    "High vegetation",     # 5
    "Building",            # 6
    "Low point",           # 7 noise
    "Model key-point",     # 8
    "Water",               # 9
    "Rail",                # 10
    "Road surface",        # 11
    "Overlap",             # 12
    "Wire guard",          # 13
    "Wire conductor",      # 14
    "Transmission tower",  # 15
    "Wire connector",      # 16
    "Bridge deck",         # 17
    "High noise",          # 18
)

BACKGROUND = (0.043, 0.059, 0.102)  # #0b0f1a
SINGLE_COLOR = (0.49, 0.62, 0.83)  # #7d9fd4
GRID_COLOR = (0.18, 0.23, 0.31)  # #2f3a4f
CLASS_COLORS = np.array(
    [
        (0.55, 0.55, 0.55),  # 0 created / never classified
        (0.75, 0.75, 0.75),  # 1 unclassified
        (0.32, 0.70, 0.24),  # 2 ground
        (0.42, 0.56, 0.24),  # 3 low vegetation
        (0.12, 0.52, 0.16),  # 4 medium vegetation
        (0.05, 0.35, 0.10),  # 5 high vegetation
        (0.25, 0.25, 0.28),  # 6 building
        (0.92, 0.25, 0.18),  # 7 low noise
        (0.95, 0.70, 0.12),  # 8 model key point
        (0.10, 0.55, 0.85),  # 9 water
        (0.55, 0.35, 0.20),  # 10 rail
        (0.70, 0.25, 0.75),  # 11 road surface
        (0.80, 0.45, 0.15),  # 12 overlap
        (0.85, 0.15, 0.55),  # 13 wire guard
        (0.75, 0.75, 0.20),  # 14 wire conductor
        (0.95, 0.20, 0.65),  # 15 transmission tower
        (0.45, 0.45, 0.50),  # 16 wire connector
        (0.85, 0.15, 0.15),  # 17 bridge deck
        (0.98, 0.05, 0.05),  # 18 high noise
    ],
    dtype=np.float32,
)
_CLASS_COLOR_LUT = np.full((256, 3), (0.65, 0.65, 0.65), dtype=np.float32)
_CLASS_COLOR_LUT[: len(CLASS_COLORS)] = CLASS_COLORS


def classification_colors(classes: np.ndarray) -> np.ndarray:
    """Map LAS classes to stable colors without allocating per class names."""
    values = np.asarray(classes)
    if values.dtype == np.uint8:
        return _CLASS_COLOR_LUT[values]
    values = np.asarray(classes, dtype=np.int64)
    colors = np.full((values.size, 3), (0.65, 0.65, 0.65), dtype=np.float32)
    valid = (values >= 0) & (values < len(CLASS_COLORS))
    colors[valid] = CLASS_COLORS[values[valid]]
    return colors


INTEGRATED_AUTO_POINT_CAP = 6_000_000

_detected_gpu_class: str | None = None


def classify_gpu(vendor: str, renderer: str) -> str:
    """Bucket a GL renderer: discrete/integrated/software/unknown.

    Heuristic from the renderer string (documented approximation, unit
    tested): unknown renderers are never punished (treated as discrete).
    """
    text = f"{vendor or ''} {renderer or ''}".lower()
    if any(mark in text for mark in (
            "llvmpipe", "swiftshader", "softpipe", "software")):
        return "software"
    if "arc" in text:
        return "discrete"  # Intel Arc dGPUs report an Intel vendor
    if any(mark in text for mark in (
            "intel", "mali", "adreno", "videocore", "apple",
            "radeon graphics", " vega", "vega ")):
        return "integrated"
    if any(mark in text for mark in ("nvidia", "geforce", "quadro",
                                     "radeon rx", "radeon pro", "tesla")):
        return "discrete"
    return "unknown"


def note_gpu_renderer(vendor, renderer) -> None:
    """Cache the machine GPU class from a live GL context (first wins)."""
    global _detected_gpu_class
    if _detected_gpu_class is not None:
        return
    try:
        def text(value) -> str:
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")
            return str(value or "")
        _detected_gpu_class = classify_gpu(text(vendor), text(renderer))
    except Exception:
        pass


def _auto_point_budget(available_bytes: int, gpu_class: str | None) -> int:
    """Auto point budget: 10% of available RAM, GPU-aware (pure)."""
    try:
        adaptive = int(int(available_bytes) * 0.10 / POINT_MEMORY_ESTIMATE)
    except (TypeError, ValueError, OverflowError):
        return DISPLAY_POINT_BUDGET
    if gpu_class in ("integrated", "software"):
        adaptive = min(adaptive, INTEGRATED_AUTO_POINT_CAP)
    return max(MIN_DISPLAY_POINT_BUDGET,
               min(MAX_DISPLAY_POINT_BUDGET, adaptive))


PREVIEW_CACHE_VERSION = 1
PREVIEW_CACHE_KEEP = 5


def preview_cache_dir(root=None) -> Path:
    """Local (non-roaming) dir for decimated preview clouds."""
    from lynceus.plugins.store import local_data_root

    override = root if root is not None else os.environ.get(
        "LYNCEUS_PREVIEW_CACHE_DIR")
    base = Path(override) if override else local_data_root()
    return base / "previews"


def preview_fingerprint(path: str) -> str | None:
    """Short identity of a source file (path, size, mtime)."""
    try:
        info = os.stat(path)
    except OSError:
        return None
    raw = f"{path}|{info.st_size}|{info.st_mtime_ns}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def preview_cache_name(fingerprint: str, budget: int) -> str:
    return f"preview_{fingerprint}_{int(budget)}.npz"


def _prune_preview_cache(directory: Path, keep: int = PREVIEW_CACHE_KEEP,
                         fingerprint: str | None = None) -> None:
    """Keep the newest entries (optionally only one file's lineage)."""
    try:
        entries = sorted(directory.glob("preview_*.npz"),
                         key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return
    if fingerprint is not None:
        entries = [p for p in entries if f"_{fingerprint}_" not in p.name]
    for stale in entries[keep:]:
        try:
            stale.unlink()
        except OSError:
            pass


def save_preview_cache(directory: Path, fingerprint: str, budget: int,
                       positions: np.ndarray, rgb: np.ndarray | None,
                       classes: np.ndarray | None,
                       origin: np.ndarray) -> Path | None:
    """Persist a loader-result preview (atomic tmp+rename, then prune)."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / preview_cache_name(fingerprint, budget)
        tmp = target.with_suffix(".tmp")
        payload = {
            "positions": np.ascontiguousarray(positions, dtype=np.float32),
            "origin": np.ascontiguousarray(origin, dtype=np.float64),
            "budget": np.array([int(budget),
                                1 if rgb is not None else 0,
                                1 if classes is not None else 0],
                               dtype=np.int64),
        }
        if rgb is not None:
            payload["rgb"] = np.ascontiguousarray(rgb, dtype=np.float32)
        if classes is not None:
            payload["classes"] = np.ascontiguousarray(classes, dtype=np.uint8)
        with open(tmp, "wb") as handle:
            np.savez(handle, **payload)
        os.replace(tmp, target)
        _prune_preview_cache(directory)
        return target
    except (OSError, ValueError):
        return None


def load_preview_cache(path: Path) -> tuple | None:
    """Read a cached preview; None when missing or unreadable."""
    try:
        with np.load(path, allow_pickle=False) as stored:
            positions = np.asarray(stored["positions"], dtype=np.float32)
            origin = np.asarray(stored["origin"], dtype=np.float64)
            budget, has_rgb, has_classes = (
                int(v) for v in stored["budget"])
            rgb = (np.asarray(stored["rgb"], dtype=np.float32)
                   if has_rgb and "rgb" in stored else None)
            classes = (np.asarray(stored["classes"], dtype=np.uint8)
                       if has_classes and "classes" in stored else None)
    except (OSError, ValueError, KeyError):
        return None
    if positions.ndim != 2 or positions.shape[1] != 3:
        return None
    count = positions.shape[0]
    if ((rgb is not None and rgb.shape[0] != count)
            or (classes is not None and classes.shape[0] != count)):
        return None
    return positions, rgb, classes, origin, int(budget)


def find_cached_preview(directory: Path, fingerprint: str,
                        budget: int) -> tuple[Path, int] | None:
    """Smallest cached budget of this file covering the request."""
    try:
        options = []
        for path in directory.glob(f"preview_{fingerprint}_*.npz"):
            try:
                stored = int(path.stem.rsplit("_", 1)[1])
            except (ValueError, IndexError):
                continue
            if stored >= budget:
                options.append((stored, path))
    except OSError:
        return None
    if not options:
        return None
    options.sort()
    return options[0][1], options[0][0]


def reservoir_downsample(positions: np.ndarray, rgb: np.ndarray | None,
                         classes: np.ndarray | None, cap: int,
                         seed: int) -> tuple:
    """Uniform subsample of resident arrays (deterministic by seed)."""
    count = positions.shape[0]
    if count <= cap or cap <= 0:
        return positions, rgb, classes
    scores = np.random.default_rng(seed).random(count)
    pick = np.argpartition(scores, cap - 1)[:cap]
    return (positions[pick],
            rgb[pick] if rgb is not None else None,
            classes[pick] if classes is not None else None)


def _settings_point_budget() -> int | None:
    """Preferences point budget (explicit preset) or None for auto."""
    try:
        from PySide6.QtCore import QSettings

        from lynceus.ui.settings_keys import (
            DISPLAY_BUDGET_DEFAULT,
            DISPLAY_BUDGET_KEY,
            SETTINGS_APP,
            SETTINGS_ORG,
        )

        raw = QSettings(SETTINGS_ORG, SETTINGS_APP).value(
            DISPLAY_BUDGET_KEY, DISPLAY_BUDGET_DEFAULT
        )
        if raw is None or str(raw).strip().lower() == "auto":
            return None
        return max(
            MIN_DISPLAY_POINT_BUDGET,
            min(MAX_DISPLAY_POINT_BUDGET, int(raw)),
        )
    except Exception:
        return None


def safe_display_point_budget() -> int:
    """Choose a point budget from current system memory with hard safety bounds."""
    try:
        import os
        import psutil

        preset = _settings_point_budget()
        if preset is not None:
            return preset
        override = os.environ.get("LYNCEUS_VIEWER_POINT_BUDGET", "").strip()
        if override:
            return max(
                MIN_DISPLAY_POINT_BUDGET,
                min(MAX_DISPLAY_POINT_BUDGET, int(override)),
            )
        available = int(psutil.virtual_memory().available)
        return _auto_point_budget(available, _detected_gpu_class)
    except (ImportError, TypeError, ValueError, OSError):
        return DISPLAY_POINT_BUDGET


def _localize_coordinates(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    origin: np.ndarray,
) -> np.ndarray:
    """Rebase projected coordinates before float32 conversion for OpenGL."""
    return np.column_stack((x - origin[0], y - origin[1], z - origin[2])).astype(
        np.float32, copy=False
    )


def _bounded_sample_merge(
    current: tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None] | None,
    scores: np.ndarray,
    positions: np.ndarray,
    rgb: np.ndarray | None,
    classes: np.ndarray | None,
    limit: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Keep only the lowest deterministic random priorities seen so far."""
    if current is not None and current[0].size >= limit:
        candidate_mask = scores < current[0].max()
        if not np.any(candidate_mask):
            return current
        scores = scores[candidate_mask]
        positions = positions[candidate_mask]
        if rgb is not None:
            rgb = rgb[candidate_mask]
        if classes is not None:
            classes = classes[candidate_mask]

    if current is None:
        merged = (scores, positions, rgb, classes)
    else:
        old_scores, old_positions, old_rgb, old_classes = current
        merged = (
            np.concatenate((old_scores, scores)),
            np.concatenate((old_positions, positions)),
            np.concatenate((old_rgb, rgb)) if old_rgb is not None else None,
            np.concatenate((old_classes, classes)) if old_classes is not None else None,
        )

    if merged[0].size <= limit:
        return merged
    selected = np.argpartition(merged[0], limit - 1)[:limit]
    return tuple(values[selected] if values is not None else None for values in merged)


POINT_VERTEX = """
#version 330 core
in vec3 aPos;
in vec3 aColor;
uniform mat4 uMVP;
uniform float uPointSize;
out vec3 vColor;
void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    gl_PointSize = uPointSize;
    vColor = aColor;
}
"""

LINE_VERTEX = """
#version 330 core
in vec3 aPos;
in vec3 aColor;
uniform mat4 uMVP;
out vec3 vColor;
void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    vColor = aColor;
}
"""

COLOR_FRAGMENT = """
#version 330 core
in vec3 vColor;
out vec4 fragColor;
void main() {
    fragColor = vec4(vColor, 1.0);
}
"""


class _PointCloudLoader:
    def __init__(self, tiles: list[dict], on_progress=None):
        self._tiles = tiles
        self._on_progress = on_progress
        self._progress_done = 0
        self._progress_total = 0

    def _account(self, count: int) -> None:
        """Determinate progress: decoded points vs header total."""
        callback = self._on_progress
        if callback is None or self._progress_total <= 0:
            return
        self._progress_done += max(0, int(count))
        try:
            callback(min(self._progress_done, self._progress_total),
                     self._progress_total)
        except Exception:
            pass

    def run(self, cancelled):
        try:
            point_budget = safe_display_point_budget()
            tile_cap = max(1, point_budget // max(1, len(self._tiles)))
            normalized_tiles = [
                tile if isinstance(tile, dict) else {"file": tile}
                for tile in self._tiles
            ]
            bounds = []
            totals = []
            for tile in normalized_tiles:
                if cancelled.is_set():
                    return None
                path = tile.get("file")
                if not path:
                    continue
                with laspy.open(path) as reader:
                    bounds.append(
                        (
                            np.asarray(reader.header.mins, dtype=np.float64),
                            np.asarray(reader.header.maxs, dtype=np.float64),
                        )
                    )
                    totals.append(
                        int(getattr(reader.header, "point_count", 0) or 0))
            if not bounds:
                return None
            self._progress_total = sum(totals)
            self._progress_done = 0
            origin = (
                np.min([item[0] for item in bounds], axis=0)
                + np.max([item[1] for item in bounds], axis=0)
            ) / 2.0
            cached = self._try_preview_cache(
                normalized_tiles, point_budget, origin, cancelled)
            if cached is not None:
                return cached

            pos_parts: list[np.ndarray] = []
            rgb_parts: list[np.ndarray | None] = []
            class_parts: list[np.ndarray | None] = []
            has_rgb = False
            has_classes = False
            for tile in normalized_tiles:
                if cancelled.is_set():
                    return None
                tile_pos, tile_rgb, tile_classes, tile_has_rgb, tile_has_classes = (
                    self._load_tile(tile, tile_cap, origin, cancelled)
                )
                if tile_pos is not None:
                    pos_parts.append(tile_pos)
                    rgb_parts.append(tile_rgb)
                    class_parts.append(tile_classes)
                    has_rgb |= tile_has_rgb
                    has_classes |= tile_has_classes

            if not pos_parts:
                return None

            positions = np.concatenate(pos_parts)
            rgb = (
                np.concatenate(
                    [
                        part
                        if part is not None
                        else np.tile(
                            np.array(SINGLE_COLOR, dtype=np.float32),
                            (pos_parts[index].shape[0], 1),
                        )
                        for index, part in enumerate(rgb_parts)
                    ]
                )
                if has_rgb
                else None
            )
            classes = (
                np.concatenate(
                    [
                        part
                        if part is not None
                        else np.full(pos_parts[index].shape[0], 255, dtype=np.uint8)
                        for index, part in enumerate(class_parts)
                    ]
                )
                if has_classes
                else None
            )
            self._save_preview_cache(
                normalized_tiles, point_budget, origin, cancelled,
                positions, rgb, classes)
            return positions, rgb, classes, None
        except MemoryError as exc:
            raise RuntimeError("Not enough memory to load the point cloud") from exc
        except Exception as exc:
            raise RuntimeError(f"Error loading point cloud: {exc}") from exc

    @staticmethod
    def _cacheable_single_file(tiles) -> str | None:
        """Source path when the payload is one plain file (else None)."""
        if len(tiles) != 1:
            return None
        tile = tiles[0]
        if not isinstance(tile, dict):
            return None
        if tile.get("crop_core", False):
            return None
        path = tile.get("file")
        return path if path else None

    def _try_preview_cache(self, tiles, budget: int, origin,
                           cancelled):
        """Instant uniform sample: exact hit, or downsample of a larger one.

        Misses (and multi-file payloads) fall through to the full decode;
        the winner is persisted afterwards by ``_save_preview_cache``.
        """
        path = self._cacheable_single_file(tiles)
        if path is None or cancelled.is_set():
            return None
        fingerprint = preview_fingerprint(path)
        if fingerprint is None:
            return None
        try:
            directory = preview_cache_dir()
        except Exception:
            return None
        found = find_cached_preview(directory, fingerprint, budget)
        if found is None:
            return None
        cached_path, stored_budget = found
        stored = load_preview_cache(cached_path)
        if stored is None:
            return None
        positions, rgb, classes, _origin, _stored = stored
        if positions.shape[0] > budget:
            seed = int.from_bytes(
                bytes.fromhex(fingerprint), "little") % (2 ** 31)
            positions, rgb, classes = reservoir_downsample(
                positions, rgb, classes, budget, seed)
        self._progress_done = self._progress_total
        if self._on_progress is not None and self._progress_total > 0:
            try:
                self._on_progress(self._progress_total, self._progress_total)
            except Exception:
                pass
        return positions, rgb, classes, None

    def _save_preview_cache(self, tiles, budget: int, origin,
                            cancelled, positions, rgb, classes) -> None:
        """Persist this uniform sample for instant future loads."""
        if cancelled.is_set():
            return
        path = self._cacheable_single_file(tiles)
        if path is None:
            return
        fingerprint = preview_fingerprint(path)
        if fingerprint is None:
            return
        try:
            directory = preview_cache_dir()
        except Exception:
            return
        save_preview_cache(directory, fingerprint, budget, positions, rgb,
                           classes, origin)

    def _load_tile(
        self, tile: dict, tile_cap: int, origin: np.ndarray, cancelled
    ):
        path = tile.get("file")
        if not path:
            return None, None, None, False, False
        has_core = tile.get("crop_core", False) and all(
            key in tile
            for key in ("core_x_min", "core_y_min", "core_x_max", "core_y_max")
        )
        with laspy.open(
            path, laz_backend=laspy.LazBackend.LazrsParallel
        ) as reader:
            header = reader.header
            dimensions = set(header.point_format.dimension_names)
            has_rgb = {"red", "green", "blue"}.issubset(dimensions)
            has_classes = "classification" in dimensions
            file_mins = np.asarray(header.mins, dtype=np.float64)
            file_maxs = np.asarray(header.maxs, dtype=np.float64)
            is_copc = any(
                str(getattr(vlr, "user_id", "")).strip("\x00").lower() == "copc"
                and getattr(vlr, "record_id", None) == 1
                for vlr in (*(header.vlrs or ()), *(header.evlrs or ()))
            )
            total = int(getattr(header, "point_count", 0) or 0)
            if is_copc:
                return self._load_copc_tile(
                    tile, tile_cap, origin, cancelled, header, file_mins,
                    file_maxs, has_rgb, has_classes, has_core,
                )
            return self._load_plain_tile(
                reader, tile, tile_cap, total, origin, cancelled,
                has_rgb, has_classes, has_core,
            )

    def _load_copc_tile(
        self, tile, tile_cap, origin, cancelled, header, file_mins,
        file_maxs, has_rgb, has_classes, has_core,
    ):
        """COPC resolution query (own reader); falls back to plain scan."""
        path = tile.get("file")
        seed = int.from_bytes(
            hashlib.sha1(path.encode("utf-8")).digest()[:8], "little"
        )
        rng = np.random.default_rng(seed)
        sample = None

        def consume(points) -> None:
            nonlocal sample
            if cancelled.is_set():
                return
            x = np.asarray(points.x)
            y = np.asarray(points.y)
            z = np.asarray(points.z)
            self._account(int(x.size))
            if has_core:
                mask = (
                    (x >= tile["core_x_min"])
                    & (x < tile["core_x_max"])
                    & (y >= tile["core_y_min"])
                    & (y < tile["core_y_max"])
                )
                x, y, z = x[mask], y[mask], z[mask]
            if not x.size:
                return
            positions = _localize_coordinates(x, y, z, origin)
            rgb = None
            if has_rgb:
                channels = [
                    np.asarray(getattr(points, channel), dtype=np.float32)
                    for channel in ("red", "green", "blue")
                ]
                if has_core:
                    channels = [channel[mask] for channel in channels]
                scale = (
                    255.0
                    if max(float(channel.max()) for channel in channels) <= 255
                    else 65535.0
                )
                rgb = np.column_stack(channels).astype(np.float32, copy=False) / scale
            classes = (
                np.asarray(points.classification, dtype=np.uint8)
                if has_classes
                else None
            )
            if has_core and classes is not None:
                classes = classes[mask]
            scores = rng.random(positions.shape[0])
            sample = _bounded_sample_merge(
                sample, scores, positions, rgb, classes, tile_cap
            )

        queried = False
        try:
            from laspy import copc

            query_mins = file_mins.copy()
            query_maxs = file_maxs.copy()
            if has_core:
                query_mins[:2] = (tile["core_x_min"], tile["core_y_min"])
                query_maxs[:2] = (tile["core_x_max"], tile["core_y_max"])
            extent = np.maximum(query_maxs - query_mins, 1e-6)
            resolution = max(
                float(np.cbrt(np.prod(extent) / tile_cap)), 1e-3
            )
            copc_reader = getattr(copc, "CopcReader", None)
            open_reader = getattr(copc_reader, "open", None)
            bounds_type = getattr(copc, "Bounds", None)
            if callable(open_reader) and bounds_type is not None:
                with open_reader(path) as copc_reader:
                    query = getattr(copc_reader, "query", None)
                    if callable(query):
                        points = query(
                            bounds=bounds_type(query_mins, query_maxs),
                            resolution=resolution,
                        )
                        consume(points)
                        queried = True
        except MemoryError:
            raise
        except Exception:
            queried = False

        if not queried:
            with laspy.open(
                path, laz_backend=laspy.LazBackend.LazrsParallel
            ) as reader:
                total = int(getattr(reader.header, "point_count", 0) or 0)
                self._consume_chunks(
                    reader, tile, tile_cap, total,
                    origin, cancelled, has_rgb, has_classes, has_core,
                    consume,
                )
        if sample is None:
            return None, None, None, has_rgb, has_classes
        _, positions, rgb, classes = sample
        return positions, rgb, classes, has_rgb, has_classes

    def _load_plain_tile(
        self, reader, tile, tile_cap, total, origin, cancelled,
        has_rgb, has_classes, has_core,
    ):
        """Single-open chunk scan with a uniform reservoir past the cap."""
        seed = int.from_bytes(
            hashlib.sha1(tile.get("file", "").encode("utf-8")).digest()[:8],
            "little",
        )
        rng = np.random.default_rng(seed)
        sample = None
        # Fast path: the header count fits the cap, so every point is kept
        # and the reservoir (scores + repeated argpartition/concatenates,
        # quadratic over chunks) is pure overhead.
        exact = 0 < total <= tile_cap
        pos_list: list[np.ndarray] = []
        rgb_list: list[np.ndarray] = []
        cls_list: list[np.ndarray] = []

        def consume(points) -> None:
            nonlocal sample
            if cancelled.is_set():
                return
            x = np.asarray(points.x)
            y = np.asarray(points.y)
            z = np.asarray(points.z)
            self._account(int(x.size))
            if has_core:
                mask = (
                    (x >= tile["core_x_min"])
                    & (x < tile["core_x_max"])
                    & (y >= tile["core_y_min"])
                    & (y < tile["core_y_max"])
                )
                x, y, z = x[mask], y[mask], z[mask]
            if not x.size:
                return
            positions = _localize_coordinates(x, y, z, origin)
            rgb = None
            if has_rgb:
                channels = [
                    np.asarray(getattr(points, channel), dtype=np.float32)
                    for channel in ("red", "green", "blue")
                ]
                if has_core:
                    channels = [channel[mask] for channel in channels]
                scale = (
                    255.0
                    if max(float(channel.max()) for channel in channels) <= 255
                    else 65535.0
                )
                rgb = np.column_stack(channels).astype(np.float32, copy=False) / scale
            classes = (
                np.asarray(points.classification, dtype=np.uint8)
                if has_classes
                else None
            )
            if has_core and classes is not None:
                classes = classes[mask]
            if exact:
                pos_list.append(positions)
                if rgb is not None:
                    rgb_list.append(rgb)
                if classes is not None:
                    cls_list.append(classes)
                return
            scores = rng.random(positions.shape[0])
            sample = _bounded_sample_merge(
                sample, scores, positions, rgb, classes, tile_cap
            )

        self._consume_chunks(
            reader, tile, tile_cap, total, origin, cancelled, has_rgb,
            has_classes, has_core, consume,
        )
        if exact:
            if not pos_list:
                return None, None, None, has_rgb, has_classes
            return (
                np.concatenate(pos_list),
                np.concatenate(rgb_list) if has_rgb else None,
                np.concatenate(cls_list) if has_classes else None,
                has_rgb,
                has_classes,
            )
        if sample is None:
            return None, None, None, has_rgb, has_classes
        _, positions, rgb, classes = sample
        return positions, rgb, classes, has_rgb, has_classes

    @staticmethod
    def _consume_chunks(reader, tile, tile_cap, total, origin, cancelled,
                        has_rgb, has_classes, has_core, consume) -> None:
        """Plain sequential scan (single open, shared by all paths)."""
        for chunk in reader.chunk_iterator(VIEWER_CHUNK):
            if cancelled.is_set():
                return
            consume(chunk)

# ---------------------------------------------------------------------------
# OpenGL view
# ---------------------------------------------------------------------------
def _interleaved_points(
    positions: np.ndarray,
    rgb: np.ndarray | None,
    classes: np.ndarray | None,
    color_mode: str,
) -> np.ndarray:
    """Pack positions + colors for one VBO upload (single allocation).

    Every color source is float32, so filling the halves in place is
    value-identical to the old hstack-then-astype while skipping a full
    extra buffer and pass (2.4 GB at 100M points).
    """
    count = positions.shape[0]
    out = np.empty((count, 6), dtype=np.float32)
    out[:, :3] = positions
    if color_mode == "rgb" and rgb is not None:
        out[:, 3:] = rgb
    elif color_mode == "classification" and classes is not None:
        out[:, 3:] = classification_colors(classes)
    elif color_mode == "elevation":
        out[:, 3:] = elevation_colors(positions[:, 2])
    else:
        out[:, 3:] = np.asarray(SINGLE_COLOR, dtype=np.float32)
    return out


def _shuffled_order(count: int) -> np.ndarray:
    """Deterministic draw order for density subsampling (built lazily).

    The eager arange+shuffle cost ~400 MB + seconds at 100M points, yet
    density 100 (the default) never reads it.
    """
    order = np.arange(count, dtype=np.int32)
    np.random.default_rng(0).shuffle(order)
    return order


class _GLView(QOpenGLWidget):
    upload_requested = Signal()
    points_uploaded = Signal(int)
    upload_failed = Signal(str)
    # Floats, not ints: byte counts past 2 GiB overflow shiboken's C int
    # conversion at emit time (exact up to 2**53, plenty for buffers).
    upload_progress = Signal(float, float)
    render_failed = Signal(str)
    upload_chunk_bytes = 4 * 1024 * 1024
    # One QOpenGLBuffer.allocate() takes a C int: anything past 2 GiB
    # raises OverflowError in the binding, and several integrated drivers
    # further cap single allocations around 256-512 MB. Point data is
    # striped over several buffers of at most this many points
    # (10M x 24 B = 240 MB), so dense clouds upload anywhere.
    vbo_split_points = 10_000_000

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumSize(200, 200)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        fmt = QSurfaceFormat()
        fmt.setDepthBufferSize(24)
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        self.setFormat(fmt)

        # Orbital camera state.
        self._yaw = -45.0
        self._pitch = 30.0
        self._distance = 10.0
        self._target = QVector3D(0.0, 0.0, 0.0)
        self._fov = 45.0

        # data
        self._source_positions: np.ndarray | None = None
        self._source_rgb: np.ndarray | None = None
        self._source_classes: np.ndarray | None = None
        self._positions: np.ndarray | None = None
        self._rgb: np.ndarray | None = None
        self._classes: np.ndarray | None = None
        self._color_mode = "elevation"  # rgb | classification | elevation | single
        self._density = 100
        self._point_size = 3.0
        self._display_order: np.ndarray | None = None
        self._display_count = 0
        self._upload_timer = QTimer(self)
        self._upload_timer.setInterval(1)
        self._upload_timer.timeout.connect(self._upload_next_points_chunk)
        self._upload_total_bytes = 0
        self._upload_done_bytes = 0

        # GL resources
        self._points_program: QOpenGLShaderProgram | None = None
        self._lines_program: QOpenGLShaderProgram | None = None
        self._loc_mvp_p = -1
        self._loc_size_p = -1
        self._loc_mvp_l = -1
        self._lines_vbo: QOpenGLBuffer | None = None
        self._vao: QOpenGLVertexArrayObject | None = None
        # Displayed point stripes: [(buffer, row count)], each buffer
        # under 2 GiB (see vbo_split_points). In-flight upload pieces live
        # in _upload_*; retired buffers wait here for a current context.
        self._points_vbos: list = []
        self._upload_pieces: list[np.ndarray] = []
        self._upload_piece = 0
        self._upload_written = 0
        self._upload_vbo: QOpenGLBuffer | None = None
        self._staged_vbos: list = []
        self._retired_vbos: list[QOpenGLBuffer] = []
        self._upload_count = 0
        # First render/upload failure latches here: paintGL returns early
        # afterwards instead of raising once per frame (PySide aborts on
        # repeated exceptions inside paint/event overrides).
        self._render_broken: str | None = None
        self._points_count = 0
        self._lines_count = 0

        # Interaction state.
        self._last_mouse = QPoint()
        self._drag_mode = None  # None | "orbit" | "pan"

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def set_points(
        self,
        positions: np.ndarray,
        rgb: np.ndarray | None,
        classes: np.ndarray | None = None,
        display_order: np.ndarray | None = None,
        color_mode: str | None = None,
    ) -> None:
        self._source_positions = positions
        self._source_rgb = rgb
        self._source_classes = classes
        self._density = 100
        self._display_order = display_order
        if color_mode is not None:
            self._color_mode = color_mode
        self._refresh_display(reset_view=True)
        self.upload_requested.emit()

    def set_color_mode(self, mode: str) -> None:
        if mode == "rgb" and self._source_rgb is None:
            mode = "elevation"
        if mode == "classification" and self._source_classes is None:
            mode = "elevation"
        self._color_mode = mode
        self.upload_requested.emit()

    def set_density(self, percent: int) -> None:
        self._density = max(10, min(100, int(percent)))
        self._refresh_display(reset_view=False)

    def set_point_size(self, size: int) -> None:
        self._point_size = max(1.0, min(12.0, float(size)))
        self.update()

    def _refresh_display(self, reset_view: bool) -> None:
        if self._source_positions is None:
            return
        source_count = self._source_positions.shape[0]
        self._display_count = max(1, int(source_count * self._density / 100))
        self._positions = self._source_positions
        self._rgb = self._source_rgb
        self._classes = self._source_classes
        if reset_view:
            self._fit_to_data()
        if self._points_program is not None:
            self.upload_requested.emit()
            if reset_view:
                self._upload_axes()
        self.update()

    def point_count(self) -> int:
        return self._display_count

    def color_mode(self) -> str:
        return self._color_mode

    def density(self) -> int:
        return self._density

    # ------------------------------------------------------------------
    # OpenGL
    # ------------------------------------------------------------------

    def initializeGL(self) -> None:
        self._vao = QOpenGLVertexArrayObject(self)
        self._vao.create()
        self._vao.bind()

        # Nearest-point-wins: without the depth test the paint order decides
        # (the reference grid landed on top of the cloud and read as seams,
        # and far points overwrote near ones).
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)

        self._points_program = self._make_program(POINT_VERTEX)
        self._lines_program = self._make_program(LINE_VERTEX)

        # uniform locations (PySide6 resolves float uniforms via int location)
        self._loc_mvp_p = self._points_program.uniformLocation("uMVP")
        self._loc_size_p = self._points_program.uniformLocation("uPointSize")
        self._loc_mvp_l = self._lines_program.uniformLocation("uMVP")
        self._check_programs()
        try:
            note_gpu_renderer(glGetString(GL_VENDOR),
                              glGetString(GL_RENDERER))
        except Exception:
            pass

        self._lines_vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self._lines_vbo.create()

        # A recreated context invalidates every buffer handle: drop them
        # (numpy pieces survive, so a pending upload simply restarts).
        self._points_vbos = []
        self._upload_vbo = None
        self._staged_vbos = []
        self._retired_vbos = []
        self._points_count = 0
        if self._upload_pieces:
            self._upload_piece = 0
            self._upload_written = 0

        self._upload_axes()
        if self._upload_pieces:
            self._upload_timer.start()

    def _check_programs(self) -> None:
        """Fail loud once when the driver rejects our shaders.

        A dead program yields attribute location -1, and every later
        setAttributeBuffer then poisons the GL error flag — surfacing as
        an inscrutable error at the next checked call, once per frame.
        """
        problems = []
        for program in (self._points_program, self._lines_program):
            if program is None or not program.isLinked():
                problems.append(
                    str(program.log()) if program is not None else "null"
                )
                continue
            for name in ("aPos", "aColor"):
                if program.attributeLocation(name) < 0:
                    problems.append(f"missing attribute {name}")
        if problems:
            self._fail_render_once(
                "GPU shaders rejected by the driver: "
                + " / ".join(problems)
            )

    def _make_program(self, vertex: str) -> QOpenGLShaderProgram:
        program = QOpenGLShaderProgram(self)
        if not program.addShaderFromSourceCode(
            QOpenGLShader.ShaderTypeBit.Vertex, vertex
        ):
            print("GL VERTEX ERROR:", program.log(), file=sys.stderr)
        if not program.addShaderFromSourceCode(
            QOpenGLShader.ShaderTypeBit.Fragment, COLOR_FRAGMENT
        ):
            print("GL FRAGMENT ERROR:", program.log(), file=sys.stderr)
        if not program.link():
            print("GL LINK ERROR:", program.log(), file=sys.stderr)
        return program

    @staticmethod
    def _split_pieces(data: np.ndarray, max_points: int) -> list[np.ndarray]:
        """Row views over point data, each under the VBO byte ceiling."""
        total = data.shape[0]
        step = max(1, int(max_points))
        return [data[start:start + step] for start in range(0, total, step)]

    def queue_points_upload(self, data: np.ndarray, point_count: int) -> None:
        self._retire_point_buffers()
        self._upload_pieces = self._split_pieces(data, self.vbo_split_points)
        self._upload_piece = 0
        self._upload_written = 0
        self._upload_vbo = None
        self._staged_vbos = []
        self._upload_count = point_count
        self._upload_total_bytes = int(data.nbytes)
        self._upload_done_bytes = 0
        self._points_count = 0
        if self.isValid():
            self._upload_timer.start()
        self.update()

    def cancel_pending_points_upload(self) -> None:
        self._upload_pieces = []
        self._upload_piece = 0
        self._upload_written = 0
        self._upload_done_bytes = 0
        self._upload_vbo = None
        self._staged_vbos = []
        self._points_count = 0

    def _retire_point_buffers(self) -> None:
        """Park displayed/in-flight buffers for destruction on a live context.

        queue() runs on the GUI thread without a current context, so GL
        handles are only destroyed inside the upload tick (or initializeGL).
        """
        self._retired_vbos.extend(
            vbo for vbo, _count in self._points_vbos)
        self._points_vbos = []
        if self._upload_vbo is not None:
            self._retired_vbos.append(self._upload_vbo)
            self._upload_vbo = None
        for vbo, _count in self._staged_vbos:
            self._retired_vbos.append(vbo)
        self._staged_vbos = []

    def _destroy_retired(self) -> None:
        while self._retired_vbos:
            vbo = self._retired_vbos.pop()
            try:
                vbo.destroy()
            except Exception:
                pass

    def _upload_pending_points(self) -> bool:
        if self._upload_piece >= len(self._upload_pieces):
            return False
        data = self._upload_pieces[self._upload_piece]
        if self._upload_vbo is None:
            vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
            vbo.create()
            vbo.bind()
            # Piece bytes stay under 2 GiB by construction: allocate() and
            # write() take C ints and overflow past INT_MAX.
            vbo.allocate(int(data.nbytes))
            vbo.release()
            self._upload_vbo = vbo
            self._upload_written = 0

        bytes_per_point = int(data.strides[0])
        points_per_chunk = max(1, self.upload_chunk_bytes // bytes_per_point)
        start = self._upload_written // bytes_per_point
        end = min(data.shape[0], start + points_per_chunk)
        chunk = data[start:end].tobytes()

        vbo = self._upload_vbo
        assert vbo is not None
        vbo.bind()
        vbo.write(int(start * bytes_per_point), chunk, len(chunk))
        vbo.release()
        self._upload_written = end * bytes_per_point

        if end >= data.shape[0]:
            self._staged_vbos.append((vbo, int(data.shape[0])))
            self._upload_done_bytes += int(data.nbytes)
            self._upload_vbo = None
            self._upload_piece += 1

        if self._upload_piece >= len(self._upload_pieces):
            self._points_vbos = self._staged_vbos
            self._staged_vbos = []
            self._upload_pieces = []
            self._points_count = self._upload_count
            self.points_uploaded.emit(self._points_count)
            return False

        return True

    def _upload_next_points_chunk(self) -> None:
        if not self._upload_pieces:
            self._upload_timer.stop()
            return
        if not self.isValid():
            return
        self.makeCurrent()
        try:
            self._destroy_retired()
            try:
                self._upload_pending_points()
            except Exception as exc:
                # Fail loud once (stale timer retries spammed the console
                # with one traceback per tick).
                self._upload_timer.stop()
                self._upload_pieces = []
                self._upload_piece = 0
                self._upload_written = 0
                self._upload_vbo = None
                self._staged_vbos = []
                self.upload_failed.emit(str(exc))
            else:
                if not self._upload_pieces:
                    # Just finished: points_uploaded is already queued with
                    # the final count, so no trailing progress emission.
                    pass
                else:
                    done = self._upload_done_bytes + self._upload_written
                    try:
                        self.upload_progress.emit(done, self._upload_total_bytes)
                    except Exception:
                        pass
        finally:
            self.doneCurrent()
        self.update()
        if not self._upload_pieces:
            self._upload_timer.stop()

    def _upload_axes(self) -> None:
        if self._lines_vbo is None or self._positions is None:
            return
        positions = self._positions
        min_pt = positions.min(axis=0)
        max_pt = positions.max(axis=0)
        center = (min_pt + max_pt) / 2.0
        radius = float(np.max(max_pt - min_pt)) or 1.0
        length = radius * 0.8
        origin = min_pt
        axis_colors = {
            "x": (0.80, 0.25, 0.25),
            "y": (0.25, 0.75, 0.30),
            "z": (0.25, 0.45, 0.90),
        }
        lines: list[np.ndarray] = []
        for axis in ("x", "y", "z"):
            end = origin.copy()
            end[{"x": 0, "y": 1, "z": 2}[axis]] += length
            lines.append(
                np.array(
                    [origin, end], dtype=np.float32
                )
            )
            # Vertex colors are assigned when the line buffer is interleaved.
        grid_min = min_pt + (max_pt - min_pt) * 0.15
        grid_max = max_pt - (max_pt - min_pt) * 0.15
        # Half a metre under the lowest point: with the depth test the grid is
        # occluded by the cloud anyway, and the drop keeps coplanar points
        # from z-fighting the floor lines.
        z_floor = float(min_pt[2]) - max(
            0.5, 0.02 * float(max_pt[2] - min_pt[2])
        )
        steps = 6
        for i in range(steps + 1):
            t = i / steps
            x = grid_min[0] + (grid_max[0] - grid_min[0]) * t
            y = grid_min[1] + (grid_max[1] - grid_min[1]) * t
            lines.append(
                np.array(
                    [[x, grid_min[1], z_floor], [x, grid_max[1], z_floor]],
                    dtype=np.float32,
                )
            )
            lines.append(
                np.array(
                    [[grid_min[0], y, z_floor], [grid_max[0], y, z_floor]],
                    dtype=np.float32,
                )
            )

        positions_all = np.concatenate(lines)
        colors_all = np.full((positions_all.shape[0], 3), GRID_COLOR, dtype=np.float32)
        # Axis colors occupy the first three lines (two vertices each).
        for i, axis in enumerate(("x", "y", "z")):
            colors_all[i * 2] = axis_colors[axis]
            colors_all[i * 2 + 1] = axis_colors[axis]
        interleaved = np.hstack([positions_all, colors_all]).astype(np.float32)
        self._lines_count = interleaved.shape[0]
        self._lines_vbo.bind()
        self._lines_vbo.allocate(interleaved.tobytes(), interleaved.nbytes)
        self._lines_vbo.release()

        self._grid_center = center
        self._grid_radius = radius

    def _bind_attributes(self, program: QOpenGLShaderProgram, vbo: QOpenGLBuffer) -> None:
        stride = 6 * 4
        pos_loc = program.attributeLocation("aPos")
        color_loc = program.attributeLocation("aColor")
        vbo.bind()
        program.enableAttributeArray(pos_loc)
        program.setAttributeBuffer(pos_loc, GL_FLOAT, 0, 3, stride)
        program.enableAttributeArray(color_loc)
        program.setAttributeBuffer(color_loc, GL_FLOAT, 12, 3, stride)
        vbo.release()

    def _fail_render_once(self, message: str) -> None:
        """Latch the first render pathology: one signal, then silence."""
        if self._render_broken is None:
            self._render_broken = str(message)
            try:
                self.render_failed.emit(self._render_broken)
            except Exception:
                pass

    def paintGL(self) -> None:
        if self._render_broken is not None:
            return
        try:
            self._paintGL_guarded()
        except Exception as exc:
            self._fail_render_once(f"{exc}")

    def _paintGL_guarded(self) -> None:
        import time

        started = time.perf_counter()
        # Drain pre-existing flags: Qt/driver calls outside this frame may
        # leave a stale error behind, and blaming it on the frame would
        # latch a healthy viewer on the very first paint.
        for _ in range(8):
            try:
                if not glGetError():
                    break
            except Exception:
                break
        if self._vao is not None:
            self._vao.bind()
        try:
            glClearColor(*BACKGROUND, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            mvp = self._mvp()

            if (
                self._points_vbos
                and self._points_program is not None
            ):
                self._points_program.bind()
                self._points_program.setUniformValue(self._loc_mvp_p, mvp)
                glUniform1f(self._loc_size_p, float(self._point_size))
                glPointSize(float(self._point_size))
                for vbo, count in self._points_vbos:
                    if vbo is None or not vbo.isCreated() or count <= 0:
                        continue
                    self._bind_attributes(self._points_program, vbo)
                    glDrawArrays(GL_POINTS, 0, int(count))

            if self._lines_count and self._lines_program is not None:
                self._lines_program.bind()
                self._lines_program.setUniformValue(self._loc_mvp_l, mvp)
                self._bind_attributes(self._lines_program, self._lines_vbo)
                glDrawArrays(GL_LINES, 0, self._lines_count)
            pending = glGetError()
            if pending:
                raise RuntimeError(f"GL error {pending:#x} after frame")
        finally:
            if self._vao is not None:
                self._vao.release()
            host = self.parent()
            sample = getattr(host, "_on_frame_sample", None)
            if callable(sample):
                try:
                    sample((time.perf_counter() - started) * 1000.0)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Camera.
    # ------------------------------------------------------------------

    def _mvp(self) -> QMatrix4x4:
        yaw = math.radians(self._yaw)
        pitch = math.radians(self._pitch)
        eye = self._target + QVector3D(
            math.cos(yaw) * math.cos(pitch) * self._distance,
            math.sin(yaw) * math.cos(pitch) * self._distance,
            math.sin(pitch) * self._distance,
        )
        view = QMatrix4x4()
        view.lookAt(eye, self._target, QVector3D(0.0, 0.0, 1.0))

        proj = QMatrix4x4()
        aspect = max(0.1, self.width() / max(1, self.height()))
        near = max(0.01, self._distance * 0.001)
        far = max(near * 2, self._distance * 100.0)
        proj.perspective(self._fov, aspect, near, far)
        return proj * view

    def _fit_to_data(self) -> None:
        if self._positions is None:
            return
        min_pt = self._positions.min(axis=0)
        max_pt = self._positions.max(axis=0)
        center = (min_pt + max_pt) / 2.0
        radius = float(np.max(max_pt - min_pt)) or 1.0
        self._target = QVector3D(float(center[0]), float(center[1]), float(center[2]))
        self._distance = radius / math.tan(math.radians(self._fov) / 2.0) * 1.3
        self._yaw = -45.0
        self._pitch = 30.0
        if self._lines_vbo is not None:
            self._upload_axes()
        self.update()

    def reset_view(self) -> None:
        self._fit_to_data()

    # ------------------------------------------------------------------
    # Interaction.
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        self._last_mouse = event.position().toPoint()
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_mode = "orbit"
        elif event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton):
            self._drag_mode = "pan"
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        pos = event.position().toPoint()
        delta = pos - self._last_mouse
        self._last_mouse = pos
        if self._drag_mode == "orbit":
            self._yaw -= delta.x() * 0.3
            self._pitch = max(-89.0, min(89.0, self._pitch - delta.y() * 0.3))
            self.update()
        elif self._drag_mode == "pan":
            yaw = math.radians(self._yaw)
            pitch = math.radians(self._pitch)
            forward = QVector3D(
                math.cos(yaw) * math.cos(pitch),
                math.sin(yaw) * math.cos(pitch),
                math.sin(pitch),
            )
            right = QVector3D.crossProduct(forward, QVector3D(0, 0, 1)).normalized()
            up = QVector3D.crossProduct(right, forward).normalized()
            factor = self._distance * 0.002
            self._target += right * (-delta.x() * factor) + up * (delta.y() * factor)
            self.update()
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        self._drag_mode = None
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:
        self._fit_to_data()
        event.accept()

    def wheelEvent(self, event) -> None:
        steps = event.angleDelta().y() / 120.0
        self._distance = max(0.01, self._distance * (0.85 ** steps))
        self.update()
        event.accept()

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key.Key_C:
            order = ["rgb", "classification", "elevation", "single"]
            if self._rgb is None and self._color_mode == "rgb":
                self._color_mode = "elevation"
            elif self._classes is None and self._color_mode == "classification":
                self._color_mode = "elevation"
            else:
                index = order.index(self._color_mode)
                self._color_mode = order[(index + 1) % len(order)]
            self.upload_requested.emit()
        elif key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self._point_size = min(30.0, self._point_size + 1.0)
            self.update()
        elif key == Qt.Key.Key_Minus:
            self._point_size = max(1.0, self._point_size - 1.0)
            self.update()
        elif key == Qt.Key.Key_F:
            self._fit_to_data()
        elif key == Qt.Key.Key_R:
            self.reset_view()
        else:
            super().keyPressEvent(event)


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------
class PointCloud3DViewer(BaseViewer):
    """3D point-cloud preview with background loading and orbital navigation."""

    load_progress = Signal(float, float)

    def __init__(self, kind: str, parent: QWidget | None = None):
        super().__init__(kind, parent)
        self.load_progress.connect(self._on_load_progress)
        control_bar = QWidget(self)
        control_bar.setObjectName("pointCloudControls")
        control_bar.setMaximumHeight(32)
        controls = QHBoxLayout(control_bar)
        controls.setContentsMargins(6, 2, 6, 2)
        controls.setSpacing(4)
        color_label = QLabel(t("Color"))
        color_label.setProperty("origText_en", "Color")
        controls.addWidget(color_label)
        self._color_combo = QComboBox(self)
        for index, (label, mode) in enumerate(
            (
                ("RGB", "rgb"),
                ("Classification", "classification"),
                ("Elevation", "elevation"),
                ("Single color", "single"),
            )
        ):
            self._color_combo.addItem(t(label), mode)
            self._color_combo.setProperty(f"origItem{index}_en", label)
        self._color_combo.setFixedWidth(112)
        self._color_combo.setMaximumHeight(24)
        self._color_combo.setProperty(
            "origTip_en",
            "Color points by RGB, LAS class, elevation, or one color",
        )
        self._color_combo.setToolTip(
            t("Color points by RGB, LAS class, elevation, or one color")
        )
        self._color_combo.currentIndexChanged.connect(self._on_color_changed)
        controls.addWidget(self._color_combo)

        density_label = QLabel(t("Density"))
        density_label.setProperty("origText_en", "Density")
        controls.addWidget(density_label)
        self._density_slider = QSlider(Qt.Orientation.Horizontal, self)
        self._density_slider.setRange(10, 100)
        self._density_slider.setValue(100)
        # Re-uploading a multi-million-point VBO is expensive. Apply density
        # once the handle is released instead of once per mouse pixel.
        self._density_slider.setTracking(False)
        self._density_slider.setFixedWidth(82)
        self._density_slider.setMaximumHeight(20)
        self._density_slider.setProperty(
            "origTip_en", "Percentage of the loaded points drawn"
        )
        self._density_slider.setToolTip(
            t("Percentage of the loaded points drawn")
        )
        self._density_slider.valueChanged.connect(self._on_density_changed)
        controls.addWidget(self._density_slider)
        self._density_label = QLabel("100%")
        controls.addWidget(self._density_label)

        size_label = QLabel(t("Size"))
        size_label.setProperty("origText_en", "Size")
        controls.addWidget(size_label)
        self._size_slider = QSlider(Qt.Orientation.Horizontal, self)
        self._size_slider.setRange(1, 24)
        self._size_slider.setValue(3)
        self._size_slider.setFixedWidth(64)
        self._size_slider.setMaximumHeight(20)
        self._size_slider.setProperty("origTip_en", "OpenGL point size")
        self._size_slider.setToolTip(t("OpenGL point size"))
        self._size_slider.valueChanged.connect(self._on_size_changed)
        controls.addWidget(self._size_slider)
        self._loaded_label = QLabel("-- pts")
        self._loaded_label.setMinimumWidth(72)
        controls.addWidget(self._loaded_label)
        self.install_help_button(controls)

        self._layout.insertWidget(0, control_bar)
        self._gl = _GLView(self)
        self._gl.upload_requested.connect(self._request_gl_upload)
        self._gl.points_uploaded.connect(
            self._on_gl_uploaded, Qt.ConnectionType.QueuedConnection
        )
        self._gl.upload_failed.connect(
            self._on_gl_upload_failed, Qt.ConnectionType.QueuedConnection
        )
        self._gl.upload_progress.connect(
            self._on_gl_upload_progress, Qt.ConnectionType.QueuedConnection
        )
        self._frame_sample_at = 0.0
        self._gl.render_failed.connect(
            self._on_gl_render_failed, Qt.ConnectionType.QueuedConnection
        )
        self._layout.addWidget(self._gl)
        self._gl.hide()

        # Floating classification legend over the viewport (top-right):
        # an inline bar with up to 19 swatch+label pairs eats half the
        # viewer, so the legend floats instead and only shows in
        # classification mode.
        self._class_legend = QWidget(self._gl)
        self._class_legend.setObjectName("pointCloudLegend")
        self._legend_rows = QVBoxLayout(self._class_legend)
        self._legend_rows.setContentsMargins(6, 6, 6, 6)
        self._legend_rows.setSpacing(2)
        self._class_legend.hide()
        self._legend_key = None

    def help_text(self) -> str | None:
        return (
            "<b>Navigate</b><br>Drag: orbit · Wheel: zoom · "
            "Double-click or F: fit · R: reset view<br><b>Color</b><br>"
            "C cycles RGB / Classification / Elevation / Single color"
            "<br><b>Size</b><br>+ / -: point size · Density: share of "
            "loaded points drawn"
        )

    def set_payload(self, payload) -> None:
        tiles = payload.get("tiles") if isinstance(payload, dict) else None
        if not tiles and isinstance(payload, dict) and payload.get("file"):
            tiles = [{"file": payload["file"]}]
        if not tiles:
            self.show_placeholder(t("No data available"))
            self._gl.hide()
            self._class_legend.hide()
            return

        self._gl.hide()
        loader = _PointCloudLoader(tiles, on_progress=self.load_progress.emit)
        self.start_async_load(loader.run, self._on_loaded, self._on_failed)

    def _on_load_progress(self, done: float, total: float) -> None:
        self._loaded_label.setText(f"{int(done):,} / {int(total):,} pts")

    def _on_gl_upload_progress(self, done: int, total: int) -> None:
        self._loaded_label.setText(
            f"↑ {done / 1048576:.0f} / {total / 1048576:.0f} MB")

    def _on_frame_sample(self, ms: float) -> None:
        import time

        now = time.monotonic()
        if now - self._frame_sample_at < 0.5:
            return
        self._frame_sample_at = now
        text = self._loaded_label.text()
        base = text.split(" · ")[0]
        self._loaded_label.setText(f"{base} · {ms:.0f} ms/f")

    def _on_color_changed(self, _index: int) -> None:
        self._gl.set_color_mode(self._color_combo.currentData())
        self._refresh_class_legend()

    def _refresh_class_legend(self) -> None:
        """Swatches for the ASPRS classes present (classification mode).

        Rebuilt only when the present-class set or the color mode changes;
        hidden outside classification mode so RGB/elevation stay clean.
        """
        classes = getattr(self._gl, "_source_classes", None)
        mode = self._color_combo.currentData()
        entries = legend_entries(classes)
        key = (mode, tuple(code for code, _name, _rgb in entries))
        if key == self._legend_key:
            self._place_legend()
            self._class_legend.setVisible(
                bool(entries) and mode == "classification")
            return
        self._legend_key = key
        layout = self._legend_rows
        while layout.count():
            child = layout.takeAt(0)
            row = child.widget()
            row_layout = child.layout()
            if row_layout is not None:
                while row_layout.count():
                    sub = row_layout.takeAt(0).widget()
                    if sub is not None:
                        sub.deleteLater()
            if row is not None:
                row.deleteLater()
        if entries and mode == "classification":
            for _code, name, rgb in entries:
                row = QWidget(self._class_legend)
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(4)
                swatch = QLabel()
                pixmap = QPixmap(14, 14)
                pixmap.fill(QColor(int(rgb[0] * 255), int(rgb[1] * 255),
                                   int(rgb[2] * 255)))
                swatch.setPixmap(pixmap)
                swatch.setFixedSize(16, 16)
                row_layout.addWidget(swatch)
                label = QLabel(t(name))
                label.setObjectName("viewerLegendText")
                row_layout.addWidget(label)
                layout.addWidget(row)
            self._class_legend.show()
            self._place_legend()
        else:
            self._class_legend.hide()

    def _place_legend(self) -> None:
        """Pin the floating legend top-right of the viewport."""
        legend = self._class_legend
        if not legend.isVisible():
            return
        legend.setFixedSize(legend.sizeHint())
        legend.move(_legend_position(
            self._gl.width(), self._gl.height(),
            legend.width(), legend.height(),
        ))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._place_legend()

    def _on_density_changed(self, value: int) -> None:
        self._density_label.setText(f"{value}%")
        self._gl.set_density(value)

    def _on_size_changed(self, value: int) -> None:
        self._gl.set_point_size(value)

    def _on_loaded(self, data) -> None:
        if data is None:
            self.show_placeholder(t("No data available"))
            return
        positions, rgb, classes, display_order = data
        rgb_index = self._color_combo.findData("rgb")
        class_index = self._color_combo.findData("classification")
        self._color_combo.model().item(rgb_index).setEnabled(rgb is not None)
        self._color_combo.model().item(class_index).setEnabled(classes is not None)
        if classes is not None:
            mode, color_index = "classification", class_index
        elif rgb is not None:
            mode, color_index = "rgb", rgb_index
        else:
            mode, color_index = "elevation", self._color_combo.findData("elevation")
        self._color_combo.blockSignals(True)
        self._color_combo.setCurrentIndex(color_index)
        self._color_combo.blockSignals(False)
        self._placeholder.hide()
        self._progress.show()
        # The GL viewport shows as soon as data is decoded (never alongside
        # the placeholder: both expand, and that split the window half/half).
        # Showing it also creates the GL context, which the upload timer
        # needs: a hidden QOpenGLWidget never initializes and the points
        # would wait forever with "Loading..." on screen.
        self._gl.show()
        self._gl.set_point_size(self._size_slider.value())
        self._gl.set_points(positions, rgb, classes, display_order, mode)
        self._loaded_label.setText(f"{self._gl.point_count():,} pts")
        self._refresh_class_legend()

    def _request_gl_upload(self) -> None:
        positions = self._gl._source_positions
        if positions is None:
            return
        rgb = self._gl._source_rgb
        classes = self._gl._source_classes
        order = self._gl._display_order
        color_mode = self._gl.color_mode()
        count = max(1, int(positions.shape[0] * self._gl.density() / 100))
        if count < positions.shape[0] and (
            order is None or order.shape[0] != positions.shape[0]
        ):
            # Built on first subsampled draw only (density < 100).
            order = _shuffled_order(positions.shape[0])
            self._gl._display_order = order
        self._gl.cancel_pending_points_upload()

        def prepare(cancelled):
            if cancelled.is_set():
                return None
            indices = order[:count] if count < positions.shape[0] else None
            selected_positions = positions if indices is None else positions[indices]
            selected_rgb = rgb if indices is None or rgb is None else rgb[indices]
            selected_classes = (
                classes if indices is None or classes is None else classes[indices]
            )
            data = _interleaved_points(
                selected_positions, selected_rgb, selected_classes, color_mode
            )
            return data, count

        def prepared(result) -> None:
            if result is None:
                return
            data, point_count = result
            self._progress.show()
            self._gl.queue_points_upload(data, point_count)

        self.start_async_load(
            prepare,
            prepared,
            lambda exc: self.show_placeholder(
                t("Error preparing point-cloud preview: {exc}").format(
                    exc=exc)
            ),
            show_loading=False,
        )

    def _on_gl_uploaded(self, point_count: int) -> None:
        self._progress.hide()
        self._placeholder.hide()
        self._gl.show()
        self._loaded_label.setText(f"{point_count:,} pts")
        self.setWindowTitle(
            f"{viewer_title(self._kind)} ({point_count:,} pts)"
            + (f" — {self._product_title}" if self._product_title else "")
        )

    def _on_gl_upload_failed(self, message: str) -> None:
        self._progress.hide()
        self._gl.hide()
        self._class_legend.hide()
        self.show_placeholder(
            t("Error uploading points to the GPU: {exc}").format(exc=message)
        )

    def _on_gl_render_failed(self, message: str) -> None:
        self._progress.hide()
        self._gl.hide()
        self._class_legend.hide()
        self.show_placeholder(
            t("Error rendering points on the GPU: {exc}").format(exc=message)
        )

    def _on_failed(self, message: str) -> None:
        self._gl.hide()
        self._class_legend.hide()
        self.show_placeholder(message)

# Registration uses domain PortType values as the single data contract.
for _kind in ("point_cloud", "classified_point_cloud", "tiles"):
    register(_kind, PointCloud3DViewer)
