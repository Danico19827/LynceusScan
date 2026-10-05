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
import sys

import laspy
import numpy as np
from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_FLOAT,
    GL_LINES,
    GL_POINTS,
    glClear,
    glClearColor,
    glDrawArrays,
    glPointSize,
    glUniform1f,
)

from PySide6.QtCore import QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QMatrix4x4, QSurfaceFormat, QVector3D
from PySide6.QtOpenGL import (
    QOpenGLBuffer,
    QOpenGLShader,
    QOpenGLShaderProgram,
    QOpenGLVertexArrayObject,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QSlider, QWidget

from lynceus.plugins.locale import t
from lynceus.ui.viewers.base import BaseViewer, viewer_title
from lynceus.ui.viewers.palette import elevation_colors
from lynceus.ui.viewers.registry import register

# ---------------------------------------------------------------------------
# OpenGL constants (avoid relying on constant bindings)
# ---------------------------------------------------------------------------
DISPLAY_POINT_BUDGET = 1_500_000
MIN_DISPLAY_POINT_BUDGET = 750_000
MAX_DISPLAY_POINT_BUDGET = 12_000_000
POINT_MEMORY_ESTIMATE = 88  # retained arrays, upload preparation, and VBO allowance
VIEWER_CHUNK = 1_000_000

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
        adaptive = int((available * 0.10) / POINT_MEMORY_ESTIMATE)
        return max(MIN_DISPLAY_POINT_BUDGET, min(MAX_DISPLAY_POINT_BUDGET, adaptive))
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
    def __init__(self, tiles: list[dict]):
        self._tiles = tiles

    def run(self, cancelled):
        try:
            point_budget = safe_display_point_budget()
            tile_cap = max(1, point_budget // max(1, len(self._tiles)))
            normalized_tiles = [
                tile if isinstance(tile, dict) else {"file": tile}
                for tile in self._tiles
            ]
            bounds = []
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
            if not bounds:
                return None
            origin = (
                np.min([item[0] for item in bounds], axis=0)
                + np.max([item[1] for item in bounds], axis=0)
            ) / 2.0

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
            display_order = np.arange(positions.shape[0], dtype=np.int32)
            np.random.default_rng(0).shuffle(display_order)
            return positions, rgb, classes, display_order
        except MemoryError as exc:
            raise RuntimeError("Not enough memory to load the point cloud") from exc
        except Exception as exc:
            raise RuntimeError(f"Error loading point cloud: {exc}") from exc

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
        with laspy.open(path) as metadata_reader:
            header = metadata_reader.header
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
        if is_copc:
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
                    with open_reader(path) as reader:
                        query = getattr(reader, "query", None)
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
            with laspy.open(path) as reader:
                for chunk in reader.chunk_iterator(VIEWER_CHUNK):
                    if cancelled.is_set():
                        return None, None, None, has_rgb, has_classes
                    consume(chunk)

        if sample is None:
            return None, None, None, has_rgb, has_classes
        _, positions, rgb, classes = sample
        return (
            positions,
            rgb,
            classes,
            has_rgb,
            has_classes,
        )
# ---------------------------------------------------------------------------
# OpenGL view
# ---------------------------------------------------------------------------
def _interleaved_points(
    positions: np.ndarray,
    rgb: np.ndarray | None,
    classes: np.ndarray | None,
    color_mode: str,
) -> np.ndarray:
    count = positions.shape[0]
    if color_mode == "rgb" and rgb is not None:
        colors = rgb
    elif color_mode == "classification" and classes is not None:
        colors = classification_colors(classes)
    elif color_mode == "elevation":
        colors = elevation_colors(positions[:, 2])
    else:
        colors = np.tile(np.array(SINGLE_COLOR, dtype=np.float32), (count, 1))
    return np.hstack([positions, colors]).astype(np.float32, copy=False)


class _GLView(QOpenGLWidget):
    upload_requested = Signal()
    points_uploaded = Signal(int)
    upload_chunk_bytes = 4 * 1024 * 1024

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
        self._pending_points: np.ndarray | None = None
        self._pending_points_offset = 0
        self._pending_points_count = 0
        self._upload_timer = QTimer(self)
        self._upload_timer.setInterval(1)
        self._upload_timer.timeout.connect(self._upload_next_points_chunk)

        # GL resources
        self._points_program: QOpenGLShaderProgram | None = None
        self._lines_program: QOpenGLShaderProgram | None = None
        self._loc_mvp_p = -1
        self._loc_size_p = -1
        self._loc_mvp_l = -1
        self._points_vbo: QOpenGLBuffer | None = None
        self._lines_vbo: QOpenGLBuffer | None = None
        self._vao: QOpenGLVertexArrayObject | None = None
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
        self._display_order = display_order if display_order is not None else np.arange(
            positions.shape[0], dtype=np.int32
        )
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
        if self._points_vbo is not None:
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

        self._points_program = self._make_program(POINT_VERTEX)
        self._lines_program = self._make_program(LINE_VERTEX)

        # uniform locations (PySide6 resolves float uniforms via int location)
        self._loc_mvp_p = self._points_program.uniformLocation("uMVP")
        self._loc_size_p = self._points_program.uniformLocation("uPointSize")
        self._loc_mvp_l = self._lines_program.uniformLocation("uMVP")

        self._points_vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self._points_vbo.create()
        self._lines_vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self._lines_vbo.create()

        self._upload_axes()
        if self._pending_points is not None:
            self._upload_timer.start()

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

    def queue_points_upload(self, data: np.ndarray, point_count: int) -> None:
        self._pending_points = data
        self._pending_points_offset = 0
        self._pending_points_count = point_count
        self._points_count = 0
        if self._points_vbo is not None and self.isValid():
            self._upload_timer.start()
        self.update()

    def cancel_pending_points_upload(self) -> None:
        self._pending_points = None
        self._pending_points_offset = 0
        self._points_count = 0

    def _upload_pending_points(self) -> bool:
        data = self._pending_points
        if data is None or self._points_vbo is None:
            return data is not None

        bytes_per_point = int(data.strides[0])
        points_per_chunk = max(1, self.upload_chunk_bytes // bytes_per_point)
        start = self._pending_points_offset
        end = min(data.shape[0], start + points_per_chunk)
        chunk = data[start:end].tobytes()

        self._points_vbo.bind()
        if start == 0:
            self._points_vbo.allocate(data.nbytes)
        self._points_vbo.write(start * bytes_per_point, chunk, len(chunk))
        self._points_vbo.release()
        self._pending_points_offset = end

        if end >= data.shape[0]:
            self._points_count = self._pending_points_count
            self._pending_points = None
            self._pending_points_offset = 0
            self.points_uploaded.emit(self._points_count)
            return False

        return True

    def _upload_next_points_chunk(self) -> None:
        if self._pending_points is None:
            self._upload_timer.stop()
            return
        if self._points_vbo is None or not self.isValid():
            return
        self.makeCurrent()
        try:
            self._upload_pending_points()
        finally:
            self.doneCurrent()
        self.update()
        if self._pending_points is None:
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
        steps = 6
        for i in range(steps + 1):
            t = i / steps
            x = grid_min[0] + (grid_max[0] - grid_min[0]) * t
            y = grid_min[1] + (grid_max[1] - grid_min[1]) * t
            z_floor = float(min_pt[2])
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

    def paintGL(self) -> None:
        if self._vao is not None:
            self._vao.bind()
        try:
            glClearColor(*BACKGROUND, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            mvp = self._mvp()

            if (
                self._points_count
                and self._points_program is not None
            ):
                self._points_program.bind()
                self._points_program.setUniformValue(self._loc_mvp_p, mvp)
                glUniform1f(self._loc_size_p, float(self._point_size))
                glPointSize(float(self._point_size))
                self._bind_attributes(self._points_program, self._points_vbo)
                glDrawArrays(GL_POINTS, 0, self._points_count)

            if self._lines_count and self._lines_program is not None:
                self._lines_program.bind()
                self._lines_program.setUniformValue(self._loc_mvp_l, mvp)
                self._bind_attributes(self._lines_program, self._lines_vbo)
                glDrawArrays(GL_LINES, 0, self._lines_count)
        finally:
            if self._vao is not None:
                self._vao.release()

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

    def __init__(self, kind: str, parent: QWidget | None = None):
        super().__init__(kind, parent)
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

        self._layout.insertWidget(0, control_bar)
        self._gl = _GLView(self)
        self._gl.upload_requested.connect(self._request_gl_upload)
        self._gl.points_uploaded.connect(
            self._on_gl_uploaded, Qt.ConnectionType.QueuedConnection
        )
        self._layout.addWidget(self._gl)
        self._gl.hide()

    def set_payload(self, payload) -> None:
        tiles = payload.get("tiles") if isinstance(payload, dict) else None
        if not tiles and isinstance(payload, dict) and payload.get("file"):
            tiles = [{"file": payload["file"]}]
        if not tiles:
            self.show_placeholder(t("No data available"))
            self._gl.hide()
            return

        self._gl.hide()
        loader = _PointCloudLoader(tiles)
        self.start_async_load(loader.run, self._on_loaded, self._on_failed)

    def _on_color_changed(self, _index: int) -> None:
        self._gl.set_color_mode(self._color_combo.currentData())

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
        self._placeholder.setText(t("Loading..."))
        self._placeholder.show()
        self._progress.show()
        self._gl.show()
        self._gl.set_point_size(self._size_slider.value())
        self._gl.set_points(positions, rgb, classes, display_order, mode)
        self._loaded_label.setText(f"{self._gl.point_count():,} pts")

    def _request_gl_upload(self) -> None:
        positions = self._gl._source_positions
        if positions is None:
            return
        rgb = self._gl._source_rgb
        classes = self._gl._source_classes
        order = self._gl._display_order
        color_mode = self._gl.color_mode()
        count = max(1, int(positions.shape[0] * self._gl.density() / 100))
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
                f"Error preparing point-cloud preview: {exc}"
            ),
            show_loading=False,
        )

    def _on_gl_uploaded(self, point_count: int) -> None:
        self._progress.hide()
        self._placeholder.hide()
        self._loaded_label.setText(f"{point_count:,} pts")
        self.setWindowTitle(
            f"{viewer_title(self._kind)} ({point_count:,} pts)"
            + (f" — {self._product_title}" if self._product_title else "")
        )

    def _on_failed(self, message: str) -> None:
        self.show_placeholder(message)

# Registration uses domain PortType values as the single data contract.
for _kind in ("point_cloud", "classified_point_cloud", "tiles"):
    register(_kind, PointCloud3DViewer)
