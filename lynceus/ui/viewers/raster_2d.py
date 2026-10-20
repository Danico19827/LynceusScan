# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""2D viewer for raster products read with rasterio.

The GeoTIFF is cropped to valid data and decimated for display. Elevation
colors, subtle NODATA checkerboard shading, hillshade for DTM/DSM, zoom, pan,
fit-to-view, and world-coordinate value inspection are supported.
"""

import numpy as np
import rasterio
from affine import Affine
from pathlib import Path
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QWidget,
)

from lynceus.plugins.locale import t
from lynceus.processing.qml_style import (
    RAMPS,
    read_palette_entries,
    read_ramp_stops,
)
from lynceus.ui.viewers.base import BaseViewer
from lynceus.ui.viewers.palette import elevation_colors
from lynceus.ui.viewers.registry import register

NODATA_FALLBACK = -9999.0
DISPLAY_MAX_CELLS = 4_000_000
RASTER_STATS_SAMPLE_MAX = 250_000
CHECKER_A = (26, 34, 51)
CHECKER_B = (19, 26, 40)
# Zoom refine: reload the viewport at native resolution once magnified past
# this ratio over the current image (debounced, async, superseding).
_REFINE_DEBOUNCE_MS = 300
_REFINE_RATIO = 1.5
_REFINE_MARGIN = 0.5
# Smooth zoom: per-tick easing toward the wheel target (QGIS-like glide).
_ZOOM_TICK_MS = 15
_ZOOM_EASE = 0.25
_ZOOM_MIN_REL = 0.01
_ZOOM_MAX_ABS = 500.0


def _zoom_factor(delta: float) -> float:
    """Zoom step for a wheel delta: a 120-unit notch is ~x1.2, smooth
    trackpad deltas land in between (gradual, never jumps)."""
    try:
        step = float(delta)
    except (TypeError, ValueError):
        return 1.0
    if step == 0.0:
        return 1.0
    return pow(1.0015, step)


def _affine_coeffs(transform) -> tuple | None:
    """A 2D affine as ``(a, b, c, d, e, f)``; None when unreadable."""
    try:
        return (float(transform.a), float(transform.b), float(transform.c),
                float(transform.d), float(transform.e), float(transform.f))
    except (AttributeError, TypeError, ValueError):
        return None


def _sig_moved(requested: tuple, current: tuple, tol: float = 1e-6) -> bool:
    """Whether a viewport scene-rect drifted past float noise (pure)."""
    scale = max(1.0, abs(current[2] - current[0])
                + abs(current[3] - current[1]))
    return any(abs(a - b) > tol * scale for a, b in zip(requested, current))


def display_max_cells() -> int:
    """Raster cells drawn in 2D previews (Preferences > Display)."""
    try:
        from PySide6.QtCore import QSettings

        from lynceus.ui.settings_keys import (
            DISPLAY_RASTER_CELLS_KEY,
            SETTINGS_APP,
            SETTINGS_ORG,
        )

        value = int(
            QSettings(SETTINGS_ORG, SETTINGS_APP).value(
                DISPLAY_RASTER_CELLS_KEY, DISPLAY_MAX_CELLS
            )
        )
        return value if value > 0 else DISPLAY_MAX_CELLS
    except Exception:
        return DISPLAY_MAX_CELLS


def _hex_rgb(hex_color: str) -> tuple[int, int, int]:
    return (int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16))


def _ramp_colors(
    array: np.ndarray,
    minimum: float,
    maximum: float,
    stops: list[tuple[float, tuple[int, int, int]]],
) -> np.ndarray:
    """Map values through QML ramp stops (QGIS formula).

    Linear interpolation between item values over [minimum, maximum];
    outside values clip to the edge colors (QGIS ``clip="0"``).
    """
    positions = np.array([pos for pos, _ in stops], dtype=np.float64)
    flat = array.astype(np.float64).ravel()
    if maximum <= minimum:
        flat = np.zeros_like(flat)
    rgb = np.empty(array.shape + (3,), dtype=np.float64)
    for channel in range(3):
        channel_vals = np.array([color[channel] for _, color in stops], dtype=np.float64)
        rgb[..., channel] = np.interp(flat, positions, channel_vals).reshape(array.shape)
    return np.clip(rgb, 0.0, 255.0).astype(np.uint8)


def _invalid_mask(array: np.ndarray, nodata: float) -> np.ndarray:
    """Invalid-cell mask: the file tag plus NaN, always.

    A raster without a tag (or with NaN voids beside a tag) must never read
    its voids as data, whatever the fallback says.
    """
    mask = array == nodata
    try:
        mask = mask | ~np.isfinite(array)
    except TypeError:
        pass
    return mask


def _continuous_ramp(
    sidecar_qml: str | None,
) -> tuple[float, float, list] | None:
    """QML ramp as ``(minimum, maximum, [(value, rgb), ...])`` or None.

    The same mapping QGIS renders from the product sidecar; without a
    continuous sidecar the legacy elevation ramp applies (foreign files).
    """
    if not sidecar_qml:
        return None
    parsed = read_ramp_stops(sidecar_qml)
    if parsed is None:
        return None
    minimum, maximum, stops = parsed
    return minimum, maximum, [
        (value, _hex_rgb(color)) for value, color, _ in stops
    ]


def _discrete_style(
    array: np.ndarray,
    nodata: float,
    sidecar_qml: str | None = None,
) -> tuple[dict[int, tuple[int, int, int]], list[tuple[int, str]]] | None:
    """Return a categorical LUT when the raster represents discrete classes.

    Prefer the pipeline's paletted ``.qml`` sidecar so the preview matches
    QGIS. Without a sidecar, use the generic class palette only for small
    integer class sets; otherwise return None and use the continuous ramp.
    """
    if sidecar_qml:
        entries = read_palette_entries(sidecar_qml)
        if entries is not None:
            lut = {value: _hex_rgb(color) for value, color, _ in entries}
            labels = [(value, label) for value, _, label in entries]
            return lut, labels
    mask = _invalid_mask(array, nodata)
    valid_count = int((~mask).sum())
    if valid_count == 0:
        return None

    stride = max(1, int(np.ceil(array.size / RASTER_STATS_SAMPLE_MAX)))
    sample = array.ravel()[::stride]
    sample = sample[~_invalid_mask(sample, nodata)]
    values = np.unique(sample)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return None
    # Range first: huge finite strays (e.g. an untagged foreign tag) are
    # integral after rint and would overflow the int cast with a warning.
    if float(values.min()) < 0 or float(values.max()) >= 10:
        return None
    if not np.all(np.isclose(values, np.rint(values))):
        return None
    classes = values.astype(int)
    if not (2 <= classes.size <= 8 and classes.min() >= 0 and classes.max() < 10):
        return None
    stops = RAMPS["classes"]["stops"]
    lut = {int(value): _hex_rgb(color) for value, color, _ in stops}
    labels = [(int(value), label) for value, _, label in stops]
    return lut, labels


def _raster_to_image(
    array: np.ndarray,
    nodata: float,
    ramp: tuple[float, float, list] | None = None,
    lut: dict[int, tuple[int, int, int]] | None = None,
) -> tuple[QImage | None, tuple[float, float]]:
    """Convert an array to an RGB QImage and return its value range.

    NODATA uses a subtle checkerboard. ``ramp`` is the pipeline ``.qml``
    mapping ``(minimum, maximum, [(value, rgb), ...])`` — the same ramp
    and stretch QGIS renders, so previews stop looking like a different
    product. Without it (foreign files) the legacy elevation ramp with a
    2-98 percentile stretch applies. ``lut`` is the discrete path.
    """
    mask = _invalid_mask(array, nodata)
    valid_count = int((~mask).sum())
    if valid_count == 0:
        return None, (0.0, 0.0)
    flat = array.ravel()
    stride = max(1, int(np.ceil(flat.size / RASTER_STATS_SAMPLE_MAX)))
    stats = flat[::stride]
    stats = stats[~_invalid_mask(stats, nodata)]
    if stats.size == 0:
        stats = array[~mask]
    vmin, vmax = np.percentile(stats, (2, 98))
    if vmax <= vmin:
        vmin, vmax = float(stats.min()), float(stats.max())

    h, w = array.shape
    if lut is not None:
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for cls, (r, g, b) in lut.items():
            rgb[array == cls] = (r, g, b)
    elif ramp is not None:
        minimum, maximum, stops = ramp
        rgb = _ramp_colors(array, minimum, maximum, stops)
        vmin, vmax = float(minimum), float(maximum)
    else:
        if vmax == vmin:
            norm = np.zeros_like(array, dtype=float)
        else:
            norm = np.clip((array - vmin) / (vmax - vmin), 0.0, 1.0)
        rgb = (elevation_colors(norm) * 255.0).astype(np.uint8)

    checker = np.bitwise_xor(
        np.arange(h, dtype=np.uint8)[:, None] & 1,
        np.arange(w, dtype=np.uint8)[None, :] & 1,
    )
    image = rgb.copy()
    image[mask] = CHECKER_A
    image[mask & (checker == 1)] = CHECKER_B
    img = QImage(
        image.data, w, h, image.strides[0], QImage.Format.Format_RGB888
    ).copy()
    return img, (float(vmin), float(vmax))


def _checker_image(h: int, w: int) -> QImage:
    """NODATA checkerboard tile (zoomed into a void keeps showing the cut)."""
    checker = np.bitwise_xor(
        np.arange(h, dtype=np.uint8)[:, None] & 1,
        np.arange(w, dtype=np.uint8)[None, :] & 1,
    )
    image = np.empty((h, w, 3), dtype=np.uint8)
    image[...] = CHECKER_A
    image[checker == 1] = CHECKER_B
    return QImage(
        image.data, w, h, image.strides[0], QImage.Format.Format_RGB888
    ).copy()


def _valid_bounds(array: np.ndarray, nodata: float) -> tuple | None:
    """Valid-data bounding box as ``(r0, r1, c0, c1)`` (inclusive)."""
    valid = ~_invalid_mask(array, nodata)
    rows_ok = np.any(valid, axis=1)
    cols_ok = np.any(valid, axis=0)
    if not rows_ok.any():
        return None
    r0, r1 = np.flatnonzero(rows_ok)[[0, -1]]
    c0, c1 = np.flatnonzero(cols_ok)[[0, -1]]
    return (int(r0), int(r1), int(c0), int(c1))


def _render_products(array, transform, nodata, path, origin=(0, 0),
                     step=(1.0, 1.0)):
    """Shared render step (overview and viewport windows alike).

    ``origin``/``step`` locate ``array`` in the full pixel grid; both
    survive cropping/decimation by exact grid arithmetic, so scene
    placement is origin-times-step with no inverse composition. Returns
    ``(image, vrange, discrete, ramp, origin_full, scale, array, nodata)``
    with the sidecar-driven styles, or all-None when not visualizable.
    """
    bounds = _valid_bounds(array, nodata)
    if bounds is None:
        return None, None, None, None, None, None, None, None
    r0, r1, c0, c1 = bounds
    array = array[r0 : r1 + 1, c0 : c1 + 1]
    ox, oy = origin
    sx, sy = step
    origin_full = (ox + c0 * sx, oy + r0 * sy)
    h, w = array.shape
    budget = display_max_cells()
    if h * w > budget:
        s = int(np.ceil((h * w / budget) ** 0.5))
        array, _ = Raster2DViewer._decimate_average(array, transform, s,
                                                   nodata)
        sx, sy = sx * s, sy * s
    discrete = _discrete_style(
        array, nodata, sidecar_qml=str(Path(path).with_suffix(".qml")))
    lut = discrete[0] if discrete is not None else None
    ramp = None
    if lut is None:
        ramp = _continuous_ramp(
            str(Path(path).with_suffix(".qml")))
    image, vrange = _raster_to_image(array, nodata, ramp=ramp, lut=lut)
    if image is None:
        return None, None, None, None, None, None, None, None
    h, w = array.shape
    return (image, vrange, discrete, ramp, origin_full, (sx, sy),
            array, nodata)


class _ImageView(QGraphicsView):
    """Zoom/pan view over the raster image."""

    mouse_moved = Signal(float, float, float)
    refine_requested = Signal()  # viewport settled after zoom/pan
    reset_requested = Signal()  # double-click: back to the full overview

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        # Zoom stays where the cursor points (never recenters on its own).
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._item: QGraphicsPixmapItem | None = None
        self._array: np.ndarray | None = None
        self._transform = None
        self._nodata = NODATA_FALLBACK
        # Single image on screen (origin in full pixels, uniform scale to
        # scene units); swaps are atomic, so nothing can ever misalign.
        self._item: QGraphicsPixmapItem | None = None
        self._array: np.ndarray | None = None
        self._origin: tuple[float, float] = (0.0, 0.0)
        self._image_scale: tuple[float, float] = (1.0, 1.0)
        # Fixed full-resolution transform (scene→world); set by the viewer.
        self._full_transform = None
        # Pending fit: when embedded in a tab, the payload arrives
        # before the view has a real size (fitInView against a phantom
        # viewport leaves the content mis-framed).
        self._pending_fit = False
        # Smooth zoom: the wheel sets a target, a ticker eases toward it.
        self._zoom_target: float | None = None
        self._fit_scale = 0.0
        self._zoom_timer = QTimer(self)
        self._zoom_timer.setInterval(_ZOOM_TICK_MS)
        self._zoom_timer.timeout.connect(self._ease_zoom)
        # Native-resolution reload once magnified: debounced so a zoom
        # gesture fires one request, superseding any in flight.
        self._refine_timer = QTimer(self)
        self._refine_timer.setSingleShot(True)
        self._refine_timer.setInterval(_REFINE_DEBOUNCE_MS)
        self._refine_timer.timeout.connect(self.refine_requested.emit)
        self.horizontalScrollBar().valueChanged.connect(
            lambda _v: self._refine_timer.start())
        self.verticalScrollBar().valueChanged.connect(
            lambda _v: self._refine_timer.start())

    def set_image(self, image: QImage, origin, scale: float,
                  array: np.ndarray, nodata: float, fit: bool = True):
        """Show exactly one image: scene units are full-resolution pixels.

        ``origin`` is the image origin in full pixels, ``scale`` maps image
        pixels to scene units, so placement is origin-times-scale — exact
        for any affine, with no inverse composition anywhere. Replaces any
        previous image atomically (no overlay can ever misalign); the view
        transform is never touched here.
        """
        from PySide6.QtGui import QTransform

        self._scene.clear()
        self._item = self._scene.addPixmap(QPixmap.fromImage(image))
        ox, oy = origin
        sx, sy = scale
        self._item.setTransform(QTransform(sx, 0.0, 0.0, sy, ox, oy))
        self._array = array
        self._origin = (float(ox), float(oy))
        self._image_scale = (float(sx), float(sy))
        self._nodata = nodata
        self._zoom_target = None
        if fit:
            self.fit_to_view()
        self._pending_fit = fit

    def fit_to_view(self) -> None:
        if self._item is not None:
            self._zoom_target = None
            self.fitInView(self._item, Qt.AspectRatioMode.KeepAspectRatio)
            self._fit_scale = self.transform().m11()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._maybe_fit()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._maybe_fit()

    def _maybe_fit(self) -> None:
        """Fit the payload once the viewport has a usable size."""
        if self._pending_fit and self.isVisible():
            self._pending_fit = False
            self.fit_to_view()

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if not delta:
            delta = event.pixelDelta().y()
        if delta:
            current = self.transform().m11()
            target = self._zoom_target or current
            lo = (self._fit_scale or current) * _ZOOM_MIN_REL
            target = min(max(target * _zoom_factor(delta), lo),
                         _ZOOM_MAX_ABS)
            self._zoom_target = target
            if not self._zoom_timer.isActive():
                self._zoom_timer.start()
        self._refine_timer.start()
        event.accept()

    def _ease_zoom(self) -> None:
        """Glide the scale toward the wheel target (fluid, no jumps)."""
        if self._zoom_target is None:
            self._zoom_timer.stop()
            return
        current = self.transform().m11()
        if current <= 0:
            self._zoom_timer.stop()
            self._zoom_target = None
            return
        step = (self._zoom_target / current) ** _ZOOM_EASE
        if abs(step - 1.0) < 0.002:
            step = self._zoom_target / current
            self._zoom_target = None
            self._zoom_timer.stop()
            # Settled below scrollbar granularity: no valueChanged fires,
            # so schedule the coverage check explicitly.
            self._refine_timer.start()
        self.scale(step, step)

    def mouseDoubleClickEvent(self, event) -> None:
        self.reset_requested.emit()
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        super().mouseMoveEvent(event)
        item, array = self._item, self._array
        full = self._full_transform
        if item is None or array is None or full is None:
            return
        scene_pos = self.mapToScene(event.position().toPoint())
        img_pos = item.mapFromScene(scene_pos)
        col = int(round(img_pos.x()))
        row = int(round(img_pos.y()))
        h, w = array.shape
        if 0 <= row < h and 0 <= col < w:
            coeffs = _affine_coeffs(full)
            if coeffs is None:
                return
            ta, tb, tc, td, te, tf = coeffs
            # Scene units are full pixels: offset by the image origin.
            ox, oy = self._origin
            sx, sy = self._image_scale
            fcol = ox + (col + 0.5) * sx
            frow = oy + (row + 0.5) * sy
            x = tc + ta * fcol + tb * frow
            y = tf + td * fcol + te * frow
            self.mouse_moved.emit(x, y, float(array[row, col]))
            return


class Raster2DViewer(BaseViewer):
    """2D view of a GeoTIFF with elevation colormap."""

    def help_text(self) -> str | None:
        return (
            "<b>Navigate</b><br>Wheel: smooth zoom (native detail loads "
            "past 1.5x) · Drag: pan · Double-click: back to overview"
            "<br><b>Inspect</b><br>Hover the image: world coordinates "
            "and cell value"
        )

    def __init__(self, kind: str, parent=None):
        super().__init__(kind, parent)
        self._image_view = _ImageView(self)
        self._image_view.hide()
        self._legend = QWidget(self)
        self._legend_layout = QHBoxLayout(self._legend)
        self._legend_layout.setContentsMargins(0, 0, 0, 0)
        self._legend_layout.setSpacing(6)
        self._legend.hide()
        # Rebuilding the legend reflows the layout (viewport resize =
        # visible tremor), so identical legends are kept as-is.
        self._legend_key = None
        self._info = QLabel("")
        self._info.setObjectName("viewerInfo")
        info_bar = QWidget(self)
        info_layout = QHBoxLayout(info_bar)
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(4)
        info_layout.addWidget(self._info, 1)
        self._layout.addWidget(self._image_view)
        self._layout.addWidget(self._legend)
        self._layout.addWidget(info_bar)
        self.install_help_button(info_layout)
        self._image_view.mouse_moved.connect(self._on_mouse_moved)
        self._image_view.refine_requested.connect(self._on_refine_request)
        self._image_view.reset_requested.connect(self._on_reset_request)
        # Single on-screen image: full-pixel extent + cells per world unit
        # (None until the first load lands).
        self._image = None
        # Full-resolution raster bounds for viewport windows: (w, h, transform).
        self._raster_meta: tuple | None = None
        # First overview load shows the placeholder; reloads over a visible
        # image stay silent (placeholder + old image split the view).
        self._expecting_first_load = False

    def _set_legend(self, labels: list[tuple[int, str]], colors: dict[int, tuple[int, int, int]]) -> None:
        """Legend with swatches and names for categorical rasters."""
        key = ("discrete", tuple(labels), tuple(sorted(colors.items())))
        if key == self._legend_key:
            return
        self._legend_key = key
        while self._legend_layout.count():
            item = self._legend_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for value, name in labels:
            r, g, b = colors[value]
            pm = QPixmap(14, 14)
            pm.fill(QColor(r, g, b))
            swatch = QLabel()
            swatch.setPixmap(pm)
            swatch.setFixedSize(18, 18)
            swatch.setAlignment(Qt.AlignmentFlag.AlignCenter)
            text = QLabel(name)
            text.setObjectName("viewerLegendText")
            self._legend_layout.addWidget(swatch)
            self._legend_layout.addWidget(text)
            self._legend_layout.addSpacing(8)
        self._legend.show()

    def _hide_legend(self) -> None:
        self._legend_key = None
        while self._legend_layout.count():
            item = self._legend_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._legend.hide()

    def _set_gradient_legend(
        self, ramp: tuple[float, float, list]
    ) -> None:
        """QGIS-style gradient bar with min/max labels (numeric, no i18n)."""
        minimum, maximum, stops = ramp
        key = ("gradient", minimum, maximum,
               tuple((value, color) for value, color in stops))
        if key == self._legend_key:
            return
        self._legend_key = key
        while self._legend_layout.count():
            item = self._legend_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        minimum, maximum, stops = ramp
        width = 160
        strip = np.empty((14, width, 3), dtype=np.uint8)
        positions = np.array([pos for pos, _ in stops], dtype=np.float64)
        grid = np.linspace(minimum, maximum, width)
        for channel in range(3):
            channel_vals = np.array(
                [color[channel] for _, color in stops], dtype=np.float64)
            strip[:, :, channel] = np.interp(grid, positions, channel_vals)
        pixmap = QPixmap(width, 14)
        image = QImage(
            strip.data, width, 14, strip.strides[0],
            QImage.Format.Format_RGB888,
        ).copy()
        pixmap.convertFromImage(image)
        bar = QLabel()
        bar.setPixmap(pixmap)
        bar.setFixedSize(width + 2, 16)
        self._legend_layout.addWidget(QLabel(f"{minimum:.2f}"))
        self._legend_layout.addWidget(bar)
        self._legend_layout.addWidget(QLabel(f"{maximum:.2f}"))
        self._legend.show()

    def set_payload(self, payload) -> None:
        path = payload.get("file") if isinstance(payload, dict) else None
        self._source_path = str(path) if path else ""
        if not path:
            self.show_placeholder(t("No data available"))
            self._image_view.hide()
            self._legend.hide()
            self._info.clear()
            return

        path = str(path)
        self._image = None
        self._raster_meta = None
        self._expecting_first_load = True
        self._image_view.hide()
        self._start_overview()

    def _track_image(self, origin, scale, array, full) -> None:
        """Coverage snapshot of the image currently on screen."""
        h, w = array.shape
        ox, oy = origin
        sx, sy = scale
        coeffs = _affine_coeffs(full)
        if coeffs is None:
            return
        a, b, c, d, e, f = coeffs
        corners = [(ox, oy), (ox + w * sx, oy),
                   (ox, oy + h * sy), (ox + w * sx, oy + h * sy)]
        worlds = [(c + a * px + b * py, f + d * px + e * py)
                  for px, py in corners]
        xs = [p[0] for p in worlds]
        ys = [p[1] for p in worlds]
        world_w = max(xs) - min(xs)
        world_h = max(ys) - min(ys)
        if world_w <= 0 or world_h <= 0:
            return
        self._image = {
            "extent": (ox, oy, ox + w * sx, oy + h * sy),
            "have": min(w / world_w, h / world_h),
        }

    def _start_overview(self) -> None:
        """Load the full-extent decimated overview (initial view)."""
        path = self._source_path
        max_cells = display_max_cells()
        first = self._expecting_first_load
        self._expecting_first_load = False

        def load(cancelled):
            if cancelled.is_set():
                return None
            with rasterio.open(str(path)) as src:
                nodata = src.nodata if src.nodata is not None else NODATA_FALLBACK
                full = (src.width, src.height, src.transform)
                scale = max(1.0, np.sqrt(src.width * src.height / max_cells))
                out_width = max(1, int(np.ceil(src.width / scale)))
                out_height = max(1, int(np.ceil(src.height / scale)))
                array = src.read(
                    1,
                    out_shape=(out_height, out_width),
                    resampling=rasterio.enums.Resampling.nearest,
                ).astype(np.float32, copy=False)
                step = (src.width / out_width, src.height / out_height)
                transform = src.transform * Affine.scale(
                    src.width / out_width, src.height / out_height
                )
            if cancelled.is_set():
                return None
            rendered = _render_products(array, transform, nodata, path,
                                        origin=(0, 0), step=step)
            if rendered[0] is None:
                return None
            (image, vrange, discrete, ramp, origin_full, out_step,
             array, nodata) = rendered
            return (image, origin_full, out_step, array, nodata, vrange,
                    discrete, ramp, full)

        def loaded(result) -> None:
            if result is None:
                self.show_placeholder(t("Raster has no valid data"))
                self._image_view.hide()
                self._hide_legend()
                self._info.clear()
                return
            (image, origin, scale, array, nodata, vrange, discrete, ramp,
             full) = result
            # Hide first and force the reflow: fitting against the
            # placeholder-shrunk viewport chronically under-fits, leaving
            # the view zoomed out (viewport outside the image) which the
            # refine slot then "fixes" with another overview, forever.
            self._placeholder.hide()
            self._layout.activate()
            self._image_view.set_image(image, origin, scale, array, nodata)
            self._image_view._full_transform = full[2]
            self._image_view.show()
            self.setWindowTitle(self._compose_title())
            self._raster_meta = full
            self._track_image(origin, scale, array, full[2])
            valid_count = int((~_invalid_mask(array, nodata)).sum())
            self._info.setText(
                t("min {a}  max {b}  cells {c}").format(
                    a=f"{vrange[0]:.2f}",
                    b=f"{vrange[1]:.2f}",
                    c=f"{valid_count:,}",
                )
            )
            if discrete is not None:
                self._set_legend(discrete[1], discrete[0])
            elif ramp is not None:
                self._set_gradient_legend(ramp)
            else:
                self._hide_legend()

        def failed(exc: str) -> None:
            self.show_placeholder(
                t("Error reading raster: {exc}").format(exc=exc)
                + (f"\n{self._source_path}" if self._source_path else "")
            )
            self._image_view.hide()
            self._hide_legend()
            self._info.clear()

        self.start_async_load(load, loaded, failed,
                             show_loading=first)

    def _on_reset_request(self) -> None:
        """Double-click: back to the full overview, never a framed tile."""
        self._image_view.fit_to_view()
        self._start_overview()

    @staticmethod
    def _window_for_pixels(x0, y0, x1, y1, raster_w, raster_h,
                           budget) -> tuple | None:
        """Pixel window for a full-grid rect: ``(Window, out_shape)``.

        Pure helper: clamps to the raster (floor/ceil outward), refuses
        empty rects, and decimates straight to the budget instead of
        refusing over-budget windows — every viewport gets exactly one
        image. ``out_shape`` None means native resolution.
        """
        from rasterio.windows import Window

        c0 = max(0, int(np.floor(min(x0, x1))))
        c1 = min(int(raster_w), int(np.ceil(max(x0, x1))))
        r0 = max(0, int(np.floor(min(y0, y1))))
        r1 = min(int(raster_h), int(np.ceil(max(y0, y1))))
        if c1 <= c0 or r1 <= r0 or budget <= 0:
            return None
        w, h = c1 - c0, r1 - r0
        if w * h <= budget:
            return Window(c0, r0, w, h), None
        scale = max(1.0, np.sqrt(w * h / budget))
        out_w = max(1, int(w / scale))
        out_h = max(1, int(h / scale))
        return Window(c0, r0, w, h), (out_h, out_w)

    def _viewport_covered(self, req) -> bool:
        """Whether the current image covers a full-pixel rect (exact)."""
        image = self._image
        if not image:
            return False
        x0, y0, x1, y1 = image["extent"]
        slack = 1.0
        return (req[0] >= x0 - slack and req[1] >= y0 - slack
                and req[2] <= x1 + slack and req[3] <= y1 + slack)

    def _on_refine_request(self) -> None:
        """Reload the viewport at the right resolution once settled.

        Single image, single transform: the window always covers the
        viewport by construction and atomically replaces the previous
        image, so no overlay can ever misalign or teleport the view.
        """
        view = self._image_view
        meta = self._raster_meta
        if meta is None or view._item is None:
            return
        viewport = view.viewport().rect()
        if viewport.width() <= 0 or viewport.height() <= 0:
            return
        # Scene units are full-resolution pixels (invariant): the viewport
        # rect already is the request in file pixels — no mapping, no
        # inverse, nothing to drift.
        scene_rect = view.mapToScene(viewport).boundingRect()
        req = (scene_rect.left(), scene_rect.top(),
               scene_rect.right(), scene_rect.bottom())
        raster_w, raster_h, full_transform = meta
        coeffs = _affine_coeffs(full_transform)
        if coeffs is None:
            return
        fa, fb, fc, fd, fe, ff = coeffs
        corners = [(req[0], req[1]), (req[2], req[1]),
                   (req[0], req[3]), (req[2], req[3])]
        worlds = [(fc + fa * px + fb * py, ff + fd * px + fe * py)
                  for px, py in corners]
        world_w = max(p[0] for p in worlds) - min(p[0] for p in worlds)
        world_h = max(p[1] for p in worlds) - min(p[1] for p in worlds)
        if world_w <= 0 or world_h <= 0:
            return
        # Physical pixels: on hidpi the backing store holds dpr screen
        # pixels per scene unit, so the refine owes that much data.
        try:
            dpr = float(view.devicePixelRatioF() or 1.0)
        except Exception:
            dpr = 1.0
        need = max(viewport.width() * dpr / world_w,
                   viewport.height() * dpr / world_h)
        image = self._image or {}
        covered = self._viewport_covered(req)
        if covered and need <= image.get("have", 0) * _REFINE_RATIO:
            return
        raster_w, raster_h, full_transform = meta
        try:
            native_have = min(1.0 / abs(fa), 1.0 / abs(fe))
        except ArithmeticError:
            return
        if covered and image.get("have", 0) >= native_have:
            # Native pixels on screen: magnifying only interpolates.
            return
        margin = _REFINE_MARGIN
        windowed = self._window_for_pixels(
            req[0] - (req[2] - req[0]) * margin,
            req[1] - (req[3] - req[1]) * margin,
            req[2] + (req[2] - req[0]) * margin,
            req[3] + (req[3] - req[1]) * margin,
            raster_w, raster_h, display_max_cells())
        if windowed is None:
            return
        # Revision-tag the request with the viewport it was built for: a
        # GB-scale window takes seconds to read, and applying it after the
        # user moved on paints a sharp patch of somewhere-else (teleport
        # feeling with exact math). Stale arrivals are dropped instead.
        self._start_windowed(windowed, (scene_rect.left(), scene_rect.top(),
                                       scene_rect.right(),
                                       scene_rect.bottom()))

    @staticmethod
    def _viewport_moved(requested, view, tol: float = 1e-6) -> bool:
        """Whether the viewport left the scene rect a refine was built for."""
        rect = view.mapToScene(view.viewport().rect()).boundingRect()
        current = (rect.left(), rect.top(), rect.right(), rect.bottom())
        return _sig_moved(requested, current, tol)

    def _start_windowed(self, windowed, viewport_sig=None) -> None:
        """Load one viewport window and atomically swap it in.

        ``windowed`` is ``(Window, out_shape)`` from ``_window_for_pixels``:
        native when it fits the budget, decimated straight to it otherwise.
        The swap replaces the single image (never an overlay), so nothing
        can misalign; the view transform is never touched.
        """
        path = self._source_path
        window, out_shape = windowed

        def load(cancelled):
            if cancelled.is_set():
                return None
            with rasterio.open(str(path)) as src:
                nodata = src.nodata if src.nodata is not None else NODATA_FALLBACK
                if out_shape is None:
                    array = src.read(1, window=window).astype(np.float32)
                    step = (1.0, 1.0)
                else:
                    array = src.read(
                        1, window=window, out_shape=out_shape,
                        resampling=rasterio.enums.Resampling.nearest,
                    ).astype(np.float32, copy=False)
                    step = (window.width / out_shape[1],
                            window.height / out_shape[0])
                transform = src.window_transform(window)
            if cancelled.is_set():
                return None
            origin = (window.col_off, window.row_off)
            rendered = _render_products(array, transform, nodata, path,
                                        origin=origin, step=step)
            if rendered[0] is None:
                # Zoomed into a void: checkerboard tile (same cut QGIS
                # shows) instead of a blank swap.
                h, w = array.shape
                void = np.full((h, w), nodata, dtype=np.float32)
                return (_checker_image(h, w), origin, step, void, nodata,
                        (0.0, 0.0), None, None, True)
            (image, vrange, discrete, ramp, origin_full, out_step,
             array, nodata) = rendered
            return (image, origin_full, out_step, array, nodata, vrange,
                    discrete, ramp, False)

        def loaded(result) -> None:
            if result is None:
                return
            if (viewport_sig is not None and self._viewport_moved(
                    viewport_sig, self._image_view)):
                # The user moved on while this window was reading: applying
                # it would paint somewhere-else sharp over the new view.
                # Drop it; the movement already scheduled a fresh request.
                self._image_view._refine_timer.start()
                return
            (image, origin, scale, array, nodata, vrange, discrete, ramp,
             is_void) = result
            self._image_view.set_image(image, origin, scale, array, nodata,
                                       fit=False)
            self._image_view._full_transform = self._raster_meta[2]
            self._track_image(origin, scale, array, self._raster_meta[2])
            if is_void:
                return
            if discrete is not None:
                self._set_legend(discrete[1], discrete[0])
            elif ramp is not None:
                self._set_gradient_legend(ramp)
            else:
                self._hide_legend()

        def failed(_exc: str) -> None:
            # A failed reload keeps the current image: zoom stays soft
            # rather than blanking the view.
            return

        self.start_async_load(load, loaded, failed, show_loading=False)

    @staticmethod
    def _crop_valid(array, transform, nodata):
        """Crop to valid-data bounds; the transform update is exact for
        any affine (rotation terms survive, unlike a scale-only shift)."""
        bounds = _valid_bounds(array, nodata)
        if bounds is None:
            return None, transform
        r0, r1, c0, c1 = bounds
        array = array[r0 : r1 + 1, c0 : c1 + 1]
        coeffs = _affine_coeffs(transform)
        if coeffs is None:
            return array, transform
        a, b, c, d, e, f = coeffs
        return array, Affine(
            a, b, c + a * c0 + b * r0, d, e, f + d * c0 + e * r0
        )

    @staticmethod
    def _decimate_average(array, transform, step, nodata):
        """Downsample by integer nodata-aware block averaging (display).

        NODATA cells never dilute valid neighbours (QGIS-pyramid manners);
        fully-void blocks stay NODATA. The extent is preserved exactly by
        padding the tail with NODATA; the transform scales exactly,
        rotation included. Display-only: never feeds fingerprints.
        """
        step = max(2, int(step))
        h, w = array.shape
        pad_h = (-h) % step
        pad_w = (-w) % step
        if pad_h or pad_w:
            array = np.pad(
                array, ((0, pad_h), (0, pad_w)),
                mode="constant", constant_values=nodata,
            )
        hh, ww = array.shape
        blocks = array.reshape(hh // step, step, ww // step, step)
        ok = ~_invalid_mask(blocks, nodata)
        sums = np.where(ok, blocks.astype(np.float64), 0.0).sum(axis=(1, 3))
        counts = ok.sum(axis=(1, 3))
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(
                counts > 0, sums / np.maximum(counts, 1), nodata
            ).astype(np.float32)
        coeffs = _affine_coeffs(transform)
        if coeffs is None:
            return out, transform
        a, b, c, d, e, f = coeffs
        return out, Affine(a * step, b * step, c, d * step, e * step, f)

    def _on_mouse_moved(self, x: float, y: float, value: float) -> None:
        import math

        if value == self._image_view._nodata or (
            isinstance(value, float) and math.isnan(value)
        ):
            self._info.setText(t("x {x}  y {y}").format(x=f"{x:.2f}", y=f"{y:.2f}"))
        else:
            self._info.setText(
                t("x {x}  y {y}  value {v}").format(
                    x=f"{x:.2f}", y=f"{y:.2f}", v=f"{value:.2f}"
                )
            )


# The registry uses the PortType VALUES of the domain (single contract)
_RASTER_KINDS = (
    "raster",
    "dtm_tile",
    "dsm_tile",
    "dtm_mosaic",
    "dsm_mosaic",
    "chm_mosaic",
    "strata_raster",
)
for _kind in _RASTER_KINDS:
    register(_kind, Raster2DViewer)