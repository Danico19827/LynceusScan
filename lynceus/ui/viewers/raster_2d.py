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
from PySide6.QtCore import Qt, Signal
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
from lynceus.processing.qml_style import RAMPS, read_palette_entries
from lynceus.ui.viewers.base import BaseViewer
from lynceus.ui.viewers.palette import elevation_colors
from lynceus.ui.viewers.registry import register

NODATA_FALLBACK = -9999.0
DISPLAY_MAX_CELLS = 4_000_000
RASTER_STATS_SAMPLE_MAX = 250_000
CHECKER_A = (26, 34, 51)
CHECKER_B = (19, 26, 40)


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


def _hillshade(array: np.ndarray, cell_size: float = 1.0) -> np.ndarray:
    """Return normalized hillshade using a northwest light source."""
    gy, gx = np.gradient(array, cell_size, cell_size)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az = np.radians(315.0)
    alt = np.radians(45.0)
    shade = (
        np.sin(alt) * np.cos(slope)
        + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    )
    return np.clip(shade, 0.0, 1.0)


def _hex_rgb(hex_color: str) -> tuple[int, int, int]:
    return (int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16))


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
    mask = array == nodata
    valid_count = int((~mask).sum())
    if valid_count == 0:
        return None

    stride = max(1, int(np.ceil(array.size / RASTER_STATS_SAMPLE_MAX)))
    sample = array.ravel()[::stride]
    sample = sample[sample != nodata]
    values = np.unique(sample)
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
    shaded: bool = False,
    lut: dict[int, tuple[int, int, int]] | None = None,
) -> tuple[QImage | None, tuple[float, float]]:
    """Convert an array to an RGB QImage and return its value range.

    NODATA uses a subtle checkerboard. ``shaded`` overlays hillshade for
    continuous DTM/DSM products, never for a discrete LUT.
    """
    mask = array == nodata
    valid_count = int((~mask).sum())
    if valid_count == 0:
        return None, (0.0, 0.0)
    flat = array.ravel()
    stride = max(1, int(np.ceil(flat.size / RASTER_STATS_SAMPLE_MAX)))
    stats = flat[::stride]
    stats = stats[stats != nodata]
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
    else:
        if vmax == vmin:
            norm = np.zeros_like(array, dtype=float)
        else:
            norm = np.clip((array - vmin) / (vmax - vmin), 0.0, 1.0)
        rgb = (elevation_colors(norm) * 255.0).astype(np.uint8)
        if shaded:
            fill = float(np.median(stats))
            work = np.where(mask, fill, array)
            shade = _hillshade(work)
            rgb = np.clip(rgb * (0.45 + 0.55 * shade[..., None]), 0.0, 255.0)

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


class _ImageView(QGraphicsView):
    """Zoom/pan view over the raster image."""

    mouse_moved = Signal(float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self._item: QGraphicsPixmapItem | None = None
        self._array: np.ndarray | None = None
        self._transform = None
        self._nodata = NODATA_FALLBACK
        # Pending fit: when embedded in a tab, the payload arrives
        # before the view has a real size (fitInView against a phantom
        # viewport leaves the content mis-framed).
        self._pending_fit = False

    def set_image(self, image: QImage, transform, array: np.ndarray, nodata: float):
        self._scene.clear()
        self._item = self._scene.addPixmap(QPixmap.fromImage(image))
        self._array = array
        self._transform = transform
        self._nodata = nodata
        self.fit_to_view()
        self._pending_fit = True

    def fit_to_view(self) -> None:
        if self._item is not None:
            self.fitInView(self._item, Qt.AspectRatioMode.KeepAspectRatio)

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
        factor = 1.25 if event.angleDelta().y() > 0 else 1 / 1.25
        self.scale(factor, factor)
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:
        self.fit_to_view()
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        super().mouseMoveEvent(event)
        if self._item is None or self._array is None:
            return
        img_pos = self._item.mapFromScene(
            self.mapToScene(event.position().toPoint())
        )
        col = int(round(img_pos.x()))
        row = int(round(img_pos.y()))
        h, w = self._array.shape
        if 0 <= row < h and 0 <= col < w:
            t = self._transform
            x = t.c + (col + 0.5) * t.a
            y = t.f + (row + 0.5) * t.e
            self.mouse_moved.emit(x, y, float(self._array[row, col]))


class Raster2DViewer(BaseViewer):
    """2D view of a GeoTIFF with elevation colormap."""

    def __init__(self, kind: str, parent=None):
        super().__init__(kind, parent)
        self._image_view = _ImageView(self)
        self._image_view.hide()
        self._legend = QWidget(self)
        self._legend_layout = QHBoxLayout(self._legend)
        self._legend_layout.setContentsMargins(0, 0, 0, 0)
        self._legend_layout.setSpacing(6)
        self._legend.hide()
        self._info = QLabel("")
        self._info.setObjectName("viewerInfo")
        self._layout.addWidget(self._image_view)
        self._layout.addWidget(self._legend)
        self._layout.addWidget(self._info)
        self._image_view.mouse_moved.connect(self._on_mouse_moved)

    def _set_legend(self, labels: list[tuple[int, str]], colors: dict[int, tuple[int, int, int]]) -> None:
        """Legend with swatches and names for categorical rasters."""
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
        while self._legend_layout.count():
            item = self._legend_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._legend.hide()

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
        max_cells = display_max_cells()
        self._image_view.hide()

        def load(cancelled):
            if cancelled.is_set():
                return None
            with rasterio.open(str(path)) as src:
                nodata = src.nodata if src.nodata is not None else NODATA_FALLBACK
                scale = max(1.0, np.sqrt(src.width * src.height / max_cells))
                out_width = max(1, int(np.ceil(src.width / scale)))
                out_height = max(1, int(np.ceil(src.height / scale)))
                array = src.read(
                    1,
                    out_shape=(out_height, out_width),
                    resampling=rasterio.enums.Resampling.nearest,
                ).astype(np.float32, copy=False)
                transform = src.transform * Affine.scale(
                    src.width / out_width, src.height / out_height
                )
            if cancelled.is_set():
                return None
            array, transform = self._crop_and_decimate(array, transform, nodata)
            if array is None:
                return None
            discrete = _discrete_style(
                array, nodata, sidecar_qml=str(Path(path).with_suffix(".qml"))
            )
            lut = discrete[0] if discrete is not None else None
            image, vrange = _raster_to_image(
                array,
                nodata,
                shaded=self._kind in ("DTM", "DSM") and lut is None,
                lut=lut,
            )
            if image is None:
                return None
            return image, transform, array, nodata, vrange, discrete

        def loaded(result) -> None:
            if result is None:
                self.show_placeholder(t("Raster has no valid data"))
                self._image_view.hide()
                self._hide_legend()
                self._info.clear()
                return
            image, transform, array, nodata, vrange, discrete = result
            self._image_view.set_image(image, transform, array, nodata)
            self._image_view.show()
            self._placeholder.hide()
            self.setWindowTitle(self._compose_title())
            valid_count = int((array != nodata).sum())
            self._info.setText(
                t("min {a}  max {b}  cells {c}").format(
                    a=f"{vrange[0]:.2f}",
                    b=f"{vrange[1]:.2f}",
                    c=f"{valid_count:,}",
                )
            )
            if discrete is not None:
                self._set_legend(discrete[1], discrete[0])
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

        self.start_async_load(load, loaded, failed)

    @staticmethod
    def _crop_and_decimate(array, transform, nodata):
        """Crop to valid-data bounds and decimate for display when needed."""
        valid = array != nodata
        rows_ok = np.any(valid, axis=1)
        cols_ok = np.any(valid, axis=0)
        if not rows_ok.any():
            return None, transform
        r0, r1 = np.flatnonzero(rows_ok)[[0, -1]]
        c0, c1 = np.flatnonzero(cols_ok)[[0, -1]]
        array = array[r0 : r1 + 1, c0 : c1 + 1]
        transform = Affine(
            transform.a, 0.0, transform.c + c0 * transform.a,
            0.0, transform.e, transform.f + r0 * transform.e,
        )
        h, w = array.shape
        budget = display_max_cells()
        if h * w > budget:
            step = int(np.ceil((h * w / budget) ** 0.5))
            array = array[::step, ::step]
            transform = Affine(
                transform.a * step, 0.0, transform.c,
                0.0, transform.e * step, transform.f,
            )
        return array, transform

    def _on_mouse_moved(self, x: float, y: float, value: float) -> None:
        if value == self._image_view._nodata:
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