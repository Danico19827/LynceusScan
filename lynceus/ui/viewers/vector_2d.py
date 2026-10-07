# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""2D viewer for vector products such as polygon GeoPackages.

Reads a layer with the PROJ-free vector reader (sqlite3 + shapely) and
renders it as RGBA. Numeric metrics use an RdYlGn choropleth, metric
selector, and legend; NODATA is gray. A sidecar ``.meta.json`` controls
the metric list and order through ``metrics_computed``. Without metadata,
generic bookkeeping columns are excluded. Registered for ``vector`` and
``grid_metrics`` products. The reader deliberately avoids
geopandas/pyproj/pyogrio: the GUI process already loads rasterio's PROJ
build, and mixing both stacks in one process crashes natively.
"""

import json
from pathlib import Path

import numpy as np

from PySide6.QtCore import QPointF, QRect, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QWidget,
)

from lynceus.plugins.locale import t
from lynceus.processing.qml_style import BOOKKEEPING_COLS
from lynceus.processing.vector_table import read_vector_features
from lynceus.ui.branding import THEME_COLORS
from lynceus.ui.viewers.base import BaseViewer
from lynceus.ui.viewers.palette import metric_colors
from lynceus.ui.viewers.registry import register

FILL = QColor(77, 163, 255, 56)
BORDER = QColor(127, 196, 255, 230)
NODATA_FILL = QColor(58, 64, 82, 130)
NODATA_BORDER = QColor(128, 136, 158, 230)
VECTOR_NODATA = -9999.0
MAX_IMAGE_PX = 2000
MAX_CACHED_IMAGES = 2
RENDER_CANCEL_CHECK_INTERVAL = 8192
CRS_LABEL_LEN = 40


def short_crs_label(crs) -> str:
    """Compact CRS label for info bars (EPSG code when resolvable).

    A raw WKT runs to thousands of characters and blows the preview
    window past the screen; the short label keeps the product
    identifiable without layout damage. The vector reader resolves the
    label from the GeoPackage metadata (no PROJ), so a plain string is
    the normal input; CRS-like objects stay accepted for API
    compatibility.
    """
    if crs is None or crs == "":
        return "—"
    if isinstance(crs, str):
        text = crs
        return text if len(text) <= CRS_LABEL_LEN else text[:CRS_LABEL_LEN] + "…"
    try:
        epsg = crs.to_epsg()
    except Exception:
        epsg = None
    if epsg:
        return f"EPSG:{epsg}"
    try:
        name = crs.name
    except Exception:
        name = ""
    text = str(name or crs)
    return text if len(text) <= CRS_LABEL_LEN else text[:CRS_LABEL_LEN] + "…"


class _PanView(QGraphicsView):
    """Zoomable and pannable graphics view for the rendered vector image."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._item: QGraphicsPixmapItem | None = None
        self._pending_fit = False

    def set_image(self, image: QImage) -> None:
        self._scene.clear()
        self._item = self._scene.addPixmap(QPixmap.fromImage(image))
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
        """Reframes the payload once there is a real viewport size."""
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


class _RampBar(QWidget):
    """RdYlGn gradient bar + gray NODATA cell, for the legend."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(12)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        steps = 64
        for i in range(steps):
            t = i / (steps - 1)
            r, g, b = (c * 255.0 for c in _ramp_color(t))
            painter.fillRect(
                QRect(int(i * self.width() / steps), 0, self.width() // steps + 1,
                      self.height()),
                QColor(int(r), int(g), int(b)),
            )
        painter.fillRect(
            QRect(int(self.width() * 0.94), 0, self.width() // 16 + 1,
                  self.height()),
            NODATA_FILL,
        )


def _ramp_color(t: float):
    """Interpolated color of the RdYlGn ramp (positions 0..1)."""
    import numpy as np

    rgb = metric_colors(np.array([float(t)], dtype=np.float32))[0]
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]))


def _as_float(values: np.ndarray) -> np.ndarray:
    """Numeric view of a column, NODATA/NaN for uncoercible values."""
    if values.dtype.kind in "fiu":
        return values.astype(np.float64)
    out = np.empty(len(values), dtype=np.float64)
    for index, value in enumerate(values):
        try:
            out[index] = float(value)
        except (TypeError, ValueError):
            out[index] = np.nan
    return out


class Vector2DViewer(BaseViewer):
    """2D view of polygon and point geometries from a GeoPackage.

    With numerically relevant metric columns it shows a choropleth per
    feature with a field selector and legend; otherwise a uniform fill.
    Points render as discs (treetops); polygons keep the box fast path
    when they form an axis-aligned grid, else the painter path.
    """

    def __init__(self, kind: str, parent=None):
        super().__init__(kind, parent)
        self._features = None
        self._fields: list[str] = []
        self._default_metric: str | None = None
        self._meta_bk: set = set()
        # Rendered images by (field, borders): metric switches and the
        # borders toggle reuse them instead of repainting thousands of cells.
        self._img_cache: dict = {}

        self._metric_row = QWidget(self)
        m_layout = QHBoxLayout(self._metric_row)
        m_layout.setContentsMargins(4, 4, 4, 0)
        metric_label = QLabel(t("Metric:"), self._metric_row)
        metric_label.setProperty("origText_en", "Metric:")
        m_layout.addWidget(metric_label)
        self._metric_combo = QComboBox(self._metric_row)
        m_layout.addWidget(self._metric_combo, 1)
        self._borders_check = QCheckBox(t("Borders"), self._metric_row)
        self._borders_check.setProperty("origText_en", "Borders")
        # Borders always start off (dense grids moiré): the user enables
        # them per preview when needed. No persistence by design.
        self._borders_check.setChecked(False)
        self._borders_check.toggled.connect(self._on_borders_toggled)
        m_layout.addWidget(self._borders_check)
        self._metric_row.hide()

        self._view = _PanView(self)
        self._view.hide()

        self._legend_row = QWidget(self)
        l_layout = QHBoxLayout(self._legend_row)
        l_layout.setContentsMargins(8, 2, 8, 0)
        self._legend_min = QLabel("", self._legend_row)
        self._legend_max = QLabel("", self._legend_row)
        self._legend_nodata = QLabel("NODATA", self._legend_row)
        self._legend_bar = _RampBar(self._legend_row)
        for lbl in (self._legend_min, self._legend_max, self._legend_nodata):
            lbl.setObjectName("viewerInfo")
        l_layout.addWidget(self._legend_min)
        l_layout.addWidget(self._legend_bar, 1)
        l_layout.addWidget(self._legend_max)
        l_layout.addWidget(self._legend_nodata)
        self._legend_row.hide()

        self._info = QLabel(self)
        self._info.setObjectName("viewerInfo")
        self._info.setWordWrap(True)
        self._info.hide()

        self._layout.addWidget(self._metric_row)
        self._layout.addWidget(self._view)
        self._layout.addWidget(self._legend_row)
        self._layout.addWidget(self._info)

        self._metric_combo.currentIndexChanged.connect(self._on_metric_changed)

    def set_payload(self, payload) -> None:
        path = payload.get("file") if isinstance(payload, dict) else None
        path_str = str(path) if path else ""
        if not path_str.lower().endswith(".gpkg"):
            self._show_placeholder(t("No vector data available"))
            return

        self._metric_row.hide()
        self._view.hide()
        self._legend_row.hide()
        self._info.hide()

        borders = self._borders_check.isChecked()

        def load(cancelled):
            if cancelled.is_set():
                return None
            meta = self._read_meta(path_str)
            metric_order = (
                [str(metric) for metric in meta.get("metrics_computed", ())]
                if "metrics_computed" in meta
                else None
            )
            features = read_vector_features(
                path_str, fields=metric_order, with_geometry=True
            )
            if cancelled.is_set():
                return None
            if len(features) == 0:
                return {"empty": True}

            bookkeeping = {
                str(column) for column in (meta.get("bookkeeping_cols") or ())
            }
            fields = self._numeric_fields(features, metric_order, bookkeeping)
            default_metric = meta.get("default_metric")
            field = (
                default_metric
                if default_metric in fields
                else fields[0] if fields else None
            )
            image = self._render_field(features, field, borders=borders)
            if image is None and field is not None:
                field = None
                image = self._render_field(features, None, borders=borders)
            return features, meta, metric_order, bookkeeping, fields, field, image

        def loaded(result) -> None:
            if result is None:
                return
            if isinstance(result, dict) and result.get("empty"):
                self._show_placeholder(t("Layer has no features"))
                return
            features, meta, metric_order, bookkeeping, fields, field, image = result
            self._features = features
            self._img_cache = {}
            self._metric_order = metric_order
            self._meta_bk = bookkeeping
            self._default_metric = meta.get("default_metric")
            self._fields = fields
            self._metric_combo.blockSignals(True)
            self._metric_combo.clear()
            for metric in fields:
                self._metric_combo.addItem(metric, metric)
            if fields:
                self._metric_combo.setCurrentIndex(fields.index(field or fields[0]))
                self._metric_row.show()
            else:
                self._metric_row.hide()
            self._metric_combo.blockSignals(False)
            if image is None:
                self._show_placeholder(t("No polygonal features to render"))
            elif field is None:
                self._img_cache[(None, borders)] = image
                self._apply_uniform_image(image)
            else:
                self._img_cache[(field, borders)] = image
                self._apply_field_image(image, field)

        def failed(exc: str) -> None:
            self._show_placeholder(
                t("Cannot read GeoPackage: {exc}").format(exc=exc)
                + (f"\n{path_str}" if path_str else "")
            )

        self._view.hide()
        self.start_async_load(load, loaded, failed)

    def _show_placeholder(self, text: str) -> None:
        self._features = None
        self._fields = []
        self._metric_order = None
        self._img_cache = {}
        self._metric_row.hide()
        self._view.hide()
        self._legend_row.hide()
        self._info.clear()
        self._info.hide()
        self.show_placeholder(text)

    @staticmethod
    def _read_meta(path_str: str) -> dict:
        """Sidecar ``<file>.meta.json`` declared by the producing node.

        The node (not the core) says which columns are ``metrics_computed``,
        which are ``bookkeeping_cols`` and which is the ``default_metric``;
        the viewer only honors it generically. An extension without a sidecar
        falls back to the generic backup (numeric columns minus universal ids).
        """
        try:
            meta = json.loads(
                Path(path_str).with_suffix(".meta.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return {}
        return meta if isinstance(meta, dict) else {}

    @staticmethod
    def _numeric_fields(
        features, allowed: list[str] | None = None, bk: set | None = None
    ) -> list[str]:
        if allowed is not None:
            return [c for c in allowed if c in features.columns]
        excluded = set(BOOKKEEPING_COLS) | set(bk or ())
        return [
            c for c in features.fields
            if c not in excluded and c in features.numeric
        ]

    def _default_field(self) -> str:
        if self._default_metric in self._fields:
            return self._default_metric
        return self._fields[0]

    def _on_metric_changed(self, _index: int) -> None:
        self._refresh()

    def _on_borders_toggled(self, checked: bool) -> None:
        self._refresh()

    def _refresh(self) -> None:
        if self._features is None or not self._fields:
            return
        field = self._metric_combo.currentData() or self._fields[0]
        key = (field, self._borders_check.isChecked())
        image = self._img_cache.get(key)
        if image is not None:
            self.cancel_async_loads()
            self._apply_field_image(image, field)
            return

        features = self._features
        borders = self._borders_check.isChecked()

        def load(cancelled):
            if cancelled.is_set():
                return None
            image = self._render_field(
                features, field, borders=borders, cancelled=cancelled
            )
            if cancelled.is_set():
                return None
            uniform = image is None
            if uniform:
                image = self._render_field(
                    features, None, borders=borders, cancelled=cancelled
                )
            return image, uniform

        def loaded(result) -> None:
            if result is None:
                return
            image, uniform = result
            if image is None:
                self._show_placeholder(t("No polygonal features to render"))
                return
            cache_key = (None if uniform else field, borders)
            self._cache_image(cache_key, image)
            if uniform:
                self._apply_uniform_image(image)
            else:
                self._apply_field_image(image, field)

        self.start_async_load(
            load,
            loaded,
            lambda exc: self._show_placeholder(
                t("Cannot read GeoPackage: {exc}").format(exc=exc)
            ),
            show_loading=False,
        )

    def _apply_field_image(self, image, field: str) -> None:
        img, (vmin, vmax, n_valid) = image
        self._view.set_image(img)
        self._view.show()
        self._placeholder.hide()
        self._info.show()
        self.setWindowTitle(self._compose_title())
        if n_valid == 0:
            self._legend_row.hide()
            self._info.setText(
                t("{n} features | CRS {crs} | {field}: all values NODATA").format(
                    n=len(self._features), crs=short_crs_label(self._features.crs_label),
                    field=field,
                )
            )
            return
        self._legend_min.setText(t("min {v}").format(v=f"{vmin:.2f}"))
        self._legend_max.setText(t("max {v}").format(v=f"{vmax:.2f}"))
        self._legend_row.show()
        self._info.setText(
            t("{n} features | CRS {crs} | {field}: {m} valid cells").format(
                n=len(self._features),
                crs=short_crs_label(self._features.crs_label),
                field=field,
                m=f"{n_valid:,}",
            )
        )

    def _render_uniform(self) -> None:
        if self._features is None:
            return
        key = (None, self._borders_check.isChecked())
        image = self._img_cache.get(key)
        if image is not None:
            self.cancel_async_loads()
            self._apply_uniform_image(image)
            return

        features = self._features
        borders = self._borders_check.isChecked()

        def load(cancelled):
            if cancelled.is_set():
                return None
            return self._render_field(
                features, None, borders=borders, cancelled=cancelled
            )

        def loaded(image) -> None:
            if image is None:
                self._show_placeholder(t("No polygonal features to render"))
                return
            self._cache_image(key, image)
            self._apply_uniform_image(image)

        self.start_async_load(
            load,
            loaded,
            lambda exc: self._show_placeholder(
                t("Cannot read GeoPackage: {exc}").format(exc=exc)
            ),
            show_loading=False,
        )

    def _apply_uniform_image(self, image) -> None:
        img, _ = image
        self._view.set_image(img)
        self._view.show()
        self._placeholder.hide()
        self._legend_row.hide()
        self._info.show()
        self.setWindowTitle(self._compose_title())
        cols = list(self._features.fields)
        self._info.setText(
            t("{n} features | CRS {crs} | attrs: {cols}").format(
                n=len(self._features),
                crs=short_crs_label(self._features.crs_label),
                cols=", ".join(cols[:8]),
            )
        )

    def _cache_image(self, key, image) -> None:
        if len(self._img_cache) >= MAX_CACHED_IMAGES:
            self._img_cache.pop(next(iter(self._img_cache)))
        self._img_cache[key] = image

    @staticmethod
    def _render_field(
        features, field: str | None, borders: bool = False, cancelled=None
    ):
        """VectorFeatures of polygons/points -> QImage over a dark background.

        With a numeric field it paints a choropleth (2-98% stretch, NODATA in
        gray / uniform if there is no valid data). Without a field it uses the
        previous uniform fill. Cell borders are only stroked when ``borders``
        is set: at fit-to-view zoom with dense grids the 1-px strokes would
        otherwise dominate the fills. Points render as filled discs with the
        same per-feature colors. Returns `(img, (vmin, vmax, n_valid))`
        (n_valid=0 if the field is all NODATA or there is no field) or None
        if there is nothing drawable.
        """
        from shapely.geometry import MultiPoint, MultiPolygon, Point, Polygon

        if features.bounds is None or features.geometries is None:
            return None
        minx, miny, maxx, maxy = features.bounds
        if not all(np.isfinite([minx, miny, maxx, maxy])):
            return None
        if cancelled is not None and cancelled.is_set():
            return None
        w_geo = max(maxx - minx, 1e-9)
        h_geo = max(maxy - miny, 1e-9)
        scale = MAX_IMAGE_PX / max(w_geo, h_geo)
        w_px = max(1, min(MAX_IMAGE_PX, int(w_geo * scale)))
        h_px = max(1, min(MAX_IMAGE_PX, int(h_geo * scale)))

        def to_px(x, y):
            return QPointF((x - minx) * scale, (maxy - y) * scale)

        field_data = None
        nodata_mask = None
        if field is not None and field in features.columns:
            vals = _as_float(features.columns[field])
            nodata_mask = ~np.isfinite(vals) | (vals == VECTOR_NODATA)
            valid = vals[~nodata_mask]
            if valid.size >= 2:
                vmin, vmax = np.percentile(valid, (2, 98))
                if vmax <= vmin:
                    vmin, vmax = float(valid.min()), float(valid.max())
                if vmax == vmin:
                    t = np.zeros_like(vals)
                else:
                    t = np.clip((vals - vmin) / (vmax - vmin), 0.0, 1.0)
                field_data = (t, vmin, vmax, int(valid.size))

        fast = (
            Vector2DViewer._try_raster_cells(
                features, field_data, nodata_mask, minx, maxy, scale,
                w_px, h_px, cancelled,
            )
            if not borders
            else None
        )
        if fast is not None:
            # Dense axis-aligned grids: numpy rasterization (no per-feature
            # Python). Falls back to the painter path for anything else.
            return fast
        if cancelled is not None and cancelled.is_set():
            return None

        img = QImage(QSize(w_px, h_px), QImage.Format.Format_RGB32)
        img.fill(QColor(THEME_COLORS["canvas_bg"]))
        painter = QPainter(img)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rgb_all = metric_colors(field_data[0]) if field_data is not None else None

        def _paint_box(poly) -> bool:
            """Fill (and optionally stroke) axis-aligned boxes directly.

            Grid cells are rectangles: a single fillRect/drawRect replaces
            per-vertex QPointF construction + drawPolygon. Returns False
            for anything else (holes included) so the generic path runs.
            """
            if poly.interiors:
                return False
            coords = poly.exterior.coords
            if len(coords) != 5:
                return False
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            if len(set(xs)) != 2 or len(set(ys)) != 2:
                return False
            x0 = (min(xs) - minx) * scale
            x1 = (max(xs) - minx) * scale
            y0 = (maxy - max(ys)) * scale
            y1 = (maxy - min(ys)) * scale
            rect = QRectF(x0, y0, x1 - x0, y1 - y0)
            if borders:
                painter.drawRect(rect)
            else:
                painter.fillRect(rect, painter.brush())
            return True

        point_radius = 3.0
        n_drawn = 0
        for i, geom in enumerate(features.geometries):
            if (
                cancelled is not None
                and i % RENDER_CANCEL_CHECK_INTERVAL == 0
                and cancelled.is_set()
            ):
                painter.end()
                return None
            dots: list
            if isinstance(geom, Point):
                dots = [geom]
            elif isinstance(geom, MultiPoint):
                dots = list(geom.geoms)
            else:
                dots = []
            polys: list
            if isinstance(geom, Polygon):
                polys = [geom]
            elif isinstance(geom, MultiPolygon):
                polys = list(geom.geoms)
            elif dots:
                polys = []
            else:
                continue
            if field_data is not None and not nodata_mask[i]:
                rgb = rgb_all[i]
                painter.setBrush(
                    QColor(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
                )
                painter.setPen(BORDER if borders else Qt.PenStyle.NoPen)
            elif field_data is not None:
                painter.setBrush(NODATA_FILL)
                painter.setPen(NODATA_BORDER if borders else Qt.PenStyle.NoPen)
            else:
                painter.setBrush(FILL)
                painter.setPen(BORDER if borders else Qt.PenStyle.NoPen)
            for dot in dots:
                center = to_px(dot.x, dot.y)
                painter.drawEllipse(
                    QRectF(center.x() - point_radius,
                           center.y() - point_radius,
                           point_radius * 2.0, point_radius * 2.0)
                )
                n_drawn += 1
            for poly in polys:
                if _paint_box(poly):
                    n_drawn += 1
                    continue
                painter.drawPolygon(
                    QPolygonF([to_px(x, y) for x, y in poly.exterior.coords])
                )
                for interior in poly.interiors:
                    painter.drawPolygon(
                        QPolygonF([to_px(x, y) for x, y in interior.coords])
                    )
                n_drawn += 1
        painter.end()
        if not n_drawn:
            return None
        if field_data is not None:
            return img, (float(field_data[1]), float(field_data[2]), field_data[3])
        return img, (0.0, 0.0, 0)

    @staticmethod
    def _try_raster_cells(
        features, field_data, nodata_mask, minx, maxy, scale, w_px, h_px,
        cancelled=None,
    ):
        """Numpy rasterization for dense axis-aligned box grids."""
        import shapely

        geometries = features.geometries
        if geometries is None or features.feature_count == 0:
            return None
        try:
            kind = shapely.get_type_id(geometries)
            box = shapely.bounds(geometries)
            bw = box[:, 2] - box[:, 0]
            bh = box[:, 3] - box[:, 1]
            area = shapely.area(geometries)
            perim = shapely.length(geometries)
        except Exception:
            return None
        if cancelled is not None and cancelled.is_set():
            return None
        if kind.size == 0 or not bool(
            (kind == shapely.GeometryType.POLYGON).all()
        ):
            return None
        if not bool(
            np.isclose(area, bw * bh, rtol=1e-9).all()
            and np.isclose(perim, 2.0 * (bw + bh), rtol=1e-9).all()
        ):
            return None
        if field_data is not None:
            rgb = (metric_colors(field_data[0]) * 255).astype(np.uint8)
            nod = np.asarray(nodata_mask, dtype=bool)
            gray = np.array(NODATA_FILL.getRgb()[:3], dtype=np.uint8)
            colors = np.empty((features.feature_count, 3), dtype=np.uint8)
            colors[nod] = gray
            colors[~nod] = rgb[~nod]
            stats = (float(field_data[1]), float(field_data[2]), field_data[3])
        else:
            fill = np.array(FILL.getRgb()[:3], dtype=np.uint8)
            colors = np.tile(fill, (features.feature_count, 1))
            stats = (0.0, 0.0, 0)
        x0 = np.floor((box[:, 0] - minx) * scale).astype(int)
        x1 = np.ceil((box[:, 2] - minx) * scale).astype(int)
        y0 = np.floor((maxy - box[:, 3]) * scale).astype(int)
        y1 = np.ceil((maxy - box[:, 1]) * scale).astype(int)
        x1 = np.maximum(x1, x0 + 1)
        y1 = np.maximum(y1, y0 + 1)
        np.clip(x0, 0, w_px, out=x0)
        np.clip(x1, 0, w_px, out=x1)
        np.clip(y0, 0, h_px, out=y0)
        np.clip(y1, 0, h_px, out=y1)
        background = np.array(
            QColor(THEME_COLORS["canvas_bg"]).getRgb()[:3], dtype=np.uint8
        )
        pixels = np.empty((h_px, w_px, 3), dtype=np.uint8)
        pixels[:, :] = background
        for i in range(features.feature_count):
            if (
                cancelled is not None
                and i % RENDER_CANCEL_CHECK_INTERVAL == 0
                and cancelled.is_set()
            ):
                return None
            if y1[i] > y0[i] and x1[i] > x0[i]:
                pixels[y0[i]:y1[i], x0[i]:x1[i]] = colors[i]
        image = QImage(
            pixels.data, w_px, h_px, pixels.strides[0], QImage.Format.Format_RGB888
        ).copy()
        return image, stats


for _kind in ("vector", "grid_metrics"):
    register(_kind, Vector2DViewer)
