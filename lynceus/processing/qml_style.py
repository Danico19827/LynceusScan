# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Generate QGIS (.qml) styles for pipeline products.

Each node declares its own style through ``PROCESSING_SPECS["qml"]``. A spec
may reference a core raster ramp or vector field, define an inline palette,
or use ``{"self_styled": True}`` when the node writes dynamic classes itself.
Missing specs use filename conventions and generic continuous or categorical
fallbacks. Valid raster/vector products receive a sidecar whenever styling is
possible.

Styles belong to their producing node. Extensions declare inline ramps/classes
in their own module and are read when the node runs. ``RAMPS`` is the standard
core catalog; ``generic`` and ``classes`` are generic fallbacks.

The controller invokes ``style_outputs(outputs, specs)`` once per run. Sidecars
are written next to products, so QGIS loads them automatically; the 2D viewer
reads the same sidecar through ``read_palette_entries``. This module remains
Qt-free and safe to call from the worker thread.

Products without valid raster/vector data are skipped silently; styling is an
optional enhancement and never fails the run.
"""

from __future__ import annotations

import logging
import xml.sax.saxutils as saxutils
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Repository NODATA convention for vector products.
VECTOR_NODATA = -9999.0

# Default ramp for rasters without a known style.
DEFAULT_RAMP = "generic"

# Generic bookkeeping columns that are never styled as metrics. Product-specific
# columns belong to the producing node and are declared in its sidecar.
BOOKKEEPING_COLS = {"gid", "fid", "ogc_fid"}

_RASTER_EXT = {".tif", ".tiff"}
# No SHP: reading it would pull geopandas/pyogrio (a second PROJ build)
# into the GUI process, which crashes natively next to rasterio's PROJ.
_VECTOR_EXT = {".gpkg", ".geojson"}

RAMPS: dict[str, dict] = {
    "dtm": {
        "name": "Terrain",
        "stops": [
            (0.00, "#333399", "Min"),
            (0.20, "#00b2b2", ""),
            (0.40, "#99eb85", ""),
            (0.60, "#ccbe7d", ""),
            (0.80, "#997c76", ""),
            (1.00, "#ffffff", "Max"),
        ],
    },
    "dsm": {
        "name": "Viridis",
        "stops": [
            (0.00, "#440154", "Min"),
            (0.20, "#3b528b", ""),
            (0.40, "#21918c", ""),
            (0.60, "#5ec962", ""),
            (0.80, "#b8de29", ""),
            (1.00, "#fde725", "Max"),
        ],
    },
    "chm": {
        "name": "RdYlGn",
        "stops": [
            (0.00, "#d7191c", "Min"),
            (0.20, "#fdae61", ""),
            (0.40, "#ffffbf", ""),
            (0.60, "#a6d96a", ""),
            (0.80, "#1a9850", ""),
            (1.00, "#006837", "Max"),
        ],
    },
    "hillshade": {
        "name": "Gray",
        "stops": [
            (0.00, "#000000", "Min"),
            (0.20, "#333333", ""),
            (0.40, "#666666", ""),
            (0.60, "#999999", ""),
            (0.80, "#cccccc", ""),
            (1.00, "#ffffff", "Max"),
        ],
    },
    "penetration": {
        "name": "RdYlBu_r",
        "stops": [
            (0.00, "#313695", "Min"),
            (0.20, "#4575b4", ""),
            (0.40, "#74add1", ""),
            (0.60, "#fee090", ""),
            (0.80, "#f46d43", ""),
            (1.00, "#a50026", "Max"),
        ],
    },
    "intensity": {
        "name": "Greys",
        "stops": [
            (0.00, "#000000", "Min"),
            (0.20, "#333333", ""),
            (0.40, "#666666", ""),
            (0.60, "#999999", ""),
            (0.80, "#cccccc", ""),
            (1.00, "#ffffff", "Max"),
        ],
    },
    "coverage": {
        "name": "Blues",
        "stops": [
            (0.00, "#f7fbff", "Min"),
            (0.20, "#cfe1f2", ""),
            (0.40, "#93c4de", ""),
            (0.60, "#4a97c9", ""),
            (0.80, "#1764ab", ""),
            (1.00, "#08306b", "Max"),
        ],
    },
    "slope": {
        "name": "YlOrRd",
        "stops": [
            (0.00, "#ffffcc", "Min"),
            (0.20, "#fee186", ""),
            (0.40, "#fdaa48", ""),
            (0.60, "#fc5a2d", ""),
            (0.80, "#d30f20", ""),
            (1.00, "#800026", "Max"),
        ],
    },
    "aspect": {
        "name": "HSV",
        "stops": [
            (0.00, "#ff0000", "N (0°)"),
            (0.125, "#ffff00", "NE (45°)"),
            (0.25, "#00ff00", "E (90°)"),
            (0.375, "#00ffff", "SE (135°)"),
            (0.50, "#0000ff", "S (180°)"),
            (0.625, "#ff00ff", "SW (225°)"),
            (0.75, "#ff0000", "W (270°)"),
            (0.875, "#ff8080", "NW (315°)"),
            (1.00, "#ff0000", "N (360°)"),
        ],
    },
    "tri": {
        "name": "Oranges",
        "stops": [
            (0.00, "#fff5eb", "Min"),
            (0.20, "#fdd8b3", ""),
            (0.40, "#fda761", ""),
            (0.60, "#f3701b", ""),
            (0.80, "#c44001", ""),
            (1.00, "#7f2704", "Max"),
        ],
    },
    "twi": {
        "name": "RdBu",
        "stops": [
            (0.00, "#67001f", "Dry"),
            (0.20, "#d7191c", ""),
            (0.40, "#fdae61", ""),
            (0.60, "#abd9e9", ""),
            (0.80, "#2166ac", ""),
            (1.00, "#053061", "Wet"),
        ],
    },
    "vertical": {
        "name": "Viridis",
        "stops": [
            (0.00, "#440154", "Low"),
            (0.20, "#3b528b", ""),
            (0.40, "#21918c", ""),
            (0.60, "#5ec962", ""),
            (0.80, "#b8de29", ""),
            (1.00, "#fde725", "High"),
        ],
    },
    "gaps": {
        # Explicit-spec only: no filename pattern selects it (see PATTERNS).
        "name": "Greens",
        "stops": [
            (0.00, "#f7fcf5", "Min"),
            (0.20, "#d9f0d3", ""),
            (0.40, "#addd8e", ""),
            (0.60, "#78c679", ""),
            (0.80, "#31a354", ""),
            (1.00, "#006d2c", "Max"),
        ],
    },
    "echo_ratio": {
        "name": "YlGnBu (Echo Ratio)",
        "stops": [
            (0.00, "#ffffcc", "Hard surface"),
            (0.20, "#a1dab4", ""),
            (0.40, "#41b6c4", ""),
            (0.60, "#2c7fb8", ""),
            (0.80, "#253494", ""),
            (1.00, "#081d58", "Complex canopy"),
        ],
    },
    "density": {
        "name": "Blues (Density)",
        "stops": [
            (0.00, "#f7fbff", "Sparse"),
            (0.20, "#c6dbef", ""),
            (0.40, "#6baed6", ""),
            (0.60, "#3182bd", ""),
            (0.80, "#08519c", ""),
            (1.00, "#08306b", "Dense"),
        ],
    },
    "ground_density": {
        "name": "Greens (Ground)",
        "stops": [
            (0.00, "#f7fcf5", "Few points"),
            (0.20, "#d9f0d3", ""),
            (0.40, "#addd8e", ""),
            (0.60, "#78c679", ""),
            (0.80, "#31a354", ""),
            (1.00, "#006d2c", "Many points"),
        ],
    },
    "ground_std": {
        "name": "OrRd (Z Std)",
        "stops": [
            (0.00, "#fff5eb", "Flat"),
            (0.20, "#fdd8b3", ""),
            (0.40, "#fda761", ""),
            (0.60, "#f3701b", ""),
            (0.80, "#c44001", ""),
            (1.00, "#7f2704", "Rough"),
        ],
    },
    "ndvi": {
        "name": "RdYlGn (NDVI)",
        "stops": [
            (0.00, "#a50026", "-1.0"),
            (0.15, "#d73027", "-0.7"),
            (0.35, "#fc8d59", "-0.3"),
            (0.50, "#ffffbf", "0.0"),
            (0.65, "#a6d96a", "+0.3"),
            (0.85, "#1a9850", "+0.7"),
            (1.00, "#006837", "+1.0"),
        ],
    },
    "ndgi": {
        "name": "RdYlGn (NDGI)",
        "stops": [
            (0.00, "#d73027", "No vegetation (-0.5)"),
            (0.15, "#fc8d59", ""),
            (0.35, "#fee08b", ""),
            (0.50, "#ffffbf", "Neutral (0.0)"),
            (0.65, "#a6d96a", ""),
            (0.85, "#1a9850", ""),
            (1.00, "#006837", "Vegetation (+0.5)"),
        ],
    },
    "classes": {
        "name": "Classes",
        "discrete": True,
        "stops": [
            (0, "#d7191c", "Class 0"),
            (1, "#fdae61", "Class 1"),
            (2, "#ffffbf", "Class 2"),
            (3, "#a6d96a", "Class 3"),
            (4, "#1a9641", "Class 4"),
            (5, "#4575b4", "Class 5"),
            (6, "#f46d43", "Class 6"),
            (7, "#74add1", "Class 7"),
        ],
    },
    "generic": {
        "name": "Turbo",
        "stops": [
            (0.00, "#30123b", "Min"),
            (0.20, "#4675ed", ""),
            (0.40, "#1bcfd4", ""),
            (0.60, "#61fc6c", ""),
            (0.80, "#d1e834", ""),
            (1.00, "#fda01f", "Max"),
        ],
    },
}

PATTERNS: list[tuple[str, str]] = [
    ("dtm", "dtm"),
    ("dsm", "dsm"),
    ("chm", "chm"),
    ("hillshade", "hillshade"),
    ("canopy_penetration", "penetration"),
    ("intensity", "intensity"),
    ("coverage", "coverage"),
    ("slope", "slope"),
    ("aspect", "aspect"),
    ("tri.tif", "tri"),
    ("twi.tif", "twi"),
    ("vertical_profile", "vertical"),
    ("ndgi", "ndgi"),
    ("echo_ratio", "echo_ratio"),
    ("density.tif", "density"),
    ("ground_density", "ground_density"),
    ("ground_std", "ground_std"),
]


# ---------------------------------------------------------------------------
# Rasters
# ---------------------------------------------------------------------------

def detect_ramp(filename: str) -> Optional[str]:
    name_lower = filename.lower()
    for pattern, ramp_key in PATTERNS:
        if pattern in name_lower:
            return ramp_key
    return None


def _read_raster_stats(
    raster_path: Path,
) -> Optional[tuple[float, float, Optional[float]]]:
    try:
        import numpy as np
        import rasterio

        with rasterio.open(str(raster_path)) as src:
            band = src.read(1)
            nodata = src.nodata
            if nodata is not None:
                valid = band[band != nodata]
            else:
                valid = band[~np.isnan(band)]
            if valid.size == 0:
                logger.warning(f"No valid data in: {raster_path.name}")
                return None
            return float(valid.min()), float(valid.max()), nodata
    except Exception as e:
        logger.error(f"Error reading raster {raster_path.name}: {e}")
        return None


def _looks_discrete(raster_path: Path) -> bool:
    """Return whether a raster looks categorical: few integer classes in 0..9.

    Used only for generic fallback styling: a small integer set is likely a
    mask or class raster rather than a continuous surface.
    """
    try:
        import numpy as np
        import rasterio

        with rasterio.open(str(raster_path)) as src:
            band = src.read(1)
            nodata = src.nodata
        if nodata is not None:
            valid = band[band != nodata]
        else:
            valid = band[~np.isnan(band)]
        if valid.size == 0:
            return False
        values = np.unique(valid)
        if not np.allclose(values, values.astype(int)):
            return False
        classes = values.astype(int)
        return 2 <= classes.size <= 8 and classes.min() >= 0 and classes.max() < 10
    except Exception:
        return False


def _build_qml(
    raster_path: Path,
    entry: dict,
    min_val: float,
    max_val: float,
    reversed_: bool = False,
) -> str:
    stops = entry["stops"]
    if reversed_:
        stops = [(1.0 - pos, color, label) for pos, color, label in reversed(stops)]
    stops_list = [f"{norm};{color}" for norm, color, _ in stops]
    stops_str = ";".join(stops_list)
    color1 = stops[0][1]
    color2 = stops[-1][1]

    items_xml = []
    for norm_pos, hex_color, label in stops:
        val = min_val + norm_pos * (max_val - min_val)
        display = label if label else f"{val:.2f}"
        items_xml.append(
            f'<item value="{val:.6f}" label="{saxutils.escape(display)}" '
            f'color="{hex_color}" alpha="255"/>'
        )
    items_str = "\n          ".join(items_xml)

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34.0" styleCategories="Rendering|LayerConfiguration" minScale="1e+08" maxScale="0">
  <pipe>
    <rasterrenderer type="singlebandpseudocolor" band="1" opacity="1" alphaBand="-1" classificationMax="{max_val:.6f}" classificationMin="{min_val:.6f}" nodataColor="" useContinuousLegend="1">
      <rasterTransparency/>
      <minMaxOrigin>
        <limits>MinMax</limits>
        <extent>WholeRaster</extent>
        <statAccuracy>Exact</statAccuracy>
        <cumulativeCutLower>0.02</cumulativeCutLower>
        <cumulativeCutUpper>0.98</cumulativeCutUpper>
        <stdDevFactor>2</stdDevFactor>
      </minMaxOrigin>
      <rastershader>
        <colorrampshader minimumValue="{min_val:.6f}" maximumValue="{max_val:.6f}" classificationMode="2" clip="0" colorRampType="INTERPOLATED" labelPrecision="4">
          <colorramp type="gradient" name="{saxutils.escape(entry['name'])}">
            <Option type="Map">
              <Option type="QString" name="color1" value="{color1}"/>
              <Option type="QString" name="color2" value="{color2}"/>
              <Option type="QString" name="stops" value="{stops_str}"/>
            </Option>
            <prop k="color1" v="{color1}"/>
            <prop k="color2" v="{color2}"/>
            <prop k="discrete" v="0"/>
            <prop k="rampType" v="gradient"/>
            <prop k="stops" v="{stops_str}"/>
          </colorramp>
          {items_str}
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
  </pipe>
</qgis>"""


def _build_discrete_qml(raster_path: Path, entry: dict) -> str:
    """Build a paletted QML with one named entry per class.

    QGIS displays the resulting class labels in the layer tree.
    """
    entries_xml = []
    for value, hex_color, label in entry["stops"]:
        entries_xml.append(
            f'      <paletteEntry alpha="255" color="{hex_color}" '
            f'label="{saxutils.escape(label)}" value="{int(value)}"/>'
        )
    entries_str = "\n".join(entries_xml)

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34.0" styleCategories="Rendering|LayerConfiguration" minScale="1e+08" maxScale="0">
  <pipe>
    <rasterrenderer type="paletted" band="1" opacity="1" alphaBand="-1">
      <rasterTransparency/>
      <minMaxOrigin>
        <limits>MinMax</limits>
        <extent>WholeRaster</extent>
        <statAccuracy>Exact</statAccuracy>
        <cumulativeCutLower>0.02</cumulativeCutLower>
        <cumulativeCutUpper>0.98</cumulativeCutUpper>
        <stdDevFactor>2</stdDevFactor>
      </minMaxOrigin>
      <colorPalette>
{entries_str}
      </colorPalette>
    </rasterrenderer>
  </pipe>
</qgis>"""


def generate_discrete_qml(
    raster_path: str,
    entries,
    name: str,
) -> bool:
    """Write a paletted ``.qml`` with one named entry per class.

    This public helper lets a barrier style dynamic run classes. The node
    declares ``{"self_styled": True}`` so the controller does not overwrite it.
    Return True only when the sidecar is written.
    """
    rpath = Path(raster_path).resolve()
    if not rpath.exists():
        logger.warning(f"Raster not found: {rpath}")
        return False
    if rpath.suffix.lower() not in _RASTER_EXT:
        return False
    entry = {"name": name, "discrete": True, "stops": list(entries)}
    try:
        qml_content = _build_discrete_qml(rpath, entry)
    except Exception as e:  # Styling must never fail the pipeline.
        logger.error(f"Error generating QML for {rpath.name}: {e}")
        return False
    qml_path = rpath.with_suffix(".qml")
    qml_path.write_text(qml_content, encoding="utf-8")
    logger.info(f"QML style: {qml_path.name} ({name}, {len(entry['stops'])} classes)")
    return True


def generate_qml(
    raster_path: str,
    ramp_key: Optional[str] = None,
    reversed_: bool = False,
    entry: Optional[dict] = None,
) -> bool:
    """Generate a raster ``<file>.qml`` sidecar.

    ``entry`` is the node-normalized style (``name``/``stops``/``discrete``
    optional) declared by the node; ``ramp_key`` references the core
    catalog. Missing styles use filename conventions and generic fallbacks.
    ``reversed_=True`` reverses a continuous ramp. Return True only if written.
    """
    rpath = Path(raster_path).resolve()
    if not rpath.exists():
        logger.warning(f"Raster not found: {rpath}")
        return False
    if rpath.suffix.lower() not in _RASTER_EXT:
        return False

    if entry is None or not entry:
        if ramp_key is None:
            ramp_key = detect_ramp(rpath.name)
        if not ramp_key or ramp_key not in RAMPS:
            ramp_key = DEFAULT_RAMP
            if _looks_discrete(rpath):
                ramp_key = "classes"
            logger.info(f"Generic style for: {rpath.name} ({ramp_key})")
        entry = RAMPS[ramp_key]
    else:
        ramp_key = None

    stats = _read_raster_stats(rpath)
    if stats is None:
        return False
    min_val, max_val, _ = stats

    if entry.get("discrete"):
        try:
            qml_content = _build_discrete_qml(rpath, entry)
        except Exception as e:
            logger.error(f"Error generating QML for {rpath.name}: {e}")
            return False
        qml_path = rpath.with_suffix(".qml")
        qml_path.write_text(qml_content, encoding="utf-8")
        logger.info(f"QML style: {qml_path.name} ({entry.get('name', 'discrete')})")
        return True

    if min_val == max_val:
        # Flat raster: widen symmetrically so the ramp has a range.
        min_val -= 1.0
        max_val += 1.0

    # The catalog-ramp clamps below apply only to catalog ramps (custom
    # entries bring their own min/max, applied right after and winning).

    if ramp_key == "chm":
        min_val = 0.0
    elif ramp_key == "penetration":
        min_val = 0.0
        max_val = 1.0
    elif ramp_key == "hillshade":
        min_val = 0.0
        max_val = 255.0
    elif ramp_key == "coverage":
        min_val = 0.0
        max_val = max(max_val, 5.0)
    elif ramp_key == "aspect":
        min_val = 0.0
        max_val = 360.0
    elif ramp_key == "vertical":
        min_val = 0.0
    elif ramp_key == "echo_ratio":
        min_val = 0.0
        max_val = 1.0
    elif ramp_key == "density":
        min_val = 0.0
        max_val = max(max_val, 1.0)
    elif ramp_key == "ground_density":
        min_val = 0.0
        max_val = max(max_val, 1.0)
    elif ramp_key == "ground_std":
        min_val = 0.0
        max_val = max(max_val, 1.0)
    elif ramp_key == "ndgi":
        min_val = -0.5
        max_val = 0.5

    if entry.get("min") is not None:
        min_val = float(entry["min"])
    if entry.get("max") is not None:
        max_val = float(entry["max"])

    try:
        qml_content = _build_qml(rpath, entry, min_val, max_val, reversed_)
    except Exception as e:
        logger.error(f"Error generating QML for {rpath.name}: {e}")
        return False

    qml_path = rpath.with_suffix(".qml")
    qml_path.write_text(qml_content, encoding="utf-8")
    logger.info(
        f"QML style: {qml_path.name} ({entry.get('name', ramp_key)}, "
        f"{min_val:.1f}–{max_val:.1f})"
    )
    return True


# ---------------------------------------------------------------------------
# Vector products.
# ---------------------------------------------------------------------------

def _read_vector_table(path: Path):
    """Read a complete GeoPackage/GeoJSON attribute table without PROJ."""
    from lynceus.processing.vector_table import read_vector_features

    return read_vector_features(path, with_geometry=False)


def _column_stats(features, field: str) -> Optional[tuple[float, float, int]]:
    """Return finite column min/max/count excluding NODATA."""
    import numpy as np

    values = features.columns.get(field)
    if values is None:
        return None
    try:
        vals = values.astype(float)
    except (TypeError, ValueError):
        try:
            vals = np.array([float(v) for v in values], dtype=float)
        except (TypeError, ValueError):
            return None
    valid = vals[np.isfinite(vals) & (vals != VECTOR_NODATA)]
    if valid.size == 0:
        return None
    return float(valid.min()), float(valid.max()), int(valid.size)


def _meta_sidecar(path: Path) -> dict:
    """Read a product's ``<file>.meta.json`` sidecar.

    The producing node declares computed metrics and bookkeeping columns; the
    core consumes them generically without domain-specific column names.
    """
    try:
        import json

        return json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _default_vector_field(
    features,
    planned: Optional[list] = None,
    bk_cols: Optional[list] = None,
) -> Optional[str]:
    """Return the first useful numeric column for generic styling.

    Prefer node-declared metrics, exclude bookkeeping columns and geometry,
    then fall back to the first remaining numeric column.
    """
    excluded = set(BOOKKEEPING_COLS) | set(bk_cols or ())
    numeric_cols = [
        c for c in features.fields
        if c not in excluded and c in features.numeric
    ]
    if planned:
        for c in planned:
            if c in numeric_cols:
                return c
    return numeric_cols[0] if numeric_cols else None


def _generic_classes(features, field: str):
    """5 graduated classes from the real range of the column."""
    stats = _column_stats(features, field)
    if stats is None:
        return None
    vmin, vmax, _ = stats
    if vmax <= vmin:
        vmax = vmin + 1.0
    colors = ["#d73027", "#fc8d59", "#ffffbf", "#a6d96a", "#1a9641"]
    classes = []
    for i in range(5):
        lo = vmin + i * (vmax - vmin) / 5.0
        hi = vmin + (i + 1) * (vmax - vmin) / 5.0
        classes.append((lo, hi, colors[i], f"{lo:.1f} \u2013 {hi:.1f}"))
    return classes


def _vector_geometry_kind(vector_path: str) -> str | None:
    """Layer geometry kind from metadata, without scanning features.

    Returns ``"point"``, ``"line"`` or ``"polygon"`` (None when unknown):
    GPKG declares one type per layer in ``gpkg_geometry_columns``;
    GeoJSON takes the first feature geometry from a bounded head read.
    Unknown kinds keep the legacy fill symbols.
    """
    suffix = Path(vector_path).suffix.lower()
    type_name: str | None = None
    if suffix == ".gpkg":
        import sqlite3

        try:
            con = sqlite3.connect(f"file:{vector_path}?mode=ro", uri=True)
            try:
                row = con.execute(
                    "SELECT geometry_type_name FROM gpkg_geometry_columns"
                    " LIMIT 1"
                ).fetchone()
            finally:
                con.close()
        except Exception:
            return None
        if row and row[0]:
            type_name = str(row[0])
    elif suffix == ".geojson":
        import re

        try:
            with open(vector_path, encoding="utf-8") as handle:
                head = handle.read(131072)
        except OSError:
            return None
        match = re.search(
            r'"geometry"\s*:\s*\{\s*"type"\s*:\s*"(\w+)"', head)
        if match:
            type_name = match.group(1)
    if not type_name:
        return None
    name = type_name.upper()
    if "POINT" in name:
        return "point"
    if "LINE" in name or "CURVE" in name or "STRING" in name:
        return "line"
    if "POLYGON" in name or "SURFACE" in name:
        return "polygon"
    return None


def _symbol_xml(index: int, color: str, geom_kind: str) -> str:
    """One graduated-class symbol for the layer geometry kind."""
    if geom_kind == "point":
        return (
            f'      <symbol alpha="1" force_rhr="0" type="marker" name="{index}">'
            f'<layer pass="0" class="SimpleMarker" locked="0">'
            f'<prop k="color" v="{color}"/>'
            f'<prop k="name" v="circle"/>'
            f'<prop k="size" v="2.5"/>'
            f'<prop k="size_unit" v="MM"/>'
            f'<prop k="outline_style" v="no"/>'
            f'<prop k="outline_color" v="#000000"/>'
            f'<prop k="outline_width" v="0"/>'
            f'</layer></symbol>'
        )
    if geom_kind == "line":
        return (
            f'      <symbol alpha="1" force_rhr="0" type="line" name="{index}">'
            f'<layer pass="0" class="SimpleLine" locked="0">'
            f'<prop k="line_color" v="{color}"/>'
            f'<prop k="line_width" v="0.6"/>'
            f'<prop k="line_width_unit" v="MM"/>'
            f'</layer></symbol>'
        )
    return (
        f'      <symbol alpha="1" force_rhr="0" type="fill" name="{index}">'
        f'<layer pass="0" class="SimpleFill" locked="0">'
        f'<prop k="color" v="{color}"/>'
        # No outline: dense grids moiré in QGIS with thousands of
        # stroked cells (same reason the 2D viewer paints fill-only).
        f'<prop k="outline_style" v="no"/>'
        f'<prop k="outline_color" v="#000000"/>'
        f'<prop k="outline_width" v="0"/>'
        f'</layer></symbol>'
    )


def _build_vector_qml(field: str, classes, name: str,
                      geom_kind: str = "polygon") -> str:
    ranges_xml = []
    symbols_xml = []
    for i, (_lo, _hi, color, label) in enumerate(classes):
        ranges_xml.append(
            f'        <range symbol="{i}" lower="{_lo:.1f}" '
            f'upper="{_hi:.1f}" label="{saxutils.escape(label)}"/>'
        )
        symbols_xml.append(_symbol_xml(i, color, geom_kind))
    ranges_str = "\n".join(ranges_xml)
    symbols_str = "\n".join(symbols_xml)
    color1 = classes[0][2]
    color2 = classes[-1][2]

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<qgis version="3.34.0" styleCategories="Symbology">
  <renderer-v2 type="graduatedSymbol" attr="{saxutils.escape(field)}" graduatedMethod="GraduatedColor">
    <ranges>
{ranges_str}
    </ranges>
    <symbols>
{symbols_str}
    </symbols>
    <colorramp type="gradient" name="{saxutils.escape(name)}">
      <prop k="color1" v="{color1}"/>
      <prop k="color2" v="{color2}"/>
      <prop k="discrete" v="0"/>
      <prop k="rampType" v="gradient"/>
    </colorramp>
    <rotation/>
    <sizescale/>
  </renderer-v2>
</qgis>"""


def style_vector_file(
    vector_path: str,
    field: Optional[str] = None,
    classes: Optional[list] = None,
    name: Optional[str] = None,
) -> bool:
    """Generate a node-curated or generic graduated vector ``.qml``.

    ``field``, ``classes``, and ``name`` may come from the node. Missing
    classes are computed from the selected field's real range. ``field``
    is required when ``classes`` is given (caught early instead of
    failing inside the XML builder). Symbol types follow the layer
    geometry (marker/line/fill); unknown kinds keep legacy fills.
    """
    vpath = Path(vector_path).resolve()
    if not vpath.exists():
        logger.warning(f"Vector file not found: {vpath}")
        return False
    if vpath.suffix.lower() not in _VECTOR_EXT:
        return False
    if classes is not None and field is None:
        logger.warning(f"Vector style for {vpath.name} needs a field with classes")
        return False
    geom_kind = _vector_geometry_kind(str(vpath)) or "polygon"

    stem = vpath.stem
    style_name = (name or f"{stem} \u2014 {field}") if field else stem

    if classes is None:
        try:
            features = _read_vector_table(vpath)
        except Exception as exc:
            logger.warning(f"Cannot read {vpath.name}: {exc}")
            return False
        if field is None:
            meta = _meta_sidecar(vpath)
            field = _default_vector_field(
                features,
                planned=meta.get("metrics_computed"),
                bk_cols=meta.get("bookkeeping_cols"),
            )
            if field is None:
                return False
        classes = _generic_classes(features, field)
        if classes is None:
            return False
        style_name = (name or f"{stem} \u2014 {field}") if field else stem

    try:
        qml_content = _build_vector_qml(field, classes, style_name,
                                        geom_kind)
    except Exception as e:
        logger.error(f"Error generating vector QML for {vpath.name}: {e}")
        return False

    qml_path = vpath.with_suffix(".qml")
    qml_path.write_text(qml_content, encoding="utf-8")
    logger.info(f"Vector QML style: {qml_path.name} (field={field})")
    return True


# ---------------------------------------------------------------------------
# Per-run orchestration
# ---------------------------------------------------------------------------

# Public aliases of the private extension sets (used by viewers/tests).
RASTER_EXT = _RASTER_EXT
VECTOR_EXT = _VECTOR_EXT


def generate_for_file(path: str, spec=None) -> bool:
    """Generates the ``.qml`` of a product according to its type.

    ``spec`` is what the node declares: a core catalog ramp id (raster), a
    field name (vector) or a self-contained ``dict``:
    ``{"ramp": <id>, "reversed": bool}`` / ``{"stops": [...], "discrete":
    bool, "name": ..., "min": ..., "max": ...}`` for rasters and
    ``{"field": ..., "classes": [...], "name": ...}`` for vectors. Without a
    spec it resolves by convention from the file name and, for rasters without
    a known ramp, falls back to the generic one.
    """
    ext = Path(path).suffix.lower()

    if isinstance(spec, dict) and spec.get("self_styled"):
        # The node writes its own .qml in its barrier (run-dependent classes,
        # e.g. Height Strata): the engine neither generates nor overwrites.
        return False

    if isinstance(spec, dict):
        if "ramp" in spec:
            ramp_key = spec["ramp"] if spec["ramp"] in RAMPS else None
            return generate_qml(
                path,
                ramp_key=ramp_key,
                reversed_=bool(spec.get("reversed")),
            )
        if "stops" in spec or "discrete" in spec:
            entry = {
                "name": str(spec.get("name") or Path(path).stem),
                "stops": spec["stops"],
                "discrete": bool(spec.get("discrete", False)),
            }
            if spec.get("min") is not None:
                entry["min"] = spec["min"]
            if spec.get("max") is not None:
                entry["max"] = spec["max"]
            return generate_qml(path, entry=entry)
        if "field" in spec:
            return style_vector_file(
                path,
                field=spec["field"],
                classes=spec.get("classes"),
                name=spec.get("name"),
            )
        return False

    if ext in _RASTER_EXT:
        ramp = spec if isinstance(spec, str) and spec in RAMPS else None
        return generate_qml(path, ramp_key=ramp)
    if ext in _VECTOR_EXT:
        field = spec if isinstance(spec, str) else None
        return style_vector_file(path, field=field)
    return False


def read_palette_entries(qml_path: str) -> Optional[list[tuple[int, str, str]]]:
    """Reads the ``<paletteEntry>`` entries of a generated paletted ``.qml``.

    Returns ``(value, color, label)`` per class — the same source of truth as
    QGIS and the app 2D viewer, with the styles each node declared. None if the
    qml does not exist, is not paletted or cannot be parsed (pure domain,
    no Qt).
    """
    try:
        import xml.etree.ElementTree as ET

        root = ET.parse(qml_path).getroot()
    except (ET.ParseError, OSError, ValueError):
        return None
    if root.find(".//renderer-paletted") is None and root.find(
        ".//rasterrenderer[@type='paletted']"
    ) is None:
        # The hyphenated tag only exists in foreign QMLs; ours uses
        # rasterrenderer type='paletted'. Both accepted here.
        return None
    entries = []
    for el in root.iter("paletteEntry"):
        try:
            entries.append(
                (int(el.get("value")), str(el.get("color")), str(el.get("label") or ""))
            )
        except (TypeError, ValueError):
            continue
    return entries or None


def style_outputs(outputs: dict, qml_specs: dict[str, dict]) -> int:
    """Generates the ``.qml`` files for the primary products of a run.

    ``outputs`` is the payload dict {iid: payload}; ``qml_specs`` is
    {iid: PROCESSING_SPECS["qml"]} (basename -> spec mapping) resolved with
    ``discover_node_capabilities``. Only string-valued payload entries with
    a raster/vector extension are considered (one primary ``.qml`` per
    product; tile lists and metadata never styled). Returns how many
    ``.qml`` files were written. Best-effort: never raises.
    """
    written = 0
    for iid, payload in outputs.items():
        if not isinstance(payload, dict):
            continue
        spec_table = qml_specs.get(iid) or {}
        for value in payload.values():
            if not isinstance(value, str) or not value:
                continue
            ext = Path(value).suffix.lower()
            if ext not in _RASTER_EXT and ext not in _VECTOR_EXT:
                continue
            spec = spec_table.get(Path(value).name)
            try:
                if generate_for_file(value, spec):
                    written += 1
            except Exception as exc:  # styling never breaks the run
                logger.warning("QML skip %s: %s", Path(value).name, exc)
    return written