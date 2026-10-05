# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Node data contracts for the Qt-free domain layer.

Defines port types and their compatibility matrix. The UI and pipeline DAG
share this contract so valid connections remain consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Set


class PortType(Enum):
    """Data types flowing between nodes and defining connection contracts."""

    # Sources.
    TILES = "tiles"                          # Virtual tile list after tiling.

    # Per-tile point clouds.
    POINT_CLOUD = "point_cloud"              # Unclassified LAZ tile.
    CLASSIFIED_POINT_CLOUD = "classified_point_cloud"  # Classified LAZ tile.

    # Per-tile rasters.
    DTM_TILE = "dtm_tile"
    DSM_TILE = "dsm_tile"

    # Mosaics produced by barrier tasks.
    DTM_MOSAIC = "dtm_mosaic"
    DSM_MOSAIC = "dsm_mosaic"
    CHM_MOSAIC = "chm_mosaic"

    # Metrics and analysis.
    GRID_METRICS = "grid_metrics"
    GRID_VEGETATION = "grid_vegetation"  # Per-cell vegetation stats table.

    # Analysis rasters.
    CANOPY_PENETRATION = "canopy_penetration"  # Ground/total ratio per cell.
    COVERAGE_RASTER = "coverage_raster"        # Unique flight lines per cell.
    INTENSITY_ORTHO = "intensity_ortho"        # Mean intensity per cell.

    # Post-mosaic analysis products.
    STRATA_RASTER = "strata_raster"          # Categorical vertical strata raster.
    VECTOR = "vector"                        # GeoPackage geometry product.
    TABLE_CSV = "table_csv"                  # CSV statistics table.
    TABLE_GPKG = "table_gpkg"                # GeoPackage table.
    TABLE = "table"                          # Generic CSV or GeoPackage table.
    RASTER = "raster"                        # Generic combined raster.


# ---------------------------------------------------------------------------
# Canonical port registry for the Qt-free domain layer. IDs are strings; the
# PortType enum is a convenience alias for built-in port types.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PortSpec:
    """Specification for one port type.

    ``compatible`` contains input IDs that this output can feed. ``viewer_kind``
    optionally identifies a generic viewer for the product.
    """

    port_id: str
    display_name: str
    color: str
    compatible: tuple[str, ...]
    viewer_kind: str | None = None


@dataclass(frozen=True, slots=True)
class PortDef:
    """Declaration of a node input or output port.

    ``required`` and ``group`` are enforced by the canvas/controller contract;
    a group allows mutually exclusive input alternatives.
    """

    port_type: PortType | str
    name: str | None = None
    required: bool = True
    group: str | None = None


_PORT_SPECS: Dict[str, PortSpec] = {}


def register_spec(spec: PortSpec) -> None:
    """Register or replace a port specification."""
    _PORT_SPECS[spec.port_id] = spec


def get_spec(port_id: str) -> PortSpec | None:
    return _PORT_SPECS.get(port_id)


def all_specs() -> tuple[PortSpec, ...]:
    return tuple(_PORT_SPECS.values())


def _normalize_id(port) -> str:
    """Normalize a PortType or string to its canonical string ID."""
    if isinstance(port, PortType):
        return port.value
    return str(port)


def port_metadata(item) -> PortDef:
    """Normalize an INPUTS/OUTPUTS declaration to PortDef.

    Accepted forms are PortType, string, ``(type, label)``, and PortDef.
    """
    if isinstance(item, PortDef):
        return item
    if isinstance(item, tuple) and len(item) == 2:
        return PortDef(port_type=item[0], name=item[1])
    return PortDef(port_type=item)


def _seed_builtin_specs() -> None:
    """Seed built-in port specifications from the canonical constants."""
    display_names: Dict[str, str] = {
        "tiles": "Tiles",
        "point_cloud": "Point Cloud",
        "classified_point_cloud": "Classified Point Cloud",
        "dtm_tile": "DTM Tile",
        "dsm_tile": "DSM Tile",
        "dtm_mosaic": "DTM Mosaic",
        "dsm_mosaic": "DSM Mosaic",
        "chm_mosaic": "CHM Mosaic",
        "grid_metrics": "Grid Metrics",
        "grid_vegetation": "Grid Vegetation",
        "canopy_penetration": "Canopy Penetration",
        "coverage_raster": "Flightline Coverage",
        "intensity_ortho": "Intensity Orthophoto",
        "strata_raster": "Strata Raster",
        "vector": "Vector",
        "table_csv": "Table (CSV)",
        "table_gpkg": "Table (GeoPackage)",
        "table": "Table",
        "raster": "Raster",
    }
    colors: Dict[str, str] = {
        "tiles": "#8496ab",
        "point_cloud": "#e8a33d",
        "classified_point_cloud": "#e8a33d",
        "dtm_tile": "#4caf6d",
        "dsm_tile": "#2fbfa8",
        "dtm_mosaic": "#4caf6d",
        "dsm_mosaic": "#2fbfa8",
        "chm_mosaic": "#a4d43d",
        "grid_metrics": "#9c6bd6",
        "grid_vegetation": "#3fae6a",
        "canopy_penetration": "#2e8b57",
        "coverage_raster": "#e07b39",
        "intensity_ortho": "#a05bb5",
        "strata_raster": "#d66bb0",
        "vector": "#4a90d9",
        "table_csv": "#c2956b",
        "table_gpkg": "#bf8b5a",
        "table": "#b98c6b",
        "raster": "#7cc7a8",
    }
    compatibility: Dict[str, Set[str]] = {
        "tiles": {"tiles"},
        "point_cloud": {"point_cloud", "classified_point_cloud"},
        "classified_point_cloud": {"classified_point_cloud", "point_cloud"},
        "dtm_tile": {"dtm_tile"},
        "dsm_tile": {"dsm_tile"},
        "dtm_mosaic": {"dtm_mosaic"},
        "dsm_mosaic": {"dsm_mosaic"},
        "chm_mosaic": {"chm_mosaic"},
        "grid_metrics": {"grid_metrics"},
        "grid_vegetation": {"grid_vegetation"},
        "canopy_penetration": {"canopy_penetration"},
        "coverage_raster": {"coverage_raster"},
        "intensity_ortho": {"intensity_ortho"},
        "strata_raster": {"strata_raster"},
        "vector": {"vector"},
        "table_csv": {"table_csv", "table"},
        "table_gpkg": {"table_gpkg", "table"},
        "table": {"table"},
        "raster": {"raster"},
    }
    viewer_kinds: Dict[str, str] = {
        "canopy_penetration": "raster",
        "coverage_raster": "raster",
        "intensity_ortho": "raster",
        "raster": "raster",
    }
    for port_id, display_name in display_names.items():
        register_spec(
            PortSpec(
                port_id=port_id,
                display_name=display_name,
                color=colors[port_id],
                compatible=tuple(compatibility.get(port_id, ())),
                viewer_kind=viewer_kinds.get(port_id),
            )
        )


_seed_builtin_specs()


def can_connect(output_port: PortType | str, input_port: PortType | str) -> bool:
    """Return whether an output port can connect to an input port."""
    out_id = _normalize_id(output_port)
    in_id = _normalize_id(input_port)
    spec = get_spec(out_id)
    if spec is not None:
        return in_id in spec.compatible
    return False


def get_display_name(port: PortType | str) -> str:
    """Return the display name for a port type."""
    port_id = _normalize_id(port)
    spec = get_spec(port_id)
    if spec is not None:
        return spec.display_name
    return port_id.replace("_", " ").title()


def get_port_color(port: PortType | str) -> str:
    """Return the hex color associated with a port type."""
    port_id = _normalize_id(port)
    spec = get_spec(port_id)
    if spec is not None:
        return spec.color
    return "#888888"
