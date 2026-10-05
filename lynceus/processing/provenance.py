# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Provenance metadata stamped into every pipeline product (no Qt).

Each session product (mosaic, import, CSV, GeoPackage, exported LAZ) and
every intermediate per-tile LAZ carries a JSON document identifying the
system that generated it, the node that produced it (with its author,
license and resolved parameters), the session, and the sources. (Per-tile
rasters are not individually stamped; only their merged mosaics are.) The
document always includes two mandatory pieces of the system contract:

- **system identity**: ``system``, ``system_version``, ``license``,
  ``copyright`` -- never removable, not editable by the pipeline operator;
- **system disclaimer**: generic liability text (product as-is, no warranty
  or damages attributable to the developer, responsibility rests with whoever
  shares/distributes or decides from it).

A node's creator may additionally declare ``NODE_DISCLAIMER`` (or
``disclaimer`` in a pack manifest), embedded as ``node_disclaimer`` when
present -- e.g. a node whose algorithm was generated with AI assistance can
state that precision is unverified and that no liability for inaccuracy is
accepted. That is informational for product recipients; legal acceptance of
extensions is gated separately by the EULA (lynceus/plugins/store.py).

Format support:

- **GeoTIFF**: rasterio dataset tags (default domain).
- **LAS/LAZ**: ``system_identifier`` / ``generating_software`` header fields
  plus a user VLR holding the JSON document.
- **GeoPackage**: ``gpkg_metadata`` / ``gpkg_metadata_reference`` rows.
- **CSV / other**: a ``<file>.meta.json`` sidecar (merged with any existing
  node metadata such as ``metrics_computed``).

Embedding is best-effort: a failed stamp never fails the product write.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lynceus import __version__
from lynceus.plugins.registry import manager

SCHEMA = "lynceus-provenance"
SCHEMA_VERSION = 1

SYSTEM_NAME = "LynceusScan"
SYSTEM_LICENSE = "GPL-3.0-or-later"
SYSTEM_COPYRIGHT = "Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>"
SYSTEM_DISCLAIMER = (
    "This product is provided \"as is\", without warranty of any kind, "
    "express or implied. The developer and distributors of LynceusScan "
    "assume no responsibility for any decision made, or damage caused, by "
    "the use of this product. Full responsibility for its use, distribution, "
    "and any decisions derived from it rests with the person or organization "
    "that shares it."
)

TIFF_TAG = "lynceus_provenance"
VLR_USER_ID = SYSTEM_NAME
VLR_RECORD_ID = 10001
VLR_DESCRIPTION = "LynceusScan product provenance"

_HEADER_SYSTEM_FIELD = SYSTEM_NAME[:32]
# LAS header fields are 32 chars by spec: a versioned
# "LynceusScan X.Y.Z" string can clip on major/minor growth.
_HEADER_SOFTWARE_FIELD = f"{SYSTEM_NAME} {__version__}"[:32]

# ctx keys that never count as node parameters.
_INTERNAL_KEYS = {
    "session_dir",
    "crs",
    "cell_size",
    "meta",
    "acceleration",
    "provenance",
    "node_fps",
    "node_iid",
    "module_id",
    "point_source_dir",
    "_point_cloud_sources",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _resolved_parameters(ctx: dict) -> dict:
    """Best-effort snapshot of the node's resolved config from the ctx.

    Injected keys (paths, session internals) and non-scalar values are
    excluded; flags such as ``classified`` are kept as informative state.
    Keys are sorted so sidecars read deterministically.
    """
    parameters: dict[str, Any] = {}
    for key in sorted(ctx):
        if key in _INTERNAL_KEYS or key.startswith("_") or key.endswith("_path"):
            continue
        value = ctx[key]
        if isinstance(value, (str, int, float, bool)):
            parameters[key] = value
    return parameters


def session_blueprint(
    session_root: str | Path, sources: list, operator: dict | None = None
) -> dict:
    """Session-level blueprint injected into every task ctx.

    Task contexts embed ``ctx["provenance"]``; ``build_provenance`` layers
    the producing node's identity on top of it. ``sources`` is the list of
    input LiDAR files for this run (empty for barrier-only runs).
    ``operator`` (``{"name", "org"}``, from Preferences) is embedded only
    when at least one field is set.
    """
    root = Path(session_root).name or str(session_root)
    blueprint = {
        "system": SYSTEM_NAME,
        "system_version": __version__,
        "license": SYSTEM_LICENSE,
        "copyright": SYSTEM_COPYRIGHT,
        "disclaimer": SYSTEM_DISCLAIMER,
        "session": {
            "root": root,
            "started_at": _now_iso(),
        },
        "sources": sources,
    }
    if operator:
        name = str(operator.get("name") or "").strip()
        org = str(operator.get("org") or "").strip()
        if name or org:
            blueprint["operator"] = {"name": name, "org": org}
    return blueprint


def build_provenance(ctx: dict, product: str | dict | None = None) -> dict:
    """Build the provenance document for the current node context.

    ``ctx`` is the tile/barrier task context (``ctx["provenance"]`` holds the
    session blueprint, ``ctx["node_fps"]`` the cascade fingerprints). The
    optional ``product`` describes the produced file: a string becomes
    ``{"name": ...}``; a dict is embedded as-is.
    """
    blueprint = dict(ctx.get("provenance") or {})
    node_id = str(ctx.get("module_id") or "")
    node_fps = ctx.get("node_fps") or {}
    iid = str(ctx.get("node_iid") or "")

    node: dict[str, Any] = {
        "node_id": node_id,
        "name": node_id.rsplit(".", 1)[-1] or node_id,
        "author": "",
        "license": SYSTEM_LICENSE,
        "instance": iid,
        "fingerprint": str(node_fps.get(iid) or ""),
    }
    node_disclaimer: str | None = None
    if node_id:
        try:
            info = manager.node_info(node_id)
        except Exception:
            info = None
        if info is not None:
            node["name"] = info.name or node["name"]
            node["author"] = info.author or ""
            node["license"] = info.license_label or SYSTEM_LICENSE
            if info.source:
                node["source"] = info.source
            if info.extension_id:
                node["extension_id"] = info.extension_id
            node_disclaimer = info.disclaimer

    parameters = _resolved_parameters(ctx)
    if parameters:
        node["parameters"] = parameters

    doc = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "system": blueprint.get("system", SYSTEM_NAME),
        "system_version": blueprint.get("system_version", __version__),
        "license": blueprint.get("license", SYSTEM_LICENSE),
        "copyright": blueprint.get("copyright", SYSTEM_COPYRIGHT),
        "disclaimer": blueprint.get("disclaimer", SYSTEM_DISCLAIMER),
    }
    operator = blueprint.get("operator")
    if isinstance(operator, dict) and (operator.get("name") or operator.get("org")):
        doc["operator"] = {
            "name": str(operator.get("name") or ""),
            "org": str(operator.get("org") or ""),
        }
    doc["node"] = node
    if node_disclaimer:
        doc["node_disclaimer"] = node_disclaimer
    session = blueprint.get("session")
    if session:
        doc["session"] = session
    sources = blueprint.get("sources")
    if sources:
        doc["sources"] = sources
    if product is not None:
        doc["product"] = {"name": product} if isinstance(product, str) else product
    # The blueprint carries `started_at`, never `generated_at`: this always
    # stamps the build time. Kept as-is on purpose (output stability).
    doc["generated_at"] = blueprint.get("generated_at") or _now_iso()
    return doc


def las_vlr(doc: dict) -> Any:
    """Serialize the provenance document as an ASCII-safe LAS VLR."""
    import laspy

    payload = json.dumps(doc, ensure_ascii=True).encode("utf-8")
    return laspy.VLR(
        user_id=VLR_USER_ID,
        record_id=VLR_RECORD_ID,
        description=VLR_DESCRIPTION,
        record_data=payload,
    )


def fill_las_header(header: Any, doc: dict) -> None:
    """Stamp a laspy header with system fields plus the provenance VLR."""
    try:
        header.system_identifier = _HEADER_SYSTEM_FIELD
        header.generating_software = _HEADER_SOFTWARE_FIELD
        header.vlrs.append(las_vlr(doc))
    except Exception:
        # Stamping is best-effort; never break the product write.
        return


def copy_header_identity(new_header: Any, header: Any) -> None:
    """Carry bit-exact header semantics across rewrites (never fails).

    version/point_format/scales/offsets/VLRs are copied by every caller
    already; this adds the identity block the LAS spec requires to travel
    with the data: ``global_encoding`` (GPS-time semantics, copied by
    integer value: object assignment does not propagate in laspy) and
    ``file_source_id``, plus the creation stamp when the backend allows
    it. Attribution (system_identifier/generating_software) stays with
    :func:`fill_las_header`.
    """
    try:
        new_header.global_encoding.value = header.global_encoding.value
    except Exception:
        pass
    for attr in ("file_source_id",):
        try:
            setattr(new_header, attr, getattr(header, attr))
        except Exception:
            pass
    try:
        new_header.creation_date = header.creation_date
    except Exception:
        pass


def copy_product_vlrs(new_header: Any, header: Any) -> None:
    """Copy VLRs that stay valid on derived products (never fails).

    COPC hierarchy VLRs (``user_id == "copc"``) describe an octree layout
    our plain-LAZ products never have — and laspy refuses to serialize
    them ("Writing COPC is not supported"). Everything else (projection,
    LASzip, Extra Bytes) travels untouched.
    """
    try:
        records = list(header.vlrs)
    except Exception:
        return
    for vlr in records:
        try:
            if str(getattr(vlr, "user_id", "") or "").lower() == "copc":
                continue
            new_header.vlrs.append(vlr)
        except Exception:
            continue


def embed_tiff_tags(path: str | Path, doc: dict) -> None:
    """Embed the provenance document in a GeoTIFF as a dataset tag."""
    import rasterio

    payload = json.dumps(doc, ensure_ascii=True)
    try:
        with rasterio.open(str(path), "r+") as dataset:
            dataset.update_tags(**{TIFF_TAG: payload})
    except Exception:
        return


def write_sidecar(
    path: str | Path, doc: dict, node_meta: dict | None = None
) -> str:
    """Write (or merge into) the ``<file>.meta.json`` sidecar.

    Existing non-provenance keys (e.g. ``metrics_computed`` written by the
    node) are preserved; the provenance document lives under the top-level
    ``provenance`` key. Uses the core naming ``<basename>.meta.json``
    (replacing the product suffix), the same one qml_style/_meta_sidecar and
    the vector_2d viewer read. Returns the sidecar path.
    """
    sidecar = Path(path).with_suffix(".meta.json")
    data: dict[str, Any] = {}
    if sidecar.exists():
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data.pop("provenance", None)
    if node_meta:
        data.update(node_meta)
    data["provenance"] = doc
    sidecar.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return str(sidecar)


def embed_gpkg_metadata(
    path: str | Path, doc: dict, table_name: str | None = None
) -> None:
    """Insert the provenance document into the GeoPackage metadata tables."""
    import sqlite3
    import uuid

    metadata_id = str(uuid.uuid4())
    payload = json.dumps(doc, ensure_ascii=True)
    try:
        con = sqlite3.connect(str(path))
        try:
            con.executescript(
                "CREATE TABLE IF NOT EXISTS gpkg_metadata (\n"
                "  id TEXT PRIMARY KEY,\n"
                "  md_scope TEXT NOT NULL,\n"
                "  md_standard_uri TEXT NOT NULL,\n"
                "  mime_type TEXT NOT NULL,\n"
                "  metadata TEXT NOT NULL\n"
                ");\n"
                "CREATE TABLE IF NOT EXISTS gpkg_metadata_reference (\n"
                "  reference_scope TEXT NOT NULL,\n"
                "  table_name TEXT,\n"
                "  column_name TEXT,\n"
                "  row_id_value INTEGER,\n"
                "  timestamp TEXT NOT NULL DEFAULT "
                "(strftime('%Y-%m-%dT%H:%M:%fZ','now')),\n"
                "  md_file_id TEXT NOT NULL,\n"
                "  md_parent_id TEXT,\n"
                "  UNIQUE (md_file_id, column_name, row_id_value)\n"
                ");"
            )
            con.execute(
                "INSERT INTO gpkg_metadata "
                "(id, md_scope, md_standard_uri, mime_type, metadata) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    metadata_id,
                    "dataset",
                    # Example-domain URI used as an opaque metadata key
                    # (it ships inside every GPKG; never fetched).
                    "http://lynceus-scan.example/provenance",
                    "application/json",
                    payload,
                ),
            )
            scope = "table" if table_name else "geopackage"
            con.execute(
                "INSERT INTO gpkg_metadata_reference "
                "(reference_scope, table_name, column_name, row_id_value, "
                "md_file_id, md_parent_id) VALUES (?, ?, NULL, NULL, ?, NULL)",
                (scope, table_name, metadata_id),
            )
            con.commit()
        finally:
            con.close()
    except Exception:
        return