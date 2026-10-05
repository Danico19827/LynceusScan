# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Catalog of visualizable products produced by a run.

Flattens pipeline output payloads ({iid: payload}) into a list of
``OutputProduct`` with its canonical viewer. This is shared by the Outputs
gallery and QGIS sidecar styling; ``qml`` stores the discovered sidecar name.

Rules:
- Products require a known viewer (raster/vector/CSV/point cloud/JSON);
  non-file payload metadata and external input paths are discarded.
- This module is Qt-free; it only builds product records.

The UI layer performs final viewer-registry confirmation through ``viewers_for``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import json
import os

_RASTER_EXT = {".tif", ".tiff"}
_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
_VECTOR_EXT = {".gpkg", ".shp", ".geojson"}
_TABLE_EXT = {".csv", ".txt"}
_POINT_CLOUD_EXT = {".las", ".laz"}
_JSON_EXT = {".json"}

VIEWER_KIND_LABELS = {
    "raster": "Raster",
    "image": "Image",
    "vector": "Vector",
    "table_csv": "Table",
    "point_cloud": "Point Cloud",
    "json": "JSON",
}
"""English display labels for canonical viewer ids (UI translates them)."""


def viewer_kind_label(viewer: str) -> str:
    """English label for a viewer id (falls back to the id itself)."""
    return VIEWER_KIND_LABELS.get(viewer, viewer)

# Payload keys that do NOT represent products:
# - "node"/"kind": emitter metadata
# - "source_file": original EXTERNAL path of an input (not a session artifact)
# - "warnings"/"count": barrier metadata
# - "tiles": handled separately (point cloud product)
# - "tile_id"/"output"/"result": tile-task remnants, not barrier products.
_SKIP_KEYS = {
    "node",
    "kind",
    "source_file",
    "warnings",
    "count",
    "display_name",
    "tiles",
    "tile_id",
    "output",
    "result",
}


@dataclass(frozen=True)
class OutputProduct:
    """A final run product ready for visualization."""

    iid: str
    node_name: str
    name: str
    viewer: str
    path: str | None = None
    tiles: tuple[dict | str, ...] = ()
    size: int = 0
    payload_key: str = "file"
    qml: str | None = None  # reserved: basename of the generated .qml (QGIS feature)
    segment: str | None = None  # "Segment 1/5" when produced by a segment batch
    segment_index: int | None = None  # numeric batch index for gallery ordering
    flow: tuple = ()  # lineage segments (see compute_flow_segments)


def display_labels(
    bases: list[str], paths: list[str | None], iids: list[str]
) -> list[str]:
    """Gallery/preview label per product.

    Basename by default; products sharing a basename with a sibling show
    their path relative to the common root instead, so branch-scoped
    twins read as ``d1/dtm_mosaic.tif`` vs ``d2/dtm_mosaic.tif``. Pathless
    duplicates fall back to the instance id. Paths are never translated.
    """
    counts = Counter(bases)
    roots = [p for p in paths if p]
    try:
        common = os.path.commonpath(roots) if roots else ""
    except ValueError:
        common = ""
    labels = []
    for base, path, iid in zip(bases, paths, iids):
        if counts[base] < 2:
            labels.append(base)
        elif path and common and path != common:
            labels.append(os.path.relpath(path, common))
        else:
            labels.append(f"{base} — {iid}")
    return labels


def compute_flow_segments(
    node_info: dict[str, str],
    edges: list | None = None,
    custom_labels: dict[str, str] | None = None,
    source_files: dict[str, str] | None = None,
) -> dict[str, tuple]:
    """Upstream → self → downstream lineage per instance.

    ``node_info`` maps instance ids to registry names; ``edges`` are
    ``(src_iid, dst_iid, ...)`` canvas connections. Each segment is
    ``(iid, key, filename, is_self)`` where ``key`` is the custom label
    when the user renamed the node, else the registry name, and
    ``filename`` annotates file-backed nodes (loaders/inputs) with their
    source basename. Render with :func:`render_flow`.
    """
    custom_labels = custom_labels or {}
    source_files = source_files or {}
    preds: dict[str, list[str]] = {}
    succs: dict[str, list[str]] = {}
    for edge in edges or []:
        src, dst = edge[0], edge[1]
        preds.setdefault(dst, []).append(src)
        succs.setdefault(src, []).append(dst)
        preds.setdefault(src, preds.get(src, []))
        succs.setdefault(dst, succs.get(dst, []))

    def _key(nid: str) -> str:
        return custom_labels.get(nid) or node_info.get(nid, nid)

    # Display names shared by several instances are qualified globally so
    # twin branches read apart in every chain (render_flow additionally
    # qualifies duplicates inside hand-built chains).
    shared = {
        key for key, count in Counter(_key(i) for i in node_info).items()
        if count > 1
    }

    def _walk(start: str, graph: dict[str, list[str]]) -> list[str]:
        seen: list[str] = []
        seen_set = {start}
        stack = [start]
        while stack:
            node = stack.pop()
            for nxt in graph.get(node, []):
                if nxt not in seen_set:
                    seen_set.add(nxt)
                    seen.append(nxt)
                    stack.append(nxt)
        return seen

    flows: dict[str, tuple] = {}
    for iid in node_info:
        up = _walk(iid, preds)
        down = _walk(iid, succs)
        chain = list(reversed(up)) + [iid] + down
        segments = []
        for nid in chain:
            key = _key(nid)
            if key in shared:
                key = f"{key} · {nid}"
            segments.append(
                (nid, key, source_files.get(nid), nid == iid)
            )
        flows[iid] = tuple(segments)
    return flows


def render_flow(segments: tuple, tr=None) -> str:
    """Render lineage segments as ``Load (vuelo.laz) → [Generate DTM]``.

    The producing node is bracketed; file-backed nodes carry their source
    basename; display names shared by several instances are qualified with
    the instance id so twin branches read apart. ``tr`` translates names.
    """
    tr = tr or (lambda s: s)
    keys = Counter(seg[1] for seg in segments)
    parts = []
    for nid, key, filename, is_self in segments:
        label = tr(key)
        if keys[key] > 1:
            label = f"{label} · {nid}"
        if filename:
            label = f"{label} ({filename})"
        if is_self:
            label = f"[{label}]"
        parts.append(label)
    return " → ".join(parts)


def viewer_for_path(path: str) -> str | None:
    """Resolve the canonical viewer from a file extension."""
    ext = Path(path).suffix.lower()
    if ext in _RASTER_EXT:
        return "raster"
    if ext in _IMAGE_EXT:
        return "image"
    if ext in _VECTOR_EXT:
        return "vector"
    if ext in _TABLE_EXT:
        return "table_csv"
    if ext in _POINT_CLOUD_EXT:
        return "point_cloud"
    if ext in _JSON_EXT:
        return "json"
    return None


def product_for_path(path: str | None, node_name: str) -> OutputProduct | None:
    """Build a previewable product for an arbitrary file (Qt-free).

    Bridge mode (File → Open Preview, past-session browsing): any existing
    file with a known viewer opens directly, with no pipeline, session, or
    copy involved. Missing files and unknown extensions yield None, following
    the same ghost rule as run products.
    """
    if not path or not isinstance(path, str):
        return None
    if not Path(path).is_file():
        return None
    viewer = viewer_for_path(path)
    if viewer is None:
        return None
    return OutputProduct(
        iid=f"external:{path}",
        node_name=node_name,
        name=Path(path).name,
        viewer=viewer,
        path=path,
        size=_file_size(path),
        payload_key="external",
    )


def _registry_node_name(module_id: str) -> str:
    """Best-effort registry display name (module id fallback)."""
    try:
        from lynceus.plugins.registry import manager

        info = manager.node_info(module_id)
        if info is not None and getattr(info, "name", ""):
            return info.name
    except Exception:
        pass
    return module_id


def products_from_session(session_dir: str | os.PathLike) -> list[OutputProduct]:
    """Rebuild gallery products from a past session (Qt-free, read-only).

    Reads ``session_state.json`` (per-instance payloads + modules, the same
    shape the live gallery consumes) and ``session_meta.json`` sources for
    the file-backed ``source`` items. Missing/corrupt state yields []; moved
    or deleted files are skipped by the ghost rule, and canvas-only details
    (custom labels, edge lineage) are not stored in sessions, so products
    carry no flow. Nothing is executed, written, or copied.
    """
    root = Path(session_dir)
    try:
        state = json.loads((root / "session_state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(state, dict):
        return []
    outputs = state.get("node_outputs") or {}
    if not isinstance(outputs, dict):
        return []
    modules = state.get("node_modules") or {}
    node_info = {
        iid: _registry_node_name(mod)
        for iid, mod in modules.items()
        if isinstance(mod, str) and mod
    }
    for iid in outputs:
        node_info.setdefault(iid, str(iid))
    source_files: dict[str, str] = {}
    source_paths: dict[str, str] = {}
    try:
        meta = json.loads((root / "session_meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}
    sources = meta.get("sources") if isinstance(meta, dict) else None
    if isinstance(sources, list):
        for entry in sources:
            if not isinstance(entry, dict):
                continue
            iid = entry.get("iid")
            path = entry.get("file")
            if isinstance(iid, str) and isinstance(path, str) and path:
                source_files[iid] = os.path.basename(path)
                source_paths[iid] = path
    return flatten_outputs(
        outputs, node_info, source_files=source_files, source_paths=source_paths
    )


def _file_size(path: str) -> int:
    try:
        return Path(path).stat().st_size
    except OSError:
        return 0


def _qml_sidecar(path: str | None) -> str | None:
    """Basename of the QGIS `.qml` if it exists next to the product."""
    if not path:
        return None
    qml = Path(path).with_suffix(".qml")
    return qml.name if qml.is_file() else None


def flatten_outputs(
    outputs: dict,
    node_info: dict[str, str],
    segment: str | None = None,
    segment_index: int | None = None,
    edges: list | None = None,
    custom_labels: dict[str, str] | None = None,
    source_files: dict[str, str] | None = None,
    source_paths: dict[str, str] | None = None,
) -> list[OutputProduct]:
    """Flatten per-instance output payloads into visualizable products.

    ``node_info`` maps instance IDs to display names supplied by the UI. Only
    products with a known viewer are returned, with the primary file first.
    ``segment``/``segment_index`` tag products published incrementally by a
    segment batch so the gallery groups them per segment. ``edges``,
    ``custom_labels`` and ``source_files`` feed the per-product lineage
    (:func:`compute_flow_segments`); without them products carry no flow.
    ``source_paths`` maps file-backed instances to their full source path;
    each existing path with a known viewer is also emitted as a previewable
    ``source`` product (kept separate from ``source_files`` basenames, which
    stay display-only for the lineage).
    """
    products: list[OutputProduct] = []
    flows = compute_flow_segments(
        node_info, edges, custom_labels, source_files
    )
    for iid, payload in outputs.items():
        if not isinstance(payload, dict):
            continue
        node_name = node_info.get(iid, str(iid))

        tiles = payload.get("tiles")
        if isinstance(tiles, list) and tiles:
            products.append(
                OutputProduct(
                    iid=iid,
                    node_name=node_name,
                    name=f"Point Cloud ({len(tiles)} tiles)",
                    viewer="point_cloud",
                    tiles=tuple(tiles),
                    size=0,
                    payload_key="tiles",
                    segment=segment,
                    segment_index=segment_index,
                    flow=flows.get(iid, ()),
                )
            )

        for key, value in payload.items():
            if key in _SKIP_KEYS or not isinstance(value, str) or not value:
                continue
            if not Path(value).is_file():
                # Ghost payload (a barrier that advertised a file it never
                # wrote): the engine already reported it loudly; the
                # gallery must not offer a preview that can never load.
                continue
            viewer = viewer_for_path(value)
            if viewer is None:
                continue
            display_name = str(payload.get("display_name") or "").strip()
            products.append(
                OutputProduct(
                    iid=iid,
                    node_name=node_name,
                    name=display_name or Path(value).name,
                    viewer=viewer,
                    path=value,
                    size=_file_size(value),
                    payload_key=key,
                    qml=_qml_sidecar(value),
                    segment=segment,
                    segment_index=segment_index,
                    flow=flows.get(iid, ()),
                )
            )

    for iid, path in (source_paths or {}).items():
        # File-backed sources (loaders/inputs) as previewable gallery items:
        # the point-cloud viewer resolves COPC hierarchies itself, so a
        # .copc.laz source previews without decoding the whole file. Same
        # ghost rule as products: a moved/deleted source offers no preview.
        if not isinstance(path, str) or not path:
            continue
        if not Path(path).is_file():
            continue
        viewer = viewer_for_path(path)
        if viewer is None:
            continue
        products.append(
            OutputProduct(
                iid=iid,
                node_name=node_info.get(iid, str(iid)),
                name=Path(path).name,
                viewer=viewer,
                path=path,
                size=_file_size(path),
                payload_key="source",
                segment=segment,
                segment_index=segment_index,
                flow=flows.get(iid, ()),
            )
        )

    products.sort(
        key=lambda p: (
            p.node_name.lower(),
            p.payload_key != "file",  # primary product first
            p.name.lower(),
        )
    )
    return products