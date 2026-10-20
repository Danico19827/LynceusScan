# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Subgraph clipboard + file-assignment helpers (Qt-free, item 8 Fase A).

Copy/cut/paste/duplicate operate on the live graph document (the same
shape ``serialize_graph`` produces): node entries carry ``iid``/``id``/
``x``/``y``/``label``/``config``; edge entries carry ``src``/``dst``/
``src_out``/``dst_in``. Everything here is pure and unit-tested; the Qt
canvas binds it to the system clipboard.

- ``extract_subgraph`` splits a document into the selected nodes, their
  internal edges and the boundary edges crossing the selection (used by
  "duplicate and rewire": only boundary edges touch non-selected nodes).
- ``encode_subgraph``/``decode_subgraph`` wrap a subgraph as the clipboard
  payload (plain readable JSON).
- ``collect_files`` scans a folder recursively, with the extension list
  derived from a Qt file-dialog filter via ``parse_filters_extensions``.
- ``assign_ordered``/``pair_files`` build the top-down node→file mapping
  for the "Assign files..." preview (the dialog enforces the no-partial
  rule: N != M means nothing is assigned).
"""

from __future__ import annotations

import json
import os
import re

FORMAT = "lynceus-subgraph"
VERSION = 1

NODE_KEYS = ("iid", "id", "x", "y", "label", "config")


def extract_subgraph(nodes, edges, iids) -> dict:
    """Split a graph document into the selection and its boundary.

    ``nodes``/``edges`` come from ``serialize_graph``. Returns::

        {"nodes": [...],    # nodes whose iid is in ``iids`` (document order)
         "edges": [...],    # edges with BOTH endpoints inside ``iids``
         "boundary": [...]} # edges with exactly one endpoint inside ``iids``,
                            # each tagged with ``src_selected``/``dst_selected``

    External wiring is pruned from ``edges``; the requester decides whether
    to re-create it (duplicate-and-rewire) or drop it (paste in place).
    """
    iids = set(iids)
    out_nodes = [
        {k: n.get(k) for k in NODE_KEYS} for n in nodes if n.get("iid") in iids
    ]
    internal: list[dict] = []
    boundary: list[dict] = []
    for e in edges:
        if not isinstance(e, dict):
            continue
        src_in = e.get("src") in iids
        dst_in = e.get("dst") in iids
        if src_in and dst_in:
            internal.append(dict(e))
        elif src_in or dst_in:
            item = dict(e)
            item["src_selected"] = bool(src_in)
            item["dst_selected"] = bool(dst_in)
            boundary.append(item)
    return {"nodes": out_nodes, "edges": internal, "boundary": boundary}


def encode_subgraph(subgraph: dict) -> str:
    """Serialize a subgraph as the clipboard payload."""
    return json.dumps({"format": FORMAT, "version": VERSION, **subgraph})


def decode_subgraph(text) -> dict | None:
    """Parse the clipboard payload; None when it is not a valid subgraph."""
    try:
        doc = json.loads(str(text or ""))
    except (TypeError, ValueError):
        return None
    if not isinstance(doc, dict) or doc.get("format") != FORMAT:
        return None
    if doc.get("version") != VERSION:
        return None
    nodes = doc.get("nodes")
    if not isinstance(nodes, list):
        return None
    for n in nodes:
        if not isinstance(n, dict) or not n.get("iid") or not n.get("id"):
            return None
    return {
        "nodes": nodes,
        "edges": [e for e in doc.get("edges") or [] if isinstance(e, dict)],
        "boundary": [e for e in doc.get("boundary") or [] if isinstance(e, dict)],
    }


def parse_filters_extensions(filters: str) -> list[str]:
    """Extension suffixes (without dot) from a Qt file-dialog filter string.

    ``"Raster files (*.tif *.tiff);;All files (*)"`` →
    ``["tif", "tiff"]`` (deduplicated, order preserved). A string without
    parenthesized patterns yields ``[]`` (meaning "every file").
    """
    out: list[str] = []
    for group in re.findall(r"\(([^()]*)\)", str(filters or "")):
        for ext in re.findall(r"\*\.([A-Za-z0-9]+)\b", group):
            lowered = ext.lower()
            if lowered not in out:
                out.append(lowered)
    return out


def collect_files(directory, extensions=None, recursive: bool = True) -> list[str]:
    """Sorted absolute paths under ``directory`` matching ``extensions``.

    ``extensions`` is a list of suffixes WITHOUT the dot (e.g.
    ``["tif", "tiff"]``); None/empty means every file. ``recursive=False``
    only scans the top level. Missing directory -> [].
    """
    directory = os.path.abspath(os.path.expanduser(str(directory)))
    if not os.path.isdir(directory):
        return []
    exts = {
        (s if s.startswith(".") else f".{s}").lower()
        for s in (extensions or [])
        if isinstance(s, str) and s
    }
    found: list[str] = []
    for root, dirs, files in os.walk(directory):
        if not recursive:
            dirs[:] = []
        for name in files:
            if exts and os.path.splitext(name)[1].lower() not in exts:
                continue
            found.append(os.path.join(root, name))
    return sorted(found, key=lambda p: (p.lower(), p))


def pair_files(ordered_nodes: list, files: list) -> list[tuple]:
    """Top-down (node, file) preview pairing; length = min(N, M)."""
    return list(zip(ordered_nodes, files))


def assign_ordered(ordered_nodes: list, files: list) -> dict:
    """Map the first K files to the first K nodes (K = min(N, M)).

    Pure pairing for the preview; the dialog enforces the no-partial rule
    (N != M -> nothing is assigned).
    """
    return dict(zip(ordered_nodes, files))