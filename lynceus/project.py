# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Persisted LynceusScan project documents (.lynx).

Qt-free domain layer. A project stores the complete canvas: node positions and
configuration, typed edges, and view state. Loading validates the structure and
current node IDs while ignoring unknown keys for forward compatibility.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from lynceus.plugins.registry import manager

PROJECT_FORMAT = "lynceus-project"
PROJECT_VERSION = 2
PROJECT_SUFFIX = ".lynx"
FILE_FILTER = "LynceusScan Project (*.lynx)"

_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9_-]+")


class ProjectError(Exception):
    """Project validation error with a UI-safe message."""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_project(path: str | Path, data: dict) -> None:
    """Write a project document with normalized format and version metadata."""
    doc = {
        "format": PROJECT_FORMAT,
        "version": PROJECT_VERSION,
        "saved_at": now_iso(),
        **data,
    }
    path = Path(path)
    if path.suffix.lower() != PROJECT_SUFFIX:
        path = path.with_suffix(PROJECT_SUFFIX)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")


def load_project(path: str | Path) -> dict:
    """Load and validate a project, raising ProjectError on invalid input."""
    path = Path(path)
    if not path.exists():
        raise ProjectError(f"Project file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProjectError(f"Cannot read project {path.name}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ProjectError(f"Not a LynceusScan project: {path.name}")
    if raw.get("format") != PROJECT_FORMAT:
        raise ProjectError(f"Not a LynceusScan project: {path.name}")
    version = raw.get("version")
    if not isinstance(version, int) or version < 1:
        raise ProjectError(f"Invalid project version in {path.name}")
    if version > PROJECT_VERSION:
        raise ProjectError(
            f"{path.name} was saved with a newer LynceusScan version "
            f"(project v{version}, app v{PROJECT_VERSION}). Please update."
        )
    _validate_structure(raw)
    return raw


def validate_nodes(data: dict) -> list[str]:
    """Return the project node ids missing from the registry.

    Missing nodes usually belong to a disabled or removed extension: the
    project is still loadable (the nodes stay on canvas as unavailable), so
    this no longer raises. Structural errors keep raising ``ProjectError``.
    """
    return sorted(
        {
            entry.get("id", "?")
            for entry in data.get("nodes", [])
            if not _node_exists(entry.get("id"))
        }
    )


def session_dir_name(project_name: str) -> str:
    """Return a safe directory name for grouping a project's sessions."""
    slug = _UNSAFE_NAME.sub("_", project_name).strip("_")
    return slug or "project"


def _node_exists(node_id) -> bool:
    if not isinstance(node_id, str) or not node_id:
        return False
    return manager.contains(node_id)


def _validate_structure(data: dict) -> None:
    """Minimal document structure.

    - Each node has an 'id' (module) and an optional 'iid' (instance, v2).
    - Instance iids must be unique: identity is the basis of the graph, so
      several copies of the same node type are allowed.
    - Edges reference iids (v2) or module ids (legacy v1).
    """
    nodes = data.get("nodes")
    edges = data.get("edges")
    if not isinstance(nodes, list) or not all(
        isinstance(n, dict) and isinstance(n.get("id"), str)
        for n in nodes
    ):
        raise ProjectError("Malformed project: 'nodes' must be a list of entries")

    iids = [n["iid"] for n in nodes if isinstance(n.get("iid"), str) and n["iid"]]
    duplicates = sorted({i for i in iids if iids.count(i) > 1})
    if duplicates:
        raise ProjectError(
            "Malformed project: duplicated instance id(s): "
            + ", ".join(duplicates)
        )

    known = {n["id"] for n in nodes} | set(iids)
    if not isinstance(edges, list):
        raise ProjectError("Malformed project: 'edges' must be a list")
    for edge in edges:
        if not isinstance(edge, dict):
            raise ProjectError("Malformed project: invalid edge entry")
        for key in ("src", "dst"):
            if edge.get(key) not in known:
                raise ProjectError(
                    f"Malformed project: edge references unknown node "
                    f"'{edge.get(key)}'"
                )


# NOTE: the port type validation (out_type/in_type vs can_connect)
# is performed when reconstructing the graph on the canvas, where the real contract lives.
