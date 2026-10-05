# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Project templates (domain, no Qt).

A template is a plain ``.lynx`` project document with an embedded ``template``
metadata block (name, description, author, version, tags). Extra top-level
keys are ignored by ``project.load_project`` / canvas ``restore_graph``, so a
template stays a valid project you can also open directly.

Two sources (bundled first, then user-authored):
  - ``templates/`` next to the repo/app (shipped, read-only);
  - ``<data>/templates/`` (``%APPDATA%/LynceusScan/templates`` on Windows),
    where "Save as Template" lands.

Opening a template always starts an untitled copy: the caller must load the
document but keep ``_project_path`` empty, so the first save prompts Save As
and the template file itself is never overwritten.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from lynceus import project as project_io
from lynceus.plugins.store import data_root
from lynceus.resources import resource_path

logger = logging.getLogger(__name__)

TEMPLATE_META_KEY = "template"
USER_TEMPLATES_SUBDIR = "templates"

BUNDLED_TEMPLATES_DIR = resource_path(USER_TEMPLATES_SUBDIR)
"""Shipped read-only templates (``templates/`` at the repo/app root)."""


def user_templates_dir() -> Path:
    """Writable per-user templates dir (created on demand)."""
    return data_root() / USER_TEMPLATES_SUBDIR


def save_target_dir(
    bundled_dir: str | Path | None = None,
    user_dir: str | Path | None = None,
) -> Path:
    """Where "Save as Template" lands: the bundled root folder when it is
    writable (dev checkout, so authored templates ship with the app),
    otherwise the per-user dir (installed app, read-only program folder)."""
    import os

    bundled = Path(bundled_dir) if bundled_dir else BUNDLED_TEMPLATES_DIR
    try:
        bundled.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    else:
        if bundled.is_dir() and os.access(bundled, os.W_OK):
            return bundled
    target = Path(user_dir) if user_dir else user_templates_dir()
    target.mkdir(parents=True, exist_ok=True)
    return target


@dataclass(frozen=True)
class TemplateInfo:
    """One discovered template: metadata + source + validation state."""

    name: str
    description: str
    author: str
    version: str
    tags: tuple = ()
    path: str = ""
    source: str = "user"  # "bundled" | "user"
    node_count: int = 0
    valid: bool = True
    error: str = ""


def template_metadata(doc: dict, path: Path) -> dict:
    """Metadata block of a template document (defaults for missing keys)."""
    meta = doc.get(TEMPLATE_META_KEY)
    if not isinstance(meta, dict):
        meta = {}
    name = meta.get("name") or doc.get("name") or path.stem
    description = meta.get("description") or ""
    author = meta.get("author") or ""
    version = meta.get("version") or ""
    tags = meta.get("tags") or ()
    tags = tuple(t for t in tags if isinstance(t, str) and t.strip())
    return {
        "name": str(name),
        "description": str(description),
        "author": str(author),
        "version": str(version),
        "tags": tags,
    }


def list_templates(
    user_dir: str | Path | None = None,
    bundled_dir: str | Path | None = None,
) -> list[TemplateInfo]:
    """Discover templates in both sources (bundled first, then user).

    Invalid documents are reported (valid=False) instead of raising, so the
    picker can show them with an error badge.
    """
    found: list[TemplateInfo] = []
    sources = (
        ("bundled", Path(bundled_dir) if bundled_dir else BUNDLED_TEMPLATES_DIR),
        ("user", Path(user_dir) if user_dir else user_templates_dir()),
    )
    for source, directory in sources:
        if source == "user":
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                logger.warning("Templates dir not writable (%s): %s", directory, exc)
                continue
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob(f"*{project_io.PROJECT_SUFFIX}")):
            found.append(_inspect(path, source))
    return found


def _inspect(path: Path, source: str) -> TemplateInfo:
    try:
        doc = project_io.load_project(path)
    except project_io.ProjectError as exc:
        return TemplateInfo(
            name=path.stem,
            description="",
            author="",
            version="",
            path=str(path),
            source=source,
            valid=False,
            error=str(exc),
        )
    meta = template_metadata(doc, path)
    missing = project_io.validate_nodes(doc)
    if missing:
        return TemplateInfo(
            valid=False,
            error=(
                "Missing node(s): " + ", ".join(missing[:5])
                + ("..." if len(missing) > 5 else "")
            ),
            node_count=len(doc.get("nodes", [])),
            path=str(path),
            source=source,
            **meta,
        )
    return TemplateInfo(
        node_count=len(doc.get("nodes", [])),
        path=str(path),
        source=source,
        **meta,
    )


def build_template_doc(graph: dict, meta: dict) -> dict:
    """Project document for "Save as Template": graph + metadata block."""
    tags = meta.get("tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    doc = dict(graph)
    doc["name"] = meta.get("name") or "template"
    doc[TEMPLATE_META_KEY] = {
        "name": meta.get("name") or "template",
        "description": meta.get("description") or "",
        "author": meta.get("author") or "",
        "version": meta.get("version") or "",
        "tags": [str(t) for t in tags],
    }
    return doc


def template_file_name(name: str) -> str:
    """Safe ``.lynx`` file name for a template title."""
    slug = "".join(c if (c.isalnum() or c in ("-", "_", " ")) else "_" for c in name)
    slug = "_".join(slug.split()).strip("_") or "template"
    return f"{slug}{project_io.PROJECT_SUFFIX}"
