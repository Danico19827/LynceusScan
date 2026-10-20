# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Shared file-queue mechanics for file-backed input nodes.

No NODE_ID here: the AST discovery ignores this module, like
_product_input.py. The queue lives ONCE in this base instead of once per
node: Load LAS/LAZ, Input Table, Input GeoPackage, Input Raster and any
third-party input (``input_file_key`` convention) all share it.

A queue is a plain list of paths (``files`` in the persisted
config: flat ``config.files`` for the Load node, ``config.pipeline.files``
for product inputs — one list per node, shared across Input Raster
strategies). ``file_path`` stays the effective current file (the queue
head when a queue is set), so single-file behavior is untouched.

The GUI driver chains one engine run per file with a per-run file
override (never mutating the saved project); the CLI pre-merges
``--source`` the same way. Everything here is pure/Qt-free and unit
tested; ``t()`` strings are covered in all 9 locale packs.
"""

from __future__ import annotations

import os

from lynceus.plugins.locale import t


def normalize_queue(files) -> list[str]:
    """Deduplicated path list preserving pick order (no existence check)."""
    seen: set[str] = set()
    queue: list[str] = []
    for item in files or []:
        if not isinstance(item, str):
            continue
        path = item.strip()
        if not path or path in seen:
            continue
        seen.add(path)
        queue.append(path)
    return queue


def set_queue(paths) -> list[str]:
    """Replace the whole queue with a fresh file selection."""
    return normalize_queue(paths)


def add_to_queue(queue, paths) -> tuple[list[str], int]:
    """Append new files, skipping duplicates already queued."""
    current = normalize_queue(queue)
    seen = set(current)
    added = 0
    for item in paths or []:
        if not isinstance(item, str):
            continue
        path = item.strip()
        if not path or path in seen:
            continue
        seen.add(path)
        current.append(path)
        added += 1
    return current, added


def remove_from_queue(queue, index: int) -> list[str]:
    """Queue without the entry at ``index`` (out of range: unchanged)."""
    current = normalize_queue(queue)
    try:
        idx = int(index)
    except (TypeError, ValueError):
        return current
    if 0 <= idx < len(current):
        del current[idx]
    return current


def move_in_queue(queue, frm: int, to: int) -> list[str]:
    """Queue with one entry relocated (out of range: unchanged)."""
    current = normalize_queue(queue)
    try:
        src, dst = int(frm), int(to)
    except (TypeError, ValueError):
        return current
    if not 0 <= src < len(current) or not 0 <= dst < len(current):
        return current
    current.insert(dst, current.pop(src))
    return current


def partition_existing(queue) -> tuple[list[str], list[str]]:
    """Split into (existing files, missing entries) preserving order."""
    existing: list[str] = []
    missing: list[str] = []
    for path in normalize_queue(queue):
        (existing if os.path.isfile(path) else missing).append(path)
    return existing, missing


def effective_file(queue, current: str | None) -> str | None:
    """File a run actually uses: the queue head, else the single file."""
    head = normalize_queue(queue)
    if head:
        return head[0]
    return current or None


def queue_count_text(count: int) -> str:
    """Node label for a multi-file queue (single files keep their label)."""
    return t("{n} files queued").format(n=int(count))


def summarize_queue(results: list[dict], stopped: bool = False) -> str:
    """One-block queue report: counts plus one line per failed file.

    Each result is ``{"file": path, "ok": bool, "reason": str}``; reasons
    are engine messages (EN by policy) while the frame is translated.
    """
    total = len(results)
    ok = sum(1 for item in results if item.get("ok"))
    failed = total - ok
    if stopped:
        head = t(
            "File queue stopped: {ok} of {total} files processed "
            "({failed} failed)"
        ).format(ok=ok, total=total, failed=failed)
    else:
        head = t(
            "File queue finished: {ok} of {total} files processed "
            "({failed} failed)"
        ).format(ok=ok, total=total, failed=failed)
    lines = [head]
    for item in results:
        if item.get("ok"):
            continue
        name = os.path.basename(str(item.get("file") or "?"))
        reason = str(item.get("reason") or "").strip().splitlines()
        lines.append(f"{name}: {reason[0] if reason else '?'}")
    return "\n".join(lines)
