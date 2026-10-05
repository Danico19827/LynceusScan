# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Shared session-layout paths (no NODE_ID: ignored by discovery).

Branch isolation (F4.2): same-module instances over overlapping tiles
write tile-task outputs under an instance scope (``<kind>/<scope>/``)
and barrier finals under ``artifacts/<scope>/``. Anything without scope
keeps the legacy layout, so single-instance graphs never change a byte
on disk. Scopes are canvas instance ids (stable per project); they never
enter fingerprints.
"""

from __future__ import annotations

from pathlib import Path


def tile_scope(ctx: dict | None) -> str:
    """Instance scope for this task ("" = legacy shared layout)."""
    if not isinstance(ctx, dict):
        return ""
    return str(ctx.get("tile_scope") or "")


def scoped_dir(ctx: dict, base: str) -> Path:
    """Session-relative subdirectory for tile-task outputs of one instance."""
    session = Path(ctx["session_dir"])
    scope = tile_scope(ctx)
    out = session / base / scope if scope else session / base
    out.mkdir(parents=True, exist_ok=True)
    return out


def scoped_file(ctx: dict, name: str) -> str:
    """Session file for a barrier final of one instance."""
    session = Path(ctx["session_dir"])
    scope = tile_scope(ctx)
    out = session / scope / name if scope else session / name
    out.parent.mkdir(parents=True, exist_ok=True)
    return str(out)


def scope_for(base: str, scope: str) -> str:
    """Pure variant for engine-side resolution (no ctx needed)."""
    return f"{base}/{scope}" if base and scope else base
