# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Theme core (domain, no Qt).

A theme is a fixed set of color roles (``THEME_ROLES``) with hex values.
The built-in ``"default"`` theme is the core dark palette; theme extension
packs (kind="theme", a ``theme.json`` palette declared in payload.palette)
override any subset of roles by id. Missing roles fall back to the default,
so a partial palette is a valid theme; unknown roles and malformed hex
values are dropped with a warning and never break startup.

Only theme packs ENABLE a theme: ``available()`` lists ``"default"`` plus
installed pack ids. The active theme only changes by explicit user choice;
the preference is stored by the UI layer in QSettings ("theme"), this
module only knows ids and palettes.
"""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

DEFAULT_THEME_ID = "default"

THEME_ROLES = {
    "background": "#0b0f1a",  # deep background (window, canvas)
    "surface": "#1c2333",  # surfaces (panels, tree, menus, node fill)
    "border": "#2f3a4f",  # borders, hover, elevation, node border
    "border_strong": "#3a465c",  # button hover, tree selection
    "text": "#e6e9f0",  # primary text (node titles, lockup recolor)
    "text_body": "#aeb7c6",  # body text (node labels)
    "muted": "#8a93a6",  # secondary text (categories, wires)
    "accent": "#7d9fd4",  # selection, focus, progress
    "danger": "#e08a8a",  # close hover, errors
    "amber": "#c9a86a",  # modified parameter value, warnings
    "canvas_bg": "#0b0f1a",  # node canvas background
    "wire": "#8a93a6",  # connection wires
}
"""Fixed role contract: theme packs may only override these roles."""

_HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


class ThemeManager:
    """Holds the active palette and resolves installed theme packs."""

    def __init__(self, registry=None) -> None:
        from lynceus.plugins.registry import manager as global_manager

        self._registry = registry or global_manager
        self._theme = DEFAULT_THEME_ID
        self._palettes: dict[str, dict[str, str]] = {}
        self._dirty = True

    # -- selection ------------------------------------------------------

    def set_theme(self, theme_id: str) -> None:
        """Activate `theme_id` (falls back to default when not available)."""
        theme_id = (theme_id or "").lower()
        self._theme = theme_id if theme_id in self.available() else (
            DEFAULT_THEME_ID
        )

    @property
    def theme(self) -> str:
        return self._theme

    def available(self) -> list[str]:
        """Theme ids: always 'default', plus installed theme packs."""
        self._ensure_loaded()
        return [DEFAULT_THEME_ID] + sorted(
            tid for tid in self._palettes if tid != DEFAULT_THEME_ID
        )

    def display_name(self, theme_id: str) -> str:
        """Human label for a theme id (pack display_name, or 'Default')."""
        if theme_id == DEFAULT_THEME_ID:
            return "Default"
        self._ensure_loaded()
        manifests = self._registry.extension_manifests()
        manifest = manifests.get(theme_id)
        if manifest is not None and manifest.kind == "theme":
            return manifest.display_name or theme_id
        return theme_id

    def palette(self, theme_id: str | None = None) -> dict[str, str]:
        """Merged palette: defaults overlaid with the pack's valid roles."""
        self._ensure_loaded()
        merged = dict(THEME_ROLES)
        tid = (theme_id or self._theme).lower()
        if tid != DEFAULT_THEME_ID:
            merged.update(self._palettes.get(tid, {}))
        return merged

    # -- pack loading ---------------------------------------------------

    def _ensure_loaded(self) -> None:
        if not self._dirty:
            return
        palettes: dict[str, dict[str, str]] = {}
        for manifest in self._registry.extension_manifests().values():
            if manifest.kind != "theme" or not manifest.palette:
                continue
            try:
                data = json.loads(
                    manifest.palette_file().read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning(
                    "Theme '%s' palette unreadable (%s); using defaults",
                    manifest.id,
                    exc,
                )
                continue
            palettes[manifest.id.lower()] = _validated_roles(
                manifest.id, data
            )
        self._palettes = palettes
        self._dirty = False

    def mark_dirty(self) -> None:
        """Force a rebuild of the palettes (after imports)."""
        self._dirty = True


def _validated_roles(pack_id: str, data: object) -> dict[str, str]:
    """Keep only known roles with valid hex values (corrupt -> defaults)."""
    if not isinstance(data, dict):
        logger.warning(
            "Theme '%s' palette must be a JSON object; using defaults",
            pack_id,
        )
        return {}
    roles: dict[str, str] = {}
    for role, value in data.items():
        if role not in THEME_ROLES:
            logger.warning(
                "Theme '%s': unknown role '%s' ignored", pack_id, role
            )
            continue
        if not isinstance(value, str) or not _HEX_RE.match(value):
            logger.warning(
                "Theme '%s': role '%s' needs a #rrggbb hex value "
                "(got %r); using default",
                pack_id,
                role,
                value,
            )
            continue
        roles[role] = value.lower()
    return roles


theme_manager = ThemeManager()
"""Module-level singleton used by the UI layer."""
