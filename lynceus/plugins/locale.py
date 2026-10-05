# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Locale / translation core (domain, no Qt).

Language support via exact-match catalogs keyed by the source (English)
string. Catalogs come from locale extension packs (kind="locale", a
<lang>.json catalog declared in payload.catalog) plus per-node translations (<node>.i18n.json
sidecar, with the NODE_TRANSLATIONS constant as fallback).

Only locale packs ENABLE a language: `available()` is driven by packs, and
node-provided translations only supplement languages that a pack already
enabled. A node sidecar can never make a language selectable by itself, so
the app never ends up half-translated (Spanish nodes on an English core).

Missing keys fall back to the original English string, so dynamic text
(e.g. file paths, status messages) is never broken.

English is the default language; the active language only changes by
explicit user choice. The preference is stored by the UI layer in
QSettings ("language"); this module only knows language codes and
catalogs.
"""

from __future__ import annotations

import json
from pathlib import Path

from lynceus.plugins.registry import manager

DEFAULT_LANG = "en"


class LocaleManager:
    """Holds merged catalogs and resolves translations."""

    def __init__(self) -> None:
        self._language = DEFAULT_LANG
        self._catalogs: dict[str, dict[str, str]] = {}
        self._pack_languages: set[str] = {DEFAULT_LANG}
        self._dirty = True

    # -- language -----------------------------------------------------------

    def set_language(self, code: str) -> None:
        """Activate `code` (falls back to English when not available)."""
        code = code.lower()
        self._language = code if code in self.available() else DEFAULT_LANG

    @property
    def language(self) -> str:
        return self._language

    def available(self) -> list[str]:
        """Language codes enabled by installed locale packs (always 'en')."""
        self._ensure_loaded()
        return sorted(self._pack_languages)

    def translate(self, text: str, lang: str | None = None) -> str:
        """Exact-match translation; falls back to the original text."""
        if not text:
            return text
        self._ensure_loaded()
        code = (lang or self._language).lower()
        catalog = self._catalogs.get(code)
        if catalog is None:
            return text
        return catalog.get(text, text)

    # -- catalog building ---------------------------------------------------

    def _ensure_loaded(self) -> None:
        if not self._dirty:
            return
        catalogs: dict[str, dict[str, str]] = {DEFAULT_LANG: {}}
        pack_languages: set[str] = {DEFAULT_LANG}

        for manifest in manager.extension_manifests().values():
            if manifest.kind != "locale" or not manifest.catalog:
                continue
            try:
                data = json.loads(
                    manifest.catalog_file().read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict):
                continue
            for lang, entries in data.items():
                if not isinstance(lang, str) or not isinstance(entries, dict):
                    continue
                code = lang.lower()
                pack_languages.add(code)
                target = catalogs.setdefault(code, {})
                for src, dst in entries.items():
                    if isinstance(src, str) and isinstance(dst, str):
                        target[src] = dst

        # Node-provided translations: NODE_TRANSLATIONS constant first, then
        # the <node>.i18n.json sidecar wins over it. They only supplement
        # languages enabled by a locale pack; without a pack a node can never
        # make a language selectable (or half-translate the app).
        for info in manager.list_nodes():
            if info.source != "extension" or not info.file_path:
                continue
            if info.translations:
                _merge_catalog(
                    catalogs, info.translations, pack_languages
                )
            _merge_catalog(
                catalogs, _read_sidecar(Path(info.file_path)), pack_languages
            )

        self._catalogs = catalogs
        self._pack_languages = pack_languages
        self._dirty = False

    def mark_dirty(self) -> None:
        """Force a rebuild of the catalogs (after imports)."""
        self._dirty = True


def _merge_catalog(
    catalogs: dict[str, dict[str, str]],
    data: dict,
    enabled: set[str],
) -> None:
    """Merge `{"lang": {"source": "target"}}` entries into `catalogs`.

    Only languages present in `enabled` (locale-pack languages) are merged,
    so node-provided translations can never activate a language on their own.
    """
    if not isinstance(data, dict):
        return
    for lang, entries in data.items():
        if not isinstance(lang, str) or not isinstance(entries, dict):
            continue
        code = lang.lower()
        if code not in enabled:
            continue
        target = catalogs.setdefault(code, {})
        for src, dst in entries.items():
            if isinstance(src, str) and isinstance(dst, str):
                target[src] = dst


def _read_sidecar(node_file: Path) -> dict:
    """<node>.i18n.json next to a node file (missing/corrupt -> {})."""
    sidecar = node_file.with_name(f"{node_file.stem}.i18n.json")
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


locale_manager = LocaleManager()
"""Module-level singleton used by the UI layer."""


def t(text: str) -> str:
    """Translate `text` with the active language (English fallback)."""
    return locale_manager.translate(text)
