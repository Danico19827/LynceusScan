# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Consent store on disk (domain, no Qt).

Persistence layout under the platform data dir (Windows: %APPDATA%,
Linux: $XDG_DATA_HOME or ~/.local/share):
  LynceusScan/
    eulas.json    accepted per-pack EULAs (pack id -> sha of accepted text)
    terms.json    accepted app-level terms (version -> sha + date)
    disabled.json disabled extension ids (node/pack ids the user turned off)

There is no account and no entitlement machinery: the core is open source
(GPL-3.0-or-later) and packs inherit that license. The only "gates" are
consent records: an optional per-pack EULA and the app-level Terms shown on
first launch.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from lynceus.plugins.manifest import ExtensionManifest

logger = logging.getLogger(__name__)

# Pack resolution states
PACK_INSTALLED = "installed"
PACK_EULA_PENDING = "eula_pending"
PackState = str

_EULA_HASH = "sha256"


def _data_dir() -> Path:
    """Platform data dir for LynceusScan (Windows: %APPDATA%)."""
    home = Path.home()
    win_appdata = None
    import os

    if os.environ.get("APPDATA"):
        win_appdata = Path(os.environ["APPDATA"])
    if win_appdata is not None:
        return win_appdata / "LynceusScan"
    return (
        Path(os.environ.get("XDG_DATA_HOME", home / ".local" / "share"))
        / "LynceusScan"
    )


def data_root() -> Path:
    """Public platform data dir for LynceusScan (user templates, etc.)."""
    return _data_dir()


def local_data_root() -> Path:
    """Non-roaming platform data dir (Windows: %LOCALAPPDATA%).

    Session outputs can be gigabytes of tiles: they must never roam with
    %APPDATA% nor land inside the install dir (read-only under Program
    Files, wiped on update). Falls back to :func:`data_root`.
    """
    local = os.environ.get("LOCALAPPDATA")
    if local:
        return Path(local) / "LynceusScan"
    return _data_dir()


@dataclass
class PackStatus:
    state: PackState = PACK_INSTALLED
    reason: str = ""

    def __str__(self) -> str:
        return self.state


class ConsentStore:
    """Offline-first EULA + app-terms persistence and resolution."""

    def __init__(self, root: str | Path | None = None):
        self._root = Path(root or _data_dir()).resolve()
        self._eulas_path = self._root / "eulas.json"
        self._terms_path = self._root / "terms.json"
        self._disabled_path = self._root / "disabled.json"

    def _ensure_dirs(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)

    # -- EULA ---------------------------------------------------------------

    def _load_eulas(self) -> dict:
        if not self._eulas_path.exists():
            return {}
        try:
            data = json.loads(self._eulas_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_eulas(self, data: dict) -> None:
        self._ensure_dirs()
        self._eulas_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def eula_accept(self, key: str, text: str) -> None:
        """Record acceptance of `text` for `key` (pack id or node id)."""
        data = self._load_eulas()
        data[key] = {_EULA_HASH: _text_fingerprint(text)}
        self._save_eulas(data)

    def eula_accepted_for(self, key: str, text: str) -> bool:
        """True when this exact text was accepted for `key`."""
        if not text:
            return True
        data = self._load_eulas()
        record = data.get(key)
        return bool(
            record
            and isinstance(record, dict)
            and record.get(_EULA_HASH) == _text_fingerprint(text)
        )

    def eula_accepted(self, manifest: ExtensionManifest) -> bool:
        return self.eula_accepted_for(manifest.id, manifest.info.eula)

    # -- app terms -----------------------------------------------------------

    def _load_terms(self) -> dict:
        if not self._terms_path.exists():
            return {}
        try:
            data = json.loads(self._terms_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_terms(self, data: dict) -> None:
        self._ensure_dirs()
        self._terms_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def terms_accepted(self, version: str, text: str) -> bool:
        """True when this exact text was accepted under `version`."""
        data = self._load_terms()
        record = data.get(version)
        return bool(
            record
            and isinstance(record, dict)
            and record.get(_EULA_HASH) == _text_fingerprint(text)
        )

    def terms_accept(self, version: str, text: str) -> None:
        """Record acceptance of `text` under `version` (fingerprint + date)."""
        data = self._load_terms()
        data[version] = {
            _EULA_HASH: _text_fingerprint(text),
            "accepted_at": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
        }
        self._save_terms(data)

    # -- disabled extensions -------------------------------------------------

    def _load_disabled(self) -> list:
        if not self._disabled_path.exists():
            return []
        try:
            data = json.loads(self._disabled_path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (OSError, json.JSONDecodeError):
            return []

    def _save_disabled(self, ids: list) -> None:
        self._ensure_dirs()
        self._disabled_path.write_text(
            json.dumps(ids, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def disabled_ids(self) -> set[str]:
        """Ids (node or pack) of disabled extensions."""
        return set(self._load_disabled())

    def set_disabled(self, key: str, disabled: bool) -> None:
        """Enable/disable an extension by id (persisted in disabled.json)."""
        ids = self.disabled_ids()
        if disabled:
            ids.add(key)
        else:
            ids.discard(key)
        self._save_disabled(sorted(ids))

    # -- resolution ---------------------------------------------------------

    def resolve(self, manifest: ExtensionManifest) -> PackStatus:
        """Overall state for an extension pack (consent only)."""
        if manifest.info.has_eula and not self.eula_accepted(manifest):
            return PackStatus(PACK_EULA_PENDING, "EULA acceptance required")
        return PackStatus(PACK_INSTALLED, "Installed")

    def resolve_node(self, node_id: str, eula_text: str) -> PackStatus:
        """Overall state for a standalone node extension (consent only)."""
        if eula_text and not self.eula_accepted_for(node_id, eula_text):
            return PackStatus(PACK_EULA_PENDING, "EULA acceptance required")
        return PackStatus(PACK_INSTALLED, "Installed")

    def node_block_reason(self, node_id: str, eula_text: str) -> str | None:
        """None = usable; otherwise a blocking message (EULA not accepted)."""
        status = self.resolve_node(node_id, eula_text)
        if status.state == PACK_INSTALLED:
            return None
        return (
            f"You must accept the EULA of '{node_id}' before using it. "
            "See File > Extensions."
        )

    def block_reason(self, manifest: ExtensionManifest) -> str | None:
        """None = usable; otherwise a human-readable blocking message.

        Used by the canvas (add-node gate, GUI thread) and by the pipeline
        (controller worker gate — never shows dialogs, just a message).
        """
        status = self.resolve(manifest)
        if status.state == PACK_INSTALLED:
            return None
        return (
            f"You must accept the EULA of '{manifest.id}' before using it. "
            "See File > Extensions."
        )


def _text_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
