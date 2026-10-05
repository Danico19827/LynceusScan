# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Extension manifest schema v2 (no Qt).

The manifest is the single contract an artifact follows: a pack with
`kind`-typed payload. kind="node", kind="locale" and kind="theme" are
supported; future kinds reuse the same lifecycle (discovery, consent,
loading) with their own payload shape.

There is no licensing machinery: the core is GPL-3.0-or-later and extensions
load in-process, so distributed packs inherit GPL (documented in
lynceus/plugins/terms.py). The manifest carries attribution and trust
metadata, with `author` required and everything else optional:

  author       required - who made the pack (display + attribution)
  contributors optional - list of people who contributed (name + optional
                          email/role/years); they keep control of their work
                           and are never relicensed to commercial terms
                          without their consent (see CLA.md). `credits` is
                          accepted as an alias.
  homepage     optional - project page (http/https URL)
  repository   optional - canonical source repository (http/https URL)
  issues       optional - bug tracker (http/https URL)
  documentation optional - docs page (http/https URL)
  donate       optional - funding/donate page (http/https URL)
  changelog    optional - release notes (http/https URL or file name)
  tags         optional - list of short keywords
  compatibility optional - list of app versions the pack is tested against
  copyright    optional - copyright line/notice (e.g. "2026 Jane Doe")
  license      optional - SPDX id; omitted means GPL-3.0-or-later
  eula         optional - consent text shown once before using the pack
  disclaimer   optional - short text embedded in the products this pack's
                          nodes produce (attribution to the node's creator,
                          e.g. algorithms generated with AI assistance whose
                          precision is unverified); node files may override
                          it with NODE_DISCLAIMER

Validation is soft: `license` only needs to look like an SPDX id (no
compatibility allowlist - the GPL inheritance rule is legal, not enforced),
and URLs must be http(s) when present. Optional fields never block loading
an existing pack, so v2 manifests without them keep working unchanged.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2
KIND_NODE = "node"
KIND_LOCALE = "locale"
KIND_THEME = "theme"
_SUPPORTED_KINDS = (KIND_NODE, KIND_LOCALE, KIND_THEME)

_PACK_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_VERSION_RE = re.compile(r"^\d+\.\d+(\.\d+)?$")
_SLUG_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
_URL_RE = re.compile(r"^https?://", re.IGNORECASE)
_SPDX_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.\-]*$")

DEFAULT_LICENSE = "GPL-3.0-or-later"
"""Project default SPDX id used when the manifest omits `license`."""


class ManifestError(ValueError):
    """Malformed or unsupported extension manifest."""


@dataclass(frozen=True)
class Contributor:
    name: str
    email: str = ""
    role: str = ""
    years: str = ""

    @property
    def label(self) -> str:
        """Display string: name (+ role/years when present)."""
        parts = [self.name]
        if self.role:
            parts.append(self.role)
        if self.years:
            parts.append(f"({self.years})")
        return " · ".join(parts)


@dataclass(frozen=True)
class PackInfo:
    author: str
    homepage: str = ""
    repository: str = ""
    license: str = ""
    eula: str = ""
    disclaimer: str = ""
    contributors: tuple[Contributor, ...] = ()
    tags: tuple[str, ...] = ()
    issues: str = ""
    documentation: str = ""
    donate: str = ""
    changelog: str = ""
    compatibility: tuple[str, ...] = ()
    copyright: str = ""

    @property
    def license_label(self) -> str:
        return self.license or DEFAULT_LICENSE

    @property
    def has_eula(self) -> bool:
        return bool(self.eula)


@dataclass(frozen=True)
class NodeSpec:
    id: str
    file: str


@dataclass(frozen=True)
class ExtensionManifest:
    schema_version: int
    id: str
    version: str
    kind: str
    display_name: str
    description: str
    min_app_version: str | None
    info: PackInfo
    nodes: tuple[NodeSpec, ...]
    root: Path
    catalog: str = ""
    palette: str = ""

    def node_by_id(self, node_id: str) -> NodeSpec | None:
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def catalog_file(self) -> Path:
        """Resolve the locale catalog file relative to the root (path-safe)."""
        return self._payload_file(self.catalog, "Catalog")

    def palette_file(self) -> Path:
        """Resolve the theme palette file relative to the root (path-safe)."""
        return self._payload_file(self.palette, "Palette")

    def _payload_file(self, name: str, label: str) -> Path:
        """Resolve a payload file relative to the manifest root (path-safe)."""
        path = (self.root / name).resolve()
        root = self.root.resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ManifestError(
                f"{label} file '{name}' escapes the extension folder"
            ) from exc
        return path

    def node_file(self, node: NodeSpec) -> Path:
        """Resolve a node file relative to the manifest root (path-safe)."""
        path = (self.root / node.file).resolve()
        root = self.root.resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ManifestError(
                f"Node file '{node.file}' escapes the extension folder"
            ) from exc
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ExtensionManifest":
        path = Path(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ManifestError(
                f"Cannot read manifest {path}: {exc}"
            ) from exc
        return cls.from_dict(data, root=path.parent)

    @classmethod
    def from_dict(cls, data: dict, root: str | Path) -> "ExtensionManifest":
        if not isinstance(data, dict):
            raise ManifestError("Manifest must be a JSON object")

        schema = data.get("schema_version")
        if schema != SCHEMA_VERSION:
            raise ManifestError(
                f"Unsupported manifest schema_version {schema!r} "
                f"(expected {SCHEMA_VERSION})"
            )

        pack_id = data.get("id")
        if not isinstance(pack_id, str) or not _PACK_ID_RE.match(pack_id):
            raise ManifestError("Manifest needs a valid 'id' for the extension")

        version = data.get("version")
        if not isinstance(version, str) or not _VERSION_RE.match(version):
            raise ManifestError(
                f"Manifest '{pack_id}' has an invalid 'version' ({version!r})"
            )

        kind = data.get("kind", KIND_NODE)
        if kind not in _SUPPORTED_KINDS:
            raise ManifestError(
                f"Extension '{pack_id}' kind '{kind}' is not supported yet"
            )

        display_name = data.get("display_name") or pack_id
        description = data.get("description") or ""
        min_app = data.get("min_app_version")

        info = cls._parse_info(pack_id, data)
        payload = data.get("payload")
        if kind == KIND_LOCALE:
            nodes: tuple[NodeSpec, ...] = ()
            catalog = cls._parse_catalog(pack_id, payload)
            palette = ""
        elif kind == KIND_THEME:
            nodes = ()
            catalog = ""
            palette = cls._parse_palette(pack_id, payload)
        else:
            nodes = cls._parse_nodes(pack_id, payload)
            catalog = ""
            palette = ""

        return cls(
            schema_version=schema,
            id=pack_id,
            version=version,
            kind=kind,
            display_name=str(display_name),
            description=str(description),
            min_app_version=str(min_app) if min_app else None,
            info=info,
            nodes=nodes,
            root=Path(root).resolve(),
            catalog=catalog,
            palette=palette,
        )

    @staticmethod
    def _parse_catalog(pack_id: str, raw: Any) -> str:
        """Locale packs declare their catalog file in payload.catalog."""
        if raw is None or not isinstance(raw, dict):
            raise ManifestError(
                f"Extension '{pack_id}': locale packs need 'payload.catalog'"
            )
        catalog = raw.get("catalog")
        if not isinstance(catalog, str) or not catalog:
            raise ManifestError(
                f"Extension '{pack_id}': locale packs need 'payload.catalog' "
                "(e.g. \"es.json\")"
            )
        if not _SLUG_RE.match(catalog):
            raise ManifestError(
                f"Extension '{pack_id}': invalid catalog file name {catalog!r}"
            )
        return catalog

    @staticmethod
    def _parse_palette(pack_id: str, raw: Any) -> str:
        """Theme packs declare their palette file in payload.palette."""
        if raw is None or not isinstance(raw, dict):
            raise ManifestError(
                f"Extension '{pack_id}': theme packs need 'payload.palette'"
            )
        palette = raw.get("palette")
        if not isinstance(palette, str) or not palette:
            raise ManifestError(
                f"Extension '{pack_id}': theme packs need 'payload.palette' "
                '(e.g. "theme.json")'
            )
        if not _SLUG_RE.match(palette):
            raise ManifestError(
                f"Extension '{pack_id}': invalid palette file name {palette!r}"
            )
        return palette

    @staticmethod
    def _parse_info(pack_id: str, data: dict) -> PackInfo:
        author = data.get("author")
        if not isinstance(author, str) or not author.strip():
            raise ManifestError(
                f"Extension '{pack_id}': 'author' is required (attribution)"
            )

        homepage = data.get("homepage") or ""
        repository = data.get("repository") or ""
        for label, value in (("homepage", homepage), ("repository", repository)):
            if value and not _URL_RE.match(str(value)):
                raise ManifestError(
                    f"Extension '{pack_id}': '{label}' must be an http(s) URL"
                )

        license_id = data.get("license") or ""
        if license_id and not _SPDX_RE.match(str(license_id)):
            raise ManifestError(
                f"Extension '{pack_id}': 'license' must be a valid SPDX id "
                f"(got {license_id!r}); omit it to default to "
                f"{DEFAULT_LICENSE}"
            )

        eula = data.get("eula") or ""
        # Optional trust metadata (soft validation; never blocks loading).
        issues = data.get("issues") or ""
        documentation = data.get("documentation") or ""
        donate = data.get("donate") or ""
        changelog = str(data.get("changelog") or "")
        copyright_line = str(data.get("copyright") or "")
        for label, value in (
            ("issues", issues),
            ("documentation", documentation),
            ("donate", donate),
        ):
            if value and not _URL_RE.match(str(value)):
                raise ManifestError(
                    f"Extension '{pack_id}': '{label}' must be an http(s) URL"
                )

        contributors = ExtensionManifest._parse_contributors(
            pack_id, data.get("contributors", data.get("credits"))
        )
        tags = ExtensionManifest._parse_string_list(
            pack_id, data.get("tags", []), "tags"
        )
        compatibility = ExtensionManifest._parse_string_list(
            pack_id, data.get("compatibility", []), "compatibility"
        )
        return PackInfo(
            author=str(author).strip(),
            homepage=str(homepage),
            repository=str(repository),
            license=str(license_id),
            eula=str(eula),
            disclaimer=str(data.get("disclaimer") or ""),
            contributors=contributors,
            tags=tags,
            issues=str(issues),
            documentation=str(documentation),
            donate=str(donate),
            changelog=changelog,
            compatibility=compatibility,
            copyright=copyright_line,
        )

    @staticmethod
    def _parse_string_list(pack_id: str, raw: Any, field: str) -> tuple[str, ...]:
        """Parse an optional list-of-strings field (soft: invalid => empty)."""
        if raw is None or raw == []:
            return ()
        if not isinstance(raw, list):
            raise ManifestError(
                f"Extension '{pack_id}': '{field}' must be a list"
            )
        values: list[str] = []
        for item in raw:
            if not isinstance(item, str) or not item.strip():
                raise ManifestError(
                    f"Extension '{pack_id}': '{field}' items must be non-empty "
                    "strings"
                )
            values.append(item.strip())
        return tuple(values)

    @staticmethod
    def _parse_contributors(
        pack_id: str, raw: Any
    ) -> tuple[Contributor, ...]:
        """Parse optional contributors (strings or {name,...} objects)."""
        if raw is None or raw == []:
            return ()
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list):
            raise ManifestError(
                f"Extension '{pack_id}': 'contributors' must be a list"
            )
        out: list[Contributor] = []
        for item in raw:
            if isinstance(item, str):
                if not item.strip():
                    raise ManifestError(
                        f"Extension '{pack_id}': contributor name is empty"
                    )
                out.append(Contributor(name=item.strip()))
                continue
            if not isinstance(item, dict):
                raise ManifestError(
                    f"Extension '{pack_id}': each contributor must be a string "
                    "or an object with a 'name'"
                )
            name = item.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ManifestError(
                    f"Extension '{pack_id}': contributor object needs a 'name'"
                )
            out.append(
                Contributor(
                    name=name.strip(),
                    email=str(item.get("email") or ""),
                    role=str(item.get("role") or ""),
                    years=str(item.get("years") or ""),
                )
            )
        return tuple(out)

    @staticmethod
    def _parse_nodes(pack_id: str, raw: Any) -> tuple[NodeSpec, ...]:
        if raw is None or not isinstance(raw, dict) or "nodes" not in raw:
            raise ManifestError(f"Extension '{pack_id}': missing payload.nodes")
        nodes_value = raw["nodes"]
        if not isinstance(nodes_value, list) or not nodes_value:
            raise ManifestError(f"Extension '{pack_id}': payload.nodes must be a list")

        nodes: list[NodeSpec] = []
        seen: set[str] = set()
        for entry in nodes_value:
            if not isinstance(entry, dict):
                raise ManifestError(
                    f"Extension '{pack_id}': each node entry must be an object"
                )
            node_id = entry.get("id")
            node_file = entry.get("file")
            if not isinstance(node_id, str) or not isinstance(node_file, str):
                raise ManifestError(
                    f"Extension '{pack_id}': node entry needs 'id' and 'file'"
                )
            if not node_id.startswith(f"{pack_id}.") or not _SLUG_RE.match(
                node_id.replace(f"{pack_id}.", "", 1)
            ):
                raise ManifestError(
                    f"Extension '{pack_id}': node id '{node_id}' must be "
                    f"'{pack_id}.<name>'"
                )
            if node_id in seen:
                raise ManifestError(
                    f"Extension '{pack_id}': duplicate node id '{node_id}'"
                )
            seen.add(node_id)
            nodes.append(NodeSpec(id=node_id, file=node_file))
        return tuple(nodes)
