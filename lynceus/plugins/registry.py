# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Node registry + extension discovery (domain, no Qt).

Resolves a global node id to its implementation. Built-in nodes keep their
module-path ids ("lynceus.nodes.*") for full backward/forward compatibility
with saved projects; external nodes use stable ids declared in their files
("<pack>.<node>" by convention). Resolution is lazy: the library reads
metadata via AST (no code execution), and a module is imported only when
the node is actually used (dragged or run).

Extensions come in two flavors:
  - standalone node files: a self-contained .py with NODE_* metadata
    (NODE_ID, NODE_NAME, NODE_AUTHOR, ...), discovered recursively under
    the extensions dir.
  - packs: a folder with manifest.json (or a .lxpkg/.lxext artifact)
    describing non-node kinds (theme, locale, ...); node packs are still
    supported through the legacy distribution path.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import logging
import pkgutil
import re
import sys
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import lynceus.nodes as nodes_package
from lynceus.nodes.ports import PortDef, PortType, register_spec, PortSpec
from lynceus.plugins.manifest import (
    DEFAULT_LICENSE,
    Contributor,
    ExtensionManifest,
    ManifestError,
)

logger = logging.getLogger(__name__)

SYSTEM_EXTENSION_ID = "system"
_META_STR_FIELDS = (
    "NODE_ID",
    "NODE_NAME",
    "NODE_CATEGORY",
    "NODE_SUBCATEGORY",
    "NODE_DESCRIPTION",
    "NODE_AUTHOR",
    "NODE_LICENSE",
    "NODE_EULA",
    "NODE_DISCLAIMER",
    "NODE_HOMEPAGE",
    "NODE_REPOSITORY",
    "NODE_FOLDER",
    "NODE_COPYRIGHT",
    "NODE_ISSUES",
    "NODE_DOCUMENTATION",
    "NODE_DONATE",
    "NODE_CHANGELOG",
    # Strategy-node variants (family + key label): a plain .py that
    # activates a strategy of a base node. Not NODE_ID nodes themselves.
    "VARIANT_OF",
    "VARIANT_KEY",
    "VARIANT_LABEL",
)
_META_PORT_FIELDS = ("INPUTS", "OUTPUTS")
_META_DICT_FIELDS = (
    "NODE_TRANSLATIONS",
    "VARIANT_TARGET",
)
_META_LIST_FIELDS = ("NODE_CONTRIBUTORS", "NODE_TAGS")

from lynceus.resources import project_root


def _default_extensions_dir() -> Path:
    """User-writable extensions dir.

    Frozen builds (installer) keep it under the platform data dir: the
    bundle dir is read-only under Program Files and is wiped on every
    update. Source checkouts use the repo ``extensions/`` folder.
    """
    import sys

    if getattr(sys, "frozen", False):
        from lynceus.plugins.store import data_root

        return data_root() / "extensions"
    return project_root() / "extensions"


DEFAULT_EXTENSIONS_DIR = _default_extensions_dir()
PACKAGE_SUFFIXES = (".lxpkg", ".lxext")
"""Distribution package suffixes (.lxpkg primary; .lxext legacy)."""
_SYNTH_RE = re.compile(r"[^A-Za-z0-9_]")


class NodeLoadError(RuntimeError):
    """A node id could not be imported."""


@dataclass(frozen=True)
class NodeInfo:
    node_id: str
    name: str
    category: str
    description: str
    inputs: tuple | None
    outputs: tuple | None
    extension_id: str
    source: str  # "builtin" | "extension"
    subcategory: str = ""
    manifest: ExtensionManifest | None = None
    file_path: str | None = None
    author: str = ""
    license_label: str = DEFAULT_LICENSE
    homepage: str = ""
    repository: str = ""
    eula: str = ""
    disclaimer: str = ""
    translations: dict | None = None
    contributors: tuple | None = None
    tags: tuple = ()
    issues: str = ""
    documentation: str = ""
    donate: str = ""
    changelog: str = ""
    copyright: str = ""


# ---------------------------------------------------------------------------
# AST metadata extraction (no execution, fast, safe)
# ---------------------------------------------------------------------------


def _ast_literal(node) -> Any:
    """Extract a literal value: str/int/float/tuple/list/dict-of-literals/None."""
    if node is None:
        return None
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        items = [_ast_literal(elt) for elt in node.elts]
        if any(
            item is None and not isinstance(elt, ast.Constant)
            for item, elt in zip(items, node.elts)
        ):
            return None
        return [item for item in items if item is not None]
    if isinstance(node, ast.Tuple):
        items = [_ast_literal(elt) for elt in node.elts]
        if all(item is not None or isinstance(elt, ast.Constant)
               for item, elt in zip(items, node.elts)):
            return tuple(items)
        return None
    if isinstance(node, ast.Dict):
        result: dict[str, Any] = {}
        for key_node, value_node in zip(node.keys, node.values):
            key = _ast_literal(key_node)
            if not isinstance(key, str):
                return None
            value = _ast_literal(value_node)
            if isinstance(value, dict):
                result[key] = value
            elif isinstance(value, str):
                result[key] = value
            elif isinstance(value, (list, tuple)):
                result[key] = value
            else:
                return None
        return result
    return None


def _contributors_from_meta(meta: dict) -> tuple[Contributor, ...]:
    """Normalise raw AST contributors (str or dict entries) to Contributors."""
    raw = meta.get("NODE_CONTRIBUTORS")
    if not raw:
        return ()
    out: list[Contributor] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            out.append(Contributor(name=item.strip()))
        elif isinstance(item, dict) and isinstance(item.get("name"), str):
            out.append(
                Contributor(
                    name=item["name"].strip(),
                    email=str(item.get("email") or ""),
                    role=str(item.get("role") or ""),
                    years=str(item.get("years") or ""),
                )
            )
    return tuple(out)


def _port_from_ast(node) -> PortType | str | PortDef | None:
    if isinstance(node, ast.Attribute):
        value = node.value
        if isinstance(value, ast.Name) and value.id == "PortType":
            try:
                return PortType[node.attr]
            except KeyError:
                return None
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            # String literal: builtin if it matches a PortType value,
            # otherwise an opaque custom id (fully valid).
            for member in PortType:
                if member.value == node.value:
                    return member
            return node.value
    if isinstance(node, ast.Call):
        # PortDef(PortType.X | "custom.id", name=..., required=..., group=...)
        func = node.func
        if not (isinstance(func, ast.Name) and func.id == "PortDef"):
            return None
        port_type = None
        name = None
        required = True
        group = None
        args = list(node.args)
        if args:
            port_type = _port_from_ast(args[0])
            args = args[1:]
        for kw in node.keywords:
            if kw.arg == "port_type" and port_type is None:
                port_type = _port_from_ast(kw.value)
            elif kw.arg == "name":
                literal = _ast_literal(kw.value)
                name = literal if isinstance(literal, str) else None
            elif kw.arg == "required":
                literal = _ast_literal(kw.value)
                required = literal if isinstance(literal, bool) else True
            elif kw.arg == "group":
                literal = _ast_literal(kw.value)
                group = literal if isinstance(literal, str) else None
        for extra in args:
            literal = _ast_literal(extra)
            if isinstance(literal, str) and name is None:
                name = literal
            elif isinstance(literal, bool):
                required = literal
        if port_type is None:
            return None
        return PortDef(port_type=port_type, name=name, required=required, group=group)
    return None


def ast_meta(path: Path) -> dict:
    """Module-level literal metadata parsed WITHOUT executing the file."""
    try:
        source = path.read_text(encoding="utf-8-sig")  # utf-8-sig: strips BOM
    except OSError:
        return {}
    try:
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return {}

    meta: dict = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        name = target.id
        if name in _META_STR_FIELDS:
            value = _ast_literal(node.value)
            if isinstance(value, str):
                meta[name] = value
        elif name in _META_PORT_FIELDS:
            value = _ast_literal(node.value)
            if isinstance(value, tuple) and all(
                isinstance(p, (PortType, str)) for p in value
            ):
                meta[name] = value
            elif isinstance(node.value, ast.Tuple):
                ports: list[PortType | str] = []
                ok = True
                for elt in node.value.elts:
                    port = _port_from_ast(elt)
                    if port is None:
                        ok = False
                        break
                    ports.append(port)
                if ok:
                    meta[name] = tuple(ports)
        elif name in _META_DICT_FIELDS:
            value = _ast_literal(node.value)
            if isinstance(value, dict):
                meta[name] = value
        elif name in _META_LIST_FIELDS:
            value = _ast_literal(node.value)
            if isinstance(value, list) and all(
                isinstance(item, (str, dict)) and (
                    isinstance(item, str) and bool(item.strip())
                    or isinstance(item, dict) and isinstance(item.get("name"), str)
                )
                for item in value
            ):
                meta[name] = value
        elif name == "PORT_TYPE_DEFS":
            value = _ast_literal(node.value)
            if isinstance(value, (list, tuple)) and all(
                isinstance(d, dict)
                and isinstance(d.get("id"), str)
                and isinstance(d.get("display_name"), str)
                and isinstance(d.get("color"), str)
                and isinstance(d.get("compatible"), (list, tuple))
                for d in value
            ):
                meta[name] = tuple(value)
    return meta


# PROCESSING_SPECS keys the UI may read without executing the module
# (file-input convention: input_file_key + file_filters; output_files is
# a {port: basename} dict of plain strings).
_SPECS_STR_KEYS = (
    "barrier_task",
    "tile_task",
    "input_file_key",
    "file_filters",
    "output_port",
    "session_file",
    "kind_group",
    "kind_label",
)
_SPECS_DICT_KEYS = ("output_files",)


def _ast_specs(path: Path) -> dict:
    """PROCESSING_SPECS entries parsed WITHOUT execution.

    Plain string values for the known string keys, string-to-string
    dicts for the known dict keys; anything exotic is skipped per key
    instead of failing the whole dict.
    """
    try:
        source = path.read_text(encoding="utf-8-sig")
    except OSError:
        return {}
    try:
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id != "PROCESSING_SPECS":
            continue
        if not isinstance(node.value, ast.Dict):
            return {}
        specs: dict = {}
        for key_node, value_node in zip(node.value.keys, node.value.values):
            if not (
                isinstance(key_node, ast.Constant)
                and isinstance(key_node.value, str)
            ):
                continue
            key = key_node.value
            if key in _SPECS_STR_KEYS and isinstance(
                value_node, ast.Constant
            ):
                if isinstance(value_node.value, str):
                    specs[key] = value_node.value
            elif key in _SPECS_DICT_KEYS and isinstance(
                value_node, ast.Dict
            ):
                table = _ast_literal(value_node)
                if isinstance(table, dict) and all(
                    isinstance(k, str) and isinstance(v, str)
                    for k, v in table.items()
                ):
                    specs[key] = dict(table)
        return specs
    return {}


def _module_spec_origin(module_name: str) -> Path | None:
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ValueError):
        return None
    if spec is None or spec.origin is None:
        return None
    path = Path(spec.origin)
    if path.suffix == ".py":
        return path
    # Frozen bundles import from the archive (no .py origin): the AST
    # discovery reads the node sources shipped as data instead.
    if module_name == "lynceus.nodes" or module_name.startswith(
        "lynceus.nodes."
    ):
        from lynceus.resources import resource_path

        candidate = resource_path("/".join(module_name.split(".")) + ".py")
        if candidate.is_file():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Plugin manager
# ---------------------------------------------------------------------------


class PluginManager:
    """Discovers extensions and resolves node ids to modules."""

    def __init__(
        self,
        extensions_dir: str | Path | None = None,
        disabled_provider: Any | None = None,
    ):
        self._extensions_dir = Path(
            extensions_dir or DEFAULT_EXTENSIONS_DIR
        ).resolve()
        self.disabled_provider = disabled_provider
        """Callable() -> set[str] of disabled extension ids (node/pack).

        Built-in nodes are always enabled (never filtered). Disabled locale
        packs never register, so their language stays out of `available()`.
        """
        self._extensions: dict[str, ExtensionManifest] = {}
        self._nodes: dict[str, NodeInfo] = {}
        self._modules: dict[str, Any] = {}
        self._variants: dict[str, dict[str, dict]] = {}
        self._variant_modules: dict[tuple[str, str], Any] = {}
        self._specs_cache: dict[str, dict] = {}
        self._discovered = False
        self._disabled: set[str] = set()

    @property
    def extensions_dir(self) -> Path:
        """Where standalone node files and packs are looked up."""
        return self._extensions_dir

    # -- discovery ----------------------------------------------------------

    def discover(self) -> None:
        self._extensions.clear()
        self._nodes.clear()
        self._specs_cache.clear()
        provider = self.disabled_provider
        self._disabled = set(provider()) if callable(provider) else set()
        self._register_builtins()
        self._register_extensions()
        self._discovered = True

    def _register_builtins(self) -> None:
        for module_info in pkgutil.walk_packages(
            nodes_package.__path__, prefix=f"{nodes_package.__name__}."
        ):
            if module_info.ispkg:
                continue
            module_name = module_info.name
            origin = _module_spec_origin(module_name)
            if origin is None:
                continue
            meta = ast_meta(origin)
            family = meta.get("VARIANT_OF")
            key = meta.get("VARIANT_KEY")
            if family and key:
                self._variant(module_name, family, key, str(meta.get("VARIANT_LABEL") or key),
                              meta.get("VARIANT_TARGET"))
                continue
            if not meta.get("NODE_NAME"):
                continue
            self._nodes[module_name] = NodeInfo(
                node_id=module_name,
                name=meta["NODE_NAME"],
                category=meta.get("NODE_CATEGORY", ""),
                subcategory=meta.get("NODE_SUBCATEGORY", ""),
                description=meta.get("NODE_DESCRIPTION", ""),
                inputs=meta.get("INPUTS"),
                outputs=meta.get("OUTPUTS"),
                extension_id=SYSTEM_EXTENSION_ID,
                source="builtin",
            )

    def _register_extensions(self) -> None:
        if not self._extensions_dir.exists():
            return
        self._register_dir(self._extensions_dir)

    def _register_dir(self, path: Path) -> None:
        """Recursively register one directory's contents.

        A folder carrying a manifest.json is a pack and owns its content (its
        inner .py files are registered from the manifest, not visited again).
        Any other folder is a node tree: direct .py files are standalone nodes
        and subfolders recurse. Package artifacts (.lxpkg/.lxext) are
        materialized wherever they sit.
        """
        if (path / "manifest.json").exists():
            try:
                self._register_extension(ExtensionManifest.load(path / "manifest.json"))
            except ManifestError as exc:
                logger.warning("Extension skipped (%s): %s", path, exc)
            except Exception as exc:  # keep app alive
                logger.warning("Extension skipped (%s): %s", path, exc)
            return

        for entry in sorted(path.iterdir()):
            if entry.name.startswith(".") or entry.name == "cache":
                continue
            try:
                if entry.is_dir():
                    self._register_dir(entry)
                elif entry.suffix.lower() in PACKAGE_SUFFIXES:
                    try:
                        self._register_extension(self._materialize_zip(entry))
                    except ManifestError as exc:
                        logger.warning(
                            "Package skipped (%s): %s. Use "
                            "Extensions > Import extension... to install a "
                            "node bundle (package without manifest.json).",
                            entry,
                            exc,
                        )
                elif entry.suffix.lower() == ".py":
                    self._register_node_file(entry)
            except ManifestError as exc:
                logger.warning("Extension skipped (%s): %s", entry, exc)
            except Exception as exc:  # keep app alive
                logger.warning("Extension skipped (%s): %s", entry, exc)

    def _register_node_file(self, path: Path) -> None:
        """Register a self-contained node .py (skips non-node helpers)."""
        meta = ast_meta(path)
        node_id = meta.get("NODE_ID")
        family = meta.get("VARIANT_OF")
        key = meta.get("VARIANT_KEY")
        if family and key and not node_id:
            # Extension variant module: activates a strategy of a base node
            # (multi-file bundles land in the same folder for atomic imports
            # via importer.py; standalone variants work as-is). A variant
            # follows its family: disabling the family id disables the
            # variant too.
            if family in self._disabled:
                return
            self._variant(
                "",
                family,
                key,
                str(meta.get("VARIANT_LABEL") or key),
                meta.get("VARIANT_TARGET"),
                file_path=str(path),
                extension_id=path.parent.name,
            )
            _seed_ports_from_meta(meta)
            return
        if not node_id:
            return  # helper module or non-node file
        _seed_ports_from_meta(meta)
        if node_id in self._disabled:
            return  # disabled extension
        if not meta.get("NODE_NAME"):
            logger.warning("Node file '%s' has no NODE_NAME; skipping", path)
            return
        if node_id in self._nodes:
            logger.warning(
                "Node id '%s' already registered (built-in or duplicate)", node_id
            )
            return
        self._nodes[node_id] = NodeInfo(
            node_id=node_id,
            name=meta["NODE_NAME"],
            category=meta.get("NODE_CATEGORY", ""),
            subcategory=meta.get("NODE_SUBCATEGORY", ""),
            description=meta.get("NODE_DESCRIPTION", ""),
            inputs=meta.get("INPUTS"),
            outputs=meta.get("OUTPUTS"),
            extension_id=path.parent.name,
            source="extension",
            file_path=str(path),
            author=meta.get("NODE_AUTHOR", ""),
            license_label=meta.get("NODE_LICENSE") or DEFAULT_LICENSE,
            homepage=meta.get("NODE_HOMEPAGE", ""),
            repository=meta.get("NODE_REPOSITORY", ""),
            eula=meta.get("NODE_EULA", ""),
            disclaimer=meta.get("NODE_DISCLAIMER", ""),
            translations=meta.get("NODE_TRANSLATIONS"),
            contributors=_contributors_from_meta(meta),
            tags=tuple(meta.get("NODE_TAGS") or ()),
            issues=meta.get("NODE_ISSUES", ""),
            documentation=meta.get("NODE_DOCUMENTATION", ""),
            donate=meta.get("NODE_DONATE", ""),
            changelog=meta.get("NODE_CHANGELOG", ""),
            copyright=meta.get("NODE_COPYRIGHT", ""),
        )

    def _variant(
        self,
        module_name: str,
        family_id: str,
        key: str,
        label: str,
        target: dict | None = None,
        file_path: str | None = None,
        extension_id: str = SYSTEM_EXTENSION_ID,
    ) -> None:
        """Register a strategy variant under ``family_id`` (first wins)."""
        variants = self._variants.setdefault(family_id, {})
        if key in variants:
            return
        variants[key] = {
            "key": key,
            "label": label,
            "target": dict(target) if isinstance(target, dict) else {},
            "module": module_name,
            "path": file_path,
            "builtin": bool(module_name),
            "extension_id": extension_id,
        }

    def _register_extension(self, manifest: ExtensionManifest) -> None:
        if manifest.id in self._disabled:
            return  # disabled extension
        if manifest.id in self._extensions:
            logger.warning(
                "Extension id '%s' already loaded (duplicate), skipping", manifest.id
            )
            return
        for node in manifest.nodes:
            file_path = manifest.node_file(node)
            meta = ast_meta(file_path)
            _seed_ports_from_meta(meta)
            declared = meta.get("NODE_ID")
            if declared and declared != node.id:
                logger.warning(
                    "Node '%s' declares NODE_ID=%r but manifest expects '%s'; "
                    "skipping",
                    node.id,
                    declared,
                    node.id,
                )
                continue
            if not meta.get("NODE_NAME"):
                logger.warning(
                    "Node '%s' has no NODE_NAME; skipping", node.id
                )
                continue
            if node.id in self._nodes:
                logger.warning(
                    "Node id '%s' already registered (built-in or duplicate)", node.id
                )
                continue
            self._nodes[node.id] = NodeInfo(
                node_id=node.id,
                name=meta["NODE_NAME"],
                category=meta.get("NODE_CATEGORY", ""),
                subcategory=meta.get("NODE_SUBCATEGORY", ""),
                description=meta.get("NODE_DESCRIPTION", ""),
                inputs=meta.get("INPUTS"),
                outputs=meta.get("OUTPUTS"),
                extension_id=manifest.id,
                source="extension",
                manifest=manifest,
                file_path=str(file_path),
                author=manifest.info.author,
                license_label=manifest.info.license_label,
                homepage=manifest.info.homepage,
                repository=manifest.info.repository,
                eula=manifest.info.eula,
                disclaimer=meta.get("NODE_DISCLAIMER") or manifest.info.disclaimer,
                translations=meta.get("NODE_TRANSLATIONS"),
                contributors=_contributors_from_meta(meta),
                tags=tuple(meta.get("NODE_TAGS") or ()),
                issues=manifest.info.issues,
                documentation=manifest.info.documentation,
                donate=manifest.info.donate,
                changelog=manifest.info.changelog,
                copyright=manifest.info.copyright,
            )
        self._extensions[manifest.id] = manifest

    @staticmethod
    def _materialize_zip(zip_path: Path) -> ExtensionManifest:
        """Extract a .lxpkg/.lxext artifact under the extensions cache, then load."""
        with zipfile.ZipFile(zip_path) as archive:
            names = archive.namelist()
            if "manifest.json" not in names:
                raise ManifestError(f"{zip_path.name}: missing manifest.json")
            manifest_data = None
            for info in archive.infolist():
                if info.filename == "manifest.json":
                    manifest_data = archive.read(info)
            if manifest_data is None:
                raise ManifestError(f"{zip_path.name}: manifest not readable")
            import json

            manifest_dict = json.loads(manifest_data.decode("utf-8"))
            pack_id = manifest_dict.get("id")
            version = manifest_dict.get("version", "0.0.0")
            target_dir = (
                zip_path.parent / "cache" / str(pack_id) / str(version)
            )
            if not (target_dir / "manifest.json").exists():
                tmp = target_dir.parent / f".{target_dir.name}.tmp"
                if tmp.exists():
                    import shutil

                    shutil.rmtree(tmp, ignore_errors=True)
                tmp.mkdir(parents=True, exist_ok=True)
                archive.extractall(tmp)
                tmp.rename(target_dir)
        return ExtensionManifest.load(target_dir / "manifest.json")

    # -- queries ------------------------------------------------------------

    def ensure_discovered(self) -> None:
        if not self._discovered:
            self.discover()

    def list_nodes(self) -> list[NodeInfo]:
        self.ensure_discovered()
        return list(self._nodes.values())

    def contains(self, node_id: str) -> bool:
        self.ensure_discovered()
        return node_id in self._nodes

    def node_info(self, node_id: str) -> NodeInfo | None:
        self.ensure_discovered()
        return self._nodes.get(node_id)

    def node_specs(self, node_id: str) -> dict:
        """String-valued PROCESSING_SPECS without executing the module.

        Powers UI conventions (e.g. the file-input widget) for builtins and
        third-party nodes alike: extension sources resolve via the
        registered file path, builtins via their module path (sources ship
        as data in frozen builds). Never imports anything.
        """
        self.ensure_discovered()
        if node_id in self._specs_cache:
            return self._specs_cache[node_id]
        specs: dict[str, str] = {}
        info = self._nodes.get(node_id)
        path: Path | None = None
        if info is not None and info.file_path:
            path = Path(info.file_path)
        else:
            path = _module_spec_origin(node_id)
        if path is not None:
            specs = _ast_specs(path)
        self._specs_cache[node_id] = specs
        return specs

    def extension_manifests(self) -> dict[str, ExtensionManifest]:
        self.ensure_discovered()
        return dict(self._extensions)

    # -- strategy variants --------------------------------------------------

    def list_variants(self, family_id: str) -> list[dict]:
        """AST metadata of a strategy family's variants, sorted by key."""
        self.ensure_discovered()
        return sorted(
            self._variants.get(family_id, {}).values(), key=lambda v: v["key"]
        )

    def has_variants(self, family_id: str) -> bool:
        self.ensure_discovered()
        return bool(self._variants.get(family_id))

    def variant_info(self, family_id: str, key: str) -> dict | None:
        self.ensure_discovered()
        return self._variants.get(family_id, {}).get(key)

    def variant_meta(self, family_id: str, key: str) -> dict:
        info = self.variant_info(family_id, key)
        return dict(info.get("target", {})) if info else {}

    def import_variant(self, family_id: str, key: str):
        """Import (and cache) the variant module implementing a strategy."""
        self.ensure_discovered()
        info = self._variants.get(family_id, {}).get(key)
        if info is None:
            raise NodeLoadError(f"Unknown strategy variant: {family_id}.{key}")
        cached = self._variant_modules.get((family_id, key))
        if cached is not None:
            return cached
        if info["builtin"]:
            module = importlib.import_module(info["module"])
        else:
            if not info["path"]:
                raise NodeLoadError(f"Variant '{family_id}.{key}' has no source file")
            path = Path(info["path"])
            family_slug = _SYNTH_RE.sub("_", family_id)
            key_slug = _SYNTH_RE.sub("_", key)
            module_name = f"_lynx_var_{family_slug}_{key_slug}"
            module = sys.modules.get(module_name)
            if module is None:
                file_spec = importlib.util.spec_from_file_location(module_name, path)
                if file_spec is None or file_spec.loader is None:
                    raise NodeLoadError(
                        f"Cannot build import spec for variant '{family_id}.{key}'"
                    )
                module = importlib.util.module_from_spec(file_spec)
                sys.modules[module_name] = module
                try:
                    file_spec.loader.exec_module(module)
                except Exception as exc:  # keep app alive
                    sys.modules.pop(module_name, None)
                    raise NodeLoadError(
                        f"Failed to import variant '{family_id}.{key}': {exc}"
                    ) from exc
        _register_module_ports(module)
        self._variant_modules[(family_id, key)] = module
        return module

    # -- ports / import -----------------------------------------------------

    def load_ports(self, node_id: str) -> tuple[tuple, tuple]:
        """INPUTS/OUTPUTS as PortType tuples (imports if AST can't tell)."""
        info = self.node_info(node_id)
        if info is None:
            return (), ()
        if info.inputs is not None and info.outputs is not None:
            return info.inputs, info.outputs
        module = self.import_node(node_id)
        inputs = tuple(getattr(module, "INPUTS", ()))
        outputs = tuple(getattr(module, "OUTPUTS", ()))
        if inputs or outputs:
            # replace() keeps every AST-read field (subcategory, attribution,
            # links); a manual rebuild would silently drop them.
            self._nodes[node_id] = replace(info, inputs=inputs, outputs=outputs)
        return inputs, outputs

    def import_node(self, node_id: str):
        """Import (and cache) the module that implements `node_id`."""
        self.ensure_discovered()
        cached = self._modules.get(node_id)
        if cached is not None:
            return cached
        info = self._nodes.get(node_id)
        if info is None:
            raise NodeLoadError(f"Unknown node id: {node_id}")
        try:
            if info.source == "builtin":
                module = importlib.import_module(node_id)
            else:
                module = self._import_from_file(info)
        except NodeLoadError:
            raise
        except Exception as exc:  # keep app alive
            raise NodeLoadError(
                f"Failed to load node '{node_id}' ({info.name}): {exc}"
            ) from exc
        # Register PORT_TYPE_DEFS from the loaded module (builtins + extensions)
        _register_module_ports(module)
        self._modules[node_id] = module
        return module

    @staticmethod
    def _import_from_file(info: NodeInfo):
        if info.manifest is not None:
            spec = info.manifest.node_by_id(info.node_id)
            if spec is None:
                raise NodeLoadError(f"Node '{info.node_id}' not in manifest")
            path = info.manifest.node_file(spec)
        else:
            if not info.file_path:
                raise NodeLoadError(f"Node '{info.node_id}' has no source file")
            path = Path(info.file_path)
        slug = _SYNTH_RE.sub("_", path.stem)
        pack = _SYNTH_RE.sub("_", info.extension_id)
        module_name = f"_lynx_ext_{pack}_{slug}"
        cached = sys.modules.get(module_name)
        if cached is not None:
            return cached
        file_spec = importlib.util.spec_from_file_location(module_name, path)
        if file_spec is None or file_spec.loader is None:
            raise NodeLoadError(
                f"Cannot build import spec for '{info.node_id}'"
            )
        module = importlib.util.module_from_spec(file_spec)
        sys.modules[module_name] = module
        try:
            file_spec.loader.exec_module(module)
        except Exception as exc:  # keep app alive
            sys.modules.pop(module_name, None)
            raise NodeLoadError(
                f"Failed to import '{info.node_id}': {exc}"
            ) from exc
        declared = getattr(module, "NODE_ID", None)
        if declared and declared != info.node_id:
            raise NodeLoadError(
                f"Module '{info.node_id}' declares NODE_ID={declared!r}"
            )
        # Register any PORT_TYPE_DEFS declared in the module (extensions only)
        _register_module_ports(module)
        return module


def _port_key(port) -> str:
    """Normalize a port ID to its canonical string value."""
    if isinstance(port, PortType):
        return port.value
    return str(port)


def _seed_ports_from_meta(meta: dict) -> None:
    """Register PORT_TYPE_DEFS found in AST metadata."""
    port_defs = meta.get("PORT_TYPE_DEFS")
    if not isinstance(port_defs, (list, tuple)):
        return
    for spec_dict in port_defs:
        if not isinstance(spec_dict, dict):
            continue
        port_id = spec_dict.get("id")
        display_name = spec_dict.get("display_name")
        color = spec_dict.get("color")
        compatible = spec_dict.get("compatible")
        if not (isinstance(port_id, str) and isinstance(display_name, str)
                and isinstance(color, str) and isinstance(compatible, (list, tuple))):
            continue
        viewer_kind = spec_dict.get("viewer_kind")
        if not isinstance(viewer_kind, str):
            viewer_kind = None
        register_spec(PortSpec(
            port_id=port_id,
            display_name=display_name,
            color=color,
            compatible=tuple(_port_key(p) for p in compatible),
            viewer_kind=viewer_kind,
        ))


def _register_module_ports(module) -> None:
    """Scan a loaded node module for PORT_TYPE_DEFS and register them."""
    port_defs = getattr(module, "PORT_TYPE_DEFS", None)
    if isinstance(port_defs, (list, tuple)):
        _seed_ports_from_meta({"PORT_TYPE_DEFS": tuple(port_defs)})


manager = PluginManager()
"""Module-level singleton used by steps/canvas/project/UI."""


def run_file_node_task(module_id: str, fn_name: str, args):
    """Spawn-safe wrapper that runs a node function by its module id.

    Extension modules are loaded from file under a synthetic name that only
    exists in this process; pickling them directly fails in the `spawn`
    workers (the child cannot re-import the synthetic name). This top-level
    function is importable by path, so workers call back into the registry,
    re-discover the extension by AST and load it from its real file.
    """
    module = manager.import_node(module_id)
    fn = getattr(module, fn_name)
    return fn(*args)
