# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Vendor CLI for extension authoring (no Qt).

Run with:  .venv\\Scripts\\python -m lynceus.tools.extension_tool <command>

Commands:
  scaffold   create the skeleton of a new pack (--kind node|locale|theme)
  pack       bundle node .py files, a folder of nodes, or a whole pack
             (folder with manifest.json) into a .lxpkg artifact
  validate   check a node file, pack folder or payload file WITHOUT
             installing it: errors fail, warnings advise

There is no licensing machinery: the core is GPL-3.0-or-later. Third-party
packs as derivative works inherit GPL. The author may relicense their own
extensions under commercial terms. The scaffold stamps attribution metadata
(author, default SPDX GPL-3.0-or-later) and a copyright/SPDX header on each
node file.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from datetime import date
from pathlib import Path, PurePosixPath
from lynceus.plugins.importer import ExtensionImportError, _safe_zip_entries

from lynceus.plugins.manifest import (
    ExtensionManifest,
    ManifestError,
    SCHEMA_VERSION,
)
from lynceus.plugins.registry import PACKAGE_SUFFIXES, ast_meta

_CURRENT_YEAR = date.today().year


# ---------------------------------------------------------------------------


def cmd_scaffold(args: argparse.Namespace) -> None:
    target = Path(args.path)
    pack_id = args.name or target.name
    author = args.author.strip()
    if not author:
        sys.exit("--author is required (attribution, manifest v2)")

    if args.kind == "locale":
        _scaffold_locale(target, pack_id, author, args)
        return

    if args.kind == "theme":
        _scaffold_theme(target, pack_id, author, args)
        return

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "id": pack_id,
        "version": "0.1.0",
        "kind": "node",
        "display_name": pack_id.replace("-", " ").title(),
        "description": "",
        "author": author,
        "license": "GPL-3.0-or-later",
        "payload": {
            "nodes": [{"id": f"{pack_id}.node1", "file": "nodes/node1.py"}]
        },
    }
    if args.homepage:
        manifest["homepage"] = args.homepage
    if args.repository:
        manifest["repository"] = args.repository
    if args.issues:
        manifest["issues"] = args.issues
    if args.documentation:
        manifest["documentation"] = args.documentation
    if args.donate:
        manifest["donate"] = args.donate
    if args.tags:
        manifest["tags"] = [
            t.strip() for t in args.tags.split(",") if t.strip()
        ]
    if args.contributors:
        manifest["contributors"] = [
            c.strip() for c in args.contributors.split(",") if c.strip()
        ]

    try:
        ExtensionManifest.from_dict(manifest, root=str(target))
    except ManifestError as exc:
        sys.exit(f"Manifest validation failed: {exc}")

    nodes_dir = target / "nodes"
    manifest_path = target / "manifest.json"
    if manifest_path.exists() and not args.force:
        sys.exit(f"{manifest_path} already exists. Use --force to overwrite.")
    nodes_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    node_path = nodes_dir / "node1.py"
    if not node_path.exists():
        header = (
            f"# SPDX-License-Identifier: GPL-3.0-or-later\n"
            f"# Copyright (c) {_CURRENT_YEAR} {author}\n"
        )
        if args.repository:
            header += f"# Source: {args.repository}\n"
        node_path.write_text(
            header
            + _NODE_SKELETON.replace("{node_id}", f"{pack_id}.node1")
            .replace("{author}", author)
            .replace("{pack_id}", pack_id),
            encoding="utf-8",
        )
    print(f"Scaffold created at {target}")
    print(f"  author: {author}")
    print(f"  license: {manifest['license']} (omit to default to GPL-3.0-or-later)")


def _scaffold_locale(target: Path, pack_id: str, author: str, args) -> None:
    """Locale pack skeleton: manifest kind='locale' + a starter catalog."""
    unsupported = [
        opt
        for opt, val in (
            ("--homepage", args.homepage),
            ("--issues", args.issues),
            ("--documentation", args.documentation),
            ("--donate", args.donate),
            ("--tags", args.tags),
            ("--contributors", args.contributors),
        )
        if val
    ]
    if unsupported:
        sys.exit(
            f"{', '.join(unsupported)} not supported for --kind locale "
            "(only --repository is kept)"
        )
    lang = args.lang or "es"
    catalog_name = f"{lang}.json"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "id": pack_id,
        "version": "0.1.0",
        "kind": "locale",
        "display_name": pack_id.replace("-", " ").title(),
        "description": f"{lang} language pack for LynceusScan",
        "author": author,
        "license": "GPL-3.0-or-later",
        "payload": {"catalog": catalog_name},
    }
    if args.repository:
        manifest["repository"] = args.repository

    try:
        ExtensionManifest.from_dict(manifest, root=str(target))
    except ManifestError as exc:
        sys.exit(f"Manifest validation failed: {exc}")

    manifest_path = target / "manifest.json"
    if manifest_path.exists() and not args.force:
        sys.exit(f"{manifest_path} already exists. Use --force to overwrite.")
    target.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    catalog_path = target / catalog_name
    if not catalog_path.exists():
        catalog_path.write_text(
            json.dumps(
                {lang: {"Source English string": "Translation"}},
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    print(f"Locale scaffold created at {target} ({catalog_name})")
    print(f"  author: {author}")


def _scaffold_theme(target: Path, pack_id: str, author: str, args) -> None:
    """Theme pack skeleton: manifest kind='theme' + a starter palette."""
    from lynceus.plugins.theme import THEME_ROLES

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "id": pack_id,
        "version": "0.1.0",
        "kind": "theme",
        "display_name": pack_id.replace("-", " ").title(),
        "description": "Color theme for LynceusScan",
        "author": author,
        "license": "GPL-3.0-or-later",
        "payload": {"palette": "theme.json"},
    }
    if args.repository:
        manifest["repository"] = args.repository
    if args.tags:
        manifest["tags"] = [
            t.strip() for t in args.tags.split(",") if t.strip()
        ]

    try:
        ExtensionManifest.from_dict(manifest, root=str(target))
    except ManifestError as exc:
        sys.exit(f"Manifest validation failed: {exc}")

    manifest_path = target / "manifest.json"
    if manifest_path.exists() and not args.force:
        sys.exit(f"{manifest_path} already exists. Use --force to overwrite.")
    target.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    palette_path = target / "theme.json"
    if not palette_path.exists():
        palette_path.write_text(
            json.dumps(dict(THEME_ROLES), indent=2) + "\n",
            encoding="utf-8",
        )
    print(f"Theme scaffold created at {target} (theme.json)")
    print(f"  author: {author}")
    print("  edit theme.json roles (any subset of the 12); missing = default")


_NODE_SKELETON = '''"""Node skeleton for extension {node_id}."""

from lynceus.nodes.ports import PortType

NODE_ID = "{node_id}"
NODE_NAME = "Node 1"
NODE_CATEGORY = "Pack"
NODE_AUTHOR = "{author}"
NODE_FOLDER = "{pack_id}"
NODE_DESCRIPTION = "TODO: describe what this node does."

# --- Custom port types (optional): declare each type the node uses
# beyond the builtin PortType enum. Each entry is a dict with:
#   id (unique string), display_name, color (hex), compatible (tuple of
#   input ids this output can feed), viewer_kind (optional: reuse an
#   existing viewer like "raster", "vector", "table_csv", "point_cloud").
# The registry reads PORT_TYPE_DEFS via AST at discovery (no execution).
PORT_TYPE_DEFS = [
    # {"id": "my.type", "display_name": "My Type", "color": "#a4d43d",
    #  "compatible": ("my.type",), "viewer_kind": "raster"},
]

# Optional trust/attribution metadata (all optional).
# NODE_CONTRIBUTORS = ["Name One", {"name": "Name Two", "role": "docs", "years": "2026"}]
# NODE_TAGS = ["lidar", "analysis"]
# NODE_HOMEPAGE = "https://example.com"
# NODE_REPOSITORY = "https://example.com/repo"
# NODE_ISSUES = "https://example.com/issues"
# NODE_DOCUMENTATION = "https://example.com/docs"
# NODE_DONATE = "https://example.com/donate"
# NODE_COPYRIGHT = "2026 {author}"

INPUTS = (PortType.POINT_CLOUD,)   # mix with custom strings: (PortType.CHM_MOSAIC, "my.type")
OUTPUTS = (PortType.POINT_CLOUD,)
'''


# ---------------------------------------------------------------------------
# validate: static checks without installing anything
# ---------------------------------------------------------------------------


def _builtin_port_id(port_id: str) -> str | None:
    """A builtin PortType value, or None (custom ids resolve via DEFS)."""
    from lynceus.nodes.ports import PortType

    for member in PortType:
        if member.value == port_id:
            return member.value
    return None


def _used_port_id(item: object) -> str | None:
    """Port id referenced by an INPUTS/OUTPUTS entry (any supported form)."""
    from lynceus.nodes.ports import PortType

    if isinstance(item, PortType):
        return item.value
    if isinstance(item, str):
        return item
    if isinstance(item, (tuple, list)) and item:
        return _used_port_id(item[0])
    port_type = getattr(item, "port_type", None)
    if isinstance(port_type, PortType):
        return port_type.value
    if isinstance(port_type, str):
        return port_type
    return None


def _calls_scoped_file(path: Path) -> bool:
    """True when any task function calls scoped_file (AST, no execution).

    Combined with a declared session_file and no output_files (mosaic
    pattern) this is the classic dangling-path bug: inputs must write
    session/<own_iid>/<session_file>, while scoped_file lands in the
    session root where nobody looks for inputs.
    """
    import ast as _ast

    try:
        tree = _ast.parse(path.read_text(encoding="utf-8-sig"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return False
    for node in _ast.walk(tree):
        if not isinstance(node, _ast.FunctionDef):
            continue
        for call in _ast.walk(node):
            if not isinstance(call, _ast.Call):
                continue
            func = call.func
            if isinstance(func, _ast.Name) and func.id == "scoped_file":
                return True
            if isinstance(func, _ast.Attribute) and func.attr == "scoped_file":
                return True
    return False


_PLACEHOLDER_TOKENS = frozenset(
    {"test", "tests", "testing", "todo", "xxx", "example", "placeholder"}
)


def _looks_like_placeholder(author: object) -> bool:
    """Placeholder author names (never ship these in a distributed pack)."""
    import re

    if not isinstance(author, str):
        return False
    lowered = author.lower()
    if "your name" in lowered:
        return True
    tokens = set(re.split(r"[^a-z]+", lowered))
    return not tokens.isdisjoint(_PLACEHOLDER_TOKENS)


def cmd_validate(args: argparse.Namespace) -> None:
    target = Path(args.path)
    if not target.exists():
        sys.exit(f"Not found: {target}")
    errors: list[str] = []
    warnings: list[str] = []

    manifest_path = (
        target / "manifest.json"
        if target.is_dir()
        else target
        if target.name == "manifest.json"
        else None
    )
    if target.suffix.lower() == ".py":
        _validate_node_file(target, errors, warnings)
    elif target.suffix.lower() in PACKAGE_SUFFIXES:
        _validate_package(target, errors, warnings)
    elif manifest_path is not None and manifest_path.exists():
        _validate_pack(manifest_path, errors, warnings)
    elif target.is_dir():
        _validate_node_folder(target, errors, warnings)
    elif target.suffix.lower() == ".json":
        _validate_payload_file(target, errors, warnings)
    else:
        sys.exit(
            f"Cannot validate '{target.name}': expected a node .py, "
            "a node folder, a package (.lxpkg/.lxext), a pack "
            "folder/manifest.json or a payload .json"
        )

    for warning in warnings:
        print(f"warning: {warning}")
    if errors:
        for error in errors:
            print(f"error: {error}")
        sys.exit(f"{len(errors)} error(s) in {target}")
    print(f"OK: {target}" + (" (with warnings)" if warnings else ""))


def _validate_node_file(
    path: Path, errors: list[str], warnings: list[str]
) -> None:
    from lynceus.plugins.registry import _ast_specs, ast_meta

    meta = ast_meta(path)
    if not meta.get("NODE_ID"):
        errors.append(f"'{path.name}': no NODE_ID declared (not a node)")
        return
    node_id = meta["NODE_ID"]
    if not meta.get("NODE_AUTHOR"):
        errors.append(
            f"'{node_id}': no NODE_AUTHOR (attribution is required)"
        )
    elif _looks_like_placeholder(meta["NODE_AUTHOR"]):
        warnings.append(
            f"'{node_id}': NODE_AUTHOR {meta['NODE_AUTHOR']!r} looks like "
            "a placeholder: use your real name before distributing"
        )
    if not meta.get("NODE_NAME"):
        warnings.append(f"'{node_id}': no NODE_NAME (shows the id instead)")
    if "INPUTS" not in meta:
        warnings.append(
            f"'{node_id}': no INPUTS declared (assumed source-only)"
        )
    if "OUTPUTS" not in meta:
        errors.append(f"'{node_id}': no OUTPUTS declared")
    own_ids = {
        e.get("id")
        for e in meta.get("PORT_TYPE_DEFS") or ()
        if isinstance(e, dict)
    }
    for side in ("INPUTS", "OUTPUTS"):
        for item in meta.get(side) or ():
            pid = _used_port_id(item)
            if (
                isinstance(pid, str)
                and pid not in own_ids
                and _builtin_port_id(pid) is None
            ):
                warnings.append(
                    f"'{node_id}': {side[:-1].lower()} port '{pid}' is "
                    "neither builtin nor declared here: it resolves at "
                    "discovery when the declaring file is installed"
                )
    for entry in meta.get("PORT_TYPE_DEFS") or ():
        for key in ("id", "display_name", "color", "compatible"):
            if key not in entry:
                errors.append(
                    f"'{node_id}': PORT_TYPE_DEFS entry missing '{key}'"
                )
        color = entry.get("color", "")
        if not (
            isinstance(color, str)
            and len(color) == 7
            and color.startswith("#")
        ):
            warnings.append(
                f"'{node_id}': port '{entry.get('id')}' color "
                f"{color!r} is not #rrggbb"
            )
        own_ids = {
            e.get("id")
            for e in meta.get("PORT_TYPE_DEFS") or ()
            if isinstance(e, dict)
        }
        for compat in entry.get("compatible") or ():
            if not isinstance(compat, str):
                continue
            if compat not in own_ids and _builtin_port_id(compat) is None:
                warnings.append(
                    f"'{node_id}': port '{entry.get('id')}' compatible "
                    f"with unknown '{compat}' (typo?)"
                )
    specs = _ast_specs(path)
    if specs.get("input_file_key") and not specs.get("barrier_task"):
        warnings.append(
            f"'{node_id}': input_file_key without barrier_task: "
            "the Browse widget shows but no import runs"
        )
    if specs.get("barrier_task") and not specs.get("tile_task"):
        if not specs.get("input_file_key") and not meta.get("INPUTS"):
            warnings.append(
                f"'{node_id}': sourceless barrier without input_file_key: "
                "runs headless with ctx['file_path'] (no Browse widget; "
                "add input_file_key + session_file for a file input)"
            )
    if specs.get("input_file_key") and not specs.get("session_file"):
        warnings.append(
            f"'{node_id}': input without session_file: "
            "the copy needs a canonical session name"
        )
    if (
        specs.get("barrier_task")
        and not specs.get("tile_task")
        and meta.get("OUTPUTS")
        and not specs.get("output_files")
        and not specs.get("session_file")
    ):
        warnings.append(
            f"'{node_id}': barrier with file outputs but no output_files "
            "or session_file: downstream nodes receive no path and reuse "
            "never matches (normalize to a fixed session_file)"
        )
    if (
        specs.get("session_file")
        and not specs.get("output_files")
        and _calls_scoped_file(path)
    ):
        errors.append(
            f"'{node_id}': declares session_file but writes via "
            "scoped_file (session root): inputs must write "
            "session/<own_iid>/<session_file>, otherwise downstream "
            "receives a dangling path"
        )


def _validate_node_folder(
    folder: Path, errors: list[str], warnings: list[str]
) -> None:
    """Validate every node .py in a folder (no manifest needed).

    Multi-file authors share port ids across files: duplicates are checked
    across the whole folder, like a pack without manifest.
    """
    found = sorted(folder.glob("*.py"))
    if not found:
        errors.append(f"No .py files found in folder: {folder}")
        return
    collected: list = []
    for node_file in found:
        _validate_node_file(node_file, errors, warnings)
        _collect_port_defs(node_file, node_file.name, collected)
    _check_duplicate_ports(collected, errors, warnings)


def _validate_pack(
    manifest_path: Path, errors: list[str], warnings: list[str]
) -> None:
    from lynceus.plugins.manifest import KIND_LOCALE, KIND_NODE, KIND_THEME

    try:
        manifest = ExtensionManifest.load(manifest_path)
    except ManifestError as exc:
        errors.append(f"Manifest error: {exc}")
        return
    pack_dir = manifest_path.parent
    if manifest.kind == KIND_NODE:
        if not manifest.nodes:
            errors.append("Node pack declares no payload.nodes")
        collected: list = []
        for node in manifest.nodes:
            node_file = pack_dir / node.file
            if not node_file.is_file():
                errors.append(f"Missing node file: {node.file}")
            else:
                _validate_node_file(node_file, errors, warnings)
                _collect_port_defs(node_file, node.file, collected)
        _check_duplicate_ports(collected, errors, warnings)
    elif manifest.kind == KIND_LOCALE:
        catalog = pack_dir / manifest.catalog
        if not catalog.is_file():
            errors.append(f"Missing catalog file: {manifest.catalog}")
        else:
            _validate_payload_file(catalog, errors, warnings)
    elif manifest.kind == KIND_THEME:
        palette = pack_dir / manifest.palette
        if not palette.is_file():
            errors.append(f"Missing palette file: {manifest.palette}")
        else:
            _validate_payload_file(palette, errors, warnings)


def _collect_port_defs(path: Path, source: str, collected: list) -> None:
    """Gather (port_id, source, canonical-json) entries for dup checks."""
    from lynceus.plugins.registry import ast_meta

    for entry in ast_meta(path).get("PORT_TYPE_DEFS") or ():
        if isinstance(entry, dict) and isinstance(entry.get("id"), str):
            collected.append(
                (
                    entry["id"],
                    source,
                    json.dumps(entry, sort_keys=True, default=str),
                )
            )


def _check_duplicate_ports(
    collected: list, errors: list[str], warnings: list[str]
) -> None:
    """Conflicting duplicate ids fail; identical repeats only warn.

    Registration is last-wins in discovery order, so a conflict silently
    rewires cables: that must never pass quietly. Identical repeats
    (producer + consumer declaring the same type) work but stay fragile
    to divergence, hence the warning.
    """
    first: dict[str, tuple[str, str]] = {}
    repeated: dict[str, set[str]] = {}
    conflicted: set[str] = set()
    for port_id, source, canon in collected:
        if port_id not in first:
            first[port_id] = (source, canon)
        elif first[port_id][1] != canon:
            conflicted.add(port_id)
            errors.append(
                f"Conflicting PORT_TYPE_DEFS '{port_id}': "
                f"{first[port_id][0]} vs {source} "
                "(last registration wins silently)"
            )
        else:
            repeated.setdefault(port_id, set()).update(
                [first[port_id][0], source]
            )
    for port_id in sorted(set(repeated) - conflicted):
        warnings.append(
            f"PORT_TYPE_DEFS '{port_id}' repeated identically in "
            f"{', '.join(sorted(repeated[port_id]))} "
            "(declare once if you can)"
        )


def _validate_payload_file(
    path: Path, errors: list[str], warnings: list[str]
) -> None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"'{path.name}': unreadable JSON ({exc})")
        return
    if not isinstance(data, dict):
        errors.append(f"'{path.name}': top level must be a JSON object")
        return
    if not data:
        warnings.append(f"'{path.name}': empty payload file")
        return
    # Theme palette: flat {role: #rrggbb} (unknown roles/invalid values
    # only warn: the app degrades to defaults, never breaks).
    if all(isinstance(value, str) for value in data.values()):
        from lynceus.plugins.theme import THEME_ROLES, _validated_roles

        kept = _validated_roles(path.stem, data)
        for role in data:
            if role not in kept:
                warnings.append(
                    f"'{path.name}': role '{role}' ignored "
                    f"(unknown or not #rrggbb; known: "
                    f"{', '.join(sorted(THEME_ROLES))})"
                )
        if len(kept) < len(THEME_ROLES):
            warnings.append(
                f"'{path.name}': partial palette "
                f"({len(kept)}/{len(THEME_ROLES)} roles, rest default)"
            )
        return
    # Locale catalog: {"lang": {"source": "target"}}.
    for lang, entries in data.items():
        if not isinstance(entries, dict):
            errors.append(
                f"'{path.name}': language '{lang}' must map to an object"
            )
            continue
        for src, dst in entries.items():
            if not isinstance(dst, str) or not dst.strip():
                warnings.append(
                    f"'{path.name}' [{lang}]: empty translation for "
                    f"{src!r} (falls back to English)"
                )
            elif dst.count("{") != dst.count("}") or dst.count(
                "{"
            ) != src.count("{"):
                warnings.append(
                    f"'{path.name}' [{lang}]: placeholder mismatch in "
                    f"{src!r}"
                )


def _validate_package(
    path: Path, errors: list[str], warnings: list[str]
) -> None:
    """Validate a finished .lxpkg/.lxext artifact without installing it.

    Unsafe zip entries fail loudly; otherwise each payload entry is
    checked in a temp dir with the same rules as its unpacked form
    (node files, pack manifest + payload, sidecar orphans).
    """
    import tempfile

    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        errors.append(f"'{path.name}' is not a valid package ({exc})")
        return
    with archive:
        try:
            entries = _safe_zip_entries(archive)
        except ExtensionImportError as exc:
            errors.append(f"'{path.name}': {exc}")
            return
        names = {info.filename for info in entries}
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            collected: list = []
            for info in entries:
                probe = tmp_dir / PurePosixPath(info.filename)
                probe.parent.mkdir(parents=True, exist_ok=True)
                probe.write_bytes(archive.read(info))
                if info.filename == "manifest.json":
                    continue
                if probe.suffix.lower() == ".py":
                    _validate_node_file(probe, errors, warnings)
                    _collect_port_defs(probe, info.filename, collected)
                elif probe.suffix.lower() == ".json":
                    _validate_payload_file(probe, errors, warnings)
            if "manifest.json" in names:
                _validate_pack(tmp_dir / "manifest.json", errors, warnings)
            else:
                # Bundle of standalone nodes: dup check across the bundle.
                _check_duplicate_ports(collected, errors, warnings)


def cmd_pack(args: argparse.Namespace) -> None:
    raw = [Path(f) for f in args.files]
    if not raw:
        sys.exit("At least one node file (.py) or a folder is required")

    # Pack mode: a folder with manifest.json or a bare manifest.json file.
    if len(raw) == 1 and _is_pack_source(raw[0]):
        _cmd_pack_pack(raw[0], args)
        return

    # Folders expand to their .py files; files without NODE_ID are skipped
    # with a warning (helpers), while explicitly named files must be nodes.
    files: list[Path] = []
    skipped: list[str] = []
    for path in raw:
        if path.is_dir():
            found = sorted(path.glob("*.py"))
            if not found:
                sys.exit(f"No .py files found in folder: {path}")
            for f in found:
                if ast_meta(f).get("NODE_ID"):
                    files.append(f)
                else:
                    skipped.append(f.name)
        else:
            files.append(path)
    if not files:
        sys.exit("No node files found (none declare NODE_ID)")

    out = Path(args.output) if args.output else Path("extensions.lxpkg")
    if not args.output and len(raw) == 1 and raw[0].is_dir():
        out = Path(raw[0].name + PACKAGE_SUFFIXES[0])
    if out.exists() and out.is_dir():
        sys.exit(f"Output is a directory, give a file name: {out}")
    if not out.suffix:
        out = Path(str(out) + PACKAGE_SUFFIXES[0])
    if out.suffix.lower() not in PACKAGE_SUFFIXES:
        sys.exit(f"Output must use one of {'/'.join(PACKAGE_SUFFIXES)}")

    nodes: list[tuple[Path, str]] = []
    for f in files:
        if not f.exists():
            sys.exit(f"File not found: {f}")
        meta = ast_meta(f)
        node_id = meta.get("NODE_ID")
        if not node_id:
            sys.exit(
                f"'{f.name}' is not a node extension: no NODE_ID declared"
            )
        if not meta.get("NODE_AUTHOR"):
            sys.exit(
                f"'{f.name}' declares no NODE_AUTHOR (attribution). "
                "Add NODE_AUTHOR = \"Your Name\" and retry."
            )
        nodes.append((f, node_id))

    # <node>.i18n.json sidecars ride along with their node files.
    sidecars: list[Path] = []
    for f, _ in nodes:
        sidecar = f.with_name(f"{f.stem}.i18n.json")
        if sidecar.exists():
            sidecars.append(sidecar)

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        seen: set[str] = set()
        for f, _ in nodes:
            if f.name in seen:
                sys.exit(
                    f"Duplicate file name in package: '{f.name}' "
                    "rename one of the node files and retry."
                )
            seen.add(f.name)
            zf.write(f, arcname=f.name)
        for sc in sidecars:
            zf.write(sc, arcname=sc.name)
    ids = ", ".join(n[1] for n in nodes)
    print(f"Packed {len(nodes)} node(s) [{ids}] -> {out}")
    if sidecars:
        print(
            f"  + {len(sidecars)} translation sidecar(s): "
            f"{', '.join(sc.name for sc in sidecars)}"
        )
    if skipped:
        print(f"Skipped {len(skipped)} non-node file(s): {', '.join(skipped)}")


def _is_pack_source(path: Path) -> bool:
    if path.is_dir() and (path / "manifest.json").exists():
        return True
    return path.is_file() and path.name == "manifest.json"


def _cmd_pack_pack(source: Path, args: argparse.Namespace) -> None:
    """Pack mode: zip a whole pack (manifest + payload files + extras)."""
    pack_dir = source if source.is_dir() else source.parent
    try:
        manifest = ExtensionManifest.load(pack_dir / "manifest.json")
    except ManifestError as exc:
        sys.exit(f"Manifest error: {exc}")

    out = Path(args.output) if args.output else Path(f"{manifest.id}.lxpkg")
    if out.exists() and out.is_dir():
        sys.exit(f"Output is a directory, give a file name: {out}")
    if not out.suffix:
        out = Path(str(out) + PACKAGE_SUFFIXES[0])
    if out.suffix.lower() not in PACKAGE_SUFFIXES:
        sys.exit(f"Output must use one of {'/'.join(PACKAGE_SUFFIXES)}")

    files: list[Path] = []
    for path in sorted(pack_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(pack_dir)
        if any(
            part.startswith(".") or part in ("cache", "__pycache__")
            for part in rel.parts
        ):
            continue
        if path.suffix.lower() == ".pyc":
            continue
        files.append(path)
    if not files:
        sys.exit(f"Pack folder is empty: {pack_dir}")

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, arcname=str(f.relative_to(pack_dir)))
    print(
        f"Packed {manifest.kind} pack [{manifest.id}] "
        f"v{manifest.version} ({len(files)} file(s)) -> {out}"
    )


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="extension_tool", description="LynceusScan extension authoring tool"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("scaffold", help="create an extension skeleton")
    p.set_defaults(func=cmd_scaffold)
    p.add_argument("path", help="target folder")
    p.add_argument("--name", help="pack id (default: folder name)")
    p.add_argument("--author", required=True, help="pack author (attribution)")
    p.add_argument("--kind", choices=("node", "locale", "theme"), default="node",
                   help="extension kind (default: node)")
    p.add_argument("--lang", default="es",
                   help="locale catalog language code (default: es)")
    p.add_argument("--homepage", help="project URL")
    p.add_argument("--repository", help="canonical source repository URL")
    p.add_argument("--issues", help="bug tracker URL (optional)")
    p.add_argument("--documentation", help="docs URL (optional)")
    p.add_argument("--donate", help="funding/donate URL (optional)")
    p.add_argument("--tags", help="comma-separated keywords (optional)")
    p.add_argument("--contributors", help="comma-separated names (optional)")
    p.add_argument("--force", action="store_true",
                   help="overwrite manifest.json (node file/catalog preserved)")

    p = sub.add_parser(
        "pack",
        help="bundle node files, a folder of nodes, or a whole pack "
        "(folder with manifest.json) into a package",
    )
    p.set_defaults(func=cmd_pack)
    p.add_argument(
        "files",
        nargs="+",
        help="node .py files, a node folder, or a pack folder/manifest.json",
    )
    p.add_argument("-o", "--output", help=f"output path (default: extensions{PACKAGE_SUFFIXES[0]}, <folder>{PACKAGE_SUFFIXES[0]} for a single folder, or <pack>{PACKAGE_SUFFIXES[0]} for a pack)")

    p = sub.add_parser(
        "validate",
        help="check a node file, pack folder or payload file without "
        "installing it (errors fail, warnings advise)",
    )
    p.set_defaults(func=cmd_validate)
    p.add_argument(
        "path",
        help="node .py file, pack folder/manifest.json or payload .json",
    )

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
