# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Extension import (domain, no Qt).

Turns a source artifact into installed extensions inside the extensions
directory, routed to the subfolder matching its kind:

  - node file (.py): standalone node extension. The module must declare
    NODE_ID and NODE_AUTHOR (read via AST, never executed). It is copied
    to extensions/nodes/<NODE_FOLDER | NODE_ID prefix>/ keeping its filename.
  - package (.lxpkg/.lxext): a ZIP. Without a manifest.json it is a
    *bundle* of standalone node files, which are validated and extracted
    into extensions/nodes/ (the package is consumed when it lives inside
    the extensions dir). With a manifest.json of kind theme/locale it is
    extracted into extensions/themes/<id>/ or extensions/locales/<id>/
    (uniform with folder packs, no cache round-trip). A node-kind pack
    with a manifest stays a legacy pack, copied as-is (materialized to
    cache/ at discovery).
  - pack folder (folder with manifest.json): copied into the subfolder for
    its kind (nodes/ for node packs, locales/ for locale packs, themes/
    for theme packs).

Import is idempotent: importing the same content twice is a no-op.
Overwriting an existing file with different content requires force=True.
"""

from __future__ import annotations

import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from lynceus.plugins.manifest import (
    KIND_LOCALE,
    KIND_NODE,
    KIND_THEME,
    ExtensionManifest,
    ManifestError,
)
from lynceus.plugins.registry import PACKAGE_SUFFIXES, ast_meta

_FOLDER_RE = re.compile(r"^[A-Za-z0-9_-]+$")

# Kind folders: everything the app imports lands under a subfolder matching
# its kind, so extensions/ stays organized by type (nodes/ locales/ themes/...).
KIND_DIR_NODE = "nodes"
KIND_DIR_LOCALE = "locales"
KIND_DIR_THEME = "themes"
_KIND_DIRS = {
    KIND_NODE: KIND_DIR_NODE,
    KIND_LOCALE: KIND_DIR_LOCALE,
    KIND_THEME: KIND_DIR_THEME,
}


def _ensure_within(root: Path, candidate: Path) -> None:
    """Guard against an import destination escaping the extensions dir."""
    if root == candidate:
        return
    if root not in candidate.parents:
        raise ExtensionImportError("Import destination escapes extensions dir")


def _same_content(a: Path, b: Path) -> bool:
    """Content equality ignoring CRLF/LF differences (Windows-safe)."""
    if a.read_bytes() == b.read_bytes():
        return True
    return _normalize_newlines(a.read_bytes()) == _normalize_newlines(
        b.read_bytes()
    )


def _normalize_newlines(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


class ExtensionImportError(ValueError):
    """The source cannot be imported as an extension."""


class ExtensionRemoveError(RuntimeError):
    """An installed extension could not be removed."""


@dataclass(frozen=True)
class ImportResult:
    kind: str  # "node" | "pack" | "bundle"
    destination: Path
    node_id: str = ""
    created: bool = True  # False when already installed (idempotent no-op)

    @property
    def message(self) -> str:
        verb = "Imported" if self.created else "Already installed"
        if self.kind == "bundle":
            return f"{verb} bundle ({self.node_id}) -> {self.destination}"
        if self.kind == "node":
            return f"{verb} node '{self.node_id}' -> {self.destination}"
        return f"{verb} pack -> {self.destination}"


def import_extension(
    source: str | Path,
    extensions_dir: str | Path,
    *,
    force: bool = False,
) -> ImportResult:
    """Install `source` (node .py, package or pack folder) into extensions."""
    src = Path(source)
    if not src.exists():
        raise ExtensionImportError(f"Source not found: {src}")
    target = Path(extensions_dir).resolve()

    if src.is_dir():
        if not (src / "manifest.json").exists():
            raise ExtensionImportError(
                f"'{src}' is not a pack: no manifest.json inside"
            )
        return _import_pack(src, target, force)
    if src.name == "manifest.json":
        return _import_pack(src.parent, target, force)
    if src.suffix.lower() in PACKAGE_SUFFIXES:
        return _import_package(src, target, force)
    if src.suffix.lower() == ".py":
        return _import_node(src, target, force)
    raise ExtensionImportError(
        f"Unsupported extension source: {src.name} "
        f"(expected .py, {'/'.join(PACKAGE_SUFFIXES)} or manifest.json)"
    )


# ---------------------------------------------------------------------------
# Node files (.py)
# ---------------------------------------------------------------------------


def _import_node(source: Path, target: Path, force: bool) -> ImportResult:
    meta = ast_meta(source)
    node_id = meta.get("NODE_ID")
    if not node_id:
        raise ExtensionImportError(
            f"'{source.name}' is not a node extension: no NODE_ID declared"
        )
    author = meta.get("NODE_AUTHOR")
    if not author:
        raise ExtensionImportError(
            f"'{source.name}' declares no NODE_AUTHOR (attribution). "
            'Add NODE_AUTHOR = "Your Name" to the file and retry.'
        )

    folder = meta.get("NODE_FOLDER")
    if not folder:
        prefix = node_id.split(".", 1)[0] if "." in node_id else ""
        folder = prefix
    if not folder or not _FOLDER_RE.match(folder):
        raise ExtensionImportError(
            f"'{source.name}': NODE_ID must be '<pack>.<node>' or the file "
            f"must declare NODE_FOLDER (got folder {folder!r})"
        )
    prefix = node_id.split(".", 1)[0] if "." in node_id else node_id
    if folder != prefix:
        raise ExtensionImportError(
            f"'{source.name}': NODE_FOLDER '{folder}' must match the "
            f"NODE_ID prefix '{prefix}' (multi-author collisions stay "
            "impossible by construction)"
        )
    _reject_installed_id(target, node_id, source)

    dest_dir = (target / KIND_DIR_NODE / folder).resolve()
    _ensure_within(target, dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest = dest_dir / source.name
    existing = _existing_with_same_id(dest_dir, node_id)
    if existing is not None and existing != dest:
        if _same_content(existing, source):
            return ImportResult("node", existing, node_id, created=False)
        if not force:
            raise ExtensionImportError(
                f"Node '{node_id}' is already installed at {existing.name} "
                "with different content. Use force=True to overwrite."
            )
        shutil.copy2(source, existing)
        return ImportResult("node", existing, node_id, created=True)
    if dest.exists():
        if _same_content(dest, source):
            return ImportResult("node", dest, node_id, created=False)
        if not force:
            raise ExtensionImportError(
                f"'{dest}' already exists with different content. "
                "Use force=True to overwrite."
            )
    shutil.copy2(source, dest)
    _copy_sidecar(source, dest_dir, force)
    return ImportResult("node", dest, node_id, created=True)


def _copy_sidecar(node_source: Path, dest_dir: Path, force: bool) -> None:
    """Copy the <node>.i18n.json sidecar next to an imported node file."""
    sidecar = node_source.with_name(f"{node_source.stem}.i18n.json")
    if not sidecar.exists():
        return
    dest = dest_dir / sidecar.name
    if dest.exists() and not _same_content(dest, sidecar) and not force:
        raise ExtensionImportError(
            f"'{dest}' already exists with different content. "
            "Use force=True to overwrite."
        )
    shutil.copy2(sidecar, dest)


def _existing_with_same_id(dest_dir: Path, node_id: str) -> Path | None:
    """A .py in `dest_dir` that already declares the same NODE_ID."""
    for candidate in dest_dir.glob("*.py"):
        if ast_meta(candidate).get("NODE_ID") == node_id:
            return candidate
    return None


def _reject_installed_id(target: Path, node_id: str, source: Path) -> None:
    """Refuse a NODE_ID already installed from a different file.

    Same id + same content is the idempotent no-op (handled by callers);
    same id from another file means two authors claim one id: renaming
    is mandatory, overwriting never happens by surprise. Skips cache and
    hidden dirs; unreadable files degrade to no-match instead of blocking.
    """
    for candidate in sorted(target.rglob("*.py")):
        try:
            rel = candidate.relative_to(target)
        except ValueError:
            continue
        if any(
            part.startswith(".") or part == "cache" for part in rel.parts[:-1]
        ):
            continue
        if candidate.resolve() == source.resolve():
            continue
        try:
            installed = ast_meta(candidate).get("NODE_ID")
        except Exception:
            continue
        if installed != node_id:
            continue
        try:
            same = _same_content(candidate, source)
        except OSError:
            same = False
        if same:
            return
        raise ExtensionImportError(
            f"Node '{node_id}' is already installed from "
            f"{candidate.name} with different content: rename one of "
            "the node ids (ids are global)."
        )


# ---------------------------------------------------------------------------
# Packages (.lxpkg / .lxext): bundle of nodes, or legacy pack with manifest
# ---------------------------------------------------------------------------


def _import_package(source: Path, target: Path, force: bool) -> ImportResult:
    try:
        archive = zipfile.ZipFile(source)
    except zipfile.BadZipFile as exc:
        raise ExtensionImportError(
            f"'{source.name}' is not a valid package: {exc}"
        ) from exc

    with archive:
        entries = _safe_zip_entries(archive)
        names = {info.filename for info in entries}
        if "manifest.json" in names:
            return _import_pack_artifact(source, target, force)

        py_entries = [info for info in entries if info.filename.endswith(".py")]
        if not py_entries:
            raise ExtensionImportError(
                f"'{source.name}' contains no node files (.py)"
            )
        i18n_entries = [
            info
            for info in entries
            if info.filename.endswith(".i18n.json")
        ]

        # Validate everything BEFORE copying anything into place.
        plans: list[tuple[str, Path, str, Path]] = []
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            for info in py_entries:
                probe = tmp_dir / PurePosixPath(info.filename).name
                probe.write_bytes(archive.read(info))
                plans.append(_bundle_plan(probe, info.filename, target))

            for filename, probe, node_id, dest_dir in plans:
                dest = dest_dir / probe.name
                existing = _existing_with_same_id(dest_dir, node_id)
                if existing is not None and existing != dest:
                    if not _same_content(existing, probe) and not force:
                        raise ExtensionImportError(
                            f"Node '{node_id}' already installed at "
                            f"{existing.name} with different content. "
                            "Use force=True to overwrite."
                        )
                elif dest.exists() and not _same_content(dest, probe) and not force:
                    raise ExtensionImportError(
                        f"'{dest}' already exists with different content. "
                        "Use force=True to overwrite."
                    )

            # Sidecar catalogs ride along, matched by node file stem.
            i18n_plans: list[tuple[str, Path, Path]] = []
            for info in i18n_entries:
                probe = tmp_dir / PurePosixPath(info.filename).name
                probe.write_bytes(archive.read(info))
                stem = PurePosixPath(info.filename).name.removesuffix(
                    ".i18n.json"
                )
                matching = next(
                    (p for p in plans if p[1].stem == stem), None
                )
                if matching is None:
                    continue  # orphan catalog: skip
                dest = matching[3] / probe.name
                i18n_plans.append((info.filename, probe, dest))

            for filename, probe, dest in i18n_plans:
                if dest.exists() and not _same_content(dest, probe) and not force:
                    raise ExtensionImportError(
                        f"'{dest}' already exists with different content. "
                        "Use force=True to overwrite."
                    )

            created_any = False
            for filename, probe, node_id, dest_dir in plans:
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = dest_dir / probe.name
                if dest.exists() and _same_content(dest, probe):
                    continue
                shutil.copy2(probe, dest)
                created_any = True
            for filename, probe, dest in i18n_plans:
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists() and _same_content(dest, probe):
                    continue
                shutil.copy2(probe, dest)

    # A package dropped inside extensions/ is consumed after extraction.
    if source.resolve().parent == target:
        source.unlink(missing_ok=True)

    ids = ", ".join(plan[2] for plan in plans)
    return ImportResult("bundle", target, node_id=ids, created=created_any)


def _bundle_plan(probe: Path, entry_name: str, target: Path):
    """Validate one node file from a bundle; return (name, probe, id, dest)."""
    meta = ast_meta(probe)
    node_id = meta.get("NODE_ID")
    if not node_id:
        raise ExtensionImportError(
            f"'{entry_name}' is not a node extension: no NODE_ID declared"
        )
    if not meta.get("NODE_AUTHOR"):
        raise ExtensionImportError(
            f"'{entry_name}' declares no NODE_AUTHOR (attribution). "
            "Add NODE_AUTHOR = \"Your Name\" and repack."
        )
    folder = meta.get("NODE_FOLDER")
    if not folder:
        folder = node_id.split(".", 1)[0] if "." in node_id else ""
    if not folder or not _FOLDER_RE.match(folder):
        raise ExtensionImportError(
            f"'{entry_name}': NODE_ID must be '<pack>.<node>' or the file "
            f"must declare NODE_FOLDER (got folder {folder!r})"
        )
    prefix = node_id.split(".", 1)[0] if "." in node_id else node_id
    if folder != prefix:
        raise ExtensionImportError(
            f"'{entry_name}': NODE_FOLDER '{folder}' must match the "
            f"NODE_ID prefix '{prefix}' (multi-author collisions stay "
            "impossible by construction)"
        )
    _reject_installed_id(target, node_id, probe)
    dest_dir = (target / KIND_DIR_NODE / folder).resolve()
    _ensure_within(target, dest_dir)
    return entry_name, probe, node_id, dest_dir


def _safe_zip_entries(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    """Zip entries without traversal/absolute paths (skips dirs)."""
    result: list[zipfile.ZipInfo] = []
    for info in archive.infolist():
        name = info.filename.replace("\\", "/")
        if name.startswith("/") or re.match(r"^[A-Za-z]:", name):
            raise ExtensionImportError(
                f"Unsafe path in package: {info.filename!r}"
            )
        parts = [p for p in name.split("/") if p not in ("", ".")]
        if any(p == ".." for p in parts):
            raise ExtensionImportError(
                f"Unsafe path in package: {info.filename!r}"
            )
        if info.is_dir():
            continue
        result.append(info)
    return result


def _manifest_from_zip(source: Path) -> ExtensionManifest | None:
    """Parse the root manifest.json inside a package, or None when absent."""
    import json

    try:
        archive = zipfile.ZipFile(source)
    except zipfile.BadZipFile as exc:
        raise ExtensionImportError(
            f"'{source.name}' is not a valid package: {exc}"
        ) from exc
    with archive:
        try:
            raw = archive.read("manifest.json")
        except KeyError:
            return None
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ExtensionImportError(
            f"'{source.name}' has an unreadable manifest.json: {exc}"
        ) from exc
    try:
        return ExtensionManifest.from_dict(data, root=source.parent)
    except ManifestError as exc:
        raise ExtensionImportError(f"Invalid manifest: {exc}") from exc


def _extract_kind_pack(
    source: Path, manifest: ExtensionManifest, target: Path, force: bool
) -> ImportResult:
    """Extract a theme/locale pack package into its kind folder.

    Uniform with folder packs: the content lands in
    ``extensions/<kind>/<id>/`` and discovery registers it directly,
    with no cache round-trip. A package dropped inside extensions/ is
    consumed after extraction (mirroring bundles).
    """
    kind_dir = _KIND_DIRS.get(manifest.kind, KIND_DIR_NODE)
    dest = (target / kind_dir / manifest.id).resolve()
    _ensure_within(target, dest)
    if dest.exists():
        if not force:
            raise ExtensionImportError(
                f"Pack '{manifest.id}' already installed at {dest}. "
                "Use force=True to overwrite."
            )
        shutil.rmtree(dest, ignore_errors=True)
    with zipfile.ZipFile(source) as archive:
        for info in _safe_zip_entries(archive):
            parts = [
                p
                for p in info.filename.replace("\\", "/").split("/")
                if p not in ("", ".")
            ]
            out = dest.joinpath(*parts)
            _ensure_within(dest, out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(archive.read(info))
    if source.resolve().parent == target:
        source.unlink(missing_ok=True)
    return ImportResult("pack", dest, created=True)


def _import_pack_artifact(source: Path, target: Path, force: bool) -> ImportResult:
    """Pack with manifest.json: extract theme/locale, else copy legacy."""
    manifest = _manifest_from_zip(source)
    if manifest is not None and manifest.kind in (KIND_THEME, KIND_LOCALE):
        return _extract_kind_pack(source, manifest, target, force)
    dest = target / source.name
    if dest.exists():
        if _same_content(dest, source):
            return ImportResult("pack", dest, created=False)
        if not force:
            raise ExtensionImportError(
                f"'{dest}' already exists with different content. "
                "Use force=True to overwrite."
            )
    shutil.copy2(source, dest)
    return ImportResult("pack", dest, created=True)


# ---------------------------------------------------------------------------
# Removal
# ---------------------------------------------------------------------------


def remove_node_file(file_path: str | Path) -> None:
    """Delete an installed standalone node (.py + .i18n.json sidecar).

    The folder is removed too when it becomes empty.
    """
    path = Path(file_path)
    sidecar = path.with_name(f"{path.stem}.i18n.json")
    path.unlink(missing_ok=True)
    sidecar.unlink(missing_ok=True)
    try:
        path.parent.rmdir()
    except OSError:
        pass  # folder not empty (sibling nodes) or busy


def remove_pack(
    manifest: ExtensionManifest, extensions_dir: str | Path
) -> None:
    """Delete an installed pack.

    Folder packs remove the whole folder. Packs materialized from a
    package (.lxpkg/.lxext) remove the cache entry and the source file.
    """
    import shutil

    root = manifest.root
    target = Path(extensions_dir).resolve()
    cache_root = target / "cache"
    try:
        root.relative_to(cache_root)
    except ValueError:
        shutil.rmtree(root, ignore_errors=True)
        return

    shutil.rmtree(root, ignore_errors=True)
    parent = root.parent
    while parent != cache_root:
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent
    for suffix in PACKAGE_SUFFIXES:
        (target / f"{manifest.id}{suffix}").unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Packs (folder with manifest.json)
# ---------------------------------------------------------------------------


def _import_pack(pack_dir: Path, target: Path, force: bool) -> ImportResult:
    manifest_path = pack_dir / "manifest.json"
    try:
        manifest = ExtensionManifest.load(manifest_path)
    except ManifestError as exc:
        raise ExtensionImportError(f"Invalid manifest: {exc}") from exc

    kind_dir = _KIND_DIRS.get(manifest.kind, KIND_DIR_NODE)
    dest = (target / kind_dir / manifest.id).resolve()
    _ensure_within(target, dest)
    if dest == pack_dir.resolve():
        return ImportResult("pack", dest, created=False)
    if dest.exists():
        if not force:
            raise ExtensionImportError(
                f"Pack '{manifest.id}' already installed at {dest}. "
                "Use force=True to overwrite."
            )
        shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(pack_dir, dest)
    return ImportResult("pack", dest, created=True)


# ---------------------------------------------------------------------------
# First-run seeding (frozen installer builds)
# ---------------------------------------------------------------------------


def seed_bundled_extensions(
    target: str | Path | None = None,
    bundled: str | Path | None = None,
) -> list[str]:
    """Copy bundled packs into the user extensions dir (frozen first runs).

    The installer ships public packs (locales, themes, optionally private
    nodes added at packaging time) inside the read-only bundle; the running
    app uses ``%APPDATA%/LynceusScan/extensions`` instead, so this copies
    every bundled pack/file that is not already installed. Existing user
    copies (including newer versions) are never overwritten: the user's
    copy always wins. Returns the seeded entry names. No-op when not
    frozen (source checkouts read the repo folder directly).
    """
    import sys

    from lynceus.resources import project_root

    if not getattr(sys, "frozen", False):
        return []
    from lynceus.plugins.registry import DEFAULT_EXTENSIONS_DIR

    src_root = Path(bundled) if bundled is not None else project_root() / "extensions"
    dest_root = Path(target) if target is not None else DEFAULT_EXTENSIONS_DIR
    if not src_root.is_dir():
        return []
    seeded: list[str] = []
    for kind_dir in sorted(
        p for p in src_root.iterdir() if p.is_dir() and not p.name.startswith(".")
    ):
        if kind_dir.name == "cache":
            continue
        dest_kind = dest_root / kind_dir.name
        for entry in sorted(kind_dir.iterdir()):
            if entry.name.startswith("."):
                continue
            dest = dest_kind / entry.name
            if dest.exists():
                continue
            if entry.is_dir():
                if not (entry / "manifest.json").is_file():
                    continue
                dest_kind.mkdir(parents=True, exist_ok=True)
                shutil.copytree(entry, dest)
                seeded.append(f"{kind_dir.name}/{entry.name}")
            elif entry.suffix.lower() == ".py":
                dest_kind.mkdir(parents=True, exist_ok=True)
                shutil.copy2(entry, dest)
                seeded.append(f"{kind_dir.name}/{entry.name}")
    return seeded
