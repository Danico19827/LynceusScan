# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Extensions dialog (Qt UI).

Lists installed extensions - standalone node files and packs - with their
consent state (installed / EULA pending) resolved against the local
ConsentStore. From here the user can import new extensions (node .py,
.lxpkg or pack folder, including drag & drop), accept EULAs and open the
extension folder.

There is no licensing machinery: extensions are open (GPL inherited from
the core, see lynceus/plugins/terms.py) and the only gate is optional EULA
consent.

No Qt in the domain layer: this is a pure adapter, translating
plugins.store / plugins.registry / plugins.importer into widgets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from lynceus.plugins.importer import (
    ExtensionImportError,
    import_extension,
    remove_node_file,
    remove_pack,
)
from lynceus.plugins.locale import t
from lynceus.plugins.manifest import DEFAULT_LICENSE, ExtensionManifest, ManifestError
from lynceus.plugins.registry import ast_meta, manager
from lynceus.plugins.store import (
    PACK_EULA_PENDING,
    PACK_INSTALLED,
    ConsentStore,
)
from lynceus.ui.widgets import CollapsibleSection

_GROUP_LABELS = {
    "node": "Nodes",
    "locale": "Languages",
    "theme": "Themes",
}


def _contributor_labels(raw) -> tuple[str, ...]:
    """Normalise contributors (list of str/dict or of Contributor) to labels."""
    if not raw:
        return ()
    labels: list[str] = []
    for item in raw:
        if hasattr(item, "label") and item.label:
            labels.append(item.label)
        elif isinstance(item, str) and item.strip():
            labels.append(item.strip())
        elif isinstance(item, dict) and item.get("name"):
            parts = [str(item["name"])]
            if item.get("role"):
                parts.append(str(item["role"]))
            if item.get("years"):
                parts.append(f"({item['years']})")
            labels.append(" · ".join(parts))
    return tuple(labels)


def _pack_meta(manifest) -> dict:
    """Unified optional-metadata dict from an ExtensionManifest."""
    info = manifest.info
    return {
        "description": manifest.description,
        "min_app_version": manifest.min_app_version,
        "contributors": _contributor_labels(info.contributors),
        "homepage": info.homepage,
        "repository": info.repository,
        "issues": info.issues,
        "documentation": info.documentation,
        "donate": info.donate,
        "changelog": info.changelog,
        "tags": tuple(info.tags),
        "compatibility": tuple(info.compatibility),
        "copyright": info.copyright,
    }


def _node_info_meta(info) -> dict:
    """Unified optional-metadata dict from a NodeInfo:
    description from the node, trust fields from the node/pack manifest."""
    return {
        "description": info.description,
        "contributors": _contributor_labels(getattr(info, "contributors", None)),
        "homepage": info.homepage,
        "repository": info.repository,
        "issues": info.issues,
        "documentation": info.documentation,
        "donate": info.donate,
        "changelog": info.changelog,
        "tags": tuple(getattr(info, "tags", ()) or ()),
        "compatibility": (),
        "copyright": info.copyright,
    }


def _ast_node_meta(raw: dict) -> dict:
    """Unified optional-metadata dict from raw AST NODE_* metadata."""
    contributors = raw.get("NODE_CONTRIBUTORS")
    if isinstance(contributors, dict):
        contributors = [contributors]
    return {
        "description": raw.get("NODE_DESCRIPTION", ""),
        "contributors": _contributor_labels(contributors),
        "homepage": raw.get("NODE_HOMEPAGE", ""),
        "repository": raw.get("NODE_REPOSITORY", ""),
        "issues": raw.get("NODE_ISSUES", ""),
        "documentation": raw.get("NODE_DOCUMENTATION", ""),
        "donate": raw.get("NODE_DONATE", ""),
        "changelog": raw.get("NODE_CHANGELOG", ""),
        "tags": tuple(raw.get("NODE_TAGS") or ()),
        "compatibility": (),
        "copyright": raw.get("NODE_COPYRIGHT", ""),
    }

_STATE_LABEL = {
    PACK_INSTALLED: "Installed",
    PACK_EULA_PENDING: "EULA pending",
}

_IMPORTABLE_SUFFIXES = (".py", ".lxpkg", ".lxext")


def block_reason(node_id: str, store: ConsentStore) -> str | None:
    """Pure (widget-free) consent check for one node id."""
    info = manager.node_info(node_id)
    if info is None or info.source == "builtin":
        return None  # built-in nodes need no consent
    try:
        if info.manifest is not None:
            return store.block_reason(info.manifest)
        return store.node_block_reason(node_id, info.eula)
    except Exception as exc:
        return f"Cannot check extension: {exc}"


def build_ui_gate(store: ConsentStore, parent: QWidget):
    """Gate usable from the GUI thread; may open the dialog to fix issues."""

    def gate(node_id: str):
        reason = block_reason(node_id, store)
        if reason is None:
            return True, None
        dlg = ExtensionsDialog(store, parent=parent)
        dlg.select_extension(node_id)
        dlg.exec()
        reason = block_reason(node_id, store)
        if reason is None:
            return True, None
        return False, reason

    return gate


def build_worker_gate(store: ConsentStore):
    """Gate for the pipeline worker thread (NO Qt objects, no dialogs)."""

    def gate(node_id: str) -> str | None:
        return block_reason(node_id, store)

    return gate


def _is_importable(path: Path) -> bool:
    if path.name == "manifest.json":
        return True
    if path.suffix.lower() in _IMPORTABLE_SUFFIXES:
        return True
    return path.is_dir() and (path / "manifest.json").exists()


# ---------------------------------------------------------------------------
# EULA view
# ---------------------------------------------------------------------------


class _EulaDialog(QDialog):
    def __init__(self, name: str, text: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{t('EULA')} — {name}")
        self.resize(560, 460)
        layout = QVBoxLayout(self)
        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._text.setPlainText(text)
        layout.addWidget(self._text)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(t("Accept"))
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(t("Decline"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


# ---------------------------------------------------------------------------
# Main dialog
# ---------------------------------------------------------------------------


class ExtensionsDialog(QDialog):
    def __init__(
        self,
        store: ConsentStore,
        parent=None,
        on_extensions_changed: Callable[[], None] | None = None,
        active_check: Callable[[], bool] | None = None,
    ):
        super().__init__(parent)
        self._store = store
        self._on_extensions_changed = on_extensions_changed
        self._active_check = active_check or (lambda: False)
        self.setWindowTitle(t("Extensions"))
        self.setProperty("origWinTitle_en", "Extensions")
        self.setModal(False)
        self.setAcceptDrops(True)
        self.resize(720, 540)

        root = QVBoxLayout(self)

        # --- extensions list (compact rows in a column) ---
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._list_host = QWidget()
        self._list_layout = QVBoxLayout(self._list_host)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.addStretch(1)
        self._scroll.setWidget(self._list_host)
        root.addWidget(self._scroll, 1)

        # --- bottom actions ---
        bottom = QHBoxLayout()
        import_btn = QPushButton(t("Import extension..."))
        import_btn.setProperty("origText_en", "Import extension...")
        import_btn.setToolTip(
            t("Import a node (.py) or a package (.lxpkg / .lxext)")
        )
        import_btn.setProperty(
            "origTip_en",
            "Import a node (.py) or a package (.lxpkg / .lxext)",
        )
        import_btn.clicked.connect(self._on_import_file)
        bottom.addWidget(import_btn)
        folder_btn = QPushButton(t("Import pack folder..."))
        folder_btn.setProperty("origText_en", "Import pack folder...")
        folder_btn.setToolTip(t("Import a folder containing a manifest.json"))
        folder_btn.setProperty(
            "origTip_en", "Import a folder containing a manifest.json"
        )
        folder_btn.clicked.connect(self._on_import_folder)
        bottom.addWidget(folder_btn)
        refresh_btn = QPushButton(t("Refresh"))
        refresh_btn.setProperty("origText_en", "Refresh")
        refresh_btn.clicked.connect(self._rebuild)
        bottom.addWidget(refresh_btn)
        bottom.addStretch(1)
        close_btn = QPushButton(t("Close"))
        close_btn.setProperty("origText_en", "Close")
        close_btn.clicked.connect(self.accept)
        bottom.addWidget(close_btn)
        root.addLayout(bottom)

        from lynceus.ui.translate import language_changed

        language_changed.connect(self._on_language_changed)

        self._rebuild()

    def _on_language_changed(self, _code: str) -> None:
        """Re-render the open dialog in the new language.

        Rows rebuild with fresh t() (construction-time strings are always
        current); chrome follows via the translate pass on pinned sources.
        """
        from lynceus.ui.translate import translate_widget

        translate_widget(self)
        self._rebuild()

    # -- import -------------------------------------------------------------

    def _on_import_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            t("Import extension"),
            "",
            "Package (*.lxpkg *.lxext);;Node (*.py);;Manifest (manifest.json)",
        )
        if path:
            self._import_paths([Path(path)])

    def _on_import_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, t("Import pack folder"), ""
        )
        if folder:
            self._import_paths([Path(folder)])

    def _import_paths(self, paths: list[Path]) -> None:
        if self._warn_if_running():
            return
        messages: list[str] = []
        errors = 0
        for path in paths:
            try:
                result = import_extension(path, manager.extensions_dir)
                messages.append(result.message)
            except ExtensionImportError as exc:
                errors += 1
                messages.append(f"{path.name}: {exc}")
        self._after_change()
        box = QMessageBox.warning if errors else QMessageBox.information
        box(
            self,
            t("Import extension"),
            "\n".join(messages),
        )

    # -- drag & drop --------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            paths = [
                Path(url.toLocalFile())
                for url in event.mimeData().urls()
                if url.isLocalFile() and _is_importable(Path(url.toLocalFile()))
            ]
            if paths:
                event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        paths = [
            Path(url.toLocalFile())
            for url in event.mimeData().urls()
            if url.isLocalFile() and _is_importable(Path(url.toLocalFile()))
        ]
        if paths:
            self._import_paths(paths)

    # -- building rows ------------------------------------------------------

    def _rebuild(self) -> None:
        while self._list_layout.count():
            item = self._list_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._list_layout.addStretch(1)

        rows = list(self._extension_entries())
        if not rows:
            info = QLabel(
                t(
                    "No extensions installed yet. Use \"Import extension...\" "
                    "to add a node (.py), a package (.lxpkg / .lxext) or a "
                    "pack folder, or drop the files here.\n"
                )
                + f"{t('Extensions directory')}: {manager.extensions_dir}"
            )
            info.setWordWrap(True)
            info.setObjectName("hintLabel")
            self._list_layout.insertWidget(
                self._list_layout.count() - 1, info
            )
            return

        for group_label, group_entries in self._group_entries(rows):
            section = CollapsibleSection(
                group_label, initially_expanded=True
            )
            for entry in group_entries:
                section.content_layout().addWidget(self._build_row(entry))
            self._list_layout.insertWidget(
                self._list_layout.count() - 1, section
            )

    @staticmethod
    def _group_entries(entries: list[dict]) -> list[tuple[str, list[dict]]]:
        """Group rows by extension kind into ordered, non-empty sections."""
        grouped: dict[str, list[dict]] = {}
        for entry in entries:
            group = entry.get("group", "node")
            grouped.setdefault(group, []).append(entry)
        order = sorted(
            grouped,
            key=lambda g: (g not in _GROUP_LABELS, _GROUP_LABELS.get(g, g)),
        )
        return [
            (t(_GROUP_LABELS.get(g, g.capitalize())), grouped[g])
            for g in order
        ]

    def _extension_entries(self) -> list[dict]:
        """Uniform rows: node files and packs (enabled and disabled), sorted by id."""
        entries: list[dict] = []
        seen: set[str] = set()
        ext_dir = manager.extensions_dir

        # --- enabled from registry (full metadata) ---
        for info in manager.list_nodes():
            if info.source != "extension":
                continue
            if info.manifest is not None:
                continue  # pack nodes are listed under their pack
            if info.node_id in seen:
                continue
            seen.add(info.node_id)
            entries.append(
                {
                    "kind": "node",
                    "group": "node",
                    "node_id": info.node_id,
                    "display_name": info.name,
                    "detail": info.node_id,
                    "author": info.author,
                    "license": info.license_label,
                    "eula": info.eula,
                    "folder": (
                        Path(info.file_path).parent if info.file_path else None
                    ),
                    "file": info.file_path,
                    "manifest": None,
                    "meta": _node_info_meta(info),
                }
            )
        for pack_id, manifest in manager.extension_manifests().items():
            if pack_id in seen:
                continue
            seen.add(pack_id)
            entries.append(
                {
                    "kind": "pack",
                    "group": manifest.kind,
                    "node_id": manifest.id,
                    "display_name": manifest.display_name,
                    "detail": f"{manifest.id}  v{manifest.version}",
                    "author": manifest.info.author,
                    "license": manifest.info.license_label,
                    "eula": manifest.info.eula,
                    "folder": manifest.root,
                    "file": None,
                    "manifest": manifest,
                    "meta": _pack_meta(manifest),
                }
            )

        # --- disabled / newly added on disk (metadata from AST/manifest) ---
        if not ext_dir.exists():
            return entries

        # Standalone node .py files (non-cache)
        for path in sorted(ext_dir.rglob("*.py")):
            rel = path.relative_to(ext_dir)
            if any(part.startswith(".") or part == "cache" for part in rel.parts[:-1]):
                continue
            meta = ast_meta(path)
            node_id = meta.get("NODE_ID")
            if not node_id or node_id in seen:
                continue
            seen.add(node_id)
            entries.append(
                {
                    "kind": "node",
                    "group": "node",
                    "node_id": node_id,
                    "display_name": meta.get("NODE_NAME") or node_id,
                    "detail": node_id,
                    "author": meta.get("NODE_AUTHOR", ""),
                    "license": meta.get("NODE_LICENSE") or DEFAULT_LICENSE,
                    "eula": meta.get("NODE_EULA", ""),
                    "folder": path.parent,
                    "file": str(path),
                    "manifest": None,
                    "meta": _ast_node_meta(meta),
                }
            )

        # Pack folders with manifest.json (non-cache)
        for path in sorted(ext_dir.rglob("manifest.json")):
            if any(part.startswith(".") or part == "cache" for part in path.parent.parts):
                continue
            try:
                manifest = ExtensionManifest.load(path)
            except ManifestError:
                continue
            if manifest.id in seen:
                continue
            seen.add(manifest.id)
            entries.append(
                {
                    "kind": "pack",
                    "group": manifest.kind,
                    "node_id": manifest.id,
                    "display_name": manifest.display_name,
                    "detail": f"{manifest.id}  v{manifest.version}",
                    "author": manifest.info.author,
                    "license": manifest.info.license_label,
                    "eula": manifest.info.eula,
                    "folder": manifest.root,
                    "file": None,
                    "manifest": manifest,
                    "meta": _pack_meta(manifest),
                }
            )

        return sorted(entries, key=lambda e: e["node_id"])

    def _build_row(self, entry: dict) -> QWidget:
        row = QWidget()
        layout = QVBoxLayout(row)
        layout.setContentsMargins(6, 6, 6, 6)

        head = QHBoxLayout()
        title = QLabel(
            f"<b>{entry['display_name']}</b>  "
            f'<span style="color:#8a93a6">{entry["detail"]}</span>'
        )
        head.addWidget(title, 1)

        status = self._entry_status(entry)
        if entry["node_id"] in self._store.disabled_ids():
            badge = QLabel(t("Disabled"))
        else:
            badge = QLabel(t(_STATE_LABEL.get(status.state, status.state)))
        badge.setObjectName("statusBadge")
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        head.addWidget(badge, 0)
        layout.addLayout(head)

        meta = f"by {entry['author'] or 'Unknown'} · {entry['license']}"
        sub = QLabel(meta)
        sub.setObjectName("hintLabel")
        layout.addWidget(sub)

        details = CollapsibleSection(t("Details"))
        details.content_layout().addWidget(self._build_details(entry))
        layout.addWidget(details)

        actions = QHBoxLayout()
        actions.addStretch(1)
        self._append_actions(actions, entry, status)
        layout.addLayout(actions)
        return row

    def _build_details(self, entry: dict) -> QWidget:
        """Detail widget listing optional metadata (contributors, links...)."""
        meta = entry.get("meta", {})
        rows: list[tuple[str, str]] = []

        desc = meta.get("description")
        if desc:
            rows.append((t("Description"), str(desc)))
        min_app = meta.get("min_app_version")
        if min_app:
            rows.append((t("Min app version"), str(min_app)))
        contributors = meta.get("contributors")
        if contributors:
            rows.append((t("Contributors"), " · ".join(contributors)))
        copyright_line = meta.get("copyright")
        if copyright_line:
            rows.append((t("Copyright"), str(copyright_line)))
        tags = meta.get("tags")
        if tags:
            rows.append((t("Tags"), ", ".join(tags)))
        for label, key in (
            ("Homepage", "homepage"),
            ("Repository", "repository"),
            ("Issues", "issues"),
            ("Documentation", "documentation"),
            ("Donate", "donate"),
            ("Changelog", "changelog"),
        ):
            value = meta.get(key)
            if value:
                rows.append((t(label), str(value)))

        body = QWidget()
        vlay = QVBoxLayout(body)
        vlay.setContentsMargins(0, 0, 0, 0)
        vlay.setSpacing(2)
        if not rows:
            empty = QLabel(t("No additional metadata."))
            empty.setObjectName("hintLabel")
            vlay.addWidget(empty)
            return body
        for label, value in rows:
            line = QLabel(f"<b>{label}:</b> {value}")
            line.setWordWrap(True)
            line.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            line.setObjectName("hintLabel")
            vlay.addWidget(line)
        return body

    def _entry_status(self, entry: dict):
        if entry["kind"] == "node":
            return self._store.resolve_node(entry["node_id"], entry["eula"])
        info = manager.node_info(entry["node_id"])
        manifest = info.manifest if info is not None else None
        if manifest is None:
            from lynceus.plugins.store import PackStatus

            return PackStatus(PACK_INSTALLED)
        return self._store.resolve(manifest)

    def _append_actions(self, layout: QHBoxLayout, entry: dict, status) -> None:
        eula = entry["eula"]
        if eula:
            accepted = self._store.eula_accepted_for(
                entry["node_id"], eula
            )
            if not accepted:
                eula_btn = QPushButton(t("Review EULA"))
                eula_btn.clicked.connect(
                    lambda _=False, e=entry: self._accept_eula(e)
                )
                layout.addWidget(eula_btn)

        open_btn = QPushButton(t("Open folder"))
        open_btn.clicked.connect(
            lambda _=False, e=entry: self._open_folder(e)
        )
        layout.addWidget(open_btn)

        disabled = entry["node_id"] in self._store.disabled_ids()
        toggle_btn = QPushButton(t("Enable") if disabled else t("Disable"))
        toggle_btn.clicked.connect(
            lambda _=False, e=entry: self._toggle_enabled(e)
        )
        layout.addWidget(toggle_btn)

        remove_btn = QPushButton(t("Remove"))
        remove_btn.clicked.connect(
            lambda _=False, e=entry: self._remove_extension(e)
        )
        layout.addWidget(remove_btn)

    # -- row actions --------------------------------------------------------

    def _toggle_enabled(self, entry: dict) -> None:
        if self._warn_if_running():
            return
        key = entry["node_id"]
        self._store.set_disabled(key, key not in self._store.disabled_ids())
        self._after_change()

    def _warn_if_running(self) -> bool:
        """Blocks extension changes while a pipeline run is active.

        Returns True when the change must be aborted (a run is in progress).
        """
        if not self._active_check():
            return False
        QMessageBox.information(
            self,
            t("Run in progress"),
            t(
                "Wait for the current run to finish before changing "
                "extensions."
            ),
        )
        return True

    def _remove_extension(self, entry: dict) -> None:
        if self._warn_if_running():
            return
        ret = QMessageBox.question(
            self,
            t("Remove extension"),
            t(
                "This will delete the extension files from the extensions "
                "directory."
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ret != QMessageBox.StandardButton.Yes:
            return
        try:
            if entry["kind"] == "node" and entry.get("file"):
                remove_node_file(entry["file"])
            elif entry["kind"] == "pack" and entry.get("manifest") is not None:
                remove_pack(entry["manifest"], manager.extensions_dir)
            else:
                return
        except Exception as exc:  # keep app alive
            QMessageBox.critical(
                self, t("Remove extension"), str(exc)
            )
            return
        self._store.set_disabled(entry["node_id"], False)
        self._after_change()

    def _after_change(self) -> None:
        """Rediscover and refresh everything after import/remove/toggle."""
        manager.discover()
        self._rebuild()
        if self._on_extensions_changed is not None:
            self._on_extensions_changed()

    def _accept_eula(self, entry: dict) -> None:
        dlg = _EulaDialog(entry["node_id"], entry["eula"], parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self._store.eula_accept(entry["node_id"], entry["eula"])
        self._rebuild()

    @staticmethod
    def _open_folder(entry: dict) -> None:
        from lynceus.ui.runs_panel import open_in_file_explorer

        folder = entry.get("folder")
        if folder is None:
            return
        open_in_file_explorer(Path(str(folder)))

    # -- helpers ------------------------------------------------------------

    def select_extension(self, extension_id: str) -> None:
        """Bring a specific extension into view (used by the add-node gate)."""
        for idx in range(self._list_layout.count()):
            item = self._list_layout.itemAt(idx)
            section = item.widget()
            if not isinstance(section, CollapsibleSection):
                continue
            target = self._find_row_in_section(section, extension_id)
            if target is None:
                continue
            if not section.is_expanded:
                section.expand()
            self._scroll.ensureWidgetVisible(target)
            return

    @staticmethod
    def _find_row_in_section(
        section: CollapsibleSection, extension_id: str
    ):
        """Row widget inside a section whose labels mention `extension_id`."""
        for i in range(section.content_layout().count()):
            item = section.content_layout().itemAt(i)
            widget = item.widget()
            if widget is None:
                continue
            if any(
                extension_id in label.text()
                for label in widget.findChildren(QLabel)
            ):
                return widget
        return None
