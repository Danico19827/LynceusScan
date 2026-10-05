# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Run-session browser shown in the lower-left panel.

Lists run directories under ``default_output_root()`` and their contents. Users can open a
session, copy its path, refresh the tree, or delete a session. Discovery uses
the same rule as the controller: session directories start with a digit and
cache directories are excluded.
"""

from __future__ import annotations

import logging
import json
import os
import shutil
import stat
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QStyle,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lynceus.plugins.locale import t
from lynceus.processing.controller import default_output_root

logger = logging.getLogger(__name__)


def _is_session_dir(path: Path) -> bool:
    """Match the controller's session-directory discovery rule."""
    if not (path.is_dir() and path.name and path.name[0].isdigit()):
        return False
    # Tiling seeds under cache/ are not sessions.
    if "cache" in [p.lower() for p in path.parts]:
        return False
    # Per-run temp tile dirs hold per-source folders named by hash, which
    # may start with a digit: never sessions (the sweep deletes them).
    if any(p.startswith("_tiles_tmp_") for p in path.parts):
        return False
    return True


def _session_children(session: Path) -> list[Path]:
    """Child entries of a session, tolerating mid-refresh deletion.

    The end-of-run sweep may remove a session between collection and
    tree building; vanishing entries yield no children instead of
    raising (the next refresh converges).
    """
    try:
        return sorted(
            session.iterdir(),
            key=lambda p: (p.is_dir(), p.name.lower()),
        )
    except OSError:
        return []


def _sessions_in(folder: Path) -> list[Path]:
    """Session directories below a folder (best-effort, guarded).

    Used to count bulk deletions and to notify the gallery per removed
    session. Temp tile dirs and unreadable entries are skipped.
    """
    found: list[Path] = []
    stack = [folder]
    while stack:
        directory = stack.pop()
        try:
            children = sorted(directory.iterdir())
        except OSError:
            continue
        for child in children:
            try:
                if not child.is_dir() or child.name.startswith("."):
                    continue
                if child.name.startswith("_tiles_tmp_"):
                    continue
            except OSError:
                continue
            try:
                if _is_session_dir(child):
                    found.append(child)
                else:
                    stack.append(child)
            except OSError:
                continue
    return found


def _rmtree_onerror(func, path, _exc_info) -> None:
    """rmtree retry for read-only files (Windows refuses them outright).

    Session files may carry the read-only bit (copied permissions,
    hardlinked reuse); chmod first, then repeat the failed call.
    """
    try:
        os.chmod(path, stat.S_IWRITE)
    except OSError:
        pass
    func(path)


def _remove_session_dir(path: Path) -> None:
    """Delete a session directory or raise with the cause.

    Verifies the directory is really gone: a silent leftover would keep
    feeding the reuse cache while the panel no longer lists it (ghost
    sessions). Raises OSError when anything remains.
    """
    shutil.rmtree(path, onerror=_rmtree_onerror)
    if Path(path).exists():
        raise OSError(f"Directory still exists after deletion: {path}")


def _tiles_cache_dirs(root: Path, max_depth: int = 4) -> list[Path]:
    """Every cache/ dir under the output root, at any nesting level.

    Caches live per session root: output/cache, output/<project>/cache,
    output/<project>/<source>/cache... A fixed 1-level scan misses the
    deeper ones and the purge button never enables. Depth-capped walk
    (never descends into a cache/ itself).
    """
    found: list[Path] = []
    stack = [(root, 0)]
    while stack:
        directory, depth = stack.pop()
        try:
            children = list(directory.iterdir())
        except OSError:
            continue
        for child in children:
            try:
                if not child.is_dir() or child.name.startswith("."):
                    continue
            except OSError:
                continue
            if child.name == "cache":
                found.append(child)
            elif depth < max_depth:
                stack.append((child, depth + 1))
    return found


def _tiles_cache_size(root: Path) -> int:
    """Total bytes under all tile caches (best-effort)."""
    total = 0
    for cache in _tiles_cache_dirs(root):
        try:
            children = list(cache.rglob("*"))
        except OSError:
            continue
        for path in children:
            try:
                if path.is_file():
                    total += path.stat().st_size
            except OSError:
                continue
    return total


def _cache_size_label(size: int) -> str:
    if size >= 1024**3:
        return f"{size / 1024**3:.1f} GB"
    return f"{size / 1024**2:.1f} MB"


def _iter_cache_files(root: Path, stop=None):
    """Yield ``(path, size)`` for files under tile caches (best-effort).

    Streaming (no materialized file list) with a cooperative ``stop``
    callback, so sizing stays interruptible on hundred-thousand-file
    trees. The check itself is nanoseconds; purge checks are coarser
    (signal traffic).
    """
    for cache in _tiles_cache_dirs(root):
        try:
            children = list(cache.rglob("*"))
        except OSError:
            continue
        for path in children:
            if stop is not None and stop():
                return
            try:
                if path.is_file():
                    yield path, path.stat().st_size
            except OSError:
                continue


def _tiles_cache_size(root: Path, stop=None) -> int:
    """Total bytes under all tile caches (best-effort)."""
    total = 0
    for _path, size in _iter_cache_files(root, stop):
        total += size
    return total


def _purge_cache_dirs(root: Path, stop=None,
                      on_progress: Callable[[int], None] | None = None,
                      run_active: Callable[[], bool] | None = None,
                      ) -> tuple[int, int, int]:
    """Delete every tile cache dir; return ``(errors, files, bytes)``.

    Best-effort with progress callback (deleted-file count) and
    cooperative stop/run-active checks every 512 entries. Read-only
    files are chmod'ed like :func:`_remove_session_dir`. Raises nothing.
    """
    errors = files = freed = 0
    checked = 0

    def _aborted() -> bool:
        try:
            if stop is not None and stop():
                return True
            return bool(run_active is not None and run_active())
        except Exception:
            return False

    for cache in _tiles_cache_dirs(root):
        if _aborted():
            break
        seen_dirs: list[Path] = []
        finished = True
        try:
            for dirpath, dirnames, filenames in os.walk(cache, topdown=True):
                dirnames[:] = [d for d in dirnames if not d.startswith(".")]
                here = Path(dirpath)
                seen_dirs.append(here)
                for name in filenames:
                    target = here / name
                    try:
                        try:
                            size = target.stat().st_size
                        except OSError:
                            size = 0
                        target.unlink(missing_ok=True)
                    except OSError:
                        try:
                            os.chmod(target, stat.S_IWRITE)
                            target.unlink(missing_ok=True)
                        except OSError:
                            errors += 1
                            continue
                    files += 1
                    freed += size
                    checked += 1
                    if checked % 512 == 0:
                        if on_progress is not None:
                            try:
                                on_progress(files)
                            except Exception:
                                pass
                        if _aborted():
                            finished = False
                            break
                if not finished:
                    break
        except OSError:
            errors += 1
        for directory in sorted(
            seen_dirs, key=lambda p: len(p.parts), reverse=True
        ):
            try:
                directory.rmdir()
            except OSError:
                pass
        try:
            if Path(cache).exists():
                errors += 1
        except OSError:
            errors += 1
    return errors, files, freed


class _PanelSignals(QObject):
    """Queued signals from panel background workers (GUI thread only).

    Byte/count payloads travel as ``object``: Qt ``int`` is 32-bit and a
    tile cache easily exceeds it (68 GB observed in the wild).
    """

    cache_sized = Signal(int, object)
    purge_progress = Signal(int)
    purge_finished = Signal(int, object, object, bool)


def open_in_file_explorer(path: Path) -> None:
    """Open a path in the operating system's file explorer."""
    target = str(path)
    try:
        if sys.platform == "win32":
            os.startfile(target)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", target])
        else:
            subprocess.Popen(["xdg-open", target])
    except OSError:
        pass


def _open_dir(path: Path) -> None:
    # Opening a whole run lands directly on its products folder.
    if _is_session_dir(path):
        artifacts = path / "artifacts"
        if artifacts.is_dir():
            path = artifacts
    # Open the parent directory when the target is a file.
    if path.is_file():
        open_in_file_explorer(path.parent)
    else:
        open_in_file_explorer(path)


class RunsPanel(QWidget):
    """Session tree with action header and context menu."""

    session_deleted = Signal(object)  # Path of the deleted session
    session_preview_requested = Signal(object)  # Session dir to browse in the gallery

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        active_check: Optional[Callable[[Path], bool]] = None,
    ):
        super().__init__(parent)
        self.setObjectName("runsPanel")
        self._session_root: Optional[Path] = None
        self._always_root: Path = default_output_root()
        self._active_check = active_check or (lambda _p: False)
        self._run_active_check: Callable[[], bool] = lambda: False
        self._signals = _PanelSignals(self)
        self._signals.cache_sized.connect(self._on_cache_size)
        self._signals.purge_progress.connect(self._on_purge_progress)
        self._signals.purge_finished.connect(self._on_purge_finished)
        self._cache_size_bytes: Optional[int] = None
        self._cache_size_seq = 0
        self._size_stop = threading.Event()
        self._purge_stop = threading.Event()
        self._purge_dialog = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 0)
        layout.setSpacing(4)

        layout.addWidget(self._build_header(), 0)

        self.tree = QTreeWidget(self)
        self.tree.setObjectName("runsTree")
        self.tree.setColumnCount(1)
        self.tree.setHeaderHidden(True)
        self.tree.setRootIsDecorated(True)
        self.tree.setItemsExpandable(True)
        self.tree.setExpandsOnDoubleClick(False)
        self.tree.setIndentation(12)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.tree.currentItemChanged.connect(
            lambda *_a: self._update_buttons_state()
        )
        layout.addWidget(self.tree, 1)

        self._empty_hint = QLabel(t("No runs yet"), self)
        self._empty_hint.setProperty("origText_en", "No runs yet")
        self._empty_hint.setObjectName("panelHint")
        self._empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_hint.setWordWrap(True)
        layout.addWidget(self._empty_hint, 1)
        self._update_empty_state()
        self._update_buttons_state()

    # ------------------------------------------------------------------
    # Header construction
    # ------------------------------------------------------------------

    def _build_header(self) -> QFrame:
        frame = QFrame(self)
        frame.setObjectName("runsHeader")
        hlay = QHBoxLayout(frame)
        hlay.setContentsMargins(4, 2, 4, 2)
        hlay.setSpacing(2)

        title = QLabel(t("Runs"), frame)
        title.setObjectName("runsTitle")
        title.setProperty("origText_en", "Runs")
        hlay.addWidget(title)
        hlay.addStretch(1)

        self.btn_refresh = self._icon_button(
            QStyle.StandardPixmap.SP_BrowserReload, "Refresh"
        )
        self.btn_refresh.clicked.connect(self.refresh)

        self.btn_open = self._icon_button(
            QStyle.StandardPixmap.SP_DialogOpenButton,
            "Open in File Explorer",
        )
        self.btn_open.clicked.connect(self._open_selected)

        self.btn_delete = self._icon_button(
            QStyle.StandardPixmap.SP_TrashIcon, "Delete"
        )
        self.btn_delete.setObjectName("runsActionDelete")
        self.btn_delete.clicked.connect(self._delete_selected)

        self.btn_cache = self._icon_button(
            QStyle.StandardPixmap.SP_DriveHDIcon, "Clear tile cache"
        )
        self.btn_cache.clicked.connect(self._clear_tile_cache)

        hlay.addWidget(self.btn_refresh)
        hlay.addWidget(self.btn_open)
        hlay.addWidget(self.btn_delete)
        hlay.addWidget(self.btn_cache)

        self._buttons = (
            self.btn_refresh, self.btn_open, self.btn_delete, self.btn_cache
        )
        return frame

    def _icon_button(self, pixmap, tooltip_key: str) -> QToolButton:
        """Create an icon-only button with a translatable tooltip."""
        btn = QToolButton(self)
        btn.setIcon(self.style().standardIcon(pixmap))
        btn.setIconSize(btn.iconSize())
        btn.setFixedSize(26, 26)
        btn.setProperty("origTip_en", tooltip_key)
        btn.setToolTip(t(tooltip_key))
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        return btn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_session_root(self, session_root: Optional[Path]) -> None:
        """Set the effective project session root.

        The panel still lists sessions from every project; the value is kept
        for future filtering or navigation features.
        """
        self._session_root = Path(session_root) if session_root else None
        self.refresh()

    def set_active_check(self, active_check: Optional[Callable[[Path], bool]]) -> None:
        """Set the callback used to identify the active session."""
        self._active_check = active_check or (lambda _p: False)

    def set_run_active_check(self, check: Optional[Callable[[], bool]]) -> None:
        """Set the callback reporting whether any pipeline run is active."""
        self._run_active_check = check or (lambda: False)

    def _collect_origins(self) -> dict[Path, list[Path]]:
        """Return ``{origin: [sessions]}`` for all runs below the output root.

        An origin directly contains sessions. Results are sorted newest first.
        """
        out = self._always_root
        out.mkdir(parents=True, exist_ok=True)
        origins: dict[Path, list[Path]] = {}
        self._collect_in(out, origins)
        return origins

    @staticmethod
    def _group_origins(
        origins: dict[Path, list[Path]], root: Path
    ) -> list[tuple[str, list[tuple[str, list[Path]]]]]:
        """Nest origins as ``[(group, [(source, [sessions])])`` for the tree.

        The first relative segment is the project group, including
        ``_default_project``: unsaved runs read as
        ``output → _default_project → las`` like any other project instead
        of spilling loose sources at the root. Groups sort with the root
        group first, then alphabetically; sources alphabetically; sessions
        keep collection order (newest first).
        """
        grouped: dict[str, dict[str, list[Path]]] = {}
        for origin, sessions in origins.items():
            try:
                rel = origin.relative_to(root)
            except ValueError:
                continue
            parts = rel.parts
            if not parts:
                group, source = "", ""
            elif len(parts) == 1:
                group, source = parts[0], ""
            else:
                group, source = parts[0], parts[1]
            target = grouped.setdefault(group, {}).setdefault(source, [])
            for session in sessions:
                if session not in target:
                    target.append(session)
        ordered = []
        for group in sorted(grouped, key=lambda g: (g != "", g.lower())):
            sources = sorted(grouped[group].items(), key=lambda kv: kv[0].lower())
            ordered.append((group, sources))
        return ordered

    @staticmethod
    @staticmethod
    def _session_badge(session: Path) -> str:
        """Session label with QA verdict/product count when reported.

        Reads ``quality_report.json`` (best-effort); sessions without one
        (legacy or still running) show the plain timestamp.
        """
        try:
            report = json.loads(
                (session / "quality_report.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return session.name
        if not isinstance(report, dict):
            return session.name
        verdict = report.get("verdict")
        counts = report.get("counts") or {}
        products = counts.get("products_written")
        label = session.name
        if verdict in ("PASS", "FAIL"):
            label += f" · {verdict}"
        if isinstance(products, int):
            label += f" · {products} products"
        return label

    def _collect_in(
        self, directory: Path, origins: dict[Path, list[Path]]
    ) -> None:
        """Register each session under its containing directory.

        Defensive per entry: one unreadable directory must never blank
        the whole panel (a failed collect used to clear the tree first
        and abort mid-way, hiding sessions that were still on disk).

        ``cache/`` and ``artifacts/`` subtrees are never descended into:
        they hold tile/product files by the hundred thousand, never
        sessions (segments live under ``lotes/``, finals beside the
        timestamp dirs). Same pre-existing convention as the purge and
        the session rule for ``cache/``. This keeps collection
        proportional to the session count, not the tile count.
        """
        try:
            children = sorted(directory.iterdir())
        except OSError as exc:
            logger.warning("Runs collect skipped %s: %s", directory, exc)
            return
        for child in children:
            try:
                if not child.is_dir() or child.name.startswith("."):
                    continue
                if child.name.startswith("_tiles_tmp_"):
                    continue
                if child.name in ("cache", "artifacts"):
                    continue
            except OSError:
                continue
                if child.name.startswith("_tiles_tmp_"):
                    continue  # per-run temp tiles, never sessions
            except OSError:
                continue
            try:
                is_session = _is_session_dir(child)
            except OSError:
                continue
            if is_session:
                origins.setdefault(directory, []).append(child)
            else:
                self._collect_in(child, origins)
        sessions = origins.get(directory)
        if sessions:
            try:
                sessions.sort(
                    key=lambda p: p.stat().st_mtime_ns, reverse=True
                )
            except OSError:
                pass

    def refresh(self) -> None:
        """Rebuild the session tree while preserving the current selection.

        Atomic: the collection runs fully before the tree is touched, so
        a failed collect keeps the previous tree instead of blanking the
        panel (which used to hide sessions that were still on disk).

        Layout is a nested ``output → project/source → sessions`` tree,
        everything collapsed except the chain holding the newest session.
        """
        selected = self._selected_path()
        try:
            origins = self._collect_origins()
            grouped = self._group_origins(origins, self._always_root)
        except Exception as exc:
            logger.warning("Runs refresh skipped: %s", exc)
            return

        newest: Path | None = None
        for _group, sources in grouped:
            for _source, sessions in sources:
                for session in sessions:
                    if newest is None or session.name > newest.name:
                        newest = session

        self.tree.clear()
        root_label = self._always_root.name or str(self._always_root)
        total = sum(
            len(sessions) for _, sources in grouped for _, sessions in sources
        )
        root_item = QTreeWidgetItem(
            [f"{root_label}  [{t('Runs')}: {total}]"]
        )
        root_item.setData(0, Qt.ItemDataRole.UserRole, str(self._always_root))
        root_item.setData(0, Qt.ItemDataRole.ToolTipRole, str(self._always_root))
        root_item.setData(0, Qt.ItemDataRole.FontRole, self._bold_font())
        self.tree.addTopLevelItem(root_item)

        for group, sources in grouped:
            if group:
                group_path = self._always_root / group
                group_item = QTreeWidgetItem(
                    [f"{group}  [{t('Runs')}: "
                     f"{sum(len(s) for _, s in sources)}]"]
                )
                group_item.setData(0, Qt.ItemDataRole.UserRole, str(group_path))
                group_item.setData(0, Qt.ItemDataRole.ToolTipRole, str(group_path))
                group_item.setData(0, Qt.ItemDataRole.FontRole, self._bold_font())
                root_item.addChild(group_item)
                parent = group_item
            else:
                parent = root_item
            for source, sessions in sources:
                if source:
                    source_path = (
                        (self._always_root / group / source)
                        if group
                        else (self._always_root / source)
                    )
                    source_item = QTreeWidgetItem(
                        [f"{source}  [{t('Runs')}: {len(sessions)}]"]
                    )
                    source_item.setData(0, Qt.ItemDataRole.UserRole, str(source_path))
                    source_item.setData(0, Qt.ItemDataRole.ToolTipRole, str(source_path))
                    parent.addChild(source_item)
                    holder = source_item
                else:
                    holder = parent
                for session in sessions:
                    self._add_session(session, holder)
        self._expand_newest(root_item, newest)
        root_item.setExpanded(True)

        if selected is not None:
            self._select_path(selected)
        self._update_empty_state()
        self._update_buttons_state()
        self._update_cache_button()

    def _expand_newest(self, root_item: QTreeWidgetItem, newest: Path | None) -> None:
        """Expand only the chain holding the newest session."""
        if newest is None:
            return
        target = str(newest)
        stack = [root_item]
        while stack:
            item = stack.pop()
            for index in range(item.childCount()):
                child = item.child(index)
                data = child.data(0, Qt.ItemDataRole.UserRole)
                if target == data or target.startswith(str(data) + os.sep):
                    child.setExpanded(True)
                    stack.append(child)

    def _bold_font(self):
        from PySide6.QtGui import QFont

        font = QFont(self.tree.font())
        font.setBold(True)
        return font

    def _add_session(self, session: Path, parent: QTreeWidgetItem) -> None:
        item = QTreeWidgetItem([self._session_badge(session)])
        item.setData(0, Qt.ItemDataRole.UserRole, str(session))
        item.setData(0, Qt.ItemDataRole.ToolTipRole, str(session))
        item.setToolTip(0, str(session))
        parent.addChild(item)

        for child in _session_children(session):
            leaf = QTreeWidgetItem([child.name])
            leaf.setData(0, Qt.ItemDataRole.UserRole, str(child))
            leaf.setData(0, Qt.ItemDataRole.ToolTipRole, str(child))
            item.addChild(leaf)

    # ------------------------------------------------------------------
    # Header actions
    # ------------------------------------------------------------------

    def _selected_path(self) -> Optional[Path]:
        item = self.tree.currentItem()
        if item is None:
            return None
        val = item.data(0, Qt.ItemDataRole.UserRole)
        return Path(val) if val else None

    def _open_selected(self) -> None:
        path = self._selected_path()
        if path is not None:
            _open_dir(path)

    def _deletable(self, path: Path | None) -> bool:
        """Sessions plus folders holding at least one session (never root)."""
        if path is None:
            return False
        try:
            if _is_session_dir(path):
                return True
            if path == self._always_root or not path.is_dir():
                return False
            return len(_sessions_in(path)) > 0
        except OSError:
            return False

    def _delete_selected(self) -> None:
        path = self._selected_path()
        if path is None:
            return
        if _is_session_dir(path):
            self._delete_session(path)
        elif path != self._always_root and path.is_dir():
            self._delete_folder(path)

    def _update_buttons_state(self) -> None:
        path = self._selected_path()
        self.btn_open.setEnabled(path is not None)
        self.btn_delete.setEnabled(self._deletable(path))

    def _update_cache_button(self) -> None:
        """Enable cache purge only when there is something to purge.

        Enablement is a cheap directory probe; the exact byte size
        arrives asynchronously (a full walk would freeze startup on
        hundred-thousand-file trees).
        """
        try:
            has_cache = bool(_tiles_cache_dirs(self._always_root))
        except Exception:
            has_cache = False
        self.btn_cache.setEnabled(has_cache)
        self._refresh_cache_size()

    def _refresh_cache_size(self) -> None:
        """Recompute the exact cache size off the GUI thread."""
        self._size_stop.set()
        self._size_stop = threading.Event()
        self._cache_size_seq += 1
        seq = self._cache_size_seq
        root = self._always_root
        stop = self._size_stop
        signals = self._signals

        def _work() -> None:
            try:
                size = _tiles_cache_size(root, stop.is_set)
            except Exception:
                return
            if not stop.is_set():
                signals.cache_sized.emit(seq, size)

        thread = threading.Thread(target=_work, daemon=True)
        thread.start()

    def _on_cache_size(self, seq: int, size: int) -> None:
        if seq == self._cache_size_seq:
            self._cache_size_bytes = size

    def shutdown(self) -> None:
        """Stop background workers (call on application exit)."""
        self._size_stop.set()
        self._purge_stop.set()
        dialog, self._purge_dialog = self._purge_dialog, None
        if dialog is not None:
            try:
                dialog.reject()
            except Exception:
                pass

    def _clear_tile_cache(self) -> None:
        """Purge every tile cache under the output root (loose + projects).

        Deleting sessions never touches the source-keyed tile cache, so a
        suspect cache (or disk-heavy legacy caches) needs this explicit
        action. Refused while a run is active: the tiler may be writing.
        Deletion runs on a worker thread with progress and cancel; the
        run starting mid-purge aborts it.
        """
        if self._run_active_check():
            QMessageBox.information(
                self,
                t("Run in progress"),
                t("Wait for the current run to finish before clearing the cache."),
            )
            return
        size = self._cache_size_bytes
        if size is not None and size <= 0:
            return
        if size is None:
            ret = QMessageBox.question(
                self,
                t("Clear tile cache"),
                t("Delete all cached tiles? "
                  "They will be rebuilt on the next run."),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
        else:
            ret = QMessageBox.question(
                self,
                t("Clear tile cache"),
                t(
                    "Tile cache uses {size}. Delete all cached tiles? "
                    "They will be rebuilt on the next run."
                ).format(size=_cache_size_label(size)),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
        if ret != QMessageBox.StandardButton.Yes:
            return
        self._purge_stop.set()
        self._purge_stop = threading.Event()
        stop = self._purge_stop
        signals = self._signals
        root = self._always_root
        run_active = self._run_active_check
        dialog = QProgressDialog(
            t("Clear tile cache"), t("Cancel"), 0, 0, self
        )
        dialog.setWindowTitle(t("Clear tile cache"))
        dialog.setMinimumDuration(0)
        dialog.canceled.connect(stop.set)
        self._purge_dialog = dialog

        def _work() -> None:
            errors, _files, _freed = _purge_cache_dirs(
                root, stop.is_set,
                lambda n: signals.purge_progress.emit(n),
                run_active,
            )
            cancelled = stop.is_set() or bool(run_active())
            signals.purge_finished.emit(errors, _files, _freed, cancelled)

        self.btn_cache.setEnabled(False)
        dialog.show()
        thread = threading.Thread(target=_work, daemon=True)
        thread.start()

    def _on_purge_progress(self, files: int) -> None:
        dialog = self._purge_dialog
        if dialog is not None:
            try:
                dialog.setLabelText(
                    t("Deleting cached tiles ({n} files)").format(n=f"{files:,}")
                )
            except Exception:
                pass

    def _on_purge_finished(
        self, errors: int, _files: int, _freed: int, cancelled: bool
    ) -> None:
        dialog, self._purge_dialog = self._purge_dialog, None
        if dialog is not None:
            try:
                dialog.close()
            except Exception:
                pass
        self.btn_cache.setEnabled(True)
        if cancelled:
            QMessageBox.information(
                self,
                t("Run in progress"),
                t("Wait for the current run to finish before clearing the cache."),
            )
        elif errors:
            QMessageBox.critical(
                self,
                t("Clear tile cache"),
                t("Could not delete tile cache"),
            )
        self.refresh()

    # ------------------------------------------------------------------
    # Context menu
    # ------------------------------------------------------------------

    def _show_context_menu(self, pos) -> None:
        item = self.tree.itemAt(pos)
        if item is None:
            return
        path = item.data(0, Qt.ItemDataRole.UserRole)
        if not path:
            return
        menu = QMenu(self)

        act_open = QAction(t("Open in File Explorer"), menu)
        act_open.triggered.connect(lambda: _open_dir(Path(path)))
        menu.addAction(act_open)

        act_copy = QAction(t("Copy Path"), menu)
        act_copy.triggered.connect(lambda: self._copy_path(Path(path)))
        menu.addAction(act_copy)

        if path and _is_session_dir(Path(path)):
            act_preview = QAction(t("Preview products"), menu)
            act_preview.triggered.connect(
                lambda: self.session_preview_requested.emit(Path(path))
            )
            menu.addAction(act_preview)
            menu.addSeparator()
            act_delete = QAction(t("Delete"), menu)
            act_delete.triggered.connect(lambda: self._delete_session(Path(path)))
            menu.addAction(act_delete)
        elif (
            path
            and Path(path) != self._always_root
            and Path(path).is_dir()
            and _sessions_in(Path(path))
        ):
            menu.addSeparator()
            act_delete = QAction(t("Delete"), menu)
            act_delete.triggered.connect(lambda: self._delete_folder(Path(path)))
            menu.addAction(act_delete)

        menu.addSeparator()
        act_refresh = QAction(t("Refresh"), menu)
        act_refresh.triggered.connect(self.refresh)
        menu.addAction(act_refresh)

        menu.exec(self.tree.viewport().mapToGlobal(pos))

    def _delete_folder(self, path: Path) -> None:
        """Delete a whole folder of runs after an explicit headcount confirm.

        Refused while any pipeline run is active (sessions inside may be
        written to). Emits one gallery-clear signal per removed session so
        previews never point at deleted products.
        """
        sessions = _sessions_in(path)
        if not sessions:
            return
        if self._run_active_check():
            QMessageBox.information(
                self,
                t("Run in progress"),
                t("Wait for the current run to finish before deleting."),
            )
            return
        ret = QMessageBox.question(
            self,
            t("Delete"),
            t("Delete {n} runs in '{name}'? This cannot be undone.").format(
                n=len(sessions), name=path.name
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ret != QMessageBox.StandardButton.Yes:
            return
        try:
            _remove_session_dir(path)
        except OSError as exc:
            QMessageBox.critical(
                self,
                t("Delete"),
                f"{t('Could not delete the run')}: {exc}",
            )
        else:
            for session in sessions:
                self.session_deleted.emit(session)
        self.refresh()

    def _delete_session(self, path: Path) -> None:
        if not _is_session_dir(path):
            return
        if self._active_check(path):
            QMessageBox.information(
                self,
                t("Runs"),
                t("This run is currently being processed."),
            )
            return
        ret = QMessageBox.question(
            self,
            t("Delete"),
            t("Are you sure you want to delete this run?"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ret != QMessageBox.StandardButton.Yes:
            return
        try:
            _remove_session_dir(path)
        except OSError as exc:
            QMessageBox.critical(
                self,
                t("Delete"),
                f"{t('Could not delete the run')}: {exc}",
            )
        else:
            self.session_deleted.emit(path)
        self.refresh()

    @staticmethod
    def _copy_path(path: Path) -> None:
        from PySide6.QtWidgets import QApplication

        QApplication.clipboard().setText(str(path))

    def _on_item_double_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        val = item.data(0, Qt.ItemDataRole.UserRole)
        if not val:
            return
        # setExpandsOnDoubleClick(False) prevents the double click from toggling
        # expansion; here it only opens the file explorer.
        _open_dir(Path(val))

    # ------------------------------------------------------------------
    # Selection and state helpers
    # ------------------------------------------------------------------

    def _select_path(self, target: Path) -> None:
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            self._select_descendant(top, target)

    def _select_descendant(self, item: QTreeWidgetItem, target: Path) -> bool:
        for i in range(item.childCount()):
            child = item.child(i)
            val = child.data(0, Qt.ItemDataRole.UserRole)
            if val and Path(val) == target:
                self.tree.setCurrentItem(child)
                self.tree.scrollToItem(child)
                self._update_buttons_state()
                return True
            if self._select_descendant(child, target):
                return True
        return False

    def _update_empty_state(self) -> None:
        has_items = self.tree.topLevelItemCount() > 0
        self.tree.setVisible(has_items)
        if not has_items:
            self._empty_hint.setText(t("No runs yet"))
        self._empty_hint.setVisible(not has_items)