# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
import importlib
import math
import os
import re
import time
import copy
import uuid
from dataclasses import dataclass
from typing import Callable

from PySide6.QtCore import QEvent, QPointF, QRect, QSettings, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QBrush, QColor, QKeySequence, QPainter, QPainterPath, QPen, QPolygonF, QRadialGradient, QShortcut
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QDialog,
    QFrame,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from lynceus.processing.controller import (
    PipelineCallbacks,
    PipelineController,
    resolve_output_root,
)
from lynceus.processing.executor import workers_for_percent
from lynceus.processing.tiler import memory_budget_for_percent
from lynceus.graph_clipboard import (
    decode_subgraph,
    encode_subgraph,
    extract_subgraph,
)
from lynceus.nodes._variants import strategy_options
from lynceus.plugins.locale import t
from lynceus.plugins.registry import manager
from lynceus.ui.nodes.canvas_toolbar import CanvasToolbar
from lynceus.ui.nodes.node_item import (
    HEADER_HEIGHT,
    PORT_PROTRUSION,
    STATUS_IDLE,
    STATUS_PROCESSING,
    NodeItem,
)
from lynceus.ui.nodes.node_library import NODE_MIME_TYPE
from lynceus.ui.settings_keys import (
    ACCEL_BACKEND_KEY,
    ACCEL_DEVICE_KEY,
    ACCEL_MODE_KEY,
    CPU_PERCENT_DEFAULT,
    CPU_PERCENT_KEY,
    KEEP_INTERMEDIATES_KEY,
    MEMORY_PERCENT_DEFAULT,
    MEMORY_PERCENT_KEY,
    OPERATOR_NAME_KEY,
    OPERATOR_ORG_KEY,
    OUTPUT_ROOT_KEY,
    PREFLIGHT_QUIET_KEY,
    SETTINGS_APP,
    SETTINGS_ORG,
    SNAP_KEY,
)
from lynceus.ui.nodes.ports import Port
from lynceus.ui.nodes.widget_factory import create_node_item
from lynceus.ui.nodes.widgets.strategy import StrategyNodeItem
from lynceus.nodes.ports import get_port_color

MIN_ZOOM = 0.1
MAX_ZOOM = 5.0
ZOOM_STEP = 1.25

TOOLBAR_MARGIN = 8
CHIP_MARGIN = 10

NOTICE_MARGIN = 10
NOTICE_MAX_WIDTH = 360
NOTICE_ROW_MAX_TEXT = 300
NOTICE_AUTOHIDE_MS = 7000
NOTICE_MAX_ROWS = 25
PERSISTENT_NOTICE_KINDS = ("warning", "error")
#: Longest failure reason shown per node in the end-of-run report (chars);
#: engine reasons can carry an OOM hint, the full text stays on disk.
FAILURE_REASON_MAX = 200

# Textual progress markers ("Tiling... 14%", "DTM 3/51 | CHM 2/51"): once a
# transient row is updated with such a message it stops auto-hiding, so it
# never flickers during sparse (>autohide gap) heavy phases. It is removed
# explicitly when the run finishes/cancels. The tiling message carries a
# rate suffix ("Tiling... 14% (2.5 M pts/s)"); the optional tail keeps it
# pinned.
_PROGRESS_RE = re.compile(r"%(\s*\(.*\))?$|\d+/\d+", re.IGNORECASE)

WIRE_COLOR = QColor("#8a93a6")
WIRE_GHOST_COLOR = QColor("#3a465c")
WIRE_WIDTH = 2.0
WIRE_HIT_MARGIN = 12

# Dim the live canvas while edits are staged against an isolated run snapshot.
PENDING_OPACITY = 0.55

# Undo/redo history (item 9): snapshot-based document history. The stack holds
# committed states (history[cursor] == the live graph); a mutation replaces the
# current slot while it falls inside the merge window, so continuous gestures
# (drags, typing, spin boxes) collapse into a single undo step.
UNDO_LIMIT = 100
UNDO_MERGE_WINDOW = 0.35

# Canvas dot-grid settings (scene units).
GRID_MINOR = 20
GRID_MAJOR_EVERY = 5
SNAP_STEP = GRID_MINOR
CANVAS_BG = QColor("#0b0f1a")
DOT_MINOR_COLOR = QColor(38, 46, 64)
DOT_MAJOR_COLOR = QColor(58, 70, 92)
ORIGIN_COLOR = QColor(47, 58, 79, 110)
# LOD: grid dots fade according to screen-space spacing. Between *_FADE_IN and
# *_FADE_OUT the color interpolates toward the background; below *_FADE_OUT the
# level is skipped.
MINOR_FADE_IN_PX = 14.0
MINOR_FADE_OUT_PX = 7.0
MAJOR_FADE_IN_PX = 8.0
MAJOR_FADE_OUT_PX = 3.0
MAX_DOTS_PER_LEVEL = 60_000


@dataclass
class Connection:
    """Visual connection between an output port and an input port."""

    source: Port
    target: Port
    path_item: QGraphicsPathItem


class NodeGraphScene(QGraphicsScene):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setSceneRect(-1_000_000, -1_000_000, 2_000_000, 2_000_000)


class _LabelEditor(QLineEdit):
    """Inline node-label editor displayed over the node header."""

    cancelled = Signal()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.cancelled.emit()
            event.accept()
            return
        super().keyPressEvent(event)


NOTICE_KIND_COLORS = {
    "info": "#7aa8d4",
    "action": "#d4a94e",
    "warning": "#d4a94e",
    "success": "#6ec87a",
    "error": "#d46a6a",
}
"""Kind colors, mirrored 1:1 by the noticeRow QSS rules below (single
source of truth for tests; the stylesheet owns the pixels)."""


def failure_report_text(
    failed: int,
    failed_nodes: dict | None = None,
    names: dict | None = None,
    peak_mb=None,
    session_dir: str = "",
) -> str:
    """Durable end-of-run failure report (pure; item 44).

    Status-bar rows are ephemeral and per-node messages can be dismissed or
    pruned, so the closing summary is the record that survives: which nodes
    failed, why, and where the session lives. ``names`` maps iid to a
    display label; unknown ids fall back to the iid.
    """
    lines = [t("Run finished with {n} failed task(s)").format(n=failed)]
    for iid in sorted(failed_nodes or {}):
        reason = " ".join(str((failed_nodes or {})[iid] or "").split())
        if len(reason) > FAILURE_REASON_MAX:
            reason = reason[: FAILURE_REASON_MAX - 1].rstrip() + "…"
        label = str((names or {}).get(iid) or iid)
        lines.append(f"• {label}: {reason}" if reason else f"• {label}")
    if peak_mb:
        lines.append(
            t("peak RAM {peak} MB").format(peak=round(float(peak_mb), 1))
        )
    if session_dir:
        lines.append(t("Session folder: {path}").format(path=session_dir))
    return "\n".join(lines)


class _NoticeRow(QFrame):
    """A single notice chip of the canvas notice tray."""

    dismissed = Signal()

    def __init__(self, text: str, kind: str, autohide: bool):
        super().__init__()
        self.setObjectName("noticeRow")
        # Kind-driven QSS (no inline stylesheets): themes keep working.
        self.setProperty("noticeKind", kind)
        self._text = text
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 5, 6)
        layout.setSpacing(8)
        dot = QLabel("")
        dot.setObjectName("noticeDot")
        dot.setFixedSize(8, 8)
        self._body = QLabel(t(text))
        self._body.setObjectName("noticeText")
        self._body.setWordWrap(True)
        self._body.setMaximumWidth(NOTICE_ROW_MAX_TEXT)
        dismiss = QToolButton()
        dismiss.setObjectName("noticeDismiss")
        dismiss.setText("×")
        dismiss.setAutoRaise(True)
        dismiss.setFixedSize(18, 18)
        dismiss.clicked.connect(self.dismissed)
        layout.addWidget(dot, 0)
        layout.addWidget(self._body, 1)
        layout.addWidget(dismiss, 0)
        if autohide:
            # Child timer: destroyed together with the row, so it never fires
            # after the row has been removed (ex. clear_all).
            self._auto_hide = QTimer(self)
            self._auto_hide.setSingleShot(True)
            self._auto_hide.timeout.connect(self.dismissed)
            self._auto_hide.start(NOTICE_AUTOHIDE_MS)

    def text(self) -> str:
        return self._text

    def set_text(self, text: str) -> None:
        self._text = text
        self._body.setText(t(text))

    def restart_autohide(self) -> None:
        if hasattr(self, "_auto_hide"):
            self._auto_hide.start(NOTICE_AUTOHIDE_MS)

    def pin(self) -> None:
        """Keep the row visible: cancel its auto-hide timer."""
        if hasattr(self, "_auto_hide"):
            self._auto_hide.stop()


class _NoticeTray(QFrame):
    """Stacked notices anchored to the lower-right corner of the canvas.

    Warnings/errors persist until dismissed; transient kinds auto-hide. The
    tray is fed on the GUI thread (queued signals from the worker).
    """

    def __init__(self, canvas):
        super().__init__(canvas.viewport())
        self._canvas = canvas
        self._rows: list[_NoticeRow] = []
        self._transient_row: _NoticeRow | None = None
        self._collapsed = False
        self.setObjectName("noticeTray")
        self.setMaximumWidth(NOTICE_MAX_WIDTH)
        self.setVisible(False)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(4)

        header = QHBoxLayout()
        header.setSpacing(4)
        self._toggle = QToolButton()
        self._toggle.setObjectName("noticeToggle")
        self._toggle.setAutoRaise(True)
        self._toggle.setFixedSize(18, 18)
        self._toggle.clicked.connect(self.toggle_expanded)
        self._summary = QLabel("0")
        self._summary.setObjectName("noticeSummary")
        self._summary.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._clear = QToolButton()
        self._clear.setObjectName("noticeClearAll")
        self._clear.setAutoRaise(True)
        self._clear.setFixedSize(18, 18)
        self._clear.setText("×")
        self._clear.setToolTip(t("Clear all notices"))
        self._clear.clicked.connect(self.clear_all)
        header.addWidget(self._toggle)
        header.addWidget(self._summary, 1)
        header.addWidget(self._clear)
        outer.addLayout(header)

        self._row_box = QVBoxLayout()
        self._row_box.setSpacing(4)
        outer.addLayout(self._row_box)

        self._update_toggle()

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def add_notice(self, text: str, kind: str) -> None:
        if kind in PERSISTENT_NOTICE_KINDS:
            # Persistent kinds stack, but identical repeated notices (ex. the
            # same warning raised per tile) are skipped so the tray stays
            # readable.
            for row in self._rows:
                if row.text() == text:
                    return
            row = _NoticeRow(text, kind, autohide=False)
            row.dismissed.connect(lambda: self._remove_row(row))
            self._rows.append(row)
            self._row_box.insertWidget(0, row)
        else:
            # Transient kinds coalesce into a single live chip: each progress
            # update ("CHM 12/35", "CHM 13/35", ...) replaces the text instead
            # of stacking a row per tile. Progress-shaped updates pin the row
            # (no auto-hide) so it never flickers during sparse heavy phases.
            if self._transient_row is not None:
                self._transient_row.set_text(text)
                if _PROGRESS_RE.search(text):
                    self._transient_row.pin()
                else:
                    self._transient_row.restart_autohide()
                self._refit()
                return
            row = _NoticeRow(text, kind, autohide=True)
            row.dismissed.connect(lambda: self._remove_row(row))
            self._rows.append(row)
            self._row_box.insertWidget(0, row)
            self._transient_row = row
        if len(self._rows) > NOTICE_MAX_ROWS:
            oldest = self._rows.pop()
            if oldest is self._transient_row:
                self._transient_row = None
            self._row_box.removeWidget(oldest)
            oldest.deleteLater()
        self._refit()

    def clear_all(self) -> None:
        while self._rows:
            row = self._rows.pop()
            self._row_box.removeWidget(row)
            row.deleteLater()
        self._transient_row = None
        self._refit()

    def discard(self, text: str) -> None:
        """Remove persistent rows matching text (run-scoped notices)."""
        for row in [r for r in self._rows if r.text() == text]:
            self._remove_row(row)

    def remove_transient(self) -> None:
        """Drop the live progress chip, keeping persistent warnings/errors."""
        if self._transient_row is not None:
            self._remove_row(self._transient_row)

    def toggle_expanded(self) -> None:
        self._collapsed = not self._collapsed
        self._update_toggle()
        for row in self._rows:
            row.setVisible(not self._collapsed)
        self._refit()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _remove_row(self, row: _NoticeRow) -> None:
        if row not in self._rows:
            return
        self._rows.remove(row)
        if row is self._transient_row:
            self._transient_row = None
        self._row_box.removeWidget(row)
        row.deleteLater()
        self._refit()

    def _update_toggle(self) -> None:
        self._toggle.setText("▸" if self._collapsed else "▾")
        self._toggle.setToolTip(
            t("Expand notices") if self._collapsed else t("Collapse notices")
        )

    def _refit(self) -> None:
        self._summary.setText(str(len(self._rows)))
        visible = bool(self._rows) and not self._collapsed
        for row in self._rows:
            row.setVisible(visible)
        if self._rows:
            if not self._collapsed:
                # Expanded: fixed width so the tray does not resize as rows
                # are added/removed (keeps the text readable and stable).
                self.setMinimumWidth(NOTICE_MAX_WIDTH)
                self.setMaximumWidth(NOTICE_MAX_WIDTH)
            else:
                # Collapsed: shrink to the header only.
                self.setMinimumWidth(0)
                self.setMaximumWidth(NOTICE_MAX_WIDTH)
            self.setVisible(True)
            self.adjustSize()
            self._canvas._position_notice_tray()
        else:
            self.setMinimumWidth(0)
            self.setMaximumWidth(NOTICE_MAX_WIDTH)
            self.setVisible(False)


def _setting_bool(key: str, default: bool) -> bool:
    """Read a bool QSetting without ever breaking on junk values."""
    try:
        value = QSettings(SETTINGS_ORG, SETTINGS_APP).value(key, default)
    except Exception:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in ("", "0", "false", "no", "off")


def _config_int(configs: dict | None, iid: str, key: str) -> int:
    """Integer config value, junk-tolerant (never breaks run start)."""
    try:
        return int((configs.get(iid) or {}).get(key) or 0)
    except (AttributeError, TypeError, ValueError):
        return 0


def _is_loader_caps(caps: dict) -> bool:
    """Capabilities predicate matching tiling source nodes."""
    try:
        return bool((caps or {}).get("triggers_tiling"))
    except AttributeError:
        return False


def _total_ram_bytes() -> float | None:
    """Physical RAM for budget advice; None when unreadable."""
    try:
        import psutil

        return float(psutil.virtual_memory().total)
    except Exception:
        return None


def _run_preferences() -> dict:
    """Performance preferences for the next run (Preferences dialog).

    Reads QSettings with safe fallbacks (a hand-edited registry never
    breaks a run): CPU/memory percentages plus the acceleration policy.
    """
    settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
    mode = settings.value(ACCEL_MODE_KEY, "auto") or "auto"
    if mode not in {"auto", "on", "off"}:
        mode = "auto"
    backend = settings.value(ACCEL_BACKEND_KEY, "auto") or "auto"
    if backend not in {"auto", "cpu", "cuda", "opencl", "directml"}:
        backend = "auto"
    op_name = str(settings.value(OPERATOR_NAME_KEY, "") or "").strip()
    op_org = str(settings.value(OPERATOR_ORG_KEY, "") or "").strip()
    return {
        "max_workers": workers_for_percent(
            settings.value(CPU_PERCENT_KEY, CPU_PERCENT_DEFAULT),
            os.cpu_count() or 4,
        ),
        "memory_budget_bytes": memory_budget_for_percent(
            settings.value(MEMORY_PERCENT_KEY, MEMORY_PERCENT_DEFAULT)
        ),
        "acceleration_mode": mode,
        "acceleration_backend": backend,
        "acceleration_device": settings.value(ACCEL_DEVICE_KEY, "") or "",
        "keep_intermediates": _setting_bool(KEEP_INTERMEDIATES_KEY, False),
        "operator": {"name": op_name, "org": op_org}
        if (op_name or op_org)
        else None,
    }


class NodeCanvasView(QGraphicsView):
    """Node canvas with zoom, pan, drag-and-drop, and pipeline delegation."""

    zoomChanged = Signal(float)
    message_requested = Signal(str, str)  # text, kind ("info"|"action"|"success"|"error")
    graph_changed = Signal()
    node_selected = Signal(object)  # NodeItem | None.
    nodes_selected = Signal(object)  # list[NodeItem]: multi-selection panel.
    node_status_changed = Signal(object, str)
    load_tile_count = Signal(int)
    outputs_updated = Signal(object)  # list[OutputProduct].
    pipeline_started = Signal()
    pipeline_finished = Signal(dict, list, int, list, dict)  # outputs, tile_results, failed, tiles, metrics
    pipeline_cancelled = Signal()
    pipeline_closed = Signal()  # pool torn down: next queued run may start
    pipeline_progress = Signal(int, int)  # done, total
    inspector_refresh_requested = Signal()  # Refresh the selected node.
    segment_started = Signal(int, int, int)  # k, n, point count of the batch.
    segment_outputs_changed = Signal(int, int, object)  # k, n, batch outputs dict.
    consolidated_outputs_ready = Signal(object)  # final/merged outputs dict.
    barrier_outputs_changed = Signal(str, object)  # iid, barrier payload.
    barrier_started = Signal(str)  # iid of the dispatched barrier.
    node_preview_requested = Signal(str, str)  # path, node EN name (bridge preview).
    open_preferences_requested = Signal(str)  # preferences section name (EN).
    history_changed = Signal(bool, bool)  # can_undo, can_redo (item 9).
    history_applied = Signal()  # after undo/redo restored a document.

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._scene = NodeGraphScene(self)
        self.setScene(self._scene)
        from lynceus.ui.fonts import font_changed
        from lynceus.ui.theming import theme_changed

        theme_changed.connect(self._on_theme_changed)
        font_changed.connect(self._on_font_changed)
        # Native scene changes (Ctrl+click multi-select, rubber band,
        # background-click deselect) bypass the item signals: funnel every
        # change through the inspector rule here (single -> item, several ->
        # multi panel, zero -> placeholder).
        self._syncing_selection = False
        self._selected_item = None
        self._inspector_multi = None
        self._inspector_busy = False
        self._scene.selectionChanged.connect(self._sync_inspector_to_scene)
        self.setAcceptDrops(True)
        # Drag events are delivered to the viewport widget: without this,
        # Qt logs "drag leave received before drag enter" on every
        # library drop (harmless, but noisy).
        self.viewport().setAcceptDrops(True)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Left-drag on empty space draws a selection band (Ctrl adds to the
        # current selection); the middle button keeps panning (handled in
        # mousePressEvent before the base class sees the event).
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self._node_items: list[NodeItem] = []
        self._connections: list[Connection] = []
        self._panning = False
        self._pan_start = QPointF()
        self._pan_h_start = 0
        self._pan_v_start = 0
        self._wire_source: Port | None = None
        self._ghost_item: QGraphicsPathItem | None = None
        self._item_by_iid: dict[str, NodeItem] = {}
        self._snap_enabled = _setting_bool(SNAP_KEY, True)
        # Restoring a graph must not auto-select an item before its config is
        # available to the Inspector.
        self._restoring = False
        # Undo/redo document history (item 9). Empty until seeded by
        # clear_history(); _undo_active suppresses recording during restores.
        self._history: list[dict] = []
        self._cursor = 0
        self._undo_active = False
        self._merge_since: float | None = None
        self._last_kind: str | None = None  # kind of the recorded mutation
        self._op_kind: str | None = None  # overrides kind during a batch op
        # Session root for the loaded project; None uses the loose output root.
        self.session_root = None

        # Inline label editor; None means inactive.
        self._label_editor: _LabelEditor | None = None
        self._label_editor_item: NodeItem | None = None

        self._pipeline = PipelineController()

        # Active-run state. While running, edits are staged against the graph
        # snapshot and are applied or discarded when the run finishes.
        self._run_active = False
        self._graph_snapshot: dict | None = None
        # File-queue driver: None, or {"iid", "files", "index", "results",
        # "order", "nodes", "edges", "configs", "source_paths", "prefs",
        # "stop", "got_finished"}. One engine run per file, chained from
        # the post-cleanup closed signal (never from finished: the old
        # pool is still alive until cleanup joins it).
        self._file_queue: dict | None = None
        self._queue_flat: list = []
        # Live gallery: per-barrier products published at completion
        # ({iid: [OutputProduct]}), plus whether a segment batch owns the
        # phase display (batches publish authoritatively at batch end).
        self._live_products: dict[str, list] = {}
        self._batch_active = False
        self._had_segments = False
        self._segment_products: dict = {}
        self._flat_products: list = []
        # Provisional-styling FYI: shown once per run at the first
        # pre-finish publish, removed when the run ends.
        self._live_notice_shown = False
        self._live_notice_text = ""

        # EULA gates: the UI gate may show dialogs; the controller gate runs in
        # the worker and must never touch Qt.
        self.consent_gate: Callable[[str], tuple[bool, str | None]] | None = None

        self._install_toolbar()
        self._toolbar.btn_snap.setChecked(self._snap_enabled)
        self._install_zoom_shortcuts()
        self._install_delete_shortcuts()
        self._install_clipboard_shortcuts()
        self._install_history_shortcuts()
        self.clear_history()

        self.node_status_changed.connect(self._on_node_status)
        self.load_tile_count.connect(self._on_load_tile_count)
        self.pipeline_finished.connect(self._on_pipeline_finished)
        self.pipeline_cancelled.connect(self._on_pipeline_cancelled)
        self.pipeline_closed.connect(self._on_pipeline_closed)
        self.segment_started.connect(self._on_segment_started)
        self.segment_outputs_changed.connect(self._on_segment_outputs)
        self.consolidated_outputs_ready.connect(self._on_consolidated_outputs)
        self.barrier_outputs_changed.connect(self._on_barrier_outputs)
        self.barrier_started.connect(self._on_barrier_started)
        self.graph_changed.connect(self._update_pending_chip)

        self._install_status_chip()
        self._install_notice_tray()
        self.message_requested.connect(self._feed_notice)
        self.pipeline_started.connect(self._notice_tray.clear_all)
        # Fit/reset/zoom change viewport scrollbars; the scrollarea reflows
        # them one event-loop tick later, so overlays re-anchor deferred.
        self.zoomChanged.connect(self._schedule_reanchor)
        self.viewport().installEventFilter(self)

    def _schedule_reanchor(self, *_args) -> None:
        """Re-anchor overlays after the scrollbars reflow.

        Called on every view change (fit/reset/zoom/wheel/restore). ``zoomChanged``
        fires before the scrollarea has resized the viewport, so the actual
        repositioning runs one event-loop tick later.
        """
        QTimer.singleShot(0, self._reanchor_overlays)

    def _reanchor_overlays(self) -> None:
        self._position_status_chip()
        self._position_notice_tray()

    def set_consent_gates(self, gate_ui, gate_worker) -> None:
        """Install UI and worker-thread consent gates from MainWindow.

        The UI gate may show dialogs when adding nodes. The worker gate returns
        a blocking reason and must not access Qt.
        """
        self.consent_gate = gate_ui
        self._pipeline.eula_checker = gate_worker

    # ------------------------------------------------------------------
    # Pending edits during a run (lightweight staging)
    # ------------------------------------------------------------------

    def is_run_active(self) -> bool:
        """Return whether a run is active on the GUI thread."""
        return self._run_active

    def _set_run_pending(self, active: bool) -> None:
        """Dim the canvas while staging edits against a run snapshot.

        Live edits remain visible but are excluded from the isolated session.
        The pending-changes chip is updated independently.
        """
        opacity = PENDING_OPACITY if active else 1.0
        for item in self._node_items:
            item.setOpacity(opacity)
        for connection in self._connections:
            connection.path_item.setOpacity(opacity)
        self._update_pending_chip()

    def _clear_run_pending(self) -> None:
        """End staging without changing the live canvas."""
        self._run_active = False
        self._graph_snapshot = None
        self._set_run_pending(False)
        self._emit_history_state()

    def _pending_diff(self) -> dict:
        """Compare the live canvas with the run-start snapshot.

        The result summarizes added/removed nodes and edges plus changed
        instance configuration for the pending-edits dialog.
        """
        snap = self._graph_snapshot or {}
        snap_nodes = {n["iid"]: n for n in snap.get("nodes", [])}
        snap_edges = {
            (e["src"], e["dst"], e.get("src_out", 0), e.get("dst_in", 0))
            for e in snap.get("edges", [])
        }

        current = self.serialize_graph()
        cur_nodes = {n["iid"]: n for n in current["nodes"]}
        cur_edges = {
            (e["src"], e["dst"], e.get("src_out", 0), e.get("dst_in", 0))
            for e in current["edges"]
        }

        added = sorted(set(cur_nodes) - set(snap_nodes))
        removed = sorted(set(snap_nodes) - set(cur_nodes))
        added_edges = len(cur_edges - snap_edges)
        removed_edges = len(snap_edges - cur_edges)
        config_changed = []
        for iid in set(snap_nodes) & set(cur_nodes):
            if snap_nodes[iid].get("config") != cur_nodes[iid].get("config"):
                config_changed.append(iid)

        return {
            "added_nodes": added,
            "removed_nodes": removed,
            "added_edges": added_edges,
            "removed_edges": removed_edges,
            "config_changed": config_changed,
        }

    def _pending_has_changes(self) -> bool:
        """True when the canvas differs from the run-start snapshot."""
        diff = self._pending_diff()
        return any(
            [*diff["added_nodes"], *diff["removed_nodes"],
             diff["added_edges"], diff["removed_edges"],
             *diff["config_changed"]]
        )

    def _install_status_chip(self) -> None:
        """Discreet corner chips while a run lasts.

        They stay out of the toolbar's way: read-only (mouse-transparent),
        positioned over the viewport and NEVER replaced by other messages.
        """
        self._pending_chip = QLabel("", self.viewport())
        self._pending_chip.setObjectName("canvasStatusChip")
        self._pending_chip.setProperty("origText_en", "Changes pending")
        self._pending_chip.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )
        self._pending_chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pending_chip.setVisible(False)

        # Segment progress chip ("Segment 2/5 · N points"), shown above the
        # pending-edits chip while a segmented run is queuing/processing.
        self._segment_chip = QLabel("", self.viewport())
        self._segment_chip.setObjectName("canvasSegmentChip")
        self._segment_chip.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )
        self._segment_chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._segment_chip.setVisible(False)

    def _position_status_chip(self) -> None:
        chip = getattr(self, "_pending_chip", None)
        if chip is None:
            return
        viewport = self.viewport()
        seg_chip = getattr(self, "_segment_chip", None)
        if seg_chip is not None and seg_chip.isVisible():
            seg_chip.adjustSize()
            seg_chip.move(
                viewport.width() - seg_chip.width() - CHIP_MARGIN,
                viewport.height() - seg_chip.height() - CHIP_MARGIN,
            )
            chip.adjustSize()
            chip.move(
                viewport.width() - chip.width() - CHIP_MARGIN,
                viewport.height() - chip.height() - seg_chip.height()
                - CHIP_MARGIN * 2,
            )
            return
        chip.adjustSize()
        chip.move(
            viewport.width() - chip.width() - CHIP_MARGIN,
            viewport.height() - chip.height() - CHIP_MARGIN,
        )

    def _install_notice_tray(self) -> None:
        """Stacked notice chips anchored above the pending-changes chip."""
        self._notice_tray = _NoticeTray(self)

    def _feed_notice(self, text: str, kind: str) -> None:
        tray = getattr(self, "_notice_tray", None)
        if tray is None or not text:
            return
        tray.add_notice(text, kind)

    def post_notice(self, text: str, kind: str = "info") -> None:
        """Public entry point for a persistent canvas notice (item 45, etc.).

        Persistent kinds (warning/error) stay until dismissed; callers must
        run this on the GUI thread.
        """
        self._feed_notice(text, kind)

    def _maybe_show_live_notice(self) -> None:
        """One FYI per run: pre-finish products render with generic styles.

        Anything visible before the run ends lacks its final `.qml` sidecar
        (generated post-DAG), so the viewer falls back to generic styling
        until the closing publish refreshes it. Shown once, removed when
        the run ends (finish/cancel/queue close).

        English by policy: engine/run notices stay in English (operators
        share them verbatim); only the surrounding UI chrome is translated.
        """
        if self._live_notice_shown:
            return
        self._live_notice_shown = True
        tray = getattr(self, "_notice_tray", None)
        if tray is None:
            return
        self._live_notice_text = (
            "Provisional styling: product previews use generic styles "
            "until the run finishes."
        )
        tray.add_notice(self._live_notice_text, "warning")

    def _hide_live_notice(self) -> None:
        text = getattr(self, "_live_notice_text", "")
        tray = getattr(self, "_notice_tray", None)
        if tray is not None and text:
            tray.discard(text)
        self._live_notice_text = ""

    def _position_notice_tray(self) -> None:
        tray = getattr(self, "_notice_tray", None)
        if tray is None or not tray.isVisible():
            return
        viewport = self.viewport()
        chip = getattr(self, "_pending_chip", None)
        seg_chip = getattr(self, "_segment_chip", None)
        tray.adjustSize()
        y = viewport.height() - tray.height() - NOTICE_MARGIN
        offset = 0
        if seg_chip is not None and seg_chip.isVisible():
            offset += seg_chip.height() + CHIP_MARGIN
        if chip is not None and chip.isVisible():
            offset += chip.height() + CHIP_MARGIN
        y -= offset
        x = viewport.width() - tray.width() - NOTICE_MARGIN
        tray.move(max(NOTICE_MARGIN, x), max(NOTICE_MARGIN, y))

    def _update_pending_chip(self) -> None:
        """Show the status chip only while a run has pending edits."""
        chip = getattr(self, "_pending_chip", None)
        if chip is None:
            return
        if self._run_active and self._pending_has_changes():
            chip.setText(t("Changes pending"))
            self._position_status_chip()
            chip.setVisible(True)
        else:
            chip.hide()
        self._position_notice_tray()

    def _ask_apply_pending(self) -> None:
        """Ask whether to apply or discard edits made during the run.

        Apply keeps the live canvas; Discard restores the run-start snapshot.
        No dialog is shown when there are no pending edits.
        """
        diff = self._pending_diff()
        if not self._pending_has_changes():
            self._clear_run_pending()
            return

        lines = []
        if diff["added_nodes"]:
            lines.append(
                t("- {n} node(s) added").format(n=len(diff["added_nodes"]))
            )
        if diff["removed_nodes"]:
            lines.append(
                t("- {n} node(s) removed").format(n=len(diff["removed_nodes"]))
            )
        if diff["added_edges"] or diff["removed_edges"]:
            lines.append(
                t(
                    "- {a} connection(s) added, {r} connection(s) removed"
                ).format(a=diff["added_edges"], r=diff["removed_edges"])
            )
        if diff["config_changed"]:
            lines.append(
                t("- configuration changed on {n} node(s)").format(
                    n=len(diff["config_changed"])
                )
            )
        details = "\n".join(lines)

        box = QMessageBox(self)
        box.setWindowTitle(t("Run finished"))
        box.setIcon(QMessageBox.Icon.Question)
        box.setText(
            t(
                "The run finished. While it was running you changed the canvas.\n"
                "These changes do not affect the finished session."
            )
        )
        box.setInformativeText(details)
        apply_btn = box.addButton(
            t("Apply changes"), QMessageBox.ButtonRole.AcceptRole
        )
        box.addButton(
            t("Discard changes"), QMessageBox.ButtonRole.DestructiveRole
        )
        box.exec()

        if box.clickedButton() is apply_btn:
            self._clear_run_pending()
            # The staged edits are now the live graph: record them so a
            # single undo reverts to the pre-run state.
            self._record_current_state("run")
        else:
            self._discard_pending_changes()

    def _discard_pending_changes(self) -> None:
        """Restore the canvas to its exact pre-run state."""
        snapshot = self._graph_snapshot or {}
        self._run_active = False
        self._graph_snapshot = None
        self.restore_graph(snapshot)
        self._set_run_pending(False)
        self._merge_since = None
        self._emit_history_state()

    # ------------------------------------------------------------------
    # Live state (GUI thread)
    # ------------------------------------------------------------------

    def _on_node_status(self, item: NodeItem, status: str) -> None:
        item.set_status(status)

    def _on_load_tile_count(self, count: int) -> None:
        for item in self._node_items:
            if item.node_id() == "lynceus.nodes.lidar.source.load_las_laz":
                item.set_tile_count(count)

    def _flatten_products(self, outputs: dict, segment: str | None = None,
                           segment_index: int | None = None,
                           file_overrides: dict | None = None) -> list:
        """Flatten run outputs into gallery products (Qt-free helper).

        ``file_overrides`` maps instance IDs to the file a queue iteration
        actually ran with: lineage basenames and source previews follow the
        iteration file, not the item's stored head.
        """
        failed_nodes = set(
            self._pipeline.session_state.get("failed_nodes", [])
        )
        node_info = {
            item.iid: item.node_name_en()
            for item in self._node_items
            if item.iid not in failed_nodes
        }
        custom_labels = {
            iid: item.label()
            for iid, item in self._item_by_iid.items()
            if item.label()
        }
        overrides = dict(file_overrides or {})
        source_files: dict[str, str] = {}
        source_paths: dict[str, str] = {}
        for iid, item in self._item_by_iid.items():
            if iid in overrides:
                path = overrides[iid]
            else:
                file_path = getattr(item, "file_path", None)
                if not callable(file_path):
                    continue
                try:
                    path = file_path()
                except Exception:
                    continue
            if path:
                source_files[iid] = os.path.basename(path)
                source_paths[iid] = path
        edges = [
            (c.source.node.iid, c.target.node.iid)
            for c in self._connections
        ]
        from lynceus.ui.outputs_model import flatten_outputs

        return flatten_outputs(
            outputs, node_info, segment=segment,
            segment_index=segment_index, edges=edges,
            custom_labels=custom_labels, source_files=source_files,
            source_paths=source_paths,
        )

    @Slot(int, int, int)
    def _on_segment_started(self, k: int, n: int, points: int) -> None:
        """Show the segment progress chip while batches are queued/processing."""
        self._had_segments = True
        self._batch_active = True
        chip = getattr(self, "_segment_chip", None)
        if chip is None:
            return
        chip.setText(
            t("Segment {k}/{n} · {points} points").format(
                k=k, n=n, points=f"{points:,}"
            )
        )
        chip.setVisible(True)
        self._position_status_chip()

    @Slot(int, int, object)
    def _on_segment_outputs(self, k: int, n: int, outputs: dict) -> None:
        """Publish a finished segment's products to the gallery immediately."""
        self._batch_active = False
        if not outputs:
            return
        self._maybe_show_live_notice()
        if self._file_queue is not None:
            # Queue iteration: tag with the file block and accumulate; the
            # gallery refreshes from the accumulated list at each finish.
            label = self._queue_file_label()
            self._queue_iter = getattr(self, "_queue_iter", [])
            self._queue_iter.extend(self._flatten_products(
                outputs,
                segment=f"{label} · Segment {k}/{n}",
                segment_index=k,
                file_overrides=self._queue_override(),
            ))
            return
        self._segment_products[k] = self._flatten_products(
            outputs,
            segment="Segment {k}/{n}".format(k=k, n=n),
            segment_index=k,
        )
        self._emit_gallery()

    @Slot(object)
    def _on_consolidated_outputs(self, outputs: dict) -> None:
        """Store the final/merged products (consolidated, no segment tag)."""
        if not outputs:
            return
        self._maybe_show_live_notice()
        # Live entries republished here authoritatively would duplicate.
        for iid in outputs:
            self._live_products.pop(iid, None)
        if self._file_queue is not None:
            self._queue_iter = getattr(self, "_queue_iter", [])
            self._queue_iter.extend(self._flatten_products(
                outputs,
                segment=self._queue_file_label(),
                segment_index=self._file_queue["index"],
                file_overrides=self._queue_override(),
            ))
            return
        self._flat_products = self._flatten_products(outputs)
        self._emit_gallery()

    @Slot(str, object)
    def _on_barrier_outputs(self, iid: str, payload: dict) -> None:
        """Publish one finished barrier to the gallery immediately.

        Products appear under their node as merges complete (e.g. the CHM
        is previewable while downstream nodes still run) instead of waiting
        for the whole run. Segment batches are skipped: the batch end
        publishes authoritatively and a live entry would duplicate it.
        In file-queue mode the iteration accumulates silently (emitting now
        would hide the previous iterations) and flushes at its finish.
        """
        if not isinstance(payload, dict) or not payload:
            return
        if self._batch_active:
            return
        if self._file_queue is not None:
            flattened = self._flatten_products(
                {iid: payload},
                segment=self._queue_file_label(),
                segment_index=self._file_queue["index"],
                file_overrides=self._queue_override(),
            )
        else:
            flattened = self._flatten_products({iid: payload})
        if not flattened:
            return
        self._maybe_show_live_notice()
        self._live_products[iid] = flattened
        if self._file_queue is None:
            self._emit_gallery()

    @Slot(str)
    def _on_barrier_started(self, iid: str) -> None:
        """Sticky phase chip while a merge/barrier runs (big files).

        Tile counters freeze during a long single merge; the chip names the
        running barrier until the next phase or the finish hides it.
        """
        if self._batch_active:
            return  # the segment chip owns the phase display
        chip = getattr(self, "_segment_chip", None)
        if chip is None:
            return
        item = self._item_by_iid.get(iid)
        try:
            label = item.node_name_en() if item is not None else iid
        except Exception:
            label = iid
        chip.setText(f"Merging {label}...")
        chip.setVisible(True)
        self._position_status_chip()

    def _emit_gallery(self) -> None:
        """Rebuild the gallery from accumulated segment + consolidated products."""
        products = []
        for k in sorted(self._segment_products):
            products.extend(self._segment_products[k])
        for _iid, live in self._live_products.items():
            products.extend(live)
        products.extend(self._flat_products)
        self.outputs_updated.emit(products)

    def _clear_node_statuses(self, only_processing: bool = False) -> None:
        """Return node outlines to the base color.

        ``only_processing`` clears just the pulsing nodes (cancel of a stray
        queued status); otherwise every node goes back to IDLE, so a cancelled
        run never leaves blinking outlines behind.
        """
        for item in self._node_items:
            if not only_processing or item.status() == STATUS_PROCESSING:
                item.set_status(STATUS_IDLE)

    def _on_pipeline_finished(
        self, outputs: dict, tile_results: list, failed: int, tiles: list,
        metrics: dict,
    ) -> None:
        """Publish visualizable products to the gallery on the GUI thread.

        Outputs are indexed by instance ID; failed or removed nodes are omitted.
        Segmented runs already published every segment incrementally and the
        consolidated products via ``consolidated_outputs_ready``; the flat run
        outputs are only re-emitted for non-segmented runs.
        """
        if self._file_queue is not None:
            # Queue iteration: record, stash its gallery block and wait for
            # the post-cleanup closed signal to advance (never here: the
            # pool is still alive until cleanup joins it).
            self._record_queue_result(outputs, failed, tiles, metrics)
            return
        self._sync_toolbar_state("idle")
        chip = getattr(self, "_segment_chip", None)
        if chip is not None:
            chip.hide()
        tray = getattr(self, "_notice_tray", None)
        if tray is not None:
            tray.remove_transient()
        # Safety net: nodes still pulsing after the final statuses (e.g. a node
        # whose "done" never reached the GUI) return to the base outline.
        self._clear_node_statuses(only_processing=True)
        if not self._had_segments:
            self._flat_products = self._flatten_products(outputs)
            self._live_products = {}
            self._emit_gallery()
        self._hide_live_notice()

        # Staging: apply/discard the changes made during the run.
        self._ask_apply_pending()

        # Persistent failure summary (item 44): errors in the tray survive
        # until dismissed, unlike the ephemeral status-bar message.
        if failed > 0 and metrics:
            self._feed_notice(self._failure_summary_text(failed, metrics), "error")

    def _failure_summary_text(self, failed: int, metrics: dict) -> str:
        """Persistent end-of-run report: failed nodes, reasons, session."""
        failed_nodes = metrics.get("failed_nodes")
        names: dict = {}
        for iid in failed_nodes or {}:
            item = self._item_by_iid.get(iid)
            try:
                names[iid] = item.display_name() if item is not None else iid
            except Exception:
                names[iid] = iid
        return failure_report_text(
            failed,
            failed_nodes,
            names,
            metrics.get("peak_memory_mb"),
            metrics.get("session_dir") or "",
        )

    def _on_pipeline_cancelled(self) -> None:
        """Reset the toolbar and discard staging after cancellation."""
        if self._file_queue is not None:
            # A queue iteration was cancelled: stop the whole queue; the
            # post-cleanup closed signal finishes with a partial summary.
            self._file_queue["stop"] = True
            return
        self._sync_toolbar_state("idle")
        # A cancelled run leaves nodes in "processing" (the worker never emits
        # a final status): restore every outline to the base color and stop
        # the pulse timers.
        self._clear_node_statuses()
        chip = getattr(self, "_segment_chip", None)
        if chip is not None:
            chip.hide()
        tray = getattr(self, "_notice_tray", None)
        if tray is not None:
            tray.remove_transient()
        self._hide_live_notice()
        self._clear_run_pending()

    # ------------------------------------------------------------------
    # File-queue driver (GUI thread)
    # ------------------------------------------------------------------

    def _manage_file_queue(self, item) -> None:
        """Open the queue manager for one file-backed node item."""
        from lynceus.ui.nodes.widgets.file_queue_dialog import (
            FileQueueDialog,
        )

        dialog = FileQueueDialog(
            self, item.file_queue(), item._queue_filters())
        dialog.exec()
        item.set_file_queue(dialog.queue())

    def _queue_file_label(self) -> str:
        """Gallery/chip label of the iteration being published."""
        queue = self._file_queue or {}
        files = queue.get("files", [])
        index = int(queue.get("index", 0))
        name = os.path.basename(files[index]) if index < len(files) else "?"
        total = len(files)
        return t("File {k} of {n}: {name}").format(
            k=index + 1, n=total, name=name)

    def _queue_override(self) -> dict:
        """Per-run file override for the current queue iteration."""
        queue = self._file_queue or {}
        files = queue.get("files", [])
        index = int(queue.get("index", 0))
        if not queue.get("iid") or index >= len(files):
            return {}
        return {queue["iid"]: files[index]}

    def _start_file_queue(self, item, nodes: list, edges: list,
                          source_paths: dict, configs: dict,
                          run_prefs: dict) -> None:
        """Begin a chained queue: one engine run per existing file."""
        from lynceus.nodes import _file_queue as fq

        existing, missing = fq.partition_existing(item.file_queue())
        order = list(item.file_queue())
        results = [
            {"file": path, "ok": False, "reason": "File not found"}
            for path in missing
        ]
        if not existing:
            self._emit(fq.summarize_queue(results), "warning")
            self._sync_toolbar_state("idle")
            self._clear_run_pending()
            return
        if len(existing) > 1:
            self._warn_queue_merge(item, nodes, edges, configs)
        self._file_queue = {
            "iid": item.iid,
            "files": existing,
            "index": 0,
            "results": results,
            "order": order,
            "nodes": list(nodes),
            "edges": list(edges),
            "configs": copy.deepcopy(configs),
            "source_paths": dict(source_paths),
            "prefs": dict(run_prefs),
            "stop": False,
            "got_finished": False,
        }
        self._queue_flat = []
        self._launch_queue_iteration()

    def _warn_queue_merge(self, item, nodes: list, edges: list,
                          configs: dict) -> None:
        """Heads-up when a chain re-runs a consolidation for every file."""
        from lynceus.processing.steps import effective_node_caps

        merge_iids = {
            iid
            for iid, nid in nodes or []
            if effective_node_caps(nid, configs.get(iid)).get("consolidator")
        }
        if not merge_iids:
            return
        children: dict[str, set] = {}
        for edge in edges or []:
            if len(edge) < 2:
                continue
            children.setdefault(edge[0], set()).add(edge[1])
        seen: set = set()
        stack = [item.iid]
        reachable: set = set()
        while stack:
            iid = stack.pop()
            if iid in seen:
                continue
            seen.add(iid)
            if iid in merge_iids:
                reachable.add(iid)
            stack.extend(children.get(iid, ()))
        if reachable:
            names = [
                self._item_by_iid.get(iid).display_name()
                for iid in sorted(reachable)
                if self._item_by_iid.get(iid) is not None
            ]
            self._emit(
                t(
                    "File queue: the merge/consolidate node(s) will re-run "
                    "for every queued file ({names})"
                ).format(names=", ".join(names)),
                "warning",
            )

    def _launch_queue_iteration(self) -> None:
        """Start the engine run for the current queue file (GUI thread).

        Runs on frozen run arguments (nodes/edges/configs/source_paths from
        queue start) plus the per-file override: live edits stay staged and
        never leak into later iterations. Called from the post-cleanup
        closed signal, so the previous pool is already joined.
        """
        queue = self._file_queue
        if queue is None:
            return
        queue["got_finished"] = False
        self._queue_iter = []
        # Fresh per-iteration accumulators (stashed into _queue_flat on
        # finish); node outlines pulse again from idle.
        self._had_segments = False
        self._segment_products = {}
        self._flat_products = []
        self._live_products = {}
        self._batch_active = False
        scope_iids = {iid for iid, _ in queue["nodes"]}
        for iid, item in self._item_by_iid.items():
            if iid in scope_iids:
                item.set_status(STATUS_IDLE)
        chip = getattr(self, "_segment_chip", None)
        if chip is not None:
            chip.setText(self._queue_file_label())
            chip.setVisible(True)
            self._position_status_chip()
        prefs = queue["prefs"]
        self._pipeline.set_acceleration_mode(prefs["acceleration_mode"])
        self._pipeline.set_acceleration_backend(
            prefs["acceleration_backend"], prefs["acceleration_device"]
        )
        self.pipeline_started.emit()
        self._pipeline.run(
            queue["nodes"], queue["edges"], queue["source_paths"],
            self._pipeline_callbacks(), session_root=self.session_root,
            configs=copy.deepcopy(queue["configs"]),
            max_workers=prefs["max_workers"],
            memory_budget_bytes=prefs["memory_budget_bytes"],
            operator=prefs["operator"],
            keep_intermediates=prefs["keep_intermediates"],
            file_overrides=self._queue_override(),
        )

    def _record_queue_result(self, outputs: dict, failed: int,
                             tiles: list, metrics: dict) -> None:
        """Stash one iteration: result row + its gallery block."""
        queue = self._file_queue
        if queue is None:
            return
        files = queue["files"]
        index = int(queue["index"])
        path = files[index] if index < len(files) else "?"
        blocked = next(
            (str(key) for key in (metrics or {})
             if str(key).startswith("blocked_by_")),
            None,
        )
        if blocked is not None:
            ok, reason = False, f"Blocked ({blocked})"
        elif failed:
            names = []
            failed_map = (
                metrics.get("failed_nodes")
                if isinstance(metrics, dict) else None
            )
            if isinstance(failed_map, dict) and failed_map:
                failed_iids = sorted(failed_map)
            else:
                try:
                    failed_iids = self._pipeline.session_state.get(
                        "failed_nodes", [])
                except Exception:
                    failed_iids = []
            for iid in failed_iids:
                item = self._item_by_iid.get(iid)
                try:
                    names.append(
                        item.display_name() if item is not None else iid)
                except Exception:
                    names.append(iid)
            detail = ", ".join(names[:3])
            if len(names) > 3:
                detail += ", …"
            reason = f"{failed} failed task(s): {detail}" if detail else (
                f"{failed} failed task(s)")
            ok = False
        else:
            ok, reason = True, ""
        queue["results"].append({"file": path, "ok": ok, "reason": reason})
        queue["got_finished"] = True
        label = self._queue_file_label()
        per_file = list(getattr(self, "_queue_iter", []))
        for _iid, live in self._live_products.items():
            per_file.extend(live)
        if not self._had_segments:
            fresh = {
                iid: out for iid, out in outputs.items()
                if iid not in self._live_products
            }
            per_file.extend(self._flatten_products(
                fresh, segment=label, segment_index=index,
                file_overrides=self._queue_override(),
            ))
        else:
            per_file.extend(self._flat_products)
        self._queue_flat.extend(per_file)
        self._queue_iter = []
        self._live_products = {}
        if per_file:
            self._maybe_show_live_notice()
        self.outputs_updated.emit(list(self._queue_flat))
        self._clear_node_statuses(only_processing=True)

    def _on_pipeline_closed(self) -> None:
        """Advance the file queue once the run is fully closed (GUI thread)."""
        queue = self._file_queue
        if queue is None:
            return
        if not queue["got_finished"]:
            # No finished signal arrived (pipeline error or cancel): record
            # the iteration before moving on so the summary stays exact.
            files = queue["files"]
            index = int(queue["index"])
            path = files[index] if index < len(files) else "?"
            reason = "Cancelled" if queue["stop"] else "Pipeline error"
            queue["results"].append(
                {"file": path, "ok": False, "reason": reason})
        if queue["stop"] or int(queue["index"]) >= len(queue["files"]) - 1:
            self._finish_queue()
            return
        queue["index"] = int(queue["index"]) + 1
        self._launch_queue_iteration()

    def _finish_queue(self) -> None:
        """Publish the queue summary and close staging (GUI thread)."""
        from lynceus.nodes import _file_queue as fq

        queue = self._file_queue
        self._file_queue = None
        if queue is None:
            return
        order = {path: pos for pos, path in enumerate(queue["order"])}
        results = sorted(
            queue["results"],
            key=lambda row: order.get(row.get("file"), 2 ** 31),
        )
        summary = fq.summarize_queue(results, stopped=bool(queue["stop"]))
        all_ok = all(row.get("ok") for row in results)
        self._emit(summary, "success" if all_ok else "warning")
        chip = getattr(self, "_segment_chip", None)
        if chip is not None:
            chip.hide()
        self._hide_live_notice()
        self._sync_toolbar_state("idle")
        # Staging: pending edits accumulated across the whole queue resolve
        # once, like a single run.
        self._ask_apply_pending()

    # ------------------------------------------------------------------
    # Floating toolbar
    # ------------------------------------------------------------------

    def _install_toolbar(self) -> None:
        self._toolbar = CanvasToolbar(self)
        self._toolbar.zoom_in_requested.connect(self.zoom_in)
        self._toolbar.zoom_out_requested.connect(self.zoom_out)
        self._toolbar.reset_requested.connect(self.reset_view)
        self._toolbar.fit_requested.connect(self.fit_content)
        self._toolbar.run_requested.connect(self.run_pipeline)
        self._toolbar.cancel_requested.connect(self.cancel_pipeline)
        self._toolbar.pause_requested.connect(self.pause_pipeline)
        self._toolbar.resume_requested.connect(self.resume_pipeline)
        self._toolbar.snap_toggled.connect(self.set_snap_enabled)
        self.zoomChanged.connect(
            lambda scale: self._toolbar.set_zoom_percent(round(scale * 100))
        )
        self._position_toolbar()

    def _position_toolbar(self) -> None:
        if hasattr(self, "_toolbar"):
            self._toolbar.move(TOOLBAR_MARGIN, TOOLBAR_MARGIN)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._position_toolbar()
        self._position_status_chip()
        self._position_notice_tray()

    # ------------------------------------------------------------------
    # Zoom / view
    # ------------------------------------------------------------------

    def zoom_in(self) -> None:
        self._zoom_by(ZOOM_STEP)

    def zoom_out(self) -> None:
        self._zoom_by(1 / ZOOM_STEP)

    def _zoom_by(self, factor: float) -> None:
        new_scale = self.transform().m11() * factor
        if MIN_ZOOM <= new_scale <= MAX_ZOOM:
            self.scale(factor, factor)
            self.zoomChanged.emit(self.transform().m11())

    def fit_content(self) -> None:
        if not self._node_items:
            self.reset_view()
            return
        bounds = self._scene.itemsBoundingRect().adjusted(-40, -40, 40, 40)
        self.fitInView(bounds, Qt.AspectRatioMode.KeepAspectRatio)
        self.zoomChanged.emit(self.transform().m11())

    def _install_zoom_shortcuts(self) -> None:
        QShortcut(QKeySequence("Ctrl+="), self, activated=self.zoom_in)
        QShortcut(QKeySequence("Ctrl+-"), self, activated=self.zoom_out)
        QShortcut(QKeySequence("Ctrl+G"), self, activated=self.toggle_snap)

    def _install_delete_shortcuts(self) -> None:
        # Window-scoped: deleting works wherever the focus is (library,
        # inspector, viewport), unlike the old view-only keyPressEvent.
        QShortcut(
            QKeySequence(Qt.Key.Key_Delete), self, activated=self.delete_selected
        )
        QShortcut(
            QKeySequence(Qt.Key.Key_Backspace),
            self,
            activated=self.delete_selected,
        )

    def delete_selected(self) -> None:
        """Delete selected nodes (with their cables) and loose selected cables.

        Skipped while editing text (rename editor, inspector fields, combo
        boxes, spin boxes), where BKSP/DEL keep their editing meaning.
        """
        if self._focus_is_editing():
            return
        # One undo step for the whole keypress: removed cables and nodes
        # belong to the same gesture.
        self._op_kind = "delete"
        try:
            for connection in list(self._connections):
                if connection.path_item.isSelected():
                    self._disconnect(connection)
            self._syncing_selection = True
            try:
                for item in list(self._node_items):
                    if item.isSelected():
                        self._remove_node(item)
            finally:
                self._syncing_selection = False
        finally:
            self._op_kind = None
        # The inspector follows the single-selection rule after bulk deletes.
        self._sync_inspector_to_scene()

    @staticmethod
    def _focus_is_editing() -> bool:
        """True when keyboard focus sits in a text-editing widget."""
        focus = QApplication.focusWidget()
        return isinstance(focus, (QLineEdit, QTextEdit, QComboBox, QAbstractSpinBox))

    # ------------------------------------------------------------------
    # Selection and clipboard (item 8 Fase A)
    # ------------------------------------------------------------------

    def _install_clipboard_shortcuts(self) -> None:
        """Window-scoped copy/cut/paste/duplicate/select-all shortcuts.

        Like the delete shortcuts, they work wherever the focus is; the
        handlers skip while a text field is being edited so typing keeps
        its normal clipboard meaning there.
        """
        for seq, fn in (
            (QKeySequence.StandardKey.Copy, self._shortcut_copy),
            (QKeySequence.StandardKey.Cut, self._shortcut_cut),
            (QKeySequence.StandardKey.Paste, self._shortcut_paste),
            (QKeySequence("Ctrl+D"), self._shortcut_duplicate),
            (QKeySequence.StandardKey.SelectAll, self._shortcut_select_all),
        ):
            QShortcut(seq, self, activated=fn)

    def _shortcut_copy(self) -> None:
        if not self._focus_is_editing():
            self.copy_selection()

    def _shortcut_cut(self) -> None:
        if not self._focus_is_editing():
            self.cut_selection()

    def _shortcut_paste(self) -> None:
        if not self._focus_is_editing():
            self.paste()

    def _shortcut_duplicate(self) -> None:
        if not self._focus_is_editing():
            self.duplicate()

    def _shortcut_select_all(self) -> None:
        if not self._focus_is_editing():
            self.select_all()

    # ------------------------------------------------------------------
    # History (item 9: undo/redo)
    # ------------------------------------------------------------------

    def _install_history_shortcuts(self) -> None:
        """Window-scoped undo/redo shortcuts (single binding, text-safe)."""
        for seq, fn in (
            (QKeySequence.StandardKey.Undo, self._shortcut_undo),
            (QKeySequence.StandardKey.Redo, self._shortcut_redo),
            (QKeySequence("Ctrl+Shift+Z"), self._shortcut_redo),
        ):
            QShortcut(seq, self, activated=fn)

    def _shortcut_undo(self) -> None:
        if not self._focus_is_editing():
            self.undo()

    def _shortcut_redo(self) -> None:
        if not self._focus_is_editing():
            self.redo()

    def _restore_history(self, doc: dict) -> None:
        """Rebuild the graph from a history snapshot without recording.

        Node items are recreated by ``restore_graph``; the selection is
        preserved by instance id so undo/redo keeps the Inspector context.
        """
        selected = {it.iid for it in self.selected_nodes()}
        self._undo_active = True
        try:
            self.restore_graph(doc)
        finally:
            self._undo_active = False
        if selected:
            self._syncing_selection = True
            try:
                for item in self._node_items:
                    item.set_selected(item.iid in selected)
            finally:
                self._syncing_selection = False
            self._sync_inspector_to_scene()

    def _graph_mutated(self, kind: str) -> None:
        """Record the mutation and emit the change signal (single funnel)."""
        self._record_current_state(kind)
        self.graph_changed.emit()

    def _record_current_state(self, kind: str) -> None:
        """Fold the live graph into the history (post-mutation).

        Called on every real canvas mutation (the graph_changed funnels).
        Continuous gestures (drag, typing, spin boxes) keep replacing the
        current slot while they reuse the same ``kind`` inside
        ``UNDO_MERGE_WINDOW``, so they collapse into one undo step; a later
        mutation truncates the redo future.
        """
        if self._undo_active or self._restoring or self._run_active:
            return
        kind = self._op_kind or kind
        now = time.monotonic()
        if (
            self._last_kind == kind
            and self._merge_since is not None
            and now - self._merge_since <= UNDO_MERGE_WINDOW
            and self._cursor == len(self._history) - 1
        ):
            # The current gesture is still underway: refresh the window clock
            # and replace the current slot so undo lands on the final pose.
            if self._history:
                doc = self.document_snapshot()
                if doc != self._history[self._cursor]:
                    self._history[self._cursor] = doc
            self._merge_since = now
            return
        del self._history[self._cursor + 1 :]
        self._history.append(self.document_snapshot())
        if len(self._history) > UNDO_LIMIT + 1:
            del self._history[0]
        self._cursor = len(self._history) - 1
        self._last_kind = kind
        self._merge_since = now
        self._emit_history_state()

    def undo(self) -> None:
        """Step back one committed state (no-op during a run)."""
        if self._run_active or self._undo_active or self._cursor <= 0:
            return
        self._cursor -= 1
        self._restore_history(self._history[self._cursor])
        self._merge_since = None
        self._last_kind = None
        self._emit_history_state()
        self.history_applied.emit()

    def redo(self) -> None:
        """Re-apply the most recently undone state (no-op during a run)."""
        if self._run_active or self._undo_active or self._cursor >= len(self._history) - 1:
            return
        self._cursor += 1
        self._restore_history(self._history[self._cursor])
        self._merge_since = None
        self._last_kind = None
        self._emit_history_state()
        self.history_applied.emit()

    def can_undo(self) -> bool:
        return not self._run_active and self._cursor > 0

    def can_redo(self) -> bool:
        return not self._run_active and self._cursor < len(self._history) - 1

    def _emit_history_state(self) -> None:
        self.history_changed.emit(self.can_undo(), self.can_redo())

    def clear_history(self) -> None:
        """Reset history to the live graph (open/new/template boundaries)."""
        self._history = [self.document_snapshot()]
        self._cursor = 0
        self._merge_since = None
        self._last_kind = None
        self._emit_history_state()

    def document_snapshot(self) -> dict:
        """Project document without view state (history + dirty comparison)."""
        return {
            k: v for k, v in self.serialize_graph().items() if k != "view"
        }

    def select_all(self) -> None:
        """Select every node on the canvas."""
        if not self._node_items:
            return
        self._syncing_selection = True
        try:
            for item in self._node_items:
                item.set_selected(True)
        finally:
            self._syncing_selection = False
        # The funnel updates state atomically (multi -> panel).
        self._sync_inspector_to_scene()

    def copy_selection(self) -> None:
        """Write the selected subgraph to the system clipboard."""
        selected = self.selected_nodes()
        if not selected:
            return
        doc = self.serialize_graph()
        iids = [item.iid for item in selected]
        payload = extract_subgraph(doc["nodes"], doc["edges"], set(iids))
        QApplication.clipboard().setText(encode_subgraph(payload))

    def cut_selection(self) -> None:
        """Copy the selection to the clipboard and delete it."""
        if not self.selected_nodes():
            return
        self.copy_selection()
        self.delete_selected()

    def _clipboard_subgraph(self) -> dict | None:
        return decode_subgraph(QApplication.clipboard().text())

    def paste(self, offset: QPointF | None = None) -> list[NodeItem] | None:
        """Paste a subgraph from the clipboard (in place, cascade via API).

        New instance ids keep pasted copies isolated; internal edges are
        re-created through ``_create_connection`` so compatibility and
        exclusivity rules hold. External wires never cross the paste.
        """
        payload = self._clipboard_subgraph()
        if payload is None or not payload["nodes"]:
            self.message_requested.emit(
                t("Clipboard does not contain a node graph"), "warning"
            )
            return None
        new_items, _remap = self._materialize_subgraph(
            payload["nodes"], payload["edges"], offset or QPointF(24, 24)
        )
        return new_items

    def duplicate(self, rewire: bool = False) -> list[NodeItem] | None:
        """Clone the current selection next to it.

        ``rewire=True`` ("duplicate and rewire") also re-creates boundary
        edges: input copies keep their providers, output copies feed every
        free same-type input slot of the original destinations.
        """
        selected = self.selected_nodes()
        if not selected:
            return None
        doc = self.serialize_graph()
        iids = {item.iid for item in selected}
        payload = extract_subgraph(doc["nodes"], doc["edges"], iids)
        new_items, remap = self._materialize_subgraph(
            payload["nodes"], payload["edges"], QPointF(24, 24)
        )
        if rewire:
            self._rewire_boundary(payload["boundary"], remap)
        return new_items

    def _materialize_subgraph(
        self, nodes_doc: list, edges_doc: list, offset: QPointF,
    ) -> tuple[list[NodeItem], dict[str, NodeItem]]:
        """Instantiate a subgraph with fresh ids (paste/duplicate core).

        Guarded by ``_restoring`` so the Inspector does not churn per node;
        the pasted items are selected as a unit afterwards. Returns the new
        items plus the ``{old_iid: new_item}`` remap used by rewiring.
        """
        old_to_new: dict[str, NodeItem] = {}
        new_items: list[NodeItem] = []
        if not nodes_doc:
            return new_items, old_to_new
        self._restoring = True
        try:
            for entry in nodes_doc:
                node_id = entry.get("id")
                if not isinstance(node_id, str):
                    continue
                try:
                    item = self.add_node(node_id)
                except Exception:
                    continue
                old_iid = entry.get("iid")
                item.iid = uuid.uuid4().hex[:8]
                try:
                    item.setPos(
                        self.snap_point(
                            QPointF(float(entry.get("x", 0)),
                                    float(entry.get("y", 0))) + offset
                        )
                    )
                except (TypeError, ValueError):
                    pass
                label = entry.get("label")
                if isinstance(label, str) and label:
                    item.set_label(label)
                config = entry.get("config")
                if isinstance(config, dict):
                    item.apply_config(config)
                if isinstance(old_iid, str):
                    old_to_new[old_iid] = item
                new_items.append(item)
            for edge in edges_doc:
                self._create_doc_edge(edge, old_to_new)
        finally:
            self._restoring = False
        if new_items:
            self._syncing_selection = True
            try:
                for item in self._node_items:
                    item.set_selected(item in new_items)
            finally:
                self._syncing_selection = False
            self._sync_inspector_to_scene()
        self._graph_mutated("paste")
        return new_items, old_to_new

    def _create_doc_edge(self, edge: dict, old_to_new: dict) -> None:
        """Re-create one internal edge across a remapped node set."""
        src = old_to_new.get(edge.get("src"))
        dst = old_to_new.get(edge.get("dst"))
        if src is None or dst is None:
            return
        outs, ins = src.outputs(), dst.inputs()
        out_i, in_i = edge.get("src_out"), edge.get("dst_in")
        if not isinstance(out_i, int) or not isinstance(in_i, int):
            return
        if not (0 <= out_i < len(outs) and 0 <= in_i < len(ins)):
            return
        if not outs[out_i].can_connect_to(ins[in_i]) or ins[in_i].is_full():
            return
        self._create_connection(outs[out_i], ins[in_i])

    def _rewire_boundary(self, boundary: list, old_to_new: dict) -> None:
        """Re-create boundary edges of a duplicated selection.

        Inbound (external provider -> copy) targets the same input index on
        the fresh copy; outbound (copy -> external consumer) lands on the
        first FREE input of the same type on the original destination
        ("siguiente slot libre"); incompatible or full slots are skipped.
        """
        live = {item.iid: item for item in self._node_items}
        for edge in boundary:
            out_i = edge.get("src_out")
            in_i = edge.get("dst_in")
            if not isinstance(out_i, int) or not isinstance(in_i, int):
                continue
            if edge.get("src_selected") and edge.get("dst_selected"):
                continue  # not a boundary edge
            if not edge.get("src_selected") and not edge.get("dst_selected"):
                continue  # external edge (both endpoints outside)
            if edge.get("dst_selected"):
                # Inbound: the original external provider -> the fresh copy.
                src_item = live.get(edge.get("src"))
                dst_item = old_to_new.get(edge.get("dst"))
                if src_item is None or dst_item is None:
                    continue
                outs, ins = src_item.outputs(), dst_item.inputs()
                if not (0 <= out_i < len(outs)) or not (0 <= in_i < len(ins)):
                    continue
                if not ins[in_i].is_full() and outs[out_i].can_connect_to(ins[in_i]):
                    self._create_connection(outs[out_i], ins[in_i])
                continue
            # Outbound: the fresh copy -> the original consumer's first free
            # input of the same type ("next free slot").
            src_item = old_to_new.get(edge.get("src"))
            dst_item = live.get(edge.get("dst"))
            if src_item is None or dst_item is None:
                continue
            outs = src_item.outputs()
            if not (0 <= out_i < len(outs)):
                continue
            in_type = edge.get("in_type")
            for port in dst_item.inputs():
                if port.port_type_id != in_type or port.is_full():
                    continue
                if port.group_conflict() is not None:
                    continue
                if outs[out_i].can_connect_to(port):
                    self._create_connection(outs[out_i], port)
                    break

    def _queueable_selected(self) -> list:
        """Selected items that accept a file queue (selection order)."""
        return [
            item for item in self.selected_nodes()
            if getattr(item, "supports_file_queue", False)
        ]

    def _on_fleet_action(self, item: NodeItem, action: str) -> None:
        """Node context-menu fleet actions: the whole selection when the
        clicked item belongs to it, otherwise just the clicked item."""
        selected = self.selected_nodes()
        targets = selected if item in selected else [item]
        if action == "delete":
            self._syncing_selection = True
            try:
                for target in list(targets):
                    self._remove_node(target)
            finally:
                self._syncing_selection = False
            self._sync_inspector_to_scene()
        elif action == "duplicate":
            self._sync_selection_to(targets)
            self.duplicate()
        elif action == "duplicate_rewire":
            self._sync_selection_to(targets)
            self.duplicate(rewire=True)
        elif action == "assign_files":
            self._sync_selection_to(targets)
            self.assign_files_to_selection()
        elif action == "set_strategy":
            self._batch_set_strategy("")
        elif action == "paste":
            self.paste()
        elif action == "select_all":
            self.select_all()

    def _sync_selection_to(self, targets: list) -> None:
        """Make exactly ``targets`` selected (used before fleet operations)."""
        self._syncing_selection = True
        try:
            for other in self._node_items:
                other.set_selected(other in targets)
        finally:
            self._syncing_selection = False

    def _batch_set_strategy(self, key: str) -> None:
        """Apply one strategy to every selected strategy node (homogeneous)."""
        for item in self.selected_nodes():
            if isinstance(item, StrategyNodeItem) and key != item.strategy():
                self.set_strategy(item, key)

    def assign_files_to_selection(self) -> None:
        """Assign matched folder files to the queueable selection."""
        items = self._queueable_selected()
        if not items:
            return
        from lynceus.ui.nodes.widgets.assign_files_dialog import AssignFilesDialog

        AssignFilesDialog(self, items).exec()

    def contextMenuEvent(self, event) -> None:
        """Empty-canvas right-click menu (items own their own menu)."""
        super().contextMenuEvent(event)
        if event.isAccepted():
            return
        target = self.itemAt(event.pos())
        while target is not None and not isinstance(target, NodeItem):
            target = target.parentItem()
        if target is not None:
            return
        self._show_canvas_menu(event.globalPos())

    def _show_canvas_menu(self, global_pos) -> None:
        """Fleet menu for the empty canvas: select-all, paste, and the
        selection-dependent batch actions (available after a rubber band)."""
        menu = QMenu(self)
        select_all = menu.addAction(t("Select All"))
        select_all.triggered.connect(self.select_all)
        paste = menu.addAction(t("Paste"))
        paste.setShortcut(QKeySequence.StandardKey.Paste)
        paste.triggered.connect(lambda: self.paste())
        paste.setEnabled(self._clipboard_subgraph() is not None)
        selected = self.selected_nodes()
        if selected:
            menu.addSeparator()
            duplicate = menu.addAction(t("Duplicate"))
            duplicate.setShortcut(QKeySequence("Ctrl+D"))
            duplicate.triggered.connect(lambda: self.duplicate())
            rewire = menu.addAction(t("Duplicate and reconnect"))
            rewire.triggered.connect(lambda: self.duplicate(rewire=True))
            strategy = self._strategy_selection_menu(menu)
            if strategy is not None:
                menu.addSeparator()
                menu.addMenu(strategy)
            if self._queueable_selected():
                assign = menu.addAction(t("Assign files..."))
                assign.triggered.connect(self.assign_files_to_selection)
            menu.addSeparator()
            delete = menu.addAction(t("Delete"))
            delete.setShortcut(QKeySequence.StandardKey.Delete)
            delete.triggered.connect(self.delete_selected)
        menu.exec(global_pos)

    def _strategy_selection_menu(self, parent) -> None:
        """Build a "Set strategy..." submenu over the strategy families in
        the selection; None when the selection has no strategy nodes."""
        families: dict[str, list] = {}
        for item in self.selected_nodes():
            if isinstance(item, StrategyNodeItem):
                families.setdefault(item.node_id(), []).append(item)
        if not families:
            return None
        from PySide6.QtWidgets import QMenu

        top = QMenu(t("Set strategy..."), parent)
        for node_id, items in sorted(families.items()):
            if len(families) > 1:
                sub = top.addMenu(self._node_name(node_id))
            else:
                sub = top
            for key, label in strategy_options(node_id):
                action = sub.addAction(label)
                action.setCheckable(True)
                action.setChecked(
                    all(item.strategy() == key for item in items)
                )
                action.triggered.connect(
                    lambda _checked=False, k=key: self._batch_set_strategy(k)
                )
            none_action = sub.addAction(t("(none)"))
            none_action.setCheckable(True)
            none_action.setChecked(
                all(item.strategy() == "" for item in items)
            )
            none_action.triggered.connect(lambda: self._batch_set_strategy(""))
        return top

    # ------------------------------------------------------------------
    # Background grid and snap
    # ------------------------------------------------------------------

    def drawBackground(self, painter, rect) -> None:
        painter.fillRect(rect, CANVAS_BG)
        # Whisper of depth: radial falloff toward the exposed edges. One
        # fill, derived live from the background so themes follow. The
        # radius overshoots the view so no circle edge is ever visible.
        # Frame of reference is the viewport (stable across repaints):
        # centering on the exposed rect shades every partial repaint
        # differently, leaving dirty rectangles behind (rubber band on
        # light themes).
        vp = self.mapToScene(self.viewport().rect()).boundingRect()
        center = vp.center()
        radius = max(vp.width(), vp.height()) * 1.2
        if radius > 0:
            vignette = QRadialGradient(center, radius)
            vignette.setColorAt(0.0, CANVAS_BG)
            vignette.setColorAt(1.0, QColor(CANVAS_BG).darker(106))
            painter.fillRect(rect, QBrush(vignette))
        painter.save()
        # Keep grid dots sharp by disabling antialiasing.
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        self._draw_origin_axes(painter, rect)

        scale = self.transform().m11()
        major_step = GRID_MINOR * GRID_MAJOR_EVERY
        self._draw_dot_level(
            painter, rect, GRID_MINOR, DOT_MINOR_COLOR,
            GRID_MINOR * scale, MINOR_FADE_IN_PX, MINOR_FADE_OUT_PX,
        )
        self._draw_dot_level(
            painter, rect, major_step, DOT_MAJOR_COLOR,
            major_step * scale, MAJOR_FADE_IN_PX, MAJOR_FADE_OUT_PX,
        )
        painter.restore()

    def _draw_origin_axes(self, painter: QPainter, rect) -> None:
        """Draw subtle axes through the origin for orientation."""
        painter.setPen(QPen(ORIGIN_COLOR, 0))
        if rect.left() <= 0 <= rect.right():
            painter.drawLine(QPointF(0, rect.top()), QPointF(0, rect.bottom()))
        if rect.top() <= 0 <= rect.bottom():
            painter.drawLine(QPointF(rect.left(), 0), QPointF(rect.right(), 0))

    @staticmethod
    def _grid_fade(screen_spacing: float, fade_in: float, fade_out: float) -> float:
        """1.0 = fully visible level; 0.0 = hidden (smoothstep)."""
        if screen_spacing >= fade_in:
            return 1.0
        if screen_spacing <= fade_out:
            return 0.0
        t = (screen_spacing - fade_out) / (fade_in - fade_out)
        return t * t * (3.0 - 2.0 * t)

    def _draw_dot_level(
        self,
        painter: QPainter,
        rect,
        step: int,
        color: QColor,
        screen_spacing: float,
        fade_in: float,
        fade_out: float,
    ) -> None:
        factor = self._grid_fade(screen_spacing, fade_in, fade_out)
        if factor <= 0.0:
            return
        left = math.floor(rect.left() / step) * step
        top = math.floor(rect.top() / step) * step
        cols = math.floor((rect.right() - left) / step) + 1
        rows = math.floor((rect.bottom() - top) / step) + 1
        if cols <= 0 or rows <= 0 or cols * rows > MAX_DOTS_PER_LEVEL:
            return
        if factor < 1.0:
            color = QColor(
                round(CANVAS_BG.red() + (color.red() - CANVAS_BG.red()) * factor),
                round(CANVAS_BG.green() + (color.green() - CANVAS_BG.green()) * factor),
                round(CANVAS_BG.blue() + (color.blue() - CANVAS_BG.blue()) * factor),
            )
        points = QPolygonF()
        for col in range(cols):
            x = left + col * step
            for row in range(rows):
                points.append(QPointF(x, top + row * step))
        # Cosmetic zero-width pen: one screen pixel.
        painter.setPen(QPen(color, 0))
        painter.drawPoints(points)

    def set_snap_enabled(self, enabled: bool) -> None:
        self._snap_enabled = bool(enabled)

    def snap_enabled(self) -> bool:
        return self._snap_enabled

    def snap_point(self, pos: QPointF) -> QPointF:
        """Round a scene position to the grid when snapping is enabled."""
        if not self._snap_enabled:
            return pos
        return QPointF(
            round(pos.x() / SNAP_STEP) * SNAP_STEP,
            round(pos.y() / SNAP_STEP) * SNAP_STEP,
        )

    def toggle_snap(self) -> None:
        self._toolbar.btn_snap.toggle()

    # ------------------------------------------------------------------
    # Pipeline (delegated to PipelineController)
    # ------------------------------------------------------------------

    def _typed_edges(self) -> list:
        """Return typed instance edges for the pipeline engine."""
        return [
            (
                c.source.node.iid,
                c.target.node.iid,
                c.source.port_type,
                c.target.port_type,
            )
            for c in self._connections
        ]

    def _pipeline_nodes(self) -> list:
        """Return ``(iid, module_id)`` pairs participating in the run.

        Wired nodes and source nodes with no inputs are included. Input-bearing
        orphan nodes are drafts and are excluded; an isolated loader still
        creates its tile session. Unavailable nodes (missing or disabled
        extension) never participate. Orphan file-inputs with nothing usable
        selected (no file, or a vanished one) are ignored instead of failing
        the run on "file not found".
        """
        wired: set[str] = set()
        for edge in self._typed_edges():
            wired.add(edge[0])
            wired.add(edge[1])
        sources: set[str] = set()
        for item in self._node_items:
            if item.iid in wired:
                continue
            if not item.is_available():
                continue
            # Live ports: strategy items resolve their variant's ports before
            # this snapshot (the engine does the same per instance).
            if not item.inputs():
                if self._orphan_input_empty(item):
                    continue
                sources.add(item.iid)
        return [
            (item.iid, item.node_id())
            for item in self._node_items
            if item.iid in wired or item.iid in sources
        ]

    @staticmethod
    def _orphan_input_empty(item) -> bool:
        """True when an unwired file-input node has nothing usable selected."""
        file_path = getattr(item, "file_path", None)
        if not callable(file_path):
            return False
        try:
            path = file_path()
        except Exception:
            return False
        if not path:
            return True
        file_missing = getattr(item, "file_missing", None)
        if not callable(file_missing):
            return False
        try:
            return bool(file_missing())
        except Exception:
            return False

    def _pipeline_configs(self) -> dict:
        """Per-instance configuration edited from the Inspector."""
        return {
            item.iid: item.pipeline_config()
            for item in self._node_items
            if item.pipeline_config()
        }

    def _on_status(self, text: str, kind: str = "info") -> None:
        """Pipeline message adapter: assigns a default kind to shield
        against emitters that only pass text (never breaks the signal)."""
        self.message_requested.emit(text, kind)

    def _pipeline_callbacks(self) -> PipelineCallbacks:
        return PipelineCallbacks(
            on_message=self._on_status,
            on_node_status=self._on_worker_node_status,
            on_tile_count=self.load_tile_count.emit,
            on_finished=self.pipeline_finished.emit,
            on_progress=self.pipeline_progress.emit,
            on_cancelled=self.pipeline_cancelled.emit,
            on_segment_started=self.segment_started.emit,
            on_segment_outputs=self.segment_outputs_changed.emit,
            on_consolidated_outputs=self.consolidated_outputs_ready.emit,
            on_run_closed=self.pipeline_closed.emit,
            on_barrier_outputs=self.barrier_outputs_changed.emit,
            on_barrier_started=self.barrier_started.emit,
        )

    def _on_worker_node_status(self, iid: str, status: str) -> None:
        # After a cancel/finish a queued "processing" event could still land;
        # the run is over, so late statuses must not re-pulse the nodes.
        if not self._run_active:
            return
        item = self._item_by_iid.get(iid)
        if item is not None:
            self.node_status_changed.emit(item, status)

    def _run_preflight(
        self, nodes, source_paths: dict, queued_files: list[str] | None,
    ):
        """Header-only run sizing (item 44): overrides dict, or None to abort.

        Best-effort and never a gate: unreadable sources just yield no
        advice. Shown once per run/queue; the operator can suppress it
        permanently or jump to the Performance preferences.
        """
        from lynceus.processing.preflight import build_preflight
        from lynceus.processing.steps import discover_node_capabilities

        if _setting_bool(PREFLIGHT_QUIET_KEY, False):
            return {}
        source_list = (
            list(queued_files) if queued_files
            else [p for p in source_paths.values() if p]
        )
        if not source_list:
            return {}
        prefs = _run_preferences()
        factors = []
        configs = self._pipeline_configs()
        merge_on = False
        has_mosaic = False
        loader_iids: list[str] = []
        for iid, _ in nodes:
            item = self._item_by_iid.get(iid)
            nid = item.node_id() if item is not None else ""
            if not nid:
                continue
            caps = discover_node_capabilities(nid)
            if caps.get("tile_task") is not None:
                factors.append(caps.get("tile_weight_factor", 1.0))
            if caps.get("triggers_tiling"):
                loader_iids.append(iid)
                if _config_int(configs, iid, "merge_sparse_below") > 0:
                    merge_on = True
            outputs = caps.get("output_files") or {}
            if any(
                str(name).lower().endswith((".tif", ".tiff"))
                for name in outputs.values()
            ):
                has_mosaic = True
        seg_cap = 0
        seg_independent = False
        if loader_iids:
            first_cfg = configs.get(loader_iids[0]) or {}
            seg_cap = _config_int(configs, loader_iids[0], "max_points_per_part")
            # Schema default merges (True); only an explicit False counts
            # as independent mode (junk stays on the safe default).
            if "merge_segments" in first_cfg:
                seg_independent = not first_cfg["merge_segments"]
        advice = build_preflight(
            source_list, factors,
            workers=prefs["max_workers"],
            budget_bytes=prefs["memory_budget_bytes"],
            cache_enabled=prefs["keep_intermediates"],
            merge_enabled=merge_on,
            has_mosaic=has_mosaic,
            total_ram_bytes=_total_ram_bytes(),
            seg_cap=seg_cap,
            segment_independent=seg_independent,
        )
        if advice is None or not advice.has_advice:
            return {}
        from lynceus.ui.preflight_dialog import (
            OPEN_PERFORMANCE,
            SHOW_LOADER,
            PreflightDialog,
        )

        dialog = PreflightDialog(self, advice)
        code = dialog.exec()
        if dialog.dont_show_again():
            QSettings(SETTINGS_ORG, SETTINGS_APP).setValue(
                PREFLIGHT_QUIET_KEY, True
            )
        if code == OPEN_PERFORMANCE:
            self.open_preferences_requested.emit("Performance")
            return None
        if code == SHOW_LOADER:
            self._focus_loader(nodes)
            return None
        if code != QDialog.DialogCode.Accepted:
            return None
        if dialog.apply_merge_floor() and advice.merge_floor_suggest > 0:
            self._apply_merge_floor(nodes, advice.merge_floor_suggest)
        if dialog.apply_segments() > 0:
            self._apply_segments(nodes, dialog.apply_segments())
        overrides: dict = {}
        if dialog.enable_cache() and not prefs["keep_intermediates"]:
            overrides["keep_intermediates"] = True
        if dialog.apply_workers() > 0:
            overrides["max_workers"] = dialog.apply_workers()
        if dialog.apply_memory() > 0:
            overrides["memory_budget_bytes"] = dialog.apply_memory()
        if dialog.apply_resolution() > 0:
            self._apply_node_config(
                nodes,
                "resolution",
                dialog.apply_resolution(),
                lambda caps: isinstance(caps.get("config_schema"), dict)
                and "resolution" in caps["config_schema"],
            )
        return overrides

    def _merge_enabled(self, nodes) -> bool:
        """Whether any loader in scope already fuses sparse tiles."""
        from lynceus.processing.steps import discover_node_capabilities

        configs = self._pipeline_configs()
        for iid, _ in nodes:
            item = self._item_by_iid.get(iid)
            nid = item.node_id() if item is not None else ""
            if not nid:
                continue
            if not discover_node_capabilities(nid).get("triggers_tiling"):
                continue
            if _config_int(configs, iid, "merge_sparse_below") > 0:
                return True
        return False

    def _focus_loader(self, nodes) -> None:
        """Select the first loader so its Tiling settings show in context."""
        from lynceus.processing.steps import discover_node_capabilities

        for iid, _ in nodes:
            item = self._item_by_iid.get(iid)
            if item is None:
                continue
            try:
                caps = discover_node_capabilities(item.node_id())
            except Exception:
                continue
            if caps.get("triggers_tiling"):
                self._sync_selection_to([item])
                self.ensureVisible(item)
                return

    def _apply_node_config(self, nodes, key: str, value, requires) -> int:
        """Write one config key into matching nodes (project state).

        Same funnel the Inspector uses per edit (stale mark + dirty + undo
        history + refresh when selected), so a programmatic apply is
        indistinguishable from typing the value by hand. ``requires`` takes
        node capabilities; junk current values are overwritten, identical
        ones skipped without dirtying. Returns the applied count.
        """
        from lynceus.processing.steps import discover_node_capabilities

        applied = 0
        for iid, _ in nodes:
            item = self._item_by_iid.get(iid)
            if item is None:
                continue
            try:
                caps = discover_node_capabilities(item.node_id())
            except Exception:
                continue
            if not requires(caps):
                continue
            config = item.pipeline_config()
            if config.get(key) == value:
                continue
            config[key] = value
            item.set_pipeline_config(config)
            self.on_config_changed(iid)
            if self._selected_item is item:
                self.inspector_refresh_requested.emit()
            applied += 1
        return applied

    def _apply_merge_floor(self, nodes, floor: int) -> None:
        """Write the suggested floor into loader configs (project state).

        Same funnel the Inspector uses per edit (stale mark + dirty + undo
        history + refresh when selected), so a one-click apply is
        indistinguishable from typing the value by hand: fingerprints,
        manifest and saved projects all carry it. No phantom per-run
        overrides.
        """
        try:
            floor_value = int(floor)
        except (TypeError, ValueError):
            return
        if floor_value <= 0:
            return
        self._apply_node_config(
            nodes,
            "merge_sparse_below",
            floor_value,
            lambda caps: bool(caps.get("triggers_tiling")),
        )

    def _apply_segments(self, nodes, cap: int) -> None:
        """Write the suggested batch cap + independent mode into loaders.

        Same Inspector funnel as every programmatic apply (dirty + undo +
        refresh). Both keys travel in fingerprints, so reuse stays correct.
        """
        try:
            cap_value = int(cap)
        except (TypeError, ValueError):
            return
        if cap_value <= 0:
            return
        self._apply_node_config(
            nodes, "max_points_per_part", cap_value, _is_loader_caps)
        self._apply_node_config(nodes, "merge_segments", False, _is_loader_caps)

    def run_pipeline(self) -> None:
        if self._run_active:
            self._emit(
                t(
                    "A run is already in progress; cancel it or wait for it "
                    "to finish"
                ),
                "warning",
            )
            return

        if not self._node_items:
            self._emit(t("Add at least one node to run a pipeline"))
            return

        unavailable = [
            item.display_name()
            for item in self._node_items
            if not item.is_available()
        ]
        if unavailable:
            self._emit(
                t(
                    "Node(s) use a missing or disabled extension: {names} — "
                    "enable it in File → Extensions, re-import it, or remove "
                    "the node(s) before running"
                ).format(names=", ".join(sorted(unavailable))),
                "warning",
            )
            return

        wired_iids = {
            iid for edge in self._typed_edges() for iid in (edge[0], edge[1])
        }
        # Only wired strategy nodes block the run: an orphan without strategy
        # is an empty draft and is ignored by _pipeline_nodes.
        unselected = [
            item.display_name()
            for item in self._node_items
            if item.iid in wired_iids
            and isinstance(item, StrategyNodeItem)
            and not item.strategy()
        ]
        if unselected:
            self._emit(
                t(
                    "Select a product strategy in node(s): {names} — choose "
                    "the strategy with the selector on the node or "
                    "in the Inspector before running"
                ).format(names=", ".join(sorted(unselected))),
                "warning",
            )
            return

        for item in self._node_items:
            item.set_status(STATUS_IDLE)
        self._item_by_iid = {item.iid: item for item in self._node_items}

        nodes = self._pipeline_nodes()
        if not nodes:
            # Nodes exist but none participate (all orphans without
            # connected inputs that are not sources): no pipeline to build.
            self._emit(
                t(
                    "No connected pipeline to run — connect nodes or add a "
                    "source"
                )
            )
            return
        in_scope = {iid for iid, _ in nodes}
        # File queue: exactly one in-scope input with several queued files
        # chains one run per file; more than one queued input disables the
        # queue with a notice (single classic run on each head file).
        queued_items = [
            item for item in self._node_items
            if item.iid in in_scope
            and getattr(item, "supports_file_queue", False)
            and len(item.file_queue()) > 1
        ]
        queue_item = None
        queue_first: str | None = None
        queue_existing: list[str] = []
        if len(queued_items) > 1:
            self._emit(
                t(
                    "File queue disabled: {names} have queued files — "
                    "run one queue at a time"
                ).format(names=", ".join(sorted(
                    item.display_name() for item in queued_items))),
                "warning",
            )
        elif len(queued_items) == 1:
            from lynceus.nodes import _file_queue as fq

            existing, _missing = fq.partition_existing(
                queued_items[0].file_queue())
            if existing:
                queue_item = queued_items[0]
                queue_first = existing[0]
                queue_existing = existing
            # All queued files vanished: fall through to the classic single
            # run, which reports the vanished head file as usual.
        load_items = [
            item
            for item in self._node_items
            if item.iid in in_scope
            and item.node_id() == "lynceus.nodes.lidar.source.load_las_laz"
        ]
        source_paths: dict[str, str] = {}
        for item in load_items:
            if queue_item is not None and item.iid == queue_item.iid:
                # Prefiltered at detection: the queue head for this run.
                fp = queue_first
            else:
                fp = item.file_path()
            if not fp:
                if item.iid not in wired_iids:
                    # Unreachable: file-less orphan loaders never enter the
                    # scope; stay silent instead of failing the run.
                    continue
                self._emit(
                    t("Set a LiDAR file in every Load node")
                )
                return
            if not os.path.isfile(fp):
                if item.iid not in wired_iids:
                    continue
                self._emit(t("Source file not found: {fp}").format(fp=fp))
                return
            source_paths[item.iid] = fp
        # source_paths == {} => grafo barreras-only (Input Raster/Table/GPKG, ...)
        # (orphan loaders without a file are ignored by _pipeline_nodes).
        # Pre-flight advisor (item 44): header-only sizing, once per run/queue.
        # None means the operator aborted or jumped to Performance; a dict
        # carries per-run preference overrides (e.g. cache for this run).
        preflight_overrides = self._run_preflight(
            nodes, source_paths, queue_existing if queue_item else None
        )
        if preflight_overrides is None:
            self._sync_toolbar_state("idle")
            return
        # Item 43: configurable output root (Preferences > General). Mapped
        # each run so a live change applies on the next run without reopening.
        self.session_root = resolve_output_root(
            str(QSettings(SETTINGS_ORG, SETTINGS_APP).value(OUTPUT_ROOT_KEY, ""))
        )
        # Isolated snapshot of the canvas: the run executes this; any later
        # edit (nodes/cables/config) stays "pending" and does not
        # affect the session until the next run (see _ask_apply_pending).
        self._graph_snapshot = self.serialize_graph()
        self._run_active = True
        self._emit_history_state()  # undo/redo disable while the run lasts
        # Reset gallery accumulation for this run: segments publish themselves
        # incrementally; the consolidated/final products fill the last block.
        self._had_segments = False
        self._segment_products = {}
        self._flat_products = []
        self._live_products = {}
        self._batch_active = False
        self._live_notice_shown = False
        self._live_notice_text = ""
        chip = getattr(self, "_segment_chip", None)
        if chip is not None:
            chip.hide()
        self.outputs_updated.emit([])
        self.pipeline_started.emit()
        self._set_run_pending(True)
        self._sync_toolbar_state("running")
        run_prefs = _run_preferences()
        run_prefs.update(preflight_overrides or {})
        self._pipeline.set_acceleration_mode(run_prefs["acceleration_mode"])
        self._pipeline.set_acceleration_backend(
            run_prefs["acceleration_backend"], run_prefs["acceleration_device"]
        )
        if queue_item is not None:
            self._start_file_queue(
                queue_item, nodes, self._typed_edges(), source_paths,
                self._pipeline_configs(), run_prefs,
            )
            return
        self._pipeline.run(
            nodes, self._typed_edges(), source_paths,
            self._pipeline_callbacks(), session_root=self.session_root,
            configs=self._pipeline_configs(),
            max_workers=run_prefs["max_workers"],
            memory_budget_bytes=run_prefs["memory_budget_bytes"],
            operator=run_prefs["operator"],
            keep_intermediates=run_prefs["keep_intermediates"],
        )

    def cancel_pipeline(self) -> None:
        if self._file_queue is not None:
            # Stop cancels the whole queue: the current run cancels now and
            # the post-cleanup closed signal finishes with a partial summary.
            self._file_queue["stop"] = True
        self._sync_toolbar_state("cancelling")
        self.message_requested.emit(t("Cancelling pipeline…"), "action")
        self._pipeline.cancel()

    def pause_pipeline(self) -> None:
        self._pipeline.pause()
        self._sync_toolbar_state("paused")

    def resume_pipeline(self) -> None:
        self._pipeline.resume()
        self._sync_toolbar_state("running")

    def _sync_toolbar_state(self, state: str | None = None) -> None:
        """Applies the Run/Pause/Cancel state machine of the toolbar.

        If no explicit state is given, it derives it from the controller
        (source of truth). Invoked on every transition and on finish/cancel.
        """
        state = state or getattr(self._pipeline, "state", "idle")
        if hasattr(self, "_toolbar"):
            self._toolbar.set_state(state)

    def _emit(self, message: str, kind: str = "info") -> None:
        self.message_requested.emit(message, kind)

    # ------------------------------------------------------------------
    # Instance labels (inline editing over the header)
    # ------------------------------------------------------------------

    def _on_node_preview_requested(self, item: NodeItem) -> None:
        """Forward a body double-click on a file-backed node to the bridge preview."""
        preview_file = getattr(item, "preview_file", None)
        path = preview_file() if callable(preview_file) else None
        if path:
            self.node_preview_requested.emit(path, item.node_name_en())

    def _start_label_edit(self, item: NodeItem) -> None:
        """Opens a QLineEdit over the node header for its label."""
        if self._label_editor is not None:
            self._close_label_editor()
        if item not in self._node_items:
            return
        scale = max(self.transform().m11(), 0.05)
        top_left = self.mapFromScene(
            item.mapToScene(QPointF(PORT_PROTRUSION + 8, 4))
        )
        width = int(
            (item.boundingRect().width() - 2 * PORT_PROTRUSION - 44) * scale
        )
        height = int((HEADER_HEIGHT - 8) * scale)
        editor = _LabelEditor(self.viewport())
        editor.setObjectName("labelEditor")
        editor.setGeometry(
            QRect(top_left.x(), top_left.y(), max(80, width), max(18, height))
        )
        font = editor.font()
        from lynceus.ui.fonts import font_scale

        font.setPointSizeF(max(7.0, 10.0 * scale * font_scale()))
        editor.setFont(font)
        editor.setText(item.label())
        editor.selectAll()
        editor.returnPressed.connect(self._commit_label_editor)
        editor.cancelled.connect(self._cancel_label_editor)
        editor.installEventFilter(self)
        editor.show()
        editor.setFocus()
        self._label_editor = editor
        self._label_editor_item = item

    def eventFilter(self, obj, event) -> bool:
        # Label editor: commit on focus loss.
        if (
            self._label_editor is not None
            and obj is self._label_editor
            and event.type() == QEvent.Type.FocusOut
        ):
            self._commit_label_editor()
        # Viewport: re-anchor overlays when scrollbars/panel expansion resize
        # the viewport without a view ``resizeEvent``.
        if (
            obj is self.viewport()
            and event.type() == QEvent.Type.Resize
        ):
            self._position_status_chip()
            self._position_notice_tray()
        return super().eventFilter(obj, event)

    def _cancel_label_editor(self) -> None:
        self._close_label_editor()

    def _commit_label_editor(self) -> None:
        item = self._label_editor_item
        text = ""
        if self._label_editor is not None:
            text = self._label_editor.text().strip()
        self._close_label_editor()
        if item is None or item.scene() is None:
            return
        item.set_label(text)
        self._graph_mutated("label")
        if self._selected_item is item:
            self.inspector_refresh_requested.emit()

    def _close_label_editor(self) -> None:
        editor = self._label_editor
        self._label_editor = None
        self._label_editor_item = None
        if editor is not None:
            editor.removeEventFilter(self)
            editor.deleteLater()

    # ------------------------------------------------------------------
    # Node management
    # ------------------------------------------------------------------

    def add_node(self, node_id: str, scene_pos: QPointF | None = None) -> NodeItem:
        """Adds a node to the canvas. Without a position, places it in the
        center of the current view with a small cascading offset."""
        if self.consent_gate is not None:
            ok, message = self.consent_gate(node_id)
            if not ok:
                if message:
                    self.message_requested.emit(message, "error")
                return None
        if scene_pos is None:
            scene_pos = self.mapToScene(self.viewport().rect().center())
            offset = (len(self._node_items) % 8) * 24
            scene_pos += QPointF(offset, offset)

        item = create_node_item(
            node_id,
            self._node_name(node_id),
            self._node_category(node_id),
            self._node_inputs(node_id),
            self._node_outputs(node_id),
            specs=manager.node_specs(node_id),
        )
        try:
            item.set_available(manager.contains(node_id))
        except Exception:  # registry hiccup: treat as unavailable
            item.set_available(False)
        item.setFlags(
            item.flags()
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemIsFocusable
        )
        item.setPos(
            self.snap_point(scene_pos - QPointF(item.boundingRect().width() / 2, 0))
        )
        item.snap_provider = self.snap_point
        item.setZValue(len(self._node_items) + 1)
        if self._run_active:
            item.setOpacity(PENDING_OPACITY)

        item.close_requested.connect(lambda: self._remove_node(item))
        item.selected.connect(self._select_node)
        item.moved.connect(self._update_node_wires)
        item.moved.connect(lambda _item: self._graph_mutated("move"))
        item.port_pressed.connect(self._start_wire)
        item.message_requested.connect(
            lambda text: self.message_requested.emit(text, "info")
        )
        item.label_edit_requested.connect(self._start_label_edit)
        item.preview_requested.connect(self._on_node_preview_requested)
        if hasattr(item, "queue_manage_requested"):
            item.queue_manage_requested.connect(
                lambda it=item: self._manage_file_queue(it)
            )
        item.fleet_action_requested.connect(self._on_fleet_action)
        item.selected_items_provider = self.selected_nodes
        if isinstance(item, StrategyNodeItem):
            item.strategy_change_requested.connect(
                lambda key, it=item: self.set_strategy(it, key)
            )
        if hasattr(item, "file_changed"):
            item.file_changed.connect(
                lambda _it=item: self._graph_mutated("file")
            )

        self._scene.addItem(item)
        self._node_items.append(item)
        self._select_node(item)
        self._graph_mutated("add")
        return item

    def _on_theme_changed(self, _theme_id: str) -> None:
        """Repaint the scene with the refreshed theme constants."""
        self._scene.update()
        self.viewport().update()

    def _on_font_changed(self, _family: str) -> None:
        """Repaint the scene: node chrome sizes follow the font scale."""
        for item in self._node_items:
            item.refresh_layout()
        self._scene.update()
        self.viewport().update()

    def refresh_node_availability(self) -> None:
        """Re-syncs every node's availability against the registry.

        Called after extensions change: disabling/removing an extension marks
        its canvas nodes unavailable (kept but blocked from runs); enabling or
        re-importing restores them. A node restored while its module was
        missing was created without ports; when it becomes available again its
        ports are rebuilt so it can be connected and used normally.
        """
        for item in self._node_items:
            try:
                restored = False
                available = manager.contains(item.node_id())
                if available and not item.is_available():
                    restored = True
                item.set_available(available)
                if restored and not item.inputs() and not item.outputs():
                    inputs, outputs = manager.load_ports(item.node_id())
                    if inputs or outputs:
                        item.rebuild_ports(inputs, outputs)
            except Exception:  # registry hiccup: treat as unavailable
                item.set_available(False)

    def _remove_node(self, item: NodeItem) -> None:
        if item in self._node_items:
            self._item_by_iid.pop(item.iid, None)
            for connection in list(self._connections):
                if (
                    connection.source.node is item
                    or connection.target.node is item
                ):
                    self._disconnect(connection)
            item.on_node_removed()
            self._node_items.remove(item)
            self._scene.removeItem(item)
            if self.selected_item() is item and not self._syncing_selection:
                self._blank_inspector()
            self._graph_mutated("remove")

    def selected_nodes(self) -> list:
        """Currently selected items (native and custom flags are synced)."""
        return [item for item in self._node_items if item.isSelected()]

    def _select_node(self, item: NodeItem) -> None:
        # Collapse + raise; the funnel owns _selected_item/emission.
        self._syncing_selection = True
        try:
            for other in self._node_items:
                other.set_selected(other is item)
            item.setZValue(max((i.zValue() for i in self._node_items), default=0) + 1)
        finally:
            self._syncing_selection = False
        if not self._restoring:
            self._sync_inspector_to_scene()

    def _sync_inspector_to_scene(self) -> None:
        """Inspector rule from the live scene selection.

        Exactly one selected node shows its config; several show the
        multi-selection panel (``nodes_selected``); zero shows the
        placeholder. Emits only on change (scene signals fire per item
        during bulk updates, and several canvas paths already blank it).
        """
        if self._restoring or self._syncing_selection:
            return
        selected = self.selected_nodes()
        if len(selected) == 1:
            item = selected[0]
            if self._selected_item is not item:
                self._selected_item = item
                self._inspector_multi = None
                self._inspector_busy = True
                self.node_selected.emit(item)
            return
        if len(selected) > 1:
            self._selected_item = None
            items = list(selected)
            if items != self._inspector_multi:
                self._inspector_multi = items
                self._inspector_busy = True
                self.nodes_selected.emit(items)
            return
        self._selected_item = None
        self._inspector_multi = None
        if self._inspector_busy:
            self._inspector_busy = False
            self.node_selected.emit(None)

    def _blank_inspector(self) -> None:
        """Force the placeholder (bulk deletes, graph clear, background click)."""
        self._selected_item = None
        self._inspector_multi = None
        if self._inspector_busy:
            self._inspector_busy = False
            self.node_selected.emit(None)

    def selected_item(self) -> NodeItem | None:
        return getattr(self, "_selected_item", None)

    def mark_node_stale(self, iid: str) -> None:
        """Configuration edited: a completed node is no longer up to date."""
        for item in self._node_items:
            if item.iid == iid:
                item.mark_stale()
                return

    def on_config_changed(self, iid: str) -> None:
        """Inspector edit: configuration changes never affect a running session,
        so the node just goes stale (strategy changes arrive via
        ``on_strategy_requested``, which keeps canvas and Inspector in sync).

        The value is already written by the Inspector when this arrives, so
        the mutation joins the undo history here (post-mutation record) and
        marks the project dirty.
        """
        for item in self._node_items:
            if item.iid == iid:
                item.mark_stale()
                self._graph_mutated("config")
                return

    def on_strategy_requested(self, iid: str, key: str) -> None:
        """Inspector wants a strategy change: route it through the single
        mutator so cables are cut and ports rebuilt in both views."""
        for item in self._node_items:
            if item.iid != iid:
                continue
            self.set_strategy(item, key)
            return

    def set_strategy(self, item: NodeItem, key: str) -> None:
        """Apply a strategy to a strategy node, cutting its cables first.

        Ports are rebuilt from the selected variant, so every edge touching
        the item must be dropped before the rebuild (stale ports would dangle).
        The Inspector mirrors the change via ``config_changed``.
        """
        if not isinstance(item, StrategyNodeItem) or key == item.strategy():
            return
        # One undo step for the whole change: cutting the cables belongs to it.
        self._op_kind = "strategy"
        try:
            for connection in [
                c
                for c in self._connections
                if c.source.node is item or c.target.node is item
            ]:
                self._disconnect(connection)
            item.set_strategy(key)
            item.mark_stale()
        finally:
            self._op_kind = None
        self._update_node_wires(item)
        if self._selected_item is item:
            self.inspector_refresh_requested.emit()
        self._graph_mutated("strategy")

    def on_node_renamed(self, _iid: str) -> None:
        """Renamed from the Inspector: cosmetic, marks the project dirty."""
        self._graph_mutated("label")

    def retranslate_items(self) -> None:
        """Re-apply translations to the item labels."""
        for item in self._node_items:
            item.retranslate()
        self._update_pending_chip()

    def _update_node_wires(self, item: NodeItem) -> None:
        for connection in self._connections:
            if (
                connection.source.node is item
                or connection.target.node is item
            ):
                self._update_connection_path(connection)

    # ------------------------------------------------------------------
    # Projects (graph serialization)
    # ------------------------------------------------------------------

    def clear_graph(self) -> None:
        """Removes all nodes and connections from the canvas."""
        # Restoring/opening must not pollute the undo history with the
        # intermediate disconnects this method emits.
        prev_restoring = self._restoring
        self._restoring = True
        try:
            for connection in list(self._connections):
                self._disconnect(connection)
            self._syncing_selection = True
            try:
                for item in list(self._node_items):
                    item.on_node_removed()
                    self._scene.removeItem(item)
                self._node_items.clear()
            finally:
                self._syncing_selection = False
        finally:
            self._restoring = prev_restoring
        self._blank_inspector()

    def serialize_graph(self) -> dict:
        """Live graph as a project document: nodes, edges and view.

        Nodes carry their iid (instance identity); edges reference iids.
        """
        nodes = []
        for item in self._node_items:
            nodes.append(
                {
                    "iid": item.iid,
                    "id": item.node_id(),
                    "x": item.pos().x(),
                    "y": item.pos().y(),
                    "label": item.label(),
                    "config": item.capture_config(),
                }
            )
        edges = []
        for c in self._connections:
            src_item = c.source.node
            dst_item = c.target.node
            edges.append(
                {
                    "src": src_item.iid,
                    "src_out": src_item.outputs().index(c.source),
                    "dst": dst_item.iid,
                    "dst_in": dst_item.inputs().index(c.target),
                    "out_type": c.source.port_type_id,
                    "in_type": c.target.port_type_id,
                }
            )
        center = self.mapToScene(self.viewport().rect().center())
        return {
            "nodes": nodes,
            "edges": edges,
            "view": {
                "scale": self.transform().m11(),
                "cx": center.x(),
                "cy": center.y(),
            },
        }

    def restore_graph(self, data: dict) -> list[str]:
        """Rebuild the graph from an already-validated project document.

        Returns warnings (skipped edges, ambiguous endpoints in legacy
        projects); unrecoverable errors are validated earlier with
        project.load_project.
        """
        warnings: list[str] = []
        self.clear_graph()
        self._restoring = True
        items_by_iid: dict[str, NodeItem] = {}
        items_by_module: dict[str, list[NodeItem]] = {}
        for entry in data.get("nodes", []):
            node_id = entry["id"]
            try:
                item = self.add_node(node_id)
            except Exception as exc:  # broken module or invalid id
                warnings.append(
                    t("Node skipped ({node_id}): {exc}").format(
                        node_id=node_id, exc=exc
                    )
                )
                continue
            saved_iid = entry.get("iid")
            if isinstance(saved_iid, str) and saved_iid:
                item.iid = saved_iid  # stable across project saves
            items_by_iid.setdefault(item.iid, item)
            items_by_module.setdefault(node_id, []).append(item)
            try:
                item.setPos(float(entry.get("x", 0)), float(entry.get("y", 0)))
            except (TypeError, ValueError):
                pass
            try:
                entry_label = entry.get("label")
                if isinstance(entry_label, str):
                    item.set_label(entry_label)  # empty clears previous
                config = entry.get("config")
                if isinstance(config, dict):
                    item.apply_config(config)
            except Exception as exc:  # corrupt per-node state: keep defaults
                warnings.append(
                    t("Node kept with defaults ({node_id}): {exc}").format(
                        node_id=node_id, exc=exc
                    )
                )

        def resolve_endpoint(ref):
            """Endpoint iid: by iid (v2) or by module (legacy v1).

            Legacy: no ambiguity when the type appears once; with several
            copies it allocates the first in document order (deterministic).
            """
            if ref in items_by_iid:
                return items_by_iid[ref], None
            candidates = items_by_module.get(ref, [])
            if not candidates:
                return None, t("edge endpoint not resolvable: {ref}").format(
                    ref=ref
                )
            warning = (
                t("legacy edge matched by node type ({ref})").format(ref=ref)
                if len(candidates) > 1
                else None
            )
            return candidates[0], warning

        for edge in data.get("edges", []):
            src_ref, dst_ref = edge.get("src"), edge.get("dst")
            label = f"{src_ref} -> {dst_ref}"
            src_node, w1 = resolve_endpoint(src_ref)
            dst_node, w2 = resolve_endpoint(dst_ref)
            for warn in (w1, w2):
                if warn:
                    warnings.append(warn)
            if src_node is None or dst_node is None:
                warnings.append(t("Edge skipped: {label}").format(label=label))
                continue
            outs, ins = src_node.outputs(), dst_node.inputs()
            out_i = edge.get("src_out", 0)
            in_i = edge.get("dst_in", 0)
            if not isinstance(out_i, int) or not isinstance(in_i, int) or not (
                0 <= out_i < len(outs) and 0 <= in_i < len(ins)
            ):
                warnings.append(
                    t("Edge skipped (bad port index): {label}").format(
                        label=label
                    )
                )
                continue
            if not outs[out_i].can_connect_to(ins[in_i]):
                warnings.append(
                    t("Edge skipped (incompatible types): {label}").format(
                        label=label
                    )
                )
                continue
            if ins[in_i].is_full():
                warnings.append(
                    t("Edge skipped (input already connected): {label}").format(
                        label=label
                    )
                )
                continue
            self._create_connection(outs[out_i], ins[in_i])

        view = data.get("view")
        if isinstance(view, dict) and view:
            try:
                scale = float(view.get("scale", 1.0))
            except (TypeError, ValueError):
                scale = 1.0
            scale = max(MIN_ZOOM, min(MAX_ZOOM, scale))
            try:
                cx = float(view.get("cx", 0))
                cy = float(view.get("cy", 0))
            except (TypeError, ValueError):
                cx = cy = 0.0
            self.resetTransform()
            self.scale(scale, scale)
            self.centerOn(cx, cy)
            self.zoomChanged.emit(self.transform().m11())
        self._restoring = False
        self._selected_item = None
        self._syncing_selection = True
        try:
            for other in self._node_items:
                other.set_selected(False)
        finally:
            self._syncing_selection = False
        self._sync_inspector_to_scene()
        return warnings

    # ------------------------------------------------------------------
    # Connections
    # ------------------------------------------------------------------

    def _start_wire(self, port: Port) -> None:
        self._wire_source = port
        self._ghost_item = QGraphicsPathItem()
        ghost = QColor(get_port_color(port.port_type_id)).darker(125)
        self._ghost_item.setPen(
            QPen(ghost, WIRE_WIDTH, Qt.PenStyle.DashLine,
                 Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        )
        self._ghost_item.setZValue(0)
        self._scene.addItem(self._ghost_item)
        self.viewport().grabMouse()
        self._update_ghost(self._port_scene_pos(port))

    def _update_ghost(self, scene_pos: QPointF) -> None:
        if self._wire_source is None or self._ghost_item is None:
            return
        start = self._port_scene_pos(self._wire_source)
        self._ghost_item.setPath(self._bezier(start, scene_pos))

    def _finish_wire(self, scene_pos: QPointF) -> None:
        source = self._wire_source
        self._wire_source = None
        self.viewport().releaseMouse()
        if self._ghost_item is not None:
            self._scene.removeItem(self._ghost_item)
            self._ghost_item = None
        if source is None:
            return

        target = self._input_at(scene_pos)
        if target is None:
            return
        if target.is_full():
            self.message_requested.emit(t("Input already connected"), "error")
            return

        self._create_connection(source, target)

    def _create_connection(self, source: Port, target: Port) -> None:
        if not source.can_connect_to(target):
            self.message_requested.emit(
                t("Incompatible connection: {a} → {b}").format(
                    a=source.port_type_id, b=target.port_type_id
                ),
                "error",
            )
            return

        conflict = target.group_conflict()
        if conflict is not None:
            self.message_requested.emit(
                t(
                    "Cannot connect '{target}': it is mutually exclusive "
                    "with '{conflict}', which is already connected"
                ).format(target=target.name, conflict=conflict.name),
                "error",
            )
            return

        path_item = QGraphicsPathItem()
        path_item.setPen(QPen(QColor(get_port_color(source.port_type_id)),
                                 WIRE_WIDTH, Qt.PenStyle.SolidLine,
                                 Qt.PenCapStyle.RoundCap,
                                 Qt.PenJoinStyle.RoundJoin))
        path_item.setFlags(
            path_item.flags() | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
        )
        path_item.setZValue(0)
        if self._run_active:
            path_item.setOpacity(PENDING_OPACITY)
        self._scene.addItem(path_item)

        connection = Connection(source, target, path_item)
        self._connections.append(connection)
        source.connections.append(connection)
        target.connections.append(connection)
        self._update_connection_path(connection)
        source.node.on_connection_changed()
        target.node.on_connection_changed()
        self._graph_mutated("connect")

    def _disconnect(self, connection: Connection) -> None:
        if connection in self._connections:
            self._connections.remove(connection)
            if connection in connection.source.connections:
                connection.source.connections.remove(connection)
            if connection in connection.target.connections:
                connection.target.connections.remove(connection)
            self._scene.removeItem(connection.path_item)
            connection.source.node.on_connection_changed()
            connection.target.node.on_connection_changed()
            self._graph_mutated("disconnect")

    def _update_connection_path(self, connection: Connection) -> None:
        start = self._port_scene_pos(connection.source)
        end = self._port_scene_pos(connection.target)
        connection.path_item.setPath(self._bezier(start, end))

    def _port_scene_pos(self, port: Port) -> QPointF:
        return port.node.mapToScene(port.node.port_center(port))

    def _input_at(self, scene_pos: QPointF) -> Port | None:
        for item in self._node_items:
            for port in item.inputs():
                center = self._port_scene_pos(port)
                if (center - scene_pos).manhattanLength() <= WIRE_HIT_MARGIN:
                    return port
        return None

    @staticmethod
    def _bezier(start: QPointF, end: QPointF) -> QPainterPath:
        dx = max(40.0, abs(end.x() - start.x()) * 0.5)
        path = QPainterPath(start)
        path.cubicTo(
            start.x() + dx, start.y(), end.x() - dx, end.y(), end.x(), end.y()
        )
        return path

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------

    def wheelEvent(self, event) -> None:
        if self._label_editor is not None:
            self._commit_label_editor()
        factor = ZOOM_STEP if event.angleDelta().y() > 0 else 1 / ZOOM_STEP
        self._zoom_by(factor)

    def reset_view(self) -> None:
        self.resetTransform()
        self.centerOn(0, 0)
        self.zoomChanged.emit(self.transform().m11())

    def set_zoom_100(self) -> None:
        self.resetTransform()
        self.zoomChanged.emit(self.transform().m11())

    def mousePressEvent(self, event) -> None:
        if self._label_editor is not None:
            self._commit_label_editor()
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = event.position()
            self._pan_h_start = self.horizontalScrollBar().value()
            self._pan_v_start = self.verticalScrollBar().value()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.MouseButton.RightButton:
            # Right-click never reaches the scene: a press on empty space
            # would clear a multi-selection before its menu opens, showing
            # the bare-canvas menu instead of the fleet one. The menu itself
            # travels via contextMenuEvent (items own theirs, the canvas
            # shows selection actions when anything is selected).
            event.accept()
            return
        # Left-press on empty space falls through to the base class, which
        # owns the rubber-band selection (replace by default, Ctrl to add).
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._wire_source is not None:
            self._update_ghost(self.mapToScene(event.position().toPoint()))
            event.accept()
            return
        if self._panning:
            delta = event.position() - self._pan_start
            self.horizontalScrollBar().setValue(
                self._pan_h_start - round(delta.x())
            )
            self.verticalScrollBar().setValue(self._pan_v_start - round(delta.y()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._wire_source is not None:
            self._finish_wire(self.mapToScene(event.position().toPoint()))
            event.accept()
            return
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    # NOTE: deletion lives in window-scoped QShortcuts, not in keyPressEvent,
    # so Delete/Backspace work wherever the focus is (see
    # _install_delete_shortcuts).

    # ------------------------------------------------------------------
    # Drop from the library
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasFormat(NODE_MIME_TYPE):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasFormat(NODE_MIME_TYPE):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:
        node_id = bytes(event.mimeData().data(NODE_MIME_TYPE)).decode("utf-8")
        scene_pos = self.mapToScene(event.position().toPoint())
        self.add_node(node_id, scene_pos)
        self.viewport().update()
        event.acceptProposedAction()

    # ------------------------------------------------------------------
    # Node metadata
    # ------------------------------------------------------------------

    def _node_name(self, node_id: str) -> str:
        info = manager.node_info(node_id)
        if info is not None:
            return info.name
        try:
            return getattr(importlib.import_module(node_id), "NODE_NAME", node_id)
        except ImportError:
            return node_id

    def _node_category(self, node_id: str) -> str:
        info = manager.node_info(node_id)
        if info is not None:
            return info.category
        try:
            return getattr(importlib.import_module(node_id), "NODE_CATEGORY", "")
        except ImportError:
            return ""

    def _node_inputs(self, node_id: str) -> tuple:
        return manager.load_ports(node_id)[0]

    def _node_outputs(self, node_id: str) -> tuple:
        return manager.load_ports(node_id)[1]
