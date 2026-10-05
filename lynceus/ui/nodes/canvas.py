# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
import importlib
import math
import os
import re
from dataclasses import dataclass
from typing import Callable

from PySide6.QtCore import QEvent, QPointF, QRect, QSettings, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QKeySequence, QPainter, QPainterPath, QPen, QPolygonF, QShortcut
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QFrame,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from lynceus.processing.controller import PipelineCallbacks, PipelineController
from lynceus.processing.executor import workers_for_percent
from lynceus.processing.tiler import memory_budget_for_percent
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

# Textual progress markers ("Tiling... 14%", "DTM 3/51 | CHM 2/51"): once a
# transient row is updated with such a message it stops auto-hiding, so it
# never flickers during sparse (>autohide gap) heavy phases. It is removed
# explicitly when the run finishes/cancels.
_PROGRESS_RE = re.compile(r"%$|\d+/\d+", re.IGNORECASE)

WIRE_COLOR = QColor("#8a93a6")
WIRE_GHOST_COLOR = QColor("#3a465c")
WIRE_WIDTH = 2.0
WIRE_HIT_MARGIN = 12

# Dim the live canvas while edits are staged against an isolated run snapshot.
PENDING_OPACITY = 0.55

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


class _NoticeRow(QFrame):
    """A single notice chip of the canvas notice tray."""

    dismissed = Signal()

    def __init__(self, text: str, kind: str, autohide: bool):
        super().__init__()
        self.setObjectName("noticeRow")
        self._text = text
        color = NOTICE_KIND_COLORS.get(kind, "#c9d1e3")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 5, 6)
        layout.setSpacing(8)
        dot = QLabel("")
        dot.setFixedSize(8, 8)
        dot.setStyleSheet(
            f"background-color: {color}; border-radius: 4px; padding: 0px;"
        )
        self._body = QLabel(t(text))
        self._body.setObjectName("noticeText")
        self._body.setWordWrap(True)
        self._body.setMaximumWidth(NOTICE_ROW_MAX_TEXT)
        self._body.setStyleSheet(f"color: {color};")
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
    node_status_changed = Signal(object, str)
    load_tile_count = Signal(int)
    outputs_updated = Signal(object)  # list[OutputProduct].
    pipeline_started = Signal()
    pipeline_finished = Signal(dict, list, int, list, dict)  # outputs, tile_results, failed, tiles, metrics
    pipeline_cancelled = Signal()
    pipeline_progress = Signal(int, int)  # done, total
    inspector_refresh_requested = Signal()  # Refresh the selected node.
    segment_started = Signal(int, int, int)  # k, n, point count of the batch.
    segment_outputs_changed = Signal(int, int, object)  # k, n, batch outputs dict.
    consolidated_outputs_ready = Signal(object)  # final/merged outputs dict.
    node_preview_requested = Signal(str, str)  # path, node EN name (bridge preview).

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
        # change through the single-or-None inspector rule here.
        self._syncing_selection = False
        self._selected_item = None
        self._scene.selectionChanged.connect(self._sync_inspector_to_scene)
        self.setAcceptDrops(True)
        # Drag events are delivered to the viewport widget: without this,
        # Qt logs "drag leave received before drag enter" on every
        # library drop (harmless, but noisy).
        self.viewport().setAcceptDrops(True)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
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

        # EULA gates: the UI gate may show dialogs; the controller gate runs in
        # the worker and must never touch Qt.
        self.consent_gate: Callable[[str], tuple[bool, str | None]] | None = None

        self._install_toolbar()
        self._toolbar.btn_snap.setChecked(self._snap_enabled)
        self._install_zoom_shortcuts()
        self._install_delete_shortcuts()

        self.node_status_changed.connect(self._on_node_status)
        self.load_tile_count.connect(self._on_load_tile_count)
        self.pipeline_finished.connect(self._on_pipeline_finished)
        self.pipeline_cancelled.connect(self._on_pipeline_cancelled)
        self.segment_started.connect(self._on_segment_started)
        self.segment_outputs_changed.connect(self._on_segment_outputs)
        self.consolidated_outputs_ready.connect(self._on_consolidated_outputs)
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
        else:
            self._discard_pending_changes()

    def _discard_pending_changes(self) -> None:
        """Restore the canvas to its exact pre-run state."""
        snapshot = self._graph_snapshot or {}
        self._run_active = False
        self._graph_snapshot = None
        self.restore_graph(snapshot)
        self._set_run_pending(False)

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
                          segment_index: int | None = None) -> list:
        """Flatten run outputs into gallery products (Qt-free helper)."""
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
        source_files: dict[str, str] = {}
        source_paths: dict[str, str] = {}
        for iid, item in self._item_by_iid.items():
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
        if not outputs:
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
        self._flat_products = self._flatten_products(outputs)
        self._emit_gallery()

    def _emit_gallery(self) -> None:
        """Rebuild the gallery from accumulated segment + consolidated products."""
        products = []
        for k in sorted(self._segment_products):
            products.extend(self._segment_products[k])
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
            self._emit_gallery()

        # Staging: apply/discard the changes made during the run.
        self._ask_apply_pending()

    def _on_pipeline_cancelled(self) -> None:
        """Reset the toolbar and discard staging after cancellation."""
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
        self._clear_run_pending()

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
        # The inspector follows the single-selection rule after bulk deletes.
        self._sync_inspector_to_scene()

    @staticmethod
    def _focus_is_editing() -> bool:
        """True when keyboard focus sits in a text-editing widget."""
        focus = QApplication.focusWidget()
        return isinstance(focus, (QLineEdit, QTextEdit, QComboBox, QAbstractSpinBox))

    # ------------------------------------------------------------------
    # Background grid and snap
    # ------------------------------------------------------------------

    def drawBackground(self, painter, rect) -> None:
        painter.fillRect(rect, CANVAS_BG)
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
        )

    def _on_worker_node_status(self, iid: str, status: str) -> None:
        # After a cancel/finish a queued "processing" event could still land;
        # the run is over, so late statuses must not re-pulse the nodes.
        if not self._run_active:
            return
        item = self._item_by_iid.get(iid)
        if item is not None:
            self.node_status_changed.emit(item, status)

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
        load_items = [
            item
            for item in self._node_items
            if item.iid in in_scope
            and item.node_id() == "lynceus.nodes.lidar.source.load_las_laz"
        ]
        source_paths: dict[str, str] = {}
        for item in load_items:
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
        # Isolated snapshot of the canvas: the run executes this; any later
        # edit (nodes/cables/config) stays "pending" and does not
        # affect the session until the next run (see _ask_apply_pending).
        self._graph_snapshot = self.serialize_graph()
        self._run_active = True
        # Reset gallery accumulation for this run: segments publish themselves
        # incrementally; the consolidated/final products fill the last block.
        self._had_segments = False
        self._segment_products = {}
        self._flat_products = []
        chip = getattr(self, "_segment_chip", None)
        if chip is not None:
            chip.hide()
        self.outputs_updated.emit([])
        self.pipeline_started.emit()
        self._set_run_pending(True)
        self._sync_toolbar_state("running")
        run_prefs = _run_preferences()
        self._pipeline.set_acceleration_mode(run_prefs["acceleration_mode"])
        self._pipeline.set_acceleration_backend(
            run_prefs["acceleration_backend"], run_prefs["acceleration_device"]
        )
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
        self.graph_changed.emit()
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
        item.moved.connect(lambda _item: self.graph_changed.emit())
        item.port_pressed.connect(self._start_wire)
        item.message_requested.connect(
            lambda text: self.message_requested.emit(text, "info")
        )
        item.label_edit_requested.connect(self._start_label_edit)
        item.preview_requested.connect(self._on_node_preview_requested)
        item.selected_items_provider = self.selected_nodes
        if isinstance(item, StrategyNodeItem):
            item.strategy_change_requested.connect(
                lambda key, it=item: self.set_strategy(it, key)
            )
        if hasattr(item, "file_changed"):
            item.file_changed.connect(
                lambda _it=item: self.graph_changed.emit()
            )

        self._scene.addItem(item)
        self._node_items.append(item)
        self._select_node(item)
        self.graph_changed.emit()
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
                self._selected_item = None
                self.node_selected.emit(None)
            self.graph_changed.emit()

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
        """Single-or-None inspector rule from the live scene selection.

        Exactly one selected node shows its config; zero or several show
        the placeholder. Emits only on change (scene signals fire per item
        during bulk updates, and several canvas paths already emit None).
        """
        if self._restoring or self._syncing_selection:
            return
        selected = self.selected_nodes()
        if len(selected) == 1:
            item = selected[0]
            if self._selected_item is not item:
                self._selected_item = item
                self.node_selected.emit(item)
        elif self._selected_item is not None:
            self._selected_item = None
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
        ``on_strategy_requested``, which keeps canvas and Inspector in sync)."""
        for item in self._node_items:
            if item.iid == iid:
                item.mark_stale()
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
        for connection in [
            c
            for c in self._connections
            if c.source.node is item or c.target.node is item
        ]:
            self._disconnect(connection)
        item.set_strategy(key)
        item.mark_stale()
        self._update_node_wires(item)
        if self._selected_item is item:
            self.inspector_refresh_requested.emit()
        self.graph_changed.emit()

    def on_node_renamed(self, _iid: str) -> None:
        """Renamed from the Inspector: cosmetic, marks the project dirty."""
        self.graph_changed.emit()

    def retranslate_items(self) -> None:
        """Re-apply translations to the item labels."""
        for item in self._node_items:
            item.retranslate()

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
        self._selected_item = None
        self.node_selected.emit(None)

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
            label = entry.get("label")
            if isinstance(label, str) and label:
                item.set_label(label)
            config = entry.get("config")
            if isinstance(config, dict):
                item.apply_config(config)

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

        view = data.get("view") or {}
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
            QPen(ghost, WIRE_WIDTH, Qt.PenStyle.DashLine)
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
        path_item.setPen(QPen(QColor(get_port_color(source.port_type_id)), WIRE_WIDTH))
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
        self.graph_changed.emit()

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
            self.graph_changed.emit()

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
        super().mousePressEvent(event)
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.itemAt(event.position().toPoint()) is None
        ):
            self._selected_item = None
            self._syncing_selection = True
            try:
                for item in self._node_items:
                    item.set_selected(False)
            finally:
                self._syncing_selection = False
            self._sync_inspector_to_scene()

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
