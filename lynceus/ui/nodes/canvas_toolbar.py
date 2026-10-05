# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QToolButton, QVBoxLayout, QWidget

from lynceus.plugins.locale import t


class CanvasToolbar(QWidget):
    """Collapsible floating toolbar for canvas navigation and execution.

    Expanded mode shows all controls; collapsed mode leaves a small tab.
    """

    zoom_in_requested = Signal()
    zoom_out_requested = Signal()
    reset_requested = Signal()
    fit_requested = Signal()
    snap_toggled = Signal(bool)
    run_requested = Signal()
    cancel_requested = Signal()
    pause_requested = Signal()
    resume_requested = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("canvasToolbar")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._build_bar()
        self._build_tab()

        root.addWidget(self._bar)
        root.addWidget(self._tab)

        self.set_state("idle")
        self.set_collapsed(False)

    def _build_bar(self) -> None:
        self._bar = QWidget(self)
        self._bar.setObjectName("canvasToolbarBar")
        layout = QHBoxLayout(self._bar)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.btn_zoom_out = self._make_button("\u2212", "Zoom out (Ctrl+-)")
        self.btn_zoom_out.clicked.connect(self.zoom_out_requested)

        self.btn_zoom_in = self._make_button("+", "Zoom in (Ctrl+=)")
        self.btn_zoom_in.clicked.connect(self.zoom_in_requested)

        self.zoom_label = QLabel("100%")
        self.zoom_label.setObjectName("zoomLabel")
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        separator = QFrame(self._bar)
        separator.setObjectName("toolbarSeparator")
        separator.setFixedSize(1, 18)

        self.btn_reset = self._make_button("Reset", "Back to origin (Ctrl+0)")
        self.btn_reset.clicked.connect(self.reset_requested)

        self.btn_fit = self._make_button("Fit", "Fit all nodes in view")
        self.btn_fit.clicked.connect(self.fit_requested)

        self.btn_snap = self._make_button("Snap", "Snap nodes to grid (Ctrl+G)")
        self.btn_snap.setCheckable(True)
        self.btn_snap.setChecked(True)
        # Use toggled so keyboard shortcuts and programmatic changes are covered.
        self.btn_snap.toggled.connect(self.snap_toggled)

        separator2 = QFrame(self._bar)
        separator2.setObjectName("toolbarSeparator")
        separator2.setFixedSize(1, 18)

        self.btn_run = self._make_button("Run", "Execute the pipeline")
        self.btn_run.setObjectName("btnRun")
        self.btn_run.clicked.connect(self.run_requested)

        self.btn_pause = self._make_button("Pause", "Pause pipeline (soft)")
        self.btn_pause.setObjectName("btnPause")
        self.btn_pause.clicked.connect(self._on_pause_clicked)

        self.btn_cancel = self._make_button("Cancel", "Cancel pipeline")
        self.btn_cancel.setObjectName("btnCancel")
        self.btn_cancel.clicked.connect(self.cancel_requested)

        self.btn_collapse = self._make_button("\u00ab", "Collapse toolbar")
        self.btn_collapse.clicked.connect(self.toggle)

        for widget in (
            self.btn_zoom_out,
            self.btn_zoom_in,
            self.zoom_label,
            separator,
            self.btn_reset,
            self.btn_fit,
            self.btn_snap,
            separator2,
            self.btn_run,
            self.btn_pause,
            self.btn_cancel,
            self.btn_collapse,
        ):
            layout.addWidget(widget)

    def _on_pause_clicked(self) -> None:
        # Use the English source property rather than translated text so the
        # toggle remains stable across language changes.
        if self.btn_pause.property("origText_en") == "Pause":
            self.pause_requested.emit()
        else:
            self.resume_requested.emit()

    def set_state(self, state: str) -> None:
        """Apply the Run/Pause/Cancel state machine to the toolbar.

        idle       -> Run on, Pause off, Cancel off
        running    -> Run off, Pause on ("Pause"), Cancel on
        paused     -> Run off, Pause on ("Resume"), Cancel on
        cancelling -> Run off, Pause off, Cancel off
        """
        paused = state == "paused"
        cancelling = state == "cancelling"

        self.btn_run.setEnabled(state == "idle")
        self.btn_cancel.setEnabled(state in ("running", "paused"))

        pause_enabled = state in ("running", "paused")
        if paused:
            self.btn_pause.setProperty("origText_en", "Resume")
            self.btn_pause.setText(t("Resume"))
            self.btn_pause.setProperty("origTip_en", "Resume pipeline")
            self.btn_pause.setToolTip(t("Resume pipeline"))
        else:
            self.btn_pause.setProperty("origText_en", "Pause")
            self.btn_pause.setText(t("Pause"))
            self.btn_pause.setProperty("origTip_en", "Pause pipeline (soft)")
            self.btn_pause.setToolTip(t("Pause pipeline (soft)"))
        self.btn_pause.setEnabled(pause_enabled)

        if cancelling:
            self.btn_pause.setEnabled(False)

    def _build_tab(self) -> None:
        self._tab = QWidget(self)
        self._tab.setObjectName("canvasToolbarTab")
        layout = QHBoxLayout(self._tab)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(0)

        self.btn_expand = self._make_button("\u00bb", "Expand toolbar")
        self.btn_expand.clicked.connect(self.toggle)

        layout.addWidget(self.btn_expand)

    def _make_button(self, text: str, tooltip: str) -> QToolButton:
        button = QToolButton(self)
        button.setProperty("origText_en", text)
        button.setText(t(text))
        button.setProperty("origTip_en", tooltip)
        button.setToolTip(t(tooltip))
        # QSS does not provide a cursor property.
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        return button

    def set_collapsed(self, collapsed: bool) -> None:
        self._bar.setVisible(not collapsed)
        self._tab.setVisible(collapsed)
        self.reflow()

    def reflow(self) -> None:
        """Resize the toolbar after translated text changes its contents.

        ``translate_widget`` changes button text and therefore its
        sizeHint, but the internal QHBoxLayout of `_bar` and the root
        QVBoxLayout cache their old hints. Both layouts must be invalidated
        and updateGeometry forced on the buttons so Qt recomputes and resizes
        the floating bar (otherwise long translated text leaves buttons
        clipped until collapsed/expanded).
        """
        for btn in self.findChildren(QToolButton):
            btn.updateGeometry()
        for bar in (self, self._bar):
            lay = bar.layout()
            if lay is not None:
                lay.invalidate()
                lay.activate()
        hint = self.sizeHint()
        self.resize(hint)

    def toggle(self) -> None:
        self.set_collapsed(not self._tab.isVisible())

    def set_zoom_percent(self, percent: int) -> None:
        self.zoom_label.setText(f"{percent}%")
