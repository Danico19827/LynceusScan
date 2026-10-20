# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Pre-flight advisor dialog (Qt).

Shown once before a run (or once per file queue) only when the header-based
heuristics find actionable advice. It never gates the run: the operator can
always continue. Outcomes: continue (with an optional per-run cache
override), abort, or jump to Performance.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from lynceus.plugins.locale import t
from lynceus.processing.preflight import Preflight

#: result codes returned by ``exec()`` (Accepted/Rejected are Qt's).
OPEN_PERFORMANCE = 2
SHOW_LOADER = 3


def _card(layout: QVBoxLayout, *widgets) -> None:
    """Group widgets in a card container (visual grouping only)."""
    card = QWidget()
    card.setObjectName("preflightCard")
    # Plain QWidgets never paint QSS backgrounds without this.
    card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    inner = QVBoxLayout(card)
    inner.setContentsMargins(10, 8, 10, 8)
    inner.setSpacing(4)
    for widget in widgets:
        inner.addWidget(widget)
    layout.addWidget(card)


def _fmt_points(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f} G"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f} M"
    if n >= 1_000:
        return f"{n / 1_000:.1f} k"
    return str(n)


def _fmt_gb(nbytes: float) -> str:
    return f"{nbytes / (1024 ** 3):.1f} GB"


class PreflightDialog(QDialog):
    """Advisory sizing + one-click cache / Performance shortcuts."""

    def __init__(self, parent, advice: Preflight):
        super().__init__(parent)
        self._advice = advice
        self._enable_cache = advice.ask_cache and not advice.cache_enabled
        self._apply_merge = advice.merge_floor_suggest > 0
        self._workers_n = advice.workers_suggest
        self._resolution_m = advice.resolution_suggest
        self._ram_bytes = advice.ram_budget_suggest
        self._segments_cap = advice.segments_cap_suggest
        self._dont_show = False
        self.setWindowTitle(t("Pre-flight advisor"))
        self.setProperty("origWinTitle_en", "Pre-flight advisor")
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 12)
        layout.setSpacing(10)

        extent = advice.area_m2 or 0.0
        if extent:
            headline = t(
                "This run holds about {points} points over ~{km2} km\u00b2 "
                "(~{tiles} tiles of ~{per_tile} points each)."
            ).format(
                points=_fmt_points(advice.total_points),
                km2=f"{extent / 1_000_000:.2f}",
                tiles=advice.tile_count_est,
                per_tile=_fmt_points(advice.per_tile_points),
            )
        else:
            headline = t(
                "This run holds about {points} points "
                "(~{per_tile} points per tile)."
            ).format(
                points=_fmt_points(advice.total_points),
                per_tile=_fmt_points(advice.per_tile_points),
            )
        title = QLabel(headline)
        title.setWordWrap(True)
        title.setObjectName("preflightHeadline")
        layout.addWidget(title)

        # Optional widgets default to None up front: every branch below
        # only overwrites when its advice fires, so _on_accept never
        # touches a missing attribute (merge/cache-only dialogs crashed
        # here before this guard existed).
        self._cache_box: QCheckBox | None = None
        self._workers_box: QCheckBox | None = None
        self._loader_button = None
        self._merge_box: QCheckBox | None = None
        self._segments_box: QCheckBox | None = None
        self._resolution_box: QCheckBox | None = None
        self._ram_box: QCheckBox | None = None
        if advice.ask_cache:
            self._cache_box = QCheckBox(
                t(
                    "Enable cache for this run (reuse intermediates and share "
                    "tile cache) so a re-run never restarts from scratch"
                )
            )
            self._cache_box.setChecked(self._enable_cache)
            _card(layout, self._cache_box)

        if advice.ask_ram:
            ram = QLabel(
                t(
                    "Estimated peak memory ~{peak} against a ~{budget} budget: "
                    "the engine will throttle, or this run may fail. Consider "
                    "lowering the CPU share or raising the memory budget."
                ).format(
                    peak=_fmt_gb(advice.est_peak_bytes),
                    budget=(
                        _fmt_gb(advice.budget_bytes)
                        if advice.budget_bytes
                        else "?"
                    ),
                )
            )
            ram.setWordWrap(True)
            perf = QPushButton(t("Open Performance\u2026"))
            perf.setProperty("origText_en", "Open Performance\u2026")
            perf.clicked.connect(lambda: self.done(OPEN_PERFORMANCE))
            _card(layout, ram, perf)
            if advice.workers_suggest > 0:
                self._workers_box = QCheckBox(
                    t(
                        "Use {n} workers for this run (lower memory peak)"
                    ).format(n=advice.workers_suggest)
                )
                self._workers_box.setChecked(True)
                _card(layout, self._workers_box)

        if advice.merge_floor_suggest > 0:
            merge = QLabel(
                t(
                    "Many small tiles (~{tiles} averaging ~{per_tile} "
                    "points). Consider fusing sparse neighbors in the "
                    "loader's tiling settings (suggested floor ~{floor} "
                    "points) to cut file, task and merge overhead."
                ).format(
                    tiles=f"{advice.tile_count_est:,}",
                    per_tile=_fmt_points(advice.per_tile_points),
                    floor=_fmt_points(advice.merge_floor_suggest),
                )
            )
            merge.setWordWrap(True)
            self._merge_box = QCheckBox(
                t(
                    "Apply suggested floor (~{floor} points) to source "
                    "loaders"
                ).format(floor=_fmt_points(advice.merge_floor_suggest))
            )
            self._merge_box.setChecked(True)
            _card(layout, merge, self._merge_box)
        else:
            self._merge_box = None

        if advice.segments_cap_suggest > 0:
            segments = QLabel(
                t(
                    "Full mosaic would not fit (~{size}); independent "
                    "segments (~{n} batches of ~{cap} points) keep full "
                    "quality with bounded memory. Consolidate nodes are "
                    "skipped (per-segment products stay separate)."
                ).format(
                    size=_fmt_gb(advice.mosaic_bytes_est),
                    n=f"{advice.segments_count_est:,}",
                    cap=_fmt_points(advice.segments_cap_suggest),
                )
            )
            segments.setWordWrap(True)
            self._segments_box = QCheckBox(
                t(
                    "Run independent segments (~{n} batches, no final merge)"
                ).format(n=f"{advice.segments_count_est:,}")
            )
            self._segments_box.setChecked(True)
            _card(layout, segments, self._segments_box)
        else:
            self._segments_box = None

        if advice.merge_floor_suggest > 0 or advice.segments_cap_suggest > 0:
            self._loader_button = QPushButton(t("Show in Inspector"))
            self._loader_button.setProperty("origText_en", "Show in Inspector")
            self._loader_button.clicked.connect(
                lambda: self.done(SHOW_LOADER))
            layout.addWidget(self._loader_button)
        else:
            self._loader_button = None

        if advice.ask_mosaic:
            if advice.ram_budget_suggest > 0:
                mosaic = QLabel(
                    t(
                        "Full-extent mosaics may reach ~{size} (at 1 m "
                        "cells) against a ~{budget} budget, but fit in "
                        "this machine's RAM. Raising the memory budget "
                        "keeps full resolution."
                    ).format(
                        size=_fmt_gb(advice.mosaic_bytes_est),
                        budget=(
                            _fmt_gb(advice.budget_bytes)
                            if advice.budget_bytes
                            else "?"
                        ),
                    )
                )
                mosaic.setWordWrap(True)
                self._ram_box = QCheckBox(
                    t(
                        "Raise memory to ~{mem} for this run"
                    ).format(mem=_fmt_gb(advice.ram_budget_suggest))
                )
                self._ram_box.setChecked(True)
                _card(layout, mosaic, self._ram_box)
                self._resolution_box = None
            else:
                mosaic = QLabel(
                    t(
                        "Full-extent mosaics may reach ~{size} (at 1 m cells) "
                        "against a ~{budget} budget — the engine merges in "
                        "windows at full resolution instead (slower, "
                        "identical pixels)."
                    ).format(
                        size=_fmt_gb(advice.mosaic_bytes_est),
                        budget=(
                            _fmt_gb(advice.budget_bytes)
                            if advice.budget_bytes
                            else "?"
                        ),
                    )
                )
                mosaic.setWordWrap(True)
                self._resolution_box = QCheckBox(
                    t(
                        "Set {res} m resolution in raster nodes"
                    ).format(res=f"{advice.resolution_suggest:g}")
                )
                self._resolution_box.setChecked(True)
                _card(layout, mosaic, self._resolution_box)
                self._ram_box = None
        else:
            self._resolution_box = None
            self._ram_box = None

        self._never = QCheckBox(t("Don't show this advisor again"))
        layout.addWidget(self._never)

        buttons = QDialogButtonBox(self)
        run_btn = buttons.addButton(
            t("Continue"), QDialogButtonBox.ButtonRole.AcceptRole
        )
        run_btn.setProperty("origText_en", "Continue")
        cancel_btn = buttons.addButton(
            t("Cancel run"), QDialogButtonBox.ButtonRole.RejectRole
        )
        cancel_btn.setProperty("origText_en", "Cancel run")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self) -> None:
        self._dont_show = self._never.isChecked()
        if self._cache_box is not None:
            self._enable_cache = self._cache_box.isChecked()
        if self._merge_box is not None:
            self._apply_merge = self._merge_box.isChecked()
        if self._workers_box is not None:
            self._workers_n = (
                self._advice.workers_suggest
                if self._workers_box.isChecked() else 0
            )
        if self._resolution_box is not None:
            self._resolution_m = (
                self._advice.resolution_suggest
                if self._resolution_box.isChecked() else 0.0
            )
        if self._ram_box is not None:
            self._ram_bytes = (
                self._advice.ram_budget_suggest
                if self._ram_box.isChecked() else 0.0
            )
        if self._segments_box is not None:
            self._segments_cap = (
                self._advice.segments_cap_suggest
                if self._segments_box.isChecked() else 0
            )
        self.accept()

    def enable_cache(self) -> bool:
        """Whether the operator asked to force cache on for this run."""
        return self._enable_cache

    def apply_merge_floor(self) -> bool:
        """Whether the suggested floor goes into the loader configs."""
        return self._apply_merge

    def apply_workers(self) -> int:
        """Worker override for this run (0 = keep current)."""
        return self._workers_n

    def apply_resolution(self) -> float:
        """Target product resolution in meters (0.0 = keep current)."""
        return self._resolution_m

    def apply_memory(self) -> float:
        """Memory budget override in bytes (0.0 = keep current)."""
        return self._ram_bytes

    def apply_segments(self) -> int:
        """Batch cap for independent segments (0 = keep current)."""
        return self._segments_cap

    def dont_show_again(self) -> bool:
        return self._dont_show
