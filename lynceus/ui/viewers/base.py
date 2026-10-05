# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
from threading import Event

from PySide6.QtCore import QObject, QRunnable, QSize, Qt, QThreadPool, Signal
from PySide6.QtWidgets import (
    QLabel,
    QProgressBar,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from lynceus.plugins.locale import t
from lynceus.ui.outputs_model import viewer_kind_label

PREVIEW_THREAD_POOL = QThreadPool()
PREVIEW_THREAD_POOL.setMaxThreadCount(2)


class _PreviewSignals(QObject):
    loaded = Signal(int, object)
    failed = Signal(int, str)


class _PreviewTask(QRunnable):
    def __init__(self, generation, loader, cancelled, signals):
        super().__init__()
        self.generation = generation
        self.loader = loader
        self.cancelled = cancelled
        self.signals = signals

    def run(self) -> None:
        if self.cancelled.is_set():
            return
        try:
            result = self.loader(self.cancelled)
        except Exception as exc:
            if not self.cancelled.is_set():
                self.signals.failed.emit(self.generation, str(exc))
            return
        if not self.cancelled.is_set():
            self.signals.loaded.emit(self.generation, result)


def default_preview_size() -> QSize:
    """Initial preview window size.

    Historical 1000x700 as the minimum, growing with the screen up to ~45%
    of the available height (10:7 aspect, clamped to fit). Falls back to
    1000x700 without a screen.
    """
    from PySide6.QtWidgets import QApplication

    fallback = QSize(1000, 700)
    screen = QApplication.primaryScreen()
    if screen is None:
        return fallback
    avail = screen.availableGeometry()
    if avail.height() <= 0 or avail.width() <= 0:
        return fallback
    height = min(max(700, int(avail.height() * 0.45)), avail.height())
    width = min(max(1000, int(height * 10 / 7)), avail.width())
    return QSize(width, height)


def viewer_title(kind: str) -> str:
    """Translated window title for a viewer id ('Preview — Raster')."""
    return f"{t('Preview')} — {t(viewer_kind_label(kind))}"


class BaseViewer(QWidget):
    """Base viewer that can be shown as a window or embedded.

    Without a parent it is a top-level window (.show()); with a parent it is
    embedded. Viewer variants implement set_payload() and replace the
    placeholder with their rendering.
    """

    def __init__(self, kind: str, parent: QWidget | None = None):
        super().__init__(parent)
        self._kind = kind
        self._product_title = ""
        self.setWindowTitle(viewer_title(kind))
        self.resize(640, 480)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._placeholder = QLabel(t("No data available"))
        self._placeholder.setObjectName("viewerPlaceholder")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Styled by app.qss (QLabel#viewerPlaceholder) so it follows themes.
        self._layout.addWidget(self._placeholder)
        self._progress = QProgressBar(self)
        self._progress.setObjectName("previewLoadingProgress")
        self._progress.setRange(0, 0)
        self._progress.setTextVisible(False)
        self._progress.setMaximumHeight(4)
        self._progress.hide()
        self._layout.addWidget(self._progress)
        self._load_generation = 0
        self._async_jobs = {}
        self._placeholder.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

    def set_product_title(self, node_name: str, label: str) -> None:
        """Identify the preview: node + product in the window title.

        Branch-scoped twins read as e.g. ``Generate DTM — d1/dtm_mosaic``.
        Call before set_payload; render paths re-compose the title.
        """
        self._product_title = (
            f"{node_name} — {label}" if node_name else label
        )
        self.setWindowTitle(self._compose_title())

    def _compose_title(self) -> str:
        base = viewer_title(self._kind)
        if self._product_title:
            return f"{base} — {self._product_title}"
        return base

    def set_payload(self, payload) -> None:
        """Receive the data to visualize (overridden in variants)."""
        self.show_placeholder(t("No data available"))

    def start_async_load(
        self, loader, on_loaded, on_failed, show_loading: bool = True
    ) -> None:
        """Run file I/O and data preparation outside the GUI thread.

        Loaders receive a cancellation event and must return plain data. They
        must not access widgets; callbacks are delivered on the GUI thread.
        """
        self._load_generation += 1
        generation = self._load_generation
        for old_generation, job in tuple(self._async_jobs.items()):
            job[0].set()
            self._async_jobs.pop(old_generation, None)

        cancelled = Event()
        signals = _PreviewSignals()
        self._async_jobs[generation] = (cancelled, on_loaded, on_failed, signals)
        signals.loaded.connect(self._async_load_succeeded)
        signals.failed.connect(self._async_load_failed)
        if show_loading:
            self._placeholder.setText(t("Loading..."))
            self._placeholder.show()
        self._progress.show()
        PREVIEW_THREAD_POOL.start(
            _PreviewTask(generation, loader, cancelled, signals)
        )

    def _async_load_succeeded(self, generation: int, result) -> None:
        job = self._async_jobs.pop(generation, None)
        if job is None or generation != self._load_generation:
            return
        self._progress.hide()
        job[1](result)

    def _async_load_failed(self, generation: int, message: str) -> None:
        job = self._async_jobs.pop(generation, None)
        if job is None or generation != self._load_generation:
            return
        self._progress.hide()
        job[2](message)

    def cancel_async_loads(self) -> None:
        self._load_generation += 1
        for cancelled, *_rest in self._async_jobs.values():
            cancelled.set()
        self._async_jobs.clear()
        self._progress.hide()

    def show_placeholder(self, text: str) -> None:
        self.cancel_async_loads()
        self._progress.hide()
        self._placeholder.setText(text)
        self._placeholder.show()

    def closeEvent(self, event) -> None:
        self.cancel_async_loads()
        super().closeEvent(event)
