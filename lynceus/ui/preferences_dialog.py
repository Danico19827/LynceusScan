# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Application preferences window (Qt UI).

Sections on the left, settings pages on the right: General (language,
start window) and Performance (CPU/memory budgets, acceleration policy).
Values persist in QSettings and apply on Ok; Cancel discards; Restore
defaults resets the widgets (saved only on Ok). Performance settings take
effect on the next run (the worker pool is created per run) and never
enter node fingerprints.
"""

from __future__ import annotations

import os

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from lynceus.plugins.locale import locale_manager, t
from lynceus.ui.fonts import available_families
from lynceus.ui.theming import available_themes
from lynceus.processing.acceleration import available_acceleration_devices
from lynceus.processing.executor import workers_for_percent
from lynceus.processing.tiler import (
    MEMORY_BUDGET_FLOOR_BYTES,
    memory_budget_for_percent,
    system_memory_bytes,
)
from lynceus.ui.settings_keys import (
    ACCEL_BACKEND_KEY,
    ACCEL_DEVICE_KEY,
    ACCEL_MODE_KEY,
    CPU_PERCENT_DEFAULT,
    CPU_PERCENT_KEY,
    DISPLAY_BUDGET_DEFAULT,
    DISPLAY_BUDGET_KEY,
    DISPLAY_RASTER_CELLS_DEFAULT,
    DISPLAY_RASTER_CELLS_KEY,
    DISPLAY_TABLE_ROWS_DEFAULT,
    DISPLAY_TABLE_ROWS_KEY,
    FONT_FAMILY_DEFAULT,
    FONT_FAMILY_KEY,
    FONT_SIZE_DEFAULT,
    FONT_SIZE_KEY,
    KEEP_INTERMEDIATES_KEY,
    MEMORY_PERCENT_DEFAULT,
    MEMORY_PERCENT_KEY,
    OPERATOR_NAME_KEY,
    OPERATOR_ORG_KEY,
    RECENT_LIMIT_DEFAULT,
    RECENT_LIMIT_KEY,
    SETTINGS_APP,
    SETTINGS_ORG,
    SHOW_START_KEY,
    SNAP_KEY,
    THEME_DEFAULT,
    THEME_KEY,
)

ACCEL_MODES = ("auto", "on", "off")
ACCEL_BACKENDS = ("auto", "cpu", "cuda", "opencl", "directml")


def _settings() -> QSettings:
    return QSettings(SETTINGS_ORG, SETTINGS_APP)


def _int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _gb(bytes_value: int) -> str:
    return f"{bytes_value / 1024**3:.1f} GB"


def _desc(text: str, parent=None) -> QLabel:
    """Dimmed description line under a setting (Inspector-style help)."""
    label = QLabel(t(text), parent)
    label.setObjectName("panelHint")
    label.setProperty("origText_en", text)
    label.setWordWrap(True)
    return label


def _T(text: str, parent=None) -> QLabel:
    """Translated static label with the English source pinned.

    The translate pass captures the current text as the original on its
    first run; pages built with t() at construction would strand that pass
    in the old language, so the source is pinned up front instead.
    """
    label = QLabel(t(text), parent)
    label.setProperty("origText_en", text)
    return label


class _GeneralPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        form = QFormLayout(self)
        self.language = QComboBox(self)
        for code in locale_manager.available():
            self.language.addItem(code.upper(), code)
        form.addRow(_T("Language"), self.language)
        form.addRow(_desc(
            "Interface language. Applies immediately when you press Ok.",
            self,
        ))
        self.theme = QComboBox(self)
        for tid, label in available_themes():
            self.theme.addItem(label, tid)
        form.addRow(_T("Theme"), self.theme)
        form.addRow(_desc(
            "Color theme. Applies immediately when you press Ok.",
            self,
        ))
        self.font = QComboBox(self)
        self.font.addItem(t("System default"), "")
        # Each family renders in itself; pin an explicit size so the style
        # never resolves a default (-1) size with a setPointSize warning.
        base_size = self.font.font().pointSize()
        if base_size <= 0:
            base_size = 9
        for family in available_families():
            self.font.addItem(family, family)
            item_font = QFont(family)
            item_font.setPointSize(base_size)
            self.font.setItemData(
                self.font.count() - 1, item_font, Qt.ItemDataRole.FontRole
            )
        form.addRow(_T("Font"), self.font)
        form.addRow(_desc(
            "Interface font. Applies immediately when you press Ok.",
            self,
        ))
        self.font_size = QSpinBox(self)
        self.font_size.setRange(70, 150)
        self.font_size.setSuffix("%")
        form.addRow(_T("Font size"), self.font_size)
        form.addRow(_desc(
            "Interface font size. Applies immediately when you press Ok.",
            self,
        ))
        self.show_start = QCheckBox(t("Show start window on launch"), self)
        self.show_start.setProperty(
            "origText_en", "Show start window on launch"
        )
        form.addRow("", self.show_start)
        form.addRow(_desc(
            "Open the launcher with recent projects and templates "
            "on startup. Disable to start directly on a blank canvas.",
            self,
        ))
        self.recent_limit = QSpinBox(self)
        self.recent_limit.setRange(5, 30)
        form.addRow(_T("Recent projects limit"), self.recent_limit)
        form.addRow(_desc(
            "How many recent projects the launcher and the File menu remember.",
            self,
        ))
        self.snap = QCheckBox(t("Snap nodes to grid"), self)
        self.snap.setProperty("origText_en", "Snap nodes to grid")
        form.addRow("", self.snap)
        form.addRow(_desc(
            "When on, nodes align to the grid while dragging. "
            "Toggle anytime with Ctrl+G.",
            self,
        ))
        self.load()

    def load(self) -> None:
        settings = _settings()
        code = str(settings.value("language", "en") or "en").lower()
        index = self.language.findData(code)
        self.language.setCurrentIndex(index if index >= 0 else 0)
        tid = str(
            settings.value(THEME_KEY, THEME_DEFAULT) or THEME_DEFAULT
        ).lower()
        tindex = self.theme.findData(tid)
        self.theme.setCurrentIndex(tindex if tindex >= 0 else 0)
        fam = str(
            settings.value(FONT_FAMILY_KEY, FONT_FAMILY_DEFAULT)
            or FONT_FAMILY_DEFAULT
        )
        findex = self.font.findData(fam)
        self.font.setCurrentIndex(findex if findex >= 0 else 0)
        size = _int(
            settings.value(FONT_SIZE_KEY, FONT_SIZE_DEFAULT),
            FONT_SIZE_DEFAULT,
        )
        self.font_size.setValue(min(150, max(70, size)))
        show = settings.value(SHOW_START_KEY, True)
        self.show_start.setChecked(str(show).lower() not in ("false", "0", ""))
        recent = _int(settings.value(RECENT_LIMIT_KEY, RECENT_LIMIT_DEFAULT),
                      RECENT_LIMIT_DEFAULT)
        self.recent_limit.setValue(min(30, max(5, recent)))
        snap = settings.value(SNAP_KEY, True)
        self.snap.setChecked(str(snap).lower() not in ("false", "0", ""))

    def save(self) -> None:
        settings = _settings()
        settings.setValue("language", self.language.currentData() or "en")
        settings.setValue(THEME_KEY, self.theme.currentData() or THEME_DEFAULT)
        settings.setValue(
            FONT_FAMILY_KEY, self.font.currentData() or FONT_FAMILY_DEFAULT
        )
        settings.setValue(FONT_SIZE_KEY, self.font_size.value())
        settings.setValue(SHOW_START_KEY, self.show_start.isChecked())
        settings.setValue(RECENT_LIMIT_KEY, self.recent_limit.value())
        settings.setValue(SNAP_KEY, self.snap.isChecked())

    def restore_defaults(self) -> None:
        self.language.setCurrentIndex(max(0, self.language.findData("en")))
        self.theme.setCurrentIndex(
            max(0, self.theme.findData(THEME_DEFAULT))
        )
        self.font.setCurrentIndex(
            max(0, self.font.findData(FONT_FAMILY_DEFAULT))
        )
        self.font_size.setValue(FONT_SIZE_DEFAULT)
        self.show_start.setChecked(True)
        self.recent_limit.setValue(RECENT_LIMIT_DEFAULT)
        self.snap.setChecked(True)


class _PerformancePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._cores = max(1, os.cpu_count() or 4)
        self._ram = system_memory_bytes()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        form = QFormLayout()
        self.cpu_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.cpu_slider.setRange(10, 100)
        self.cpu_slider.setSingleStep(5)
        self.cpu_label = QLabel(self)
        self.cpu_slider.valueChanged.connect(self._refresh_cpu_label)
        cpu_row = QHBoxLayout()
        cpu_row.addWidget(self.cpu_slider, 1)
        cpu_row.addWidget(self.cpu_label)
        form.addRow(_T("CPU limit"), cpu_row)
        form.addRow(_desc(
            "Share of CPU cores used per run. 100% is fastest. "
            "Recommended: 50-75% when working alongside other apps.",
            self,
        ))

        self.mem_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.mem_slider.setRange(5, 100)
        self.mem_slider.setSingleStep(5)
        self.mem_label = QLabel(self)
        self.mem_slider.valueChanged.connect(self._refresh_mem_label)
        mem_row = QHBoxLayout()
        mem_row.addWidget(self.mem_slider, 1)
        mem_row.addWidget(self.mem_label)
        form.addRow(_T("Memory budget"), mem_row)
        form.addRow(_desc(
            "RAM ceiling for tiling per run. Recommended: 25%. "
            "Never drops below 1 GiB.",
            self,
        ))
        layout.addLayout(form)

        accel_form = QFormLayout()
        self.accel_mode = QComboBox(self)
        for mode in ACCEL_MODES:
            self.accel_mode.addItem(t(mode.capitalize()), mode)
        accel_form.addRow(_T("Acceleration"), self.accel_mode)
        self.accel_backend = QComboBox(self)
        for backend in ACCEL_BACKENDS:
            self.accel_backend.addItem(
                t("Auto") if backend == "auto" else backend.upper(), backend
            )
        self.accel_backend.currentIndexChanged.connect(self._reload_devices)
        accel_form.addRow(_T("Backend"), self.accel_backend)
        self.accel_device = QComboBox(self)
        accel_form.addRow(_T("Device"), self.accel_device)
        layout.addLayout(accel_form)
        layout.addWidget(_desc(
            "Hardware acceleration for numeric passes. Auto lets the app "
            "decide; Off forces CPU. Specific backends fall back to CPU "
            "with a warning when unavailable.",
            self,
        ))

        self.keep_intermediates = QCheckBox(
            t("Keep intermediate files and session cache"), self
        )
        layout.addWidget(self.keep_intermediates)
        layout.addWidget(_desc(
            "Keep per-tile intermediates and the shared tile cache so "
            "identical re-runs reuse previous sessions. Off (default): "
            "runs keep only final products and always recompute.",
            self,
        ))

        hint = QLabel(
            t("Performance settings apply to the next run."),
            self,
        )
        hint.setObjectName("hintLabel")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch(1)
        self.load()

    def _refresh_cpu_label(self) -> None:
        pct = self.cpu_slider.value()
        workers = workers_for_percent(pct, self._cores)
        self.cpu_label.setText(
            t("{n} cores ({p}%)").format(n=workers, p=pct)
        )

    def _refresh_mem_label(self) -> None:
        pct = self.mem_slider.value()
        budget = memory_budget_for_percent(pct, self._ram or None)
        floor_note = ""
        if budget <= MEMORY_BUDGET_FLOOR_BYTES:
            floor_note = " " + t("(minimum)")
        self.mem_label.setText(
            t("{size} ({p}%)").format(size=_gb(budget), p=pct) + floor_note
        )

    def _reload_devices(self) -> None:
        backend = self.accel_backend.currentData() or "auto"
        self.accel_device.clear()
        self.accel_device.addItem(t("Auto"), "")
        for dev in available_acceleration_devices():
            if backend != "auto" and dev.backend != backend:
                continue
            if not dev.available:
                continue
            self.accel_device.addItem(f"{dev.name} ({dev.backend})", dev.device_id)

    def load(self) -> None:
        settings = _settings()
        cpu = _int(settings.value(CPU_PERCENT_KEY, CPU_PERCENT_DEFAULT),
                   CPU_PERCENT_DEFAULT)
        self.cpu_slider.setValue(min(100, max(10, cpu)))
        self._refresh_cpu_label()
        mem = _int(settings.value(MEMORY_PERCENT_KEY, MEMORY_PERCENT_DEFAULT),
                   MEMORY_PERCENT_DEFAULT)
        self.mem_slider.setValue(min(100, max(5, mem)))
        self._refresh_mem_label()
        mode = str(settings.value(ACCEL_MODE_KEY, "auto") or "auto")
        self.accel_mode.setCurrentIndex(
            max(0, list(ACCEL_MODES).index(mode) if mode in ACCEL_MODES else 0)
        )
        backend = str(settings.value(ACCEL_BACKEND_KEY, "auto") or "auto")
        self.accel_backend.setCurrentIndex(
            max(0, list(ACCEL_BACKENDS).index(backend)
                if backend in ACCEL_BACKENDS else 0)
        )
        self._reload_devices()
        device = str(settings.value(ACCEL_DEVICE_KEY, "") or "")
        index = self.accel_device.findData(device)
        self.accel_device.setCurrentIndex(index if index >= 0 else 0)
        keep = settings.value(KEEP_INTERMEDIATES_KEY, False)
        self.keep_intermediates.setChecked(
            str(keep).lower() not in ("false", "0", "")
        )

    def save(self) -> None:
        settings = _settings()
        settings.setValue(CPU_PERCENT_KEY, self.cpu_slider.value())
        settings.setValue(MEMORY_PERCENT_KEY, self.mem_slider.value())
        settings.setValue(ACCEL_MODE_KEY, self.accel_mode.currentData() or "auto")
        settings.setValue(ACCEL_BACKEND_KEY,
                          self.accel_backend.currentData() or "auto")
        settings.setValue(ACCEL_DEVICE_KEY, self.accel_device.currentData() or "")
        settings.setValue(
            KEEP_INTERMEDIATES_KEY, self.keep_intermediates.isChecked()
        )

    def restore_defaults(self) -> None:
        self.cpu_slider.setValue(CPU_PERCENT_DEFAULT)
        self.mem_slider.setValue(MEMORY_PERCENT_DEFAULT)
        self.accel_mode.setCurrentIndex(0)
        self.accel_backend.setCurrentIndex(0)
        self._reload_devices()
        self.accel_device.setCurrentIndex(0)
        self.keep_intermediates.setChecked(False)


BUDGET_PRESETS = (
    ("Auto", "auto"),
    ("0.75M", 750_000),
    ("1.5M", 1_500_000),
    ("3M", 3_000_000),
    ("6M", 6_000_000),
    ("12M", 12_000_000),
)
RASTER_CELL_PRESETS = (
    ("1M", 1_000_000),
    ("2M", 2_000_000),
    ("4M", 4_000_000),
    ("8M", 8_000_000),
)
TABLE_ROW_PRESETS = (
    ("1k", 1_000),
    ("2k", 2_000),
    ("5k", 5_000),
    ("10k", 10_000),
    ("20k", 20_000),
)


def _preset_combo(presets, parent=None) -> QComboBox:
    combo = QComboBox(parent)
    for label, data in presets:
        combo.addItem(label, data)
    return combo


def _select_preset(combo: QComboBox, raw, fallback: int = 0) -> None:
    """Select a preset by value (settings may store ints as strings)."""
    if isinstance(raw, str) and raw.strip().lower() == "auto":
        index = combo.findData("auto")
    else:
        try:
            index = combo.findData(int(raw))
        except (TypeError, ValueError):
            index = -1
    combo.setCurrentIndex(index if index >= 0 else fallback)


class _DisplayPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        form = QFormLayout(self)
        self.budget = _preset_combo(BUDGET_PRESETS, self)
        form.addRow(_T("Point budget"), self.budget)
        form.addRow(_desc(
            "How many points previews draw. Auto adapts to available RAM. "
            "Higher values show more detail but use more memory.",
            self,
        ))
        self.raster_cells = _preset_combo(RASTER_CELL_PRESETS, self)
        form.addRow(_T("Raster cells"), self.raster_cells)
        form.addRow(_desc(
            "Maximum raster cells drawn in 2D previews. Larger rasters "
            "are decimated.",
            self,
        ))
        self.table_rows = _preset_combo(TABLE_ROW_PRESETS, self)
        form.addRow(_T("Table rows"), self.table_rows)
        form.addRow(_desc(
            "Maximum rows shown in table previews.",
            self,
        ))
        self.load()

    def load(self) -> None:
        settings = _settings()
        _select_preset(
            self.budget,
            settings.value(DISPLAY_BUDGET_KEY, DISPLAY_BUDGET_DEFAULT),
        )
        _select_preset(
            self.raster_cells,
            settings.value(
                DISPLAY_RASTER_CELLS_KEY, DISPLAY_RASTER_CELLS_DEFAULT
            ),
            fallback=2,
        )
        _select_preset(
            self.table_rows,
            settings.value(
                DISPLAY_TABLE_ROWS_KEY, DISPLAY_TABLE_ROWS_DEFAULT
            ),
            fallback=2,
        )

    def save(self) -> None:
        settings = _settings()
        settings.setValue(DISPLAY_BUDGET_KEY, self.budget.currentData())
        settings.setValue(
            DISPLAY_RASTER_CELLS_KEY, self.raster_cells.currentData()
        )
        settings.setValue(
            DISPLAY_TABLE_ROWS_KEY, self.table_rows.currentData()
        )

    def restore_defaults(self) -> None:
        _select_preset(self.budget, DISPLAY_BUDGET_DEFAULT)
        _select_preset(
            self.raster_cells, DISPLAY_RASTER_CELLS_DEFAULT, fallback=2
        )
        _select_preset(
            self.table_rows, DISPLAY_TABLE_ROWS_DEFAULT, fallback=2
        )


class _OperatorPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        form = QFormLayout(self)
        self.name = QLineEdit(self)
        self.org = QLineEdit(self)
        form.addRow(_T("Name"), self.name)
        form.addRow(_T("Organization"), self.org)
        form.addRow(_desc(
            "Optional, embedded in the provenance of every product you "
            "produce. Leave empty to omit your identity from the output "
            "files (your responsibility as the producer is governed by the "
            "Terms & Conditions regardless).",
            self,
        ))
        form.addRow(_desc(
            "Changing the operator recomputes cached products.",
            self,
        ))
        self.load()

    def load(self) -> None:
        settings = _settings()
        self.name.setText(
            str(settings.value(OPERATOR_NAME_KEY, "") or "")
        )
        self.org.setText(str(settings.value(OPERATOR_ORG_KEY, "") or ""))

    def save(self) -> None:
        settings = _settings()
        settings.setValue(OPERATOR_NAME_KEY, self.name.text().strip())
        settings.setValue(OPERATOR_ORG_KEY, self.org.text().strip())

    def restore_defaults(self) -> None:
        self.name.clear()
        self.org.clear()


class _AboutPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        from lynceus import __version__ as app_version
        from lynceus import links

        form = QFormLayout(self)
        for label, value in (
            (t("Version"), app_version),
            (t("License"), "GPL-3.0-or-later"),
            (t("Built with"), "Qt for Python (PySide6) \u00b7 LGPL-3.0"),
            (t("Contact"), "lynceusscan@gmail.com"),
            (t("Repository"), links.CORE_REPO_URL),
        ):
            field = QLabel(value, self)
            field.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            form.addRow(label, field)
        site = QLabel(
            f'<a href="{links.WEBSITE_URL}">{links.WEBSITE_URL}</a>', self
        )
        site.setOpenExternalLinks(True)
        site.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.LinksAccessibleByMouse
        )
        form.addRow(_T("Website"), site)

    def load(self) -> None:
        return None

    def save(self) -> None:
        return None

    def restore_defaults(self) -> None:
        return None


class PreferencesDialog(QDialog):
    """Sections-based application preferences."""

    def __init__(self, parent=None, on_apply=None):
        super().__init__(parent)
        self._on_apply_callback = on_apply
        self.setWindowTitle(t("Preferences"))
        self.setProperty("origWinTitle_en", "Preferences")
        self.setModal(True)
        self.resize(560, 420)
        # Guard against oversized-font geometry explosions: cap to the
        # primary screen (the font filter keeps pathological faces out,
        # this keeps any large-but-valid family reachable + closable).
        from PySide6.QtGui import QGuiApplication

        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry().size()
            self.setMaximumSize(
                int(avail.width() * 0.95), int(avail.height() * 0.9)
            )

        root = QVBoxLayout(self)
        content = QHBoxLayout()
        self.nav = QListWidget(self)
        self.nav.setObjectName("prefsNav")
        self.nav.setFixedWidth(150)
        self.pages = QStackedWidget(self)
        # English section names: QListWidgetItem is outside the translate
        # pass, so they are re-applied manually on language changes.
        self._nav_en = [
            "General", "Performance", "Display", "Operator", "About",
        ]
        self._pages = [
            (t("General"), _GeneralPage(self)),
            (t("Performance"), _PerformancePage(self)),
            (t("Display"), _DisplayPage(self)),
            (t("Operator"), _OperatorPage(self)),
            (t("About"), _AboutPage(self)),
        ]
        for label, page in self._pages:
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, self.pages.count())
            self.nav.addItem(item)
            self.pages.addWidget(page)
        self.nav.setCurrentRow(0)
        self.nav.currentRowChanged.connect(self.pages.setCurrentIndex)
        content.addWidget(self.nav)
        content.addWidget(self.pages, 1)
        root.addLayout(content, 1)
        from lynceus.ui.translate import language_changed

        language_changed.connect(self._on_language_changed)

        bottom = QHBoxLayout()
        restore = QPushButton(t("Restore defaults"), self)
        restore.setProperty("origText_en", "Restore defaults")
        restore.clicked.connect(self._on_restore)
        bottom.addWidget(restore)
        bottom.addStretch(1)
        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Apply
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        self._buttons.accepted.connect(self._on_accept)
        self._buttons.rejected.connect(self.reject)
        self._buttons.clicked.connect(self._on_button_clicked)
        bottom.addWidget(self._buttons)
        root.addLayout(bottom)
        # Standard buttons follow the OS locale, not the app language: pin
        # our own text so Apply translates with everything else.
        apply_btn = self._buttons.button(
            QDialogButtonBox.StandardButton.Apply
        )
        apply_btn.setProperty("origText_en", "Apply")
        apply_btn.setText(t("Apply"))

    def _on_restore(self) -> None:
        for _label, page in self._pages:
            page.restore_defaults()

    def _on_language_changed(self, _code: str) -> None:
        """Re-run the translate pass over the open dialog (Apply path)."""
        from lynceus.ui.translate import translate_widget

        for row, name_en in enumerate(self._nav_en):
            item = self.nav.item(row)
            if item is not None:
                item.setText(t(name_en))
        translate_widget(self)

    def _on_accept(self) -> None:
        for _label, page in self._pages:
            page.save()
        self.accept()

    def _on_button_clicked(self, button) -> None:
        if (
            self._buttons.buttonRole(button)
            == QDialogButtonBox.ButtonRole.ApplyRole
            and self._on_apply_callback is not None
        ):
            for _label, page in self._pages:
                page.save()
            self._on_apply_callback()
