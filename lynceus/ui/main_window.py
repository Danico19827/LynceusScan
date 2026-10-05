# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
from pathlib import Path

import psutil
import time as _time
from PySide6.QtCore import QEvent, QPoint, QRect, QSettings, QSize, QTimer, Qt, QUrl
from PySide6.QtGui import (
    QAction,
    QDesktopServices,
    QFont,
    QKeySequence,
    QMouseEvent,
)
from PySide6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QSizePolicy,
    QToolButton,
    QWidget,
)
from lynceus import project as project_io
from lynceus import templates as templates_io
from lynceus.plugins.locale import locale_manager, t
from lynceus.plugins.registry import manager
from lynceus.plugins.store import ConsentStore
from lynceus.resources import resource_path
from lynceus.processing.controller import default_output_root
from lynceus.ui.extensions_dialog import (
    ExtensionsDialog,
    build_ui_gate,
    build_worker_gate,
)
from lynceus.ui.node_library import populate_library
from lynceus.ui.settings_keys import (
    RECENT_KEY,
    RECENT_LIMIT_DEFAULT,
    RECENT_LIMIT_KEY,
    SETTINGS_APP,
    SETTINGS_ORG,
)
from lynceus.ui.start_dialog import TemplateMetaDialog, TemplatePickerDialog
from lynceus.ui.translate import language_changed, translate_widget
from lynceus.ui.ui_loader import load_ui_inplace

MAINWINDOW_UI_PATH = str(resource_path("lynceus/ui/mainwindow.ui"))

RESIZE_MARGIN = 8
MIN_WINDOW_SIZE = (400, 300)

_EDGE_CURSORS = {
    "left": Qt.CursorShape.SizeHorCursor,
    "right": Qt.CursorShape.SizeHorCursor,
    "top": Qt.CursorShape.SizeVerCursor,
    "bottom": Qt.CursorShape.SizeVerCursor,
    "topleft": Qt.CursorShape.SizeFDiagCursor,
    "bottomright": Qt.CursorShape.SizeFDiagCursor,
    "topright": Qt.CursorShape.SizeBDiagCursor,
    "bottomleft": Qt.CursorShape.SizeBDiagCursor,
}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._store = ConsentStore()
        manager.disabled_provider = lambda: self._store.disabled_ids()
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        load_ui_inplace(MAINWINDOW_UI_PATH, self)
        self.setWindowTitle("LynceusScan")
        self._init_status_dot()
        self._init_runs_panel()
        populate_library(self.libraryNodes)
        self.libraryNodes.node_add_requested.connect(self.canvasView.add_node)
        self.canvasView.message_requested.connect(self._on_status_message)
        self.canvasView.node_selected.connect(self.inspectorPanel.set_node)
        self.inspectorPanel.config_changed.connect(
            self.canvasView.on_config_changed
        )
        self.inspectorPanel.strategy_change_requested.connect(
            self.canvasView.on_strategy_requested
        )
        self.inspectorPanel.label_changed.connect(self.canvasView.on_node_renamed)
        self.canvasView.inspector_refresh_requested.connect(
            self.inspectorPanel.refresh
        )
        self.canvasView.outputs_updated.connect(
            self.outputDock.set_products
        )
        self.canvasView.node_preview_requested.connect(self._on_node_preview)
        self._connect_view_actions()
        self._init_project_state()
        self._connect_project_actions()
        self._install_extensions_menu()
        self._install_template_actions()
        self._install_preview_action()
        self._install_tools_menu()
        self._install_menu_logo()
        self.canvasView.set_consent_gates(
            build_ui_gate(self._store, self),
            build_worker_gate(self._store),
        )
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        right_width = max(self.rightPanel.minimumWidth(), 320)
        middle_width = max(
            self.splitter.width()
            - self.nodesPanel.minimumWidth()
            - right_width,
            0,
        )
        self.splitter.setSizes(
            [
                self.nodesPanel.minimumWidth(),
                middle_width,
                right_width,
            ]
        )
        self.rightSplitter.setStretchFactor(0, 4)
        self.rightSplitter.setStretchFactor(1, 6)
        # Taller default for the outputs gallery. Applied once (any user
        # drag marks it done, so manual adjustments are never overridden).
        self._right_split_init_done = False
        self.rightSplitter.splitterMoved.connect(
            lambda *_: setattr(self, "_right_split_init_done", True)
        )
        QTimer.singleShot(0, self._maybe_init_right_splitter)

        left_splitter = getattr(self, "nodesLeftSplitter", None)
        if left_splitter is not None:
            left_splitter.setStretchFactor(0, 3)
            left_splitter.setStretchFactor(1, 2)
            left_splitter.setSizes([320, 220])

        self._dragging_window = False
        self._drag_offset = QPoint()
        self._resize_edge: str | None = None
        self._resize_origin = QPoint()
        self._resize_geometry = QRect()
        self._window_maximized = False
        self._normal_geometry: QRect | None = None

        self._install_window_controls()
        self._enable_edge_tracking()

        QApplication.instance().installEventFilter(self)

        self._init_metrics_display()
        self._setup_language()

    # ---------- language ----------

    def _setup_language(self) -> None:
        settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
        saved = settings.value("language", "en")
        locale_manager.set_language(saved or "en")
        language_changed.connect(self._on_language_changed)
        translate_widget(self)

    def _set_language(self, code: str) -> None:
        locale_manager.set_language(code)
        QSettings(SETTINGS_ORG, SETTINGS_APP).setValue("language", code)
        language_changed.emit(locale_manager.language)

    def _on_language_changed(self, _code: str) -> None:
        # Rebuilds first (fresh English text), then a single translate pass
        # over the whole tree using origText_en, so switching back to English
        # really reverts everything.
        populate_library(self.libraryNodes)
        self.canvasView.retranslate_items()
        self.inspectorPanel.refresh()
        translate_widget(self)
        self._refresh_status_elision()
        # The floating overlay toolbar must reflow to fit longer Spanish text.
        # A plain adjustSize() reuses stale layout hints; reflow() invalidates
        # and resizes so buttons are never clipped.
        toolbar = getattr(self.canvasView, "_toolbar", None)
        if toolbar is not None:
            toolbar.reflow()
        runs = getattr(self, "runsPanel", None)
        if runs is not None:
            runs.refresh()

    def _on_extensions_changed(self) -> None:
        from lynceus.plugins.theme import theme_manager
        from lynceus.ui.settings_keys import THEME_DEFAULT, THEME_KEY
        from lynceus.ui.theming import apply_theme

        locale_manager.mark_dirty()
        if locale_manager.language not in locale_manager.available():
            self._set_language("en")
        theme_manager.mark_dirty()
        if theme_manager.theme not in theme_manager.available():
            QSettings(SETTINGS_ORG, SETTINGS_APP).setValue(
                THEME_KEY, THEME_DEFAULT
            )
            apply_theme(THEME_DEFAULT)
        populate_library(self.libraryNodes)
        self.canvasView.refresh_node_availability()

    # ---------- View menu actions ----------

    def _connect_view_actions(self) -> None:
        reset_action = getattr(self, "actionReset_View", None)
        if reset_action is not None:
            reset_action.setShortcut(QKeySequence("Ctrl+0"))
            reset_action.triggered.connect(self.canvasView.reset_view)

        zoom_action = getattr(self, "actionZoom100", None) or getattr(
            self, "actionZoom_100", None
        )
        if zoom_action is None:
            view_menu = getattr(self, "viewMenu", None)
            self.actionZoom100 = QAction("Zoom 100%", self)
            self.actionZoom100.setShortcut(QKeySequence("Ctrl+1"))
            if view_menu is not None:
                view_menu.addAction(self.actionZoom100)
            zoom_action = self.actionZoom100
        else:
            zoom_action.setShortcut(QKeySequence("Ctrl+1"))
        zoom_action.triggered.connect(self.canvasView.set_zoom_100)

    # ---------- project (state and File menu actions) ----------

    def _init_project_state(self) -> None:
        self._project_path: str | None = None
        self._template_name: str | None = None
        self._dirty = False
        self._recent_paths = self._load_recent()
        self._recent_menu: QMenu | None = None
        self.canvasView.graph_changed.connect(self._on_graph_changed)
        self._update_title()

    def _connect_project_actions(self) -> None:
        new_action = getattr(self, "actionNew_Project", None)
        if new_action is not None:
            new_action.setShortcut(QKeySequence("Ctrl+N"))
            new_action.triggered.connect(self.new_project)

        open_action = getattr(self, "actionOpen_File", None)
        if open_action is not None:
            open_action.setShortcut(QKeySequence("Ctrl+O"))
            open_action.triggered.connect(self.open_project)

        save_action = getattr(self, "actionSave", None)
        if save_action is not None:
            save_action.setShortcut(QKeySequence("Ctrl+S"))
            save_action.triggered.connect(self.save_project)

        save_as_action = getattr(self, "actionSave_As", None)
        if save_as_action is not None:
            save_as_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
            save_as_action.triggered.connect(self.save_project_as)

        copy_action = getattr(self, "actionSave_a_Copy", None)
        if copy_action is not None:
            copy_action.triggered.connect(self.save_project_copy)

        exit_action = getattr(self, "actionExit", None)
        if exit_action is not None:
            exit_action.setShortcut(QKeySequence("Ctrl+Q"))
            exit_action.triggered.connect(self.close)

        self._install_recent_menu()

    def _install_recent_menu(self) -> None:
        file_menu = getattr(self, "fileMenu", None)
        if file_menu is None:
            return
        self._recent_menu = QMenu("Open Recent", file_menu)
        self._recent_menu.menuAction().setProperty("origText_en", "Open Recent")
        before = getattr(self, "actionSave", None)
        if before is not None:
            file_menu.insertMenu(before, self._recent_menu)
        else:
            file_menu.addMenu(self._recent_menu)
        self._rebuild_recent_menu()

    def _install_extensions_menu(self) -> None:
        file_menu = getattr(self, "fileMenu", None)
        if file_menu is None:
            return
        action = QAction("Extensions...", self)
        action.triggered.connect(self.show_extensions_dialog)
        self.file_extensions_action = action
        before = getattr(self, "actionExit", None)
        if before is not None:
            file_menu.insertAction(before, action)
        else:
            file_menu.addAction(action)

    def _install_template_actions(self) -> None:
        file_menu = getattr(self, "fileMenu", None)
        if file_menu is None:
            return
        new_from = QAction(t("New From Template..."), self)
        new_from.setProperty("origText_en", "New From Template...")
        new_from.triggered.connect(self.new_from_template)
        save_tpl = QAction(t("Save as Template..."), self)
        save_tpl.setProperty("origText_en", "Save as Template...")
        save_tpl.triggered.connect(self.save_as_template)
        before = getattr(self, "actionExit", None)
        if before is not None:
            file_menu.insertAction(before, new_from)
            file_menu.insertAction(before, save_tpl)
        else:
            file_menu.addAction(new_from)
            file_menu.addAction(save_tpl)

    def _install_preview_action(self) -> None:
        file_menu = getattr(self, "fileMenu", None)
        if file_menu is None:
            return
        action = QAction(t("Open Preview..."), self)
        action.setProperty("origText_en", "Open Preview...")
        action.setShortcut(QKeySequence("Ctrl+Shift+O"))
        action.triggered.connect(self.open_preview_file)
        self.file_preview_action = action
        before = getattr(self, "actionExit", None)
        if before is not None:
            file_menu.insertAction(before, action)
        else:
            file_menu.addAction(action)

    def open_preview_file(self) -> None:
        """Open a floating preview for any supported file (bridge mode).

        No pipeline runs, no session is created, and nothing is copied: the
        file is visualized in place through its canonical viewer.
        """
        from lynceus.ui.outputs_model import viewer_for_path

        preview_filter = (
            "Point Cloud (*.las *.laz);;Raster (*.tif *.tiff);;"
            "Vector (*.gpkg *.shp *.geojson);;Table (*.csv *.txt);;"
            "JSON (*.json);;All Files (*)"
        )
        path, _ = QFileDialog.getOpenFileName(
            self, t("Open Preview"), "", preview_filter
        )
        if not path:
            return
        if viewer_for_path(path) is None:
            self._on_status_message(
                t("Cannot preview '{name}': no viewer for this file type").format(
                    name=Path(path).name
                ),
                "warning",
            )
            return
        self.outputDock.products.open_external_file(path, t("External file"))

    def new_from_template(self) -> None:
        if self.canvasView.is_run_active():
            QMessageBox.information(
                self, t("Run in progress"),
                t("Cancel or wait for the current run before opening a template."),
            )
            return
        if not self._confirm_discard():
            return
        infos = templates_io.list_templates()
        if not [i for i in infos if i.valid]:
            QMessageBox.information(
                self,
                t("New From Template"),
                t(
                    "No templates yet. Build a workflow and save it with "
                    "File > Save as Template."
                ),
            )
            return
        dlg = TemplatePickerDialog(infos, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        selected = dlg.selected()
        if selected is not None:
            self._open_path(selected.path, from_template=True)

    def save_as_template(self) -> bool:
        suggested = ""
        if self._project_path:
            suggested = Path(self._project_path).stem
        elif self._template_name:
            suggested = self._template_name
        dlg = TemplateMetaDialog(suggested, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return False
        meta = dlg.meta()
        doc = templates_io.build_template_doc(
            self.canvasView.serialize_graph(), meta
        )
        dest = (
            templates_io.save_target_dir()
            / templates_io.template_file_name(meta["name"])
        )
        if dest.exists():
            ret = QMessageBox.question(
                self,
                t("Save as Template"),
                t("A template with this name already exists. Overwrite it?"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if ret != QMessageBox.StandardButton.Yes:
                return False
        try:
            project_io.save_project(dest, doc)
        except OSError as exc:
            QMessageBox.critical(
                self,
                t("Save as Template"),
                t("Could not save template: {exc}").format(exc=exc),
            )
            return False
        self._on_status_message(t("Template saved: ") + dest.name, "info")
        return True
    def _install_tools_menu(self) -> None:
        menubar = getattr(self, "menubar", None)
        if menubar is None:
            return
        tools_menu = QMenu(t("Tools"), menubar)
        tools_menu.menuAction().setProperty("origText_en", "Tools")
        prefs = QAction(t("Preferences..."), self)
        prefs.setProperty("origText_en", "Preferences...")
        prefs.triggered.connect(self.show_preferences)
        tools_menu.addAction(prefs)
        menubar.addMenu(tools_menu)
        self._install_help_menu(menubar)

    def _install_help_menu(self, menubar) -> None:
        """Help menu: outbound links to the website, docs and trackers."""
        from lynceus import links

        help_menu = QMenu(t("Help"), menubar)
        help_menu.menuAction().setProperty("origText_en", "Help")
        for label, url in (
            ("Documentation", links.DOCS_URL),
            ("Get Extensions", links.EXTENSIONS_CATALOG_URL),
            ("Report Issue", links.ISSUES_URL),
        ):
            action = QAction(t(label), self)
            action.setProperty("origText_en", label)
            action.triggered.connect(
                lambda _=False, u=url: QDesktopServices.openUrl(QUrl(u))
            )
            help_menu.addAction(action)
        menubar.addMenu(help_menu)

    def _install_menu_logo(self) -> None:
        """Brand button at the left of the menubar: opens the website."""
        from lynceus import links
        from lynceus.ui.theming import theme_changed

        menubar = getattr(self, "menubar", None)
        if menubar is None:
            return
        logo = QToolButton(menubar)
        logo.setObjectName("menuLogo")
        logo.setIconSize(QSize(22, 22))
        logo.setFixedSize(28, 28)
        logo.setCursor(Qt.CursorShape.PointingHandCursor)
        logo.setToolTip(t("Open website"))
        logo.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl(links.WEBSITE_URL))
        )
        menubar.setCornerWidget(logo, Qt.Corner.TopLeftCorner)
        self._menu_logo = logo
        self._refresh_menu_logo()
        theme_changed.connect(self._on_theme_changed_logo)

    def _on_theme_changed_logo(self, _theme_id: str) -> None:
        """Retint the menubar logo with the new theme text color."""
        self._refresh_menu_logo()

    def _refresh_menu_logo(self) -> None:
        from PySide6.QtGui import QColor, QIcon

        from lynceus.ui.branding import (
            ICON_SQUARE_SVG,
            THEME_COLORS,
            render_svg,
            tint_pixmap,
        )

        logo = getattr(self, "_menu_logo", None)
        if logo is None:
            return
        pixmap = render_svg(ICON_SQUARE_SVG, 44)
        if pixmap.isNull():
            return
        logo.setIcon(
            QIcon(tint_pixmap(pixmap, QColor(THEME_COLORS["text"])))
        )

    def show_preferences(self) -> None:
        from lynceus.ui.fonts import saved_size
        from lynceus.ui.preferences_dialog import PreferencesDialog
        from lynceus.ui.settings_keys import (
            FONT_FAMILY_DEFAULT,
            FONT_FAMILY_KEY,
            THEME_DEFAULT,
            THEME_KEY,
        )

        baseline = {
            "language": str(
                QSettings(SETTINGS_ORG, SETTINGS_APP).value("language", "en")
                or "en"
            ),
            "theme": str(
                QSettings(SETTINGS_ORG, SETTINGS_APP).value(
                    THEME_KEY, THEME_DEFAULT
                )
                or THEME_DEFAULT
            ).lower(),
            "font": str(
                QSettings(SETTINGS_ORG, SETTINGS_APP).value(
                    FONT_FAMILY_KEY, FONT_FAMILY_DEFAULT
                )
                or FONT_FAMILY_DEFAULT
            ),
            "font_size": saved_size(),
        }
        dialog = PreferencesDialog(
            self, on_apply=lambda: self._apply_prefs_changes(baseline)
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._apply_prefs_changes(baseline)

    def _apply_prefs_changes(self, baseline: dict) -> None:
        """Apply language/theme/font deltas vs `baseline` (Ok/Apply share)."""
        from lynceus.ui.fonts import apply_font, saved_size
        from lynceus.ui.settings_keys import (
            FONT_FAMILY_DEFAULT,
            FONT_FAMILY_KEY,
            THEME_DEFAULT,
            THEME_KEY,
        )
        from lynceus.ui.theming import apply_theme

        current_language = str(
            QSettings(SETTINGS_ORG, SETTINGS_APP).value("language", "en")
            or "en"
        )
        if current_language != baseline["language"]:
            self._set_language(current_language)
            baseline["language"] = locale_manager.language
        current_theme = str(
            QSettings(SETTINGS_ORG, SETTINGS_APP).value(
                THEME_KEY, THEME_DEFAULT
            )
            or THEME_DEFAULT
        ).lower()
        if current_theme != baseline["theme"]:
            apply_theme(current_theme)
            from lynceus.plugins.theme import theme_manager

            baseline["theme"] = theme_manager.theme
        current_font = str(
            QSettings(SETTINGS_ORG, SETTINGS_APP).value(
                FONT_FAMILY_KEY, FONT_FAMILY_DEFAULT
            )
            or FONT_FAMILY_DEFAULT
        )
        current_size = saved_size()
        if (
            current_font != baseline["font"]
            or current_size != baseline["font_size"]
        ):
            # The stylesheet carries the size: rebuild it even when only
            # the size moved (the family path alone leaves QSS sizes stale).
            baseline["font"] = apply_font(current_font)
            apply_theme()
            baseline["font_size"] = current_size

    def show_extensions_dialog(self) -> None:
        ExtensionsDialog(
            self._store,
            parent=self,
            on_extensions_changed=self._on_extensions_changed,
            active_check=lambda: self.canvasView.is_run_active(),
        ).show()

    def _on_graph_changed(self) -> None:
        self._set_dirty(True)

    def _set_dirty(self, dirty: bool) -> None:
        if self._dirty == dirty:
            return
        self._dirty = dirty
        self._update_title()

    def _update_title(self) -> None:
        title = "LynceusScan"
        if self._project_path:
            title += f" - {Path(self._project_path).stem}"
        elif self._template_name:
            title += f" - {self._template_name} ({t('template')})"
        if self._dirty:
            title += "*"
        self.setWindowTitle(title)

    def new_project(self) -> None:
        if self.canvasView.is_run_active():
            QMessageBox.information(
                self, t("Run in progress"),
                t("Cancel or wait for the current run before starting a new project."),
            )
            return
        if not self._confirm_discard():
            return
        self.canvasView.clear_graph()
        self._project_path = None
        self._template_name = None
        self._sync_session_root()
        self._set_dirty(False)

    def open_project(self) -> None:
        if self.canvasView.is_run_active():
            QMessageBox.information(
                self, t("Run in progress"),
                t("Cancel or wait for the current run before opening a project."),
            )
            return
        if not self._confirm_discard():
            return
        start_dir = (
            str(Path(self._project_path).parent)
            if self._project_path
            else self._last_project_dir()
        )
        path, _ = QFileDialog.getOpenFileName(
            self, t("Open Project"), start_dir, project_io.FILE_FILTER
        )
        if path:
            self._open_path(path)

    def _open_path(self, path: str, from_template: bool = False) -> None:
        try:
            data = project_io.load_project(path)
            missing = project_io.validate_nodes(data)
        except project_io.ProjectError as exc:
            QMessageBox.critical(self, t("Open Project"), str(exc))
            return
        if missing:
            QMessageBox.information(
                self,
                t("Open Project"),
                t(
                    "The project contains node(s) whose extension is missing "
                    "or disabled:\n  {names}\n\nThey were kept but won't run "
                    "until you enable or re-import the extension, or remove "
                    "the node(s)."
                ).format(names=", ".join(missing)),
            )
        warnings = self.canvasView.restore_graph(data)
        if from_template:
            # Untitled copy: the template file itself is never overwritten;
            # the first save prompts Save As.
            self._project_path = None
            meta = data.get(templates_io.TEMPLATE_META_KEY) or {}
            self._template_name = str(
                meta.get("name") or data.get("name") or Path(path).stem
            )
        else:
            self._project_path = str(Path(path).resolve())
            self._template_name = None
            self._add_recent(self._project_path)
        self._sync_session_root()
        self._set_dirty(False)
        if warnings:
            self._on_status_message(
                " | ".join(t(w) for w in warnings), "warning"
            )

    def save_project(self) -> bool:
        if not self._project_path:
            return self.save_project_as()
        if not self._write_project(self._project_path):
            return False
        self._set_dirty(False)
        self._add_recent(self._project_path)
        return True

    def save_project_as(self) -> bool:
        suggested = self._project_path or str(
            Path(self._last_project_dir())
            / (
                templates_io.template_file_name(self._template_name)
                if self._template_name
                else "project.lynx"
            )
        )
        path, _ = QFileDialog.getSaveFileName(
            self, t("Save Project As"), suggested, project_io.FILE_FILTER
        )
        if not path:
            return False
        if not self._write_project(path):
            return False
        self._project_path = str(Path(path).resolve())
        self._template_name = None
        self._sync_session_root()
        self._set_dirty(False)
        self._add_recent(self._project_path)
        return True

    def save_project_copy(self) -> bool:
        suggested = self._project_path or str(
            Path(self._last_project_dir()) / "project.lynx"
        )
        path, _ = QFileDialog.getSaveFileName(
            self, t("Save a Copy"), suggested, project_io.FILE_FILTER
        )
        if not path:
            return False
        ok = self._write_project(path)
        if ok:
            self._add_recent(str(Path(path).resolve()))
        return ok

    def _write_project(self, path: str) -> bool:
        data = self.canvasView.serialize_graph()
        data["name"] = Path(path).stem
        try:
            project_io.save_project(path, data)
        except OSError as exc:
            QMessageBox.critical(
                self,
                t("Save Project"),
                t("Could not save project: {exc}").format(exc=exc),
            )
            return False
        return True

    def _confirm_discard(self) -> bool:
        """True if we can continue (no changes, saved or discarded)."""
        if not self._dirty:
            return True
        ret = QMessageBox.warning(
            self,
            t("Unsaved Changes"),
            t("The project has unsaved changes. Save them?"),
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
        )
        if ret == QMessageBox.StandardButton.Save:
            return self.save_project()
        return ret == QMessageBox.StandardButton.Discard

    def closeEvent(self, event) -> None:
        if not self._confirm_discard():
            event.ignore()
            return
        runs = getattr(self, "runsPanel", None)
        shutdown = getattr(runs, "shutdown", None)
        if callable(shutdown):
            shutdown()
        super().closeEvent(event)

    # ---------- project: sessions by folder ----------

    def _sync_session_root(self) -> None:
        if self._project_path:
            slug = project_io.session_dir_name(Path(self._project_path).stem)
            self.canvasView.session_root = default_output_root() / slug
        elif self._template_name:
            # Template runs group under the template name (untitled copies
            # would otherwise pile up in _default_project/).
            slug = project_io.session_dir_name(self._template_name)
            self.canvasView.session_root = default_output_root() / slug
        else:
            self.canvasView.session_root = None
        runs = getattr(self, "runsPanel", None)
        if runs is not None:
            runs.set_session_root(self.canvasView.session_root)

    # ---------- project: recent files ----------

    def _last_project_dir(self) -> str:
        settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
        last = settings.value("last_project_dir", "", type=str)
        return last if last and Path(last).is_dir() else ""

    def _load_recent(self) -> list[str]:
        settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
        paths = settings.value(RECENT_KEY, []) or []
        if isinstance(paths, str):
            paths = [paths]
        return [p for p in paths if isinstance(p, str)]

    def _save_recent(self) -> None:
        settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
        settings.setValue(RECENT_KEY, self._recent_paths)

    def _recent_limit(self) -> int:
        try:
            limit = int(
                QSettings(SETTINGS_ORG, SETTINGS_APP).value(
                    RECENT_LIMIT_KEY, RECENT_LIMIT_DEFAULT
                )
            )
        except (TypeError, ValueError):
            limit = RECENT_LIMIT_DEFAULT
        return min(50, max(1, limit))

    def _add_recent(self, path: str) -> None:
        norm = str(Path(path).resolve())
        parent = str(Path(norm).parent)
        QSettings(SETTINGS_ORG, SETTINGS_APP).setValue("last_project_dir", parent)
        rest = [p for p in self._recent_paths if p.lower() != norm.lower()]
        self._recent_paths = ([norm] + rest)[: self._recent_limit()]
        self._save_recent()
        self._rebuild_recent_menu()

    def _rebuild_recent_menu(self) -> None:
        if self._recent_menu is None:
            return
        self._recent_menu.clear()
        self._recent_menu.menuAction().setProperty("origText_en", "Open Recent")
        for path in self._recent_paths:
            exists = Path(path).exists()
            label = f"{Path(path).stem}  ({Path(path).parent})"
            action = self._recent_menu.addAction(label)
            if not exists:
                action.setText(f"{label} - missing")
            action.setEnabled(exists)
            action.triggered.connect(lambda _=False, p=path: self._open_recent(p))
        self._recent_menu.addSeparator()
        clear = self._recent_menu.addAction(t("Clear List"))
        clear.setProperty("origText_en", "Clear List")
        clear.triggered.connect(self._clear_recent)

    def _clear_recent(self) -> None:
        self._recent_paths = []
        self._save_recent()
        self._rebuild_recent_menu()

    def _open_recent(self, path: str) -> None:
        if not self._confirm_discard():
            return
        self._open_path(path)

    # ---------- window buttons ----------

    def _maybe_init_right_splitter(self, retries: int = 8) -> None:
        """Split inspector/outputs 50/50 once, when the layout is real.

        setGeometry() does not lay out synchronously: reading the splitter
        height too early yields the small pre-maximize value and the floors
        below would eat the whole panel. So this validates against the
        available screen height and retries (120 ms, capped) until the
        window is truly maximized. A user drag also marks it done.
        """
        if self._right_split_init_done:
            return
        splitter = getattr(self, "rightSplitter", None)
        if splitter is None or splitter.count() < 2:
            return
        total = splitter.height()
        try:
            avail = self.screen().availableGeometry().height()
        except (AttributeError, RuntimeError):
            avail = 0
        if total <= 0 or (avail > 0 and total < avail - 150):
            if retries > 0:
                QTimer.singleShot(
                    120,
                    lambda: self._maybe_init_right_splitter(retries - 1),
                )
            return
        outputs_h = max(260, total // 2)
        inspector_h = total - outputs_h
        if inspector_h < 220 and total > 480:
            inspector_h = 220
            outputs_h = total - inspector_h
        splitter.setSizes([inspector_h, outputs_h])
        self._right_split_init_done = True

    def _install_window_controls(self) -> None:
        container = QWidget(self.menubar)
        container.setObjectName("windowControls")
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.btn_minimize = QToolButton(container)
        self.btn_minimize.setObjectName("btnMinimize")
        self.btn_minimize.setText("\u2013")
        self.btn_minimize.setFixedSize(32, 26)
        self.btn_minimize.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_minimize.clicked.connect(self.showMinimized)

        self.btn_maximize = QToolButton(container)
        self.btn_maximize.setObjectName("btnMaximize")
        self.btn_maximize.setFixedSize(32, 26)
        self.btn_maximize.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_maximize.clicked.connect(self._toggle_maximize)

        self.btn_close = QToolButton(container)
        self.btn_close.setObjectName("btnClose")
        self.btn_close.setText("\u2715")
        self.btn_close.setFixedSize(32, 26)
        self.btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_close.clicked.connect(self.close)

        layout.addWidget(self.btn_minimize)
        layout.addWidget(self.btn_maximize)
        layout.addWidget(self.btn_close)

        self.menubar.setCornerWidget(container, Qt.Corner.TopRightCorner)
        self._update_maximize_glyph()

    def _toggle_maximize(self) -> None:
        if self._window_maximized:
            self.restore_window()
        else:
            self.maximize_window()

    def maximize_window(self) -> None:
        self._normal_geometry = self.geometry()
        screen_geometry = self.screen().availableGeometry()
        self.setGeometry(screen_geometry)
        self._window_maximized = True
        self._update_maximize_glyph()
        self._maybe_init_right_splitter()

    def restore_window(self) -> None:
        if self._normal_geometry is not None and self._normal_geometry.isValid():
            self.setGeometry(self._normal_geometry)
        self._window_maximized = False
        self._update_maximize_glyph()

    def _update_maximize_glyph(self) -> None:
        self.btn_maximize.setText(
            "\u2750" if self._window_maximized else "\u25a1"
        )

    def changeEvent(self, event) -> None:
        if (
            event.type() == QEvent.Type.WindowStateChange
            and hasattr(self, "btn_maximize")
        ):
            self._update_maximize_glyph()
        super().changeEvent(event)

    # ---------- edge-resize cursor tracking ----------

    def _enable_edge_tracking(self) -> None:
        for widget in (
            self,
            self.menubar,
            self.centralwidget,
            self.statusbar,
            self.nodesPanel,
            self.canvasPanel,
        ):
            widget.setMouseTracking(True)

    # ---------- metrics display ----------

    _COLOR_NORMAL = "#6ec87a"
    _COLOR_WARN = "#d4a94e"
    _COLOR_CRIT = "#d46a6a"
    _COLOR_IDLE = "#8a93a6"
    _KIND_COLORS = {
        "info": "#7aa8d4",
        "action": "#d4a94e",
        "warning": "#d4a94e",
        "success": "#6ec87a",
        "error": "#d46a6a",
    }
    # Transient kinds only replace a message with equal-or-lower priority, so
    # warnings/errors stay visible instead of being covered by the final
    # "Pipeline finished" success message.
    _KIND_PRIORITY = {
        "idle": 0,
        "info": 1,
        "action": 1,
        "success": 1,
        "warning": 2,
        "error": 3,
    }

    def _init_status_dot(self) -> None:
        """Persistent dot in the status bar corner plus a message label
        colored by category.

        The label yields to the metric segments: it keeps the full text
        but only shows what fits (elided), so a long message can never
        stretch the frameless window beyond the screen.
        """
        self._status_kind = "idle"
        self._status_dot = QLabel("")
        self._status_dot.setFixedSize(10, 10)
        self._status_dot.setObjectName("statusDot")
        self._apply_dot_color(self._COLOR_IDLE)
        self._status_text = QLabel("")
        self._status_text.setObjectName("statusMessage")
        self._status_text.setStyleSheet(f"color: {self._COLOR_IDLE};")
        self._status_text.setVisible(False)
        self._status_text.setMinimumWidth(0)
        self._status_text.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self._status_full = ""
        self.statusbar.addWidget(self._status_dot)
        self.statusbar.addWidget(self._status_text)
        self.statusbar.installEventFilter(self)

    def _apply_dot_color(self, color: str) -> None:
        self._status_dot.setStyleSheet(
            f"background-color: {color}; border-radius: 5px; padding: 0px;"
        )

    def _on_status_message(self, message: str, kind: str = "info") -> None:
        priority = self._KIND_PRIORITY.get(kind, 1)
        if priority < self._KIND_PRIORITY.get(self._status_kind, 0):
            return  # keep the higher-priority message visible
        self._status_kind = kind
        color = self._KIND_COLORS.get(kind, self._COLOR_IDLE)
        self._apply_dot_color(color)
        translated = t(message)
        if translated:
            # Stored as received (callers pre-translate, so a language
            # switch keeps the old text like before; only the width adapts).
            self._status_full = translated
            self._status_text.setToolTip(translated)
            self._status_text.setStyleSheet(f"color: {color};")
            self._status_text.setVisible(True)
            self._refresh_status_elision()
        else:
            self._status_full = ""
            self._status_text.setVisible(False)

    def _refresh_status_elision(self) -> None:
        """Show the status message elided to the label's current width."""
        if not self._status_text.isVisible() or not self._status_full:
            return
        available = max(50, self._status_text.width() - 4)
        self._status_text.setText(
            self._status_text.fontMetrics().elidedText(
                self._status_full, Qt.TextElideMode.ElideRight, available
            )
        )

    def _init_runs_panel(self) -> None:
        """Configures the session browser on the bottom-left panel."""
        runs = getattr(self, "runsPanel", None)
        if runs is None:
            return
        runs.set_active_check(
            lambda p: self.canvasView._pipeline is not None
            and self.canvasView._pipeline.session_dir == p
            and self.canvasView._pipeline.state in ("running", "paused", "cancelling")
        )
        runs.set_run_active_check(self.canvasView.is_run_active)
        runs.set_session_root(self.canvasView.session_root)
        runs.session_deleted.connect(self._on_session_deleted)
        runs.session_preview_requested.connect(self._on_session_preview)
        self.canvasView.pipeline_finished.connect(runs.refresh)
        self.canvasView.pipeline_cancelled.connect(runs.refresh)

    def _on_session_deleted(self, path) -> None:
        """Clears the output panel ONLY if the deleted session is the one being
        displayed (the last run processed, session_dir)."""
        pipe = getattr(self.canvasView, "_pipeline", None)
        if pipe is not None and pipe.session_dir is not None:
            if Path(pipe.session_dir) == Path(path):
                self.outputDock.clear_outputs()

    def _on_session_preview(self, path) -> None:
        """Load a past session's surviving products into the gallery.

        Bridge mode: read-only reconstruction (no run, no copies); swept or
        deleted files simply offer no preview via the ghost rule.
        """
        from lynceus.ui.outputs_model import products_from_session

        products = products_from_session(Path(path))
        if not products:
            self._on_status_message(
                t("No previewable products in this session"), "warning"
            )
            return
        self.outputDock.set_products(products)

    def _on_node_preview(self, path: str, node_name: str) -> None:
        """Open a floating preview from a body double-click on a node.

        Bridge mode, same as File → Open Preview: the file is visualized in
        place with no run involved.
        """
        if not self.outputDock.products.open_external_file(path, node_name):
            self._on_status_message(
                t("Cannot preview '{name}': no viewer for this file type").format(
                    name=Path(path).name
                ),
                "warning",
            )

    def _init_metrics_display(self) -> None:
        self._seg_timer = self._create_segment()
        self._seg_ram = self._create_segment()
        self._seg_cpu = self._create_segment()
        self._seg_tasks = self._create_segment()
        self._seg_pool = self._create_segment()

        for seg in (
            self._seg_timer,
            self._seg_ram,
            self._seg_cpu,
            self._seg_tasks,
            self._seg_pool,
        ):
            self.statusbar.addPermanentWidget(seg)
            seg.setVisible(False)

        self._metrics_timer = QTimer(self)
        self._metrics_timer.setInterval(1000)
        self._metrics_timer.timeout.connect(self._update_metrics_display)

        self._metrics_start_time = 0.0
        self._tasks_done = 0
        self._tasks_total = 0

        self.canvasView.pipeline_started.connect(self._on_pipeline_started)
        self.canvasView.pipeline_finished.connect(self._on_pipeline_finished_metrics)
        self.canvasView.pipeline_progress.connect(self._on_pipeline_progress)

    def _create_segment(self) -> QLabel:
        label = QLabel("")
        label.setFont(QFont("Consolas", 8))
        label.setStyleSheet(f"color: {self._COLOR_IDLE}; padding: 0 6px;")
        return label

    def _set_segment(self, label: QLabel, text: str, color: str) -> None:
        label.setText(text)
        label.setStyleSheet(f"color: {color}; padding: 0 6px;")
        label.setVisible(bool(text.strip()))

    @staticmethod
    def _color_for_threshold(
        ratio: float,
        green: str = "#6ec87a",
        amber: str = "#d4a94e",
        red: str = "#d46a6a",
    ) -> str:
        if ratio >= 0.80:
            return red
        if ratio >= 0.60:
            return amber
        return green

    def _on_pipeline_started(self) -> None:
        self._status_kind = "idle"
        self._metrics_start_time = _time.monotonic()
        self._tasks_done = 0
        self._tasks_total = 0
        self._seg_timer.setVisible(True)
        self._seg_ram.setVisible(True)
        self._seg_cpu.setVisible(True)
        self._metrics_timer.start()
        self._update_metrics_display()

    def _on_pipeline_finished_metrics(self, *args) -> None:
        self._metrics_timer.stop()
        self._update_metrics_display(final=True)
        # Final segments stay visible until the next run resets them: hiding
        # them on a timer meant returning users found no trace of the run.

    def _on_pipeline_progress(self, done: int, total: int) -> None:
        self._tasks_done = done
        self._tasks_total = total

    def _update_metrics_display(self, final: bool = False) -> None:
        elapsed = (
            _time.monotonic() - self._metrics_start_time
            if self._metrics_start_time
            else 0
        )
        mins, secs = divmod(int(elapsed), 60)

        # --- Timer ---
        timer_color = self._COLOR_NORMAL if not final else self._COLOR_IDLE
        self._set_segment(self._seg_timer, f" {mins:02d}:{secs:02d} ", timer_color)

        # --- RAM ---
        try:
            gui_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception:
            gui_mb = 0.0
        total_ram_mb = psutil.virtual_memory().total / (1024 * 1024)

        if final:
            metrics = getattr(self.canvasView._pipeline, "last_metrics", {})
            peak = metrics.get("peak_memory_mb", gui_mb)
            ram_ratio = peak / total_ram_mb if total_ram_mb else 0.0
            self._set_segment(
                self._seg_ram, f" Peak {peak:.0f} MB ",
                self._color_for_threshold(ram_ratio),
            )
        else:
            ram_ratio = gui_mb / total_ram_mb if total_ram_mb else 0.0
            self._set_segment(
                self._seg_ram, f" RAM {gui_mb:.0f} MB ",
                self._color_for_threshold(ram_ratio),
            )

        # --- CPU (system) ---
        cpu_pct = psutil.cpu_percent(interval=None)
        self._set_segment(
            self._seg_cpu, f" CPU Sys {cpu_pct:.0f}% ",
            self._color_for_threshold(cpu_pct / 100.0),
        )

        # --- Tasks ---
        if self._tasks_total > 0:
            task_color = (
                self._COLOR_NORMAL
                if self._tasks_done >= self._tasks_total
                else self._COLOR_IDLE
            )
            self._set_segment(
                self._seg_tasks,
                f" {self._tasks_done}/{self._tasks_total} tasks ",
                task_color,
            )
        else:
            self._set_segment(self._seg_tasks, "", self._COLOR_IDLE)

        # --- Pool ---
        if final:
            metrics = getattr(self.canvasView._pipeline, "last_metrics", {})
            occ = metrics.get("occupancy", 0)
            self._set_segment(
                self._seg_pool, f" Pool {occ * 100:.0f}% ",
                self._color_for_threshold(occ),
            )
        else:
            self._set_segment(self._seg_pool, "", self._COLOR_IDLE)

    # ---------- move window from menubar ----------

    def _menubar_item_at(self, pos: QPoint):
        for action in self.menubar.actions():
            if self.menubar.actionGeometry(action).contains(pos):
                return action
        return None

    # ---------- edge resize ----------

    def _edge_at(self, global_pos: QPoint) -> str | None:
        rect = self.geometry()
        x = global_pos.x() - rect.x()
        y = global_pos.y() - rect.y()
        left = x <= RESIZE_MARGIN
        right = x >= rect.width() - RESIZE_MARGIN
        top = y <= RESIZE_MARGIN
        bottom = y >= rect.height() - RESIZE_MARGIN
        if top and left:
            return "topleft"
        if top and right:
            return "topright"
        if bottom and left:
            return "bottomleft"
        if bottom and right:
            return "bottomright"
        if left:
            return "left"
        if right:
            return "right"
        if top:
            return "top"
        if bottom:
            return "bottom"
        return None

    def _apply_resize(self, global_pos: QPoint) -> None:
        rect = self._resize_geometry
        edge = self._resize_edge
        dx = global_pos.x() - self._resize_origin.x()
        dy = global_pos.y() - self._resize_origin.y()
        new = QRect(rect)
        if "left" in edge:
            new.setLeft(rect.left() + dx)
        if "right" in edge:
            new.setRight(rect.right() + dx)
        if "top" in edge:
            new.setTop(rect.top() + dy)
        if "bottom" in edge:
            new.setBottom(rect.bottom() + dy)
        if (
            new.width() >= MIN_WINDOW_SIZE[0]
            and new.height() >= MIN_WINDOW_SIZE[1]
        ):
            self.setGeometry(new)

    # ---------- global event filter ----------

    def eventFilter(self, obj, event) -> bool:
        if obj is getattr(self, "statusbar", None):
            if event.type() == QEvent.Type.Resize:
                self._refresh_status_elision()
            return super().eventFilter(obj, event)
        if not isinstance(obj, QWidget) or not self.isAncestorOf(obj):
            return super().eventFilter(obj, event)
        if not isinstance(event, QMouseEvent):
            return super().eventFilter(obj, event)

        etype = event.type()
        buttons = event.buttons()

        if etype == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            if isinstance(obj, QAbstractButton):
                return super().eventFilter(obj, event)
            global_pos = event.globalPosition().toPoint()
            if not self._window_maximized:
                edge = self._edge_at(global_pos)
                if edge:
                    self._resize_edge = edge
                    self._resize_origin = global_pos
                    self._resize_geometry = self.geometry()
                    event.accept()
                    return True
                if obj is self.menubar:
                    if self._menubar_item_at(event.position().toPoint()) is None:
                        self._dragging_window = True
                        self._drag_offset = (
                            global_pos - self.frameGeometry().topLeft()
                        )
                        event.accept()
                        return True

        elif etype == QEvent.Type.MouseMove:
            global_pos = event.globalPosition().toPoint()
            if self._resize_edge and buttons & Qt.MouseButton.LeftButton:
                self._apply_resize(global_pos)
                event.accept()
                return True
            if self._dragging_window and buttons & Qt.MouseButton.LeftButton:
                self.move(global_pos - self._drag_offset)
                event.accept()
                return True
            if not self._window_maximized:
                edge = self._edge_at(global_pos)
                cursor = _EDGE_CURSORS.get(edge)
                if cursor is not None:
                    self.setCursor(cursor)
                else:
                    self.unsetCursor()

        elif etype == QEvent.Type.MouseButtonDblClick:
            if (
                obj is self.menubar
                and self._menubar_item_at(event.position().toPoint()) is None
            ):
                self._toggle_maximize()
                event.accept()
                return True

        elif etype == QEvent.Type.MouseButtonRelease:
            self._dragging_window = False
            self._resize_edge = None

        return super().eventFilter(obj, event)
