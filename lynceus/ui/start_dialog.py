# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Start window shown before the main window (Qt UI).

Four ways in: a blank canvas, an existing project, a recent project, or a
project template (base workflows with descriptive metadata). Closing the
dialog without choosing falls back to a blank canvas.

``TemplateListWidget`` is shared with the in-app "New From Template" picker;
``TemplateMetaDialog`` collects the metadata for "Save as Template".
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from lynceus import project as project_io
from lynceus import templates as templates_io
from lynceus.plugins.locale import t
from lynceus.ui.settings_keys import SETTINGS_APP, SETTINGS_ORG


class TemplateListWidget(QWidget):
    """Template cards: list on top, selected-template details below."""

    selection_changed = Signal()

    def __init__(self, templates: list, parent=None):
        super().__init__(parent)
        self._templates = list(templates)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.list = QListWidget(self)
        self.list.setObjectName("templateList")
        for info in self._templates:
            label = info.name
            if info.source == "bundled":
                label += f"  [{t('Built-in')}]"
            else:
                label += f"  [{t('My Templates')}]"
            if not info.valid:
                label += f"  ({t('Invalid')})"
            item = QListWidgetItem(label)
            item.setToolTip(info.description or info.name)
            item.setData(Qt.ItemDataRole.UserRole, info.path)
            if not info.valid:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            self.list.addItem(item)
        self.list.currentItemChanged.connect(lambda *_: self._on_selected())
        layout.addWidget(self.list, 1)

        self.details = QTextEdit(self)
        self.details.setObjectName("templateDetails")
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(130)
        layout.addWidget(self.details)
        for row in range(self.list.count()):
            if self.list.item(row).flags() & Qt.ItemFlag.ItemIsEnabled:
                self.list.setCurrentRow(row)
                break
        self._on_selected()

    def _on_selected(self) -> None:
        info = self.selected()
        if info is None:
            self.details.setPlainText(t("Select a template to see its details."))
        elif not info.valid:
            self.details.setPlainText(f"{info.name}\n{info.error}")
        else:
            lines = [info.name]
            if info.description:
                lines += ["", info.description]
            meta = []
            if info.author:
                meta.append(f"{t('Author')}: {info.author}")
            if info.version:
                meta.append(f"{t('Version')}: {info.version}")
            if info.tags:
                meta.append(f"{t('Tags')}: {', '.join(info.tags)}")
            meta.append(f"{t('Nodes')}: {info.node_count}")
            lines += ["", " | ".join(meta)]
            self.details.setPlainText("\n".join(lines))
        self.selection_changed.emit()

    def selected(self):
        """The selected valid TemplateInfo (None if none/invalid)."""
        item = self.list.currentItem()
        if item is None:
            return None
        for info in self._templates:
            if info.path == item.data(Qt.ItemDataRole.UserRole):
                return info if info.valid else None
        return None


class StartDialog(QDialog):
    """Launcher: blank canvas, open project, recents, templates."""

    def __init__(
        self,
        recent_paths: list[str],
        templates: list,
        show_on_startup: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("LynceusScan")
        self.setModal(True)
        self.resize(880, 560)

        self._action: tuple[str, str | None, bool] = ("blank", None, False)

        root = QVBoxLayout(self)
        title = QLabel("<b>LynceusScan</b>")
        title.setObjectName("startTitle")
        root.addWidget(title)
        root.addWidget(QLabel(t("Where do you want to start?")))

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(self._build_left(recent_paths))
        splitter.addWidget(self._build_right(templates))
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        root.addWidget(splitter, 1)

        bottom = QHBoxLayout()
        self._show_box = QCheckBox(t("Show on startup"), self)
        self._show_box.setChecked(show_on_startup)
        bottom.addWidget(self._show_box)
        bottom.addStretch(1)
        close = QPushButton(t("Close"), self)
        close.clicked.connect(self.reject)
        bottom.addWidget(close)
        root.addLayout(bottom)

    # -- left: actions + recents -------------------------------------------

    def _build_left(self, recent_paths: list[str]) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        new_btn = QPushButton(t("New Blank Canvas"), panel)
        new_btn.setObjectName("startPrimary")
        new_btn.clicked.connect(lambda: self._accept("blank", None, False))
        layout.addWidget(new_btn)

        open_btn = QPushButton(t("Open Project..."), panel)
        open_btn.clicked.connect(self._choose_file)
        layout.addWidget(open_btn)

        layout.addWidget(QLabel(t("Recent Projects"), panel))
        self._recent_list = QListWidget(panel)
        self._recent_list.setObjectName("recentList")
        shown = 0
        for path in recent_paths:
            if not Path(path).is_file():
                continue
            item = QListWidgetItem(
                f"{Path(path).stem}  ({Path(path).parent})"
            )
            item.setToolTip(path)
            item.setData(Qt.ItemDataRole.UserRole, path)
            self._recent_list.addItem(item)
            shown += 1
        if shown == 0:
            empty = QListWidgetItem(t("No recent projects"))
            empty.setFlags(empty.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            self._recent_list.addItem(empty)
        self._recent_list.itemActivated.connect(self._open_recent_item)
        layout.addWidget(self._recent_list, 1)
        return panel

    # -- right: templates ---------------------------------------------------

    def _build_right(self, templates: list) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 0, 0, 0)
        layout.addWidget(QLabel(t("Templates"), panel))
        if templates:
            self._template_list = TemplateListWidget(templates, panel)
            layout.addWidget(self._template_list, 1)
            use_btn = QPushButton(t("Use Template"), panel)
            use_btn.setObjectName("startPrimary")
            use_btn.clicked.connect(self._use_template)
            self._template_list.selection_changed.connect(
                lambda: use_btn.setEnabled(
                    self._template_list.selected() is not None
                )
            )
            use_btn.setEnabled(self._template_list.selected() is not None)
            layout.addWidget(use_btn)
        else:
            hint = QLabel(
                t(
                    "No templates yet. Build a workflow and save it with "
                    "File > Save as Template."
                ),
                panel,
            )
            hint.setObjectName("hintLabel")
            hint.setWordWrap(True)
            hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addStretch(1)
            layout.addWidget(hint)
            layout.addStretch(1)
            self._template_list = None
        return panel

    # -- actions ------------------------------------------------------------

    @property
    def action(self) -> tuple[str, str | None, bool]:
        """("blank"|"open", path or None, from_template)."""
        return self._action

    @property
    def show_on_startup(self) -> bool:
        return self._show_box.isChecked()

    def _accept(self, kind: str, path: str | None, from_template: bool) -> None:
        self._action = (kind, path, from_template)
        self.accept()

    def _choose_file(self) -> None:
        settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
        start_dir = settings.value("last_project_dir", "", type=str) or ""
        path, _ = QFileDialog.getOpenFileName(
            self, t("Open Project"), start_dir, project_io.FILE_FILTER
        )
        if path:
            self._accept("open", path, False)

    def _open_recent_item(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        if path and Path(path).is_file():
            self._accept("open", str(path), False)

    def _use_template(self) -> None:
        info = (
            self._template_list.selected() if self._template_list else None
        )
        if info is not None:
            self._accept("open", info.path, True)


class TemplatePickerDialog(QDialog):
    """In-app "New From Template" picker (shares TemplateListWidget)."""

    def __init__(self, templates: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("New From Template"))
        self.setModal(True)
        self.resize(640, 480)
        layout = QVBoxLayout(self)
        self._list = TemplateListWidget(templates, self)
        layout.addWidget(self._list, 1)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Open
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Open).setText(
            t("Use Template")
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self) -> None:
        if self._list.selected() is None:
            return
        self.accept()

    def selected(self):
        return self._list.selected()


class TemplateMetaDialog(QDialog):
    """Metadata editor for "Save as Template"."""

    def __init__(self, suggested_name: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("Save as Template"))
        self.setModal(True)
        self.resize(480, 320)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self._name = QLineEdit(suggested_name, self)
        self._description = QLineEdit(self)
        self._author = QLineEdit(self)
        self._version = QLineEdit("1.0", self)
        self._tags = QLineEdit(self)
        self._tags.setPlaceholderText(t("comma, separated, tags"))
        form.addRow(t("Name"), self._name)
        form.addRow(t("Description"), self._description)
        form.addRow(t("Author"), self._author)
        form.addRow(t("Version"), self._version)
        form.addRow(t("Tags"), self._tags)
        layout.addLayout(form)
        dest = QLabel(str(templates_io.save_target_dir()), self)
        dest.setObjectName("hintLabel")
        dest.setWordWrap(True)
        layout.addWidget(QLabel(t("Save to:"), self))
        layout.addWidget(dest)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self) -> None:
        if not self._name.text().strip():
            self._name.setFocus()
            return
        self.accept()

    def meta(self) -> dict:
        return {
            "name": self._name.text().strip(),
            "description": self._description.text().strip(),
            "author": self._author.text().strip(),
            "version": self._version.text().strip(),
            "tags": self._tags.text().strip(),
        }
