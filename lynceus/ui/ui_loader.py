# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
import importlib
import re

from PySide6.QtUiTools import QUiLoader


def _parse_custom_widgets(ui_path):
    """Read the <customwidgets> section and return {class_name: module_path}."""
    with open(ui_path, encoding="utf-8") as fh:
        text = fh.read()
    mapping = {}
    section = re.search(r"<customwidgets>.*?</customwidgets>", text, re.S)
    if not section:
        return mapping
    for block in re.findall(r"<customwidget>.*?</customwidget>", section.group(0), re.S):
        class_m = re.search(r"<class>([^<]+)</class>", block)
        header_m = re.search(r"<header>([^<]+)</header>", block)
        if class_m and header_m:
            mapping[class_m.group(1)] = header_m.group(1)
    return mapping


def _is_assignable(class_name, host):
    """True if ``class_name`` (a Qt widget class) can be the base of ``host``."""
    meta = host.metaObject()
    while meta is not None:
        if meta.className() == class_name:
            return True
        meta = meta.superClass()
    return False


class _InPlaceLoader(QUiLoader):
    """QUiLoader that reuses an existing widget as the UI root and resolves the
    promoted custom classes declared in the ``.ui`` file.

    PySide6 does not ship PyQt6's ``uic.loadUi(path, widget)`` helper, so this
    reproduces its behaviour:
      - the root widget that matches the host instance is the host itself;
      - promoted widgets (e.g. ``NodeLibrary``) are realised from their headers
        instead of their base Qt class.
    """

    def __init__(self, host, custom_widgets):
        super().__init__()
        self._host = host
        self._custom = custom_widgets

    def createWidget(self, class_name, parent=None, name=""):
        if parent is None and self._host is not None:
            if _is_assignable(class_name, self._host):
                self._host.setObjectName(name or self._host.objectName())
                return self._host
        module_path = self._custom.get(class_name)
        if module_path is not None:
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            instance = cls(parent)
            if name:
                instance.setObjectName(name)
            return instance
        return super().createWidget(class_name, parent, name)


def load_ui_inplace(ui_path, host):
    """Populate ``host`` from a ``.ui`` file, mirroring PyQt6's
    ``uic.loadUi(ui_path, host)``. Returns ``host``."""
    custom_widgets = _parse_custom_widgets(ui_path)
    loader = _InPlaceLoader(host, custom_widgets)
    loaded = loader.load(ui_path)
    if loaded is not None and loaded is not host:
        children = list(loaded.findChildren(object))
        for child in children:
            child.setParent(host)
        del loaded
    return host