# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Right panel: per-node configuration inspector.

The form is generated from config_schema of the module; values are stored
per instance (iid) and the engine applies them as overrides over the
defaults when compiling the DAG (build_dag).
"""

from PySide6.QtCore import QPointF, Qt, QEvent, Signal
from PySide6.QtGui import QPainter, QPainterPath
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStyle,
    QStyleOptionSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from lynceus.nodes._variants import STRATEGY_FIELD, strategy_options
from lynceus.plugins.locale import t
from lynceus.processing.steps import (
    discover_node_capabilities,
    effective_node_caps,
)

_INT32_MAX = 2**31 - 1
_FLOAT_MAX = 1e9


def _value_differs(value, default) -> bool:
    """Whether a field value differs from its schema default.

    Floats are compared with a small tolerance (the double spin box only
    displays a limited number of decimals, so a raw comparison could flag
    spurious differences).
    """
    if default is None:
        return False
    try:
        if isinstance(value, float) or isinstance(default, float):
            return abs(float(value) - float(default)) > 1e-6
        return value != default
    except (TypeError, ValueError):
        return value != default


def _update_spin_cursor(spin, event) -> None:
    """Set the pointing-hand cursor over the up/down buttons of a spin box.

    The spin buttons are style-drawn sub-controls (not real child widgets),
    so we locate them via QStyle.subControlRect() and react to the hover
    position. Over the editable area the normal I-beam cursor is kept.
    """
    etype = event.type()
    if etype in (
        QEvent.Type.MouseMove,
        QEvent.Type.HoverMove,
        QEvent.Type.Enter,
    ):
        opt = QStyleOptionSpinBox()
        spin.initStyleOption(opt)
        up = spin.style().subControlRect(
            QStyle.ComplexControl.CC_SpinBox,
            opt,
            QStyle.SubControl.SC_SpinBoxUp,
            spin,
        )
        down = spin.style().subControlRect(
            QStyle.ComplexControl.CC_SpinBox,
            opt,
            QStyle.SubControl.SC_SpinBoxDown,
            spin,
        )
        pos = event.position().toPoint()
        if up.contains(pos) or down.contains(pos):
            spin.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            spin.setCursor(Qt.CursorShape.IBeamCursor)
    elif etype == QEvent.Type.Leave:
        spin.setCursor(Qt.CursorShape.IBeamCursor)


def _paint_spin_arrows(spin, event) -> None:
    """Redraw the up/down arrows of a style-sheet spin box.

    Once a stylesheet defines ::up-button/::down-button (required for correct
    click hit-testing on Windows), Qt stops painting the default arrows, so
    we draw them on top of the transparent sub-control area.
    """
    opt = QStyleOptionSpinBox()
    spin.initStyleOption(opt)
    style = spin.style()
    up = style.subControlRect(
        QStyle.ComplexControl.CC_SpinBox,
        opt,
        QStyle.SubControl.SC_SpinBoxUp,
        spin,
    )
    down = style.subControlRect(
        QStyle.ComplexControl.CC_SpinBox,
        opt,
        QStyle.SubControl.SC_SpinBoxDown,
        spin,
    )
    if up.width() <= 0 or down.width() <= 0:
        return

    painter = QPainter(spin)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = opt.palette.text().color()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        _paint_arrow(painter, up, True)
        _paint_arrow(painter, down, False)
    finally:
        painter.end()


def _paint_arrow(painter: QPainter, rect, pointing_up: bool) -> None:
    """Fill a small triangle centered in `rect` (vertically centered too)."""
    cx = (rect.left() + rect.right()) / 2.0
    half = 3.0
    tip_off = 2.0
    cy = (rect.top() + rect.bottom()) / 2.0
    if pointing_up:
        tip = QPointF(cx, cy - tip_off)
        base_y = cy + tip_off
    else:
        tip = QPointF(cx, cy + tip_off)
        base_y = cy - tip_off
    path = QPainterPath()
    path.moveTo(tip)
    path.lineTo(QPointF(cx - half, base_y))
    path.lineTo(QPointF(cx + half, base_y))
    path.closeSubpath()
    painter.drawPath(path)


class _CursorSpinBox(QSpinBox):
    """QSpinBox that shows a pointing hand when hovering its buttons."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseTracking(True)

    def event(self, event) -> bool:
        _update_spin_cursor(self, event)
        return super().event(event)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        _paint_spin_arrows(self, event)


class _CursorDoubleSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox that shows a pointing hand when hovering its buttons."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseTracking(True)

    def event(self, event) -> bool:
        _update_spin_cursor(self, event)
        return super().event(event)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        _paint_spin_arrows(self, event)


class InspectorPanel(QWidget):
    """Inspector: configuration form for the selected node."""

    config_changed = Signal(str)  # iid of edited instance
    label_changed = Signal(str)  # iid of renamed instance
    strategy_change_requested = Signal(str, str)  # iid, variant key

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("inspectorPanel")
        self._item = None
        self._building = False

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(6)

        self._title = QLabel(t("Inspector"))
        self._title.setProperty("origText_en", "Inspector")
        self._title.setObjectName("inspectorTitle")
        root.addWidget(self._title)

        self._subtitle = QLabel("")
        self._subtitle.setObjectName("inspectorSubtitle")
        self._subtitle.setWordWrap(True)
        root.addWidget(self._subtitle)

        self._scroll = QScrollArea()
        self._scroll.setObjectName("inspectorScrollArea")
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._form_host = QWidget()
        self._form_host.setObjectName("inspectorFormHost")
        self._form = QFormLayout(self._form_host)
        self._form.setContentsMargins(0, 2, 6, 2)
        self._form.setHorizontalSpacing(12)
        self._form.setVerticalSpacing(7)
        self._form.setLabelAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._scroll.setWidget(self._form_host)
        root.addWidget(self._scroll, 1)

        self._footer = QLabel(t("Changes apply on the next run"))
        self._footer.setProperty(
            "origText_en", "Changes apply on the next run"
        )
        self._footer.setObjectName("panelHint")
        root.addWidget(self._footer)

        self._rebuild()

    # ---------- API ----------

    def set_node(self, item) -> None:
        """Show configuration for the node (None shows placeholder)."""
        if self._item is item:
            return
        self._item = item
        self._rebuild()

    def refresh(self) -> None:
        """Re-read current node state (e.g. after inline rename)."""
        self._rebuild()

    # ---------- form construction ----------

    def _rebuild(self) -> None:
        self._building = True
        # Capture the available width from the scroll viewport BEFORE clearing
        # the form. The viewport width reflects the current panel size and
        # avoids stale cached widths from the previous node (which would
        # cause wrapped labels to be sized for the wrong width, producing
        # visual clipping). Use geometry() to get the visible width
        # excluding the scrollbar area.
        vp_geo = self._scroll.viewport().geometry()
        scrollbar_w = (
            self._scroll.verticalScrollBar().geometry().width()
            if self._scroll.verticalScrollBar().isVisible() else 0
        )
        self._available_w = max(100, vp_geo.width() - scrollbar_w - 6 - 12)
        # Cap the form host to the visible width: widgetResizable cannot
        # shrink the host below its minimumSizeHint, and if some label
        # forces a large minimum the form overflows horizontally, sliding
        # the field column (and the numeric spin buttons) under the
        # vertical scrollbar.
        self._form_host.setMaximumWidth(max(100, vp_geo.width()))
        # Invalidate layout cache so QFormLayout recomputes from scratch
        # when the new widgets are added.
        self._form.invalidate()
        self._form_host.updateGeometry()
        try:
            while self._form.count():
                row = self._form.takeAt(0)
                widget = row.widget()
                if widget is not None:
                    widget.deleteLater()

            if self._item is None:
                self._title.setText(t("Inspector"))
                self._title.setProperty("origText_en", "Inspector")
                self._subtitle.setText("")
                placeholder = QLabel(
                    t("Select a node\nto inspect its configuration")
                )
                placeholder.setObjectName("inspectorPlaceholder")
                placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self._form.addRow(placeholder)
                return

            # Effective capabilities (strategy-aware): variant params
            # extend the base schema, so per-variant settings are visible.
            try:
                caps = effective_node_caps(
                    self._item.node_id(), self._item.pipeline_config()
                )
            except Exception:
                caps = discover_node_capabilities(self._item.node_id())
            schema = caps.get("config_schema") or {}
            # A strategy family without exposed options renders its variants
            # (the family base can't know them: variants are discovered).
            if isinstance(schema, dict) and STRATEGY_FIELD in schema:
                strategy_spec = dict(schema[STRATEGY_FIELD])
                if not strategy_spec.get("options"):
                    opts = strategy_options(self._item.node_id())
                    if opts:
                        strategy_spec["options"] = [
                            ("(none) — no strategy", ""),
                            *[(label, key) for key, label in opts],
                        ]
                schema = dict(schema)
                schema[STRATEGY_FIELD] = strategy_spec
            description = caps.get("description", "")
            module_id = self._item.node_id()
            title_orig = (
                self._item.label()
                if self._item.label()
                else self._item.node_name_en()
            )
            self._title.setText(t(title_orig))
            self._title.setProperty("origText_en", title_orig)
            self._subtitle.setText(
                f"{module_id.rsplit('.', 1)[-1]}  ·  #{self._item.iid}"
            )

            # Instance label
            label_edit = QLineEdit(self._item.label())
            label_edit.setPlaceholderText(self._item.node_name())
            label_edit.setClearButtonEnabled(True)
            label_edit.setToolTip(t("Custom display name (empty = node type)"))
            label_edit.editingFinished.connect(self._on_label_edited)
            label_orig = "Label"
            form_label = QLabel("Label")
            form_label.setProperty("origText_en", label_orig)
            self._form.addRow(form_label, label_edit)

            # Node description (always visible, not collapsible)
            if description:
                self._add_description_visible(description)

            if not schema:
                note_orig = "This node has no configurable options"
                note = QLabel("This node has no configurable options")
                note.setProperty("origText_en", note_orig)
                note.setObjectName("inspectorPlaceholder")
                note.setWordWrap(True)
                self._form.addRow(note)
                return

            current = self._item.pipeline_config()

            # Group schema items
            groups: dict[str, list[tuple[str, dict]]] = {}
            for key, spec in schema.items():
                group = spec.get("group", "")
                groups.setdefault(group, []).append((key, spec))

            for group_name, items in groups.items():
                # Group header
                if group_name:
                    header = QLabel(t(group_name).upper())
                    header.setProperty("origText_en", group_name)
                    header.setObjectName("inspectorGroupHeader")
                    header.setContentsMargins(0, 8, 0, 2)
                    header.setWordWrap(True)
                    header.setMinimumSize(0, 0)
                    self._form.addRow(header)

                for key, spec in items:
                    value = current.get(key, spec.get("default"))
                    is_advanced = spec.get("advanced", False)

                    # Field label
                    field_orig = str(key).replace("_", " ")
                    field_label = QLabel("")
                    field_label.setProperty("origText_en", field_orig)
                    field_label.setText(t(field_orig))
                    field_label.setWordWrap(True)
                    field_label.setMinimumSize(0, 0)
                    description_text = spec.get("description", "")
                    impact_text = spec.get("impact", "")
                    tooltip_parts = []
                    if description_text:
                        tooltip_parts.append(t(description_text))
                    if impact_text:
                        tooltip_parts.append(t(impact_text))
                    if tooltip_parts:
                        tooltip = "\n\n".join(tooltip_parts)
                        field_label.setToolTip(tooltip)

                    # Widget
                    widget = self._make_field(key, spec, value)
                    if tooltip_parts:
                        widget.setToolTip("\n\n".join(tooltip_parts))
                    self._form.addRow(field_label, widget)

                    # Impact sub-label (only for non-advanced fields)
                    if impact_text and not is_advanced:
                        impact_label = QLabel(t(impact_text))
                        impact_label.setProperty("origText_en", impact_text)
                        impact_label.setObjectName("inspectorImpact")
                        impact_label.setWordWrap(True)
                        impact_label.setMinimumHeight(0)
                        # A zero horizontal minimum stops the word-wrapped
                        # text from inflating the form's minimum width and
                        # pushing fields under the vertical scrollbar.
                        impact_label.setMinimumSize(0, 0)
                        impact_label.setSizePolicy(
                            QSizePolicy.Policy.Preferred,
                            QSizePolicy.Policy.Preferred,
                        )
                        # Compute the wrapped height for the visible width.
                        # heightForWidth() is explicit, so the label keeps
                        # the correct height even when the form gives it a
                        # narrower column than available_w.
                        available_w = getattr(self, "_available_w", 320)
                        h = impact_label.heightForWidth(available_w)
                        if h <= 0:
                            impact_label.resize(available_w, 0)
                            impact_label.adjustSize()
                            h = impact_label.height()
                        impact_label.setMinimumHeight(h)
                        # Span both columns
                        self._form.addRow(impact_label)

            # adjustSize() was called on each wrapped label before addRow(),
            # so the QFormLayout row heights now reflect the correct text heights.
        finally:
            self._building = False

    def _add_description_visible(self, description: str) -> None:
        """Add a visible (non-collapsible) node description section."""
        desc_label = QLabel(t(description))
        desc_label.setProperty("origText_en", description)
        desc_label.setObjectName("inspectorDescription")
        desc_label.setWordWrap(True)
        desc_label.setTextFormat(Qt.TextFormat.RichText)
        desc_label.setMinimumHeight(0)
        # Zero horizontal minimum: the rich-text body must not inflate the
        # form's minimum width (it would push fields under the scrollbar).
        desc_label.setMinimumSize(0, 0)
        desc_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        # Compute the wrapped height for the visible width. heightForWidth()
        # is explicit, so the label keeps the correct height even when the
        # form gives it a narrower column than available_w.
        available_w = getattr(self, "_available_w", 320)
        h = desc_label.heightForWidth(available_w)
        if h <= 0:
            desc_label.resize(available_w, 0)
            desc_label.adjustSize()
            h = desc_label.height()
        desc_label.setMinimumHeight(h)
        self._form.addRow(desc_label)

    def _make_field(self, key: str, spec: dict, value):
        """Create the appropriate widget based on type/options in the schema."""
        ftype = str(spec.get("type", "str")).lower()
        options = spec.get("options") or spec.get("enum")

        if options:
            combo = QComboBox()
            for option in options:
                if (
                    isinstance(option, (tuple, list))
                    and len(option) == 2
                    and isinstance(option[1], (str, int, float))
                ):
                    label, data = option[0], option[1]
                else:
                    label, data = option, option
                combo.addItem(t(str(label)), data)
            if value is not None:
                index = combo.findData(value)
                if index < 0:
                    index = combo.findText(str(value))
                if index >= 0:
                    combo.setCurrentIndex(index)
            combo.currentIndexChanged.connect(
                lambda index, k=key, cb=combo: self._on_value(k, cb.itemData(index))
            )
            return combo

        if ftype == "bool":
            box = QCheckBox()
            box.setChecked(bool(value))
            box.toggled.connect(
                lambda checked, k=key: self._on_value(k, bool(checked))
            )
            return box

        if ftype == "int":
            minimum = spec.get("minimum", -_INT32_MAX)
            maximum = spec.get("maximum", _INT32_MAX)
            spin = _CursorSpinBox()
            spin.setRange(int(minimum), int(maximum))
            # Thousands grouping follows the system locale (e.g. 100.000.000),
            # so big point caps are readable at a glance. Editing/parsing is
            # handled by Qt; the stored value stays a plain int.
            spin.setGroupSeparatorShown(True)
            try:
                spin.setValue(int(value))
            except (TypeError, ValueError):
                pass
            spin.valueChanged.connect(
                lambda v, k=key: self._on_value(k, int(v))
            )
            return self._wrap_numeric_field(key, spec, spin)

        if ftype == "float":
            spin = _CursorDoubleSpinBox()
            spin.setDecimals(3)
            spin.setRange(
                float(spec.get("minimum", -_FLOAT_MAX)),
                float(spec.get("maximum", _FLOAT_MAX)),
            )
            spin.setSingleStep(0.05)
            try:
                spin.setValue(float(value))
            except (TypeError, ValueError):
                pass
            spin.valueChanged.connect(
                lambda v, k=key: self._on_value(k, float(v))
            )
            return self._wrap_numeric_field(key, spec, spin)

        edit = QLineEdit()
        if value is not None:
            edit.setText(str(value))
            edit.setCursorPosition(0)
        edit.textEdited.connect(
            lambda text, k=key: self._on_value(k, str(text))
        )
        return edit

    def _wrap_numeric_field(self, key, spec, spin) -> QWidget:
        """Wrap a spin box with a reset button and 'modified' highlight.

        Returns a container holding [spin | reset]. The reset button is
        visible only while the current value differs from the schema
        default; the spin gets the dynamic property `modified` so the QSS
        can highlight it.
        """
        container = QWidget()
        container.setObjectName("numericField")
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(spin, 1)

        default = spec.get("default")
        reset_btn = QToolButton()
        reset_btn.setObjectName("paramReset")
        reset_btn.setText("\u21ba")  # rotated open arrow (↺)
        reset_btn.setAutoRaise(True)
        reset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        reset_btn.setToolTip(t("Reset to default"))
        reset_btn.setProperty("origTip_en", "Reset to default")
        if default is None:
            reset_btn.setEnabled(False)
        layout.addWidget(reset_btn)

        def _refresh() -> None:
            diff = _value_differs(spin.value(), default)
            reset_btn.setVisible(diff)
            spin.setProperty("modified", diff)
            st = spin.style()
            st.unpolish(spin)
            st.polish(spin)
            spin.update()

        def _on_reset() -> None:
            if default is not None:
                spin.setValue(default)

        spin.valueChanged.connect(_refresh)
        reset_btn.clicked.connect(_on_reset)
        # Initial state (based on the value already set on the spin).
        _refresh()
        return container

    def _on_value(self, key: str, value) -> None:
        """Write value to the instance and signal stale mark."""
        if self._building or self._item is None:
            return
        if key == STRATEGY_FIELD:
            # Strategy edits go through the canvas (single mutator): it cuts
            # cables, rebuilds ports and refreshes both views. Nothing is
            # written here, otherwise the canvas guard would no-op.
            self.strategy_change_requested.emit(self._item.iid, str(value or ""))
            return
        config = self._item.pipeline_config()
        config[key] = value
        self._item.set_pipeline_config(config)
        self.config_changed.emit(self._item.iid)

    def _on_label_edited(self) -> None:
        """Apply edited label (cosmetic: does not mark stale)."""
        if self._building or self._item is None:
            return
        sender = self.sender()
        if isinstance(sender, QLineEdit):
            self._item.set_label(sender.text())
            self.label_changed.emit(self._item.iid)
