# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
import os
import uuid

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QGraphicsItem, QGraphicsObject

from lynceus.nodes.ports import get_display_name, get_port_color, port_metadata
from lynceus.plugins.locale import t
from lynceus.ui.nodes.ports import Port

NODE_WIDTH = 240  # Minimum frame width.
PORT_PROTRUSION = 6  # Port marker protrusion from the node edge.
PORT_DOT_SIZE = 12
PORT_HIT_RADIUS = 9
PORT_LABEL_SPACING = 6
HEADER_HEIGHT = 30
ROW_HEIGHT = 18
ROW_SPACING = 4
CONTENT_PADDING = 4
CORNER_RADIUS = 6

FILL = QColor("#1c2333")
BORDER = QColor("#2f3a4f")
BORDER_SELECTED = QColor("#7d9fd4")
TITLE_COLOR = QColor("#e6e9f0")
CATEGORY_COLOR = QColor("#8a93a6")
LABEL_COLOR = QColor("#aeb7c6")
DOT_FILL = QColor("#8a93a6")
DOT_BORDER = QColor("#2f3a4f")
DOT_HOVER = QColor("#7d9fd4")
CLOSE_COLOR = QColor("#8a93a6")
CLOSE_HOVER = QColor("#e08a8a")
BUTTON_FILL = QColor("#2f3a4f")
BUTTON_FILL_HOVER = QColor("#3a465c")
BUTTON_TEXT = QColor("#e6e9f0")

# Processing states.
STATUS_IDLE = "idle"
STATUS_PROCESSING = "processing"
STATUS_DONE = "done"
STATUS_CACHED = "cached"
STATUS_ERROR = "error"

STATUS_BORDER = {
    STATUS_IDLE: BORDER,
    STATUS_DONE: QColor("#6fbf73"),
    STATUS_CACHED: QColor("#5a9463"),
    STATUS_ERROR: QColor("#d97a7a"),
}
PULSE_BORDER = QColor("#7d9fd4")
PULSE_DIM_BORDER = QColor("#4d6a9e")
PULSE_INTERVAL_MS = 250
UNAVAILABLE_OPACITY = 0.45


def _port_label_font() -> QFont:
    from lynceus.ui.fonts import scaled_point_size

    font = QFont()
    font.setPointSize(scaled_point_size(8))
    return font


class NodeItem(QGraphicsObject):
    """Vector-painted node item with header, ports, and content.

    The item remains sharp at every zoom. Its dimensions grow with port labels
    and port count; port markers protrude horizontally, never vertically.
    """

    close_requested = Signal()
    selected = Signal(object)
    moved = Signal(object)
    port_pressed = Signal(object)
    message_requested = Signal(str)
    label_edit_requested = Signal(object)  # the item itself
    preview_requested = Signal(object)  # file-backed item double-clicked on its body

    def __init__(
        self,
        node_id: str,
        node_name: str,
        category: str,
        inputs: tuple = (),
        outputs: tuple = (),
        parent=None,
    ):
        super().__init__(parent)
        self._node_id = node_id
        # Instance identity is stable in saved projects; module IDs are not
        # unique because multiple copies of a node type may coexist.
        self.iid = uuid.uuid4().hex[:8]
        # Keep the registry's English name as the translation source.
        self._node_name_en = node_name
        self._node_name = t(node_name)
        # A custom instance label is cosmetic and does not affect execution.
        self._label = ""
        self._category = category
        self._selected = False
        self._dragging = False
        self._drag_offset = QPointF()
        self._defer_select = False  # collapse-to-single on release, not press
        self._press_moved = False
        self._hover_region = None  # None | "close" | Port | "content" | "button"
        self._hovered_port: Port | None = None
        self._input_ports: list[Port] = []
        self._output_ports: list[Port] = []
        self.snap_provider = None  # callable QPointF -> QPointF (set by canvas)
        self.selected_items_provider = None  # callable -> list (set by canvas)
        # Inspector overrides merged over module defaults during DAG compilation.
        self._pipeline_config: dict = {}
        self._frame_width = NODE_WIDTH
        self._height = HEADER_HEIGHT + 20

        self.setAcceptHoverEvents(True)

        self._status = STATUS_IDLE
        self._pulse_on = False
        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(PULSE_INTERVAL_MS)
        self._pulse_timer.timeout.connect(self._toggle_pulse)
        self._available = True

        self._update_tooltip()

        self._add_ports(inputs, outputs)

    def _add_ports(self, inputs: tuple, outputs: tuple) -> None:
        """Builds the input/output ports from port metadata tuples.

        Shared by __init__ and `rebuild_ports` so a placeholder node that
        gains its module (re-enabled extension) gets identical ports.
        """
        for item in inputs:
            d = port_metadata(item)
            if d.name is not None:
                self.add_input(t(d.name), d.port_type, name_en=d.name,
                               required=d.required, group=d.group)
            else:
                self.add_input(t(get_display_name(d.port_type)), d.port_type,
                               name_en=get_display_name(d.port_type),
                               required=d.required, group=d.group)

        for item in outputs:
            d = port_metadata(item)
            if d.name is not None:
                self.add_output(t(d.name), d.port_type, name_en=d.name,
                                required=d.required, group=d.group)
            else:
                self.add_output(t(get_display_name(d.port_type)), d.port_type,
                                name_en=get_display_name(d.port_type),
                                required=d.required, group=d.group)

        self._recompute()

    def rebuild_ports(self, inputs: tuple, outputs: tuple) -> None:
        """Rebuilds ports on a node that was unavailable (no module).

        Used when a disabled/missing extension is re-enabled: the node keeps
        its identity, position and config but gains its connectable ports.
        A placeholder has no ports and, therefore, no connections to lose.
        """
        self._input_ports = []
        self._output_ports = []
        self._add_ports(inputs, outputs)
        self._update_tooltip()
        self.update()

    # ---------- Processing state ----------

    def set_status(self, status: str) -> None:
        self._status = status
        if status == STATUS_PROCESSING:
            self._pulse_on = True
            self._pulse_timer.start()
        else:
            self._pulse_timer.stop()
            self._pulse_on = False
        self.update()

    def status(self) -> str:
        return self._status

    # ---------- instance label ----------

    def display_name(self) -> str:
        """Custom label if set; otherwise the type name."""
        return self._label or self._node_name

    def set_label(self, label: str) -> None:
        """Set or clear the visible instance label."""
        label = (label or "").strip()
        if label == self._label:
            return
        self._label = label
        self._update_tooltip()
        self.update()

    def retranslate(self) -> None:
        """Reapply translations to registry names and labeled ports."""
        self._node_name = t(self._node_name_en)
        for port in self._input_ports + self._output_ports:
            if port.name_en is not None:
                port.name = t(port.name_en)
        self._update_tooltip()
        self._recompute()
        self.update()

    def label(self) -> str:
        return self._label

    def _update_tooltip(self) -> None:
        tip = f"{self._node_name}  ·  #{self.iid}"
        if self._label:
            tip = f"{self._label}\n{tip}"
        if not self._available:
            tip += "\n" + t("Unavailable — extension missing or disabled")
        self.setToolTip(tip)

    def _toggle_pulse(self) -> None:
        self._pulse_on = not self._pulse_on
        self.update()

    def _border_color(self) -> QColor:
        if self._status == STATUS_PROCESSING:
            return PULSE_BORDER if self._pulse_on else PULSE_DIM_BORDER
        return STATUS_BORDER.get(self._status, BORDER)

    # ---------- API ----------

    def node_id(self) -> str:
        return self._node_id

    def set_available(self, available: bool) -> None:
        """Flags node availability (its module registered in the registry).

        Unavailable nodes (disabled or missing extension) stay on the canvas
        keeping their configuration, but are dimmed and blocked from runs.
        """
        available = bool(available)
        if available == self._available:
            return
        self._available = available
        self.setOpacity(1.0 if available else UNAVAILABLE_OPACITY)
        self._update_tooltip()
        self.update()

    def is_available(self) -> bool:
        return self._available

    def node_name(self) -> str:
        return self._node_name

    def node_name_en(self) -> str:
        return self._node_name_en

    def inputs(self) -> list[Port]:
        return list(self._input_ports)

    def outputs(self) -> list[Port]:
        return list(self._output_ports)

    def add_input(self, name: str, port_type, name_en: str | None = None,
                  required: bool = True, group: str | None = None) -> Port:
        return self._add_port("input", name, port_type, name_en,
                              required=required, group=group)

    def add_output(self, name: str, port_type, name_en: str | None = None,
                   required: bool = True, group: str | None = None) -> Port:
        return self._add_port("output", name, port_type, name_en,
                              required=required, group=group)

    def _add_port(self, kind: str, name: str, port_type,
                  name_en: str | None = None,
                  required: bool = True, group: str | None = None) -> Port:
        max_connections = None if kind == "output" else 1
        port = Port(self, kind, name, port_type, max_connections, name_en,
                    required=required, group=group)
        (self._input_ports if kind == "input" else self._output_ports).append(port)
        self._recompute()
        return port

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        # Native Qt selection stays in lockstep: deletion and hit paths read
        # isSelected(), painting reads _selected; itemChange() mirrors back
        # selections made natively (rubber-band later, accessibility tools).
        if self.isSelected() != selected:
            self.setSelected(selected)
        self.update()

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedChange:
            self._selected = bool(value)
            self.update()
        return super().itemChange(change, value)

    def clear_data(self) -> None:
        """Hook to clear visual data (overridden in nodes with data)."""
        pass

    # ---------- pipeline configuration (Inspector) ----------

    def set_pipeline_config(self, config: dict) -> None:
        """Replaces the per-instance configuration (config_schema keys)."""
        self._pipeline_config = dict(config)

    def pipeline_config(self) -> dict:
        return dict(self._pipeline_config)

    def mark_stale(self) -> None:
        """After editing the configuration, a green (done) node goes idle."""
        if self._status in (STATUS_DONE, STATUS_CACHED):
            self.set_status(STATUS_IDLE)

    def capture_config(self) -> dict:
        """Configurable node state to persist into a project."""
        if not self._pipeline_config:
            return {}
        return {"pipeline": dict(self._pipeline_config)}

    def apply_config(self, config: dict) -> None:
        """Restores the state from a project (overridden per node)."""
        cfg = config.get("pipeline")
        if isinstance(cfg, dict):
            self._pipeline_config = {
                k: v for k, v in cfg.items() if isinstance(k, str)
            }

    def content_height(self) -> float:
        return 0.0

    # ---------- responsive layout ----------

    def _side_width(self, ports: list[Port]) -> float:
        if not ports:
            return 0.0
        fm = QFontMetrics(_port_label_font())
        widest = max(fm.horizontalAdvance(port.name) for port in ports)
        return PORT_DOT_SIZE + PORT_LABEL_SPACING + widest

    def _ports_height(self) -> float:
        rows = max(len(self._input_ports), len(self._output_ports))
        if not rows:
            return 0.0
        return rows * (ROW_HEIGHT + ROW_SPACING) + CONTENT_PADDING

    def refresh_layout(self) -> None:
        """Recompute metrics-driven geometry (font scale changes)."""
        self._recompute()

    def _recompute(self) -> None:
        left = 10 + self._side_width(self._input_ports)
        right = 10 + self._side_width(self._output_ports)
        self._frame_width = max(
            NODE_WIDTH, left + right + 10, int(self.content_height()) + 20
        )
        self._width = self._frame_width + 2 * PORT_PROTRUSION

        ports_height = self._ports_height()
        content_height = self.content_height()
        body = ports_height + (8 + content_height if content_height else 0)
        self._height = HEADER_HEIGHT + body + 16

        self.prepareGeometryChange()
        self.update()

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self._width, self._height)

    def _port_y(self, index: int) -> float:
        return (
            HEADER_HEIGHT
            + CONTENT_PADDING
            + index * (ROW_HEIGHT + ROW_SPACING)
            + ROW_HEIGHT / 2
        )

    def port_center(self, port: Port) -> QPointF:
        """Port dot center in item coordinates."""
        if port.kind == "input":
            x = PORT_PROTRUSION
        else:
            x = self._frame_width + PORT_PROTRUSION
        ports = self._input_ports if port.kind == "input" else self._output_ports
        return QPointF(x, self._port_y(ports.index(port)))

    def _content_rect(self) -> QRectF:
        return QRectF(
            PORT_PROTRUSION + 10,
            HEADER_HEIGHT + CONTENT_PADDING + self._ports_height() + 8,
            self._frame_width - 20,
            self.content_height(),
        )

    # ---------- painting ----------

    def paint(self, painter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        frame = QRectF(PORT_PROTRUSION, 1, self._frame_width, self._height - 2)
        border = self._border_color() if self._status != STATUS_IDLE else (
            BORDER_SELECTED if self._selected else BORDER
        )
        painter.setBrush(FILL)
        painter.setPen(QPen(border, 1.2))
        painter.drawRoundedRect(frame, CORNER_RADIUS, CORNER_RADIUS)

        self._paint_header(painter, frame)
        accent = self.header_accent()
        if accent is not None:
            painter.setPen(QPen(accent, 2))
            painter.drawLine(
                QPointF(frame.x() + 2, frame.y() + HEADER_HEIGHT),
                QPointF(frame.right() - 2, frame.y() + HEADER_HEIGHT),
            )
        self._paint_close(painter, frame)
        for port in self._input_ports:
            self._paint_port(painter, port)
        for port in self._output_ports:
            self._paint_port(painter, port)

        self.paint_content(painter, self._content_rect())

    def header_accent(self) -> QColor | None:
        """Accent line under the header (None = no line)."""
        return None

    def _paint_header(self, painter: QPainter, frame: QRectF) -> None:
        category_font = self._font(7.5, bold=True)
        painter.setFont(category_font)
        painter.setPen(CATEGORY_COLOR)
        category_text = t(self._category).upper()
        painter.drawText(QPointF(frame.x() + 10, 18), category_text)

        title_font = self._font(10, bold=True)
        painter.setFont(title_font)
        painter.setPen(TITLE_COLOR)
        x_title = (
            frame.x()
            + 10
            + QFontMetrics(category_font).horizontalAdvance(category_text)
            + 10
        )
        painter.drawText(QPointF(x_title, 18), self.display_name())

    def _paint_close(self, painter: QPainter, frame: QRectF) -> None:
        close_rect = QRectF(frame.right() - 26, 5, 16, 16)
        painter.setFont(self._font(10))
        painter.setPen(CLOSE_HOVER if self._hover_region == "close" else CLOSE_COLOR)
        painter.drawText(close_rect, Qt.AlignmentFlag.AlignCenter, "\u2715")

    def _paint_port(self, painter: QPainter, port: Port) -> None:
        center = self.port_center(port)
        hovered = self._hovered_port is port
        base = QColor(get_port_color(port.port_type_id))
        if hovered:
            fill, border = DOT_HOVER, DOT_HOVER
        elif port.required:
            fill, border = base, base.darker(140)
        else:
            # Optional port: hollow ring (same edge color)
            fill = QColor(base)
            fill.setAlpha(0)
            border = base
        painter.setBrush(fill)
        painter.setPen(QPen(border, 1.2))
        painter.drawEllipse(center, PORT_DOT_SIZE / 2, PORT_DOT_SIZE / 2)

        font = _port_label_font()
        fm = QFontMetrics(font)
        painter.setFont(font)
        painter.setPen(LABEL_COLOR)
        baseline = center.y() + (fm.ascent() - fm.descent()) / 2
        label = port.name
        if port.kind == "input":
            painter.drawText(
                QPointF(center.x() + PORT_DOT_SIZE / 2 + PORT_LABEL_SPACING, baseline),
                label,
            )
        else:
            width = fm.horizontalAdvance(label)
            painter.drawText(
                QPointF(
                    center.x() - PORT_DOT_SIZE / 2 - PORT_LABEL_SPACING - width,
                    baseline,
                ),
                label,
            )

    def paint_content(self, painter: QPainter, rect: QRectF) -> None:
        pass

    @staticmethod
    def _font(size: float, bold: bool = False) -> QFont:
        from lynceus.ui.fonts import font_scale

        font = QFont()
        font.setBold(bold)
        font.setPointSizeF(max(1.0, size * font_scale()))
        return font

    @staticmethod
    def paint_button(
        painter: QPainter,
        rect: QRectF,
        text: str,
        hovered: bool,
        font_size: float = 9,
    ) -> None:
        """Draws a reusable painted button for any content."""
        from lynceus.ui.fonts import font_scale

        painter.setBrush(BUTTON_FILL_HOVER if hovered else BUTTON_FILL)
        painter.setPen(QPen(BORDER, 1))
        painter.drawRoundedRect(rect, 4, 4)
        font = QFont()
        font.setPointSizeF(max(1.0, font_size * font_scale()))
        painter.setFont(font)
        painter.setPen(BUTTON_TEXT)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    # ---------- interaction ----------

    def _hit_region(self, pos: QPointF):
        close_rect = QRectF(self._frame_width + PORT_PROTRUSION - 26, 5, 16, 16)
        if close_rect.contains(pos):
            return "close"
        if pos.y() <= HEADER_HEIGHT:
            return "title"
        for port in self._input_ports + self._output_ports:
            if (pos - self.port_center(port)).manhattanLength() <= PORT_HIT_RADIUS:
                return port
        if self._content_rect().contains(pos):
            return "content"
        return "body"

    def on_content_click(self, region) -> None:
        pass

    def on_connection_changed(self) -> None:
        """Hook: called when something connects/disconnects to its ports."""

    def on_node_removed(self) -> None:
        """Hook: called when the node is removed from the canvas."""

    def mousePressEvent(self, event) -> None:
        region = self._hit_region(event.pos())
        if region == "close":
            self.close_requested.emit()
            event.accept()
            return
        if isinstance(region, Port):
            if region.kind == "output":
                self.port_pressed.emit(region)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and (
            event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            # Ctrl+click: let Qt handle native multi-selection, but prevent
            # content actions (Browse, menus). Don't accept here; if we do,
            # Qt's default selection handler won't run.
            super().mousePressEvent(event)
            return
        if region in ("content", "button"):
            self.on_content_click(region)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_offset = event.scenePos() - self.pos()
            self._press_moved = False
            multi = False
            provider = self.selected_items_provider
            if self._selected and callable(provider):
                try:
                    multi = sum(1 for _ in provider()) > 1
                except Exception:
                    multi = False
            # Pressing a member of a multi-selection defers the collapse to
            # release: a press+drag moves the formation, a bare click
            # collapses to this node. Pressing outside any selection
            # collapses immediately, as before.
            self._defer_select = multi
            if not multi:
                self.selected.emit(self)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._dragging:
            new_pos = event.scenePos() - self._drag_offset
            if self.snap_provider is not None:
                new_pos = self.snap_provider(new_pos)
            delta = new_pos - self.pos()
            moved = not delta.isNull()
            self._press_moved = self._press_moved or moved
            self.setPos(new_pos)
            self.moved.emit(self)
            if not delta.isNull() and self._selected:
                # The leader snaps; companions follow by the same delta so
                # the formation holds while multi-dragging a selection.
                provider = self.selected_items_provider
                peers = []
                if callable(provider):
                    try:
                        peers = [p for p in provider() if p is not self]
                    except Exception:
                        peers = []
                for peer in peers:
                    peer.setPos(peer.pos() + delta)
                    peer.moved.emit(peer)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._dragging:
            self._dragging = False
            if self._defer_select and not self._press_moved:
                # Bare click on a member: collapse the multi-selection here.
                self._defer_select = False
                self.selected.emit(self)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def preview_file(self) -> str | None:
        """Existing file backing this node, if it is file-backed.

        Loaders and product inputs expose ``file_path``; anything else (or a
        missing/vanished file) yields None so double-click stays a no-op.
        """
        file_path = getattr(self, "file_path", None)
        if not callable(file_path):
            return None
        try:
            path = file_path()
        except Exception:
            return None
        if not path or not os.path.isfile(path):
            return None
        return path

    def mouseDoubleClickEvent(self, event) -> None:
        region = self._hit_region(event.pos())
        if region == "title":
            self.label_edit_requested.emit(self)
            event.accept()
            return
        if region == "body" and self.preview_file():
            self.preview_requested.emit(self)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def hoverMoveEvent(self, event) -> None:
        region = self._hit_region(event.pos())
        self._hover_region = region if isinstance(region, (str, Port)) else None
        self._hovered_port = region if isinstance(region, Port) else None
        if isinstance(region, Port):
            self.setToolTip(
                f"{region.name} — "
                + (t("Required") if region.required else t("Optional"))
            )
        else:
            self._update_tooltip()
        self.update()
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        self._hover_region = None
        self._hovered_port = None
        self.update()
        super().hoverLeaveEvent(event)

    def contextMenuEvent(self, event) -> None:
        from PySide6.QtWidgets import QMenu

        menu = QMenu()
        rename_action = menu.addAction(t("Rename..."))
        rename_action.triggered.connect(
            lambda: self.label_edit_requested.emit(self)
        )
        menu.exec(event.screenPos())
        event.accept()
