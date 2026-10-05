# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Connectable node-port model used by the UI.

Data contracts (PortType and compatibility) live in the domain layer;
this module only bridges the port model and its canvas item.
"""

from __future__ import annotations

from lynceus.nodes.ports import PortType, can_connect


class Port:
    """Pure node-port model without widget dependencies.

    NodeItem calculates the geometry; the canvas uses scene coordinates for
    connection edges.
    """

    def __init__(
        self,
        node,
        kind: str,
        name: str,
        port_type: PortType | str,
        max_connections: int | None = 1,
        name_en: str | None = None,
        required: bool = True,
        group: str | None = None,
    ):
        self.node = node              # owner NodeItem
        self.kind = kind              # "input" | "output"
        self.name = name              # readable name (e.g. "Point Cloud")
        self.name_en = name_en        # English source if translatable; None = custom
        # Accepts an enum (builtin) or str (canonical, possibly custom).
        # The canonical string id always lives in .port_type_id.
        self.port_type = port_type
        if isinstance(port_type, PortType):
            self.port_type_id: str = port_type.value
        else:
            self.port_type_id = str(port_type)
        self.max_connections = max_connections  # None means unlimited.
        self.required = required      # Required inputs block execution when loose.
        self.group = group            # Mutual-exclusion group, or None.
        self.connections: list = []   # Connection objects managed by the canvas.

    def is_full(self) -> bool:
        if self.max_connections is None or self.max_connections <= 0:
            return False
        return len(self.connections) >= self.max_connections

    def can_connect_to(self, other: "Port") -> bool:
        """Return whether this port can connect to another port."""
        if self.kind == other.kind:
            return False
        if self.kind == "output":
            return can_connect(self.port_type_id, other.port_type_id)
        return can_connect(other.port_type_id, self.port_type_id)

    def group_conflict(self) -> "Port | None":
        """Return a connected input that conflicts with this exclusive group."""
        if self.kind != "input" or self.group is None:
            return None
        for other in self.node.inputs():
            if other is not self and other.group == self.group and other.connections:
                return other
        return None

    def __repr__(self) -> str:
        return f"Port({self.kind}, {self.name}, {self.port_type_id})"
