# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Output dock containing the pipeline products.

The single "All Products" gallery includes CSV statistics and point clouds.
Double-click opens a floating preview; the context menu opens the containing
folder. Per-node embedded preview tabs were removed.
"""

from PySide6.QtWidgets import QVBoxLayout, QWidget

from lynceus.ui.outputs_model import OutputProduct
from lynceus.ui.outputs_panel import OutputsPanel


class OutputDock(QWidget):
    """Results dock containing the visualizable product gallery."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("outputDock")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.products = OutputsPanel(self)
        layout.addWidget(self.products)

    def set_products(self, products: list[OutputProduct]) -> None:
        """Replace gallery products after a pipeline run."""
        self.products.set_products(products)

    def clear_outputs(self) -> None:
        """Clear products from the active session."""
        self.products.clear()