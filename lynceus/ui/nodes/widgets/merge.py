# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""UI item for merge-strategy nodes (variant selector only).

Same contract as ``StrategyNodeItem`` (selected variant in
``_pipeline_config[STRATEGY_FIELD]``, ports rebuilt from the variant,
canvas as single mutator cutting cables), but without file rows: a merge
instance only picks which product it consolidates.
"""

from lynceus.ui.nodes.widgets.strategy import SelectorNodeItem


class MergeNodeItem(SelectorNodeItem):
    """Consolidate item: strategy selector, no file rows."""

    selector_title = "Merge:"
