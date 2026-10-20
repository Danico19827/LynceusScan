# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Node library search: matcher predicate and tree filtering."""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QTreeWidget

from lynceus.plugins.registry import manager
from lynceus.ui.node_library import (
    apply_library_filter,
    node_matches,
    populate_library,
)

LOAD_ID = "lynceus.nodes.lidar.source.load_las_laz"
DTM_ID = "lynceus.nodes.lidar.terrain.generate_dtm"


class MatcherTests(unittest.TestCase):
    def test_empty_query_matches_all(self) -> None:
        self.assertTrue(node_matches("", "t", "e", "i"))
        self.assertTrue(node_matches("   ", "t", "e", "i"))

    def test_case_insensitive_substring(self) -> None:
        self.assertTrue(node_matches("dtm", "x", "Generate DTM", "i"))
        self.assertTrue(node_matches("DTM", "Generate DTM", "x", "i"))

    def test_id_matches(self) -> None:
        self.assertTrue(node_matches("generate_dtm", "x", "y", DTM_ID))

    def test_no_match(self) -> None:
        self.assertFalse(node_matches("zzz", "Generate DTM",
                                      "Generate DTM", DTM_ID))


class FilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self._app = QApplication.instance() or QApplication([])
        self.assertTrue(manager.contains(LOAD_ID))
        self.assertTrue(manager.contains(DTM_ID))
        self.tree = QTreeWidget()
        populate_library(self.tree)

    def tearDown(self) -> None:
        self.tree.deleteLater()

    def _visible_leaves(self) -> list[str]:
        leaves = []
        for ci in range(self.tree.topLevelItemCount()):
            category = self.tree.topLevelItem(ci)
            if category.isHidden():
                continue
            for si in range(category.childCount()):
                child = category.child(si)
                if child.childCount():
                    if child.isHidden():
                        continue
                    for li in range(child.childCount()):
                        leaf = child.child(li)
                        if not leaf.isHidden():
                            leaves.append(str(
                                leaf.data(0, Qt.ItemDataRole.UserRole)
                                or leaf.text(0)))
                elif not child.isHidden():
                    leaves.append(str(
                        child.data(0, Qt.ItemDataRole.UserRole)
                        or child.text(0)))
        return leaves

    def test_empty_filter_shows_everything_collapsed(self) -> None:
        count = apply_library_filter(self.tree, "")
        self.assertGreater(count, 10)
        leaves = self._visible_leaves()
        self.assertEqual(len(leaves), count)

    def test_english_name_matches(self) -> None:
        leaves = self._visible_leaves()
        total = len(leaves)
        count = apply_library_filter(self.tree, "DTM")
        shown = self._visible_leaves()
        self.assertGreater(count, 0)
        self.assertLess(count, total)
        self.assertTrue(any("dtm" in leaf.lower() for leaf in shown))

    def test_id_matches(self) -> None:
        count = apply_library_filter(self.tree, "generate_dtm")
        shown = self._visible_leaves()
        self.assertGreaterEqual(count, 1)
        self.assertTrue(any("generate_dtm" in leaf for leaf in shown))

    def test_no_match_hides_categories(self) -> None:
        count = apply_library_filter(self.tree, "zzz-no-such-node")
        self.assertEqual(count, 0)
        hidden_cats = [
            self.tree.topLevelItem(ci).isHidden()
            for ci in range(self.tree.topLevelItemCount())
        ]
        self.assertTrue(hidden_cats)
        self.assertTrue(all(hidden_cats))

    def test_filter_survives_repopulate(self) -> None:
        first = apply_library_filter(self.tree, "DTM")
        populate_library(self.tree)
        second = apply_library_filter(self.tree, "DTM")
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
