# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Canvas run feedback: live gallery, phase chip, selection readability.

- Barrier products publish to the gallery at completion (no waiting for
  the whole run), without duplicates at finish/consolidation, and stay
  silent during segment batches (authoritative batch publish wins).
- The phase chip names the running merge while a single long barrier
  holds the run.
- Right-click never clears a multi-selection (fleet menu stays reachable).
- Selected nodes keep a visible selection ring on done/error outlines.
"""

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from lynceus.plugins.registry import manager
from lynceus.ui.nodes.canvas import NodeCanvasView

LOAD_ID = "lynceus.nodes.lidar.source.load_las_laz"
DTM_ID = "lynceus.nodes.lidar.terrain.generate_dtm"


def _right_press(view, pos) -> None:
    from PySide6.QtCore import QPointF

    event = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        QPointF(pos),
        QPointF(pos),
        Qt.MouseButton.RightButton,
        Qt.MouseButton.RightButton,
        Qt.KeyboardModifier.NoModifier,
    )
    view.mousePressEvent(event)


class RightClickSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._app = QApplication.instance() or QApplication([])
        self.assertTrue(manager.contains(LOAD_ID))
        self.assertTrue(manager.contains(DTM_ID))
        self.view = NodeCanvasView()
        self.a = self.view.add_node(LOAD_ID, QPointF(0, 0))
        self.b = self.view.add_node(DTM_ID, QPointF(500, 0))

    def tearDown(self) -> None:
        self.view.deleteLater()

    def test_right_press_keeps_multi_selection(self) -> None:
        self.a.setSelected(True)
        self.b.setSelected(True)
        self.assertEqual(self.view.selected_nodes(), [self.a, self.b])
        # Empty corner of the viewport (no item under cursor).
        _right_press(self.view, QPointF(5, 5))
        self.assertEqual(self.view.selected_nodes(), [self.a, self.b])

    def test_right_press_on_member_keeps_selection(self) -> None:
        self.a.setSelected(True)
        self.b.setSelected(True)
        center = self.view.mapFromScene(self.a.pos())
        _right_press(
            self.view, QPointF(float(center.x()), float(center.y())))
        self.assertEqual(self.view.selected_nodes(), [self.a, self.b])


class SelectionRingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._app = QApplication.instance() or QApplication([])
        self.assertTrue(manager.contains(LOAD_ID))
        self.view = NodeCanvasView()

    def tearDown(self) -> None:
        self.view.deleteLater()

    def _ring_pixel(self, status: str, selected: bool):
        from PySide6.QtGui import QImage, QPainter
        from PySide6.QtWidgets import QStyleOptionGraphicsItem

        from lynceus.ui.nodes.node_item import (
            BORDER_SELECTED,
            PORT_PROTRUSION,
        )

        item = self.view.add_node(LOAD_ID, QPointF(0, 0))
        item.set_status(status)
        item.set_selected(selected)
        image = QImage(600, 300, QImage.Format.Format_RGB32)
        image.fill("#000000")
        painter = QPainter(image)
        option = QStyleOptionGraphicsItem()
        option.rect = item.boundingRect().toRect()
        item.paint(painter, option, None)
        painter.end()
        # Left edge stroke center of the outer selection ring.
        color = image.pixelColor(int(PORT_PROTRUSION - 3.5), 20)
        return color, BORDER_SELECTED

    @staticmethod
    def _distance(a, b) -> float:
        return (
            (a.red() - b.red()) ** 2
            + (a.green() - b.green()) ** 2
            + (a.blue() - b.blue()) ** 2
        ) ** 0.5

    def test_done_selected_shows_ring(self) -> None:
        color, ring = self._ring_pixel("done", True)
        self.assertLess(self._distance(color, ring), 40.0)

    def test_done_unselected_shows_no_ring(self) -> None:
        color, ring = self._ring_pixel("done", False)
        self.assertGreater(self._distance(color, ring), 100.0)

    def test_error_selected_shows_ring(self) -> None:
        color, ring = self._ring_pixel("error", True)
        self.assertLess(self._distance(color, ring), 40.0)


class LiveGalleryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._app = QApplication.instance() or QApplication([])
        self.assertTrue(manager.contains(LOAD_ID))
        self.assertTrue(manager.contains(DTM_ID))
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.view = NodeCanvasView()
        self.addCleanup(self.view.deleteLater)
        self.a = self.view.add_node(LOAD_ID, QPointF(0, 0))
        self.b = self.view.add_node(DTM_ID, QPointF(500, 0))
        self.emitted: list = []
        self.view.outputs_updated.connect(self.emitted.append)
        self.fa = self._product_file("a_mosaic.tif")
        self.fb = self._product_file("b_mosaic.tif")

    def _product_file(self, name: str) -> str:
        path = Path(self._tmp.name) / name
        path.write_bytes(b"tif")
        return str(path)

    def test_barrier_publishes_under_its_node_immediately(self) -> None:
        self.view._on_barrier_outputs(self.a.iid, {"file": self.fa})
        self.assertEqual(len(self.emitted), 1)
        names = [p.node_name for p in self.emitted[-1]]
        self.assertEqual(len(names), 1)
        self.assertIn(self.a.iid, self.view._live_products)
        second = {"file": self.fb}
        self.view._on_barrier_outputs(self.b.iid, second)
        self.assertEqual(len(self.emitted), 2)
        by_iid = [p.iid for p in self.emitted[-1]]
        self.assertEqual(by_iid, [self.a.iid, self.b.iid])

    def test_finish_replaces_live_without_dupes(self) -> None:
        self.view._graph_snapshot = self.view.serialize_graph()
        self.view._on_barrier_outputs(self.a.iid, {"file": self.fa})
        self.view._on_barrier_outputs(self.b.iid, {"file": self.fb})
        before = len(self.emitted)
        self.view._on_pipeline_finished(
            {self.a.iid: {"file": self.fa},
             self.b.iid: {"file": self.fb}},
            [], 0, [], {},
        )
        self.assertGreater(len(self.emitted), before)
        final = self.emitted[-1]
        self.assertEqual(len(final), 2)
        self.assertEqual(self.view._live_products, {})

    def test_batch_suppresses_live_until_segment_close(self) -> None:
        self.view._on_segment_started(1, 2, 100)
        self.view._on_barrier_outputs(self.a.iid, {"file": self.fa})
        self.assertEqual(self.emitted, [])
        self.assertEqual(self.view._live_products, {})
        self.view._on_segment_outputs(
            1, 2, {self.a.iid: {"file": self.fa}})
        self.assertEqual(len(self.emitted), 1)

    def test_consolidated_replaces_republished_live(self) -> None:
        self.view._on_barrier_outputs(self.a.iid, {"file": self.fa})
        self.assertIn(self.a.iid, self.view._live_products)
        self.view._on_consolidated_outputs(
            {self.a.iid: {"file": self.fa}})
        self.assertNotIn(self.a.iid, self.view._live_products)

    def test_queue_finish_skips_live_published_iids(self) -> None:
        self.view._file_queue = {
            "iid": self.a.iid,
            "files": [self.fa],
            "index": 0,
            "results": [],
            "order": [self.fa],
            "nodes": [],
            "edges": [],
            "configs": {},
            "source_paths": {},
            "prefs": {},
            "stop": False,
            "got_finished": False,
        }
        self.view._on_barrier_outputs(self.a.iid, {"file": self.fa})
        self.assertEqual(self.emitted, [])
        self.view._record_queue_result(
            {self.a.iid: {"file": self.fa}}, 0, [], {})
        paths = [p.path for p in self.emitted[-1]
                 if p.payload_key == "file"]
        self.assertEqual(paths.count(self.fa), 1)


class PhaseChipTests(unittest.TestCase):
    def setUp(self) -> None:
        self._app = QApplication.instance() or QApplication([])
        self.assertTrue(manager.contains(DTM_ID))
        self.view = NodeCanvasView()
        self.addCleanup(self.view.deleteLater)
        self.node = self.view.add_node(DTM_ID, QPointF(0, 0))
        self.chip = self.view._segment_chip

    def test_barrier_start_shows_phase(self) -> None:
        self.view._on_barrier_started(self.node.iid)
        self.assertFalse(self.chip.isHidden())
        self.assertTrue(self.chip.text().startswith("Merging "))
        self.assertTrue(self.chip.text().endswith("..."))

    def test_batch_owns_chip_during_segments(self) -> None:
        self.view._on_segment_started(1, 2, 100)
        text = self.chip.text()
        self.view._on_barrier_started(self.node.iid)
        self.assertEqual(self.chip.text(), text)


class LiveNoticeTests(unittest.TestCase):
    """One provisional-styling FYI per run, gone when the run ends."""

    def setUp(self) -> None:
        self._app = QApplication.instance() or QApplication([])
        self.assertTrue(manager.contains(LOAD_ID))
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.view = NodeCanvasView()
        self.addCleanup(self.view.deleteLater)
        self.node = self.view.add_node(LOAD_ID, QPointF(0, 0))
        self.fa = self._product_file("n_mosaic.tif")

    def _product_file(self, name: str) -> str:
        path = Path(self._tmp.name) / name
        path.write_bytes(b"tif")
        return str(path)

    def _notice_texts(self) -> list:
        return [row.text() for row in self.view._notice_tray._rows]

    def _live_rows(self) -> list:
        return [x for x in self._notice_texts()
                if "Provisional styling" in x]

    def test_first_live_publish_shows_notice_once(self) -> None:
        self.view._on_barrier_outputs(self.node.iid, {"file": self.fa})
        self.assertEqual(len(self._live_rows()), 1)
        fb = self._product_file("m_mosaic.tif")
        self.view._on_barrier_outputs(self.node.iid, {"file": fb})
        self.assertEqual(len(self._live_rows()), 1)

    def test_finish_hides_notice(self) -> None:
        self.view._on_barrier_outputs(self.node.iid, {"file": self.fa})
        self.assertEqual(len(self._live_rows()), 1)
        self.view._graph_snapshot = self.view.serialize_graph()
        self.view._on_pipeline_finished(
            {self.node.iid: {"file": self.fa}}, [], 0, [], {})
        self.assertEqual(self._live_rows(), [])

    def test_segment_publish_shows_notice(self) -> None:
        self.view._on_segment_started(1, 1, 10)
        self.view._on_segment_outputs(
            1, 1, {self.node.iid: {"file": self.fa}})
        self.assertEqual(len(self._live_rows()), 1)


class ProgressPinTests(unittest.TestCase):
    """Rate-suffixed tiling messages still pin the transient notice row."""

    def test_tiling_rate_message_pins(self) -> None:
        from lynceus.ui.nodes.canvas import _PROGRESS_RE

        self.assertIsNotNone(
            _PROGRESS_RE.search("Tiling... 14% (2.5 M pts/s)"))
        self.assertIsNotNone(_PROGRESS_RE.search("Tiling... 14%"))
        self.assertIsNotNone(_PROGRESS_RE.search("DTM 3/51 | CHM 2/51"))
        self.assertIsNone(_PROGRESS_RE.search("Run finished"))


class QueueResultTests(unittest.TestCase):
    """Queue failure rows: run-scoped map wins over the last-batch state."""

    def setUp(self) -> None:
        self._app = QApplication.instance() or QApplication([])
        self.assertTrue(manager.contains(LOAD_ID))
        self.view = NodeCanvasView()
        self.a = self.view.add_node(LOAD_ID, QPointF(0, 0))
        # The canvas rebuilds the map at run start; tests seed it directly.
        self.view._item_by_iid = {self.a.iid: self.a}

    def tearDown(self) -> None:
        self.view.deleteLater()

    def _queue(self) -> dict:
        return {
            "iid": self.a.iid,
            "files": ["one.laz"],
            "index": 0,
            "results": [],
            "order": ["one.laz"],
            "nodes": [],
            "edges": [],
            "configs": {},
            "source_paths": {},
            "prefs": {},
            "stop": False,
            "got_finished": False,
        }

    def test_failure_row_prefers_run_scoped_metrics(self) -> None:
        self.view._file_queue = self._queue()
        # Stale per-batch state must not shadow the run-scoped map.
        self.view._pipeline.session_state["failed_nodes"] = ["stale"]
        self.view._record_queue_result(
            {}, 1, [], {"failed_nodes": {self.a.iid: "boom"}})
        row = self.view._file_queue["results"][0]
        self.assertFalse(row["ok"])
        self.assertIn("1 failed task(s)", row["reason"])
        self.assertIn(self.a.display_name(), row["reason"])
        self.assertNotIn("stale", row["reason"])

    def test_failure_row_falls_back_to_session_state(self) -> None:
        self.view._file_queue = self._queue()
        self.view._pipeline.session_state["failed_nodes"] = [self.a.iid]
        self.view._record_queue_result({}, 1, [], {})
        row = self.view._file_queue["results"][0]
        self.assertIn(self.a.display_name(), row["reason"])


class FailureReportTests(unittest.TestCase):
    """End-of-run summary: failed nodes, reasons, session folder (item 44)."""

    @staticmethod
    def _report(**kwargs) -> str:
        from lynceus.ui.nodes.canvas import failure_report_text

        return failure_report_text(**kwargs)

    def test_lists_nodes_reasons_and_session(self) -> None:
        from lynceus.plugins.locale import t

        text = self._report(
            failed=2,
            failed_nodes={"b": "kaboom", "a": "root cause"},
            names={"b": "CHM"},
            peak_mb=1234.567,
            session_dir="C:/out/ses",
        )
        lines = text.splitlines()
        self.assertEqual(
            lines[0], t("Run finished with {n} failed task(s)").format(n=2))
        # Sorted by iid; "a" has no display name (iid fallback).
        self.assertEqual(lines[1], "• a: root cause")
        self.assertEqual(lines[2], "• CHM: kaboom")
        self.assertEqual(
            lines[3], t("peak RAM {peak} MB").format(peak=1234.6))
        self.assertEqual(
            lines[4], t("Session folder: {path}").format(path="C:/out/ses"))

    def test_reason_truncated_single_line(self) -> None:
        from lynceus.ui.nodes.canvas import FAILURE_REASON_MAX

        text = self._report(
            failed=1, failed_nodes={"n": "x" * 500 + "\nsecond line"})
        shown = text.splitlines()[1].split(": ", 1)[1]
        self.assertLessEqual(len(shown), FAILURE_REASON_MAX)
        self.assertTrue(shown.endswith("…"))
        self.assertNotIn("second line", shown)

    def test_minimal_report_without_context(self) -> None:
        self.assertEqual(len(self._report(failed=3).splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
