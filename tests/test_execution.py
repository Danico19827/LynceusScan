# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Execution behavior: cancel, restore tolerance, disable flow, i18n side.

Thread pools only (same run_dag code path, no Windows spawn in-suite):
cancel marks pending failed; unknown saved config keys travel harmlessly;
disabling with an open project marks nodes unavailable; sidecars merge
into enabled languages; long/unicode filenames copy through barriers.
"""

import os
import tempfile
import unittest
from multiprocessing.pool import ThreadPool
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _ok(_ctx=None):
    return {"file": "x"}


class CancelTests(unittest.TestCase):
    def test_preset_cancel_fails_everything_fast(self) -> None:
        from lynceus.processing.executor import run_dag
        from lynceus.processing.steps import Task

        tasks = [
            Task(
                task_id=f"t{i}",
                iid=f"n{i}",
                module_id="m",
                kind="barrier",
                fn=_ok,
                args=({"v": i},),
            )
            for i in range(4)
        ]
        pool = ThreadPool(processes=2)
        try:
            outputs, _tiles, failed, _stats = run_dag(
                pool, tasks, cancel_flag=lambda: True
            )
        finally:
            pool.close()
            pool.join()
        self.assertEqual(outputs, {})
        self.assertEqual(failed, 4)

    def test_cancel_mid_run_stops_dispatch(self) -> None:
        import threading
        import time

        from lynceus.processing.executor import run_dag
        from lynceus.processing.steps import Task

        started = threading.Event()

        def slow(_ctx=None):
            started.set()
            time.sleep(0.5)
            return {"file": "slow"}

        tasks = [
            Task(
                task_id="slow",
                iid="n0",
                module_id="m",
                kind="barrier",
                fn=slow,
                args=({},),
            ),
            Task(
                task_id="t1",
                iid="n1",
                module_id="m",
                kind="barrier",
                fn=_ok,
                args=({},),
                deps={"slow"},
            ),
        ]
        pool = ThreadPool(processes=1)
        try:
            # Cancel trips the moment the first task starts: dependents
            # never dispatch.
            outputs, _tiles, failed, _stats = run_dag(
                pool, tasks, cancel_flag=started.is_set
            )
        finally:
            pool.close()
            pool.join()
        self.assertTrue(started.wait(timeout=30))
        self.assertIn("n0", outputs)
        self.assertEqual(failed, 1)


class RestoreToleranceTests(unittest.TestCase):
    def test_unknown_saved_keys_travel_harmlessly(self) -> None:
        from PySide6.QtCore import QPointF
        from PySide6.QtWidgets import QApplication

        from lynceus.ui.nodes.canvas import NodeCanvasView

        QApplication.instance() or QApplication([])
        view = NodeCanvasView()
        try:
            item = view.add_node(
                "lynceus.nodes.lidar.source.load_las_laz", QPointF(0, 0)
            )
            self.assertIsNotNone(item)
            doc = {
                "nodes": [
                    {
                        "id": "lynceus.nodes.lidar.source.load_las_laz",
                        "iid": "x1",
                        "x": 0,
                        "y": 0,
                        "label": "",
                        "config": {
                            "pipeline": {
                                "tile_size_m": 50.0,
                                "removed_param": "old",
                                "renamed": 3,
                            }
                        },
                    }
                ],
                "edges": [],
            }
            warnings = view.restore_graph(doc)
            self.assertIsInstance(warnings, list)
            restored = next(
                i
                for i in view._node_items
                if i.iid == "x1"
            )
            config = restored.pipeline_config()
            self.assertEqual(config.get("tile_size_m"), 50.0)
            # Unknown keys travel without breaking the inspector.
            self.assertEqual(config.get("removed_param"), "old")
        finally:
            view.deleteLater()


class DisableFlowTests(unittest.TestCase):
    def test_disabled_marks_open_nodes_unavailable(self) -> None:
        from PySide6.QtCore import QPointF
        from PySide6.QtWidgets import QApplication

        from lynceus.plugins.registry import manager
        from lynceus.ui.nodes.canvas import NodeCanvasView

        QApplication.instance() or QApplication([])
        view = NodeCanvasView()
        try:
            item = view.add_node(
                "lynceus.nodes.lidar.source.load_las_laz", QPointF(0, 0)
            )
            self.assertTrue(item.is_available())
            with mock.patch.object(
                manager, "contains", return_value=False
            ):
                view.refresh_node_availability()
                self.assertFalse(item.is_available())
            view.refresh_node_availability()
            self.assertTrue(item.is_available())
        finally:
            view.deleteLater()


class SidecarMergeTests(unittest.TestCase):
    def test_sidecar_supplements_enabled_language(self) -> None:
        from lynceus.plugins.locale import _merge_catalog, _read_sidecar

        with tempfile.TemporaryDirectory() as tmp:
            node = Path(tmp) / "n.py"
            node.write_text("NODE_ID = 'x.y'\n", encoding="utf-8")
            (node.with_name("n.i18n.json")).write_text(
                '{"es": {"Hello": "Hola"}, "xx": {"Hello": "Xola"}}',
                encoding="utf-8",
            )
            data = _read_sidecar(node)
            catalogs: dict = {}
            _merge_catalog(catalogs, data, {"es"})
            self.assertEqual(catalogs["es"]["Hello"], "Hola")
            self.assertNotIn("xx", catalogs)


class LongPathTests(unittest.TestCase):
    def test_long_unicode_names_copy(self) -> None:
        from lynceus.nodes._product_input import barrier_input_file

        with tempfile.TemporaryDirectory() as tmp:
            name = "áéí_" + "n" * 180 + ".csv"
            src = Path(tmp) / name
            src.write_text("h\n1\n", encoding="utf-8")
            session = Path(tmp) / "session"
            ctx = {
                "session_dir": str(session),
                "file_path": str(src),
                "node_iid": "t1",
                "module_id": "m",
            }
            payload = barrier_input_file(
                ctx,
                {
                    "node_id": "m",
                    "node_name": "M",
                    "output_port": "table_csv",
                    "session_file": "table.csv",
                    "kind_label": "Table",
                    "kind_group": "table_csv",
                },
            )
            self.assertTrue(Path(payload["file"]).is_file())


if __name__ == "__main__":
    unittest.main()
