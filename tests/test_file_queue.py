# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Tests for the shared file-queue base (Qt-free) and engine overrides."""

from __future__ import annotations

import csv
import os
import tempfile
import threading
import unittest
from pathlib import Path

from lynceus.nodes import _file_queue as fq


class QueueMechanicsTests(unittest.TestCase):
    def test_normalize_dedupes_and_trims(self) -> None:
        self.assertEqual(
            fq.normalize_queue([" b ", "a", "b", "", None, 7, "a"]),
            ["b", "a"],
        )
        self.assertEqual(fq.normalize_queue(None), [])

    def test_set_replaces(self) -> None:
        self.assertEqual(fq.set_queue(["a", "b", "a"]), ["a", "b"])

    def test_add_skips_duplicates(self) -> None:
        queue, added = fq.add_to_queue(["a"], ["b", "a", "c"])
        self.assertEqual(queue, ["a", "b", "c"])
        self.assertEqual(added, 2)

    def test_remove_and_move(self) -> None:
        self.assertEqual(fq.remove_from_queue(["a", "b", "c"], 1), ["a", "c"])
        self.assertEqual(fq.remove_from_queue(["a"], 5), ["a"])
        self.assertEqual(
            fq.move_in_queue(["a", "b", "c"], 0, 2), ["b", "c", "a"])
        self.assertEqual(fq.move_in_queue(["a"], 0, 9), ["a"])

    def test_partition_existing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            real = Path(tmp) / "real.csv"
            real.write_text("x\n", encoding="utf-8")
            existing, missing = fq.partition_existing(
                [str(real), str(Path(tmp) / "gone.csv")])
            self.assertEqual(existing, [str(real)])
            self.assertEqual(missing, [str(Path(tmp) / "gone.csv")])

    def test_effective_file(self) -> None:
        self.assertEqual(fq.effective_file(["q1", "q2"], "single"), "q1")
        self.assertEqual(fq.effective_file([], "single"), "single")
        self.assertIsNone(fq.effective_file([], None))

    def test_count_text(self) -> None:
        self.assertIn("3", fq.queue_count_text(3))

    def test_summary_counts_and_failed_lines(self) -> None:
        text = fq.summarize_queue([
            {"file": "/d/a.csv", "ok": True, "reason": ""},
            {"file": "/d/b.csv", "ok": False, "reason": "boom\ndetail"},
        ])
        self.assertIn("1 of 2", text)
        self.assertIn("b.csv: boom", text)
        self.assertNotIn("a.csv", text)

    def test_summary_stopped_variant(self) -> None:
        text = fq.summarize_queue(
            [{"file": "a", "ok": True, "reason": ""}], stopped=True)
        self.assertIn("1 of 1", text)
        self.assertNotEqual(
            text,
            fq.summarize_queue([{"file": "a", "ok": True, "reason": ""}]),
        )


class EngineOverrideTests(unittest.TestCase):
    """file_overrides swap one input per run without touching the project."""

    def setUp(self) -> None:
        self._root = Path(tempfile.mkdtemp(prefix="lynceus_queue_"))
        self.addCleanup(_retry_rmtree, self._root)
        self._old_appdata = os.environ.get("APPDATA")
        os.environ["APPDATA"] = str(self._root / "appdata")
        from lynceus.plugins.registry import manager

        self._old_disabled = manager.disabled_provider
        self.addCleanup(
            setattr, manager, "disabled_provider", self._old_disabled)

    def tearDown(self) -> None:
        if self._old_appdata is None:
            os.environ.pop("APPDATA", None)
        else:
            os.environ["APPDATA"] = self._old_appdata

    def _write_csv(self, name: str, rows: list[list[str]]) -> Path:
        path = self._root / name
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["x"])
            writer.writerows(rows)
        return path

    def test_override_swaps_input_product(self) -> None:
        from lynceus.plugins import terms
        from lynceus.plugins.store import ConsentStore
        from lynceus.processing.controller import (
            PipelineCallbacks,
            PipelineController,
        )

        ConsentStore().terms_accept(terms.TERMS_VERSION, terms.terms_blob())
        first = self._write_csv("a.csv", [["1"]])
        second = self._write_csv("b.csv", [["2"]])
        other = self._write_csv("c.csv", [["3"]])
        nodes = [
            ("in1", "lynceus.nodes.table.input_table"),
            ("in2", "lynceus.nodes.table.input_table"),
            ("cat", "lynceus.nodes.table.concatenate_tables"),
        ]
        edges = [
            ("in1", "cat", "table_csv", "table"),
            ("in2", "cat", "table_csv", "table"),
        ]
        configs = {
            "in1": {"file_path": str(first)},
            "in2": {"file_path": str(second)},
        }
        state: dict = {}
        done = threading.Event()

        def on_finished(outputs, _results=None, failed: int = 0,
                        tiles=None, extra=None) -> None:
            state["outputs"] = outputs
            state["failed"] = int(failed or 0)
            done.set()

        ctrl = PipelineController()
        ctrl.run(
            nodes, edges, {},
            PipelineCallbacks(on_finished=on_finished),
            session_root=str(self._root / "out"),
            configs=configs,
            file_overrides={"in2": str(other)},
        )
        self.assertTrue(done.wait(timeout=300), "run finished")
        self.assertEqual(state.get("failed"), 0)
        payload = state["outputs"]["cat"]
        merged = Path(payload["file"])
        with open(merged, newline="", encoding="utf-8") as handle:
            rows = [row for row in csv.reader(handle) if row]
        # a.csv + c.csv (override), NOT b.csv.
        self.assertEqual([row[0] for row in rows[1:]], ["1", "3"])
        # The caller's configs were not mutated by the override.
        self.assertEqual(configs["in2"], {"file_path": str(second)})


def _retry_rmtree(path) -> None:
    import shutil
    import time

    for _ in range(100):
        try:
            shutil.rmtree(path)
            return
        except OSError:
            time.sleep(0.2)
    shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
