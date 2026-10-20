# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Tests for the headless CLI (no Qt, no GUI).

Covers exit codes, registry listing, validation gates and one real
end-to-end barrier run (concatenate two CSVs). A subprocess test proves
a fresh interpreter runs without Qt.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from lynceus.cli import (
    EXIT_BLOCKED,
    EXIT_OK,
    EXIT_USAGE,
    main,
)

INPUT_TABLE = "lynceus.nodes.table.input_table"
CONCAT = "lynceus.nodes.table.concatenate_tables"


def _retry_rmtree(path) -> None:
    """Remove a tree, retrying briefly: on Windows a pool worker can still
    hold a session file for a moment after on_finished (WinError 32)."""
    import shutil
    import time

    for _ in range(100):
        try:
            shutil.rmtree(path)
            return
        except OSError:
            time.sleep(0.2)
    shutil.rmtree(path, ignore_errors=True)


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> Path:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _concat_project(path: Path, first: Path, second: Path) -> Path:
    doc = {
        "format": "lynceus-project",
        "version": 2,
        "nodes": [
            {"iid": "in1", "id": INPUT_TABLE, "x": 0.0, "y": 0.0,
             "label": "", "config": {"pipeline": {"file_path": str(first)}}},
            {"iid": "in2", "id": INPUT_TABLE, "x": 0.0, "y": 120.0,
             "label": "", "config": {"pipeline": {"file_path": str(second)}}},
            {"iid": "cat", "id": CONCAT, "x": 280.0, "y": 60.0,
             "label": "", "config": {}},
        ],
        "edges": [
            {"src": "in1", "src_out": 0, "dst": "cat", "dst_in": 0,
             "out_type": "table_csv", "in_type": "table"},
            {"src": "in2", "src_out": 0, "dst": "cat", "dst_in": 1,
             "out_type": "table_csv", "in_type": "table"},
        ],
    }
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


class CliAppDataMixin(unittest.TestCase):
    def setUp(self) -> None:
        self._root = Path(tempfile.mkdtemp(prefix="lynceus_cli_"))
        self.addCleanup(_retry_rmtree, self._root)
        self._old_appdata = os.environ.get("APPDATA")
        os.environ["APPDATA"] = str(self._root / "appdata")
        from lynceus.plugins.registry import manager

        self._old_disabled = manager.disabled_provider
        self.addCleanup(setattr, manager, "disabled_provider",
                        self._old_disabled)

    def tearDown(self) -> None:
        if self._old_appdata is None:
            os.environ.pop("APPDATA", None)
        else:
            os.environ["APPDATA"] = self._old_appdata

    def _seed_terms(self) -> None:
        from lynceus.plugins import terms
        from lynceus.plugins.store import ConsentStore

        ConsentStore().terms_accept(terms.TERMS_VERSION, terms.terms_blob())

    def _run_main(self, argv: list[str]):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(argv)
        return code, out.getvalue(), err.getvalue()


class NodesListTests(CliAppDataMixin):
    def test_list_lists_builtin_nodes(self) -> None:
        code, out, _err = self._run_main(["nodes", "list"])
        self.assertEqual(code, EXIT_OK)
        self.assertIn("lynceus.nodes.table.concatenate_tables", out)

    def test_list_json_parses(self) -> None:
        code, out, _err = self._run_main(["nodes", "list", "--json"])
        self.assertEqual(code, EXIT_OK)
        rows = json.loads(out)
        by_id = {row["id"]: row for row in rows}
        self.assertIn("lynceus.nodes.table.concatenate_tables", by_id)
        self.assertEqual(by_id[INPUT_TABLE]["category"], "Table")


class ValidateTests(CliAppDataMixin):
    def test_missing_file_is_usage_error(self) -> None:
        code, out, _err = self._run_main(
            ["validate", str(self._root / "nope.lynx")])
        self.assertEqual(code, EXIT_USAGE)
        self.assertFalse(json.loads(out)["valid"])

    def test_unknown_node_is_usage_error(self) -> None:
        proj = self._root / "bad.lynx"
        proj.write_text(json.dumps({
            "format": "lynceus-project", "version": 2,
            "nodes": [{"iid": "x", "id": "no.such.node", "config": {}}],
            "edges": [],
        }), encoding="utf-8")
        code, out, _err = self._run_main(["validate", str(proj)])
        self.assertEqual(code, EXIT_USAGE)
        self.assertIn("no.such.node", out)

    def test_empty_graph_fails(self) -> None:
        proj = self._root / "empty.lynx"
        proj.write_text(json.dumps({
            "format": "lynceus-project", "version": 2,
            "nodes": [], "edges": [],
        }), encoding="utf-8")
        code, out, _err = self._run_main(["validate", str(proj)])
        self.assertEqual(code, EXIT_USAGE)


class TermsGateTests(CliAppDataMixin):
    def test_run_without_terms_is_blocked(self) -> None:
        root = self._root
        _write_csv(root / "a.csv", ["x"], [["1"]])
        _write_csv(root / "b.csv", ["x"], [["2"]])
        proj = _concat_project(root / "p.lynx", root / "a.csv", root / "b.csv")
        code, _out, err = self._run_main(
            ["run", str(proj), "--output", str(root / "out")])
        self.assertEqual(code, EXIT_BLOCKED)
        self.assertIn("Terms", err)


class RunTests(CliAppDataMixin):
    def test_concatenate_end_to_end(self) -> None:
        from lynceus.cli import EXIT_OK as OK

        self._seed_terms()
        root = self._root
        _write_csv(root / "a.csv", ["x", "y"], [["1", "2"]])
        _write_csv(root / "b.csv", ["y", "z"], [["3", "4"]])
        proj = _concat_project(root / "p.lynx", root / "a.csv", root / "b.csv")
        code, out, _err = self._run_main(
            ["run", str(proj), "--output", str(root / "out"), "--json"])
        self.assertEqual(code, OK)
        report = json.loads(out)
        self.assertEqual(report["status"], "ok")
        self.assertTrue(report["session_dir"])
        self.assertGreaterEqual(len(report["products"]), 1)
        merged = Path(report["products"][0]["path"])
        self.assertTrue(merged.is_file())
        with open(merged, newline="", encoding="utf-8") as handle:
            rows = [row for row in csv.reader(handle) if row]
        self.assertEqual(rows[0], ["x", "y", "z"])
        self.assertEqual(len(rows), 3)

    def test_source_overrides_any_file_input(self) -> None:
        self._seed_terms()
        root = self._root
        _write_csv(root / "a.csv", ["x"], [["1"]])
        _write_csv(root / "b.csv", ["x"], [["2"]])
        _write_csv(root / "c.csv", ["x"], [["3"]])
        proj = _concat_project(root / "p.lynx", root / "a.csv", root / "b.csv")
        code, out, _err = self._run_main(
            ["run", str(proj), "--output", str(root / "out"), "--json",
             "--source", f"in2={root / 'c.csv'}"])
        self.assertEqual(code, EXIT_OK)
        report = json.loads(out)
        self.assertEqual(report["status"], "ok")
        merged = Path(report["products"][0]["path"])
        with open(merged, newline="", encoding="utf-8") as handle:
            rows = [row for row in csv.reader(handle) if row]
        self.assertEqual([row[0] for row in rows[1:]], ["1", "3"])

    def test_source_unknown_iid_is_usage_error(self) -> None:
        self._seed_terms()
        root = self._root
        _write_csv(root / "a.csv", ["x"], [["1"]])
        _write_csv(root / "b.csv", ["x"], [["2"]])
        proj = _concat_project(root / "p.lynx", root / "a.csv", root / "b.csv")
        code, _out, err = self._run_main(
            ["run", str(proj), "--source", "nope=/tmp/x.csv"])
        self.assertEqual(code, EXIT_USAGE)
        self.assertIn("not a file input", err)

    def test_run_missing_project_is_usage_error(self) -> None:
        self._seed_terms()
        code, _out, err = self._run_main(
            ["run", str(self._root / "nope.lynx")])
        self.assertEqual(code, EXIT_USAGE)
        self.assertIn("Project not found", err)

    def test_run_missing_source_is_blocked(self) -> None:
        self._seed_terms()
        root = self._root
        proj = Path(root / "p.lynx")
        proj.write_text(json.dumps({
            "format": "lynceus-project", "version": 2,
            "nodes": [
                {"iid": "l1",
                 "id": "lynceus.nodes.lidar.source.load_las_laz",
                 "x": 0.0, "y": 0.0, "label": "",
                 "config": {"file_path": str(root / "gone.laz"),
                            "pipeline": {}}},
                {"iid": "d1", "id": "lynceus.nodes.lidar.clean.decimate",
                 "x": 280.0, "y": 0.0, "label": "",
                 "config": {"pipeline": {}}},
            ],
            "edges": [
                {"src": "l1", "src_out": 0, "dst": "d1", "dst_in": 0,
                 "out_type": "point_cloud", "in_type": "point_cloud"},
            ],
        }), encoding="utf-8")
        code, _out, _err = self._run_main(
            ["run", str(proj), "--output", str(root / "out")])
        self.assertEqual(code, EXIT_BLOCKED)


class HeadlessSubprocessTests(unittest.TestCase):
    def test_fresh_interpreter_has_no_qt(self) -> None:
        import subprocess
        import sys

        proc = subprocess.run(
            [sys.executable, "-m", "lynceus.cli", "nodes", "list"],
            capture_output=True, text=True, cwd=str(Path.cwd()),
            timeout=300,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertIn("lynceus.nodes.table.concatenate_tables", proc.stdout)


if __name__ == "__main__":
    unittest.main()
