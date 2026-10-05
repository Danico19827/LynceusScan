# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Tests for Concatenate Tables (stacking semantics, not key joins)."""

import csv
import sqlite3
import tempfile
import unittest
from pathlib import Path

from lynceus.nodes.table.concatenate_tables import barrier_concatenate_tables


def _write_csv(path: Path, header: list[str], rows: list[list[str]],
               delimiter: str = ",") -> Path:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter=delimiter)
        if header:
            writer.writerow(header)
        writer.writerows(rows)
    return path


def _write_gpkg(path: Path, columns: list[str], rows: list[tuple]) -> Path:
    con = sqlite3.connect(str(path))
    try:
        defs = ", ".join(f'"{c}" TEXT' for c in columns)
        con.execute(f'CREATE TABLE data ({defs})')
        con.executemany(
            f"INSERT INTO data VALUES ({', '.join('?' * len(columns))})", rows
        )
        con.commit()
    finally:
        con.close()
    return path


def _read_result(path: Path, delimiter: str = ",") -> tuple[list[str], list[list[str]]]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        rows = [row for row in reader if row]
    if not rows:
        return [], []
    return rows[0], rows[1:]


def _ctx(session: Path, extra: dict | None = None) -> dict:
    session.mkdir(parents=True, exist_ok=True)
    ctx = {"session_dir": str(session)}
    if extra:
        ctx.update(extra)
    return ctx


class ConcatenateTablesTests(unittest.TestCase):
    def test_union_headers_and_empty_cells(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = _write_csv(root / "a.csv", ["x", "y"], [["1", "2"]])
            b = _write_csv(root / "b.csv", ["y", "z"], [["3", "4"]])
            out = barrier_concatenate_tables(
                _ctx(root / "session", {"in0_path": str(a), "in1_path": str(b)})
            )
            header, rows = _read_result(Path(out["file"]))
            self.assertEqual(header, ["x", "y", "z"])
            self.assertEqual(rows, [["1", "2", ""], ["", "3", "4"]])
            self.assertEqual(out["rows"], 2)
            self.assertNotIn("warnings", out)

    def test_header_only_inputs_keep_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = _write_csv(root / "a.csv", ["x", "y"], [])
            b = _write_csv(root / "b.csv", ["y", "z"], [])
            out = barrier_concatenate_tables(
                _ctx(root / "session", {"in0_path": str(a), "in1_path": str(b)})
            )
            header, rows = _read_result(Path(out["file"]))
            self.assertEqual(header, ["x", "y", "z"])
            self.assertEqual(rows, [])
            self.assertEqual(out["rows"], 0)
            self.assertTrue(out.get("warnings"))

    def test_fully_empty_inputs_stay_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = _write_csv(root / "a.csv", [], [])
            b = _write_csv(root / "b.csv", [], [])
            out = barrier_concatenate_tables(
                _ctx(root / "session", {"in0_path": str(a), "in1_path": str(b)})
            )
            self.assertEqual(Path(out["file"]).read_text(encoding="utf-8"), "")

    def test_delimiter_respected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = _write_csv(root / "a.csv", ["x"], [["1"]], delimiter=";")
            b = _write_csv(root / "b.csv", ["x"], [["2"]], delimiter=";")
            out = barrier_concatenate_tables(
                _ctx(root / "session", {
                    "in0_path": str(a), "in1_path": str(b), "delimiter": ";",
                })
            )
            header, rows = _read_result(Path(out["file"]), delimiter=";")
            self.assertEqual(header, ["x"])
            self.assertEqual(rows, [["1"], ["2"]])

    def test_gpkg_geometry_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = _write_csv(root / "a.csv", ["id"], [["7"]])
            b = _write_gpkg(root / "b.gpkg", ["id", "geom"], [("8", "POINT(0 0)")])
            out = barrier_concatenate_tables(
                _ctx(root / "session", {"in0_path": str(a), "in1_path": str(b)})
            )
            header, rows = _read_result(Path(out["file"]))
            self.assertEqual(header, ["id"])
            self.assertEqual(rows, [["7"], ["8"]])

    def test_missing_side_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = _write_csv(root / "a.csv", ["x"], [["1"]])
            with self.assertRaises(RuntimeError):
                barrier_concatenate_tables(
                    _ctx(root / "session", {
                        "in0_path": str(a),
                        "in1_path": str(root / "nope.csv"),
                    })
                )


if __name__ == "__main__":
    unittest.main()
