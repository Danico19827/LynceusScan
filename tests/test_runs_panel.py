# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Tests for session deletion robustness (S2).

Deleting a session must really remove it from disk (a silent leftover
keeps feeding the reuse cache while the panel no longer lists it).
Qt-free: only the pure deletion helpers (no widgets instantiated).
"""

from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from lynceus.ui.runs_panel import (
    _is_session_dir,
    _iter_cache_files,
    _purge_cache_dirs,
    _remove_session_dir,
    _session_children,
    _tiles_cache_dirs,
    _tiles_cache_size,
)


def _make_session(root: Path, name: str = "20260101_000000") -> Path:
    sess = root / name
    (sess / "artifacts" / "a").mkdir(parents=True)
    (sess / "artifacts" / "a" / "dtm_mosaic.tif").write_bytes(b"mosaic")
    (sess / "provenance.json").write_bytes(b"{}")
    return sess


class SessionDirTests(unittest.TestCase):
    def test_digit_start_is_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sess = _make_session(Path(tmp))
            self.assertTrue(_is_session_dir(sess))

    def test_cache_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache" / "tiles" / "20260101_000000"
            cache.mkdir(parents=True)
            self.assertFalse(_is_session_dir(cache))

    def test_files_are_not_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "20260101_000000.txt"
            target.write_bytes(b"x")
            self.assertFalse(_is_session_dir(target))

    def test_temp_tiles_never_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fp_dir = Path(tmp) / "_tiles_tmp_x" / "0bda4fb085b8f638"
            fp_dir.mkdir(parents=True)
            (fp_dir / "t.laz").write_bytes(b"x")
            self.assertFalse(_is_session_dir(fp_dir))


class SessionChildrenTests(unittest.TestCase):
    def test_lists_children(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sess = _make_session(Path(tmp))
            names = [p.name for p in _session_children(sess)]
            self.assertIn("artifacts", names)
            self.assertIn("provenance.json", names)

    def test_vanished_session_yields_no_children(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "20260101_000000"
            self.assertEqual(_session_children(missing), [])


class SessionsInTests(unittest.TestCase):
    def _tree(self, root: Path) -> None:
        for rel in (
            "las/20260102_000000",
            "las/20260101_000000",
            "yuto/20260101_120000",
        ):
            (root / rel).mkdir(parents=True)
        (root / "_tiles_tmp_x" / "0bda1234").mkdir(parents=True)
        (root / "las" / "notes.txt").write_bytes(b"x")

    def test_collects_nested_sessions_skipping_temps(self) -> None:
        from lynceus.ui.runs_panel import _sessions_in

        with tempfile.TemporaryDirectory() as tmp:
            self._tree(Path(tmp))
            found = sorted(
                p.relative_to(tmp).as_posix()
                for p in _sessions_in(Path(tmp))
            )
            self.assertEqual(
                found,
                [
                    "las/20260101_000000",
                    "las/20260102_000000",
                    "yuto/20260101_120000",
                ],
            )

    def test_empty_folder_yields_nothing(self) -> None:
        from lynceus.ui.runs_panel import _sessions_in

        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(_sessions_in(Path(tmp)), [])


class TilesCacheTests(unittest.TestCase):
    def _cache_tree(self, root: Path) -> None:
        loose = root / "cache" / "tiles" / "fp1"
        loose.mkdir(parents=True)
        (loose / "t0.laz").write_bytes(b"0" * 100)
        proj = root / "proj" / "cache" / "tiles" / "fp2"
        proj.mkdir(parents=True)
        (proj / "t0.laz").write_bytes(b"0" * 200)
        deep = root / "proj" / "source" / "cache" / "tiles" / "fp3"
        deep.mkdir(parents=True)
        (deep / "t0.laz").write_bytes(b"0" * 300)

    def test_finds_loose_and_project_caches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._cache_tree(Path(tmp))
            found = _tiles_cache_dirs(Path(tmp))
            self.assertEqual(len(found), 3)

    def test_size_sums_all_caches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._cache_tree(Path(tmp))
            self.assertEqual(_tiles_cache_size(Path(tmp)), 600)

    def test_empty_root_has_no_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(_tiles_cache_dirs(Path(tmp)), [])
            self.assertEqual(_tiles_cache_size(Path(tmp)), 0)

    def test_purge_removes_cache_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._cache_tree(Path(tmp))
            for cache in _tiles_cache_dirs(Path(tmp)):
                _remove_session_dir(cache)
            self.assertEqual(_tiles_cache_size(Path(tmp)), 0)


class GroupOriginsTests(unittest.TestCase):
    def _origins(self, root: Path) -> dict:
        sessions = {}
        for rel in (
            "_default_project/las/20260102_000000",
            "_default_project/las/20260101_000000",
            "_default_project/yuto/20260101_120000",
            "myproj/las/20260103_000000",
            "_default_project/_no_source/20260101_000000",
        ):
            origin = root / Path(rel).parent
            sessions.setdefault(origin, []).append(root / rel)
        return sessions

    def test_default_project_groups_like_others(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            from lynceus.ui.runs_panel import RunsPanel

            grouped = RunsPanel._group_origins(
                self._origins(Path(tmp)), Path(tmp)
            )
            groups = {group: dict(sources) for group, sources in grouped}
            self.assertEqual(set(groups), {"_default_project", "myproj"})
            self.assertEqual(
                set(groups["_default_project"]), {"las", "yuto", "_no_source"}
            )
            self.assertEqual(
                [s.name for s in groups["_default_project"]["las"]],
                ["20260102_000000", "20260101_000000"],
            )

    def test_loose_sessions_list_directly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            from lynceus.ui.runs_panel import RunsPanel

            root = Path(tmp)
            origins = {root: [root / "20260101_000000"]}
            grouped = RunsPanel._group_origins(origins, root)
            self.assertEqual(len(grouped), 1)
            group, sources = grouped[0]
            self.assertEqual(group, "")
            self.assertEqual(sources[0][0], "")
            self.assertEqual(
                [s.name for s in sources[0][1]], ["20260101_000000"]
            )


class SessionBadgeTests(unittest.TestCase):
    def _write_meta(self, sess: Path, meta: dict) -> None:
        import json

        (sess / "session_meta.json").write_text(
            json.dumps(meta), encoding="utf-8"
        )

    def test_badge_with_quality_report(self) -> None:
        import json

        from lynceus.ui.runs_panel import RunsPanel

        with tempfile.TemporaryDirectory() as tmp:
            sess = Path(tmp) / "20260101_000000"
            sess.mkdir()
            (sess / "quality_report.json").write_text(
                json.dumps({
                    "verdict": "PASS",
                    "counts": {"products_written": 4},
                }),
                encoding="utf-8",
            )
            self.assertEqual(
                RunsPanel._session_badge(sess),
                "PASS · 20260101_000000 · 4 products",
            )

    def test_badge_adds_source_from_meta(self) -> None:
        import json

        from lynceus.ui.runs_panel import RunsPanel

        with tempfile.TemporaryDirectory() as tmp:
            sess = Path(tmp) / "20260101_000000"
            sess.mkdir()
            (sess / "quality_report.json").write_text(
                json.dumps({
                    "verdict": "PASS",
                    "counts": {"products_written": 4},
                }),
                encoding="utf-8",
            )
            self._write_meta(sess, {
                "pipeline_status": "completed",
                "source_file": "",
                "sources": [{"iid": "a1", "file": "/data/vuelo.laz"}],
            })
            self.assertEqual(
                RunsPanel._session_badge(sess),
                "PASS · 20260101_000000 · vuelo.laz · 4 products",
            )

    def test_badge_running_status_without_report(self) -> None:
        from lynceus.ui.runs_panel import RunsPanel

        with tempfile.TemporaryDirectory() as tmp:
            sess = Path(tmp) / "20260101_000000"
            sess.mkdir()
            self._write_meta(sess, {
                "pipeline_status": "running",
                "sources": [{"iid": "a1", "file": "/data/vuelo.laz"}],
            })
            self.assertEqual(
                RunsPanel._session_badge(sess),
                "RUNNING · 20260101_000000 · vuelo.laz",
            )

    def test_badge_verdict_beats_meta_status(self) -> None:
        import json

        from lynceus.ui.runs_panel import RunsPanel

        with tempfile.TemporaryDirectory() as tmp:
            sess = Path(tmp) / "20260101_000000"
            sess.mkdir()
            (sess / "quality_report.json").write_text(
                json.dumps({"verdict": "FAIL", "counts": {}}),
                encoding="utf-8",
            )
            self._write_meta(sess, {"pipeline_status": "completed"})
            self.assertEqual(
                RunsPanel._session_badge(sess),
                "FAIL · 20260101_000000",
            )

    def test_badge_cancelled_beats_fail_verdict(self) -> None:
        import json

        from lynceus.ui.runs_panel import RunsPanel

        with tempfile.TemporaryDirectory() as tmp:
            sess = Path(tmp) / "20260101_000000"
            sess.mkdir()
            (sess / "quality_report.json").write_text(
                json.dumps({"verdict": "FAIL", "counts": {}}),
                encoding="utf-8",
            )
            self._write_meta(sess, {"pipeline_status": "cancelled"})
            self.assertEqual(
                RunsPanel._session_badge(sess),
                "CANCELLED · 20260101_000000",
            )

    def test_badge_multi_source_lists_two_plus_count(self) -> None:
        from lynceus.ui.runs_panel import RunsPanel

        with tempfile.TemporaryDirectory() as tmp:
            sess = Path(tmp) / "20260101_000000"
            sess.mkdir()
            self._write_meta(sess, {
                "pipeline_status": "completed",
                "sources": [
                    {"file": "/d/a.laz"},
                    {"file": "/d/b.laz"},
                    {"file": "/d/c.laz"},
                ],
            })
            self.assertEqual(
                RunsPanel._session_badge(sess),
                "COMPLETED · 20260101_000000 · a.laz, b.laz +1",
            )

    def test_badge_tolerates_corrupt_metadata(self) -> None:
        from lynceus.ui.runs_panel import RunsPanel

        with tempfile.TemporaryDirectory() as tmp:
            sess = Path(tmp) / "20260101_000000"
            sess.mkdir()
            (sess / "quality_report.json").write_bytes(b"not json")
            (sess / "session_meta.json").write_bytes(b"[[[")
            self.assertEqual(
                RunsPanel._session_badge(sess), "20260101_000000"
            )

    def test_badge_without_report_is_plain(self) -> None:
        from lynceus.ui.runs_panel import RunsPanel

        with tempfile.TemporaryDirectory() as tmp:
            sess = Path(tmp) / "20260101_000000"
            sess.mkdir()
            self.assertEqual(
                RunsPanel._session_badge(sess), "20260101_000000"
            )


class RemoveSessionTests(unittest.TestCase):
    def test_removes_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sess = _make_session(Path(tmp))
            _remove_session_dir(sess)
            self.assertFalse(sess.exists())

    def test_removes_readonly_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sess = _make_session(Path(tmp))
            locked = sess / "artifacts" / "a" / "dtm_mosaic.tif"
            os.chmod(locked, stat.S_IREAD)
            try:
                _remove_session_dir(sess)
            finally:
                if locked.exists():
                    os.chmod(locked, stat.S_IWRITE)
            self.assertFalse(sess.exists())


class CollectPruneTests(unittest.TestCase):
    """Session collection never descends into tile/product containers."""

    def _tree(self, root: Path) -> None:
        # Real sessions (timestamp dirs, nested segment, _no_source).
        for rel in (
            "proj/src/20260101_000000",
            "proj/src/20260102_000000",
            "proj/src/lotes/segment_0000",
            "proj/src/_no_source/20260103_000000",
        ):
            (root / rel).mkdir(parents=True)
        # Decoys: session-looking dirs planted where products live.
        for rel in (
            "proj/src/20260101_000000/artifacts/dtm",
            "proj/src/20260101_000000/artifacts/20260109_000000",
            "proj/cache/tiles/fp/20260109_000000",
            "cache/tiles/fp2/20260109_000000",
        ):
            target = root / rel
            target.mkdir(parents=True)
            (target / "session_meta.json").write_text("{}", encoding="utf-8")
        for rel in (
            "proj/src/20260101_000000/artifacts/dtm/t0.tif",
            "cache/tiles/fp/t0.laz",
        ):
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"x")

    def test_collect_skips_product_subtrees(self) -> None:
        from lynceus.ui.runs_panel import RunsPanel

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._tree(root)
            panel = RunsPanel.__new__(RunsPanel)
            origins: dict = {}
            panel._collect_in(root, origins)
            found = sorted(
                str(p.relative_to(root).as_posix())
                for sessions in origins.values()
                for p in sessions
            )
            self.assertEqual(found, [
                "proj/src/20260101_000000",
                "proj/src/20260102_000000",
                "proj/src/_no_source/20260103_000000",
            ])

    def test_enable_probes_dirs_not_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertFalse(bool(_tiles_cache_dirs(root)))
            (root / "cache" / "tiles" / "fp").mkdir(parents=True)
            self.assertTrue(bool(_tiles_cache_dirs(root)))

    def test_iter_cache_files_stop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache" / "tiles" / "fp"
            cache.mkdir(parents=True)
            for i in range(600):
                (cache / f"t{i}.laz").write_bytes(b"x")
            seen = list(_iter_cache_files(root, lambda: True))
            self.assertEqual(seen, [])
            full = list(_iter_cache_files(root, None))
            self.assertEqual(len(full), 600)
            self.assertEqual(sum(size for _, size in full), 600)


class PurgeWorkerTests(unittest.TestCase):
    def _cache(self, root: Path, name: str = "cache", count: int = 6,
               locked: bool = True) -> tuple[Path, int]:
        base = root / name / "tiles" / "fp"
        base.mkdir(parents=True)
        for i in range(count):
            (base / f"t{i}.laz").write_bytes(b"0123456789")
        total = count
        if locked:
            blocked = base / "locked.laz"
            blocked.write_bytes(b"0123456789")
            os.chmod(blocked, stat.S_IREAD)
            total += 1
        return root / name, total

    def test_purge_deletes_and_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache, total = self._cache(root, count=600)
            seen: list[int] = []
            errors, files, freed = _purge_cache_dirs(
                root, None, seen.append, lambda: False
            )
            self.assertEqual(errors, 0)
            self.assertEqual(files, total)
            self.assertEqual(freed, total * 10)
            self.assertTrue(seen)
            self.assertFalse(cache.exists())

    def test_purge_stops_on_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = self._cache(root)[0]
            errors, files, freed = _purge_cache_dirs(
                root, lambda: True, None, lambda: False
            )
            # Aborted before touching anything (checked per cache dir).
            self.assertEqual((files, freed), (0, 0))
            self.assertTrue(cache.exists())

    def test_purge_aborts_when_run_starts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = self._cache(root, count=600)[0]
            calls = {"n": 0}

            def _active() -> bool:
                calls["n"] += 1
                return calls["n"] > 1

            errors, files, _freed = _purge_cache_dirs(
                root, None, None, _active
            )
            # First 512 gone, then the run check aborts the rest.
            self.assertEqual(files, 512)
            self.assertTrue(cache.exists())
            self.assertGreater(calls["n"], 1)


class PanelSignalsTests(unittest.TestCase):
    def test_large_payloads_do_not_overflow(self) -> None:
        from lynceus.ui.runs_panel import _PanelSignals

        seen: dict = {}
        signals = _PanelSignals()
        signals.cache_sized.connect(
            lambda seq, size: seen.update(seq=seq, size=size)
        )
        signals.cache_sized.emit(3, 68907621167)
        self.assertEqual(seen, {"seq": 3, "size": 68907621167})
        done: dict = {}
        signals.purge_finished.connect(
            lambda e, f, b, c: done.update(
                errors=e, files=f, freed=b, cancelled=c
            )
        )
        signals.purge_finished.emit(1, 205922, 68907621167, True)
        self.assertEqual(
            done,
            {"errors": 1, "files": 205922, "freed": 68907621167,
             "cancelled": True},
        )


class CollapsedTreeTests(unittest.TestCase):
    """Item 41: sessions list collapsed in one line; files load on expand."""

    def setUp(self) -> None:
        from unittest.mock import PropertyMock, patch

        from PySide6.QtWidgets import QApplication

        from lynceus.ui.runs_panel import RunsPanel

        self._app = QApplication.instance() or QApplication([])
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self._sessions: dict[str, Path] = {}
        for name, with_report in (
            ("20260101_000000", True),
            ("20260102_000000", False),
        ):
            sess = self.root / "proj" / "src" / name
            (sess / "artifacts").mkdir(parents=True)
            (sess / "artifacts" / "dtm_mosaic.tif").write_bytes(b"tif")
            (sess / "provenance.json").write_bytes(b"{}")
            import json

            (sess / "session_meta.json").write_text(
                json.dumps({
                    "pipeline_status": "completed",
                    "sources": [{"iid": "a1", "file": "/data/vuelo.laz"}],
                }),
                encoding="utf-8",
            )
            if with_report:
                (sess / "quality_report.json").write_text(
                    json.dumps({
                        "verdict": "PASS",
                        "counts": {"products_written": 4},
                    }),
                    encoding="utf-8",
                )
            # Deterministic newest-first order (collect sorts by mtime).
            stamp = 1_700_000_000 + int(name[6:8])
            os.utime(sess, (stamp, stamp))
            self._sessions[name] = sess
        self._patcher = patch.object(
            RunsPanel, "_always_root",
            new_callable=PropertyMock, return_value=self.root,
        )
        self._patcher.start()
        self.addCleanup(self._patcher.stop)
        self.panel = RunsPanel()
        self.addCleanup(self.panel.shutdown)
        self.addCleanup(self.panel.deleteLater)
        self.panel.refresh()

    def _row(self, path: Path):
        """Tree item holding a path, or None."""
        from PySide6.QtCore import Qt

        role = Qt.ItemDataRole.UserRole
        stack = [
            self.panel.tree.topLevelItem(i)
            for i in range(self.panel.tree.topLevelItemCount())
        ]
        while stack:
            item = stack.pop()
            if item.data(0, role) == str(path):
                return item
            for i in range(item.childCount()):
                stack.append(item.child(i))
        return None

    def test_counters_live_only_at_root(self) -> None:
        root = self.panel.tree.topLevelItem(0)
        self.assertTrue(root.text(0).endswith("[Runs: 2]"))
        group = root.child(0)
        self.assertEqual(group.text(0), "proj")
        source = group.child(0)
        self.assertEqual(source.text(0), "src")

    def test_sessions_collapsed_in_one_line(self) -> None:
        for name, _ in (("20260101_000000", 0), ("20260102_000000", 1)):
            row = self._row(self._sessions[name])
            self.assertIsNotNone(row)
            self.assertFalse(row.isExpanded())
            self.assertEqual(row.childCount(), 0)
        self.assertEqual(
            self._row(self._sessions["20260101_000000"]).text(0),
            "PASS · 20260101_000000 · vuelo.laz · 4 products",
        )
        self.assertEqual(
            self._row(self._sessions["20260102_000000"]).text(0),
            "COMPLETED · 20260102_000000 · vuelo.laz",
        )

    def test_newest_chain_expanded_down_to_session(self) -> None:
        root = self.panel.tree.topLevelItem(0)
        self.assertTrue(root.isExpanded())
        self.assertTrue(root.child(0).isExpanded())
        self.assertTrue(root.child(0).child(0).isExpanded())
        newest = self._row(self._sessions["20260102_000000"])
        self.assertFalse(newest.isExpanded())

    def test_expand_loads_files_once(self) -> None:
        row = self._row(self._sessions["20260101_000000"])
        row.setExpanded(True)
        names = sorted(row.child(i).text(0) for i in range(row.childCount()))
        self.assertEqual(
            names,
            ["artifacts", "provenance.json", "quality_report.json",
             "session_meta.json"],
        )
        count = row.childCount()
        row.setExpanded(False)
        row.setExpanded(True)
        self.assertEqual(row.childCount(), count)

    def test_empty_session_loses_expander(self) -> None:
        from PySide6.QtWidgets import QTreeWidgetItem

        empty = self.root / "proj" / "src" / "20260103_000000"
        empty.mkdir(parents=True)
        os.utime(empty, (1_700_000_099, 1_700_000_099))
        self.panel.refresh()
        row = self._row(empty)
        self.assertIsNotNone(row)
        row.setExpanded(True)
        self.assertEqual(row.childCount(), 0)
        self.assertEqual(
            row.childIndicatorPolicy(),
            QTreeWidgetItem.ChildIndicatorPolicy.DontShowIndicator,
        )

    def test_file_selection_survives_refresh(self) -> None:
        row = self._row(self._sessions["20260101_000000"])
        row.setExpanded(True)
        target = self._sessions["20260101_000000"] / "artifacts"
        leaf = self._row(target)
        self.assertIsNotNone(leaf)
        self.panel.tree.setCurrentItem(leaf)
        self.panel.refresh()
        current = self.panel.tree.currentItem()
        self.assertIsNotNone(current)
        from PySide6.QtCore import Qt

        self.assertEqual(
            current.data(0, Qt.ItemDataRole.UserRole), str(target))


if __name__ == "__main__":
    unittest.main()
