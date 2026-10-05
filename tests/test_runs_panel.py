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
                "20260101_000000 · PASS · 4 products",
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


if __name__ == "__main__":
    unittest.main()
