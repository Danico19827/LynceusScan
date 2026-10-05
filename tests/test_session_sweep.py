# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Tests for finals-only sessions (S1).

The end-of-run sweep keeps barrier/merge products at the artifacts root
while deleting per-tile intermediates in subdirectories. Stdlib +
engine/domain modules only (no Qt, no real runs).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lynceus.processing.controller import PipelineController

DTM = "lynceus.nodes.lidar.terrain.generate_dtm"
CLASSIFY = "lynceus.nodes.lidar.terrain.classify_ground"
INPUT_TABLE = "lynceus.nodes.table.input_table"
GRID = "lynceus.nodes.lidar.analysis.grid_metrics"
EXPORT = "lynceus.nodes.lidar.output.export_point_cloud"


def _make_session(root: Path) -> Path:
    sess = root / "20260101_000000"
    arts = sess / "artifacts"
    (arts / "dtm").mkdir(parents=True)
    (arts / "classified").mkdir(parents=True)
    (arts / "in1").mkdir(parents=True)
    (arts / "dtm" / "t0.tif").write_bytes(b"tile")
    (arts / "dtm_mosaic.tif").write_bytes(b"mosaic")
    (arts / "dtm_mosaic.qml").write_bytes(b"qml")
    (arts / "classified" / "t0.laz").write_bytes(b"cloud")
    (arts / "in1" / "table.csv").write_bytes(b"a,b\n1,2\n")
    prov = {
        "nodes": {
            "a": {
                "module": DTM,
                "fp": "x",
                "files": ["dtm/t0.tif", "dtm_mosaic.tif", "dtm_mosaic.qml"],
            },
            "c": {
                "module": CLASSIFY,
                "fp": "y",
                "files": ["classified/t0.laz"],
            },
            "in1": {
                "module": INPUT_TABLE,
                "fp": "z",
                "files": ["in1/table.csv"],
            },
        }
    }
    (sess / "provenance.json").write_text(json.dumps(prov), encoding="utf-8")
    return sess


class SweepTests(unittest.TestCase):
    def _controller(self) -> PipelineController:
        ctrl = PipelineController()
        ctrl._modules = {"a": DTM, "c": CLASSIFY, "in1": INPUT_TABLE}
        ctrl._run_configs = {}
        return ctrl

    def test_sweep_keeps_finals_deletes_tile_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sess = _make_session(Path(tmp))
            removed = self._controller()._sweep_session(sess)
            self.assertEqual(removed, 2)
            arts = sess / "artifacts"
            self.assertTrue((arts / "dtm_mosaic.tif").is_file())
            self.assertTrue((arts / "dtm_mosaic.qml").is_file())
            self.assertTrue((arts / "in1" / "table.csv").is_file())
            self.assertFalse((arts / "dtm").exists())
            self.assertFalse((arts / "classified").exists())
            self.assertTrue((sess / "provenance.json").is_file())

    def test_sweep_without_provenance_keeps_everything(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sess = Path(tmp) / "20260101_000000"
            arts = sess / "artifacts" / "a" / "dtm"
            arts.mkdir(parents=True)
            target = arts / "t0.tif"
            target.write_bytes(b"tile")
            removed = self._controller()._sweep_session(sess)
            self.assertEqual(removed, 0)
            self.assertTrue(target.is_file())

    def test_publishable_strips_tiles_only_when_off(self) -> None:
        outputs = {"a": {"file": "m.tif", "tiles": [{"file": "t.laz"}]}}
        off = PipelineController()
        off._run_keep_intermediates = False
        cleaned = off._publishable_outputs(outputs)
        self.assertEqual(cleaned, {"a": {"file": "m.tif"}})
        # Original payload untouched (publish works on the copy shape).
        self.assertIn("tiles", outputs["a"])
        on = PipelineController()
        on._run_keep_intermediates = True
        self.assertEqual(on._publishable_outputs(outputs), outputs)

    def test_tiles_dir_off_uses_run_temp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctrl = PipelineController()
            ctrl._run_keep_intermediates = False
            ctrl._run_tiles_tmp = None
            first = ctrl._run_tiles_dir_for(tmp, "fp1", "laz")
            second = ctrl._run_tiles_dir_for(tmp, "fp2", "laz")
            self.assertTrue(first.is_dir())
            self.assertEqual(first.parent, second.parent)
            self.assertTrue(first.parent.name.startswith("_tiles_tmp_"))
            self.assertNotEqual(first, second)

    def test_tiles_dir_on_uses_shared_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctrl = PipelineController()
            ctrl._run_keep_intermediates = True
            shared = ctrl._run_tiles_dir_for(tmp, "fp1", "laz")
            self.assertEqual(shared, Path(tmp) / "cache" / "tiles" / "fp1")


class BundleCompleteTests(unittest.TestCase):
    def test_all_present_is_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            arts = Path(tmp) / "artifacts"
            arts.mkdir()
            (arts / "m.tif").write_bytes(b"x")
            entry = {"files": ["m.tif"]}
            self.assertTrue(
                PipelineController._bundle_complete(Path(tmp), entry)
            )

    def test_one_missing_is_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            arts = Path(tmp) / "artifacts"
            arts.mkdir()
            (arts / "m.tif").write_bytes(b"x")
            entry = {"files": ["m.tif", "tiles/t0.tif"]}
            self.assertFalse(
                PipelineController._bundle_complete(Path(tmp), entry)
            )

    def test_no_files_counts_as_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(
                PipelineController._bundle_complete(Path(tmp), {"files": []})
            )


class RebasePayloadTests(unittest.TestCase):
    def test_rewrites_previous_session_paths(self) -> None:
        old = Path("/old/sess/artifacts")
        new = Path("/new/sess/artifacts")
        payload = {
            "file": str(old / "m.tif"),
            "csv": str(old / "m.csv"),
            "tiles": [{"file": str(old / "t" / "t0.laz")}],
            "count": 4,
        }
        rebased = PipelineController._rebase_payload(payload, "/old/sess", new)
        self.assertEqual(rebased["file"], str(new / "m.tif"))
        self.assertEqual(rebased["tiles"][0]["file"], str(new / "t" / "t0.laz"))
        self.assertEqual(rebased["count"], 4)

    def test_leaves_cache_and_external_paths(self) -> None:
        payload = {
            "file": "/cache/tiles/fp/t0.laz",
            "source": "/data/input.laz",
        }
        rebased = PipelineController._rebase_payload(
            payload, "/old/sess", Path("/new/sess/artifacts")
        )
        self.assertEqual(rebased, payload)


class GhostProductTests(unittest.TestCase):
    def _controller(self):
        from types import SimpleNamespace

        ctrl = PipelineController()
        ctrl._modules = {"g": "lynceus.nodes.lidar.analysis.grid_metrics"}
        ctrl._run_configs = {}
        messages = []
        statuses = []
        cb = SimpleNamespace(
            on_node_status=lambda iid, st: statuses.append((iid, st)),
            on_message=lambda text, kind="info": messages.append((text, kind)),
        )
        return ctrl, cb, messages, statuses

    def test_missing_barrier_file_is_rejected_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctrl, cb, messages, statuses = self._controller()
            ghost = str(Path(tmp) / "grid_metrics.gpkg")
            outputs = {"g": {"file": ghost, "kind": "grid_metrics"}}
            rejected = ctrl._reject_ghost_products(outputs, ["g"], {}, cb)
            self.assertEqual(rejected, 1)
            self.assertEqual(outputs, {})
            self.assertIn(("g", "error"), statuses)
            self.assertTrue(
                any(kind == "error" for _, kind in messages)
            )

    def test_existing_barrier_file_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctrl, cb, _messages, statuses = self._controller()
            real = str(Path(tmp) / "grid_metrics.gpkg")
            Path(real).write_bytes(b"x")
            outputs = {"g": {"file": real}}
            rejected = ctrl._reject_ghost_products(outputs, ["g"], {}, cb)
            self.assertEqual(rejected, 0)
            self.assertIn("g", outputs)
            self.assertEqual(statuses, [])

    def test_reused_payloads_are_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctrl, cb, _messages, statuses = self._controller()
            ghost = str(Path(tmp) / "grid_metrics.gpkg")
            outputs = {"g": {"file": ghost}}
            rejected = ctrl._reject_ghost_products(
                outputs, ["g"], {}, cb, exclude={"g"}
            )
            self.assertEqual(rejected, 0)
            self.assertIn("g", outputs)
            self.assertEqual(statuses, [])


class RoleSplitTests(unittest.TestCase):
    def test_dual_splits_by_depth(self) -> None:
        caps = {"tile_task": "t", "barrier_task": "b"}
        barrier, tile = PipelineController._entry_role_files(
            caps, ["m.tif", "m.qml", "tiles/t0.tif", "cls/t0.laz"]
        )
        self.assertEqual(sorted(barrier), ["m.qml", "m.tif"])
        self.assertEqual(sorted(tile), ["cls/t0.laz", "tiles/t0.tif"])

    def test_pure_roles_take_whole_bundle(self) -> None:
        barrier, tile = PipelineController._entry_role_files(
            {"barrier_task": "b"}, ["a/b/c.laz", "m.tif"]
        )
        self.assertEqual(sorted(barrier), ["a/b/c.laz", "m.tif"])
        self.assertEqual(tile, [])
        barrier, tile = PipelineController._entry_role_files(
            {"tile_task": "t"}, ["cls/t0.laz"]
        )
        self.assertEqual(barrier, [])
        self.assertEqual(tile, ["cls/t0.laz"])

    def test_no_tasks_no_roles(self) -> None:
        self.assertEqual(
            PipelineController._entry_role_files({}, ["m.tif"]), ([], [])
        )


def _prev_with(root: Path, name: str, entries: dict) -> Path:
    """Fake previous session: {iid: (module, fp, [files])} on disk."""
    prev = root / name
    arts = prev / "artifacts"
    nodes = {}
    for iid, (module, fp, files) in entries.items():
        for rel in files:
            target = arts / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"x")
        nodes[iid] = {"module": module, "fp": fp, "files": list(files)}
    (prev / "provenance.json").write_text(
        json.dumps({"nodes": nodes}), encoding="utf-8"
    )
    return prev


class EffectiveReuseTests(unittest.TestCase):
    GRID_FILES = [
        "grid_metrics/t0.gpkg",
        "grid_metrics.gpkg",
        "grid_metrics.csv",
    ]

    def _controller(self, modules: dict) -> PipelineController:
        ctrl = PipelineController()
        ctrl._modules = dict(modules)
        ctrl._run_configs = {}
        return ctrl

    def _matched(self, prev: Path, entries: dict) -> dict:
        data = json.loads((prev / "provenance.json").read_text(encoding="utf-8"))
        return {iid: (prev, data["nodes"][iid]) for iid in entries}

    def test_complete_bundles_reuse_everything(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prev = _prev_with(Path(tmp), "prev", {
                "c": (CLASSIFY, "F1", ["classified/t0.laz"]),
                "g": (GRID, "F2", list(self.GRID_FILES)),
            })
            ctrl = self._controller({"c": CLASSIFY, "g": GRID})
            matched = self._matched(prev, ["c", "g"])
            tile, barrier = ctrl._effective_reuse(
                ["c", "g"], [("c", "g", "point_cloud", "point_cloud")],
                matched, {"c": {"strategy": "pmf"}},
            )
            self.assertEqual(tile, {"c", "g"})
            self.assertEqual(barrier, {"g"})

    def test_swept_chain_reuses_without_files(self) -> None:
        # The reported case: merged finals kept, tile outputs swept.
        # Barrier reuses (finals present) -> grid tiles relax (own
        # barrier reuses) -> classify relaxes (consumers reuse).
        with tempfile.TemporaryDirectory() as tmp:
            prev = _prev_with(Path(tmp), "prev", {
                "c": (CLASSIFY, "F1", ["classified/t0.laz"]),
                "g": (GRID, "F2", list(self.GRID_FILES)),
            })
            for rel in ("classified/t0.laz", "grid_metrics/t0.gpkg"):
                (prev / "artifacts" / rel).unlink()
            ctrl = self._controller({"c": CLASSIFY, "g": GRID})
            matched = self._matched(prev, ["c", "g"])
            tile, barrier = ctrl._effective_reuse(
                ["c", "g"], [("c", "g", "point_cloud", "point_cloud")],
                matched, {"c": {"strategy": "pmf"}},
            )
            self.assertEqual(tile, {"c", "g"})
            self.assertEqual(barrier, {"g"})

    def test_recomputing_consumer_forces_provider_files(self) -> None:
        # Grid unmatched (fp changed): its tiles recompute reading the
        # stream, so swept classify must recompute too.
        with tempfile.TemporaryDirectory() as tmp:
            prev = _prev_with(Path(tmp), "prev", {
                "c": (CLASSIFY, "F1", ["classified/t0.laz"]),
            })
            (prev / "artifacts" / "classified" / "t0.laz").unlink()
            ctrl = self._controller({"c": CLASSIFY, "g": GRID})
            matched = self._matched(prev, ["c"])
            tile, barrier = ctrl._effective_reuse(
                ["c", "g"], [("c", "g", "point_cloud", "point_cloud")],
                matched, {"c": {"strategy": "pmf"}},
            )
            self.assertEqual(tile, set())
            self.assertEqual(barrier, set())

    def test_stream_barrier_consumer_blocks_relaxation(self) -> None:
        # Export (pure barrier over POINT_CLOUD) recomputing needs the
        # stream: with both missing, nothing reuses.
        with tempfile.TemporaryDirectory() as tmp:
            prev = _prev_with(Path(tmp), "prev", {
                "c": (CLASSIFY, "F1", ["classified/t0.laz"]),
                "e": (EXPORT, "F3", ["point_cloud_exports/e/export.laz"]),
            })
            arts = prev / "artifacts"
            (arts / "classified" / "t0.laz").unlink()
            (arts / "point_cloud_exports" / "e" / "export.laz").unlink()
            ctrl = self._controller({"c": CLASSIFY, "e": EXPORT})
            edges = [("c", "e", "point_cloud", "point_cloud")]
            matched = self._matched(prev, ["c", "e"])
            tile, barrier = ctrl._effective_reuse(
                ["c", "e"], edges, matched, {"c": {"strategy": "pmf"}}
            )
            self.assertEqual(tile, set())
            self.assertEqual(barrier, set())
            # Stream back: classify reuses by files, export still recomputes.
            (arts / "classified" / "t0.laz").write_bytes(b"x")
            matched = self._matched(prev, ["c", "e"])
            tile, barrier = ctrl._effective_reuse(
                ["c", "e"], edges, matched, {"c": {"strategy": "pmf"}}
            )
            self.assertEqual(tile, {"c"})
            self.assertEqual(barrier, set())
            # Export back, stream swept: classify relaxes (reader reuses).
            (arts / "classified" / "t0.laz").unlink()
            (arts / "point_cloud_exports" / "e" / "export.laz").write_bytes(b"x")
            matched = self._matched(prev, ["c", "e"])
            tile, barrier = ctrl._effective_reuse(
                ["c", "e"], edges, matched, {"c": {"strategy": "pmf"}}
            )
            self.assertEqual(tile, {"c"})
            self.assertEqual(barrier, {"e"})


class DropSkippedTasksTests(unittest.TestCase):
    def _task(self, task_id, iid, kind, deps=()):
        from types import SimpleNamespace

        return SimpleNamespace(
            task_id=task_id, iid=iid, kind=kind, deps=set(deps)
        )

    def test_drops_reused_tiles_and_strips_deps(self) -> None:
        tasks = [
            self._task("c|t0", "c", "tile"),
            self._task("g|t0", "g", "tile", {"c|t0"}),
            self._task("g|merge", "g", "barrier", {"g|t0"}),
        ]
        kept = PipelineController._drop_skipped_tile_tasks(tasks, {"c"})
        self.assertEqual([t.task_id for t in kept], ["g|t0", "g|merge"])
        by_id = {t.task_id: t for t in kept}
        self.assertEqual(by_id["g|t0"].deps, set())

    def test_barriers_never_dropped(self) -> None:
        tasks = [self._task("g|merge", "g", "barrier")]
        kept = PipelineController._drop_skipped_tile_tasks(tasks, {"g"})
        self.assertEqual(len(kept), 1)

    def test_empty_skip_set_returns_same_list(self) -> None:
        tasks = [self._task("c|t0", "c", "tile")]
        self.assertIs(
            PipelineController._drop_skipped_tile_tasks(tasks, set()), tasks
        )


if __name__ == "__main__":
    unittest.main()
