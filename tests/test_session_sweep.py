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


class ExportLayoutTests(unittest.TestCase):
    """Finals-only export layout: nested legacy vs flat (no cache)."""

    def _controller(self, modules: dict) -> PipelineController:
        ctrl = PipelineController()
        ctrl._modules = dict(modules)
        ctrl._run_configs = {}
        return ctrl

    def _names(self, order, modules) -> dict:
        return self._controller(modules)._export_filenames(order, {})

    def test_single_export_keeps_clean_name(self) -> None:
        names = self._names(["e1"], {"e1": EXPORT})
        self.assertEqual(
            names, {"e1": "point_cloud_consolidated.laz"})

    def test_two_exports_suffix_deterministically(self) -> None:
        modules = {"e1": EXPORT, "e2": EXPORT}
        first = self._names(["e2", "e1"], modules)
        self.assertEqual(first, {
            "e2": "point_cloud_consolidated.laz",
            "e1": "point_cloud_consolidated_2.laz",
        })
        again = self._names(["e2", "e1"], modules)
        self.assertEqual(again, first)
        # Non-export instances never claim names.
        self.assertEqual(
            self._names(["e1", "g1"], {"e1": EXPORT, "g1": GRID}), {
                "e1": "point_cloud_consolidated.laz"})

    def test_sweep_keeps_export_nested_and_flat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sess = Path(tmp) / "20260101_000000"
            arts = sess / "artifacts"
            nested = arts / "point_cloud_exports" / "e9"
            nested.mkdir(parents=True)
            (nested / "point_cloud_consolidated.laz").write_bytes(b"laz")
            (arts / "point_cloud_consolidated.laz").write_bytes(b"laz")
            (arts / "clean_flags").mkdir(parents=True)
            (arts / "clean_flags" / "t0.laz").write_bytes(b"laz")
            prov = {"nodes": {
                "e9": {
                    "module": EXPORT,
                    "fp": "w",
                    "files": [
                        "point_cloud_exports/e9/point_cloud_consolidated.laz",
                        "point_cloud_consolidated.laz",
                    ],
                },
            }}
            (sess / "provenance.json").write_text(json.dumps(prov))
            removed = self._controller({"e9": EXPORT})._sweep_session(sess)
            self.assertEqual(removed, 1)
            self.assertTrue(
                (nested / "point_cloud_consolidated.laz").is_file())
            self.assertTrue(
                (arts / "point_cloud_consolidated.laz").is_file())
            self.assertFalse((arts / "clean_flags").exists())

    def test_flat_export_reuses_by_glob(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prev = root / "prev"
            arts = prev / "artifacts"
            arts.mkdir(parents=True)
            (arts / "point_cloud_consolidated.laz").write_bytes(b"laz")
            entry = {
                "module": EXPORT,
                "fp": "w",
                "files": ["point_cloud_consolidated.laz"],
                "output": {"file": str(
                    arts / "point_cloud_consolidated.laz")},
            }
            ctrl = self._controller({"e9": EXPORT})
            found, _ = ctrl._find_reusable([(prev, entry)], set(), EXPORT, "w")
            self.assertEqual(found, prev)
            fresh = root / "fresh" / "artifacts"
            fresh.mkdir(parents=True)
            done = ctrl._materialize_bundle(prev, fresh, entry)
            self.assertEqual(done, ["point_cloud_consolidated.laz"])
            self.assertTrue(
                (fresh / "point_cloud_consolidated.laz").is_file())

    @staticmethod
    def _write_laz(path: Path, x0: float, y0: float, count: int) -> None:
        import laspy
        import numpy as np

        header = laspy.LasHeader(point_format=0, version="1.2")
        header.scales = np.array([0.01, 0.01, 0.01])
        header.offsets = np.zeros(3)
        record = laspy.ScaleAwarePointRecord.zeros(
            count,
            point_format=header.point_format,
            scales=header.scales,
            offsets=header.offsets,
        )
        record.x = np.full(count, x0 + 50.0)
        record.y = np.full(count, y0 + 50.0)
        record.z = np.zeros(count)
        with laspy.open(str(path), mode="w", header=header) as writer:
            writer.write_points(record)

    def _tile(self, arts: Path, tid: str, x0: float, count: int) -> dict:
        path = arts / f"{tid}.laz"
        self._write_laz(path, x0, 0.0, count)
        return {
            "tile_id": tid,
            "file": str(path),
            "core_x_min": x0,
            "core_y_min": 0.0,
            "core_x_max": x0 + 100.0,
            "core_y_max": 100.0,
        }

    def test_barrier_writes_flat_when_named(self) -> None:
        from lynceus.nodes.lidar.output.export_point_cloud import (
            barrier_export_point_cloud,
        )

        with tempfile.TemporaryDirectory() as tmp:
            arts = Path(tmp) / "artifacts"
            arts.mkdir(parents=True)
            tiles = [
                self._tile(arts, "t0", 0.0, 50),
                self._tile(arts, "t1", 100.0, 60),
            ]
            ctx = {
                "session_dir": str(arts),
                "node_iid": "e9",
                "export_filenames": {"e9": "point_cloud_consolidated.laz"},
                "_point_cloud_sources": [
                    {"point_cloud_dir": "", "tiles": tiles}],
                "crop_buffer": True,
                "intermediate_ext": "laz",
                "stage_name": "",
            }
            payload = barrier_export_point_cloud(ctx)
            flat = arts / "point_cloud_consolidated.laz"
            self.assertTrue(flat.is_file())
            self.assertEqual(payload["file"], str(flat))
            self.assertEqual(payload["count"], 110)
            self.assertFalse((arts / "point_cloud_exports").exists())

    def test_barrier_keeps_nested_without_map(self) -> None:
        from lynceus.nodes.lidar.output.export_point_cloud import (
            barrier_export_point_cloud,
        )

        with tempfile.TemporaryDirectory() as tmp:
            arts = Path(tmp) / "artifacts"
            arts.mkdir(parents=True)
            tiles = [self._tile(arts, "t0", 0.0, 50)]
            ctx = {
                "session_dir": str(arts),
                "node_iid": "e9",
                "_point_cloud_sources": [
                    {"point_cloud_dir": "", "tiles": tiles}],
                "crop_buffer": True,
                "intermediate_ext": "laz",
                "stage_name": "",
            }
            payload = barrier_export_point_cloud(ctx)
            nested = (arts / "point_cloud_exports" / "e9"
                      / "point_cloud_consolidated.laz")
            self.assertTrue(nested.is_file())
            self.assertEqual(payload["file"], str(nested))


class FailedProvenanceTests(unittest.TestCase):
    """Failed/cancelled batches still record provenance (sweep needs it)."""

    def test_provenance_written_without_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sess = Path(tmp) / "20260101_000000"
            arts = sess / "artifacts"
            arts.mkdir(parents=True)
            (arts / "dtm_mosaic.tif").write_bytes(b"mosaic")
            ctrl = PipelineController()
            ctrl._modules = {"a": DTM}
            ctrl._run_configs = {}
            nodes = ctrl._save_provenance(
                sess, {}, Path(tmp), {"a": "fp-a"}, ["a"], [],
                {}, {},
            )
            prov = json.loads((sess / "provenance.json").read_text())
            self.assertIn("a", prov["nodes"])
            self.assertEqual(
                nodes["a"]["files"], ["dtm_mosaic.tif"])
            self.assertEqual(prov["nodes"]["a"]["fp"], "fp-a")


class MergeSegmentsFlagTests(unittest.TestCase):
    """Independent segments: flag travels in fingerprints; prune rewires."""

    LOAD = "lynceus.nodes.lidar.source.load_las_laz"
    DTM = "lynceus.nodes.lidar.terrain.generate_dtm"

    def _ctrl(self) -> PipelineController:
        ctrl = PipelineController()
        ctrl._modules = {"l1": self.LOAD, "c1": self.DTM}
        return ctrl

    def test_flag_travels_in_fp(self) -> None:
        ctrl = self._ctrl()
        base = {"file_path": "/d/a.laz", "merge_segments": True}
        indep = {"file_path": "/d/a.laz", "merge_segments": False}
        plain = ctrl._node_fingerprints(["l1"], [], {"l1": base}, {})
        split = ctrl._node_fingerprints(["l1"], [], {"l1": indep}, {})
        self.assertNotEqual(plain["l1"], split["l1"])

    def test_downstream_inherits_flag_change(self) -> None:
        ctrl = self._ctrl()
        iids = ["l1", "c1"]
        edges = [("l1", "c1", "point_cloud", "point_cloud")]
        plain = ctrl._node_fingerprints(
            iids, edges,
            {"l1": {"file_path": "/d/a.laz", "merge_segments": True}}, {},
        )
        split = ctrl._node_fingerprints(
            iids, edges,
            {"l1": {"file_path": "/d/a.laz", "merge_segments": False}}, {},
        )
        self.assertNotEqual(plain["c1"], split["c1"])

    def test_resolve_defaults_to_merge(self) -> None:
        ctrl = self._ctrl()
        self.assertTrue(ctrl._resolve_merge_segments({}, ["l1"]))
        self.assertTrue(ctrl._resolve_merge_segments(None, []))
        self.assertFalse(ctrl._resolve_merge_segments(
            {"l1": {"merge_segments": False}}, ["l1"]))
        self.assertTrue(ctrl._resolve_merge_segments(
            {"l1": {"merge_segments": True}}, ["l1"]))


class FailureRecordTests(unittest.TestCase):
    """Failure reasons reach disk (provenance/session_meta/results/state)."""

    def _ctrl(self, tmp: Path) -> PipelineController:
        from types import SimpleNamespace

        from lynceus.processing.controller import PipelineCallbacks

        ctrl = PipelineController()
        ctrl._modules = {"b": GRID, "c": DTM}
        ctrl._node_failed = set()
        ctrl._node_failures = {}
        ctrl._node_task_done = {}
        ctrl._node_task_total = {}
        ctrl._reused_barriers = set()
        ctrl._stage_done = {}
        ctrl._stage_total = {}
        ctrl._session_dir = tmp
        cb = PipelineCallbacks()
        prog = ctrl._make_dag_progress(cb)
        prog(0, 2, SimpleNamespace(iid="b", kind="barrier",
                                   task_id="b|merge"),
             False, RuntimeError("boom\nsecond line"))
        prog(1, 2, SimpleNamespace(iid="c", kind="barrier",
                                   task_id="c|merge"),
             False, None)
        return ctrl

    def _tmp(self) -> Path:
        import shutil

        tmp = Path(tempfile.mkdtemp(prefix="failrec_"))
        self.addCleanup(shutil.rmtree, tmp, True)
        return tmp

    def test_progress_records_first_line_and_cascade(self) -> None:
        ctrl = self._ctrl(self._tmp())
        self.assertEqual(ctrl._node_failures["b"],
                         "boom | caused by: second line")
        self.assertEqual(
            ctrl._node_failures["c"], "blocked by failed dependency")

    def test_meta_and_state_carry_reasons(self) -> None:
        tmp = self._tmp()
        ctrl = self._ctrl(tmp)
        ctrl._session_state = {}
        ctrl._update_session_meta("failed")
        meta = json.loads((tmp / "session_meta.json").read_text(
            encoding="utf-8"))
        self.assertEqual(meta["failure_reasons"]["b"],
                         "boom | caused by: second line")
        ctrl._session_state["failed_nodes"] = sorted(ctrl._node_failed)
        ctrl._session_state["failure_reasons"] = dict(ctrl._node_failures)
        ctrl._save_session_state()
        state = json.loads((tmp / "session_state.json").read_text(
            encoding="utf-8"))
        self.assertEqual(state["failure_reasons"]["c"],
                         "blocked by failed dependency")

    def test_results_appends_per_iid_errors(self) -> None:
        tmp = self._tmp()
        ctrl = self._ctrl(tmp)
        tiles = [{"tile_id": "t0", "node": GRID, "output": "x.gpkg"}]
        ctrl._write_results(tiles)
        ledger = json.loads((tmp / "results.json").read_text(
            encoding="utf-8"))
        self.assertEqual(len(ledger), 3)
        errors = {e["iid"]: e for e in ledger if "error" in e}
        self.assertEqual(errors["b"]["error"],
                         "boom | caused by: second line")
        self.assertEqual(errors["b"]["node"], GRID)
        self.assertTrue(all("tile_id" not in e for e in ledger
                            if "error" in e))

    def test_provenance_attaches_error(self) -> None:
        tmp = self._tmp()
        ctrl = self._ctrl(tmp)
        sess = tmp / "sess"
        (sess / "artifacts").mkdir(parents=True)
        nodes = ctrl._save_provenance(
            sess, {"": "fp"}, tmp, {"b": "fp-b", "c": "fp-c"},
            ["b", "c"], [], {}, {"b": {"strategy": "grid"},
                                 "c": {"strategy": "dtm"}})
        self.assertEqual(nodes["b"]["error"],
                         "boom | caused by: second line")
        self.assertEqual(nodes["c"]["error"], "blocked by failed dependency")
        on_disk = json.loads((sess / "provenance.json").read_text(
            encoding="utf-8"))
        self.assertEqual(on_disk["nodes"]["b"]["error"],
                         "boom | caused by: second line")

    def test_run_abort_persists_without_batch_epilogue(self) -> None:
        tmp = self._tmp()
        ctrl = self._ctrl(tmp)
        ctrl._node_task_total = {"b": 1, "c": 1}
        ctrl._node_failed = set()
        ctrl._node_failures = {}
        reason = ctrl._record_run_abort(RuntimeError("kaboom"))
        self.assertEqual(reason, "kaboom")
        self.assertEqual(
            ctrl._node_failures["b"], "run aborted: kaboom")
        state = json.loads((tmp / "session_state.json").read_text(
            encoding="utf-8"))
        self.assertEqual(state["pipeline_status"], "failed")
        self.assertEqual(state["failure_reasons"]["c"],
                         "run aborted: kaboom")
        meta = json.loads((tmp / "session_meta.json").read_text(
            encoding="utf-8"))
        self.assertEqual(meta["pipeline_status"], "failed")
        self.assertIn("b", meta["failure_reasons"])
        ledger = json.loads((tmp / "results.json").read_text(
            encoding="utf-8"))
        self.assertTrue(all("error" in e for e in ledger))
        self.assertEqual(
            {e["iid"] for e in ledger}, {"b", "c"})

    def test_run_abort_never_breaks_twice(self) -> None:
        ctrl = PipelineController()
        ctrl._node_task_total = {}
        # No session dir, no state: must still return the reason.
        self.assertEqual(ctrl._record_run_abort(ValueError("")),
                         "unknown error")

    def test_chained_reason_keeps_root_cause(self) -> None:
        from lynceus.processing.controller import _failure_reason

        self.assertEqual(_failure_reason(ValueError("boom")), "boom")
        self.assertEqual(_failure_reason(ValueError("")), "unknown error")
        chained = _failure_reason(RuntimeError(
            "Write failed. See previous exception for details.\n"
            "Some context\n"
            "No space left on device (os error 28)"))
        self.assertIn("Write failed", chained)
        self.assertIn("No space left on device", chained)
        single = _failure_reason(RuntimeError("x\ny\nx"))
        self.assertEqual(single, "x")
        long = _failure_reason(ValueError("z" * 500))
        self.assertLessEqual(len(long), 401)


if __name__ == "__main__":
    unittest.main()
