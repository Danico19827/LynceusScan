"""Unit tests for the executor's predictive dispatch throttle (P1b).

Pure helpers plus dispatch-order tests over a synchronous fake pool (no
real processes, no Qt). Running real pools on Windows spawn is exercised
by the app and the ad-hoc e2e script."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from lynceus.processing.executor import (
    _BYTES_PER_POINT_EST,
    _DEFAULT_BARRIER_FACTOR,
    _dispatch_allowed,
    _predictive_budget,
    _project_inflight_bytes,
    _task_est_bytes,
    run_dag,
    workers_for_percent,
)
from lynceus.processing.steps import Task
from lynceus.processing.tiler import (
    MEMORY_BUDGET_FLOOR_BYTES,
    memory_budget_for_percent,
)


def _noop() -> None:
    return None


def _tile(points: int, tile_f: float = 0.0) -> Task:
    return Task(
        task_id="t",
        iid="n",
        module_id="m",
        kind="tile",
        fn=_noop,
        args=(),
        deps=set(),
        tile_id="x",
        weight=points,
        tile_f=tile_f,
    )


class PredictiveThrottleTests(unittest.TestCase):
    def test_tile_estimate_uses_point_count_bytes(self) -> None:
        self.assertGreater(_BYTES_PER_POINT_EST, 0)
        self.assertEqual(_task_est_bytes(_tile(1_000_000)), 1_000_000 * _BYTES_PER_POINT_EST)

    def test_tile_weight_factor_scales_estimate_only(self) -> None:
        # A declared factor multiplies the admission estimate...
        base = _task_est_bytes(_tile(1_000_000))
        self.assertEqual(_task_est_bytes(_tile(1_000_000, tile_f=3.0)), 3 * base)
        self.assertEqual(_task_est_bytes(_tile(1_000_000, tile_f=1.5)), int(1.5 * base))
        # ...but the queue weight (priority) stays the raw point count.
        self.assertEqual(_tile(10, tile_f=3.0).weight, 10)
        self.assertEqual(_tile(10, tile_f=3.0).weight, _tile(10).weight)

    def test_tile_weight_factor_junk_defaults_to_one(self) -> None:
        for bad in (-1.0, 0.0, "junk"):
            base = _tile(500)
            base.tile_f = bad
            self.assertEqual(_task_est_bytes(base), 500 * _BYTES_PER_POINT_EST)

    def test_tile_weight_factor_survives_pickle(self) -> None:
        import pickle

        heavy = pickle.loads(pickle.dumps(_tile(10, tile_f=3.0)))
        self.assertEqual(_task_est_bytes(heavy), 30 * _BYTES_PER_POINT_EST)

    def test_barrier_and_unknown_weight_budget_zero(self) -> None:
        barrier = Task(
            task_id="b",
            iid="n",
            module_id="m",
            kind="barrier",
            fn=lambda: None,
            args=(),
        )
        self.assertEqual(_task_est_bytes(barrier), 0)
        self.assertEqual(_task_est_bytes(_tile(0)), 0)

    def test_projection_sums_tile_estimates(self) -> None:
        tasks = [_tile(100), _tile(200), _tile(0)]
        self.assertEqual(_project_inflight_bytes(tasks), 300 * _BYTES_PER_POINT_EST)

    def test_dispatch_allowed_within_budget(self) -> None:
        est = 100 * _BYTES_PER_POINT_EST
        self.assertTrue(_dispatch_allowed(est, 0, est))
        self.assertTrue(_dispatch_allowed(est, est, 2 * est))
        self.assertFalse(_dispatch_allowed(est, est, est + 1))

    def test_oversized_tile_runs_alone_above_budget(self) -> None:
        est = 20 * _BYTES_PER_POINT_EST
        budget = 10 * _BYTES_PER_POINT_EST
        self.assertTrue(_dispatch_allowed(est, 0, budget))
        # Never held back by something already running once past budget.
        self.assertFalse(_dispatch_allowed(est, 1, budget))

    def test_zero_estimate_never_blocks(self) -> None:
        self.assertTrue(_dispatch_allowed(0, 9 ** 18, 1024))


class PerfPreferenceMathTests(unittest.TestCase):
    def test_workers_for_percent(self) -> None:
        # Product policy: 100% leaves one core for the system.
        self.assertEqual(workers_for_percent(100, 8), 7)
        self.assertEqual(workers_for_percent(50, 8), 4)
        self.assertEqual(workers_for_percent(10, 8), 1)
        self.assertEqual(workers_for_percent(0, 8), 1)
        self.assertEqual(workers_for_percent(100, 1), 1)
        self.assertEqual(workers_for_percent(100, 2), 1)
        self.assertEqual(workers_for_percent(250, 4), 3)
        self.assertEqual(workers_for_percent("junk", 4), 3)
        self.assertGreaterEqual(workers_for_percent(50, None), 1)

    def test_default_workers_reserves_one_core(self) -> None:
        from lynceus.processing.executor import default_workers

        self.assertEqual(default_workers(12), 11)
        self.assertEqual(default_workers(2), 1)
        self.assertEqual(default_workers(1), 1)
        self.assertEqual(default_workers(0), 1)
        self.assertEqual(default_workers("junk"), 3)
        self.assertGreaterEqual(default_workers(None), 1)

    def test_memory_budget_for_percent(self) -> None:
        ram = 16 * 1024**3
        self.assertEqual(memory_budget_for_percent(25, ram), 4 * 1024**3)
        self.assertEqual(memory_budget_for_percent(100, ram), ram)
        # Floor wins over tiny percentages and unknown RAM.
        self.assertEqual(memory_budget_for_percent(1, ram), MEMORY_BUDGET_FLOOR_BYTES)
        self.assertEqual(memory_budget_for_percent(50, 0), MEMORY_BUDGET_FLOOR_BYTES)
        self.assertEqual(memory_budget_for_percent("junk", ram), 4 * 1024**3)


class ProgressReasonTests(unittest.TestCase):
    """Task failure reasons reach the UI (not just the worker log)."""

    def _run_progress(self, ok: bool, exc):
        from types import SimpleNamespace

        from lynceus.processing.controller import PipelineController

        ctrl = PipelineController()
        ctrl._modules = {"g": "lynceus.nodes.lidar.analysis.grid_metrics"}
        ctrl._node_failed = set()
        ctrl._node_failures = {}
        messages = []
        statuses = []
        cb = SimpleNamespace(
            on_node_status=lambda iid, st: statuses.append((iid, st)),
            on_message=lambda text, kind="info": messages.append((text, kind)),
            on_progress=lambda done, total: None,
        )
        task = SimpleNamespace(iid="g", kind="barrier", task_id="g|merge")
        ctrl._make_dag_progress(cb)(1, 1, task, ok, exc)
        return messages, statuses

    def test_failure_reports_reason(self) -> None:
        messages, statuses = self._run_progress(False, RuntimeError("boom"))
        self.assertIn(("g", "error"), statuses)
        self.assertTrue(
            any(kind == "error" and "boom" in text for text, kind in messages),
            messages,
        )

    def test_non_runtime_errors_report_the_same_way(self) -> None:
        # Any worker exception (not just RuntimeError) marks the node
        # failed and surfaces its message: no silent zombies by type.
        for exc in (MemoryError("out of ram"), KeyError("nope")):
            messages, statuses = self._run_progress(False, exc)
            self.assertIn(("g", "error"), statuses)
            self.assertTrue(
                any(kind == "error" and str(exc) in text for text, kind in messages),
                messages,
            )

    def test_cascade_without_exception_stays_quiet(self) -> None:
        messages, statuses = self._run_progress(False, None)
        self.assertIn(("g", "error"), statuses)
        self.assertEqual(
            [kind for _, kind in messages if kind == "error"], []
        )


def _sized_file(directory: Path, name: str, size: int) -> str:
    path = directory / name
    path.write_bytes(b"x" * size)
    return str(path)


def _barrier_task(
    task_id: str,
    ctx: dict,
    mem_factor: float = 0.0,
    fn=None,
) -> Task:
    return Task(
        task_id=task_id,
        iid=task_id.split("|")[0],
        module_id="m",
        kind="barrier",
        fn=fn or (lambda _ctx: {"ok": True}),
        args=(ctx,),
        mem_factor=mem_factor,
    )


class BarrierEstimateTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.root = Path(self._dir.name)

    def test_barrier_scales_with_input_bytes_and_factor(self) -> None:
        path = _sized_file(self.root, "in.tif", 600)
        task = _barrier_task("n|merge", {"in0_path": path}, mem_factor=10.0)
        self.assertEqual(_task_est_bytes(task), 6000)

    def test_barrier_without_factor_uses_default(self) -> None:
        self.assertGreater(_DEFAULT_BARRIER_FACTOR, 0)
        path = _sized_file(self.root, "in.tif", 100)
        task = _barrier_task("n|merge", {"in0_path": path})
        self.assertEqual(_task_est_bytes(task), 100 * _DEFAULT_BARRIER_FACTOR)

    def test_barrier_skips_missing_and_non_files(self) -> None:
        task = _barrier_task(
            "n|merge",
            {
                "in0_path": str(self.root / "gone.tif"),
                "in1_path": str(self.root),
                "session_dir": str(self.root),
                "cell_size": 1.0,
            },
        )
        self.assertEqual(_task_est_bytes(task), 0)

    def test_barrier_unwraps_extension_envelope(self) -> None:
        path = _sized_file(self.root, "in.tif", 50)
        ctx = {"in0_path": path}
        task = _barrier_task("n|merge", ctx, mem_factor=2.0)
        task.args = ("mod", "fn_name", (ctx,))
        self.assertEqual(_task_est_bytes(task), 100)

    def test_barrier_unwraps_wrapper_envelope(self) -> None:
        # Regression (Fairbanks 20261014): the F4 barrier wrapper changed
        # core args to (fn, ctx) and the positional extractor went blind
        # (est 0 MB), so the gate dispatched every merge at once.
        path = _sized_file(self.root, "in.tif", 50)
        ctx = {"in0_path": path}
        task = _barrier_task("n|merge", ctx, mem_factor=2.0)
        task.args = (lambda c: {"ok": True}, ctx)
        self.assertEqual(_task_est_bytes(task), 100)

    def test_junk_factor_falls_back_to_default(self) -> None:
        path = _sized_file(self.root, "in.tif", 40)
        task = _barrier_task("n|merge", {"in0_path": path})
        task.mem_factor = "junk"
        self.assertEqual(_task_est_bytes(task), 40 * _DEFAULT_BARRIER_FACTOR)


class ModeAwareBarrierTests(unittest.TestCase):
    """Dispatch budgets the peak of the mode that WILL run (F4)."""

    def setUp(self) -> None:
        import shutil

        self.tmp = Path(tempfile.mkdtemp(prefix="modeest_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        from affine import Affine

        from lynceus.processing.raster import write_geotiff

        for i, value in enumerate((10.0, 20.0)):
            arr = np.full((30, 30), value, dtype="float32")
            write_geotiff(self.tmp / f"t{i}.tif", arr,
                          Affine(1.0, 0.0, i * 30.0, 0.0, -1.0, 30.0),
                          crs="EPSG:32621")
        self.a = str(self.tmp / "t0.tif")
        self.b = str(self.tmp / "t1.tif")

    def _merge_task(self, ctx, copies=3.0, cell=0):
        task = _barrier_task("n|merge", ctx)
        task.merge_copies = copies
        task.merge_cell_bytes = cell
        return task

    def test_fitting_union_budgets_direct_need(self) -> None:
        ctx = {"session_dir": str(self.tmp),
               "in0_path": self.a, "in1_path": self.b}
        task = self._merge_task(ctx)
        # Union 30x60 cells x 4 B x 3 copies, exact from headers.
        self.assertEqual(_task_est_bytes(task, 10 ** 12), 30 * 60 * 4 * 3)

    def test_oversized_union_budgets_window_peak(self) -> None:
        from affine import Affine

        from lynceus.processing.raster import window_peak, write_geotiff

        # Big enough that the window floor (256 px) stays below the need:
        # union 500x1000 cells x 4 B x 3 copies = 6 MB need.
        big = []
        for i, value in enumerate((10.0, 20.0)):
            arr = np.full((500, 500), value, dtype="float32")
            path = str(self.tmp / f"big{i}.tif")
            write_geotiff(path, arr,
                          Affine(1.0, 0.0, i * 500.0, 0.0, -1.0, 500.0),
                          crs="EPSG:32621")
            big.append(path)
        ctx = {"session_dir": str(self.tmp),
               "in0_path": big[0], "in1_path": big[1]}
        task = self._merge_task(ctx)
        est = _task_est_bytes(task, 2_000_000)
        self.assertEqual(est, window_peak(2_000_000, 4))
        self.assertLess(est, 500 * 1000 * 4 * 3)

    def test_heterogeneous_falls_back_to_legacy(self) -> None:
        import rasterio

        from lynceus.processing.executor import _DEFAULT_BARRIER_FACTOR

        other = str(self.tmp / "o.tif")
        import shutil

        shutil.copy(self.a, other)
        with rasterio.open(other, "r+") as dst:
            dst.crs = rasterio.crs.CRS.from_epsg(32620)
        ctx = {"session_dir": str(self.tmp),
               "in0_path": self.a, "in1_path": other}
        task = self._merge_task(ctx)
        expect = (Path(self.a).stat().st_size
                  + Path(other).stat().st_size) * _DEFAULT_BARRIER_FACTOR
        self.assertEqual(_task_est_bytes(task, 10 ** 12), expect)

    def test_no_budget_or_copies_keep_legacy(self) -> None:
        ctx = {"session_dir": str(self.tmp),
               "in0_path": self.a, "in1_path": self.b}
        legacy = _task_est_bytes(self._merge_task(ctx), None)
        self.assertEqual(
            _task_est_bytes(self._merge_task(ctx, copies=0.0), 10 ** 12),
            legacy)
        task = self._merge_task(ctx, copies="junk")
        self.assertEqual(_task_est_bytes(task, 10 ** 12), legacy)

    def test_dem_cells_use_float64_model(self) -> None:
        ctx = {"session_dir": str(self.tmp), "dtm_mosaic_path": self.a}
        task = self._merge_task(ctx, copies=5.0, cell=8)
        # Single 30x30 input: 900 x 8 B x 5 copies, exact from headers.
        self.assertEqual(_task_est_bytes(task, 10 ** 12), 900 * 8 * 5)

    def test_caps_carry_merge_model(self) -> None:
        from lynceus.plugins.registry import manager
        from lynceus.processing.steps import (
            discover_node_capabilities,
            effective_node_caps,
        )

        manager.discover()
        dtm = discover_node_capabilities(
            "lynceus.nodes.lidar.terrain.generate_dtm")
        self.assertEqual(dtm.get("merge_copies"), 20.0)
        slope = effective_node_caps(
            "lynceus.nodes.raster.ops.terrain_derivatives",
            {"strategy": "slope"})
        self.assertEqual(slope.get("merge_copies"), 5.0)
        self.assertEqual(slope.get("merge_cell_bytes"), 8)
        grid = discover_node_capabilities(
            "lynceus.nodes.lidar.analysis.grid_metrics")
        self.assertEqual(grid.get("merge_copies"), 0.0)


class TileGridEstimateTests(unittest.TestCase):
    """Area-driven tile estimates (sparse supertiles hold big grids)."""

    def _grid_tile(self, width_m: float, height_m: float, points: int,
                   bpc: float = 96.0, res_m: float = 1.0,
                   tile_f: float = 1.5) -> Task:
        tile = {
            "tile_id": "t",
            "core_x_min": 0.0,
            "core_x_max": width_m,
            "core_y_min": 0.0,
            "core_y_max": height_m,
            "point_count": points,
        }
        return Task(
            task_id="t",
            iid="n",
            module_id="m",
            kind="tile",
            fn=_noop,
            args=(tile, {"session_dir": "s"}),
            weight=points,
            tile_f=tile_f,
            grid_bpc=bpc,
            grid_res_m=res_m,
        )

    def test_sparse_supertile_counts_area_not_points(self) -> None:
        # Fairbanks 20261014: a 2667 m supertile with 116k points holds a
        # 683 MB PMF grid admitted against an 11 MB point estimate.
        task = self._grid_tile(2667.0, 2667.0, 116_000)
        est = _task_est_bytes(task)
        point_part = int(116_000 * _BYTES_PER_POINT_EST * 1.5)
        self.assertGreaterEqual(est, 2667 * 2667 * 96)
        self.assertGreater(est, 60 * point_part)

    def test_small_dense_tile_matches_point_model(self) -> None:
        task = self._grid_tile(100.0, 100.0, 500_000)
        point_part = int(500_000 * _BYTES_PER_POINT_EST * 1.5)
        self.assertEqual(
            _task_est_bytes(task), point_part + 100 * 100 * 96)

    def test_no_grid_terms_keep_legacy_estimate(self) -> None:
        base = int(1_000_000 * _BYTES_PER_POINT_EST * 1.5)
        task = self._grid_tile(2667.0, 2667.0, 1_000_000, bpc=0.0)
        self.assertEqual(_task_est_bytes(task), base)
        task = self._grid_tile(2667.0, 2667.0, 1_000_000, res_m=0.0)
        self.assertEqual(_task_est_bytes(task), base)

    def test_missing_bounds_keep_legacy_estimate(self) -> None:
        base = _task_est_bytes(_tile(1_000_000))
        task = Task(
            task_id="t", iid="n", module_id="m", kind="tile", fn=_noop,
            args=({"tile_id": "t"}, {"session_dir": "s"}),
            weight=1_000_000, grid_bpc=96.0, grid_res_m=1.0,
        )
        self.assertEqual(_task_est_bytes(task), base)

    def test_ctx_wins_over_tile_dict(self) -> None:
        from lynceus.processing.executor import _task_ctx, _task_tile

        tile = {"tile_id": "t", "core_x_min": 0.0}
        ctx = {"session_dir": "s", "cell_size": 1.0}
        task = Task(
            task_id="t", iid="n", module_id="m", kind="tile", fn=_noop,
            args=(tile, ctx), weight=10,
        )
        self.assertEqual(_task_ctx(task), ctx)
        self.assertEqual(_task_tile(task), tile)

    def test_resolution_keys_resolve_first_positive(self) -> None:
        from lynceus.processing.steps import _grid_resolution

        specs = {"grid_resolution_keys": ("pmf_cell_size", "cell_size")}
        self.assertEqual(
            _grid_resolution({"pmf_cell_size": 2.0, "cell_size": 1.0},
                             specs), 2.0)
        self.assertEqual(
            _grid_resolution({"cell_size": 1.0}, specs), 1.0)
        self.assertEqual(_grid_resolution({}, specs), 0.0)
        self.assertEqual(_grid_resolution({"resolution": 1.0}, {}), 1.0)


class PredictiveBudgetTests(unittest.TestCase):
    def test_unset_cap_keeps_all_ram_behavior(self) -> None:
        self.assertEqual(_predictive_budget(1000, None), 700.0)
        self.assertEqual(_predictive_budget(1000, 0), 700.0)

    def test_cap_only_tightens(self) -> None:
        self.assertEqual(_predictive_budget(1000, 500), 350.0)
        self.assertEqual(_predictive_budget(1000, 10_000), 700.0)

    def test_unknown_ram_is_unbounded(self) -> None:
        self.assertEqual(_predictive_budget(0, None), float("inf"))


class _SyncFuture:
    def __init__(self) -> None:
        self._result = None

    def get(self):
        return self._result


class _SyncPool:
    """Inline pool: runs each task at submit (no processes)."""

    _processes = 8

    def apply_async(self, fn, args=(), kwds=None, callback=None,
                    error_callback=None):
        future = _SyncFuture()
        try:
            future._result = fn(*(args or ()), **(kwds or {}))
        except Exception as exc:
            if error_callback is not None:
                error_callback(exc)
        else:
            if callback is not None:
                callback(future._result)
        return future


class BarrierDispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.root = Path(self._dir.name)
        self.live = 0
        self.peak = 0

    def _tracked(self, tasks: list[Task], **kwargs):
        def on_start(_task) -> None:
            self.live += 1
            self.peak = max(self.peak, self.live)

        def on_progress(_done, _total, _task, _ok, _exc) -> None:
            self.live -= 1

        return run_dag(
            _SyncPool(),
            tasks,
            progress_callback=on_progress,
            on_task_start=on_start,
            **kwargs,
        )

    def _three_barriers(self, size: int) -> list[Task]:
        tasks = []
        for i in range(3):
            path = _sized_file(self.root, f"in{i}.tif", size)
            tasks.append(_barrier_task(f"n{i}|merge", {"in0_path": path},
                                       mem_factor=1.0))
        return tasks

    def test_heavy_barriers_serialize(self) -> None:
        outputs, _tiles, failed, _stats = self._tracked(
            self._three_barriers(600), memory_budget_bytes=1000)
        self.assertEqual(failed, 0)
        self.assertEqual(self.peak, 1)
        self.assertEqual(len(outputs), 3)

    def test_light_barriers_stay_parallel(self) -> None:
        outputs, _tiles, failed, _stats = self._tracked(
            self._three_barriers(100), memory_budget_bytes=1000)
        self.assertEqual(failed, 0)
        self.assertEqual(self.peak, 3)
        self.assertEqual(len(outputs), 3)

    def test_oversized_barrier_runs_alone(self) -> None:
        outputs, _tiles, failed, _stats = self._tracked(
            self._three_barriers(900), memory_budget_bytes=1000)
        self.assertEqual(failed, 0)
        self.assertEqual(self.peak, 1)
        self.assertEqual(len(outputs), 3)


class OomHintTests(unittest.TestCase):
    def test_memory_error_carries_estimate_and_budget(self) -> None:
        from types import SimpleNamespace

        from lynceus.processing.controller import PipelineController

        ctrl = PipelineController()
        ctrl._modules = {"g": "lynceus.nodes.lidar.analysis.grid_metrics"}
        ctrl._node_failed = set()
        ctrl._node_failures = {}
        messages = []
        cb = SimpleNamespace(
            on_node_status=lambda iid, st: None,
            on_message=lambda text, kind="info": messages.append((text, kind)),
            on_progress=lambda done, total: None,
        )
        task = SimpleNamespace(iid="g", kind="barrier", task_id="g|merge")
        ctrl._make_dag_progress(cb)(1, 1, task, False, MemoryError("out of ram"))
        errors = [text for text, kind in messages if kind == "error"]
        self.assertEqual(len(errors), 1)
        self.assertIn("out of ram", errors[0])
        self.assertIn("dispatch budget", errors[0])


class BarrierDoneTests(unittest.TestCase):
    """run_dag publishes each barrier payload at completion (live gallery)."""

    def test_barrier_payloads_publish_in_completion_order(self) -> None:
        seen: list = []
        first = _barrier_task(
            "n0|merge", {}, fn=lambda _c: {"file": "a.tif"})
        second = _barrier_task(
            "n1|merge", {}, fn=lambda _c: {"file": "b.tif"})
        second.deps = {"n0|merge"}
        outputs, _tiles, failed, _stats = run_dag(
            _SyncPool(), [first, second],
            on_barrier_done=lambda task, result: seen.append(
                (task.iid, result)),
        )
        self.assertEqual(failed, 0)
        self.assertEqual([iid for iid, _ in seen], ["n0", "n1"])
        self.assertEqual(seen[0][1], {"file": "a.tif"})
        self.assertEqual(outputs["n1"], {"file": "b.tif"})

    def test_tiles_failures_and_non_dicts_never_publish(self) -> None:
        seen: list = []

        def _boom(_ctx):
            raise RuntimeError("nope")

        tile = _tile(10)
        tile.fn = lambda: {"node": "n", "tile_id": "x"}
        tasks = [
            tile,
            _barrier_task("nb|merge", {}, fn=_boom),
            _barrier_task("nd|merge", {}, fn=lambda _c: ["not", "a dict"]),
            _barrier_task("ok|merge", {}, fn=lambda _c: {"file": "o.tif"}),
        ]
        _outputs, _tiles, failed, _stats = run_dag(
            _SyncPool(), tasks,
            on_barrier_done=lambda task, result: seen.append(
                (task.iid, result)),
        )
        self.assertEqual(failed, 1)
        self.assertEqual(seen, [("ok", {"file": "o.tif"})])

    def test_exceptional_listener_never_fails_tasks(self) -> None:
        def _bad(_task, _result):
            raise RuntimeError("listener blew up")

        tasks = [_barrier_task("n|merge", {}, fn=lambda _c: {"ok": True})]
        outputs, _tiles, failed, _stats = run_dag(
            _SyncPool(), tasks, on_barrier_done=_bad)
        self.assertEqual(failed, 0)
        self.assertEqual(outputs, {"n": {"ok": True}})


if __name__ == "__main__":
    unittest.main()