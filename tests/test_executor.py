"""Unit tests for the executor's predictive dispatch throttle (P1b).

Pure helpers only: no process pool, no Qt. Running real pools on Windows
spawn is exercised by the app and the ad-hoc e2e script."""

from __future__ import annotations

import unittest

from lynceus.processing.executor import (
    _BYTES_PER_POINT_EST,
    _dispatch_allowed,
    _project_inflight_bytes,
    _task_est_bytes,
    workers_for_percent,
)
from lynceus.processing.steps import Task
from lynceus.processing.tiler import (
    MEMORY_BUDGET_FLOOR_BYTES,
    memory_budget_for_percent,
)


def _tile(points: int) -> Task:
    return Task(
        task_id="t",
        iid="n",
        module_id="m",
        kind="tile",
        fn=lambda: None,
        args=(),
        deps=set(),
        tile_id="x",
        weight=points,
    )


class PredictiveThrottleTests(unittest.TestCase):
    def test_tile_estimate_uses_point_count_bytes(self) -> None:
        self.assertGreater(_BYTES_PER_POINT_EST, 0)
        self.assertEqual(_task_est_bytes(_tile(1_000_000)), 1_000_000 * _BYTES_PER_POINT_EST)

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
        self.assertEqual(workers_for_percent(100, 8), 8)
        self.assertEqual(workers_for_percent(50, 8), 4)
        self.assertEqual(workers_for_percent(10, 8), 1)
        self.assertEqual(workers_for_percent(0, 8), 1)
        self.assertEqual(workers_for_percent(100, 1), 1)
        self.assertEqual(workers_for_percent(250, 4), 4)
        self.assertEqual(workers_for_percent("junk", 4), 4)
        self.assertGreaterEqual(workers_for_percent(50, None), 1)

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


if __name__ == "__main__":
    unittest.main()