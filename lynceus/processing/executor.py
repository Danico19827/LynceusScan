# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Execute the processing pipeline DAG over point-cloud tiles.

The compiled canvas graph becomes an atomic task DAG. Tasks are dispatched as
soon as dependencies complete:
- tile tasks stream each tile into the next stage
- barrier tasks wait for all required tiles (barrier-only graphs run
  without any tiling stage at all)

Dispatch is lightest-first with RAM admission control: a ready task runs
only if the projected in-flight working set fits the predictive budget
(70% of system RAM); otherwise dispatch pauses until completions free
memory (the reactive 75% RSS throttle backs it up). So the pool may idle
on purpose under memory pressure instead of paging.
"""

from __future__ import annotations

import logging
import os
import queue
import time
from typing import Callable

import psutil

from lynceus.processing.steps import Task

logger = logging.getLogger(__name__)

# Metrics for the latest run: task timings and pool occupancy.
# Populated by run_dag() for diagnostics and optimization work.
last_run_stats: dict = {}

# Memory sampling interval (completed tasks).
_MEMORY_SAMPLE_EVERY = 4

# Reduce active workers when GUI plus pool memory exceeds this system ratio.
_MEMORY_RATIO_HARD_LIMIT = 0.75
# Soft knee where the linear scale-down begins.
_MEMORY_RATIO_SOFT_START = 0.60
# Minimum workers retained to avoid deadlocks.
_MIN_ACTIVE_WORKERS = 2

# Predictive dispatch: projected in-flight tile RAM cap as a system ratio.
# Stays below the reactive hard limit so both throttles form a staircase:
# dispatch stops as soon as the *estimated* working set of running tiles
# reaches the budget, instead of waiting for RSS to actually blow up.
# Set close to the hard limit (not mid-scale) so ordinary cloud sizes keep
# full concurrency; only near-paging workloads feel the proactive backpressure.
_PREDICTIVE_RAM_RATIO = 0.70
# Rough ceiling for a tile task's peak working set per buffered point
# (record array + numpy intermediates for stats and routing).
_BYTES_PER_POINT_EST = 64


def workers_for_percent(percent: int | float, total: int | None = None) -> int:
    """Pool size for a CPU percentage (Preferences > Performance).

    Pure helper so the UI math stays tested: clamps to [1, total].
    """
    try:
        cores = int(total if total is not None else (os.cpu_count() or 4))
    except (TypeError, ValueError):
        cores = 4
    cores = max(1, cores)
    try:
        pct = float(percent)
    except (TypeError, ValueError):
        pct = 100.0
    return max(1, min(cores, int(round(cores * pct / 100.0))))


def create_pool(max_workers: int | None = None):
    """Create the process pool; this must run on the main thread."""
    from multiprocessing import get_context

    workers = max_workers or (os.cpu_count() or 4)
    return get_context("spawn").Pool(processes=workers)


def _sample_memory_mb() -> float:
    """Resident Set Size of the current process in MB."""
    try:
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def _sample_pool_memory_mb(pool) -> float:
    """Sum of the RSS of the pool workers in MB (via child processes)."""
    total = 0.0
    try:
        parent = psutil.Process()
        for child in parent.children(recursive=True):
            try:
                total += child.memory_info().rss / (1024 * 1024)
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                pass
    except Exception:
        pass
    return total


def _task_est_bytes(task: Task) -> int:
    """Estimated peak working set of a task in bytes.

    Tile tasks scale with their point count (weight, also used for queue
    priority). Barriers budget zero here even though point-cloud exports
    and vector/table consolidations can peak: their inputs are already
    materialized, so the predictive gate only throttles tile dispatch.
    """
    if task.kind == "tile" and task.weight > 0:
        return task.weight * _BYTES_PER_POINT_EST
    return 0


def _project_inflight_bytes(tasks) -> int:
    """Sum of the estimated working set of the given in-flight tasks."""
    return sum(_task_est_bytes(t) for t in tasks)


def _dispatch_allowed(est_bytes: int, inflight_bytes: int, budget: int) -> bool:
    """Predictive dispatch gate.

    A task with no estimate always fits. An oversized tile whose estimate
    alone exceeds the budget runs anyway (nothing better to do with its
    working set); otherwise the projected in-flight sum must stay in budget.
    """
    if not est_bytes:
        return True
    return inflight_bytes == 0 or inflight_bytes + est_bytes <= budget


def run_dag(
    pool,
    tasks: list[Task],
    progress_callback: Callable[[int, int, Task, bool, Exception | None], None] | None = None,
    on_task_start: Callable[[Task], None] | None = None,
    cancel_flag: Callable[[], bool] | None = None,
    pause_flag: Callable[[], bool] | None = None,
) -> tuple[dict, list[dict], int, dict]:
    """Execute the DAG while keeping the worker pool busy.

    Returns ``(outputs, tile_results, failed, stats)`` for successful barriers,
    tile-task results, failed tasks including dependency cascades, and metrics.

    Callbacks run in the execution thread. ``on_task_start`` fires when a task
    is submitted; ``progress_callback`` fires when a task reaches a final state
    (with the exception on failure, None for dependency cascades).

    ``cancel_flag`` stops dispatch and marks pending tasks. ``pause_flag``
    stops dispatch while allowing in-flight tasks to complete.
    """
    if not tasks:
        return {}, [], 0, {}

    workers = getattr(pool, "_processes", None) or (os.cpu_count() or 4)
    tasks_by_id = {t.task_id: t for t in tasks}
    ready: list[Task] = [t for t in tasks if not t.deps]
    if not ready:
        raise RuntimeError("Pipeline task graph has a cycle or invalid dependencies")
    completed: set[str] = set()
    failed: set[str] = set()
    results: dict[str, object] = {}
    outputs: dict[str, dict] = {}
    tile_results: list[dict] = []

    in_flight: dict[object, Task] = {}
    futures_by_id: dict[str, object] = {}
    completions: "queue.Queue[tuple[str, Exception | None]]" = queue.Queue()

    t_start = time.monotonic()
    dispatch_ts: dict[str, float] = {}
    busy: dict[str, float] = {"tile": 0.0, "barrier": 0.0}
    task_secs: dict[str, float] = {}

    peak_memory_mb = _sample_memory_mb()

    poisoned: set[str] = set()

    def fail_cascade(task: Task) -> None:
        if task.task_id in completed:
            return
        completed.add(task.task_id)
        failed.add(task.task_id)
        if progress_callback is not None:
            progress_callback(len(completed), len(tasks), task, False, None)
        release(task.task_id, dep_failed=True)

    def release(task_id: str, dep_failed: bool) -> None:
        # Mutates Task.deps in place: only safe because build_dag rebuilds
        # the task list on every run (never reuse a list across runs).
        for other in tasks:
            if task_id not in other.deps:
                continue
            other.deps.discard(task_id)
            if dep_failed:
                poisoned.add(other.task_id)
            if not other.deps and other.task_id not in completed:
                if other.task_id in poisoned:
                    fail_cascade(other)
                else:
                    ready.append(other)

    def _effective_workers() -> int:
        """Return active workers after applying the memory throttle."""
        total_ram = psutil.virtual_memory().total
        if total_ram <= 0:
            return workers
        cur = _sample_memory_mb() + _sample_pool_memory_mb(pool)
        ratio = cur / total_ram
        if ratio >= _MEMORY_RATIO_HARD_LIMIT:
            logger.warning(
                f"[Executor] Memory throttle: {cur:.0f}MB / "
                f"{total_ram / (1024**2):.0f}MB = {ratio:.0%} >= {_MEMORY_RATIO_HARD_LIMIT:.0%} "
                f"limit -> min {_MIN_ACTIVE_WORKERS} workers"
            )
            return _MIN_ACTIVE_WORKERS
        # Scale linearly between the normal threshold and the hard limit.
        if ratio <= _MEMORY_RATIO_SOFT_START:
            return workers
        span = _MEMORY_RATIO_HARD_LIMIT - _MEMORY_RATIO_SOFT_START
        fraction = max(0.0, 1.0 - (ratio - _MEMORY_RATIO_SOFT_START) / span)
        adjusted = max(_MIN_ACTIVE_WORKERS, int(workers * fraction))
        return adjusted

    def pump() -> None:
        limit = _effective_workers()
        total_ram = psutil.virtual_memory().total
        pred_budget = total_ram * _PREDICTIVE_RAM_RATIO if total_ram > 0 else float("inf")
        # Projected in-flight RAM from tile tasks with a known point count.
        inflight_bytes = _project_inflight_bytes(in_flight.values())
        # Lightest-first admission control: the ready list is heaviest-first
        # but pop() takes the lightest end. Trying the lightest task first
        # keeps the pool full under the RAM budget (best-fit-ish); if even
        # it does not fit, nothing will, so dispatch pauses correctly.
        ready.sort(key=lambda t: t.weight, reverse=True)
        while len(in_flight) < limit and ready:
            task = ready.pop()
            est = _task_est_bytes(task)
            # Allow an oversized tile to run alone (nothing to do about its
            # working set); otherwise hold dispatch until RAM frees up.
            if not _dispatch_allowed(est, inflight_bytes, pred_budget):
                ready.append(task)
                break
            if on_task_start is not None:
                on_task_start(task)
            dispatch_ts[task.task_id] = time.monotonic()
            future = pool.apply_async(
                task.fn,
                task.args,
                callback=lambda _result, tid=task.task_id: completions.put(
                    (tid, None)
                ),
                error_callback=lambda e, tid=task.task_id: completions.put((tid, e)),
            )
            in_flight[future] = task
            futures_by_id[task.task_id] = future
            inflight_bytes += est

    def consume_completion(tid: str, exc: Exception | None) -> None:
        nonlocal peak_memory_mb
        task = tasks_by_id[tid]
        future = futures_by_id.pop(tid)
        del in_flight[future]
        completed.add(tid)

        started = dispatch_ts.pop(tid, None)
        if started is not None:
            secs = time.monotonic() - started
            task_secs[tid] = secs
            busy[task.kind] = busy.get(task.kind, 0.0) + secs

        if exc is not None:
            failed.add(tid)
            logger.error(f"[Executor] Task {tid} failed: {exc}")
        else:
            result = future.get()
            results[tid] = result
            if task.kind == "tile":
                tile_results.append(result)
            elif task.kind == "barrier" and isinstance(result, dict):
                outputs[task.iid] = result

        if len(completed) % _MEMORY_SAMPLE_EVERY == 0:
            cur = _sample_memory_mb() + _sample_pool_memory_mb(pool)
            if cur > peak_memory_mb:
                peak_memory_mb = cur

        release(tid, dep_failed=exc is not None)

        if progress_callback is not None:
            progress_callback(len(completed), len(tasks), task, exc is None, exc)

    while len(completed) < len(tasks):
        if cancel_flag and cancel_flag():
            # No pool teardown here by design (the pool is controller-owned
            # and reused): in-flight futures drain via their callbacks while
            # every pending task is marked failed below.
            for task in tasks:
                if task.task_id not in completed:
                    completed.add(task.task_id)
                    failed.add(task.task_id)
            break

        if pause_flag and pause_flag():
            while in_flight and not (cancel_flag and cancel_flag()):
                tid, exc = completions.get()
                consume_completion(tid, exc)

            while pause_flag and pause_flag() and not (cancel_flag and cancel_flag()):
                time.sleep(0.1)
            continue

        pump()
        if not in_flight and not ready:
            raise RuntimeError("Pipeline stalled (dependency cycle?)")
        tid, exc = completions.get()
        consume_completion(tid, exc)

    wall = max(0.0, time.monotonic() - t_start)
    tile_busy = busy.get("tile", 0.0)
    barrier_busy = busy.get("barrier", 0.0)
    total_busy = tile_busy + barrier_busy
    # Configured workers, not throttle-effective ones: occupancy
    # under-reports saturation whenever a throttle engaged.
    denom = wall * workers if workers else 0.0
    occupancy = (total_busy / denom) if denom > 0 else 0.0

    final_memory = _sample_memory_mb() + _sample_pool_memory_mb(pool)
    if final_memory > peak_memory_mb:
        peak_memory_mb = final_memory

    tile_weights = sorted(
        ((t.task_id, t.weight) for t in tasks if t.kind == "tile" and t.weight),
        key=lambda kv: -kv[1],
    )
    stats = {
        "wall_s": round(wall, 3),
        "workers": workers,
        "tasks_total": len(tasks),
        "tasks_failed": len(failed),
        "busy_s": round(total_busy, 3),
        "busy_tile_s": round(tile_busy, 3),
        "busy_barrier_s": round(barrier_busy, 3),
        "occupancy": round(occupancy, 4),
        "peak_memory_mb": round(peak_memory_mb, 1),
        "task_seconds": {k: round(v, 4) for k, v in task_secs.items()},
    }

    last_run_stats.clear()
    last_run_stats.update(stats)

    slowest = sorted(task_secs.items(), key=lambda kv: -kv[1])[:5]
    slow_txt = ", ".join(f"{k}={v:.1f}s" for k, v in slowest) or "n/a"
    w_max = tile_weights[0][1] if tile_weights else 0
    w_min = tile_weights[-1][1] if tile_weights else 0
    w_imbalance = (w_max / w_min) if w_min > 0 else 0
    logger.info(
        f"[Executor] tasks={len(tasks)} wall={wall:.1f}s "
        f"busy={total_busy:.1f}s (tiles {tile_busy:.1f}s / "
        f"barriers {barrier_busy:.1f}s) on {workers} workers -> "
        f"{occupancy * 100:.0f}% pool | peak RAM {peak_memory_mb:.0f} MB | "
        f"slowest: {slow_txt} | "
        f"tile weight: min={w_min:,} max={w_max:,} imbalance={w_imbalance:.1f}x"
    )

    tile_results.sort(key=lambda r: (r.get("node", ""), r.get("tile_id", "")))
    return outputs, tile_results, len(failed), stats
