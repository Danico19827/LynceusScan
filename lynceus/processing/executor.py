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
# Default working-set multiplier for a barrier task over its on-disk input
# bytes. Barriers run whole products at once (mosaics, tables, vectors);
# numpy-heavy ones upcast and hold several full-size temporaries, so the
# peak is a small multiple of the inputs. Modules with a measured profile
# declare their own PROCESSING_SPECS["memory_factor"] instead.
_DEFAULT_BARRIER_FACTOR = 4.0


def default_workers(total: int | None = None) -> int:
    """Engine worker default: all cores minus one for the system.

    The pool must never fully hog the machine (a 12/12 run starves the
    GUI event loop and the OS: the app "hangs" while healthy). Explicit
    counts still flow through untouched; only the default reserves.
    Floor of 1 keeps single-core machines working.
    """
    try:
        cores = int(total if total is not None else (os.cpu_count() or 4))
    except (TypeError, ValueError):
        cores = 4
    return max(1, cores - 1)


def workers_for_percent(percent: int | float, total: int | None = None) -> int:
    """Pool size for a CPU percentage (Preferences > Performance).

    Pure helper so the UI math stays tested: clamps to [1, total], then
    to the engine default — 100% means every core but the system's, by
    product policy (see :func:`default_workers`).
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
    return max(1, min(default_workers(cores),
                      int(round(cores * pct / 100.0))))


def create_pool(max_workers: int | None = None):
    """Create the process pool; this must run on the main thread.

    Explicit counts are respected as given (CLI power override); an
    unset count takes the engine default (one core stays with the
    system).
    """
    from multiprocessing import get_context

    try:
        workers = int(max_workers) if max_workers is not None else 0
    except (TypeError, ValueError):
        workers = 0
    if workers < 1:
        workers = default_workers()
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


def _task_ctx(task: Task) -> dict:
    """Task context dict, unwrapping any dispatch envelope.

    Core tasks carry ``(ctx,)`` (or ``(fn, ctx)`` through the barrier
    wrapper); extension tasks go through run_file_node_task as
    ``(module_id, fn_name, (ctx,))``. Tile tasks carry ``(tile, ctx)``:
    the ctx (holding ``session_dir``) wins over the tile dict, so new
    wrappers never blind the dispatch estimates again.
    """
    args = getattr(task, "args", ()) or ()
    fallback: dict = {}
    stack = [args]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            if "session_dir" in value:
                return value
            if not fallback:
                fallback = value
        elif isinstance(value, (list, tuple)):
            stack.extend(value)
    return fallback


def _task_tile(task: Task) -> dict:
    """Tile descriptor dict of a tile task ({} when absent).

    Tile tasks carry ``(tile, ctx)``: the tile is the dict WITHOUT
    ``session_dir`` (core bounds, ids, counts). Used for area-driven
    estimates; missing bounds simply yield no grid term.
    """
    args = getattr(task, "args", ()) or ()
    stack = [args]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            if "session_dir" not in value:
                return value
        elif isinstance(value, (list, tuple)):
            stack.extend(value)
    return {}


def _tile_grid_bytes(task: Task) -> int:
    """Area-driven working set of a tile task (0 when not a grid node).

    Grid-building nodes (ground filters, surface accumulators) allocate
    by tile AREA at the node's grid resolution, not by point count: a
    sparse supertile holds a 683 MB PMF grid for 116k points. Cells come
    from the tile core bounds; missing bounds or resolution keep the
    legacy point-only estimate bit for bit.
    """
    try:
        bpc = float(getattr(task, "grid_bpc", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0
    try:
        res = float(getattr(task, "grid_res_m", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0
    if bpc <= 0.0 or res <= 0.0:
        return 0
    tile = _task_tile(task)
    try:
        width = float(tile["core_x_max"]) - float(tile["core_x_min"])
        height = float(tile["core_y_max"]) - float(tile["core_y_min"])
    except (KeyError, TypeError, ValueError):
        return 0
    if width <= 0.0 or height <= 0.0:
        return 0
    import math

    cells = max(1, math.ceil(width / res)) * max(1, math.ceil(height / res))
    return int(cells * bpc)


def _barrier_input_bytes(task: Task) -> int:
    """On-disk bytes of a barrier task's already-materialized inputs.

    Three channels, all complete before dispatch: provider products
    (``*_path`` ctx keys from ``_inject_input_paths``), tile fragments
    matching the task's ``input_globs`` under its artifacts root, and
    per-segment products injected as ``segment_outputs`` (consolidators).
    Only real files count: missing paths, directories and non-string
    values are skipped, and a barrier with no file inputs budgets zero
    (unchanged).
    """
    total = 0
    try:
        ctx = _task_ctx(task)
    except AttributeError:
        return 0
    try:
        items = ctx.items()
    except AttributeError:
        items = ()
    for key, value in items:
        if not (isinstance(key, str) and key.endswith("_path")):
            continue
        if not (isinstance(value, str) and value):
            continue
        try:
            if os.path.isfile(value):
                total += os.path.getsize(value)
        except OSError:
            continue
    globs = getattr(task, "input_globs", ()) or ()
    if globs:
        try:
            root = (ctx.get("session_dir") or ctx.get("artifacts") or "")
        except AttributeError:
            root = ""
        if root:
            for path in _glob_files(root, globs):
                try:
                    total += path.stat().st_size
                except OSError:
                    continue
    segment_outputs = None
    try:
        segment_outputs = ctx.get("segment_outputs")
    except AttributeError:
        segment_outputs = None
    if segment_outputs:
        total += _nested_file_bytes(segment_outputs)
    return total


def _glob_files(root, globs: list) -> list:
    """Existing files matching glob patterns under a root (best-effort)."""
    from pathlib import Path

    found: list = []
    try:
        base = Path(root)
    except (OSError, TypeError, ValueError):
        return found
    for pattern in globs or ():
        try:
            matches = base.glob(str(pattern))
        except (OSError, ValueError):
            continue
        for path in matches:
            try:
                if path.is_file():
                    found.append(path)
            except OSError:
                continue
    return found


_RASTER_INPUT_EXTS = (".tif", ".tiff", ".vrt")


def _barrier_input_files(task: Task) -> list[str]:
    """Raster inputs of a barrier task for header-exact sizing.

    Tile fragments matching ``input_globs`` plus single-file ``*_path``
    raster inputs (mosaic consumers): the same files the writer will
    open, so the mode-aware estimate sizes the same union. Best-effort:
    missing roots, bad patterns and non-files yield fewer (never fail).
    """
    try:
        ctx = _task_ctx(task)
    except AttributeError:
        return []
    try:
        items = ctx.items()
    except AttributeError:
        items = ()
    files: list[str] = []
    seen: set[str] = set()
    for key, value in items:
        if not (isinstance(key, str) and key.endswith("_path")):
            continue
        if not (isinstance(value, str) and value):
            continue
        if value.lower().endswith(_RASTER_INPUT_EXTS) and value not in seen:
            try:
                if os.path.isfile(value):
                    seen.add(value)
                    files.append(value)
            except OSError:
                continue
    globs = getattr(task, "input_globs", ()) or ()
    if globs:
        try:
            root = (ctx.get("session_dir") or ctx.get("artifacts") or "")
        except AttributeError:
            root = ""
        if root:
            for path in _glob_files(root, globs):
                text = str(path)
                if text not in seen:
                    seen.add(text)
                    files.append(text)
    return files


def _mode_aware_barrier_est(task: Task, worker_budget) -> int | None:
    """Peak of the mode the writer WILL run (None = legacy estimate).

    The writer goes direct when the union need fits the per-worker
    budget and streams otherwise; budgeting the same decision at
    dispatch keeps windowed barriers admittable (their peak is small)
    while direct ones serialize on their true need. Any doubt (no
    declared copies, no files, unreadable headers, no budget) returns
    None and the caller keeps the legacy input-bytes estimate.
    """
    try:
        copies = float(getattr(task, "merge_copies", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    if copies <= 0.0:
        return None
    try:
        cell = int(getattr(task, "merge_cell_bytes", 0) or 4)
    except (TypeError, ValueError):
        cell = 4
    if cell <= 0:
        cell = 4
    try:
        budget = int(worker_budget or 0)
    except (TypeError, ValueError):
        return None
    if budget <= 0:
        return None
    files = _barrier_input_files(task)
    if not files:
        return None
    try:
        from lynceus.processing.raster import mosaic_need, window_peak
    except (ImportError, OSError):
        return None
    try:
        need = mosaic_need(files, copies, cell)
    except Exception:
        return None
    if need is None:
        return None
    if need <= budget:
        return need
    return window_peak(budget, cell)


def _nested_file_bytes(node) -> int:
    """Sum of existing-file sizes in nested dict/list/tuple payloads."""
    total = 0
    stack = [node]
    seen: set[str] = set()
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            stack.extend(current.values())
        elif isinstance(current, (list, tuple)):
            stack.extend(current)
        elif isinstance(current, str) and current and current not in seen:
            seen.add(current)
            try:
                if os.path.isfile(current):
                    total += os.path.getsize(current)
            except OSError:
                continue
    return total


def _task_est_bytes(task: Task, worker_budget=None) -> int:
    """Estimated peak working set of a task in bytes.

    Tile tasks scale with their point count (weight, also used for queue
    priority) times the module's ``tile_weight_factor`` when it declares one:
    neighborhood algorithms (k-NN trees, full-cloud passes) hold several
    full-size temporaries beyond the base record array, so their working set
    is a small multiple of the raw estimate. The multiplier only affects
    admission -- queue order stays by raw weight so a heavy node does not
    starve light neighbors. Grid-building nodes add an AREA term (cells ×
    ``grid_bpc`` from the tile core bounds at ``grid_res_m``): point counts
    cannot see a sparse supertile's 683 MB grid. Barrier tasks declaring
    ``merge_copies`` budget the peak of the mode the writer WILL run
    (direct need when it fits ``worker_budget``, window peak when it
    streams); without copies the legacy input-bytes × ``memory_factor``
    applies. ``worker_budget`` (per-worker bytes, None = legacy) comes
    from the dispatcher, which shares it with the writers' own budget.
    """
    if task.kind == "tile" and task.weight > 0:
        tile_f = getattr(task, "tile_f", 0.0) or 0.0
        try:
            tile_f = float(tile_f)
        except (TypeError, ValueError):
            tile_f = 0.0
        factor = tile_f if tile_f > 0.0 else 1.0
        return int(task.weight * _BYTES_PER_POINT_EST * factor) + _tile_grid_bytes(task)
    if task.kind == "barrier":
        mode_est = _mode_aware_barrier_est(task, worker_budget)
        if mode_est is not None:
            return mode_est
        factor = getattr(task, "mem_factor", 0.0) or 0.0
        try:
            factor = float(factor)
        except (TypeError, ValueError):
            factor = 0.0
        if factor <= 0.0:
            factor = _DEFAULT_BARRIER_FACTOR
        return int(_barrier_input_bytes(task) * factor)
    return 0


def _predictive_budget(total_ram: int, cap_bytes: int | None = None) -> float:
    """Dispatch budget: the performance memory preference caps system RAM.

    An explicit cap only ever tightens the gate (never widens it past RAM),
    so unset preferences keep the historical all-RAM behavior bit for bit.
    """
    base = total_ram
    if cap_bytes is not None:
        try:
            cap = int(cap_bytes)
        except (TypeError, ValueError):
            cap = 0
        if cap > 0:
            base = min(base, cap)
    return base * _PREDICTIVE_RAM_RATIO if base > 0 else float("inf")


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
    memory_budget_bytes: int | None = None,
    on_barrier_done: Callable[[Task, dict], None] | None = None,
) -> tuple[dict, list[dict], int, dict]:
    """Execute the DAG while keeping the worker pool busy.

    Returns ``(outputs, tile_results, failed, stats)`` for successful barriers,
    tile-task results, failed tasks including dependency cascades, and metrics.

    Callbacks run in the execution thread. ``on_task_start`` fires when a task
    is submitted; ``progress_callback`` fires when a task reaches a final state
    (with the exception on failure, None for dependency cascades);
    ``on_barrier_done`` fires at each successful barrier completion with the
    barrier payload, so callers can publish products incrementally instead of
    waiting for the whole DAG (the final ``outputs`` still carry everything).

    ``cancel_flag`` stops dispatch and marks pending tasks. ``pause_flag``
    stops dispatch while allowing in-flight tasks to complete.
    ``memory_budget_bytes`` caps the predictive dispatch budget (an explicit
    performance preference only tightens the gate; None keeps the
    historical all-RAM behavior).
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

    # Barrier input files are complete once their tasks are dispatchable,
    # so estimates are stable per task: resolve once, reuse everywhere.
    # The per-worker share mirrors the writers' own budget, so the mode
    # the estimate assumes (direct need vs window peak) is the mode that
    # runs.
    est_cache: dict[str, int] = {}
    try:
        _share = int(memory_budget_bytes or 0) // max(1, int(workers))
    except (TypeError, ValueError):
        _share = 0
    worker_share = _share if _share > 0 else None

    def est_of(task: Task) -> int:
        est = est_cache.get(task.task_id)
        if est is None:
            est = _task_est_bytes(task, worker_share)
            est_cache[task.task_id] = est
        return est

    def pump() -> None:
        limit = _effective_workers()
        total_ram = psutil.virtual_memory().total
        pred_budget = _predictive_budget(total_ram, memory_budget_bytes)
        # Projected in-flight RAM from dispatched tasks with a known estimate
        # (tiles by point count, barriers by materialized input bytes).
        inflight_bytes = _project_inflight_bytes(in_flight.values())
        # Lightest-first admission control: the ready list is heaviest-first
        # but pop() takes the lightest end. Trying the lightest task first
        # keeps the pool full under the RAM budget (best-fit-ish); if even
        # it does not fit, nothing will, so dispatch pauses correctly.
        ready.sort(key=lambda t: (est_of(t), t.weight), reverse=True)
        while len(in_flight) < limit and ready:
            task = ready.pop()
            est = est_of(task)
            # Allow an oversized task to run alone (nothing to do about its
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
            if isinstance(exc, MemoryError):
                est_mb = est_of(task) / (1024 * 1024)
                budget_mb = _predictive_budget(
                    psutil.virtual_memory().total, memory_budget_bytes
                ) / (1024 * 1024)
                logger.error(
                    f"[Executor] Task {tid} ran out of memory "
                    f"(est. working set {est_mb:.0f} MB vs "
                    f"{budget_mb:.0f} MB dispatch budget)"
                )
            else:
                logger.error(f"[Executor] Task {tid} failed: {exc}")
        else:
            result = future.get()
            results[tid] = result
            if task.kind == "tile":
                tile_results.append(result)
            elif task.kind == "barrier" and isinstance(result, dict):
                outputs[task.iid] = result
                if on_barrier_done is not None:
                    try:
                        on_barrier_done(task, result)
                    except Exception:
                        logger.exception(
                            f"[Executor] on_barrier_done failed for {tid}"
                        )

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
