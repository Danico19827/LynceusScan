# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Pre-flight advisor math (domain, no Qt).

Runs in the GUI thread before a run starts (once per queue, never blocking)
to size the workload from LAS/LAZ headers alone: total points, bbox area,
expected tile count, and a rough peak working set built from the executor's
per-point estimate times each tile node's ``tile_weight_factor``. The result
is advisory: thresholds and factors are documented heuristics refined with
session telemetry (``session_meta.run_metrics.peak_memory_mb``), not a hard
gate -- the executor's dispatch throttle remains the enforcement layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from lynceus.processing.executor import _BYTES_PER_POINT_EST
from lynceus.processing.point_counts import union_area_m2

#: Default tile side when the run does not override it (matches the engine).
DEFAULT_TILE_SIZE_M = 100.0

#: A cloud above this size (points) is worth the "enable cache for this
#: run" suggestion when enrichment would otherwise restart from scratch.
CACHE_POINTS_THRESHOLD = 80_000_000
#: ...or a chain this long (tile nodes in scope) warrants the same advice.
CACHE_NODE_THRESHOLD = 4

#: Advise opening Performance when the estimated peak reaches this fraction
#: of the configured memory budget: the predictive gate (budget * 0.70)
#: will already throttle, so tighter settings buy headroom.
RAM_ADVICE_RATIO = 0.70

#: Suggest sparse-tile merging when the estimated base tiles reach this
#: count while averaging below the auto floor: thousands of micro-tiles
#: mean file/task/merge overhead dominates the run.
MERGE_TILE_THRESHOLD = 20_000
#: Suggest independent segments when the full mosaic would not fit but a
#: bounded batch would: at least this many segments (fewer is pointless)
#: and at most this many (more is session spam).
SEGMENTS_MIN_COUNT = 2
SEGMENTS_MAX_COUNT = 200
#: Segment point cap targets a mosaic at most this fraction of the budget
#: (same headroom policy as the RAM advice).
SEGMENT_BUDGET_RATIO = 0.50
#: Auto floor targets this many supertiles, bounded both ways so per-task
#: working sets stay comfortable and the suggestion stays sane on small
#: clouds. Deterministic from source size (never ambient RAM) so tilings
#: stay portable across machines.
MERGE_TARGET_SUPERTILES = 3000
MERGE_FLOOR_MIN = 50_000
MERGE_FLOOR_MAX = 2_000_000

#: Full-extent mosaic working set model: float32 cells times working
#: copies (merge + post-process passes). Barriers materialize whole
#: mosaics, so this term — not per-tile math — decides feasibility on
#: wide sparse surveys. Reference 1 m cells (documented assumption in
#: the message); coarser products scale it down quadratically.
MOSAIC_BYTES_PER_CELL = 4.0
MOSAIC_WORKING_COPIES = 3.0
#: Coarsest product resolution the auto-adjust proposes (m): respects the
#: terrain nodes' schema maximum (finer control stays manual).
RESOLUTION_MAX_M = 5.0
#: Raising the memory budget only helps while the need fits in physical
#: RAM with headroom left for the OS/GUI/pool. Beyond that only a
#: coarser resolution (or the windowed engine) helps.
BUDGET_HEADROOM_RATIO = 0.90


def suggested_merge_floor(total_points: int) -> int:
    """Deterministic AUTO floor from source size (~3000 supertiles)."""
    try:
        total = int(total_points)
    except (TypeError, ValueError):
        total = 0
    return max(MERGE_FLOOR_MIN, min(
        MERGE_FLOOR_MAX, max(1, total // MERGE_TARGET_SUPERTILES)))


@dataclass
class Preflight:
    """Advisory sizing for one run (or one queue iteration).

    Everything is an estimate from headers; ``est_peak_bytes`` assumes every
    worker holds one tile of the heaviest in-scope factor at once, which is
    the steady-state ceiling for a tuned pipeline.
    """

    total_points: int
    area_m2: float | None
    tile_count_est: int
    per_tile_points: int
    heaviest_factor: float
    tile_nodes: int
    workers: int
    est_peak_bytes: float
    budget_bytes: float | None
    cache_enabled: bool
    ask_cache: bool
    ask_ram: bool
    merge_floor_suggest: int = 0
    ask_mosaic: bool = False
    mosaic_bytes_est: float = 0.0
    workers_suggest: int = 0
    resolution_suggest: float = 0.0
    ram_budget_suggest: float = 0.0
    segments_cap_suggest: int = 0
    segments_count_est: int = 0

    @property
    def has_advice(self) -> bool:
        return (
            self.ask_cache or self.ask_ram or self.merge_floor_suggest > 0
            or self.ask_mosaic or self.segments_cap_suggest > 0
        )


def build_preflight(
    source_files: list[str],
    tile_factors: list[float],
    *,
    workers: int,
    tile_size_m: float = DEFAULT_TILE_SIZE_M,
    budget_bytes: float | None = None,
    cache_enabled: bool = False,
    merge_enabled: bool = False,
    has_mosaic: bool = False,
    total_ram_bytes: float | None = None,
    seg_cap: int = 0,
    segment_independent: bool = False,
) -> Preflight | None:
    """Size a run from header data; ``None`` when nothing is readable.

    ``tile_factors`` are the ``tile_weight_factor`` values of every tile node
    in scope (1.0 for undeclared). ``budget_bytes`` is the Performance memory
    preference. Unreadable sources only shrink the estimate -- the advisor
    must never block a run, so an unreadable file yields ``None`` advice.
    ``merge_enabled`` silences the sparse-merge suggestion when the loader
    already fuses (suggest-only by design: tiling knobs live in the
    project, never as phantom per-run overrides). ``has_mosaic`` flags a
    raster-mosaic barrier in scope so the whole-grid working set (which
    per-tile math cannot see) is sized too. ``total_ram_bytes`` lets the
    mosaic advice prefer raising the budget (quality preserved) when the
    need fits in physical RAM, falling back to coarser resolution only
    when no setting could fit it. ``segment_independent`` silences the
    independent-segments suggestion when the loader already runs that
    way; ``seg_cap`` carries the configured batch cap (0 = unset) so the
    suggestion only fires when batching would actually change shape.
    """
    if not source_files:
        return None
    total_points, area = union_area_m2(source_files)
    if total_points <= 0:
        return None
    tile_size = max(1.0, float(tile_size_m))
    tile_count = 1
    if area is not None:
        tile_count = max(1, int(math.ceil(area / (tile_size * tile_size))))
    per_tile = max(1, int(math.ceil(total_points / max(1, tile_count))))
    heaviest = 1.0
    if tile_factors:
        values = []
        for f in tile_factors:
            try:
                values.append(float(f))
            except (TypeError, ValueError):
                values.append(0.0)
        heaviest = max([1.0] + [v for v in values if v > 0.0])
    ask_cache = (
        not cache_enabled
        and (total_points >= CACHE_POINTS_THRESHOLD or len(tile_factors) >= CACHE_NODE_THRESHOLD)
    )
    merge_floor_suggest = 0
    if (
        not merge_enabled
        and tile_count >= MERGE_TILE_THRESHOLD
        and per_tile < suggested_merge_floor(total_points)
    ):
        merge_floor_suggest = suggested_merge_floor(total_points)
    # Size for the recommended configuration: with a suggested merge the
    # fused tiles (not the micro-tile average) drive per-worker peaks.
    per_tile_eff = max(per_tile, merge_floor_suggest)
    est_peak = float(workers) * per_tile_eff * _BYTES_PER_POINT_EST * heaviest
    try:
        budget = float(budget_bytes) if budget_bytes is not None else 0.0
    except (TypeError, ValueError):
        budget = 0.0
    ask_ram = bool(budget > 0.0 and est_peak >= RAM_ADVICE_RATIO * budget)
    workers_suggest = 0
    if ask_ram and budget > 0.0:
        per_worker = max(
            1, int(per_tile_eff * _BYTES_PER_POINT_EST * heaviest))
        workers_suggest = max(1, int(budget * RAM_ADVICE_RATIO // per_worker))
        if workers_suggest >= max(1, int(workers or 1)):
            workers_suggest = 0
    ask_mosaic = False
    mosaic_bytes_est = 0.0
    resolution_suggest = 0.0
    ram_budget_suggest = 0.0
    if has_mosaic and area is not None and budget > 0.0:
        mosaic_bytes_est = (
            float(area) * MOSAIC_BYTES_PER_CELL * MOSAIC_WORKING_COPIES)
        if mosaic_bytes_est >= RAM_ADVICE_RATIO * budget:
            ask_mosaic = True
            need = mosaic_bytes_est / RAM_ADVICE_RATIO
            try:
                total_ram = float(total_ram_bytes) if (
                    total_ram_bytes is not None) else 0.0
            except (TypeError, ValueError):
                total_ram = 0.0
            if total_ram > 0.0 and need <= BUDGET_HEADROOM_RATIO * total_ram:
                # Fits in the machine: raise the budget (bytes, rounded
                # up to GiB), keep full resolution.
                ram_budget_suggest = (
                    math.ceil(need / 1024 ** 3) * 1024 ** 3)
            else:
                res = 1.0
                est = mosaic_bytes_est
                while est >= RAM_ADVICE_RATIO * budget and res < RESOLUTION_MAX_M:
                    res *= 2.0
                    est /= 4.0
                resolution_suggest = min(res, RESOLUTION_MAX_M)
    segments_cap_suggest = 0
    segments_count = 0
    if (
        not segment_independent
        and has_mosaic
        and area is not None
        and budget > 0.0
        and mosaic_bytes_est >= RAM_ADVICE_RATIO * budget
    ):
        # The full mosaic would not fit: size per-segment batches from
        # headers so each segment mosaic stays within headroom. A
        # configured cap that already fits is kept as-is.
        try:
            density = total_points / float(area)
        except (ArithmeticError, TypeError, ValueError):
            density = 0.0
        if density > 0.0:
            def _seg_fits(cap_value: int) -> bool:
                cells = cap_value / density
                return (
                    cells * MOSAIC_BYTES_PER_CELL * MOSAIC_WORKING_COPIES
                    <= SEGMENT_BUDGET_RATIO * budget
                )

            try:
                configured = int(seg_cap or 0)
            except (TypeError, ValueError):
                configured = 0
            if configured > 0 and _seg_fits(configured):
                candidate = configured
            else:
                candidate = int(
                    SEGMENT_BUDGET_RATIO * budget
                    / (MOSAIC_BYTES_PER_CELL * MOSAIC_WORKING_COPIES)
                    * density
                )
            if candidate > 0:
                n_seg = int(math.ceil(total_points / candidate))
                if SEGMENTS_MIN_COUNT <= n_seg <= SEGMENTS_MAX_COUNT:
                    segments_cap_suggest = int(candidate)
                    segments_count = int(n_seg)
    return Preflight(
        total_points=total_points,
        area_m2=area,
        tile_count_est=tile_count,
        per_tile_points=per_tile,
        heaviest_factor=heaviest,
        tile_nodes=len(tile_factors),
        workers=max(1, int(workers or 1)),
        est_peak_bytes=est_peak,
        budget_bytes=budget_bytes,
        cache_enabled=cache_enabled,
        ask_cache=ask_cache,
        ask_ram=ask_ram,
        merge_floor_suggest=merge_floor_suggest,
        ask_mosaic=ask_mosaic,
        mosaic_bytes_est=mosaic_bytes_est,
        workers_suggest=workers_suggest,
        resolution_suggest=resolution_suggest,
        ram_budget_suggest=ram_budget_suggest,
        segments_cap_suggest=segments_cap_suggest,
        segments_count_est=(
            segments_count if segments_cap_suggest > 0 else 0),
    )