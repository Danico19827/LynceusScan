# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Bottom-up quadtree merge of sparse base tiles (domain, no Qt).

Ficha (repo rigor rule: published algorithm + equations + deviations):
  Reference: H. Samet, "The Quadtree and Related Hierarchical Data
  Structures", ACM Computing Surveys 16(2), June 1984 (quadtree region
  representation; the bottom-up merge of homogeneous siblings).
  Merge predicate for an aligned 2x2 block G at level L (span 2**L):
    merge(G) <=> nonempty(G) AND NOT locked(G) AND sum(G) <= floor
                AND span(G) <= max_span
  Units: counts are integer points per base tile; spans are base cells.
  Declared deviations from textbook quadtree merging:
  - Empty cells contribute geometry with zero points (a merged block may
    cover empty bases; rectangularity requires full aligned squares, and
    empty members cost nothing downstream).
  - Locked cells (subdivided bases) poison all ancestors: a block touching
    subdivided data never merges, so dense subdivision keeps precedence
    for memory safety. This is a deliberate asymmetry (dense splits,
    sparse merges, never both on one base).
  - Single oversized bases (> floor) stay single: splitting them is the
    subdivision pass's job, not the merge's.

  The map is derived from integer counts with sorted traversal, so it is
  byte-deterministic across machines and worker counts. It feeds the
  tiler routing (P1): every nonempty base lands in exactly one final
  block, and downstream readers keep working on tile dicts unchanged.
"""

from __future__ import annotations

#: Safety cap for merged spans in base cells (downstream grid allocations
#: scale with extent: an 800 m supertile at 1 m stays trivial, a 10 km one
#: does not). Fixed internally in v1 (no config knob by design).
MAX_MERGE_SPAN = 8


def _identity(
    bases: list[int],
) -> tuple[dict[int, int], dict[int, dict]]:
    """No-merge map: every base routes to itself."""
    return ({b: b for b in bases}, {})


def plan_sparse_merge(
    base_counts: dict[int, int],
    subdivided: set[int] | None = None,
    *,
    cols: int,
    rows: int,
    floor: int,
    max_span: int = MAX_MERGE_SPAN,
) -> tuple[dict[int, int], dict[int, dict]]:
    """Map every nonempty base tile to its final block origin.

    Returns ``(super_of, blocks)``: ``super_of[base]`` is the origin base
    index (top-left corner cell) of the block holding it — itself when
    unmerged — and ``blocks[origin]`` holds ``{"span", "members", "points"}``
    for merged (span > 1) blocks only. Origins are corner cell indices and
    may address empty cells; blocks are always full aligned ``span`` x
    ``span`` squares, so members are a subset of the square.

    ``floor <= 0`` or ``max_span <= 1`` (or no bases) yields the identity
    map with zero work. ``subdivided`` bases never merge and poison their
    ancestor chain. Everything is integer-exact and traversal-sorted, hence
    deterministic.
    """
    bases = sorted(base_counts)
    subdivided = set(subdivided or ())
    if floor <= 0 or max_span <= 1 or not bases or cols <= 0 or rows <= 0:
        return _identity(bases)
    max_level = max(0, int(max_span).bit_length() - 1)
    if max_level <= 0:
        return _identity(bases)

    # cells: (level, gx, gy) -> [points, locked, members|None].
    # members None marks a tombstone: an empty-but-closed region that only
    # exists to poison ancestors (its points were finalized separately, so
    # it must never merge nor finalize anything itself).
    cells: dict[tuple[int, int, int], list] = {}
    for base in bases:
        col, row = base % cols, base // cols
        cells[(0, col, row)] = [
            int(base_counts[base]), base in subdivided, [base]]
    super_of: dict[int, int] = {}
    blocks: dict[int, dict] = {}

    def _origin(level: int, gx: int, gy: int) -> int:
        span = 1 << level
        return (gy * span) * cols + gx * span

    def _finalize(points: int, members: list[int], span: int,
                  origin: int) -> None:
        for base in members:
            super_of[base] = origin
        if span > 1:
            blocks[origin] = {
                "span": span,
                "members": sorted(members),
                "points": int(points),
            }

    for level in range(1, max_level + 1):
        span = 1 << level
        if span > max_span:
            break
        parents: dict[tuple[int, int], list] = {}
        for (lv, gx, gy), cell in cells.items():
            if lv != level - 1:
                continue
            parents.setdefault((gx // 2, gy // 2), []).append(
                ((gx, gy), cell))
        next_cells: dict[tuple[int, int, int], list] = {}
        for (pgx, pgy), group in sorted(parents.items()):
            total = sum(cell[0] for _, cell in group)
            locked = any(cell[1] for _, cell in group)
            members = sorted(
                base for _, cell in group for base in (cell[2] or []))
            if members and not locked and total <= floor:
                next_cells[(level, pgx, pgy)] = [total, False, members]
            else:
                for (gx, gy), cell in group:
                    if cell[2]:
                        _finalize(cell[0], cell[2], 1 << (level - 1),
                                  _origin(level - 1, gx, gy))
                next_cells[(level, pgx, pgy)] = [0, True, None]
        cells = next_cells
    for (lv, gx, gy), cell in sorted(cells.items()):
        if cell[2]:
            _finalize(cell[0], cell[2], 1 << lv, _origin(lv, gx, gy))
    return super_of, blocks
