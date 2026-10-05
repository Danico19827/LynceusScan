# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Intermediate point-cloud format selection (P2a).

Tiles and every transformed stream (classified/, clean_*) are written either
as LAZ (compressed) or LAS (raw). LAS makes encode/decode nearly free, which
dominates the wall-clock of multi-stage pipelines over huge clouds; LAZ keeps
the source file and the final export small on disk.

The selection is automatic but conservative: ``laz`` is the default because
for ordinary datasets its smaller tiles read faster than their LAS siblings
(LAZ decode is cheap and disk I/O cheaper). ``las`` only wins when the raw
footprint is so large that the codec work dwarfs the extra bytes on disk.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# A dataset whose raw LAS footprint exceeds this is large enough that
# skip-the-codec wins even on beefy workstations.
_UNCOMPRESSED_LAS_TRIGGER_BYTES = 5_000_000_000
# Average encoded bytes per point used to approximate the raw footprint.
_BYTES_PER_POINT_UNCOMPRESSED = 30

INTERMEDIATE_LAS = "las"
INTERMEDIATE_LAZ = "laz"


def choose_intermediate_format(
    total_ram: int | None = None, total_points: int | None = None
) -> str:
    """Return the intermediate extension (``"las"`` or ``"laz"``) for a run.

    Only the scale of the dataset matters: any dataset whose uncompressed
    footprint is below ``_UNCOMPRESSED_LAS_TRIGGER_BYTES`` stays on ``laz``
    (the fast path for typical clouds, and the format of legacy tile caches).
    ``total_ram`` is accepted for API symmetry and future policies, but does
    not flip the result on its own.
    """
    if total_points is not None and (
        total_points * _BYTES_PER_POINT_UNCOMPRESSED
        >= _UNCOMPRESSED_LAS_TRIGGER_BYTES
    ):
        return INTERMEDIATE_LAS
    return INTERMEDIATE_LAZ


def format_enabled() -> str:
    """Environment override for reproducible runs and tests.

    ``LYNCEUS_INTERMEDIATE_FORMAT`` may be ``las`` or ``laz``; empty
    (the default) leaves the automatic selection untouched. Unknown values
    warn and fall back to automatic (typo visibility).
    """
    value = os.environ.get("LYNCEUS_INTERMEDIATE_FORMAT", "").strip().lower()
    if value in (INTERMEDIATE_LAS, INTERMEDIATE_LAZ):
        return value
    if value:
        logger.warning(
            "Ignoring unknown LYNCEUS_INTERMEDIATE_FORMAT=%r (expected "
            "'las' or 'laz')",
            value,
        )
    return ""