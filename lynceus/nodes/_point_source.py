# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Shared point-cloud source resolution and cleaning I/O helpers.

This module is NOT a node: the registry discovery (AST) skips modules
without ``NODE_NAME``, so it can be imported by both the cleaning nodes
and any downstream node that consumes a (possibly transformed) stream.

Every tile-task that writes a transformed POINT_CLOUD declares its output
directory via ``PROCESSING_SPECS["point_cloud_dir"]`` (relative to
``session_dir``). ``build_dag`` propagates the immediate provider's folder
as ``ctx["point_source_dir"]``; ``point_src`` resolves the file to read, in
order:

1. the immediate provider's ``point_cloud_dir`` (cleaning / classification
   chain),
2. the legacy ``classified/`` folder when the ``classified`` flag is set,
3. the raw tile file.
"""

from __future__ import annotations

from pathlib import Path

import laspy
import numpy as np

from lynceus.processing.tiler import _laz_backend
from lynceus.processing import provenance

# ASPRS noise classes used by the cleaning nodes.
NOISE_LOW_CLASS = 7    # low point / noise
NOISE_HIGH_CLASS = 18  # high noise
GROUND_CLASS = 2       # never overwritten by tagging

# Sentinel the tiler uses when the source carries no CRS.
UNKNOWN_CRS = "Unknown"


def clean_crs(value) -> str | None:
    """Normalize a session CRS for writers (None when unknown).

    Mirrors ``write_geotiff``: GeoTIFF writers already fall back to None
    on unparseable CRS, while GPKG writers crash on the sentinel string.
    """
    if value in (None, "", UNKNOWN_CRS):
        return None
    return value


class MissingDimension(RuntimeError):
    """Raised when a cleaning node needs a dimension the source lacks."""


def quality_mask(record) -> np.ndarray | None:
    """Vendor QA exclusion (None when the dimension is absent).

    Withheld points (ASPRS 1.4, bit 2) are flagged by the vendor for
    exclusion from terrain and metric computation; they are culled
    unconditionally (no opt-out: the flag is vendor QA, not operator
    criteria). Returns None on legacy clouds without the dimension.
    """
    import numpy as np

    dims = set(record.point_format.dimension_names)
    if "withheld" not in dims:
        return None
    return ~np.asarray(record.withheld, dtype=bool)


def point_src(tile: dict, ctx: dict) -> str:
    """Resolve the point-cloud file a tile-task should read (see module docstring).

    The extension follows the run's intermediate format (``ctx["intermediate_ext"]``,
    default ``laz``); falls back to the raw tile file from the tiler.
    """
    ext = ctx.get("intermediate_ext", "laz")
    session_dir = Path(ctx["session_dir"])
    provider_dir = ctx.get("point_source_dir")
    if provider_dir:
        candidate = session_dir / provider_dir / f"{tile['tile_id']}.{ext}"
        if candidate.exists():
            return str(candidate)
    if ctx.get("classified"):
        candidate = session_dir / "classified" / f"{tile['tile_id']}.{ext}"
        if candidate.exists():
            return str(candidate)
    return tile["file"]


def clean_out_path(tile: dict, ctx: dict, folder: str) -> str:
    """Output path for a transformed POINT_CLOUD tile-task."""
    from lynceus.nodes._paths import scoped_dir

    ext = ctx.get("intermediate_ext", "laz")
    return str(scoped_dir(ctx, folder) / f"{tile['tile_id']}.{ext}")


def read_cloud(src: str) -> tuple:
    """Read a whole LAZ into memory; return (cloud, header, scales, offsets).

    Tiles are bounded by design, so one full read per tile is the same
    pattern used by Classify Ground (single decompression pass).
    """
    parts: list[np.ndarray] = []
    with laspy.open(src, laz_backend=_laz_backend()) as reader:
        header = reader.header
        for chunk in reader.chunk_iterator(2_000_000):
            parts.append(np.asanyarray(chunk.array))
        scales = header.scales.copy()
        offsets = header.offsets.copy()
    cloud = np.concatenate(parts) if len(parts) > 1 else (
        parts[0]
        if parts
        else laspy.ScaleAwarePointRecord.zeros(
            0,
            point_format=header.point_format,
            scales=scales,
            offsets=offsets,
        ).array
    )
    return cloud, header, scales, offsets


def write_clean_laz(
    out_path: str,
    cloud,
    header,
    scales,
    offsets,
    metadata: dict | None = None,
) -> None:
    """Write a (possibly filtered/reclassified) cloud back to LAZ."""
    if cloud.size == 0 and getattr(cloud, "dtype", None) is not None \
            and cloud.dtype.names is None:
        # Plain `np.empty(0)` (no structured dtype) cannot build a point
        # record; normalize to a proper empty record so downstream keeps
        # propagating empties instead of raising.
        cloud = laspy.ScaleAwarePointRecord.zeros(
            0,
            point_format=header.point_format,
            scales=scales,
            offsets=offsets,
        ).array
    record = laspy.ScaleAwarePointRecord(
        cloud, header.point_format, scales=scales, offsets=offsets
    )
    new_header = laspy.LasHeader(
        version=header.version, point_format=header.point_format
    )
    new_header.offsets = offsets
    new_header.scales = scales
    provenance.copy_product_vlrs(new_header, header)
    provenance.copy_header_identity(new_header, header)
    if metadata:
        provenance.fill_las_header(new_header, metadata)
    with laspy.open(
        str(out_path), mode="w", header=new_header, laz_backend=_laz_backend()
    ) as writer:
        writer.write_points(record)


def _copy_with_sidecar(src: str, out_path: str, prov_doc: dict) -> None:
    """Identity fast path: copy bytes + stamp this node's sidecar.

    Used when a node changes nothing (no-op mask, pass-through, empty
    source): LAZ decompression/recompression dominates tile time, so the
    bytes travel untouched. The embedded stamp stays truthful (it describes
    whoever last modified the bytes — the upstream node), while this node's
    document lands in the `<file>.meta.json` sidecar.
    """
    import shutil

    from lynceus.processing import provenance

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out_path)
    provenance.write_sidecar(out_path, prov_doc, {"copied_from": str(src)})


def run_clean(
    tile: dict,
    ctx: dict,
    module_id: str,
    folder: str,
    mask_fn,
    cls_value: int = NOISE_LOW_CLASS,
    force_remove: bool = False,
    protect_ground: bool = True,
) -> dict:
    """Generic cleaning tile-task.

    ``mask_fn(record) -> np.ndarray[bool]`` selects the points to KEEP
    (True = inside range / not noise). When ``action == "remove"`` (or
    ``force_remove``) the masked-out points are dropped; otherwise they are
    tagged with ``cls_value`` (non-destructive, default noise class 7).

    Tagging never overwrites ground (class 2) unless ``protect_ground`` is
    False: an explicit operator opt-in for vendor clouds where noise was
    mislabeled as ground and must be retagged before removal. The remove
    path is unaffected by this flag.

    If ``mask_fn`` raises :class:`MissingDimension`, the tile is passed
    through unchanged with a warning so the pipeline does not break.
    Vendor-withheld points are always excluded from the kept set
    (see :func:`quality_mask`).
    Identity results (nothing tagged/removed) travel as a byte copy instead
    of a recompress (see :func:`_copy_with_sidecar`).
    """
    out_path = clean_out_path(tile, ctx, folder)
    if Path(out_path).exists():
        return {"tile_id": tile["tile_id"], "node": module_id, "output": out_path}

    action = ctx.get("action", "classify")
    if action not in ("classify", "remove"):
        raise ValueError(
            f"{module_id}: unknown action {action!r} "
            "(expected 'classify' or 'remove')"
        )
    remove = force_remove or action == "remove"
    if not remove and int(cls_value) == GROUND_CLASS:
        raise ValueError(
            f"{module_id}: target_class must not be {GROUND_CLASS} (ground); "
            "tagging would create ground from outliers"
        )
    if not remove and int(cls_value) in (8, 12):
        raise ValueError(
            f"{module_id}: target_class must not be {int(cls_value)} "
            "(ASPRS-reserved in LAS 1.4+); tagging would corrupt the "
            "standard classification"
        )

    prov_doc = provenance.build_provenance(ctx, Path(out_path).name)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    src = point_src(tile, ctx)
    cloud, header, scales, offsets = read_cloud(src)
    if cloud.shape[0] == 0:
        _copy_with_sidecar(src, out_path, prov_doc)
        return {
            "tile_id": tile["tile_id"],
            "node": module_id,
            "output": out_path,
            "warnings": [f"{tile['tile_id']}: no points in source; empty tile written"],
        }
    record = laspy.ScaleAwarePointRecord(
        cloud, header.point_format, scales=scales, offsets=offsets
    )

    try:
        keep = np.asarray(mask_fn(record), dtype=bool)
    except MissingDimension as exc:
        _copy_with_sidecar(src, out_path, prov_doc)
        return {
            "tile_id": tile["tile_id"],
            "node": module_id,
            "output": out_path,
            "warnings": [f"{tile['tile_id']}: {exc}"],
        }
    vendor = quality_mask(record)
    if vendor is not None:
        keep = keep & vendor

    notkeep = ~keep
    if remove:
        dropped = int(notkeep.sum())
        if dropped == 0:
            _copy_with_sidecar(src, out_path, prov_doc)
            return {
                "tile_id": tile["tile_id"],
                "node": module_id,
                "output": out_path,
                "removed": 0,
            }
        cloud = cloud[keep]
    else:
        # Tagging never overwrites ground (class 2) unless the operator
        # opted out via protect_ground: the masks select noise/outliers,
        # but a point that already is ground stays by default.
        cls = np.asarray(record.classification)
        if protect_ground:
            tagged = notkeep & (cls != GROUND_CLASS)
        else:
            tagged = notkeep
        tagged_count = int(tagged.sum())
        if tagged_count == 0:
            _copy_with_sidecar(src, out_path, prov_doc)
            return {
                "tile_id": tile["tile_id"],
                "node": module_id,
                "output": out_path,
                "removed": 0,
                "tagged_noise": 0,
                "warnings": [
                    f"{tile['tile_id']}: no points outside the range to tag"
                ],
            }
        record.classification[tagged] = int(cls_value)
        ground_retagged = int((tagged & (cls == GROUND_CLASS)).sum())

    write_clean_laz(out_path, cloud, header, scales, offsets, metadata=prov_doc)
    result = {
        "tile_id": tile["tile_id"],
        "node": module_id,
        "output": out_path,
        "removed": dropped if remove else 0,
    }
    if not remove:
        result["tagged_noise"] = tagged_count
        result["ground_retagged"] = ground_retagged
        if ground_retagged:
            result["warnings"] = [
                f"{tile['tile_id']}: {ground_retagged} ground (class 2) "
                f"points retagged as {int(cls_value)}"
            ]
    return result