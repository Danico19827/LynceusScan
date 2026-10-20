# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Shared barrier machinery for the Merge Rasters strategy family.

This module is NOT a node (no NODE_ID): variant modules call
``merge_rasters`` with their product contract. Each variant declares up to
8 same-type input ports; wired providers arrive positionally as
``in0_path``..``in7_path`` (see ``_inject_input_paths``).
"""

from __future__ import annotations

MAX_MERGE_INPUTS = 8


def _wired_inputs(task_ctx: dict) -> list[str]:
    """Provider files actually wired, in plug order."""
    from pathlib import Path

    files = []
    for pos in range(MAX_MERGE_INPUTS):
        path = task_ctx.get(f"in{pos}_path", "") or ""
        if path and Path(path).is_file():
            files.append(str(path))
    return files


def _describe(path: str) -> dict:
    """Header facts needed for merge validation (no pixel reads)."""
    import rasterio

    with rasterio.open(path) as src:
        return {
            "crs": str(src.crs) if src.crs else "",
            "dtype": str(src.dtypes[0]) if src.dtypes else "",
            "count": int(src.count),
            "res": (abs(float(src.transform.a)), abs(float(src.transform.e))),
            "nodata": src.nodata,
        }


def merge_rasters(
    task_ctx: dict,
    *,
    session_file: str,
    kind_label: str,
    display_label: str,
    module_id: str,
    nodata_default: float = -9999.0,
) -> dict:
    """Fuse the wired same-product rasters into one shared-scale mosaic."""
    import numpy as np

    from lynceus.nodes._paths import scoped_file
    from lynceus.nodes._point_source import clean_crs
    from lynceus.processing import provenance
    from lynceus.processing.raster import (
        _raster_shape,
        _valid_box_streaming,
        finish_mosaic,
        write_geotiff,
    )

    files = _wired_inputs(task_ctx)
    if len(files) < 2:
        raise RuntimeError(
            f"Merge {display_label} needs at least two rasters wired "
            f"(got {len(files)})"
        )
    nodata = float(task_ctx.get("nodata_value", nodata_default))
    method = str(task_ctx.get("method", "first"))
    if method not in ("first", "last", "min", "max"):
        raise RuntimeError(
            f"Merge {display_label}: unknown method {method!r} "
            "(expected first/last/min/max)"
        )

    infos = [_describe(path) for path in files]
    crs_set = {info["crs"] for info in infos if info["crs"]}
    if len(crs_set) > 1:
        raise RuntimeError(
            f"Merge {display_label} needs one CRS, got "
            + ", ".join(sorted(f"'{crs}'" for crs in crs_set))
        )
    dtypes = {info["dtype"] for info in infos}
    counts = {info["count"] for info in infos}
    if len(dtypes) > 1 or len(counts) > 1:
        raise RuntimeError(
            f"Merge {display_label} needs matching dtype/bands, got "
            f"dtype={sorted(dtypes)} bands={sorted(counts)}"
        )
    warnings: list[str] = []
    resolutions = {info["res"] for info in infos}
    if len(resolutions) > 1:
        warnings.append(
            f"Merge {display_label}: inputs differ in resolution; "
            "resampling to the first input's grid (nearest)"
        )

    prov_doc = provenance.build_provenance(task_ctx, session_file)
    out_path = scoped_file(task_ctx, session_file)
    _, _, first_transform, first_crs, _ = _raster_shape(files[0])
    crs_in = str(first_crs) if first_crs else None
    crs_out = clean_crs(task_ctx.get("crs")) or crs_in
    bounds = [_valid_box_streaming(str(path), nodata) for path in files]
    if all(bound is None for bound in bounds):
        width, height, first_transform, _, _ = _raster_shape(files[0])
        grid = np.full((height, width), nodata, dtype=np.float32)
        write_geotiff(out_path, grid, first_transform, crs=crs_out,
                      nodata=nodata, metadata=prov_doc)
        return {
            "file": out_path,
            "kind": kind_label,
            "node": module_id,
            "display_name": display_label,
            "warnings": [
                f"Merge {display_label}: every input is all-NODATA; "
                "merged written as all-NODATA"
            ],
        }

    finish_mosaic(files, out_path, crs=crs_out, nodata=nodata,
                  metadata=prov_doc, method=method,
                  memory_budget_bytes=task_ctx.get(
                      "_raster_grid_memory_budget_bytes"),
                  notes=task_ctx.setdefault("_bounded_notes", []))
    payload = {
        "file": out_path,
        "kind": kind_label,
        "node": module_id,
        "display_name": display_label,
    }
    if warnings:
        payload["warnings"] = warnings
    return payload
