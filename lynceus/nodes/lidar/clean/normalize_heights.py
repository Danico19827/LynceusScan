# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Node -- Normalize Heights (height above ground).

Replaces Z with height above the wired DTM (bilinear sampling, exact on
slopes): vegetation and buildings read as height over ground. The
classified stream is deliberately NOT re-emitted (a DTM over normalized
heights would surface heights as elevation): downstream terrain products
must use the pre-normalize stream.
"""

from __future__ import annotations

from lynceus.nodes.ports import PortType

NODE_ID = "lynceus.nodes.lidar.clean.normalize_heights"
NODE_NAME = "Normalize Heights"
NODE_CATEGORY = "LiDAR"

NODE_SUBCATEGORY = "Cleaning"

NODE_DESCRIPTION = (
    "<b>Normalize Heights</b> -- Height Above Ground<br><br>"
    "Replaces Z with height above the wired DTM (bilinear sampling, "
    "exact on slopes): vegetation and buildings read as height over "
    "ground. Points outside DTM coverage or flagged withheld keep their "
    "original Z and are counted in the payload.<br><br>"
    "<b>Process:</b> Requires a wired DTM. The classified stream is NOT "
    "re-emitted: downstream terrain products must use the pre-normalize "
    "stream, or they will surface heights as elevation.<br><br>"
    "<b>Tips:</b> Normalize right before vegetation analysis; keep the "
    "classified stream for terrain."
)

INPUTS = (PortType.POINT_CLOUD, PortType.DTM_MOSAIC)
OUTPUTS = (PortType.POINT_CLOUD,)

PROCESSING_SPECS = {
    "tile_task": "tile_normalize_heights",
    "input_port": "point_cloud",
    "output_port": "point_cloud",
    "point_cloud_dir": "normalized",
    "output_globs": ("normalized/**/*.laz",),
    "config_schema": {},
}


def get_config_defaults() -> dict:
    return {}


def tile_normalize_heights(tile: dict, ctx: dict) -> dict:
    """Replace Z with height above the wired DTM (bilinear, exact)."""
    from pathlib import Path

    import laspy
    import numpy as np

    from lynceus.nodes._point_source import (
        _copy_with_sidecar,
        bilinear_dtm_reference,
        clean_out_path,
        point_src,
        quality_mask,
        read_cloud,
        write_clean_laz,
    )
    from lynceus.processing import provenance

    out_path = clean_out_path(tile, ctx, "normalized")
    if Path(out_path).exists():
        return {"tile_id": tile["tile_id"], "node": NODE_ID,
                "output": out_path}

    dtm_path = ctx.get("dtm_mosaic_path", "") or ""
    if not dtm_path or not Path(dtm_path).is_file():
        raise RuntimeError(
            f"{NODE_ID}: DTM product missing at {dtm_path!r}; wire a DTM "
            "mosaic to normalize against"
        )

    prov_doc = provenance.build_provenance(ctx, Path(out_path).name)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    src = point_src(tile, ctx)
    cloud, header, scales, offsets = read_cloud(src)
    if cloud.shape[0] == 0:
        _copy_with_sidecar(src, out_path, prov_doc)
        return {
            "tile_id": tile["tile_id"], "node": NODE_ID, "output": out_path,
            "warnings": [f"{tile['tile_id']}: no points in source; empty tile written"],
        }
    record = laspy.ScaleAwarePointRecord(
        cloud, header.point_format, scales=scales, offsets=offsets
    )
    x = np.asarray(record.x, dtype=np.float64)
    y = np.asarray(record.y, dtype=np.float64)
    z = np.asarray(record.z, dtype=np.float64)

    vendor = quality_mask(record)
    processable = np.ones(z.shape, dtype=bool) if vendor is None else vendor

    ref = bilinear_dtm_reference(x, y, str(dtm_path), NODE_ID)
    has_datum = np.isfinite(ref)
    do = processable & has_datum
    z_new = z.copy()
    z_new[do] = z[do] - ref[do]
    record.z = z_new

    normalized = int(do.sum())
    kept_original = int((~do).sum())
    withheld = int((~processable).sum())
    result = {
        "tile_id": tile["tile_id"], "node": NODE_ID, "output": out_path,
        "normalized": normalized, "kept_original": kept_original,
        "withheld": withheld,
    }
    if kept_original - withheld > 0:
        result["warnings"] = [
            f"{tile['tile_id']}: {kept_original - withheld} points outside "
            "DTM coverage kept with original Z"
        ]
    write_clean_laz(out_path, cloud, header, scales, offsets, metadata=prov_doc)
    return result
