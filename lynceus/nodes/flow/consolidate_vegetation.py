# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Variant grid vegetation of the Consolidate strategy family (no NODE_ID)."""

from __future__ import annotations

from lynceus.nodes.ports import PortDef, PortType

VARIANT_OF = "lynceus.nodes.flow.consolidate"
VARIANT_KEY = "vegetation"
VARIANT_LABEL = "Grid Vegetation"

INPUTS = (PortDef(PortType.GRID_VEGETATION),)
OUTPUTS = (PortType.GRID_VEGETATION,)

PROCESSING_SPECS = {
    "consolidator": True,
    "consolidate_task": "barrier_consolidate_vegetation",
    "output_files": {"grid_vegetation": "grid_vegetation.gpkg"},
    "output_globs": ("*/grid_vegetation.gpkg", "*/grid_vegetation.csv"),
    "qml": {"grid_vegetation.gpkg": "cover"},
}


def barrier_consolidate_vegetation(ctx: dict) -> dict:
    """Merge per-segment vegetation tables exactly by cell key."""
    from lynceus.nodes.flow._consolidate_base import (
        _consolidate_payload,
        _segment_files,
        _segment_node_meta,
        _consolidate_doc,
        _out_path,
        _sidecar_meta,
    )
    from lynceus.nodes.lidar.analysis.grid_vegetation import (
        finalize_vegetation_frame,
        merge_vegetation_frames,
    )

    import geopandas as gpd

    files = _segment_files(ctx, "grid_vegetation")
    if not files:
        raise RuntimeError(
            "Consolidation has no segment tables for grid vegetation"
        )
    frames = []
    for path in files:
        frame = gpd.read_file(path)
        if frame is not None and len(frame) > 0:
            frames.append(frame)
    if not frames:
        raise RuntimeError("Segment tables are empty; nothing to consolidate")
    nodata = float(ctx.get("nodata_value", -9999.0))
    res = float(ctx.get("resolution", 1.0))
    threshold_m = float(ctx.get("canopy_threshold_m", 2.0))
    lai_k = float(ctx.get("lai_k", 0.5))
    if files:
        # Producer parameters travel in the segment sidecar: honor them
        # instead of re-asking on the consolidator instance.
        meta = _sidecar_meta(files[0])
        if isinstance(meta.get("nodata"), (int, float)):
            nodata = float(meta["nodata"])
        if isinstance(meta.get("grid_size_m"), (int, float)):
            res = float(meta["grid_size_m"])
        if isinstance(meta.get("canopy_threshold_m"), (int, float)):
            threshold_m = float(meta["canopy_threshold_m"])
        if isinstance(meta.get("lai_k"), (int, float)):
            lai_k = float(meta["lai_k"])
    gids, cols, rows, hists, n = merge_vegetation_frames(frames)
    finals = finalize_vegetation_frame(
        gids, cols, rows, hists, n, res, threshold_m, lai_k, nodata
    )
    from lynceus.nodes.lidar.analysis.grid_vegetation import _veg_polygons

    gdf = gpd.GeoDataFrame(
        finals, geometry=_veg_polygons(cols, rows, res), crs=frames[0].crs
    )
    out = _out_path(ctx, "grid_vegetation.gpkg")
    gdf.to_file(str(out), driver="GPKG")
    csv_path = str(out.with_suffix(".csv"))
    gdf.drop(columns=["geometry"]).to_csv(csv_path, index=False)
    from lynceus.processing import provenance

    prov_doc = _consolidate_doc(ctx, "grid_vegetation.gpkg")
    provenance.embed_gpkg_metadata(str(out), prov_doc)
    provenance.write_sidecar(
        str(out), prov_doc, _segment_node_meta(ctx, "grid_vegetation")
    )
    payload = _consolidate_payload(out, ctx)
    payload["display_name"] = "Grid Vegetation"
    payload["csv"] = csv_path
    return payload
