# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Node -- consolidate processed point-cloud tiles into one LAZ product."""

from __future__ import annotations

from pathlib import Path

from lynceus.nodes.ports import PortType

NODE_ID = "lynceus.nodes.lidar.output.export_point_cloud"
NODE_NAME = "Export Point Cloud"
NODE_CATEGORY = "LiDAR"

NODE_SUBCATEGORY = "Export"

NODE_DESCRIPTION = (
    "<b>Export Point Cloud</b> -- Consolidated LAZ<br><br>"
    "Merges the processed tiles into one point-cloud file. The default export "
    "removes tile buffers by keeping only each tile's core area, preventing "
    "duplicate points along tile boundaries.<br><br>"
    "The export can consume the raw, classified, or cleaned stream connected "
    "to its input."
)

INPUTS = (PortType.POINT_CLOUD,)
# Terminal sink: the engine feeds barrier sources only from tile_task /
# triggers_tiling providers, so an output port here would invite dead-end
# connections (downstream would silently re-read upstream tiles).
OUTPUTS = ()

PROCESSING_SPECS = {
    "barrier_task": "barrier_export_point_cloud",
    "input_port": "point_cloud",
    "output_port": "point_cloud",
    "point_cloud_export": True,
    "output_globs": ("point_cloud_exports/*/point_cloud_consolidated.laz",),
    "config_schema": {
        "stage_name": {
            "type": "str",
            "default": "",
            "description": "Optional label for this export stage in the products panel.",
            "impact": "Use labels such as Raw, Classified, or Cleaned when several export nodes are present.",
            "group": "Output",
        },
        "crop_buffer": {
            "type": "bool",
            "default": True,
            "description": "Keep only each tile core and remove overlapping buffer points.",
            "impact": "Prevents duplicate points at tile boundaries. Disable only when the overlap is required for a downstream workflow.",
            "group": "Output",
        },
    },
}


def get_config_defaults() -> dict:
    return {k: v["default"] for k, v in PROCESSING_SPECS["config_schema"].items()}


def _header_copy(header):
    import laspy

    from lynceus.processing import provenance

    result = laspy.LasHeader(version=header.version, point_format=header.point_format)
    result.offsets = header.offsets.copy()
    result.scales = header.scales.copy()
    provenance.copy_product_vlrs(result, header)
    provenance.copy_header_identity(result, header)
    return result


def _output_record(chunk, reader_header, output_header, mask):
    import laspy
    import numpy as np

    # Boolean indexing already copies: one masked slice feeds both records.
    # Non-XYZ dims travel as raw integers (identical encoding); X/Y/Z are
    # renormalized through physical coordinates (per-header scale/offset).
    if mask is not None:
        sub = np.asanyarray(chunk.array)[mask]
    else:
        sub = np.asanyarray(chunk.array)
    result = laspy.ScaleAwarePointRecord(
        sub,
        output_header.point_format,
        scales=output_header.scales,
        offsets=output_header.offsets,
    )
    source_view = laspy.ScaleAwarePointRecord(
        sub,
        reader_header.point_format,
        scales=reader_header.scales,
        offsets=reader_header.offsets,
    )
    result.x = np.asarray(source_view.x)
    result.y = np.asarray(source_view.y)
    result.z = np.asarray(source_view.z)
    return result


def barrier_export_point_cloud(ctx: dict) -> dict:
    """Write one LAZ from the connected stream, cropping tile buffers."""
    import laspy
    import numpy as np

    from lynceus.processing import provenance
    from lynceus.processing.tiler import _laz_backend

    node_iid = str(ctx.get("node_iid") or "export")
    output_path = (
        Path(ctx["session_dir"])
        / "point_cloud_exports"
        / node_iid
        / "point_cloud_consolidated.laz"
    )
    crop_buffer = bool(ctx.get("crop_buffer", True))
    display_name = str(ctx.get("stage_name") or "")
    if output_path.exists():
        # Header-only count: the reuse payload keeps its full shape.
        with laspy.open(str(output_path)) as existing:
            count = int(existing.header.point_count)
        return {
            "file": str(output_path),
            "kind": "Point Cloud",
            "node": NODE_ID,
            "count": count,
            "crop_buffer": crop_buffer,
            "display_name": display_name,
        }

    prov_doc = provenance.build_provenance(ctx, "point_cloud_consolidated.laz")
    backend = _laz_backend()

    sources = ctx.get("_point_cloud_sources", [])
    if not sources:
        raise RuntimeError("No point-cloud tiles connected to export")

    ext = ctx.get("intermediate_ext", "laz")
    output_header = None
    writer = None
    count = 0
    warnings: list[str] = []
    crs_seen: list[str] = []
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        for source in sources:
            provider_dir = source.get("point_cloud_dir") or ""
            for tile in source.get("tiles", []):
                tile_path = (
                    Path(ctx["session_dir"]) / provider_dir / f"{tile['tile_id']}.{ext}"
                    if provider_dir
                    else Path(tile["file"])
                )
                if not tile_path.is_file():
                    raise RuntimeError(f"Missing point-cloud tile: {tile_path}")
                with laspy.open(str(tile_path), laz_backend=backend) as reader:
                    if output_header is None:
                        output_header = _header_copy(reader.header)
                        provenance.fill_las_header(output_header, prov_doc)
                        writer = laspy.open(
                            str(output_path),
                            mode="w",
                            header=output_header,
                            laz_backend=backend,
                        )
                    else:
                        source_names = set(reader.header.point_format.dimension_names)
                        output_names = set(output_header.point_format.dimension_names)
                        if source_names != output_names:
                            raise RuntimeError(
                                f"{tile['tile_id']}: dimension mismatch "
                                f"(extra={sorted(source_names - output_names)}, "
                                f"missing={sorted(output_names - source_names)})"
                            )
                    crs = reader.header.parse_crs()
                    crs_key = str(crs) if crs else ""
                    if crs_key not in crs_seen:
                        crs_seen.append(crs_key)
                    for chunk in reader.chunk_iterator(2_000_000):
                        x = np.asarray(chunk.x)
                        y = np.asarray(chunk.y)
                        if crop_buffer:
                            mask = (
                                (x >= tile["core_x_min"])
                                & (x < tile["core_x_max"])
                                & (y >= tile["core_y_min"])
                                & (y < tile["core_y_max"])
                            )
                            n = int(mask.sum())
                            if n == 0:
                                continue
                            chunk_mask: np.ndarray | None = mask
                        else:
                            n = int(x.size)
                            chunk_mask = None
                        record = _output_record(
                            chunk, reader.header, output_header, chunk_mask
                        )
                        writer.write_points(record)
                        count += n
    finally:
        if writer is not None:
            writer.close()

    if output_header is None:
        raise RuntimeError("No point-cloud data found in connected tiles")
    payload = {
        "file": str(output_path),
        "kind": "Point Cloud",
        "node": NODE_ID,
        "count": count,
        "crop_buffer": crop_buffer,
        "display_name": display_name,
    }
    if count == 0:
        warnings.append(
            "All points cropped by tile core bounds; export is empty."
        )
    if len(crs_seen) > 1:
        warnings.append(
            "Tiles carry different coordinate systems; merged as-is."
        )
    if warnings:
        payload["warnings"] = warnings
    return payload