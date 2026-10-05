# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Shared ground-classification tile runner (no NODE_ID: ignored by discovery).

Every Classify Ground strategy variant (PMF/CSF/SMRF) funnels its tile
through ``run_ground_classification`` with its own ground-mask function.
The two-pass streaming I/O (single decode, compact retention, incremental
write, empty-tile propagation) lives here once; methods differ only in
how the mask is computed.
"""

from __future__ import annotations

from pathlib import Path


def run_ground_classification(
    tile: dict, ctx: dict, mask_fn, node_id: str = ""
) -> dict:
    """Classify one tile with ``mask_fn(x, y, z) -> bool ground array``."""
    import laspy
    import numpy as np

    from lynceus.nodes._point_source import point_src, write_clean_laz
    from lynceus.processing import provenance
    from lynceus.processing.tiler import _laz_backend

    out_path = _classified_path(tile, ctx)
    out = Path(out_path)
    if out.exists():
        return {"tile_id": tile["tile_id"], "node": node_id, "output": out_path}

    prov_doc = provenance.build_provenance(ctx, out.name)
    out.parent.mkdir(parents=True, exist_ok=True)

    src = point_src(tile, ctx)
    backend = _laz_backend()
    output_class = int(ctx.get("output_class", 2))

    # Pass 1: single decode retaining compact x/y/z only (~24 B/pt
    # instead of full records). The ground mask needs every point before
    # any chunk can be labeled; original classes travel with the second
    # streaming pass below, so they are never retained. Vendor-withheld
    # points are excluded from the surface estimation (and never labeled
    # as ground) but keep their offsets so both passes stay aligned.
    xs: list = []
    ys: list = []
    zs: list = []
    oks: list = []
    with laspy.open(src, laz_backend=backend) as reader:
        header = reader.header
        if "classification" not in header.point_format.dimension_names:
            raise RuntimeError(
                f"{tile['tile_id']}: source has no classification dimension"
            )
        has_withheld = "withheld" in header.point_format.dimension_names
        for chunk in reader.chunk_iterator(2_000_000):
            xs.append(np.asarray(chunk.x))
            ys.append(np.asarray(chunk.y))
            zs.append(np.asarray(chunk.z))
            if has_withheld:
                oks.append(~np.asarray(chunk.withheld, dtype=bool))
        scales = header.scales.copy()
        offsets = header.offsets.copy()
        point_format = header.point_format
        version = header.version

    total = sum(a.size for a in xs)
    if total == 0:
        empty = laspy.ScaleAwarePointRecord.zeros(
            0, point_format=point_format, scales=scales, offsets=offsets
        ).array
        write_clean_laz(out_path, empty, header, scales, offsets, metadata=prov_doc)
        return {
            "tile_id": tile["tile_id"],
            "node": node_id,
            "output": out_path,
            "warnings": [
                f"{tile['tile_id']}: no points in source tile; empty classified tile written"
            ],
        }

    # float64 (native scaled coords): minimum.at-style reductions run on
    # the fast path; mixing dtypes activates NumPy's slow path.
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    z = np.concatenate(zs)
    del xs, ys, zs

    if oks:
        ok = np.concatenate(oks)
        del oks
        ground = np.zeros(total, dtype=bool)
        if ok.any():
            ground[ok] = mask_fn(x[ok], y[ok], z[ok])
        else:
            # Everything withheld: nothing to classify; the cloud travels
            # through unchanged (never labeled as ground).
            import shutil

            shutil.copy2(src, out_path)
            provenance.write_sidecar(out_path, prov_doc, {"copied_from": str(src)})
            del x, y, z
            return {
                "tile_id": tile["tile_id"],
                "node": node_id,
                "output": out_path,
                "warnings": [
                    f"{tile['tile_id']}: all points withheld; "
                    "nothing classified"
                ],
            }
    else:
        ground = mask_fn(x, y, z)
    del x, y, z

    new_header = laspy.LasHeader(version=version, point_format=point_format)
    new_header.offsets = offsets
    new_header.scales = scales
    provenance.copy_product_vlrs(new_header, header)
    provenance.copy_header_identity(new_header, header)
    provenance.fill_las_header(new_header, prov_doc)

    # Pass 2: same chunking as pass 1 (deterministic iterator) labels each
    # chunk from its mask slice and writes incrementally — no full output
    # cloud is ever retained.
    offset = 0
    with laspy.open(src, laz_backend=backend) as reader:
        with laspy.open(
            str(out), mode="w", header=new_header, laz_backend=backend
        ) as writer:
            for chunk in reader.chunk_iterator(2_000_000):
                n = len(chunk.array)
                gm = ground[offset : offset + n]
                assert n == gm.size, "chunking diverged between classify passes"
                offset += n
                record = laspy.ScaleAwarePointRecord(
                    np.asanyarray(chunk.array),
                    point_format,
                    scales=scales,
                    offsets=offsets,
                )
                record.classification[gm] = output_class
                writer.write_points(record)
    assert offset == ground.size, "classify passes covered different points"

    return {"tile_id": tile["tile_id"], "node": node_id, "output": out_path}


def _classified_path(tile: dict, ctx: dict) -> str:
    """Return the output path for a classified tile.

    Uses the run's intermediate extension (``ctx["intermediate_ext"]``,
    default ``laz``)."""
    from lynceus.nodes._paths import scoped_dir

    ext = ctx.get("intermediate_ext", "laz")
    return str(scoped_dir(ctx, "classified") / f"{tile['tile_id']}.{ext}")
