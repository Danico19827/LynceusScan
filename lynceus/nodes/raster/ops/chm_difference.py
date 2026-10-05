# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Node -- CHM Difference (A - B) between two CHM mosaics.

Barrier-only combination node (F2): imports TWO CHM mosaics (typically two
Input Raster nodes set to the CHM strategy, or two input flights) and writes
the per-cell difference A - B as a generic raster. Both inputs are the same
port type, so the two paths arrive positionally as `ctx["in0_path"]` and
`ctx["in1_path"]`.
"""

from __future__ import annotations

from lynceus.nodes._point_source import clean_crs
from lynceus.nodes.ports import PortDef, PortType

NODE_ID = "lynceus.nodes.raster.ops.chm_difference"
NODE_NAME = "CHM Difference"
NODE_CATEGORY = "Raster"
NODE_SUBCATEGORY = "Operations"

NODE_DESCRIPTION = (
    "<b>CHM Difference</b> -- Vegetation Height Change (A - B)<br><br>"
    "Subtracts two CHM mosaics cell by cell to highlight vegetation height "
    "change between two surveys or treatments: <code>output = CHM A - CHM B</code>.<br><br>"
    "<b>Process:</b> Both rasters are aligned to the grid of CHM A; cells "
    "where either input is NODATA stay NODATA. The result is a signed raster: "
    "positive values mean taller canopy in A, negative values mean taller "
    "canopy in B.<br><br>"
    "<b>Tips:</b> Feed this node with two <code>Input Raster</code> nodes set "
    "to the CHM strategy to compare a pre-made CHM against a generated one, "
    "or two LiDAR flights (before/after)."
)

INPUTS = (
    PortDef(PortType.CHM_MOSAIC, name="CHM A"),
    PortDef(PortType.CHM_MOSAIC, name="CHM B"),
)
OUTPUTS = (PortType.RASTER,)

PROCESSING_SPECS = {
    "barrier_task": "barrier_chm_difference",
    "input_ports": ("chm_mosaic", "chm_mosaic"),
    "output_files": {"raster": "chm_difference.tif"},
    "output_port": "raster",
    "output_globs": ("chm_difference.tif",),
    "qml": {"chm_difference.tif": "chm"},
    "config_schema": {
        "nodata_value": {
            "type": "float",
            "default": -9999.0,
            "minimum": -99999.0,
            "maximum": 99999.0,
            "description": "Value written to cells with no data in the GeoTIFF.",
            "impact": "Should match the convention of your GIS software. Almost always -9999.0.",
            "group": "Advanced",
            "advanced": True,
        },
    },
}


def get_config_defaults() -> dict:
    return {"nodata_value": -9999.0}


def barrier_chm_difference(ctx: dict) -> dict:
    """Compute CHM(A) - CHM(B) aligned to CHM A's grid; NODATA elsewhere."""
    from pathlib import Path

    import numpy as np

    from lynceus.processing import provenance
    from lynceus.processing.raster import (
        _valid_bounds,
        align_to_grid,
        read_geotiff,
        write_geotiff,
    )

    prov_doc = provenance.build_provenance(ctx, "chm_difference.tif")
    from lynceus.nodes._paths import scoped_file

    session = Path(ctx["session_dir"])
    a_path = Path(ctx.get("in0_path", str(session / "chm_mosaic.tif")))
    b_path = Path(ctx.get("in1_path", str(session / "chm_mosaic.tif")))
    if not a_path.is_file() or not b_path.is_file():
        raise RuntimeError(
            "CHM Difference requires two CHM mosaics (CHM A and CHM B)"
        )
    nodata = ctx.get("nodata_value", -9999.0)
    out_path = scoped_file(ctx, "chm_difference.tif")

    a_arr, a_transform, crs_a = read_geotiff(a_path)
    b_arr, b_transform, _crs_b = read_geotiff(b_path)

    valid_a = _valid_bounds(a_arr, a_transform, nodata)
    valid_b = _valid_bounds(b_arr, b_transform, nodata)
    crs_out = clean_crs(ctx.get("crs")) or crs_a
    if valid_a is None or valid_b is None:
        grid = np.full(a_arr.shape, nodata, dtype=np.float32)
        write_geotiff(
            out_path, grid, a_transform, crs=crs_out, nodata=nodata,
            metadata=prov_doc,
        )
        return {
            "file": out_path,
            "kind": "CHM Difference",
            "node": NODE_ID,
            "warnings": [
                "One of the CHM inputs has no valid data; "
                "difference written as all-NODATA"
            ],
        }

    b_aligned = align_to_grid(b_arr, b_transform, a_arr.shape, a_transform, nodata)
    both_valid = (a_arr != nodata) & (b_aligned != nodata)
    diff = np.full(a_arr.shape, nodata, dtype=np.float32)
    diff[both_valid] = a_arr[both_valid] - b_aligned[both_valid]

    write_geotiff(
        out_path, diff, a_transform, crs=crs_out, nodata=nodata, metadata=prov_doc
    )
    return {"file": out_path, "kind": "CHM Difference", "node": NODE_ID}