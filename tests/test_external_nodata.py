# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Foreign NODATA tags: voids of external rasters must stay voids.

Regression (verified on a real download): an external DTM tagged
``-3.4e38`` (not the house -9999) had its void cells treated as valid
ground — Horn computed ~90° garbage over them and the products showed
data outside the scanned zone. Barriers must honor the file's own tag
(plus NaN, always) instead of the bare ``nodata_value`` default.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

FOREIGN_TAG = np.float32(-3.4028235e38)
HOUSE_NODATA = -9999.0
SHAPE = (24, 24)
VOID = (slice(8, 16), slice(8, 16))


def _plane(shape=SHAPE) -> np.ndarray:
    rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float64)
    return (2000.0 + 0.5 * rows + 0.3 * cols).astype(np.float32)


def _write_raster(path: Path, array: np.ndarray, nodata) -> Path:
    import rasterio
    from affine import Affine

    with rasterio.open(
        str(path), "w", driver="GTiff", height=array.shape[0],
        width=array.shape[1], count=1, dtype="float32",
        crs="EPSG:32612", transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
        nodata=nodata,
    ) as ds:
        ds.write(array, 1)
    return path


def _foreign_dtm(path: Path, nan_voids: bool = False) -> Path:
    arr = _plane()
    if nan_voids:
        arr = arr.copy()
        arr[VOID] = np.nan
        return _write_raster(path, arr, None)
    arr = arr.copy()
    arr[VOID] = FOREIGN_TAG
    return _write_raster(path, arr, float(FOREIGN_TAG))


def _ctx(session: Path, extra: dict | None = None) -> dict:
    ctx = {"session_dir": str(session), "node_iid": "t1"}
    if extra:
        ctx.update(extra)
    return ctx


def _eroded_valid_mask(valid: np.ndarray) -> np.ndarray:
    """Exact Horn validity: full 3x3 window valid, rim always NODATA."""
    from scipy.ndimage import minimum_filter

    window_ok = minimum_filter(valid, size=3, mode="constant", cval=False)
    expected = np.zeros_like(valid, dtype=bool)
    expected[1:-1, 1:-1] = window_ok[1:-1, 1:-1]
    return expected


def _read_result(path: str) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as src:
        return src.read(1)


class LoadDemMaskedTests(unittest.TestCase):
    def test_foreign_tag_remapped_to_house_nodata(self) -> None:
        from lynceus.processing.raster import load_dem_masked

        with tempfile.TemporaryDirectory() as tmp:
            path = _foreign_dtm(Path(tmp) / "dtm.tif")
            work, _transform, _crs, nodata = load_dem_masked(
                path, HOUSE_NODATA)
            self.assertEqual(nodata, HOUSE_NODATA)
            self.assertEqual(work.dtype, np.float64)
            valid = np.ones(SHAPE, dtype=bool)
            valid[VOID] = False
            self.assertTrue(
                np.array_equal(work != HOUSE_NODATA, valid))
            # Void cells carry no trace of the foreign tag.
            self.assertFalse(np.isinf(work).any())

    def test_nan_voids_masked_without_tag(self) -> None:
        from lynceus.processing.raster import load_dem_masked

        with tempfile.TemporaryDirectory() as tmp:
            path = _foreign_dtm(Path(tmp) / "dtm.tif", nan_voids=True)
            work, _transform, _crs, _nodata = load_dem_masked(
                path, HOUSE_NODATA)
            valid = np.ones(SHAPE, dtype=bool)
            valid[VOID] = False
            self.assertTrue(
                np.array_equal(work != HOUSE_NODATA, valid))


class DerivativeForeignTagTests(unittest.TestCase):
    def test_slope_mask_is_exact_and_values_sane(self) -> None:
        from lynceus.nodes.raster.ops.terrain_derivative_slope import (
            barrier_terrain_slope,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dtm = _foreign_dtm(root / "ext.tif")
            out = barrier_terrain_slope(_ctx(
                root / "session",
                {"dtm_mosaic_path": str(dtm),
                 "nodata_value": HOUSE_NODATA},
            ))
            grid = _read_result(out["file"])
            valid_in = np.ones(SHAPE, dtype=bool)
            valid_in[VOID] = False
            expected = _eroded_valid_mask(valid_in)
            self.assertTrue(
                np.array_equal(grid != HOUSE_NODATA, expected),
                "voids (and only their 1px fringe) must stay NODATA",
            )
            # Tilted plane: exact Horn slope, no 90-degree garbage.
            # (1e-2 tolerance: the fixture itself is float32 at z~2000.)
            want = np.degrees(np.arctan(np.hypot(0.3, 0.5)))
            self.assertTrue(
                np.allclose(grid[expected], want, atol=1e-2))
            self.assertLess(float(grid[expected].max()), 45.0)

    def test_aspect_and_hillshade_masks_match(self) -> None:
        from lynceus.nodes.raster.ops.terrain_derivative_aspect import (
            barrier_terrain_aspect,
        )
        from lynceus.nodes.raster.ops.terrain_derivative_hillshade import (
            barrier_terrain_hillshade,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dtm = _foreign_dtm(root / "ext.tif")
            valid_in = np.ones(SHAPE, dtype=bool)
            valid_in[VOID] = False
            expected = _eroded_valid_mask(valid_in)
            for barrier in (barrier_terrain_aspect,
                            barrier_terrain_hillshade):
                out = barrier(_ctx(
                    root / "session",
                    {"dtm_mosaic_path": str(dtm),
                     "nodata_value": HOUSE_NODATA},
                ))
                grid = _read_result(out["file"])
                self.assertTrue(
                    np.array_equal(grid != HOUSE_NODATA, expected))

    def test_nan_voids_stay_masked(self) -> None:
        from lynceus.nodes.raster.ops.terrain_derivative_slope import (
            barrier_terrain_slope,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dtm = _foreign_dtm(root / "ext.tif", nan_voids=True)
            out = barrier_terrain_slope(_ctx(
                root / "session",
                {"dtm_mosaic_path": str(dtm),
                 "nodata_value": HOUSE_NODATA},
            ))
            grid = _read_result(out["file"])
            valid_in = np.ones(SHAPE, dtype=bool)
            valid_in[VOID] = False
            self.assertTrue(
                np.array_equal(grid != HOUSE_NODATA,
                               _eroded_valid_mask(valid_in)))
            self.assertFalse(np.isnan(grid).any())


class ChmDifferenceForeignTagTests(unittest.TestCase):
    def test_voids_stay_nodata_with_foreign_tags(self) -> None:
        from lynceus.nodes.raster.ops.chm_difference import (
            barrier_chm_difference,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arr = np.full(SHAPE, 10.0, dtype=np.float32)
            arr[VOID] = FOREIGN_TAG
            a_path = _write_raster(root / "a.tif", arr, float(FOREIGN_TAG))
            arr_b = np.full(SHAPE, 4.0, dtype=np.float32)
            arr_b[VOID] = FOREIGN_TAG
            b_path = _write_raster(root / "b.tif", arr_b, float(FOREIGN_TAG))
            out = barrier_chm_difference(_ctx(
                root / "session",
                {"in0_path": str(a_path), "in1_path": str(b_path),
                 "nodata_value": HOUSE_NODATA},
            ))
            grid = _read_result(out["file"])
            valid = np.ones(SHAPE, dtype=bool)
            valid[VOID] = False
            self.assertTrue(np.array_equal(grid != HOUSE_NODATA, valid))
            self.assertTrue(np.allclose(grid[valid], 6.0))


class StatsNaNTests(unittest.TestCase):
    def test_stats_exclude_nan_beside_tag(self) -> None:
        from lynceus.processing.qml_style import _read_raster_stats

        with tempfile.TemporaryDirectory() as tmp:
            arr = np.full((8, 8), 5.0, dtype=np.float32)
            arr[0, 0] = np.nan
            arr[1, 1] = HOUSE_NODATA
            path = _write_raster(Path(tmp) / "r.tif", arr, HOUSE_NODATA)
            stats = _read_raster_stats(path)
            self.assertIsNotNone(stats)
            self.assertEqual((stats[0], stats[1]), (5.0, 5.0))


class ViewerMaskTests(unittest.TestCase):
    def test_nan_always_masked(self) -> None:
        from lynceus.ui.viewers.raster_2d import _invalid_mask

        arr = np.array([[1.0, np.nan], [HOUSE_NODATA, 2.0]])
        self.assertTrue(
            np.array_equal(
                _invalid_mask(arr, HOUSE_NODATA),
                [[False, True], [True, False]],
            ))


if __name__ == "__main__":
    unittest.main()
