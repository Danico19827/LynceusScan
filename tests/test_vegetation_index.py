# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Tests for the Vegetation Index node + multiband input variant (U1).

Pixel-exact reference against brute-force numpy on a synthetic 5-band
GeoTIFF (common Blue/Green/Red/RedEdge/NIR layout). Stdlib + engine
modules only (no Qt, no real runs).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from lynceus.nodes.raster.ops.vegetation_index import barrier_vegetation_index
from lynceus.plugins.registry import manager
from lynceus.processing.raster import read_geotiff, read_geotiff_bands

NODATA = -9999.0


def _write_multispectral(path: Path) -> None:
    import rasterio

    red = np.array([[100.0, 200.0], [0.0, NODATA]], dtype=np.float32)
    nir = np.array([[300.0, 200.0], [0.0, 400.0]], dtype=np.float32)
    green = np.array([[50.0, 100.0], [25.0, 75.0]], dtype=np.float32)
    rededge = np.array([[150.0, 250.0], [50.0, 350.0]], dtype=np.float32)
    blue = np.zeros((2, 2), dtype=np.float32)
    cube = np.stack([blue, green, red, rededge, nir])
    profile = {
        "driver": "GTiff",
        "height": 2,
        "width": 2,
        "count": 5,
        "dtype": "float32",
        "crs": "EPSG:32720",
        "transform": rasterio.Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
        "nodata": NODATA,
    }
    with rasterio.open(str(path), "w", **profile) as ds:
        ds.write(cube)


def _ctx(session: Path, **overrides):
    cfg = {
        "index": "ndvi",
        "red_band": 3,
        "nir_band": 5,
        "rededge_band": 4,
        "green_band": 2,
        "soil_brightness_L": 0.5,
        "nodata_value": NODATA,
    }
    cfg.update(overrides)
    return {"session_dir": str(session), **cfg}


class ReadBandsTests(unittest.TestCase):
    def test_reads_selected_bands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "ms.tif"
            _write_multispectral(src)
            cube, _transform, crs = read_geotiff_bands(src, [5, 3])
            self.assertEqual(cube.shape, (2, 2, 2))
            self.assertEqual(crs, "EPSG:32720")
            self.assertAlmostEqual(float(cube[0, 0, 0]), 300.0)
            self.assertAlmostEqual(float(cube[1, 0, 1]), 200.0)

    def test_out_of_range_band_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "ms.tif"
            _write_multispectral(src)
            with self.assertRaises(RuntimeError):
                read_geotiff_bands(src, [6])

    def test_non_positive_band_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "ms.tif"
            _write_multispectral(src)
            with self.assertRaises(RuntimeError):
                read_geotiff_bands(src, [0])


class VegetationIndexTests(unittest.TestCase):
    def _run(self, session: Path, **overrides):
        src = session / "multispectral.tif"
        _write_multispectral(src)
        return barrier_vegetation_index(
            {**_ctx(session, **overrides), "raster_path": str(src)}
        )

    def test_ndvi_matches_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp)
            out = self._run(session)
            arr, _transform, crs = read_geotiff(out["file"])
            self.assertEqual(crs, "EPSG:32720")
            # (300-100)/(300+100)=0.5; (200-200)/400=0.0;
            # 0/0 -> NODATA; input NODATA -> NODATA.
            expected = np.array(
                [[0.5, 0.0], [NODATA, NODATA]], dtype=np.float32
            )
            np.testing.assert_allclose(arr, expected, rtol=1e-5)

    def test_unknown_session_crs_falls_back_to_raster_crs(self) -> None:
        from lynceus.nodes._point_source import UNKNOWN_CRS

        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp)
            out = self._run(session, crs=UNKNOWN_CRS)
            _, _, crs = read_geotiff(out["file"])
            self.assertEqual(crs, "EPSG:32720")

    def test_ndre_uses_rededge_band(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp)
            out = self._run(session, index="ndre")
            arr, _, _ = read_geotiff(out["file"])
            # (300-150)/(300+150)=1/3.
            self.assertAlmostEqual(float(arr[0, 0]), 1 / 3, places=5)

    def test_savi_applies_soil_factor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp)
            out = self._run(session, index="savi")
            arr, _, _ = read_geotiff(out["file"])
            # ((300-100)/(300+100+0.5))*1.5.
            self.assertAlmostEqual(
                float(arr[0, 0]), (200 / 400.5) * 1.5, places=5
            )

    def test_vari_rgb_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp)
            out = self._run(session, index="vari")
            arr, _, _ = read_geotiff(out["file"])
            # (50-100)/(50+100-0)=-1/3; (100-200)/(100+200-0)=-1/3;
            # (25-0)/(25+0-0)=1.0; input NODATA -> NODATA.
            self.assertAlmostEqual(float(arr[0, 0]), -1 / 3, places=5)
            self.assertAlmostEqual(float(arr[0, 1]), -1 / 3, places=5)
            self.assertAlmostEqual(float(arr[1, 0]), 1.0, places=5)
            self.assertEqual(float(arr[1, 1]), NODATA)

    def test_gli_rgb_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp)
            out = self._run(session, index="gli")
            arr, _, _ = read_geotiff(out["file"])
            # (100-100-0)/(100+100+0)=0; (200-200-0)/(200+200+0)=0;
            # (50-0-0)/(50+0+0)=1.0.
            self.assertAlmostEqual(float(arr[0, 0]), 0.0, places=5)
            self.assertAlmostEqual(float(arr[0, 1]), 0.0, places=5)
            self.assertAlmostEqual(float(arr[1, 0]), 1.0, places=5)

    def test_unknown_index_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp)
            with self.assertRaises(RuntimeError):
                self._run(session, index="evi")

    def test_missing_input_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp)
            with self.assertRaises(RuntimeError):
                barrier_vegetation_index(_ctx(session))

    def test_all_nodata_input_writes_nodata_mosaic(self) -> None:
        import rasterio

        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp)
            src = session / "multispectral.tif"
            profile = {
                "driver": "GTiff",
                "height": 2,
                "width": 2,
                "count": 5,
                "dtype": "float32",
                "crs": "EPSG:32720",
                "transform": rasterio.Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
                "nodata": NODATA,
            }
            with rasterio.open(str(src), "w", **profile) as ds:
                ds.write(
                    np.full((5, 2, 2), NODATA, dtype=np.float32)
                )
            out = barrier_vegetation_index(
                {**_ctx(session), "raster_path": str(src)}
            )
            arr, _, _ = read_geotiff(out["file"])
            self.assertTrue(bool((arr == NODATA).all()))
            self.assertTrue(out.get("warnings"))


class MultispectralVariantTests(unittest.TestCase):
    def test_variant_discovered_with_raster_port(self) -> None:
        manager.discover()
        keys = [v["key"] for v in manager.list_variants(
            "lynceus.nodes.raster.input_raster"
        )]
        self.assertIn("multispectral", keys)
        info = manager.node_info("lynceus.nodes.raster.input_raster")
        self.assertIsNotNone(info)

    def test_multiband_file_passes_without_band_warning(self) -> None:
        from lynceus.nodes._product_input import _raster_validation_warnings

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "ms.tif"
            _write_multispectral(src)
            single = _raster_validation_warnings(src)
            multi = _raster_validation_warnings(src, multiband=True)
            self.assertTrue(
                any("only band 1" in w for w in single),
                single,
            )
            self.assertEqual(multi, [])

    def test_single_band_multispectral_warns(self) -> None:
        import rasterio

        from lynceus.nodes._product_input import _raster_validation_warnings

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "chm.tif"
            profile = {
                "driver": "GTiff",
                "height": 2,
                "width": 2,
                "count": 1,
                "dtype": "float32",
                "crs": "EPSG:32720",
                "transform": rasterio.Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
                "nodata": NODATA,
            }
            with rasterio.open(str(src), "w", **profile) as ds:
                ds.write(
                    np.full((1, 2, 2), 10.0, dtype=np.float32)
                )
            warnings = _raster_validation_warnings(src, multiband=True)
            self.assertTrue(
                any("multiband" in w for w in warnings),
                warnings,
            )

    def test_band_warning_names_the_product(self) -> None:
        from lynceus.nodes._product_input import _raster_validation_warnings

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "ms.tif"
            _write_multispectral(src)
            warnings = _raster_validation_warnings(src, product="DSM")
            self.assertTrue(
                any("DSM" in w and "only band 1" in w for w in warnings),
                warnings,
            )


if __name__ == "__main__":
    unittest.main()
