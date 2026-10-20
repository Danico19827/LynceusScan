# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Tests for the Merge Rasters strategy family (S11: shared-scale fusion).

Exact fusion of adjacent tiles, overlap methods (first/last/min/max),
hard errors (CRS/dtype mismatch, <2 wired), warning-only resolution
mismatch, all-NODATA propagation, single-scale .qml from merged stats,
and family discovery (base inert, one instance per product).
"""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
from affine import Affine

from lynceus.nodes.raster.ops.merge_rasters_dtm import barrier_merge_rasters_dtm
from lynceus.processing.raster import read_geotiff, write_geotiff

FAMILY = "lynceus.nodes.raster.ops.merge_rasters"
CRS = "EPSG:32621"


def _write(path: Path, grid: np.ndarray, west: float, north: float,
           res: float = 1.0, crs: str = CRS) -> Path:
    write_geotiff(str(path), grid.astype(np.float32),
                  Affine(res, 0.0, west, 0.0, -res, north),
                  crs=crs, nodata=-9999.0)
    return path


def _ctx(session: Path, files: list[Path], extra: dict | None = None) -> dict:
    session.mkdir(parents=True, exist_ok=True)
    ctx = {"session_dir": str(session)}
    for pos, path in enumerate(files):
        ctx[f"in{pos}_path"] = str(path)
    if extra:
        ctx.update(extra)
    return ctx


def _tiles(root: Path) -> list[Path]:
    """Two adjacent 10x10 DTM tiles: west [0..100), east [100..200)."""
    west = np.full((10, 10), 100.0, dtype=np.float32)
    east = np.full((10, 10), 200.0, dtype=np.float32)
    return [_write(root / "w.tif", west, 0.0, 10.0),
            _write(root / "e.tif", east, 10.0, 10.0)]


class FusionTests(unittest.TestCase):
    def test_exact_fusion_adjacent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = barrier_merge_rasters_dtm(_ctx(root / "s", _tiles(root)))
            arr, transform, _crs = read_geotiff(out["file"])
            self.assertEqual(arr.shape, (10, 20))
            self.assertTrue(bool((arr[:, :10] == 100.0).all()))
            self.assertTrue(bool((arr[:, 10:] == 200.0).all()))
            self.assertEqual((transform.c, transform.f), (0.0, 10.0))
            self.assertNotIn("warnings", out)
            self.assertEqual(out["display_name"], "DTM")

    def test_overlap_methods(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = _write(root / "a.tif",
                       np.full((10, 10), 10.0, dtype=np.float32), 0.0, 10.0)
            b = _write(root / "b.tif",
                       np.full((10, 10), 20.0, dtype=np.float32), 0.0, 10.0)
            expected = {"first": 10.0, "last": 20.0, "min": 10.0, "max": 20.0}
            for method, want in expected.items():
                with self.subTest(method=method):
                    out = barrier_merge_rasters_dtm(
                        _ctx(root / f"s_{method}", [a, b],
                             {"method": method}))
                    arr, _t, _c = read_geotiff(out["file"])
                    self.assertTrue(bool((arr == want).all()))

    def test_nodata_seams_fill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = np.full((10, 10), -9999.0, dtype=np.float32)
            a[:, :5] = 50.0
            b = np.full((10, 10), -9999.0, dtype=np.float32)
            b[:, 5:] = 60.0
            pa = _write(root / "a.tif", a, 0.0, 10.0)
            pb = _write(root / "b.tif", b, 0.0, 10.0)
            out = barrier_merge_rasters_dtm(_ctx(root / "s", [pa, pb]))
            arr, _t, _c = read_geotiff(out["file"])
            self.assertTrue(bool((arr[:, :5] == 50.0).all()))
            self.assertTrue(bool((arr[:, 5:] == 60.0).all()))

    def test_all_nodata_propagates_with_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = [_write(root / f"{name}.tif",
                            np.full((4, 4), -9999.0, dtype=np.float32),
                            0.0, 4.0) for name in ("a", "b")]
            out = barrier_merge_rasters_dtm(_ctx(root / "s", files))
            arr, _t, _c = read_geotiff(out["file"])
            self.assertTrue(bool((arr == -9999.0).all()))
            self.assertTrue(out.get("warnings"))


class ValidationTests(unittest.TestCase):
    def test_needs_two_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _tiles(root)
            for subset in ([], files[:1]):
                with self.subTest(n=len(subset)):
                    with self.assertRaisesRegex(RuntimeError, "at least two"):
                        barrier_merge_rasters_dtm(_ctx(root / "s", subset))

    def test_crs_mismatch_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _tiles(root)
            other = _write(root / "c.tif",
                           np.full((10, 10), 1.0, dtype=np.float32),
                           0.0, 10.0, crs="EPSG:32721")
            with self.assertRaisesRegex(RuntimeError, "one CRS"):
                barrier_merge_rasters_dtm(_ctx(root / "s", [files[0], other]))

    def test_resolution_mismatch_warns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fine = _write(root / "f.tif",
                          np.full((10, 10), 7.0, dtype=np.float32), 0.0, 10.0,
                          res=1.0)
            coarse = _write(root / "c.tif",
                            np.full((5, 5), 8.0, dtype=np.float32), 0.0, 10.0,
                            res=2.0)
            out = barrier_merge_rasters_dtm(_ctx(root / "s", [fine, coarse]))
            self.assertTrue(any("resolution" in str(w).lower()
                                for w in out.get("warnings", [])))
            arr, _t, _c = read_geotiff(out["file"])
            self.assertTrue(bool((arr != -9999.0).any()))

    def test_unknown_method_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "unknown method"):
                barrier_merge_rasters_dtm(
                    _ctx(root / "s", _tiles(root), {"method": "median"}))


class SingleScaleQmlTests(unittest.TestCase):
    def test_qml_spans_merged_range(self) -> None:
        from lynceus.processing.qml_style import generate_for_file

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = barrier_merge_rasters_dtm(_ctx(root / "s", _tiles(root)))
            path = Path(out["file"])
            self.assertTrue(generate_for_file(str(path), "dtm"))
            text = path.with_suffix(".qml").read_text(encoding="utf-8")
            lo = float(re.search(r'minimumValue="([\d.\-]+)"', text).group(1))
            hi = float(re.search(r'maximumValue="([\d.\-]+)"', text).group(1))
            arr, _t, _c = read_geotiff(path)
            valid = arr[arr != -9999.0]
            self.assertAlmostEqual(lo, float(valid.min()), places=4)
            self.assertAlmostEqual(hi, float(valid.max()), places=4)


class FamilyDiscoveryTests(unittest.TestCase):
    def test_four_product_variants(self) -> None:
        from lynceus.plugins.registry import manager

        manager.discover()
        keys = [v["key"] for v in manager.list_variants(FAMILY)]
        self.assertEqual(keys, ["chm", "dsm", "dtm", "raster"])

    def test_base_inert_without_strategy(self) -> None:
        from lynceus.plugins.registry import manager

        manager.discover()
        info = manager.node_info(FAMILY)
        self.assertIsNotNone(info)
        inputs, outputs = manager.load_ports(FAMILY)
        self.assertEqual(inputs, ())
        self.assertEqual(outputs, ())

    def test_effective_caps_per_product(self) -> None:
        from lynceus.processing.steps import effective_node_caps

        for key, fn in (
            ("dtm", "barrier_merge_rasters_dtm"),
            ("dsm", "barrier_merge_rasters_dsm"),
            ("chm", "barrier_merge_rasters_chm"),
            ("raster", "barrier_merge_rasters_raster"),
        ):
            caps = effective_node_caps(FAMILY, {"strategy": key})
            self.assertEqual(caps.get("barrier_task").__name__, fn)

    def test_resolved_ports_eight_optional(self) -> None:
        from lynceus.nodes._variants import resolved_ports
        from lynceus.nodes.ports import PortType, port_metadata

        inputs, outputs = resolved_ports(FAMILY, {"strategy": "dtm"})
        self.assertEqual(len(inputs), 8)
        self.assertTrue(port_metadata(inputs[0]).required)
        self.assertTrue(all(not port_metadata(item).required
                            for item in inputs[1:]))
        self.assertEqual(outputs, (PortType.DTM_MOSAIC,))


if __name__ == "__main__":
    unittest.main()
