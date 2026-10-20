# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Unit tests for independent-segment mosaic VRTs: deterministic builder,
pixel equivalence with the physical merge (first-wins overlaps), mismatch
skips, union-scale QML without pixel reads, segment grouping and gallery.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from affine import Affine

from lynceus.processing.controller import _segment_raster_groups
from lynceus.processing.qml_style import (
    generate_discrete_qml,
    generate_qml,
    read_palette_entries,
    read_ramp_stops,
)
from lynceus.processing.raster import (
    build_vrt_mosaic,
    merge_geotiffs,
    write_geotiff,
)
from lynceus.ui.outputs_model import flatten_outputs, viewer_for_path

CRS = "EPSG:32621"


def _mosaic(path: Path, value: float, west: float, width: int = 10) -> Path:
    arr = np.full((10, width), value, np.float32)
    write_geotiff(path, arr, Affine(1.0, 0.0, west, 0.0, -1.0, 10.0), crs=CRS)
    return path


class BuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="vrt_"))
        self.addCleanup(
            lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))

    def _pair(self):
        a = _mosaic(self.tmp / "a.tif", 5.0, 0.0)
        b = _mosaic(self.tmp / "b.tif", 9.0, 8.0)
        return a, b

    def test_build_is_deterministic(self) -> None:
        a, b = self._pair()
        out1 = self.tmp / "m1.vrt"
        out2 = self.tmp / "sub" / "m2.vrt"
        self.assertIn("file", build_vrt_mosaic([a, b], out1))
        self.assertIn("file", build_vrt_mosaic([b, a], out1))
        self.assertIn("file", build_vrt_mosaic([a, b], out2))
        # Same inputs, same location: byte-identical (source order is fixed).
        again = self.tmp / "m1b.vrt"
        build_vrt_mosaic([b, a], again)
        first = (self.tmp / "m1.vrt").read_bytes()
        self.assertEqual(first, again.read_bytes())
        self.assertIn(b"relativeToVRT", first)
        self.assertNotIn(b"C:", first)

    def test_matches_physical_merge_first_wins(self) -> None:
        import rasterio

        a, b = self._pair()
        out = self.tmp / "m.vrt"
        build_vrt_mosaic([a, b], out)
        merge_geotiffs([a, b], CRS, self.tmp / "m.tif")
        with rasterio.open(out) as v, rasterio.open(
            self.tmp / "m.tif"
        ) as m:
            self.assertEqual((v.width, v.height), (m.width, m.height))
            self.assertEqual(v.crs, m.crs)
            self.assertEqual(v.nodata, m.nodata)
            va, ma = v.read(1), m.read(1)
        # Overlap (x 8..10) carries a=5.0: first-sorted source wins.
        self.assertTrue(np.array_equal(va, ma))
        self.assertEqual(float(va[5, 9]), 5.0)
        self.assertEqual(float(va[5, 11]), 9.0)

    def test_nodata_falls_through_on_overlaps(self) -> None:
        """Void pixels of the winning source expose the one beneath.

        The independent-segment stitch regression class: SimpleSource
        stamps the winner's NODATA over valid data along segment
        overlaps (visible as gaps); the physical merge falls through,
        so the VRT must too.
        """
        import rasterio
        from rasterio.windows import Window

        a_arr = np.full((10, 10), 5.0, np.float32)
        a_arr[:, :4] = -9999.0
        a = self.tmp / "na.tif"
        write_geotiff(
            a, a_arr, Affine(1.0, 0.0, 0.0, 0.0, -1.0, 10.0), crs=CRS,
            nodata=-9999.0)
        b_arr = np.full((10, 10), 9.0, np.float32)
        b = self.tmp / "nb.tif"
        write_geotiff(
            b, b_arr, Affine(1.0, 0.0, 3.0, 0.0, -1.0, 10.0), crs=CRS,
            nodata=-9999.0)

        vrt = self.tmp / "n.vrt"
        build_vrt_mosaic([a, b], vrt)
        merge_geotiffs([a, b], CRS, self.tmp / "n.tif")
        with rasterio.open(vrt) as v:
            full = v.read(1)
            # merge crops to valid bounds (west 3, width 10); the VRT
            # keeps the union: compare the common region.
            common = v.read(1, window=Window(3, 0, 10, 10))
        # a is void at union col 3: b shows through instead of NODATA.
        self.assertEqual(float(full[5, 3]), 9.0)
        # a valid inside the overlap: first-sorted wins over b.
        self.assertEqual(float(full[5, 5]), 5.0)
        with rasterio.open(self.tmp / "n.tif") as m:
            np.testing.assert_array_equal(common, m.read(1))


    def test_skips_mismatched_sources(self) -> None:
        a, _ = self._pair()
        other_crs = _mosaic(self.tmp / "c.tif", 1.0, 0.0)
        import rasterio

        with rasterio.open(other_crs, "r+") as dst:
            dst.crs = rasterio.crs.CRS.from_epsg(32620)
        res = build_vrt_mosaic([a, other_crs], self.tmp / "x.vrt")
        self.assertIn("CRS differs", res["skipped"])
        fine = _mosaic(self.tmp / "d.tif", 1.0, 0.0)
        arr = np.full((10, 10), 2.0, np.float32)
        write_geotiff(
            self.tmp / "e.tif", arr, Affine(2.0, 0.0, 0.0, 0.0, -2.0, 20.0),
            crs=CRS,
        )
        res = build_vrt_mosaic([fine, self.tmp / "e.tif"], self.tmp / "y.vrt")
        self.assertIn("resolution differs", res["skipped"])
        self.assertEqual(
            build_vrt_mosaic([], self.tmp / "z.vrt")["skipped"], "no sources")

    def test_skips_dtype_mismatch(self) -> None:
        import rasterio

        a, _ = self._pair()
        u8 = np.full((10, 10), 3, np.uint8)
        with rasterio.open(
            self.tmp / "u.tif", "w", driver="GTiff", height=10, width=10,
            count=1, dtype="uint8", crs=CRS,
            transform=Affine(1.0, 0.0, 20.0, 0.0, -1.0, 10.0),
        ) as dst:
            dst.write(u8, 1)
        res = build_vrt_mosaic([a, self.tmp / "u.tif"], self.tmp / "u.vrt")
        self.assertIn("dtype differs", res["skipped"])


class VrtQmlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="vrtqml_"))
        self.addCleanup(
            lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))

    def test_union_range_without_pixel_reads(self) -> None:
        low = self.tmp / "s0" / "seg_dtm_mosaic.tif"
        high = self.tmp / "s1" / "seg_dtm_mosaic.tif"
        write_geotiff(
            low, np.arange(100, dtype=np.float32).reshape(10, 10) + 10.0,
            Affine(1.0, 0.0, 0.0, 0.0, -1.0, 10.0), crs=CRS)
        write_geotiff(
            high, np.arange(100, dtype=np.float32).reshape(10, 10) + 80.0,
            Affine(1.0, 0.0, 10.0, 0.0, -1.0, 10.0), crs=CRS)
        self.assertTrue(generate_qml(str(low), ramp_key="dtm"))
        self.assertTrue(generate_qml(str(high), ramp_key="dtm"))
        from lynceus.processing.qml_style import generate_vrt_qml

        vrt = self.tmp / "final" / "dtm_mosaic.vrt"
        build_vrt_mosaic([low, high], vrt)
        self.assertTrue(generate_vrt_qml(
            str(vrt),
            [str(low.with_suffix(".qml")), str(high.with_suffix(".qml"))],
        ))
        parsed = read_ramp_stops(str(vrt.with_suffix(".qml")))
        assert parsed is not None
        lo, hi, _ = parsed
        self.assertAlmostEqual(lo, 10.0)
        self.assertAlmostEqual(hi, 179.0)

    def test_unanimous_palette_is_copied(self) -> None:
        from lynceus.processing.qml_style import generate_vrt_qml

        modal = [(0, "#000000", "void"), (1, "#ffffff", "full")]
        first = _mosaic(self.tmp / "p0.tif", 0.0, 0.0)
        second = _mosaic(self.tmp / "p1.tif", 1.0, 10.0)
        self.assertTrue(generate_discrete_qml(str(first), modal, "mask"))
        (second.with_suffix(".qml")).write_bytes(
            first.with_suffix(".qml").read_bytes())
        vrt = self.tmp / "mask.vrt"
        build_vrt_mosaic([first, second], vrt)
        self.assertTrue(generate_vrt_qml(
            str(vrt),
            [str(first.with_suffix(".qml")), str(second.with_suffix(".qml"))],
        ))
        self.assertEqual(
            read_palette_entries(str(vrt.with_suffix(".qml"))),
            read_palette_entries(str(first.with_suffix(".qml"))),
        )

    def test_heterogeneous_styles_refuse(self) -> None:
        from lynceus.processing.qml_style import generate_vrt_qml

        low = _mosaic(self.tmp / "h0.tif", 10.0, 0.0)
        high = _mosaic(self.tmp / "h1.tif", 90.0, 10.0)
        self.assertTrue(generate_qml(str(low), ramp_key="dtm"))
        self.assertTrue(generate_qml(str(high), ramp_key="chm"))
        vrt = self.tmp / "h.vrt"
        build_vrt_mosaic([low, high], vrt)
        self.assertFalse(generate_vrt_qml(
            str(vrt),
            [str(low.with_suffix(".qml")), str(high.with_suffix(".qml"))],
        ))
        self.assertFalse(generate_vrt_qml(str(vrt), []))


class GroupsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="vrtgrp_"))
        self.addCleanup(
            lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))

    def test_groups_barrier_rasters_by_relative_path(self) -> None:
        seg_dirs = [self.tmp / f"segment_{k:04d}" for k in range(2)]
        payloads = []
        for k, seg in enumerate(seg_dirs):
            (seg / "dtm").mkdir(parents=True)
            tif = _mosaic(seg / "dtm" / "dtm_mosaic.tif", float(k), k * 10.0)
            payloads.append({"dtm": {"file": str(tif), "count": k}})
        groups = _segment_raster_groups(payloads, seg_dirs, {"dtm"})
        self.assertEqual(list(groups), ["dtm/dtm_mosaic.tif"])
        members = groups["dtm/dtm_mosaic.tif"]
        self.assertEqual([iid for iid, _ in members], ["dtm", "dtm"])

    def test_single_source_and_streams_never_group(self) -> None:
        seg = self.tmp / "segment_0000"
        (seg / "dtm").mkdir(parents=True)
        tif = _mosaic(seg / "dtm" / "dtm_mosaic.tif", 1.0, 0.0)
        payloads = [{"dtm": {"file": str(tif)}}]
        self.assertEqual(
            _segment_raster_groups(payloads, [seg], {"dtm"}), {})
        stream = _mosaic(self.tmp / "s.tif", 1.0, 0.0)
        payloads = [{"tile": {"file": str(stream)}}]
        self.assertEqual(
            _segment_raster_groups(payloads, [seg], {"tile"}), {})


class BuildSegmentVrtsTests(unittest.TestCase):
    """End-to-end ``_build_segment_vrts``: which instances get a VRT.

    Regression: the instance filter used to require "barrier task and NO
    tile task", silently skipping Generate DTM/DSM (tile+barrier) while
    only barrier-only nodes like CHM got a virtual mosaic.
    """

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="vrtbuild_"))
        self.addCleanup(
            lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))

    def test_tile_and_barrier_nodes_group_too(self) -> None:
        from types import SimpleNamespace

        from lynceus.processing.controller import PipelineController

        session = self.tmp
        payloads = []
        for k in range(2):
            seg = session / "lotes" / f"segment_{k:04d}"
            (seg / "artifacts").mkdir(parents=True)
            dtm = _mosaic(
                seg / "artifacts" / "dtm_mosaic.tif", float(k), 10.0 * k)
            chm = _mosaic(
                seg / "artifacts" / "chm_mosaic.tif", 1.0 + k, 10.0 * k)
            payloads.append({
                "dtm1": {"file": str(dtm)},
                "chm1": {"file": str(chm)},
            })
        controller = PipelineController()
        controller._modules = {
            "dtm1": "lynceus.nodes.lidar.terrain.generate_dtm",
            "chm1": "lynceus.nodes.lidar.terrain.generate_chm",
        }
        messages: list = []
        cb = SimpleNamespace(
            on_message=lambda text, kind="info": messages.append((text, kind)))
        outputs: dict = {}
        controller._build_segment_vrts(
            session, [None, None], payloads, outputs, {}, cb)
        self.assertFalse([m for m in messages if m[1] == "warning"])
        for name, iid in (("dtm_mosaic", "dtm1"), ("chm_mosaic", "chm1")):
            vrt = session / "final" / "artifacts" / f"{name}.vrt"
            self.assertTrue(vrt.is_file(), name)
            self.assertEqual(outputs[iid]["vrt_mosaic"], str(vrt))
            self.assertTrue(vrt.with_suffix(".meta.json").is_file())


class GalleryTests(unittest.TestCase):
    def test_vrt_resolves_to_raster_viewer(self) -> None:
        self.assertEqual(viewer_for_path("final/dtm_mosaic.vrt"), "raster")

    def test_vrt_payload_flattens_with_qml(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="vrtgal_"))
        self.addCleanup(
            lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        tif = _mosaic(tmp / "dtm_mosaic.tif", 4.0, 0.0)
        vrt = tmp / "dtm_mosaic.vrt"
        build_vrt_mosaic([tif], vrt)
        vrt.with_suffix(".qml").write_text("<qgis/>", encoding="utf-8")
        products = flatten_outputs(
            {"dtm": {"file": str(tif), "vrt_mosaic": str(vrt)}},
            {"dtm": "Generate DTM"},
        )
        by_key = {p.payload_key: p for p in products}
        self.assertEqual(by_key["vrt_mosaic"].viewer, "raster")
        self.assertEqual(by_key["vrt_mosaic"].qml, "dtm_mosaic.qml")
        self.assertEqual(by_key["file"].viewer, "raster")


if __name__ == "__main__":
    unittest.main()
