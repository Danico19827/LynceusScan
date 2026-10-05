# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""End-to-end isolation for branch-scoped instances (F4.2a).

Runs the real DTM tile/merge writers for two same-module instances over
the SAME synthetic tile (the S-case that used to collide), then checks
the whole chain: scoped tile dirs, per-instance mosaics that never see
each other's tiles, scoped consumer paths from ``_inject_input_paths``,
scope-aware bundle recording, the barrier/tile role split and the
finals-only sweep. In-process with a synthetic LAZ (no pool, no Qt).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import laspy
import numpy as np

from lynceus.nodes.lidar.terrain.generate_dtm import (
    barrier_generate_dtm,
    tile_generate_dtm,
)
from lynceus.nodes.raster.ops.chm_difference import barrier_chm_difference
from lynceus.nodes.ports import PortType
from lynceus.processing.controller import PipelineController
from lynceus.processing.steps import (
    _inject_input_paths,
    effective_node_caps,
)
from lynceus.ui.outputs_model import (
    compute_flow_segments,
    display_labels,
    flatten_outputs,
    render_flow,
)

DTM = "lynceus.nodes.lidar.terrain.generate_dtm"
TILE_ID = "tag_c0000_r0000"


def _write_ground_laz(path: Path, z: float = 5.0) -> Path:
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = [0.01, 0.01, 0.01]
    header.offsets = [0.0, 0.0, 0.0]
    xs, ys = np.meshgrid(
        np.arange(0.5, 10.0, 1.0), np.arange(0.5, 10.0, 1.0)
    )
    n = xs.size
    record = laspy.ScaleAwarePointRecord.zeros(n, header=header)
    record.x = xs.ravel()
    record.y = ys.ravel()
    record.z = np.full(n, z)
    record.classification = np.full(n, 2, dtype=np.uint8)
    with laspy.open(str(path), mode="w", header=header) as writer:
        writer.write_points(record)
    return path


def _tile(laz: Path) -> dict:
    return {
        "tile_id": TILE_ID,
        "file": str(laz),
        "core_x_min": 0.0,
        "core_x_max": 10.0,
        "core_y_min": 0.0,
        "core_y_max": 10.0,
    }


def _ctx(arts: Path, scope: str = "") -> dict:
    ctx = {
        "session_dir": str(arts),
        "resolution": 1.0,
        "nodata_value": -9999.0,
        "interpolation_method": "min",
        "hole_fill_enabled": False,
        "smooth_enabled": False,
        "crop_to_valid": False,
    }
    if scope:
        ctx["tile_scope"] = scope
    return ctx


class ScopedDtmTests(unittest.TestCase):
    def test_two_instances_same_tile_stay_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            arts = Path(tmp) / "artifacts"
            arts.mkdir()
            laz = _write_ground_laz(Path(tmp) / "tile.laz")
            tile = _tile(laz)

            # Instance d1 tiles and merges first.
            out1 = tile_generate_dtm(tile, _ctx(arts, "d1"))
            self.assertEqual(
                out1["output"],
                str(arts / "dtm" / "d1" / f"{TILE_ID}.tif"),
            )
            res1 = barrier_generate_dtm(_ctx(arts, "d1"))
            self.assertEqual(res1["file"], str(arts / "d1" / "dtm_mosaic.tif"))
            mosaic1_bytes = Path(res1["file"]).read_bytes()

            # Instance d2 over the SAME tile lands apart; d1 is untouched.
            out2 = tile_generate_dtm(tile, _ctx(arts, "d2"))
            self.assertEqual(
                out2["output"],
                str(arts / "dtm" / "d2" / f"{TILE_ID}.tif"),
            )
            res2 = barrier_generate_dtm(_ctx(arts, "d2"))
            self.assertEqual(res2["file"], str(arts / "d2" / "dtm_mosaic.tif"))
            self.assertEqual(Path(res1["file"]).read_bytes(), mosaic1_bytes)
            # Same source tile: both mosaics carry the same ground height
            # (the raw bytes differ only in the embedded provenance tag,
            # which records each instance's own scope).
            from lynceus.processing.raster import read_geotiff

            arr1, _, _ = read_geotiff(res1["file"])
            arr2, _, _ = read_geotiff(res2["file"])
            self.assertTrue((arr1 == arr2).all())
            self.assertAlmostEqual(float(arr1[5, 5]), 5.0, places=2)

            # Legacy layout untouched when nothing is scoped.
            out0 = tile_generate_dtm(tile, _ctx(arts))
            self.assertEqual(
                out0["output"], str(arts / "dtm" / f"{TILE_ID}.tif")
            )
            res0 = barrier_generate_dtm(_ctx(arts))
            self.assertEqual(res0["file"], str(arts / "dtm_mosaic.tif"))

    def test_inject_resolves_scoped_mosaic_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            arts = Path(tmp) / "artifacts"
            (arts / "d1").mkdir(parents=True)
            (arts / "d1" / "dtm_mosaic.tif").write_bytes(b"m")
            (arts / "dtm_mosaic.tif").write_bytes(b"root")

            def _caps_of(iid: str):
                return DTM, effective_node_caps(DTM, None)

            edges = [("d1", "c", PortType.DTM_MOSAIC, PortType.POINT_CLOUD)]
            scoped = {"session_dir": str(arts)}
            _inject_input_paths(scoped, DTM, edges, _caps_of, None, {"d1": "d1"})
            self.assertEqual(
                scoped["dtm_mosaic_path"],
                str(arts / "d1" / "dtm_mosaic.tif"),
            )
            legacy = {"session_dir": str(arts)}
            _inject_input_paths(legacy, DTM, edges, _caps_of, None, {})
            self.assertEqual(
                legacy["dtm_mosaic_path"], str(arts / "dtm_mosaic.tif")
            )

    def test_inject_resolves_input_session_file(self) -> None:
        # Input-style providers (session_file, no output_files) resolve
        # to session/<src_iid>/<file>: writing anywhere else (e.g. via
        # bare scoped_file) leaves the consumer with a dangling path.
        with tempfile.TemporaryDirectory() as tmp:
            arts = Path(tmp) / "artifacts"
            (arts / "in1").mkdir(parents=True)
            (arts / "in1" / "image.png").write_bytes(b"png")

            def _caps_of(iid: str):
                return "t2.input_image", {
                    "output_port": "t2.image",
                    "session_file": "image.png",
                    "input_file_key": "file_path",
                }

            ctx = {"session_dir": str(arts), "node_iid": "flt1"}
            _inject_input_paths(
                ctx,
                "t2.image_filter",
                [("in1", "flt1", "t2.image", "t2.image")],
                _caps_of,
                None,
                {},
            )
            self.assertEqual(
                ctx.get("t2.image_path"), str(arts / "in1" / "image.png")
            )


class ScopedBundleTests(unittest.TestCase):
    def _controller(self, modules: dict) -> PipelineController:
        ctrl = PipelineController()
        ctrl._modules = modules
        ctrl._run_configs = {}
        return ctrl

    def test_bundle_records_scope_dirs_and_roles_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            arts = Path(tmp) / "artifacts"
            tile_dir = arts / "dtm" / "d1"
            tile_dir.mkdir(parents=True)
            (tile_dir / f"{TILE_ID}.tif").write_bytes(b"t")
            (arts / "d1").mkdir(exist_ok=True)
            (arts / "d1" / "dtm_mosaic.tif").write_bytes(b"m")
            (arts / "dtm_mosaic.tif").write_bytes(b"root")
            (arts / "dtm" / "d2").mkdir(parents=True)
            (arts / "dtm" / "d2" / f"{TILE_ID}.tif").write_bytes(b"foreign")

            ctrl = self._controller({"d1": DTM})
            files = ctrl._module_bundle_files(arts, DTM, None, "d1", {"d1", "d2"})
            self.assertIn("d1/dtm_mosaic.tif", files)
            self.assertIn(f"dtm/d1/{TILE_ID}.tif", files)
            # The sibling scope is never recorded in this instance's bundle
            # (materializing it would freeze stale products on recompute).
            self.assertNotIn(f"dtm/d2/{TILE_ID}.tif", files)

            caps = effective_node_caps(DTM, None)
            barrier, tile = ctrl._entry_role_files(caps, files)
            # Both the scoped final and the legacy root final read as
            # barrier-need; per-tile outputs stay tile-need.
            self.assertEqual(barrier, ["d1/dtm_mosaic.tif", "dtm_mosaic.tif"])
            self.assertEqual(tile, [f"dtm/d1/{TILE_ID}.tif"])

    def test_sweep_keeps_scoped_mosaic_drops_scoped_tiles(self) -> None:
        import json

        with tempfile.TemporaryDirectory() as tmp:
            sess = Path(tmp) / "20260101_000000"
            arts = sess / "artifacts"
            (arts / "dtm" / "d1").mkdir(parents=True)
            (arts / "d1").mkdir(parents=True)
            (arts / "dtm" / "d1" / f"{TILE_ID}.tif").write_bytes(b"t")
            (arts / "d1" / "dtm_mosaic.tif").write_bytes(b"m")
            prov = {
                "nodes": {
                    "d1": {
                        "module": DTM,
                        "fp": "x",
                        "files": [
                            f"dtm/d1/{TILE_ID}.tif",
                            "d1/dtm_mosaic.tif",
                        ],
                    },
                }
            }
            (sess / "provenance.json").write_text(
                json.dumps(prov), encoding="utf-8"
            )
            removed = self._controller({"d1": DTM})._sweep_session(sess)
            self.assertEqual(removed, 1)
            self.assertTrue((arts / "d1" / "dtm_mosaic.tif").is_file())
            self.assertFalse((arts / "dtm" / "d1").exists())


class BarrierOnlyScopeTests(unittest.TestCase):
    def _write_chm(self, path: Path, value: float) -> None:
        import numpy as np
        from affine import Affine

        from lynceus.processing.raster import write_geotiff

        write_geotiff(
            str(path),
            np.full((6, 6), value, dtype=np.float32),
            Affine(1.0, 0.0, 0.0, 0.0, -1.0, 6.0),
            crs="EPSG:32721",
            nodata=-9999.0,
        )

    def test_chm_difference_twins_write_apart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            arts = Path(tmp) / "artifacts"
            arts.mkdir()
            a = arts / "a.tif"
            b = arts / "b.tif"
            self._write_chm(a, 10.0)
            self._write_chm(b, 4.0)

            def _ctx(scope: str = "") -> dict:
                ctx = {
                    "session_dir": str(arts),
                    "in0_path": str(a),
                    "in1_path": str(b),
                    "nodata_value": -9999.0,
                }
                if scope:
                    ctx["tile_scope"] = scope
                return ctx

            res1 = barrier_chm_difference(_ctx("c1"))
            self.assertEqual(
                res1["file"], str(arts / "c1" / "chm_difference.tif")
            )
            before = Path(res1["file"]).read_bytes()
            res2 = barrier_chm_difference(_ctx("c2"))
            self.assertEqual(
                res2["file"], str(arts / "c2" / "chm_difference.tif")
            )
            # The first twin survives the second write untouched.
            self.assertEqual(Path(res1["file"]).read_bytes(), before)
            # Legacy graphs keep the canonical root product.
            res0 = barrier_chm_difference(_ctx())
            self.assertEqual(
                res0["file"], str(arts / "chm_difference.tif")
            )


class DisplayLabelTests(unittest.TestCase):
    def test_unique_basenames_keep_basename(self) -> None:
        labels = display_labels(
            ["dtm_mosaic.tif", "dsm_mosaic.tif"],
            ["/s/dtm_mosaic.tif", "/s/dsm_mosaic.tif"],
            ["d1", "d2"],
        )
        self.assertEqual(labels, ["dtm_mosaic.tif", "dsm_mosaic.tif"])

    def test_scoped_twins_show_relative_paths(self) -> None:
        import os

        labels = display_labels(
            ["dtm_mosaic.tif", "dtm_mosaic.tif"],
            ["/s/d1/dtm_mosaic.tif", "/s/d2/dtm_mosaic.tif"],
            ["d1", "d2"],
        )
        self.assertEqual(
            labels,
            [f"d1{os.sep}dtm_mosaic.tif", f"d2{os.sep}dtm_mosaic.tif"],
        )

    def test_pathless_duplicates_fall_back_to_iid(self) -> None:
        labels = display_labels(
            ["Point Cloud (4 tiles)", "Point Cloud (4 tiles)"],
            [None, None],
            ["e1", "e2"],
        )
        self.assertEqual(
            labels,
            ["Point Cloud (4 tiles) — e1", "Point Cloud (4 tiles) — e2"],
        )


class FlowLineageTests(unittest.TestCase):
    NODES = {
        "load": "Load LAS/LAZ",
        "c": "Classify Ground",
        "d1": "Generate DTM",
        "d2": "Generate DTM",
        "chm": "Generate CHM",
    }
    EDGES = [
        ("load", "c"),
        ("c", "d1"),
        ("c", "d2"),
        ("d1", "chm"),
    ]

    def test_linear_chain_marks_self(self) -> None:
        flows = compute_flow_segments(
            self.NODES, self.EDGES, source_files={"load": "vuelo.laz"}
        )
        self.assertEqual(
            render_flow(flows["c"]),
            "Load LAS/LAZ (vuelo.laz) → [Classify Ground] → "
            "Generate DTM · d1 → Generate DTM · d2 → Generate CHM",
        )

    def test_twins_share_chain_but_mark_own_scope(self) -> None:
        flows = compute_flow_segments(self.NODES, self.EDGES)
        d1 = render_flow(flows["d1"])
        d2 = render_flow(flows["d2"])
        self.assertNotEqual(d1, d2)
        self.assertIn("[Generate DTM · d1]", d1)
        self.assertIn("[Generate DTM · d2]", d2)
        # The common ancestors stay clean (no iid noise).
        self.assertTrue(d1.startswith("Load LAS/LAZ → Classify Ground → "))

    def test_custom_labels_disambiguate_naturally(self) -> None:
        flows = compute_flow_segments(
            self.NODES, self.EDGES,
            custom_labels={"d1": "DTM vuelo norte", "d2": "DTM vuelo sur"},
        )
        self.assertIn("[DTM vuelo norte]", render_flow(flows["d1"]))
        self.assertIn("[DTM vuelo sur]", render_flow(flows["d2"]))
        self.assertNotIn("· d1", render_flow(flows["d1"]))

    def test_lone_node_has_no_chain(self) -> None:
        flows = compute_flow_segments({"solo": "Input Table"}, [])
        self.assertEqual(len(flows["solo"]), 1)
        self.assertEqual(render_flow(flows["solo"]), "[Input Table]")

    def test_flatten_carries_flow_to_products(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "dtm_mosaic.tif"
            target.write_bytes(b"x")
            products = flatten_outputs(
                {"d1": {"file": str(target)}},
                {"d1": "Generate DTM", "load": "Load LAS/LAZ"},
                edges=[("load", "d1")],
                custom_labels={},
                source_files={"load": "vuelo.laz"},
            )
            self.assertEqual(len(products), 1)
            self.assertEqual(
                render_flow(products[0].flow),
                "Load LAS/LAZ (vuelo.laz) → [Generate DTM]",
            )


if __name__ == "__main__":
    unittest.main()
