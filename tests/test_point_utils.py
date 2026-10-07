# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Tests for the point-cloud utility nodes (T4).

Normalize Heights (bilinear DTM sampling, chain break, outside/withheld
handling), Filter by Return (all four modes on modern + legacy return
dimensions) and Clip by Polygon (inside/boundary/buffer, fail-loud WKT).
Tile-level end-to-end over written LAZ/GeoTIFF fixtures; stdlib + engine
modules only.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

NORMALIZE = "lynceus.nodes.lidar.clean.normalize_heights"
DENORMALIZE = "lynceus.nodes.lidar.clean.denormalize_heights"
RETURNS = "lynceus.nodes.lidar.clean.return_filter"
CLIP = "lynceus.nodes.lidar.clean.clip_polygon"


def _write_cloud(path: Path, xs, ys, zs, cls, point_format: int = 6,
                 ret=None, num=None, withheld=None) -> Path:
    import laspy

    xs = np.asarray(xs, dtype=np.float64)
    n = len(xs)
    header = laspy.LasHeader(point_format=point_format, version="1.4")
    header.scales = np.array([0.01, 0.01, 0.01])
    header.offsets = np.array([0.0, 0.0, 0.0])
    rec = laspy.ScaleAwarePointRecord.zeros(
        n, point_format=header.point_format,
        scales=header.scales, offsets=header.offsets,
    )
    rec.x = xs
    rec.y = np.asarray(ys, dtype=np.float64)
    rec.z = np.asarray(zs, dtype=np.float64)
    rec.classification = np.asarray(cls, dtype=np.uint8)
    if ret is not None:
        if point_format >= 6:
            rec.return_number = np.asarray(ret, dtype=np.uint8)
            rec.number_of_returns = np.asarray(num, dtype=np.uint8)
        else:
            rec.return_num = np.asarray(ret, dtype=np.uint8)
            rec.num_returns = np.asarray(num, dtype=np.uint8)
    if withheld is not None:
        rec.withheld = np.asarray(withheld, dtype=bool)
    with laspy.open(str(path), mode="w", header=header) as writer:
        writer.write_points(rec)
    return path


def _write_dtm(path: Path) -> Path:
    from affine import Affine

    from lynceus.processing.raster import write_geotiff

    xs = np.arange(0.5, 20.0, 1.0)
    grid = np.tile(100.0 + 0.5 * xs, (10, 1)).astype(np.float32)
    write_geotiff(str(path), grid, Affine(1.0, 0.0, 0.0, 0.0, -1.0, 10.0),
                  crs="EPSG:32721", nodata=-9999.0)
    return path


def _read_z(path: Path):
    import laspy

    with laspy.open(str(path)) as reader:
        chunks = [np.asanyarray(chunk.z)
                  for chunk in reader.chunk_iterator(2_000_000)]
    return np.concatenate(chunks) if chunks else np.zeros(0)


def _read_cls(path: Path):
    import laspy

    with laspy.open(str(path)) as reader:
        chunks = [np.asanyarray(chunk.classification)
                  for chunk in reader.chunk_iterator(2_000_000)]
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.uint8)


def _ctx(session: Path, extra: dict | None = None) -> dict:
    session.mkdir(parents=True, exist_ok=True)
    ctx = {"session_dir": str(session)}
    if extra:
        ctx.update(extra)
    return ctx


class NormalizeTests(unittest.TestCase):
    def test_normalizes_plane_and_canopy(self) -> None:
        from lynceus.nodes.lidar.clean.normalize_heights import (
            tile_normalize_heights,
        )

        rng = np.random.default_rng(3)
        # Interior to the DTM extent on purpose: border points lack a
        # full bilinear support and keep their Z (covered separately).
        x = rng.uniform(2, 18, 300)
        y = rng.uniform(2, 8, 300)
        z = 100.0 + 0.5 * x
        cls = np.full(300, 2, dtype=np.uint8)
        vx = rng.uniform(2, 18, 60)
        vy = rng.uniform(2, 8, 60)
        vz = 100.0 + 0.5 * vx + 6.0
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = _write_cloud(
                root / "t.laz",
                np.concatenate([x, vx]), np.concatenate([y, vy]),
                np.concatenate([z, vz]),
                np.concatenate([cls, np.full(60, 5, dtype=np.uint8)]),
            )
            dtm = _write_dtm(root / "dtm.tif")
            out = tile_normalize_heights(
                {"tile_id": "t", "file": str(src)},
                _ctx(root / "s", {"dtm_mosaic_path": str(dtm)}),
            )
            self.assertEqual(out["normalized"], 360)
            self.assertEqual(out["kept_original"], 0)
            self.assertNotIn("warnings", out)
            got = _read_z(Path(out["output"]))
            self.assertEqual(len(got), 360)
            self.assertTrue(bool(np.allclose(got[:300], 0.0, atol=0.02)))
            self.assertTrue(bool(np.allclose(got[300:], 6.0, atol=0.02)))
            # Classes travel untouched (only Z is replaced).
            self.assertTrue(bool((_read_cls(Path(out["output"]))[:300] == 2).all()))

    def test_outside_and_withheld_keep_original(self) -> None:
        from lynceus.nodes.lidar.clean.normalize_heights import (
            tile_normalize_heights,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = _write_cloud(
                root / "t.laz",
                [5.0, 50.0, 6.0], [5.0, 5.0, 5.0],
                [102.5, 200.0, 103.0], [2, 2, 2],
                withheld=[False, False, True],
            )
            dtm = _write_dtm(root / "dtm.tif")
            out = tile_normalize_heights(
                {"tile_id": "t", "file": str(src)},
                _ctx(root / "s", {"dtm_mosaic_path": str(dtm)}),
            )
            self.assertEqual(out["normalized"], 1)
            self.assertEqual(out["withheld"], 1)
            self.assertTrue(bool(out.get("warnings")))
            got = _read_z(Path(out["output"]))
            self.assertAlmostEqual(float(got[0]), 0.0, places=2)
            self.assertAlmostEqual(float(got[1]), 200.0, places=2)
            self.assertAlmostEqual(float(got[2]), 103.0, places=2)

    def test_missing_dtm_raises(self) -> None:
        from lynceus.nodes.lidar.clean.normalize_heights import (
            tile_normalize_heights,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = _write_cloud(root / "t.laz", [1.0], [1.0], [400.0], [2])
            with self.assertRaisesRegex(RuntimeError, "DTM product missing"):
                tile_normalize_heights(
                    {"tile_id": "t", "file": str(src)},
                    _ctx(root / "s",
                         {"dtm_mosaic_path": str(root / "nope.tif")}),
                )

    def test_breaks_classified_chain(self) -> None:
        from lynceus.nodes.lidar.clean.normalize_heights import (
            PROCESSING_SPECS,
        )

        # A DTM over normalized heights would surface heights as
        # elevation: the node must not re-emit the classified flag.
        self.assertNotIn("provides", PROCESSING_SPECS)
        self.assertNotIn("passes_flags", PROCESSING_SPECS)


class DenormalizeTests(unittest.TestCase):
    def _scene(self, root: Path):
        from lynceus.nodes.lidar.clean.normalize_heights import (
            tile_normalize_heights,
        )

        rng = np.random.default_rng(17)
        x = rng.uniform(2, 18, 200)
        y = rng.uniform(2, 8, 200)
        z = 100.0 + 0.5 * x
        cls = np.full(200, 2, dtype=np.uint8)
        src = _write_cloud(root / "t.laz", x, y, z, cls)
        dtm = _write_dtm(root / "dtm.tif")
        norm = tile_normalize_heights(
            {"tile_id": "t", "file": str(src)},
            _ctx(root / "s", {"dtm_mosaic_path": str(dtm)}),
        )
        return Path(norm["output"]), dtm, (x, y, z)

    def test_roundtrip_recovers_original(self) -> None:
        from lynceus.nodes.lidar.clean.denormalize_heights import (
            tile_denormalize_heights,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            norm_path, dtm, (x, _y, z) = self._scene(root)
            out = tile_denormalize_heights(
                {"tile_id": "t", "file": str(norm_path)},
                _ctx(root / "s2", {"dtm_mosaic_path": str(dtm)}),
            )
            self.assertEqual(out["restored"], 200)
            self.assertEqual(out["kept_original"], 0)
            self.assertNotIn("warnings", out)
            got = _read_z(Path(out["output"]))
            self.assertTrue(bool(np.allclose(got, z, atol=0.02)))

    def test_direct_on_normalized_cloud(self) -> None:
        # No original available: a vendor-style normalized cloud (canopy
        # 6 m over a datum plane) recovers absolute elevation with the
        # same DTM.
        from lynceus.nodes.lidar.clean.denormalize_heights import (
            tile_denormalize_heights,
        )

        rng = np.random.default_rng(19)
        x = rng.uniform(2, 18, 120)
        y = rng.uniform(2, 8, 120)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = _write_cloud(root / "n.laz", x, y,
                               np.full(120, 6.0),
                               np.full(120, 5, dtype=np.uint8))
            dtm = _write_dtm(root / "dtm.tif")
            out = tile_denormalize_heights(
                {"tile_id": "t", "file": str(src)},
                _ctx(root / "s", {"dtm_mosaic_path": str(dtm)}),
            )
            got = _read_z(Path(out["output"]))
            want = 100.0 + 0.5 * x + 6.0
            self.assertTrue(bool(np.allclose(got, want, atol=0.02)))

    def test_missing_dtm_raises(self) -> None:
        from lynceus.nodes.lidar.clean.denormalize_heights import (
            tile_denormalize_heights,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = _write_cloud(root / "t.laz", [1.0], [1.0], [6.0], [5])
            with self.assertRaisesRegex(RuntimeError, "DTM product missing"):
                tile_denormalize_heights(
                    {"tile_id": "t", "file": str(src)},
                    _ctx(root / "s",
                         {"dtm_mosaic_path": str(root / "nope.tif")}),
                )

    def test_breaks_classified_chain(self) -> None:
        from lynceus.nodes.lidar.clean.denormalize_heights import (
            PROCESSING_SPECS,
        )

        # Classes may exist in the file, but their provenance is
        # unknown: the flag must not be re-emitted.
        self.assertNotIn("provides", PROCESSING_SPECS)
        self.assertNotIn("passes_flags", PROCESSING_SPECS)


class ReturnFilterTests(unittest.TestCase):
    @staticmethod
    def _pulses(point_format: int = 6):
        # 4 single pulses + 3 double pulses.
        ret, num = [], []
        for _ in range(4):
            ret.append(1)
            num.append(1)
        for _ in range(3):
            ret.extend([1, 2])
            num.extend([2, 2])
        n = len(ret)
        rng = np.random.default_rng(9)
        return (rng.uniform(0, 10, n), rng.uniform(0, 10, n),
                np.full(n, 400.0), np.full(n, 1, dtype=np.uint8),
                ret, num)

    def _run(self, mode: str, point_format: int = 6) -> int:
        from lynceus.nodes.lidar.clean.return_filter import tile_return_filter

        x, y, z, cls, ret, num = self._pulses(point_format)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = _write_cloud(root / "t.laz", x, y, z, cls,
                               point_format=point_format, ret=ret, num=num)
            out = tile_return_filter(
                {"tile_id": "t", "file": str(src)},
                _ctx(root / "s", {"return_mode": mode}),
            )
            return len(_read_z(Path(out["output"])))

    def test_first_and_last(self) -> None:
        self.assertEqual(self._run("first"), 7)
        self.assertEqual(self._run("last"), 7)

    def test_single_and_multiple(self) -> None:
        self.assertEqual(self._run("single"), 4)
        self.assertEqual(self._run("multiple"), 6)

    def test_legacy_dimensions(self) -> None:
        # PDRF 3 carries return_num / num_returns: same four answers.
        self.assertEqual(self._run("first", point_format=3), 7)
        self.assertEqual(self._run("last", point_format=3), 7)
        self.assertEqual(self._run("single", point_format=3), 4)
        self.assertEqual(self._run("multiple", point_format=3), 6)

    def test_unknown_mode_raises(self) -> None:
        from lynceus.nodes.lidar.clean.return_filter import tile_return_filter

        x, y, z, cls, ret, num = self._pulses()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = _write_cloud(root / "t.laz", x, y, z, cls,
                               ret=ret, num=num)
            with self.assertRaisesRegex(ValueError, "unknown return_mode"):
                tile_return_filter(
                    {"tile_id": "t", "file": str(src)},
                    _ctx(root / "s", {"return_mode": "sideways"}),
                )


class ClipTests(unittest.TestCase):
    SQUARE = "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))"

    def _run(self, wkt: str, extra: dict | None = None):
        from lynceus.nodes.lidar.clean.clip_polygon import tile_clip_polygon

        xs = [5.0, 20.0, 0.0, 5.0]  # inside, outside, boundary, inside
        ys = [5.0, 5.0, 0.0, 5.0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = _write_cloud(root / "t.laz", xs, ys,
                               [400.0] * 4, [2] * 4)
            cfg = {"polygon_wkt": wkt}
            if extra:
                cfg.update(extra)
            out = tile_clip_polygon({"tile_id": "t", "file": str(src)},
                                    _ctx(root / "s", cfg))
            got = _read_z(Path(out["output"]))
            return out, len(got)

    def test_inside_and_boundary_kept(self) -> None:
        _out, n = self._run(self.SQUARE)
        self.assertEqual(n, 3)

    def test_buffer_reaches_outside(self) -> None:
        _out, n = self._run(self.SQUARE, {"buffer_m": 15.0})
        self.assertEqual(n, 4)

    def test_empty_wkt_blocks(self) -> None:
        from lynceus.nodes.lidar.clean.clip_polygon import tile_clip_polygon

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = _write_cloud(root / "t.laz", [1.0], [1.0], [400.0], [2])
            with self.assertRaisesRegex(RuntimeError, "polygon_wkt is empty"):
                tile_clip_polygon(
                    {"tile_id": "t", "file": str(src)},
                    _ctx(root / "s", {"polygon_wkt": "  "}),
                )

    def test_invalid_wkt_blocks(self) -> None:
        from lynceus.nodes.lidar.clean.clip_polygon import tile_clip_polygon

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = _write_cloud(root / "t.laz", [1.0], [1.0], [400.0], [2])
            with self.assertRaisesRegex(RuntimeError, "not valid WKT"):
                tile_clip_polygon(
                    {"tile_id": "t", "file": str(src)},
                    _ctx(root / "s", {"polygon_wkt": "NOT A GEOMETRY ((("}),
                )


class RegistryTests(unittest.TestCase):
    def test_nodes_registered_with_ports(self) -> None:
        from lynceus.nodes.ports import PortType, port_metadata
        from lynceus.plugins.registry import manager

        manager.discover()
        for module_id in (NORMALIZE, DENORMALIZE, RETURNS, CLIP):
            with self.subTest(node=module_id):
                self.assertIsNotNone(manager.node_info(module_id))
                inputs, outputs = manager.load_ports(module_id)
                self.assertEqual(
                    len(inputs),
                    2 if module_id in (NORMALIZE, DENORMALIZE) else 1)
                self.assertEqual(len(outputs), 1)
                first = port_metadata(inputs[0])
                self.assertEqual(first.port_type, PortType.POINT_CLOUD)
                self.assertTrue(first.required)
        for module_id in (NORMALIZE, DENORMALIZE):
            with self.subTest(node=module_id):
                _inputs, _outputs = manager.load_ports(module_id)
                dtm = port_metadata(_inputs[1])
                self.assertEqual(dtm.port_type, PortType.DTM_MOSAIC)
                self.assertTrue(dtm.required)


if __name__ == "__main__":
    unittest.main()
