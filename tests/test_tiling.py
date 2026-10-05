"""Unit tests for lotes de tiles: batch grouping, per-batch scopes, the
per-loader tiling plan, and the declarative Tile/Segment nodes.

Stays on the stdlib + the engine modules (no Qt, no real runs): the heavy
readers are exercised by the real app and the ad-hoc e2e script.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import laspy
import numpy as np

from lynceus.nodes.ports import PortType
from lynceus.plugins.registry import manager
from lynceus.processing.controller import PipelineController
from lynceus.processing.steps import discover_node_capabilities
from lynceus.processing.stream_format import choose_intermediate_format
from lynceus.processing.tiler import (
    MANIFEST_VERSION,
    MEMORY_BUDGET_FLOOR_BYTES,
    LiDARTiler,
    _has_copc_vlr,
    default_memory_budget_bytes,
)

LOAD = "lynceus.nodes.lidar.source.load_las_laz"
CLASSIFY = "lynceus.nodes.lidar.terrain.classify_ground"
CONS = "lynceus.nodes.flow.consolidate"
PC = PortType.POINT_CLOUD


class CacheFreshnessTests(unittest.TestCase):
    """_cache_is_fresh rejects manifests with missing/resized tiles."""

    def _tiler(self, tmp: Path, source_size: int = 0) -> LiDARTiler:
        source = tmp / "source.las"
        source.write_bytes(b"0" * max(source_size, 1))
        tiler = LiDARTiler.__new__(LiDARTiler)
        tiler.source_path = source
        tiler.output_dir = tmp / "tiles"
        tiler.output_dir.mkdir(parents=True, exist_ok=True)
        tiler.output_format = "laz"
        tiler.tile_size = 100.0
        tiler.buffer_m = 10.0
        tiler.max_points = 0
        return tiler

    def _manifest(self, tmp: Path, tiler: LiDARTiler, tiles: list) -> None:
        st = tiler.source_path.stat()
        doc = {
            "manifest_version": MANIFEST_VERSION,
            "output_format": "laz",
            "tile_size_m": 100.0,
            "buffer_m": 10.0,
            "max_points_per_tile": 0,
            "source_size": st.st_size,
            "source_mtime": st.st_mtime,
            "tiles_generated": tiles,
        }
        (tiler.output_dir / "tiling_manifest.json").write_text(
            json.dumps(doc), encoding="utf-8"
        )

    def test_fresh_when_tiles_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tiler = self._tiler(Path(tmp))
            tile = tiler.output_dir / "t0.laz"
            tile.write_bytes(b"12345678")
            self._manifest(Path(tmp), tiler, [
                {"tile_id": "t0", "file": str(tile),
                 "file_size": tile.stat().st_size},
            ])
            self.assertTrue(tiler._cache_is_fresh())

    def test_missing_tile_invalidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tiler = self._tiler(Path(tmp))
            self._manifest(Path(tmp), tiler, [
                {"tile_id": "t0",
                 "file": str(tiler.output_dir / "t0.laz"), "file_size": 8},
            ])
            self.assertFalse(tiler._cache_is_fresh())

    def test_resized_tile_invalidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tiler = self._tiler(Path(tmp))
            tile = tiler.output_dir / "t0.laz"
            tile.write_bytes(b"12345678")
            self._manifest(Path(tmp), tiler, [
                {"tile_id": "t0", "file": str(tile), "file_size": 7},
            ])
            self.assertFalse(tiler._cache_is_fresh())

    def test_legacy_manifest_without_sizes_checks_existence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tiler = self._tiler(Path(tmp))
            tile = tiler.output_dir / "t0.laz"
            tile.write_bytes(b"12345678")
            self._manifest(Path(tmp), tiler, [
                {"tile_id": "t0", "file": str(tile)},
            ])
            self.assertTrue(tiler._cache_is_fresh())


class LoaderConfigTests(unittest.TestCase):
    def test_loader_registered_in_lidar_category(self) -> None:
        manager.discover()
        info = manager.node_info(LOAD)
        self.assertIsNotNone(info)
        self.assertEqual(info.category, "LiDAR")
        self.assertEqual(info.subcategory, "Source")

    def test_lidar_family_subcategories(self) -> None:
        manager.discover()
        cases = {
            "lynceus.nodes.lidar.clean.elevation_range": "Cleaning",
            "lynceus.nodes.lidar.terrain.classify_ground": "Terrain",
            "lynceus.nodes.lidar.analysis.grid_metrics": "Analysis",
            "lynceus.nodes.lidar.output.export_point_cloud": "Export",
            "lynceus.nodes.raster.input_raster": "Input",
            "lynceus.nodes.raster.ops.chm_difference": "Operations",
            "lynceus.nodes.table.concatenate_tables": "Concatenate",
        }
        for node_id, expected in cases.items():
            info = manager.node_info(node_id)
            self.assertIsNotNone(info)
            self.assertEqual(info.subcategory, expected)

    def test_loader_declares_tiling_and_segmentation_schema(self) -> None:
        schema = discover_node_capabilities(LOAD)["config_schema"]
        for key in ("tile_size_m", "buffer_m", "max_points_per_tile",
                    "max_points_per_part"):
            self.assertIn(key, schema)
        self.assertEqual(schema["tile_size_m"]["group"], "Tiling")
        self.assertEqual(schema["buffer_m"]["group"], "Tiling")
        self.assertEqual(schema["max_points_per_tile"]["group"], "Tiling")
        self.assertEqual(schema["max_points_per_part"]["group"], "Segmentation")
        self.assertEqual(schema["max_points_per_part"]["default"], 0)

    def test_loader_triggers_tiling(self) -> None:
        caps = discover_node_capabilities(LOAD)
        self.assertTrue(caps["triggers_tiling"])
        self.assertNotIn("tiling_node", caps)
        self.assertNotIn("segment_node", caps)


class LotesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ctrl = PipelineController()

    def _tiles(self, count: int, points: int = 30) -> list[dict]:
        return [
            {
                "tile_id": f"t_c{c:04d}_r0000",
                "row": 0,
                "col": c,
                "sub_row": 0,
                "sub_col": 0,
                "point_count": points,
            }
            for c in range(count)
        ]

    def test_build_batches_groups_contiguously_under_cap(self) -> None:
        batches = self.ctrl._build_batches(self._tiles(10, 30), 100)
        self.assertEqual([len(b) for b in batches], [3, 3, 3, 1])
        self.assertEqual(sum(len(b) for b in batches), 10)

    def test_build_batches_without_cap_returns_single_batch(self) -> None:
        self.assertEqual(len(self.ctrl._build_batches(self._tiles(4), 0)), 1)
        self.assertEqual(len(self.ctrl._build_batches(self._tiles(4), None)), 1)

    def test_build_batches_oversized_tile_forms_own_batch(self) -> None:
        tiles = [
            {"tile_id": "a", "col": 0, "row": 0, "point_count": 500},
            {"tile_id": "b", "col": 1, "row": 0, "point_count": 10},
        ]
        batches = self.ctrl._build_batches(tiles, 100)
        self.assertEqual([len(b) for b in batches], [1, 1])

    def test_batch_scope_is_content_and_index_sensitive(self) -> None:
        t1 = [{"tile_id": "x"}, {"tile_id": "y"}]
        t2 = [{"tile_id": "x"}, {"tile_id": "z"}]
        reversed_t1 = [{"tile_id": "y"}, {"tile_id": "x"}]
        self.assertEqual(
            self.ctrl._batch_scope(t1, 0), self.ctrl._batch_scope(reversed_t1, 0)
        )
        self.assertNotEqual(
            self.ctrl._batch_scope(t1, 0), self.ctrl._batch_scope(t2, 0)
        )
        self.assertNotEqual(
            self.ctrl._batch_scope(t1, 0), self.ctrl._batch_scope(t1, 1)
        )

    def test_tiling_plan_defaults_without_config(self) -> None:
        self.ctrl._modules = {"loader": LOAD, "classify": CLASSIFY}
        plan = self.ctrl._detect_tiling_plan(
            ["loader", "classify"],
            [("loader", "classify", PC, PC)],
            {},
        )
        self.assertEqual(plan, {"loader": {}})  # auto-fallback -> LiDARTiler defaults

    def test_tiling_plan_reads_loader_config(self) -> None:
        self.ctrl._modules = {"loader": LOAD, "classify": CLASSIFY}
        plan = self.ctrl._detect_tiling_plan(
            ["loader", "classify"],
            [("loader", "classify", PC, PC)],
            {"loader": {"tile_size_m": 50.0, "buffer_m": 5.0,
                        "max_points_per_tile": 0}},
        )
        self.assertEqual(
            plan["loader"],
            {"tile_size_m": 50.0, "buffer_m": 5.0, "max_points_per_tile": 0},
        )

    def test_segment_cap_defaults_to_off_without_config(self) -> None:
        # A fresh loader (Inspector never opened) ships no per-instance config;
        # segmentation falls back to the schema default 0 (single pass).
        self.ctrl._modules = {"loader": LOAD}
        cap = self.ctrl._resolve_segment_cap({}, "loader")
        self.assertEqual(cap, 0)
        tiles = [{"tile_id": f"t{i}", "point_count": 60_000_000} for i in range(4)]
        batches = self.ctrl._build_batches(tiles, cap)
        self.assertEqual(batches, [tiles])

    def test_segment_cap_reads_loader_config(self) -> None:
        self.ctrl._modules = {"loader": LOAD}
        cap = self.ctrl._resolve_segment_cap(
            {"loader": {"max_points_per_part": 5_000_000}}, "loader"
        )
        self.assertEqual(cap, 5_000_000)
        tiles = [{"tile_id": f"t{i}", "point_count": 30_000_000} for i in range(4)]
        batches = self.ctrl._build_batches(tiles, cap)
        self.assertGreater(len(batches), 1)


class TilerMemoryBudgetTests(unittest.TestCase):
    """P1a: the tiler budget buffers BYTES, not points.

    A point-based budget (the historical 1e9 points) buffered the whole
    dataset in RAM (~30 GB packed) before flushing, swapping machines with
    << 30 GB and stalling large tilings past ~60%. These tests pin the
    byte-based, RAM-adaptive behavior."""

    def test_default_budget_is_ram_adaptive_with_floor(self) -> None:
        self.assertEqual(
            default_memory_budget_bytes(1_073_741_824), MEMORY_BUDGET_FLOOR_BYTES
        )
        eight_gb = 8 * 1_073_741_824
        self.assertEqual(default_memory_budget_bytes(eight_gb), eight_gb // 4)

    def test_byte_budget_flushes_during_run_not_only_at_end(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self._write_cloud(root / "src.laz", count=60_000)
            output = root / "tiles"

            tiler = LiDARTiler(
                source,
                output,
                tile_size=100.0,
                buffer_m=0.0,
                max_points_per_tile=0,
                chunk_size=20_000,
                buffer_flush_size=10_000_000,  # per-tile threshold never fires
                memory_budget_bytes=8_192,  # tiny: forces the byte-budget loop
            )

            flushed: list[tuple[int, int]] = []
            original = tiler._async_write_laz

            def tracked(tid: int, mem_buf: dict, io_futures: list) -> None:
                flushed.append(
                    (tid, sum(int(a.shape[0]) for a in mem_buf[tid]))
                )
                original(tid, mem_buf, io_futures)

            tiler._async_write_laz = tracked  # type: ignore[method-assign]

            result = tiler.run(cancel_flag=None)

            total = sum(c["point_count"] for c in result.tiles_generated)
            self.assertEqual(total, 60_000)
            # 4 base tiles, but with a tiny byte budget the biggest bucket is
            # flushed repeatedly mid-run -> strictly more flushes than tiles.
            self.assertGreater(len(flushed), len(result.tiles_generated))

    def test_default_budget_bytes_is_positive(self) -> None:
        self.assertGreaterEqual(default_memory_budget_bytes(), MEMORY_BUDGET_FLOOR_BYTES)

    @staticmethod
    def _write_cloud(path: Path, count: int) -> Path:
        rng = np.random.default_rng(7)
        header = laspy.LasHeader(point_format=0, version="1.2")
        header.scales = np.array([0.01, 0.01, 0.01])
        header.offsets = np.zeros(3)
        record = laspy.ScaleAwarePointRecord.zeros(
            count,
            point_format=header.point_format,
            scales=header.scales,
            offsets=header.offsets,
        )
        record.x = rng.uniform(0, 200, count)
        record.y = rng.uniform(0, 200, count)
        record.z = np.zeros(count)
        with laspy.open(str(path), mode="w", header=header) as writer:
            writer.write_points(record)
        return path


class ParallelDecodeDeterminismTests(unittest.TestCase):
    """Task C: parallel decode must be byte-deterministic.

    The full-file passes decode source spans on a bounded worker pool but
    consume them strictly in file order, so every tile (bytes + manifest +
    IDs) must equal a serial run and be independent of worker count/window.
    Pinned here so the parallel path can never drift output across machines."""

    def _run(
        self,
        source: Path,
        out: Path,
        *,
        serial: bool = False,
        workers: int | None = None,
        window: int | None = None,
        max_points: int = 0,
    ) -> object:
        tiler = LiDARTiler(
            source,
            out,
            tile_size=100.0,
            buffer_m=0.0,
            max_points_per_tile=max_points,
            chunk_size=20_000,
        )
        if serial:
            tiler._use_parallel_decode = False
        if workers is not None:
            tiler._decode_workers = workers
            tiler._decode_window = min(window or workers, tiler._decode_window)
        result = tiler.run(cancel_flag=None)
        return result

    @staticmethod
    def _hashes(result) -> dict[str, tuple[str, int]]:
        out = {}
        for tile in result.tiles_generated:
            fp = Path(tile["file"])
            out[tile["tile_id"]] = (
                hashlib.sha1(fp.read_bytes()).hexdigest(),
                int(tile["point_count"]),
            )
        return out

    def _assert_equal(self, a, b) -> None:
        self.assertEqual(
            [(t["tile_id"], t["point_count"]) for t in a.tiles_generated],
            [(t["tile_id"], t["point_count"]) for t in b.tiles_generated],
        )
        self.assertEqual(self._hashes(a), self._hashes(b))

    def test_parallel_equals_serial(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = TilerMemoryBudgetTests._write_cloud(root / "src.laz", count=60_000)
            serial = self._run(source, root / "serial", serial=True, workers=2)
            parallel = self._run(source, root / "parallel", workers=2)
            self._assert_equal(serial, parallel)

    def test_adaptive_adivision_parallel_equals_serial(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = TilerMemoryBudgetTests._write_cloud(root / "src.laz", count=60_000)
            serial = self._run(
                source, root / "serial", serial=True, workers=2, max_points=2_000
            )
            parallel = self._run(
                source, root / "parallel", workers=2, max_points=2_000
            )
            self._assert_equal(serial, parallel)

    def test_worker_count_and_window_invariant(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = TilerMemoryBudgetTests._write_cloud(root / "src.laz", count=60_000)
            light = self._run(source, root / "w2", workers=2, window=1)
            heavy = self._run(source, root / "w8", workers=8, window=4)
            self._assert_equal(light, heavy)

    def test_worker_count_invariant_with_subdivision(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = TilerMemoryBudgetTests._write_cloud(root / "src.laz", count=60_000)
            light = self._run(source, root / "w2", workers=2, window=1, max_points=2_000)
            heavy = self._run(source, root / "w8", workers=8, window=4, max_points=2_000)
            self._assert_equal(light, heavy)


class IntermediateFormatTests(unittest.TestCase):
    """P2a: `las` only for datasets huge enough to beat the codec.

    The regression that must not return: a 131M-point cloud on any machine
    stays on `laz` so it reuses the legacy tile cache and keeps its speed."""

    def test_131m_cloud_stays_laz(self) -> None:
        self.assertEqual(choose_intermediate_format(total_points=131_000_000), "laz")

    def test_huge_cloud_uses_las(self) -> None:
        self.assertEqual(choose_intermediate_format(total_points=200_000_000), "las")
        self.assertEqual(choose_intermediate_format(total_points=1_000_000_000), "las")

    def test_unknown_points_defaults_laz(self) -> None:
        self.assertEqual(choose_intermediate_format(total_points=None), "laz")
        self.assertEqual(choose_intermediate_format(), "laz")

    def test_tiling_writes_selected_format(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = TilerMemoryBudgetTests._write_cloud(root / "src.laz", count=12_000)
            for fmt in ("las", "laz"):
                output = root / f"tiles_{fmt}"
                tiler = LiDARTiler(
                    source,
                    output,
                    tile_size=100.0,
                    buffer_m=0.0,
                    output_format=fmt,
                )
                result = tiler.run(cancel_flag=None)
                self.assertGreater(len(result.tiles_generated), 0)
                for tile in result.tiles_generated:
                    self.assertTrue(tile["file"].endswith(f".{fmt}"))

    def test_legacy_manifest_reused_for_laz_not_las(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = TilerMemoryBudgetTests._write_cloud(root / "src.laz", count=12_000)
            tiles_dir = root / "tiles"
            tiles_dir.mkdir(parents=True, exist_ok=True)
            st = source.stat()
            # Legacy manifest: predates the output_format key.
            (tiles_dir / "tiling_manifest.json").write_text(
                json.dumps(
                    {
                        "manifest_version": MANIFEST_VERSION,
                        "source_size": st.st_size,
                        "source_mtime": st.st_mtime,
                        "tile_size_m": 100.0,
                        "buffer_m": 0.0,
                        "max_points_per_tile": 0,
                    }
                )
            )
            legacy_laz = LiDARTiler(
                source, tiles_dir, tile_size=100.0, buffer_m=0.0, output_format="laz"
            )
            legacy_las = LiDARTiler(
                source, tiles_dir, tile_size=100.0, buffer_m=0.0, output_format="las"
            )
            self.assertTrue(legacy_laz._cache_is_fresh())
            self.assertFalse(legacy_las._cache_is_fresh())

    def test_cached_manifest_roundtrip_with_format(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = TilerMemoryBudgetTests._write_cloud(root / "src.laz", count=12_000)
            tiles_dir = root / "tiles"
            first = LiDARTiler(
                source, tiles_dir, tile_size=100.0, buffer_m=0.0, output_format="laz"
            )
            r1 = first.run()
            self.assertGreater(len(r1.tiles_generated), 0)
            self.assertTrue(all(t["file"].endswith(".laz") for t in r1.tiles_generated))
            # A re-run on the same dir must hydrate the cached result in place
            # (same keys round-trip through TilingResult) instead of re-tiling.
            again = LiDARTiler(
                source, tiles_dir, tile_size=100.0, buffer_m=0.0, output_format="laz"
            )
            self.assertTrue(again._cache_is_fresh())
            r2 = again.run()
            self.assertEqual(len(r2.tiles_generated), len(r1.tiles_generated))
            self.assertEqual(r2.output_format, "laz")


class SubdivisionTests(unittest.TestCase):
    def test_dense_tile_subdivides_under_point_cap(self) -> None:
        """Pass A must actually run: a dense base tile splits when capped.

        Regression test: the density-histogram body was once stranded
        inside the cancel branch (unreachable), so levels stayed flat and
        no subdivision ever happened.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = TilerMemoryBudgetTests._write_cloud(root / "src.laz", count=60_000)
            output = root / "tiles"
            tiler = LiDARTiler(
                source,
                output,
                tile_size=200.0,
                buffer_m=0.0,
                max_points_per_tile=10_000,
                chunk_size=20_000,
            )
            result = tiler.run(cancel_flag=None)
            total = sum(c["point_count"] for c in result.tiles_generated)
            self.assertEqual(total, 60_000)
            # One 200 m base tile holding 60K points must split under a
            # 10K cap (best-effort bounded by MAX_SUB_DIV).
            self.assertGreater(len(result.tiles_generated), 1)


class CopcVlrTests(unittest.TestCase):
    """COPC inputs must tile without crashing on the copc VLR."""

    def test_has_copc_vlr(self) -> None:
        self.assertTrue(_has_copc_vlr([{"user_id": "COPC"}]))
        self.assertTrue(_has_copc_vlr([{"user_id": "copc"}]))
        self.assertFalse(_has_copc_vlr([{"user_id": "laszip encoded"}]))
        self.assertFalse(_has_copc_vlr([]))
        self.assertFalse(_has_copc_vlr(None))

    def test_copy_product_vlrs_strips_copc(self) -> None:
        from types import SimpleNamespace

        from lynceus.processing.provenance import copy_product_vlrs

        src = SimpleNamespace(
            vlrs=[
                SimpleNamespace(user_id="copc"),
                SimpleNamespace(user_id="LASF_Projection"),
            ]
        )
        dst = laspy.LasHeader(point_format=6, version="1.4")
        copy_product_vlrs(dst, src)
        self.assertEqual([v.user_id for v in dst.vlrs], ["LASF_Projection"])

    def test_tiles_cloud_with_copc_vlr(self) -> None:
        n = 200
        header = laspy.LasHeader(point_format=6, version="1.4")
        header.scales = np.array([0.01, 0.01, 0.01])
        header.offsets = np.array([0.0, 0.0, 0.0])
        header.vlrs.append(laspy.VLR(
            user_id="copc", record_id=1, description="",
            record_data=b"\x00",
        ))
        record = laspy.ScaleAwarePointRecord.zeros(
            n, point_format=header.point_format,
            scales=header.scales, offsets=header.offsets,
        )
        record.x = np.linspace(0.0, 99.0, n)
        record.y = np.linspace(0.0, 99.0, n)
        record.z = np.linspace(400.0, 410.0, n)
        with tempfile.TemporaryDirectory(prefix="lynceus_copc_") as tmp:
            src = os.path.join(tmp, "src.laz")
            with laspy.open(src, mode="w", header=header) as writer:
                writer.write_points(record)
            with laspy.open(src) as reader:
                self.assertTrue(_has_copc_vlr(list(reader.header.vlrs)))
            tiler = LiDARTiler(
                src, tmp, tile_size=100.0, buffer_m=0.0,
                max_points_per_tile=0,
            )
            result = tiler.run(cancel_flag=None)
            total = sum(c["point_count"] for c in result.tiles_generated)
            self.assertGreater(len(result.tiles_generated), 0)
            self.assertEqual(total, n)

    def test_tiles_real_copc_file(self) -> None:
        candidates = sorted(glob.glob(os.path.join("input", "*.copc.laz")))
        if not candidates:
            self.skipTest("no .copc.laz sample in input/")
        path = candidates[0]
        with laspy.open(path) as reader:
            self.assertTrue(_has_copc_vlr(list(reader.header.vlrs)))
            expected = reader.header.point_count
        with tempfile.TemporaryDirectory(prefix="lynceus_copc_") as tmp:
            tiler = LiDARTiler(
                path, tmp, tile_size=100.0, buffer_m=10.0,
                max_points_per_tile=0,
            )
            result = tiler.run(cancel_flag=None)
            total = sum(c["point_count"] for c in result.tiles_generated)
            self.assertGreater(len(result.tiles_generated), 0)
            self.assertEqual(total, expected)

    def test_copc_header_forces_serial_decode(self) -> None:
        # Hermetic: a plain LAZ carrying a copc VLR trips the serial gate
        # (the gate keys off the VLR; real COPC layouts break seek+read
        # with silent zero-filled records, see next test).
        n = 200
        header = laspy.LasHeader(point_format=6, version="1.4")
        header.scales = np.array([0.01, 0.01, 0.01])
        header.offsets = np.zeros(3)
        header.vlrs.append(laspy.VLR(
            user_id="copc", record_id=1, description="",
            record_data=b"\x00",
        ))
        record = laspy.ScaleAwarePointRecord.zeros(
            n, point_format=header.point_format,
            scales=header.scales, offsets=header.offsets,
        )
        record.x = np.linspace(0.0, 99.0, n)
        record.y = np.linspace(0.0, 99.0, n)
        record.z = np.linspace(400.0, 410.0, n)
        with tempfile.TemporaryDirectory(prefix="lynceus_copc_gate_") as tmp:
            src = os.path.join(tmp, "src.laz")
            with laspy.open(src, mode="w", header=header) as writer:
                writer.write_points(record)
            tiler = LiDARTiler(
                src, tmp, tile_size=100.0, buffer_m=0.0,
                max_points_per_tile=0,
            )
            self.assertTrue(tiler._use_parallel_decode)
            tiler._read_grid()
            self.assertTrue(tiler._is_copc)
            self.assertFalse(tiler._use_parallel_decode)

    def test_real_copc_first_spans_decode_nonzero(self) -> None:
        # Behavioral (needs the sample): first spans through _iter_chunks
        # must decode real coordinates. Pre-fix the parallel seek path
        # returned zero-filled records on COPC here.
        import itertools

        candidates = sorted(glob.glob(os.path.join("input", "*.copc.laz")))
        if not candidates:
            self.skipTest("no .copc.laz sample in input/")
        with tempfile.TemporaryDirectory(prefix="lynceus_copc_span_") as tmp:
            tiler = LiDARTiler(
                candidates[0], tmp, tile_size=100.0, buffer_m=10.0,
                max_points_per_tile=0, chunk_size=50_000,
            )
            tiler._read_grid()
            self.assertTrue(tiler._is_copc)
            chunks = list(itertools.islice(tiler._iter_chunks(None), 3))
            self.assertEqual(len(chunks), 3)
            for chunk in chunks:
                x = np.asarray(chunk.x)
                self.assertGreater(len(x), 0)
                self.assertGreater(float(np.abs(x).max()), 0.0)


if __name__ == "__main__":
    unittest.main()