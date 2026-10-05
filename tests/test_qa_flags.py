# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Tests for vendor QA flags (ASPRS 1.4,Informe LiDAR normative).

Withheld points are auto-excluded from surfaces/metrics/cleaning (no
opt-out); key points survive decimation; overlap/withheld fractions
surface as loader warnings and quality-report counts. Stdlib +
engine/domain modules only (no Qt).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import laspy
import numpy as np

from lynceus.nodes._point_source import quality_mask, run_clean
from lynceus.processing.controller import PipelineController

MODULE = "test.qa_flags"


def _write_cloud(path: Path, n: int = 200, point_format: int = 6,
                 withheld: np.ndarray | None = None,
                 key_point: np.ndarray | None = None,
                 overlap: np.ndarray | None = None,
                 synthetic: np.ndarray | None = None) -> Path:
    header = laspy.LasHeader(point_format=point_format, version="1.4")
    header.scales = np.array([0.01, 0.01, 0.01])
    header.offsets = np.zeros(3)
    record = laspy.ScaleAwarePointRecord.zeros(
        n,
        point_format=header.point_format,
        scales=header.scales,
        offsets=header.offsets,
    )
    record.x = np.linspace(0.0, 99.0, n)
    record.y = np.zeros(n)
    record.z = np.full(n, 10.0)
    record.classification = np.full(n, 5, dtype=np.uint8)
    if withheld is not None:
        record.withheld = np.asarray(withheld)
    if key_point is not None:
        record.key_point = np.asarray(key_point)
    if overlap is not None:
        record.overlap = np.asarray(overlap)
    if synthetic is not None:
        record.synthetic = np.asarray(synthetic)
    with laspy.open(str(path), mode="w", header=header) as writer:
        writer.write_points(record)
    return path


def _read_record(path: str):
    from lynceus.processing.tiler import _laz_backend

    with laspy.open(path, laz_backend=_laz_backend()) as reader:
        cloud, header = [], None
        for chunk in reader.chunk_iterator(2_000_000):
            cloud.append(np.asanyarray(chunk.array))
        header = reader.header
    import laspy as _laspy

    record = _laspy.ScaleAwarePointRecord(
        np.concatenate(cloud), header.point_format,
        scales=header.scales, offsets=header.offsets,
    )
    return record


def _classes(path: str) -> dict[int, int]:
    from lynceus.processing.tiler import _laz_backend

    hist = np.zeros(256, dtype=np.int64)
    with laspy.open(path, laz_backend=_laz_backend()) as reader:
        for chunk in reader.chunk_iterator(2_000_000):
            hist += np.bincount(
                np.asarray(chunk.classification), minlength=256
            )
    return {int(k): int(v) for k, v in enumerate(hist) if v}


class QualityMaskTests(unittest.TestCase):
    def test_withheld_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = _write_cloud(
                Path(tmp) / "s.laz",
                withheld=np.array([True] * 30 + [False] * 170),
            )
            mask = quality_mask(_read_record(str(src)))
            self.assertIsNotNone(mask)
            self.assertEqual(int((~mask).sum()), 30)
            self.assertEqual(int(mask.sum()), 170)

    def test_absent_dimension_returns_none(self) -> None:
        # laspy surfaces flag dims (as zeros) even on legacy PDRF 3, so a
        # real record masks everything as kept; the None path stays as a
        # defensive guard for foreign record types.
        with tempfile.TemporaryDirectory() as tmp:
            src = _write_cloud(Path(tmp) / "s.laz", point_format=3)
            mask = quality_mask(_read_record(str(src)))
            self.assertTrue(bool(np.asarray(mask).all()))

    def test_run_clean_drops_withheld_despite_keep_all(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = _write_cloud(
                root / "s.laz",
                withheld=np.array([True] * 40 + [False] * 160),
            )
            tile = {"tile_id": "t_c0000_r0000", "file": str(src)}
            ctx = {"session_dir": str(root / "s"),
                   "intermediate_ext": "laz", "action": "remove"}
            res = run_clean(
                tile, ctx, MODULE, "clean_q",
                lambda record: np.ones(len(record), dtype=bool),
            )
            self.assertEqual(res["removed"], 40)
            with laspy.open(res["output"]) as reader:
                self.assertEqual(int(reader.header.point_count), 160)


class KeyPointTests(unittest.TestCase):
    def test_decimate_keeps_key_points(self) -> None:
        from lynceus.nodes.lidar.clean.decimate import _keep_mask

        with tempfile.TemporaryDirectory() as tmp:
            # nth step 4 keeps indices 0,4,8,...; key points at 1,2,3,5,6,7
            # would all be dropped without the protection.
            key = np.zeros(20, dtype=bool)
            key[[1, 2, 3, 5, 6, 7, 9, 11]] = True
            src = _write_cloud(Path(tmp) / "s.laz", n=20, key_point=key)
            rec = _read_record(str(src))
            mask = _keep_mask(rec, {"mode": "ratio", "sampling_ratio": 25.0})
            kept_key = int((mask & key).sum())
            self.assertEqual(kept_key, int(key.sum()))
            # Regular points still thin to the target count plus key extras.
            self.assertGreaterEqual(int(mask.sum()), int(key.sum()))

    def test_decimate_legacy_format_unchanged(self) -> None:
        from lynceus.nodes.lidar.clean.decimate import _keep_mask

        with tempfile.TemporaryDirectory() as tmp:
            src = _write_cloud(Path(tmp) / "s.laz", n=20, point_format=1)
            rec = _read_record(str(src))
            mask = _keep_mask(rec, {"mode": "ratio", "sampling_ratio": 25.0})
            self.assertEqual(int(mask.sum()), 5)


class FlagWarningTests(unittest.TestCase):
    def test_thresholds(self) -> None:
        fn = PipelineController._flag_warnings
        self.assertEqual(fn("f.laz", 0, {"withheld": 5}), [])
        self.assertEqual(fn("f.laz", 100, {}), [])
        below = fn("f.laz", 10_000, {"withheld": 50, "overlap": 50})
        self.assertEqual(below, [])
        rows = fn("f.laz", 1_000, {"withheld": 30, "overlap": 5})
        self.assertEqual(len(rows), 1)
        self.assertIn("withheld", rows[0])
        self.assertIn("f.laz", rows[0])
        both = fn("f.laz", 1_000, {"withheld": 30, "overlap": 40})
        self.assertEqual(len(both), 2)


class TilerFlagTests(unittest.TestCase):
    def test_flag_census(self) -> None:
        from lynceus.processing.tiler import LiDARTiler

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            n = 100
            withheld = np.zeros(n, dtype=bool)
            withheld[:7] = True
            overlap = np.zeros(n, dtype=bool)
            overlap[10:13] = True
            key = np.zeros(n, dtype=bool)
            key[20] = True
            synth = np.zeros(n, dtype=bool)
            synth[30:32] = True
            src = _write_cloud(
                root / "s.laz", n=n, withheld=withheld, overlap=overlap,
                key_point=key, synthetic=synth,
            )
            tiler = LiDARTiler(
                src, root / "tiles", tile_size=100.0, buffer_m=0.0,
                max_points_per_tile=0,
            )
            result = tiler.run(cancel_flag=None)
            self.assertEqual(
                result.flag_counts,
                {"withheld": 7, "overlap": 3, "synthetic": 2,
                 "key_point": 1},
            )

    def test_legacy_format_reports_zeros(self) -> None:
        from lynceus.processing.tiler import LiDARTiler

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = _write_cloud(root / "s.laz", n=100, point_format=1)
            tiler = LiDARTiler(
                src, root / "tiles", tile_size=100.0, buffer_m=0.0,
                max_points_per_tile=0,
            )
            result = tiler.run(cancel_flag=None)
            # Whatever laspy surfaces on legacy formats counts truthfully
            # (zeros: no point can carry those flags).
            self.assertTrue(
                set(result.flag_counts) <= {
                    "withheld", "overlap", "synthetic", "key_point"}
            )
            self.assertTrue(
                all(v == 0 for v in result.flag_counts.values())
            )


class QualityReportFlagsTests(unittest.TestCase):
    def test_point_flags_land_in_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctrl = PipelineController()
            ctrl._run_flag_counts = {
                "loader": {
                    "total_points": 1_000,
                    "flags": {"withheld": 30, "overlap": 5},
                }
            }
            verdict = ctrl._write_quality_report(
                Path(tmp), [], {}, [], 0, [], {}, {},
            )
            report = json.loads(
                (Path(tmp) / "quality_report.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                report["counts"]["point_flags"],
                {"total_points": 1_000, "withheld": 30, "overlap": 5},
            )
            self.assertIn(verdict, ("PASS", "FAIL"))

    def test_absent_flags_omit_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctrl = PipelineController()
            ctrl._run_flag_counts = {}
            ctrl._write_quality_report(
                Path(tmp), [], {}, [], 0, [], {}, {},
            )
            report = json.loads(
                (Path(tmp) / "quality_report.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertNotIn("point_flags", report["counts"])


class SurfaceExclusionTests(unittest.TestCase):
    """Withheld points stay out of terrain surfaces and metric counts."""

    def _cloud(self, path: Path, xs, ys, zs, cls, withheld=None,
               intensity=None, gps_bit: bool = False,
               source_id: int = 0) -> Path:
        from lynceus.processing.tiler import _laz_backend

        n = len(xs)
        header = laspy.LasHeader(point_format=6, version="1.4")
        header.scales = np.array([0.01, 0.01, 0.01])
        header.offsets = np.zeros(3)
        rec = laspy.ScaleAwarePointRecord.zeros(
            n, point_format=header.point_format,
            scales=header.scales, offsets=header.offsets,
        )
        rec.x = np.asarray(xs, dtype=float)
        rec.y = np.asarray(ys, dtype=float)
        rec.z = np.asarray(zs, dtype=float)
        rec.classification = np.asarray(cls, dtype=np.uint8)
        if gps_bit:
            header.global_encoding.gps_time_type = 1
        if source_id:
            header.file_source_id = source_id
        if withheld is not None:
            rec.withheld = np.asarray(withheld, dtype=bool)
        if intensity is not None:
            rec.intensity = np.asarray(intensity, dtype=np.uint16)
        with laspy.open(str(path), mode="w", header=header,
                        laz_backend=_laz_backend()) as writer:
            writer.write_points(rec)
        return path

    def _tile(self, name: str, path: Path) -> dict:
        return {
            "tile_id": "t_c0000_r0000", "file": str(path),
            "core_x_min": 0.0, "core_x_max": 100.0,
            "core_y_min": 0.0, "core_y_max": 100.0,
        }

    def _ctx(self, session: Path, **kw) -> dict:
        ctx = {"session_dir": str(session), "resolution": 10.0}
        ctx.update(kw)
        return ctx

    def _read(self, path: str):
        from lynceus.processing.raster import read_geotiff

        arr, _, _ = read_geotiff(path)
        return arr

    def test_dtm_ignores_withheld_ground(self) -> None:
        from lynceus.nodes.lidar.terrain.generate_dtm import (
            tile_generate_dtm,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Ground z=100 all withheld; veg z=110 kept. Pre-fix the DTM
            # read 100.0 from the withheld ground.
            src = self._cloud(
                root / "s.laz", [1.0] * 20, [1.0] * 20,
                [100.0] * 10 + [110.0] * 10,
                [2] * 10 + [5] * 10,
                withheld=[True] * 10 + [False] * 10,
            )
            out = tile_generate_dtm(
                self._tile("t", src), self._ctx(root / "s")
            )
            self.assertTrue(out.get("warnings"))
            arr = self._read(out["output"])
            valid = arr[arr != -9999.0]
            self.assertEqual(len(valid), 1)
            self.assertAlmostEqual(float(valid[0]), 110.0, places=1)

    def test_dsm_ignores_withheld_spike(self) -> None:
        from lynceus.nodes.lidar.terrain.generate_dsm import (
            tile_generate_dsm,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = self._cloud(
                root / "s.laz", [1.0] * 11, [1.0] * 11,
                [110.0] * 10 + [200.0],
                [5] * 10 + [18],
                withheld=[False] * 10 + [True],
            )
            out = tile_generate_dsm(
                self._tile("t", src), self._ctx(root / "s")
            )
            arr = self._read(out["output"])
            valid = arr[arr != -9999.0]
            self.assertEqual(len(valid), 1)
            self.assertAlmostEqual(float(valid[0]), 110.0, places=1)

    def test_grid_metrics_ignores_withheld(self) -> None:
        import geopandas as gpd

        from lynceus.nodes.lidar.analysis.grid_metrics import (
            tile_grid_metrics,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = self._cloud(
                root / "s.laz", [1.0] * 4, [1.0] * 4,
                [100.0] * 4, [5] * 4,
                withheld=[True, True, False, False],
            )
            out = tile_grid_metrics(
                self._tile("t", src), self._ctx(root / "s")
            )
            frame = gpd.read_file(out["output"])
            self.assertEqual(int(frame.iloc[0]["n"]), 2)

    def test_canopy_degrades_without_usable_ground(self) -> None:
        from lynceus.nodes.lidar.analysis.canopy_penetration import (
            tile_canopy_penetration,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # All ground withheld: no usable ground -> NODATA, not 0.5.
            src = self._cloud(
                root / "s.laz", [1.0] * 8, [1.0] * 8,
                [100.0] * 4 + [110.0] * 4,
                [2] * 4 + [5] * 4,
                withheld=[True] * 4 + [False] * 4,
            )
            out = tile_canopy_penetration(
                self._tile("t", src), self._ctx(root / "s")
            )
            self.assertTrue(out.get("warnings"))
            arr = self._read(out["output"])
            self.assertTrue(bool((arr == -9999.0).all()))

    def test_intensity_ignores_withheld_spike(self) -> None:
        from lynceus.nodes.lidar.analysis.intensity_ortho import (
            tile_intensity_ortho,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = self._cloud(
                root / "s.laz", [1.0] * 5, [1.0] * 5,
                [100.0] * 5, [5] * 5,
                withheld=[False] * 4 + [True],
                intensity=[100] * 4 + [10000],
            )
            out = tile_intensity_ortho(
                self._tile("t", src),
                self._ctx(root / "s", normalize_scan_angle=False),
            )
            arr = self._read(out["output"])
            valid = arr[arr != -9999.0]
            self.assertEqual(len(valid), 1)
            self.assertAlmostEqual(float(valid[0]), 100.0, places=1)

    def test_classify_never_labels_withheld_ground(self) -> None:
        from lynceus.nodes.lidar.terrain._classify_base import (
            run_ground_classification,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = self._cloud(
                root / "s.laz", [1.0] * 6, [1.0] * 6,
                [100.0, 101.0, 200.0, 110.0, 111.0, 112.0],
                [2, 2, 5, 5, 5, 5],
                withheld=[False, False, True, False, False, False],
                gps_bit=True, source_id=6,
            )
            tile = {"tile_id": "t_c0000_r0000", "file": str(src)}
            ctx = {"session_dir": str(root / "s")}
            out = run_ground_classification(
                tile, ctx, lambda x, y, z: np.ones(x.shape, dtype=bool),
                node_id="test.classify",
            )
            got = _classes(out["output"])
            # The withheld spike keeps class 5; everything else is ground.
            self.assertEqual(got.get(5), 1)
            self.assertEqual(got.get(2), 5)
            with laspy.open(out["output"]) as reader:
                hdr = reader.header
            self.assertEqual(int(hdr.global_encoding.gps_time_type), 1)
            self.assertEqual(hdr.file_source_id, 6)


class DensityCensusTests(unittest.TestCase):
    def test_tiler_counts_first_returns_and_histograms(self) -> None:
        from lynceus.processing.tiler import LiDARTiler

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            n = 100
            header = laspy.LasHeader(point_format=6, version="1.4")
            header.scales = np.array([0.01, 0.01, 0.01])
            header.offsets = np.zeros(3)
            rec = laspy.ScaleAwarePointRecord.zeros(
                n, point_format=header.point_format,
                scales=header.scales, offsets=header.offsets,
            )
            rec.x = np.linspace(0.0, 99.0, n)
            rec.y = np.zeros(n)
            rec.z = np.full(n, 10.0)
            rec.classification = np.array(
                [2] * 30 + [5] * 70, dtype=np.uint8
            )
            rec.return_number = np.array(
                [1] * 60 + [2] * 40, dtype=np.uint8
            )
            rec.number_of_returns = np.array(
                [1] * 60 + [2] * 40, dtype=np.uint8
            )
            with laspy.open(str(root / "s.laz"), mode="w",
                            header=header) as writer:
                writer.write_points(rec)
            tiler = LiDARTiler(
                root / "s.laz", root / "tiles", tile_size=100.0,
                buffer_m=0.0, max_points_per_tile=0,
            )
            result = tiler.run(cancel_flag=None)
            self.assertEqual(
                result.density_counts["first_returns"], 60
            )
            self.assertEqual(
                result.density_counts["class_histogram"], {"2": 30, "5": 70}
            )
            self.assertEqual(
                result.density_counts["return_histogram"],
                {"1": 60, "2": 40},
            )

    def test_old_manifest_loads_without_new_fields(self) -> None:
        from lynceus.processing.tiler import TilingResult

        result = TilingResult(
            source_file="s", output_dir="o", crs="c", total_points=10,
            tile_size_m=100.0,
        )
        self.assertEqual(result.flag_counts, {})
        self.assertEqual(result.density_counts, {})


class IngestWarningTests(unittest.TestCase):
    def test_cases(self) -> None:
        from lynceus.processing.controller import PipelineController

        fn = PipelineController._ingest_warnings
        self.assertEqual(len(fn("s.laz", "Unknown", 8, "1.4")), 1)
        self.assertIn("no CRS", fn("s.laz", "Unknown", 8, "1.4")[0])
        self.assertEqual(fn("s.laz", "EPSG:25831", 8, "1.4"), [])
        legacy = fn("s.laz", "EPSG:25831", 3, "1.2")
        self.assertEqual(len(legacy), 1)
        self.assertIn("PDRF 3", legacy[0])
        self.assertEqual(fn("s.laz", None, None, None), [
            msg for msg in fn("s.laz", None, None, None)
            if "no CRS" in msg
        ])
        self.assertEqual(fn("s.laz", "", 8, "1.4"), fn("s.laz", None, 8, "1.4"))


class QualityDensityTests(unittest.TestCase):
    def test_density_block_math(self) -> None:
        from lynceus.processing.controller import PipelineController
        from lynceus.processing.tiler import TilingResult

        with tempfile.TemporaryDirectory() as tmp:
            ctrl = PipelineController()
            ctrl._tilings = {
                "loader": TilingResult(
                    source_file="s", output_dir="o", crs="c",
                    total_points=100, tile_size_m=100.0,
                    source_meta={"mins": [0, 0, 0], "maxs": [10, 20, 5]},
                )
            }
            ctrl._run_density_counts = {
                "loader": {
                    "first_returns": 60,
                    "class_histogram": {"2": 30, "5": 70},
                    "return_histogram": {"1": 60, "2": 40},
                }
            }
            ctrl._write_quality_report(
                Path(tmp), [], {}, [], 0, [], {}, {},
            )
            report = json.loads(
                (Path(tmp) / "quality_report.json").read_text(
                    encoding="utf-8"
                )
            )
            dens = report["counts"]["density"]
            self.assertEqual(dens["total_points"], 100)
            self.assertEqual(dens["area_m2"], 200.0)
            self.assertEqual(dens["npd_all_pts_m2"], 0.5)
            self.assertEqual(dens["npd_first_pts_m2"], 0.3)
            self.assertAlmostEqual(
                dens["nps_first_m"], 1.0 / (0.3 ** 0.5), places=4
            )
            self.assertEqual(
                dens["class_histogram"], {"2": 30, "5": 70}
            )


class HeaderFidelityTests(unittest.TestCase):
    def test_copy_preserves_identity(self) -> None:
        from lynceus.processing.provenance import copy_header_identity

        import laspy

        src = laspy.LasHeader(point_format=6, version="1.4")
        src.global_encoding.gps_time_type = 1
        src.file_source_id = 7
        dst = laspy.LasHeader(point_format=6, version="1.4")
        copy_header_identity(dst, src)
        self.assertEqual(int(dst.global_encoding.gps_time_type), 1)
        self.assertEqual(dst.file_source_id, 7)

    def test_write_clean_laz_round_trips_identity(self) -> None:
        from lynceus.nodes._point_source import read_cloud, write_clean_laz

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            header = laspy.LasHeader(point_format=6, version="1.4")
            header.scales = np.array([0.01, 0.01, 0.01])
            header.offsets = np.zeros(3)
            header.global_encoding.gps_time_type = 1
            header.file_source_id = 9
            rec = laspy.ScaleAwarePointRecord.zeros(
                10, point_format=header.point_format,
                scales=header.scales, offsets=header.offsets,
            )
            rec.x = np.arange(10, dtype=float)
            with laspy.open(str(root / "s.laz"), mode="w",
                            header=header) as writer:
                writer.write_points(rec)
            cloud, hdr, scales, offsets = read_cloud(str(root / "s.laz"))
            write_clean_laz(
                str(root / "o.laz"), cloud, hdr, scales, offsets
            )
            with laspy.open(str(root / "o.laz")) as reader:
                out = reader.header
            self.assertEqual(int(out.global_encoding.gps_time_type), 1)
            self.assertEqual(out.file_source_id, 9)

    def test_tiler_tiles_keep_identity(self) -> None:
        from lynceus.processing.tiler import LiDARTiler

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            header = laspy.LasHeader(point_format=6, version="1.4")
            header.scales = np.array([0.01, 0.01, 0.01])
            header.offsets = np.zeros(3)
            header.global_encoding.gps_time_type = 1
            header.file_source_id = 4
            rec = laspy.ScaleAwarePointRecord.zeros(
                50, point_format=header.point_format,
                scales=header.scales, offsets=header.offsets,
            )
            rec.x = np.linspace(0.0, 99.0, 50)
            rec.y = np.zeros(50)
            rec.z = np.full(50, 10.0)
            with laspy.open(str(root / "s.laz"), mode="w",
                            header=header) as writer:
                writer.write_points(rec)
            result = LiDARTiler(
                root / "s.laz", root / "tiles", tile_size=100.0,
                buffer_m=0.0, max_points_per_tile=0,
            ).run(cancel_flag=None)
            self.assertTrue(result.tiles_generated)
            with laspy.open(result.tiles_generated[0]["file"]) as reader:
                tiled = reader.header
            self.assertEqual(int(tiled.global_encoding.gps_time_type), 1)
            self.assertEqual(tiled.file_source_id, 4)

    def test_export_header_copy_keeps_identity(self) -> None:
        from lynceus.nodes.lidar.output.export_point_cloud import (
            _header_copy,
        )

        header = laspy.LasHeader(point_format=6, version="1.4")
        header.global_encoding.gps_time_type = 1
        header.file_source_id = 5
        out = _header_copy(header)
        self.assertEqual(int(out.global_encoding.gps_time_type), 1)
        self.assertEqual(out.file_source_id, 5)


class ReservedTargetTests(unittest.TestCase):
    def test_reserved_targets_raise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = _write_cloud(root / "s.laz")
            tile = {"tile_id": "t_c0000_r0000", "file": str(src)}
            ctx = {"session_dir": str(root / "s"),
                   "intermediate_ext": "laz"}
            mask = lambda record: np.ones(len(record), dtype=bool)
            for target in (8, 12):
                with self.subTest(target=target):
                    with self.assertRaises(ValueError):
                        run_clean(tile, ctx, MODULE, "clean_r", mask,
                                  cls_value=target)


class WaveformGateTests(unittest.TestCase):
    def test_pdrf9_refused(self) -> None:
        from types import SimpleNamespace

        from lynceus.processing.tiler import _require_no_waveform

        bad = SimpleNamespace(point_format=SimpleNamespace(id=9))
        with self.assertRaises(RuntimeError):
            _require_no_waveform(bad, "flight.laz")
        good = SimpleNamespace(point_format=SimpleNamespace(id=8))
        self.assertIsNone(_require_no_waveform(good, "flight.laz"))

    def test_crs_provenance_paths(self) -> None:
        from types import SimpleNamespace

        from lynceus.processing.tiler import _crs_provenance

        header = SimpleNamespace(point_format=SimpleNamespace(id=6), vlrs=[])
        self.assertEqual(_crs_provenance(header)[1], "none")


class CrsOriginWarningTests(unittest.TestCase):
    def test_missing_file_is_silent(self) -> None:
        from lynceus.processing.controller import PipelineController

        self.assertEqual(
            PipelineController._crs_origin_warnings("s.laz", "/nope.laz"), []
        )

    def test_plain_header_no_geokeys_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            header = laspy.LasHeader(point_format=6, version="1.4")
            header.scales = np.array([0.01, 0.01, 0.01])
            header.offsets = np.zeros(3)
            rec = laspy.ScaleAwarePointRecord.zeros(
                5, point_format=header.point_format,
                scales=header.scales, offsets=header.offsets,
            )
            with laspy.open(str(root / "s.laz"), mode="w",
                            header=header) as writer:
                writer.write_points(rec)
            from lynceus.processing.controller import PipelineController

            self.assertEqual(
                PipelineController._crs_origin_warnings(
                    "s.laz", str(root / "s.laz")), []
            )


if __name__ == "__main__":
    unittest.main()
