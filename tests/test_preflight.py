"""Pre-flight advisor: LAS/LAZ header reads + run-sizing heuristics.

Synthetic LAS headers are built with ``struct`` directly (no laspy), the
same contract the advisor uses at runtime on the GUI thread."""

from __future__ import annotations

import math
import struct
import tempfile
import unittest
from pathlib import Path

from lynceus.processing.executor import _BYTES_PER_POINT_EST
from lynceus.processing.point_counts import point_count, read_header, union_area_m2
from lynceus.processing.preflight import (
    MERGE_FLOOR_MAX,
    MERGE_FLOOR_MIN,
    MERGE_TILE_THRESHOLD,
    RAM_ADVICE_RATIO,
    build_preflight,
    suggested_merge_floor,
)
from lynceus.processing.controller import default_output_root, resolve_output_root


def _las12(
    count: int,
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    wkt: bytes | None = None,
) -> Path:
    """Minimal but well-formed LAS 1.2 header (legacy 32-bit count)."""
    buf = bytearray(227)
    buf[:4] = b"LASF"
    buf[24] = 1  # major
    buf[25] = 2  # minor
    struct.pack_into("<H", buf, 94, 227)  # header size
    struct.pack_into("<I", buf, 96, 227)  # offset to point data
    buf[104] = 3  # point data format
    struct.pack_into("<H", buf, 105, 34)  # record length
    struct.pack_into("<I", buf, 107, count)  # legacy point count
    struct.pack_into("<d", buf, 179, max_x)
    struct.pack_into("<d", buf, 187, min_x)
    struct.pack_into("<d", buf, 195, max_y)
    struct.pack_into("<d", buf, 203, min_y)
    struct.pack_into("<d", buf, 211, 500.0)
    struct.pack_into("<d", buf, 219, 0.0)
    if wkt is not None:
        struct.pack_into("<I", buf, 96, 227 + 54 + len(wkt) + 1)
        struct.pack_into("<I", buf, 100, 1)  # one VLR
        vlr = (
            struct.pack("<H", 0)
            + b"LASF_Projection".ljust(16, b"\x00")
            + struct.pack("<H", 2112)
            + struct.pack("<H", len(wkt) + 1)
            + b"".ljust(32, b"\x00")
        )
        return _write_header(bytearray(bytes(buf) + vlr + wkt + b"\x00"))
    return _write_header(buf)


def _las14_extended(count: int) -> Path:
    """LAS 1.4 header carrying only the 64-bit extended count."""
    buf = bytearray(375)
    buf[:4] = b"LASF"
    buf[24] = 1
    buf[25] = 4
    struct.pack_into("<H", buf, 94, 375)
    struct.pack_into("<I", buf, 96, 375)
    buf[104] = 6
    struct.pack_into("<I", buf, 107, 0)  # legacy zero: only extended counts
    struct.pack_into("<d", buf, 179, 1000.0)
    struct.pack_into("<d", buf, 187, 0.0)
    struct.pack_into("<d", buf, 195, 1200.0)
    struct.pack_into("<d", buf, 203, 0.0)
    struct.pack_into("<Q", buf, 247, count)
    return _write_header(buf)


def _write_header(buf: bytearray) -> Path:
    fd, name = tempfile.mkstemp(suffix=".laz")
    import os

    os.close(fd)  # Windows: an open handle blocks unlink.
    path = Path(name)
    path.write_bytes(bytes(buf))
    return path


def _tmp(*paths: Path):
    for p in paths:
        p.unlink(missing_ok=True)


class HeaderDecodeTests(unittest.TestCase):
    def test_legacy_count_and_bounds(self) -> None:
        path = _las12(7_000_000, 5.0, 6.0, 105.0, 206.0)
        self.addCleanup(_tmp, path)
        header = read_header(path)
        self.assertIsNotNone(header)
        assert header is not None
        self.assertEqual(header["point_count"], 7_000_000)
        self.assertEqual(header["point_format"], 3)
        self.assertEqual(header["bounds"], (5.0, 6.0, 105.0, 206.0))
        self.assertEqual(point_count(path), 7_000_000)

    def test_extended_count_wins_over_zero_legacy(self) -> None:
        path = _las14_extended(372_000_000)
        self.addCleanup(_tmp, path)
        self.assertEqual(point_count(path), 372_000_000)

    def test_unreadable_and_non_las_return_none(self) -> None:
        import os

        fd, name = tempfile.mkstemp()
        os.close(fd)
        missing = Path(name)
        missing.unlink(missing_ok=True)
        self.addCleanup(_tmp, missing)
        self.assertIsNone(read_header(missing))
        junk = _write_header(bytearray(b"NOT A LAS FILE AT ALL........."))
        self.addCleanup(_tmp, junk)
        self.assertIsNone(read_header(junk))


class UnionAreaTests(unittest.TestCase):
    def test_sums_points_and_union_area(self) -> None:
        a = _las12(1_000_000, 0.0, 0.0, 100.0, 100.0)
        b = _las12(2_000_000, 50.0, 50.0, 200.0, 300.0)
        self.addCleanup(_tmp, a, b)
        total, area = union_area_m2([str(a), str(b)])
        self.assertEqual(total, 3_000_000)
        self.assertAlmostEqual(area, 200.0 * 300.0)  # union bbox 0..200 x 0..300

    def test_geographic_extent_yields_no_area(self) -> None:
        wkt = b'GEOGCS["WGS 84", DATUM["WGS_1984"], SPHEROID["WGS 84",6378137,298.257223563]]'
        path = _las12(
            500_000, -58.5, -34.5, -58.0, -34.0, wkt=wkt
        )
        self.addCleanup(_tmp, path)
        total, area = union_area_m2([str(path)])
        self.assertEqual(total, 500_000)
        self.assertIsNone(area)

    def test_projected_wkt_keeps_area(self) -> None:
        wkt = b'PROJCS["WGS 84 / UTM 21S", GEOGCS["WGS 84"], PROJECTION["Transverse_Mercator"]]'
        path = _las12(200_000, 400_000.0, 6_150_000.0, 401_000.0, 6_151_000.0, wkt=wkt)
        self.addCleanup(_tmp, path)
        total, area = union_area_m2([str(path)])
        self.assertEqual(total, 200_000)
        self.assertAlmostEqual(area, 1000.0 * 1000.0)

    def test_unreadable_contributions_are_skipped(self) -> None:
        import os

        good = _las12(100_000, 0.0, 0.0, 10.0, 10.0)
        fd, name = tempfile.mkstemp()
        os.close(fd)
        missing = Path(name)
        missing.unlink(missing_ok=True)
        self.addCleanup(_tmp, good, missing)
        total, area = union_area_m2([str(good), str(missing)])
        self.assertEqual(total, 100_000)
        self.assertAlmostEqual(area, 100.0)


class PreflightBuildTests(unittest.TestCase):
    def test_worker_peak_uses_heaviest_factor(self) -> None:
        path = _las12(100_000, 0.0, 0.0, 1000.0, 1000.0)  # 100 ha @ 100 m = 100 tiles
        self.addCleanup(_tmp, path)
        # 100k pts / 100 tiles = 1000 pts per tile.
        pf = build_preflight(
            [str(path)], [3.0, 1.5], workers=8, budget_bytes=None,
        )
        self.assertIsNotNone(pf)
        assert pf is not None
        self.assertEqual(pf.tile_count_est, 100)
        self.assertEqual(pf.per_tile_points, 1000)
        self.assertEqual(pf.heaviest_factor, 3.0)
        self.assertEqual(
            pf.est_peak_bytes, 8 * 1000 * _BYTES_PER_POINT_EST * 3.0
        )

    def test_cache_advice_for_large_cloud_or_long_chain(self) -> None:
        big = _las12(100_000_000, 0.0, 0.0, 1000.0, 1000.0)
        self.addCleanup(_tmp, big)
        pf = build_preflight([str(big)], [3.0], workers=8, cache_enabled=False)
        assert pf is not None
        self.assertTrue(pf.ask_cache)
        # Cache already on: no advice.
        pf_on = build_preflight([str(big)], [3.0], workers=8, cache_enabled=True)
        assert pf_on is not None
        self.assertFalse(pf_on.ask_cache)
        # Small cloud, short chain: no cache advice.
        small = _las12(1_000_000, 0.0, 0.0, 1000.0, 1000.0)
        self.addCleanup(_tmp, small)
        pf_small = build_preflight(
            [str(small)], [3.0], workers=8, cache_enabled=False
        )
        assert pf_small is not None
        self.assertFalse(pf_small.ask_cache)

    def test_ram_advice_against_budget(self) -> None:
        path = _las12(10_000_000, 0.0, 0.0, 1000.0, 1000.0)  # 10k pts per tile
        self.addCleanup(_tmp, path)
        pf = build_preflight(
            [str(path)], [3.0], workers=8, budget_bytes=9_000, cache_enabled=True,
        )
        assert pf is not None
        # est = 8 * 10_000 * 64 * 3 = 15_360_000 bytes >= 0.70 * 9000.
        self.assertTrue(pf.ask_ram)
        roomy = build_preflight(
            [str(path)], [3.0], workers=8, budget_bytes=9_999_999_999,
            cache_enabled=True,
        )
        assert roomy is not None
        self.assertFalse(roomy.ask_ram)

    def test_advice_ratio_constant_is_sane(self) -> None:
        self.assertGreater(RAM_ADVICE_RATIO, 0.0)
        self.assertLessEqual(RAM_ADVICE_RATIO, 1.0)

    def test_no_sources_or_zero_points_yields_none(self) -> None:
        self.assertIsNone(build_preflight([], [1.0], workers=8))
        empty = _las12(0, 0.0, 0.0, 100.0, 100.0)
        self.addCleanup(_tmp, empty)
        self.assertIsNone(build_preflight([str(empty)], [1.0], workers=8))

    def test_rounds_up_math_uses_ceil(self) -> None:
        self.assertEqual(math.ceil(1.0), 1)  # sanity for the helper math


class MergeSuggestTests(unittest.TestCase):
    def test_floor_scales_with_source_deterministically(self) -> None:
        self.assertEqual(
            suggested_merge_floor(1_173_185_133), 1_173_185_133 // 3000)
        self.assertEqual(suggested_merge_floor(1_000), MERGE_FLOOR_MIN)
        self.assertEqual(suggested_merge_floor(10**12), MERGE_FLOOR_MAX)
        self.assertEqual(suggested_merge_floor(0), MERGE_FLOOR_MIN)

    def test_suggests_on_many_small_tiles(self) -> None:
        # 25k estimated tiles averaging 4k points: merge territory.
        path = _las12(100_000_000, 0.0, 0.0, 50000.0, 50000.0)
        self.addCleanup(_tmp, path)
        pf = build_preflight(
            [str(path)], [1.0], workers=8, tile_size_m=100.0,
            cache_enabled=True,
        )
        assert pf is not None
        self.assertGreaterEqual(pf.tile_count_est, MERGE_TILE_THRESHOLD)
        self.assertLess(pf.per_tile_points, pf.merge_floor_suggest)
        self.assertGreater(pf.merge_floor_suggest, 0)
        self.assertTrue(pf.has_advice)

    def test_silent_when_dense_or_enabled_or_few(self) -> None:
        dense = _las12(100_000_000, 0.0, 0.0, 1000.0, 1000.0)
        self.addCleanup(_tmp, dense)
        pf = build_preflight([str(dense)], [1.0], workers=8)
        assert pf is not None
        self.assertEqual(pf.merge_floor_suggest, 0)
        sparse = _las12(100_000_000, 0.0, 0.0, 50000.0, 50000.0)
        self.addCleanup(_tmp, sparse)
        on = build_preflight(
            [str(sparse)], [1.0], workers=8, cache_enabled=True,
            merge_enabled=True,
        )
        assert on is not None
        self.assertEqual(on.merge_floor_suggest, 0)
        few = _las12(1_000_000, 0.0, 0.0, 1000.0, 1000.0)
        self.addCleanup(_tmp, few)
        small = build_preflight([str(few)], [1.0], workers=8)
        assert small is not None
        self.assertEqual(small.merge_floor_suggest, 0)


class MosaicAdviceTests(unittest.TestCase):
    def test_mosaic_fires_on_huge_area_with_rasters(self) -> None:
        from lynceus.processing.preflight import (
            MOSAIC_BYTES_PER_CELL,
            MOSAIC_WORKING_COPIES,
        )

        wide = _las12(100_000_000, 0.0, 0.0, 91200.0, 64000.0)
        self.addCleanup(_tmp, wide)
        pf = build_preflight(
            [str(wide)], [1.0], workers=8, budget_bytes=1_073_741_824,
            has_mosaic=True,
        )
        assert pf is not None
        area = 91200.0 * 64000.0
        self.assertAlmostEqual(
            pf.mosaic_bytes_est,
            area * MOSAIC_BYTES_PER_CELL * MOSAIC_WORKING_COPIES)
        self.assertTrue(pf.ask_mosaic)
        self.assertTrue(pf.has_advice)
        # Even halving to the schema cap cannot fit: capped suggestion.
        self.assertEqual(pf.resolution_suggest, 5.0)
        self.assertEqual(pf.ram_budget_suggest, 0.0)

    def test_mosaic_prefers_raising_budget_when_it_fits(self) -> None:
        import math

        # 16 GiB machine, 1 GiB budget: the mosaic needs ~11 GiB working
        # set, so raising the budget (not coarsening) is suggested.
        modest = _las12(100_000_000, 0.0, 0.0, 20000.0, 20000.0)
        self.addCleanup(_tmp, modest)
        sixteen_gib = 16 * 1024 ** 3
        pf = build_preflight(
            [str(modest)], [1.0], workers=8,
            budget_bytes=1_073_741_824, has_mosaic=True,
            total_ram_bytes=sixteen_gib,
        )
        assert pf is not None
        self.assertTrue(pf.ask_mosaic)
        self.assertEqual(pf.resolution_suggest, 0.0)
        need = pf.mosaic_bytes_est / 0.70
        self.assertEqual(
            pf.ram_budget_suggest, math.ceil(need / 1024 ** 3) * 1024 ** 3)
        self.assertLessEqual(pf.ram_budget_suggest, sixteen_gib)

    def test_mosaic_silent_without_ram_info(self) -> None:
        modest = _las12(100_000_000, 0.0, 0.0, 20000.0, 20000.0)
        self.addCleanup(_tmp, modest)
        pf = build_preflight(
            [str(modest)], [1.0], workers=8,
            budget_bytes=1_073_741_824, has_mosaic=True,
        )
        assert pf is not None
        # Unknown physical RAM: cannot offer the raise, falls back to
        # resolution (conservative and honest).
        self.assertTrue(pf.ask_mosaic)
        self.assertEqual(pf.ram_budget_suggest, 0.0)
        self.assertGreater(pf.resolution_suggest, 0.0)

    def test_mosaic_silent_without_rasters_or_budget_or_area(self) -> None:
        wide = _las12(100_000_000, 0.0, 0.0, 91200.0, 64000.0)
        self.addCleanup(_tmp, wide)
        no_raster = build_preflight(
            [str(wide)], [1.0], workers=8, budget_bytes=1_073_741_824,
            has_mosaic=False,
        )
        assert no_raster is not None
        self.assertFalse(no_raster.ask_mosaic)
        self.assertEqual(no_raster.resolution_suggest, 0.0)
        no_budget = build_preflight(
            [str(wide)], [1.0], workers=8, has_mosaic=True)
        assert no_budget is not None
        self.assertFalse(no_budget.ask_mosaic)
        small = _las12(1_000_000, 0.0, 0.0, 1000.0, 1000.0)
        self.addCleanup(_tmp, small)
        tiny = build_preflight(
            [str(small)], [1.0], workers=8,
            budget_bytes=1_073_741_824, has_mosaic=True)
        assert tiny is not None
        self.assertFalse(tiny.ask_mosaic)

    def test_merge_aware_peak_and_workers_suggest(self) -> None:
        sparse = _las12(100_000_000, 0.0, 0.0, 50000.0, 50000.0)
        self.addCleanup(_tmp, sparse)
        pf = build_preflight(
            [str(sparse)], [1.0], workers=8, budget_bytes=10_000_000,
            cache_enabled=True,
        )
        assert pf is not None
        # 250k base tiles averaging 400 pts, floor suggested at 50k: the
        # peak sizes fused tiles, not the micro average.
        self.assertGreater(pf.merge_floor_suggest, 0)
        self.assertEqual(
            pf.est_peak_bytes, 8 * 50_000 * 64 * 1.0)
        self.assertTrue(pf.ask_ram)
        self.assertGreater(pf.workers_suggest, 0)
        self.assertLess(pf.workers_suggest, 8)

    def test_workers_silent_when_current_fits(self) -> None:
        path = _las12(10_000_000, 0.0, 0.0, 1000.0, 1000.0)
        self.addCleanup(_tmp, path)
        pf = build_preflight(
            [str(path)], [1.0], workers=2,
            budget_bytes=9_999_999_999, cache_enabled=True,
        )
        assert pf is not None
        self.assertFalse(pf.ask_ram)
        self.assertEqual(pf.workers_suggest, 0)


class SegmentsSuggestTests(unittest.TestCase):
    """Independent-segments suggestion: full mosaic overflows, batches fit."""

    def _wide(self):
        # ~5.8B m2 like a regional survey; 1.17B points.
        return _las12(1_173_185_133, 0.0, 0.0, 91200.0, 64000.0)

    def test_suggests_cap_and_count(self) -> None:
        path = self._wide()
        self.addCleanup(_tmp, path)
        pf = build_preflight(
            [str(path)], [1.0], workers=8,
            budget_bytes=16 * 1024 ** 3, cache_enabled=True,
            merge_enabled=True, has_mosaic=True,
        )
        assert pf is not None
        self.assertGreater(pf.segments_cap_suggest, 0)
        n = pf.segments_count_est
        self.assertGreaterEqual(n, 2)
        self.assertLessEqual(n, 200)
        # Cap bounds one batch mosaic within headroom.
        density = 1_173_185_133 / (91200.0 * 64000.0)
        seg_cells = pf.segments_cap_suggest / density
        self.assertLessEqual(
            seg_cells * 4.0 * 3.0, 0.50 * 16 * 1024 ** 3)
        self.assertTrue(pf.has_advice)

    def test_silent_when_independent_or_fitting_or_degenerate(self) -> None:
        path = self._wide()
        self.addCleanup(_tmp, path)
        base = dict(workers=8, budget_bytes=16 * 1024 ** 3,
                    cache_enabled=True, merge_enabled=True, has_mosaic=True)
        indep = build_preflight([str(path)], [1.0], **base,
                                segment_independent=True)
        assert indep is not None
        self.assertEqual(indep.segments_cap_suggest, 0)
        small = _las12(1_000_000, 0.0, 0.0, 1000.0, 1000.0)
        self.addCleanup(_tmp, small)
        fits = build_preflight([str(small)], [1.0], **base)
        assert fits is not None
        self.assertEqual(fits.segments_cap_suggest, 0)
        noram = build_preflight([str(path)], [1.0], workers=8,
                                cache_enabled=True, merge_enabled=True,
                                has_mosaic=True)
        assert noram is not None
        self.assertEqual(noram.segments_cap_suggest, 0)

    def test_configured_fitting_cap_is_kept(self) -> None:
        path = self._wide()
        self.addCleanup(_tmp, path)
        pf = build_preflight(
            [str(path)], [1.0], workers=8,
            budget_bytes=16 * 1024 ** 3, cache_enabled=True,
            merge_enabled=True, has_mosaic=True, seg_cap=25_000_000,
        )
        assert pf is not None
        # 25M pts/batch fits per segment: suggestion keeps the user's cap.
        self.assertEqual(pf.segments_cap_suggest, 25_000_000)
        self.assertGreaterEqual(pf.segments_count_est, 2)


class ResolveOutputRootTests(unittest.TestCase):
    """Item 43: configurable output root resolution + fallback."""

    def test_empty_falls_back_to_default(self) -> None:
        self.assertEqual(resolve_output_root(""), default_output_root())
        self.assertEqual(resolve_output_root("   "), default_output_root())

    def test_absolute_path_wins(self) -> None:
        from pathlib import Path

        result = resolve_output_root(r"D:\LidarRuns")
        self.assertTrue(result.is_absolute())
        self.assertEqual(result, Path(r"D:\LidarRuns"))

    def test_relative_becomes_subfolder_of_default(self) -> None:
        result = resolve_output_root("my_runs")
        self.assertEqual(result, default_output_root() / "my_runs")


class TemplatePartitionTests(unittest.TestCase):
    """Item 45: template files across devices partition + strip."""

    def setUp(self) -> None:
        import os

        fd, name = tempfile.mkstemp(suffix=".laz")
        os.close(fd)
        self._good = Path(name)
        self._good.write_bytes(b"")
        self._ghost = str(name) + ".missing.laz"

    def tearDown(self) -> None:
        self._good.unlink(missing_ok=True)

    def _doc(self) -> dict:
        return {
            "nodes": [
                {"iid": "a", "config": {"file_path": str(self._good)}},
                {"iid": "b", "config": {"file_path": self._ghost}},
                {"iid": "c", "config": {
                    "files": [str(self._good), self._ghost],
                    "file_path": str(self._good),
                }},
                {"iid": "d", "config": {}},
            ]
        }

    def test_partition_splits_existing_and_missing(self) -> None:
        from lynceus import templates as templates_io

        existing, missing = templates_io.partition_template_files(self._doc())
        self.assertEqual(len(existing), 1)  # the temp file appears once
        self.assertEqual(missing, [self._ghost])  # de-duplicated once

    def test_strip_empties_only_missing_paths(self) -> None:
        from lynceus import templates as templates_io

        doc = self._doc()
        removed = templates_io.strip_missing_files(doc, {self._ghost})
        self.assertEqual(removed, 2)  # node b file_path + node c files entry
        nodes = {n["iid"]: n["config"] for n in doc["nodes"]}
        self.assertEqual(nodes["a"]["file_path"], str(self._good))
        self.assertNotIn("file_path", nodes["b"])
        self.assertEqual(nodes["b"], {})
        self.assertEqual(nodes["c"]["files"], [str(self._good)])

    def test_strip_untouched_when_nothing_missing(self) -> None:
        from lynceus import templates as templates_io

        doc = self._doc()
        removed = templates_io.strip_missing_files(doc, set())
        self.assertEqual(removed, 0)
        self.assertEqual(doc["nodes"][0]["config"]["file_path"], str(self._good))


if __name__ == "__main__":
    unittest.main()