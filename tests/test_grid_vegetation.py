# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Tests for Grid Vegetation (normalized histograms + distributional finals)."""

import json
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np

from lynceus.nodes.lidar.analysis.grid_vegetation import (
    HIST_BINS,
    PERCENTILES,
    barrier_grid_vegetation,
    finalize_vegetation_frame,
    tile_grid_vegetation,
)


def _reference_finals(counts: np.ndarray, threshold_m: float = 2.0,
                      lai_k: float = 0.5) -> dict:
    """Obviously-correct scalar version of finalize_vegetation_frame."""
    total = float(counts.sum())
    assert total > 0
    canopy_bin = min(max(int(threshold_m), 0), HIST_BINS - 1)
    canopy = float(counts[canopy_bin:].sum())
    cover = canopy / total

    def pct(p: float) -> float:
        target = p / 100.0 * total
        cum = np.cumsum(counts)
        idx = int(np.searchsorted(cum, target, side="left"))
        idx = min(max(idx, 0), HIST_BINS - 1)
        prev = float(cum[idx - 1]) if idx > 0 else 0.0
        width = float(counts[idx])
        frac = 0.5 if width <= 0 else (target - prev) / width
        return idx + min(max(frac, 0.0), 1.0)

    percentiles = {p: pct(float(p)) for p in PERCENTILES}
    p10, p50, p90 = percentiles[10], percentiles[50], percentiles[90]
    span = p90 - p10
    weights = counts.astype(float)
    props3 = np.array([weights[0:2].sum(), weights[2:10].sum(),
                       weights[10:].sum()])
    props3 = props3[props3 > 0] / total
    nz = int(np.nonzero(weights)[0].max()) + 1
    h = weights[:nz]
    fhd_props = h[h > 0] / total
    return {
        "cover": cover,
        "lai": min(5.0, float(-np.log(max(1.0 - cover, 1e-9)) / lai_k)),
        "vci": float(1.0 - canopy / total),
        "fhd": float(-np.sum(fhd_props * np.log(fhd_props))),
        "shannon_h": float(-np.sum(props3 * np.log(props3))),
        "density_0_2m": float(weights[0:2].sum() / total),
        "density_2_10m": float(weights[2:10].sum() / total),
        "density_above_10m": float(weights[10:].sum() / total),
        "l_skewness": float((p90 - 2 * p50 + p10) / span) if span > 0 else 0.0,
        **{f"p{p}": float(percentiles[p]) for p in PERCENTILES},
    }


class VegetationParityTests(unittest.TestCase):
    def test_vectorized_matches_scalar_reference(self) -> None:
        rng = np.random.default_rng(21)
        for trial in range(5):
            counts = rng.integers(0, 40, HIST_BINS).astype(np.int64)
            counts[rng.integers(0, HIST_BINS, 3)] = 0
            n = np.array([int(counts.sum())])
            got = finalize_vegetation_frame(
                ["0_0"], np.array([0]), np.array([0]),
                counts.reshape(1, -1), n, 10.0, 2.0, 0.5, -9999.0,
            )
            want = _reference_finals(counts)
            for key, expected in want.items():
                self.assertAlmostEqual(
                    got[key][0] if isinstance(got[key], np.ndarray) else got[key],
                    expected, places=6, msg=f"{key} trial {trial}",
                )


def _write_cloud(path: Path, xs, ys, zs, cls) -> Path:
    n = len(xs)
    header = laspy.LasHeader(point_format=3, version="1.2")
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
    with laspy.open(str(path), mode="w", header=header) as writer:
        writer.write_points(rec)
    return path


def _tile_dict(tile_id: str, path: Path) -> dict:
    return {
        "tile_id": tile_id,
        "file": str(path),
        "core_x_min": 0.0,
        "core_x_max": 100.0,
        "core_y_min": 0.0,
        "core_y_max": 100.0,
    }


def _tile_ctx(session: Path, **kw) -> dict:
    ctx = {"session_dir": str(session), "node_iid": "gv", "tile_id": "t",
           "resolution": 10.0}
    ctx.update(kw)
    return ctx


class GridVegetationTests(unittest.TestCase):
    def test_tile_histograms_use_ground_median(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Ground at 400 (median 400); veg at 405..414 in cell (0,0).
            xs = [1.0] * 12
            ys = [1.0] * 12
            zs = [400.0] * 4 + [405.0, 406.0, 407.0, 408.0,
                                409.0, 410.0, 412.0, 414.0]
            cls = [2] * 4 + [5] * 8
            src = _write_cloud(root / "t.laz", xs, ys, zs, cls)
            out = tile_grid_vegetation(
                _tile_dict("t", src), _tile_ctx(root / "s")
            )
            frame = gpd.read_file(out["output"])
            self.assertEqual(len(frame), 1)
            row = frame.iloc[0]
            self.assertEqual(row["gid"], "0_0")
            self.assertEqual(row["n"], 8)
            # Normalized heights 5,6,7,8,9,10,12,14 -> bins 5..10,12,14.
            for b in (5, 6, 7, 8, 9, 10, 12, 14):
                self.assertEqual(row[f"h{b}"], 1, b)
            self.assertEqual(row[[f"h{b}" for b in range(61)]].sum(), 8)

    def test_barrier_distributional_finals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = root / "s"
            xs = [1.0] * 12
            ys = [1.0] * 12
            zs = [400.0] * 4 + [405.0, 406.0, 407.0, 408.0,
                                409.0, 410.0, 412.0, 414.0]
            cls = [2] * 4 + [5] * 8
            src = _write_cloud(root / "t.laz", xs, ys, zs, cls)
            tile_grid_vegetation(
                _tile_dict("t", src), _tile_ctx(session)
            )
            ctx = {"session_dir": str(session), "node_iid": "gv",
                   "resolution": 10.0}
            out = barrier_grid_vegetation(ctx)
            frame = gpd.read_file(out["file"])
            self.assertEqual(len(frame), 1)
            row = frame.iloc[0]
            # cover: all 8 above threshold 2.0
            self.assertAlmostEqual(row["cover"], 1.0, places=6)
            self.assertAlmostEqual(
                row["lai"], min(5.0, -np.log(1e-9) / 0.5), places=4
            )
            self.assertAlmostEqual(row["vci"], 0.0, places=6)
            # median of 5..14 ~ 8.5-9.5 range sanity
            self.assertGreaterEqual(row["p50"], 8.0)
            self.assertLessEqual(row["p50"], 10.0)
            self.assertGreater(row["p95"], row["p50"])
            # densities: all in 2..10 and above bands
            self.assertAlmostEqual(
                row["density_2_10m"] + row["density_above_10m"], 1.0, places=6
            )
            self.assertAlmostEqual(row["density_0_2m"], 0.0, places=6)
            twin = Path(str(out["file"]).replace(".gpkg", ".csv"))
            self.assertTrue(twin.is_file())
            self.assertIn("csv", out)
            sidecar = json.loads(
                Path(str(out["file"]).replace(".gpkg", ".meta.json")).read_text(
                    encoding="utf-8"
                )
            )
            self.assertIn("cover", sidecar["metrics_computed"])
            self.assertIn("p95", sidecar["metrics_computed"])
            self.assertEqual(sidecar["default_metric"], "cover")

    def test_no_ground_writes_empty_with_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = root / "s"
            # No class 2: no datum to normalize against -> empty partials.
            src = _write_cloud(
                root / "t.laz", [1.0] * 5, [1.0] * 5,
                [400.0, 401.0, 402.0, 403.0, 404.0], [5] * 5,
            )
            tiled = tile_grid_vegetation(
                _tile_dict("t", src), _tile_ctx(session)
            )
            self.assertTrue(tiled.get("warnings"))
            frame = gpd.read_file(tiled["output"])
            self.assertEqual(len(frame), 0)
            out = barrier_grid_vegetation(
                {"session_dir": str(session), "node_iid": "gv",
                 "resolution": 10.0}
            )
            self.assertTrue(out.get("warnings"))


def _write_dtm_plane(path: Path) -> Path:
    """1 m DTM of the z=100+0.5x plane over x in [0,200), y in [0,10)."""
    from affine import Affine

    from lynceus.processing.raster import write_geotiff

    xs = np.arange(0.5, 200.0, 1.0)
    grid = np.tile(100.0 + 0.5 * xs, (10, 1)).astype(np.float32)
    write_geotiff(
        str(path), grid, Affine(1.0, 0.0, 0.0, 0.0, -1.0, 10.0),
        crs="EPSG:32721", nodata=-9999.0,
    )
    return path


def _slope_tile(tile_id: str, path: Path, x0: float) -> dict:
    """One tile on the z=100+0.5x slope: ground on the plane, vegetation
    exactly 6 m above local ground at a cell center."""
    xs = np.concatenate([
        np.linspace(x0, x0 + 100.0, 40),  # ground spread (median reference)
        np.full(16, x0 + 25.5),  # veg, cell center -> hv exactly 6.0
    ])
    ys = np.full(xs.shape, 5.0)
    is_ground = np.arange(xs.size) < 40
    zs = np.where(
        is_ground,
        100.0 + 0.5 * xs,
        100.0 + 0.5 * xs + 6.0,
    )
    cls = np.where(is_ground, 2, 5).astype(np.uint8)
    src = _write_cloud(path, xs, ys, zs, cls)
    return {
        "tile_id": tile_id,
        "file": str(src),
        "core_x_min": x0,
        "core_x_max": x0 + 100.0,
        "core_y_min": 0.0,
        "core_y_max": 100.0,
    }


class DtmReferenceTests(unittest.TestCase):
    """Optional DTM input: exact normalization on slopes (F4.2 follow-up).

    Without a wired DTM both tiles share the same biased tile-median
    reference; with the plane DTM every cell reads its true 6 m canopy.
    """

    def _run_pair(self, root: Path, session: str,
                  dtm: Path | None) -> dict:
        tile_a = _slope_tile("s_c0000_r0000", root / "a.laz", 0.0)
        tile_b = _slope_tile("s_c0001_r0000", root / "b.laz", 100.0)
        outs = {}
        for tag, tile in (("a", tile_a), ("b", tile_b)):
            ctx = _tile_ctx(root / session)
            if dtm is not None:
                ctx["dtm_mosaic_path"] = str(dtm)
            outs[tag] = tile_grid_vegetation(tile, ctx)
        ctx = {"session_dir": str(root / session), "node_iid": "gv",
               "resolution": 10.0}
        merged = barrier_grid_vegetation(ctx)
        return outs, merged

    def test_wired_dtm_normalizes_slope_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dtm = _write_dtm_plane(root / "dtm.tif")
            outs, merged = self._run_pair(root, "wired", dtm)
            frame = gpd.read_file(merged["file"])
            self.assertEqual(len(frame), 2)
            for _, row in frame.iterrows():
                # True 6 m canopy in both tiles: full cover in the 2-10 m
                # stratum (bin 6), median height 6.5 m by histogram rule.
                self.assertEqual(row["point_count"], 16)
                self.assertAlmostEqual(row["cover"], 1.0, places=6)
                self.assertAlmostEqual(
                    row["density_2_10m"], 1.0, places=6
                )
                self.assertAlmostEqual(row["p50"], 6.5, places=6)
            for tag in ("a", "b"):
                self.assertNotIn("unnormalized_dropped", outs[tag])

    def test_legacy_median_biases_slope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_dtm_plane(root / "dtm.tif")  # present but unwired
            _, merged = self._run_pair(root, "legacy", None)
            frame = gpd.read_file(merged["file"])
            covers = sorted(frame["cover"].tolist())
            # Tile medians sit ~25 m off the local ground at the cells:
            # most of the 6 m canopy lands below the 2 m threshold.
            self.assertLess(covers[0], 1.0)

    def test_missing_dtm_falls_back_with_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tile = _slope_tile("s_c0000_r0000", root / "a.laz", 0.0)
            ctx = _tile_ctx(root / "s")
            ctx["dtm_mosaic_path"] = str(root / "nope.tif")
            out = tile_grid_vegetation(tile, ctx)
            self.assertTrue(
                any("tile-median" in w for w in out.get("warnings", []))
            )
            frame = gpd.read_file(out["output"])
            self.assertGreater(len(frame), 0)

    def test_points_outside_dtm_are_dropped_and_counted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dtm = _write_dtm_plane(root / "dtm.tif")
            # Tile far outside the DTM extent: everything unnormalizable.
            tile = _slope_tile("s_c0009_r0009", root / "far.laz", 900.0)
            ctx = _tile_ctx(root / "s")
            ctx["dtm_mosaic_path"] = str(dtm)
            out = tile_grid_vegetation(tile, ctx)
            self.assertGreater(out.get("unnormalized_dropped", 0), 0)
            frame = gpd.read_file(out["output"])
            self.assertEqual(len(frame), 0)


    def test_dtm_path_injected_for_consumer(self) -> None:
        from lynceus.nodes.ports import PortType
        from lynceus.processing.steps import (
            _inject_input_paths,
            effective_node_caps,
        )

        def _posix(path: str) -> str:
            return path.replace("\\", "/")

        DTM = "lynceus.nodes.lidar.terrain.generate_dtm"
        VEG = "lynceus.nodes.lidar.analysis.grid_vegetation"

        def _caps_of(iid: str):
            module = {"d1": DTM, "gv": VEG}[iid]
            return module, effective_node_caps(module, None)

        edges = [("d1", "gv", PortType.DTM_MOSAIC, PortType.DTM_MOSAIC)]
        legacy = {"session_dir": "/s"}
        _inject_input_paths(legacy, VEG, edges, _caps_of, None, {})
        self.assertEqual(
            _posix(legacy["dtm_mosaic_path"]), "/s/dtm_mosaic.tif"
        )
        scoped = {"session_dir": "/s"}
        _inject_input_paths(scoped, VEG, edges, _caps_of, None, {"d1": "d1"})
        self.assertEqual(
            _posix(scoped["dtm_mosaic_path"]), "/s/d1/dtm_mosaic.tif"
        )


    def test_unconnected_optional_dtm_passes_required_gate(self) -> None:
        from lynceus.nodes.ports import PortType
        from lynceus.processing.controller import (
            PipelineCallbacks,
            PipelineController,
        )

        VEG = "lynceus.nodes.lidar.analysis.grid_vegetation"
        LOAD = "lynceus.nodes.lidar.source.load_las_laz"
        ctrl = PipelineController()
        ctrl._modules = {"gv": VEG, "load": LOAD}
        cb = PipelineCallbacks()
        wired_pc = [("load", "gv", PortType.POINT_CLOUD, PortType.POINT_CLOUD)]
        # Optional DTM unconnected: silent pass.
        self.assertTrue(ctrl._check_required_ports(["gv"], wired_pc, cb, {}))
        # Required POINT_CLOUD unconnected: still blocked.
        self.assertFalse(ctrl._check_required_ports(["gv"], [], cb, {}))


class TileMergeDepsTests(unittest.TestCase):
    """Tiles consuming barrier finals wait for the provider merge.

    A vegetation tile sampling a DTM mosaic must run after the merge,
    not alongside the provider tiles (fresh runs have no mosaic yet);
    point-cloud streams keep tile-level dependencies.
    """

    DTM = "lynceus.nodes.lidar.terrain.generate_dtm"
    VEG = "lynceus.nodes.lidar.analysis.grid_vegetation"
    METRICS = "lynceus.nodes.lidar.analysis.grid_metrics"

    def _tasks(self, order, edges, modules):
        from lynceus.processing.steps import build_dag

        tiles = [{"tile_id": "t1", "file": "x.laz"}]
        return build_dag(
            order, edges, tiles, {"session_dir": "unused"},
            modules=modules, configs={},
        )

    def test_mosaic_edge_waits_for_merge(self) -> None:
        from lynceus.nodes.ports import PortType

        tasks = self._tasks(
            ["d1", "gv"],
            [("d1", "gv", PortType.DTM_MOSAIC, PortType.DTM_MOSAIC)],
            {"d1": self.DTM, "gv": self.VEG},
        )
        by_id = {t.task_id: t for t in tasks}
        self.assertEqual(by_id["gv|t1"].deps, {"d1|merge"})

    def test_stream_edge_keeps_tile_dep(self) -> None:
        from lynceus.nodes.ports import PortType

        tasks = self._tasks(
            ["m", "gv"],
            [("m", "gv", PortType.POINT_CLOUD, PortType.POINT_CLOUD)],
            {"m": self.METRICS, "gv": self.VEG},
        )
        by_id = {t.task_id: t for t in tasks}
        self.assertEqual(by_id["gv|t1"].deps, {"m|t1"})
        from lynceus.nodes._point_source import UNKNOWN_CRS, clean_crs

        self.assertIsNone(clean_crs(UNKNOWN_CRS))
        self.assertIsNone(clean_crs(None))
        self.assertIsNone(clean_crs(""))
        self.assertEqual(clean_crs("EPSG:25831"), "EPSG:25831")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = _write_cloud(
                root / "t.laz", [1.0] * 6, [1.0] * 6,
                [400.0, 400.0, 405.0, 406.0, 407.0, 408.0],
                [2, 2, 5, 5, 5, 5],
            )
            ctx = _tile_ctx(root / "s", crs=UNKNOWN_CRS)
            out = tile_grid_vegetation(_tile_dict("t", src), ctx)
            frame = gpd.read_file(out["output"])
            self.assertGreater(len(frame), 0)


if __name__ == "__main__":
    unittest.main()
