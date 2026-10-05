# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Tests for the real Grid Metrics node (per-cell partials + exact merge)."""

import json
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np

from lynceus.nodes.lidar.analysis.grid_metrics import (
    barrier_grid_metrics,
    tile_grid_metrics,
)


def _write_cloud(path: Path, xs, ys, zs, cls=None, intensity=None,
                 ret=None, nret=None) -> Path:
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
    rec.classification = np.asarray(
        cls if cls is not None else [0] * n, dtype=np.uint8
    )
    if intensity is not None:
        rec.intensity = np.asarray(intensity, dtype=np.uint16)
    if ret is not None:
        rec.return_number = np.asarray(ret, dtype=np.uint8)
    if nret is not None:
        rec.number_of_returns = np.asarray(nret, dtype=np.uint8)
    with laspy.open(str(path), mode="w", header=header) as writer:
        writer.write_points(rec)
    return path


def _tile_ctx(session: Path, **kw) -> dict:
    ctx = {"session_dir": str(session), "node_iid": "gm", "tile_id": "t",
           "resolution": 10.0}
    ctx.update(kw)
    return ctx


def _tile_dict(tile_id: str, path: Path) -> dict:
    return {
        "tile_id": tile_id,
        "file": str(path),
        "core_x_min": 0.0,
        "core_x_max": 100.0,
        "core_y_min": 0.0,
        "core_y_max": 100.0,
    }


class GridMetricsTests(unittest.TestCase):
    def test_tile_values_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Cell (0,0): 4 pts z 400..403, 2 ground, intensity 100/200/300/400.
            # Cell (1,0): 2 pts z 410,411, no ground, no intensity spread.
            xs = [1, 2, 3, 4, 15, 16]
            ys = [1, 2, 3, 4, 5, 6]
            zs = [400, 401, 402, 403, 410, 411]
            cls = [2, 2, 0, 7, 0, 0]
            inten = [100, 200, 300, 400, 1000, 1000]
            ret = [1, 1, 2, 1, 1, 2]
            nret = [2, 2, 2, 1, 1, 1]
            src = _write_cloud(root / "t.laz", xs, ys, zs, cls, inten, ret, nret)
            out = tile_grid_metrics(
                _tile_dict("t", src), _tile_ctx(root / "s")
            )
            frame = gpd.read_file(out["output"])
            self.assertEqual(len(frame), 2)
            cell = frame.set_index("gid").to_dict("index")
            c00 = cell["0_0"]
            self.assertEqual(c00["n"], 4)
            self.assertAlmostEqual(c00["z_min"], 400.0, places=2)
            self.assertAlmostEqual(c00["z_max"], 403.0, places=2)
            self.assertAlmostEqual(c00["z_sum"], 1606.0, places=2)
            self.assertEqual(c00["ground"], 2)
            self.assertEqual(c00["noise"], 1)
            self.assertAlmostEqual(c00["i_sum"] / c00["i_n"], 250.0, places=2)
            self.assertEqual(c00["class_2"], 2)
            self.assertEqual(c00["class_7"], 1)
            c10 = cell["1_0"]
            self.assertEqual(c10["n"], 2)
            self.assertEqual(c10["ground"], 0)

    def test_barrier_merges_and_finalizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = root / "s"
            # Same global cell (0,0) from two tiles: partials must add up.
            # Tile b leans ground so dominant_class has a clear winner.
            specs = {
                "a": ([1.0, 2.0], [400.0, 402.0], [2, 0]),
                "b": ([2.0, 3.0], [400.0, 402.0], [2, 2]),
            }
            for name, (xx, zz, cc) in specs.items():
                src = _write_cloud(
                    root / f"{name}.laz", xx, [1.0, 1.0], zz, cc,
                    [100, 300], [1, 2], [2, 2],
                )
                tile_grid_metrics(
                    _tile_dict(name, src), _tile_ctx(session, tile_id=name,
                                                    resolution=10.0)
                )
            ctx = {"session_dir": str(session), "node_iid": "gm",
                   "resolution": 10.0}
            out = barrier_grid_metrics(ctx)
            frame = gpd.read_file(out["file"])
            self.assertEqual(len(frame), 1)
            row = frame.iloc[0]
            self.assertEqual(row["gid"], "0_0")
            self.assertEqual(row["point_count"], 4)
            self.assertAlmostEqual(row["density"], 4 / 100.0, places=6)
            self.assertAlmostEqual(row["z_min"], 400.0, places=2)
            self.assertAlmostEqual(row["z_max"], 402.0, places=2)
            self.assertAlmostEqual(row["z_mean"], 401.0, places=2)
            self.assertAlmostEqual(row["z_range"], 2.0, places=2)
            self.assertAlmostEqual(row["ground_cover"], 0.75, places=6)
            self.assertAlmostEqual(row["intensity_mean"], 200.0, places=2)
            self.assertAlmostEqual(row["intensity_min"], 100.0, places=2)
            self.assertAlmostEqual(row["intensity_max"], 300.0, places=2)
            self.assertEqual(row["noise_count"], 0)
            self.assertEqual(row["dominant_class"], 2)
            self.assertAlmostEqual(row["first_return_cover"], 0.5, places=6)
            self.assertAlmostEqual(row["last_return_cover"], 0.5, places=6)
            self.assertAlmostEqual(row["veg_cover"], 0.0, places=6)
            twin = Path(str(out["file"]).replace(".gpkg", ".csv"))
            self.assertTrue(twin.is_file())
            self.assertIn("csv", out)
            import csv

            with open(twin, newline="", encoding="utf-8") as handle:
                header = next(csv.reader(handle))
            for col in ("dominant_class", "veg_cover", "first_return_cover",
                        "z_range", "intensity_min"):
                self.assertIn(col, header)
            sidecar = json.loads(
                Path(str(out["file"]).replace(".gpkg", ".meta.json")).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                sidecar["metrics_computed"],
                ["point_count", "density", "z_min", "z_max", "z_mean",
                 "z_std", "ground_count", "ground_cover", "intensity_mean",
                 "intensity_std", "noise_count", "dominant_class",
                 "intensity_min", "intensity_max", "z_range",
                 "first_return_cover", "last_return_cover", "veg_cover",
                 "building_cover", "water_cover"],
            )
            self.assertEqual(sidecar["default_metric"], "density")
            self.assertIn("gid", sidecar["bookkeeping_cols"])

    def test_missing_dims_degrade_to_nodata(self) -> None:
        # Sentinel partials (dim missing upstream) merge to NODATA.
        import geopandas as gpd
        from shapely.geometry import box

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = root / "s"
            tiledir = session / "grid_metrics"
            tiledir.mkdir(parents=True)
            frame = gpd.GeoDataFrame(
                {
                    "cell_col": [0],
                    "cell_row": [0],
                    "gid": ["0_0"],
                    "n": [4],
                    "z_sum": [1606.0],
                    "z_sumsq": [1606.0**2 / 4 + 5.0],
                    "z_min": [400.0],
                    "z_max": [403.0],
                    "ground": [-1],
                    "noise": [-1],
                    "i_sum": [0.0],
                    "i_sumsq": [0.0],
                    "i_n": [-1],
                    "first_ret": [4],
                    "last_ret": [4],
                    "i_min": [float("inf")],
                    "i_max": [float("-inf")],
                    **{f"class_{v}": [-1] for v in (0, 1, 2, 3, 4, 5, 6, 7, 9, 17, 18)},
                },
                geometry=[box(0, 0, 10, 10)],
                crs=None,
            )
            frame.to_file(str(tiledir / "t.gpkg"), driver="GPKG")
            out = barrier_grid_metrics(
                {"session_dir": str(session), "node_iid": "gm",
                 "resolution": 10.0}
            )
            self.assertTrue(out.get("warnings"))
            result = gpd.read_file(out["file"])
            self.assertEqual(result.iloc[0]["ground_count"], -9999.0)
            self.assertEqual(result.iloc[0]["intensity_mean"], -9999.0)
            self.assertEqual(result.iloc[0]["point_count"], 4)

    def test_unknown_crs_writes_without_projection(self) -> None:
        from lynceus.nodes._point_source import UNKNOWN_CRS

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = _write_cloud(
                root / "t.laz", [1.0, 2.0], [1.0, 2.0], [400.0, 401.0],
                [2, 0],
            )
            ctx = _tile_ctx(root / "s", crs=UNKNOWN_CRS)
            out = tile_grid_metrics(_tile_dict("t", src), ctx)
            frame = gpd.read_file(out["output"])
            self.assertGreater(len(frame), 0)


if __name__ == "__main__":
    unittest.main()
