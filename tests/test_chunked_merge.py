# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Unit tests for bounded table/vector merges: CSV/GPKG streaming concat,
sqlite keyed aggregation, grid-metrics legacy-vs-stream parity and the
fit check.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from lynceus.processing.chunked_merge import (
    append_gpkg,
    concat_csv,
    concat_gpkg,
    init_gpkg,
    stream_keyed_groups,
    tables_fit,
)


def _frames(tmp: Path, n_files: int = 3, rows: int = 5):
    paths = []
    for i in range(n_files):
        gdf = gpd.GeoDataFrame(
            {
                "gid": [f"{i}_{r}" for r in range(rows)],
                "v": [i * 100 + r for r in range(rows)],
                "w": [float(i) + r * 0.5 for r in range(rows)],
            },
            geometry=[Point(i, r) for r in range(rows)],
            crs="EPSG:32621",
        )
        path = tmp / f"part{i}.gpkg"
        gdf.to_file(path, driver="GPKG")
        paths.append(path)
    return paths


class ConcatTests(unittest.TestCase):
    def setUp(self) -> None:
        import shutil

        self.tmp = Path(tempfile.mkdtemp(prefix="concat_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_csv_union_headers_streaming(self) -> None:
        for i in range(2):
            pd.DataFrame({"a": [i], "b": [f"x{i}"]}).to_csv(
                self.tmp / f"t{i}.csv", index=False)
        pd.DataFrame({"b": ["y"], "c": [3]}).to_csv(
            self.tmp / "t2.csv", index=False)
        res = concat_csv([self.tmp / f"t{i}.csv" for i in range(3)],
                         self.tmp / "out.csv")
        self.assertEqual(res["rows"], 3)
        self.assertEqual(res["headers"], ["a", "b", "c"])
        back = pd.read_csv(self.tmp / "out.csv")
        self.assertEqual(len(back), 3)

    def test_gpkg_concat_matches_inputs(self) -> None:
        paths = _frames(self.tmp)
        res = concat_gpkg(paths, self.tmp / "out.gpkg", chunk_features=2)
        self.assertEqual(res["rows"], 15)
        back = gpd.read_file(self.tmp / "out.gpkg")
        self.assertEqual(len(back), 15)
        self.assertEqual(set(back["gid"]), {f"{i}_{r}" for i in range(3)
                                            for r in range(5)})

    def test_append_helpers(self) -> None:
        paths = _frames(self.tmp, n_files=1)
        sample = gpd.read_file(paths[0], rows=1)
        init_gpkg(self.tmp / "a.gpkg", sample)
        n = append_gpkg(self.tmp / "a.gpkg", gpd.read_file(paths[0]))
        self.assertEqual(n, 5)
        self.assertEqual(len(gpd.read_file(self.tmp / "a.gpkg")), 5)
        self.assertEqual(append_gpkg(self.tmp / "a.gpkg", None), 0)

    def test_tables_fit(self) -> None:
        paths = _frames(self.tmp, n_files=1)
        self.assertTrue(tables_fit(paths, 4.0, 10 ** 12))
        self.assertTrue(tables_fit(paths, 4.0, None))
        self.assertFalse(tables_fit(paths, 4.0, 1))


class KeyedGroupsTests(unittest.TestCase):
    def setUp(self) -> None:
        import shutil

        self.tmp = Path(tempfile.mkdtemp(prefix="keyed_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_sums_exact_ints_allclose_floats(self) -> None:
        paths = _frames(self.tmp)
        legacy = pd.concat(
            [gpd.read_file(p).drop(columns=["geometry"]) for p in paths],
            ignore_index=True,
        ).groupby("gid", sort=True)
        ref_sum = legacy["v"].sum()
        ref_min = legacy["w"].min()

        def reader(path):
            return gpd.read_file(path).drop(columns=["geometry"])

        batches = list(stream_keyed_groups(
            paths, key="gid",
            columns={"v": ("int64", "sum"), "w": ("float64", "min")},
            reader=reader, batch_rows=4))
        got = pd.concat(batches).set_index("gid").sort_index()
        self.assertEqual(got["v"].tolist(), ref_sum.tolist())
        self.assertTrue(
            np.allclose(got["w"].to_numpy(), ref_min.to_numpy()))

    def test_empty_inputs_yield_nothing(self) -> None:
        empty = self.tmp / "empty.gpkg"
        gpd.GeoDataFrame({"gid": [], "v": []}).to_file(empty, driver="GPKG")
        out = list(stream_keyed_groups(
            [empty], key="gid", columns={"v": ("int64", "sum")},
            reader=lambda p: gpd.read_file(p)))
        self.assertEqual(out, [])


class GridBarrierParityTests(unittest.TestCase):
    def setUp(self) -> None:
        import shutil

        self.tmp = Path(tempfile.mkdtemp(prefix="gridpar_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        arts = self.tmp / "arts"
        (arts / "grid_metrics").mkdir(parents=True)
        rng = np.random.default_rng(7)
        for t in range(4):
            rows = []
            for _ in range(60):
                c, r = int(rng.integers(0, 20)), int(rng.integers(0, 20))
                rows.append({
                    "cell_col": c, "cell_row": r, "gid": f"{c}_{r}",
                    "n": int(rng.integers(1, 50)),
                    "z_sum": float(rng.random() * 100),
                    "z_sumsq": float(rng.random() * 1000),
                    "z_min": float(rng.random() * 10),
                    "z_max": float(rng.random() * 10 + 10),
                    "ground": int(rng.integers(0, 20)),
                    "noise": int(rng.integers(0, 5)),
                    "i_sum": float(rng.random() * 500),
                    "i_sumsq": float(rng.random() * 5000),
                    "i_n": int(rng.integers(0, 30)),
                    "first_ret": int(rng.integers(0, 20)),
                    "last_ret": int(rng.integers(0, 20)),
                    "i_min": float(rng.random() * 50),
                    "i_max": float(rng.random() * 50 + 50),
                    **{f"class_{v}": int(rng.integers(0, 10))
                       for v in (0, 1, 2, 3, 4, 5, 6, 7, 9, 17, 18)},
                })
            frame = pd.DataFrame(rows)
            gdf = gpd.GeoDataFrame(
                frame,
                geometry=gpd.GeoSeries.from_xy(
                    frame["cell_col"] + 0.5, frame["cell_row"] + 0.5,
                    crs="EPSG:32621"),
                crs="EPSG:32621",
            )
            gdf.to_file(arts / "grid_metrics" / f"t{t}.gpkg", driver="GPKG")
        self.base_ctx = {
            "session_dir": str(arts),
            "node_iid": "g1",
            "crs": "EPSG:32621",
            "resolution": 1.0,
        }

    def _run(self, budget):
        from lynceus.nodes.lidar.analysis.grid_metrics import (
            barrier_grid_metrics,
        )

        ctx = dict(self.base_ctx)
        ctx["_raster_grid_memory_budget_bytes"] = budget
        return barrier_grid_metrics(ctx)

    def _frame(self, payload):
        frame = gpd.read_file(payload["file"]).sort_values("gid")
        return frame.reset_index(drop=True)

    def test_stream_matches_direct(self) -> None:
        direct = self._frame(self._run(10 ** 15))
        stream = self._frame(self._run(1))
        self.assertEqual(list(direct.columns), list(stream.columns))
        self.assertEqual(direct["gid"].tolist(), stream["gid"].tolist())
        for col in direct.columns:
            if col in ("gid", "geometry"):
                continue
            a = direct[col].to_numpy()
            b = stream[col].to_numpy()
            if np.issubdtype(a.dtype, np.integer):
                self.assertTrue(np.array_equal(a, b), col)
            else:
                self.assertTrue(
                    np.allclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True),
                    col)

    def test_csv_twin_and_sidecar_match(self) -> None:
        import json

        direct = self._run(10 ** 15)
        stream = self._run(1)
        for payload in (direct, stream):
            twin = Path(payload["csv"])
            self.assertTrue(twin.is_file())
            meta = json.loads(Path(payload["file"]).with_suffix(
                ".meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["n_cells"],
                             len(gpd.read_file(payload["file"])))
        d_rows = len(pd.read_csv(direct["csv"]))
        s_rows = len(pd.read_csv(stream["csv"]))
        self.assertEqual(d_rows, s_rows)


class ConsolidateParityTests(unittest.TestCase):
    """Segment-final merges: direct vs stream, same cells and values."""

    def setUp(self) -> None:
        import shutil

        self.tmp = Path(tempfile.mkdtemp(prefix="conspar_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        from lynceus.nodes.lidar.analysis.grid_metrics import (
            barrier_grid_metrics,
        )

        self.segment_files = []
        for s in range(2):
            arts = self.tmp / f"seg{s}" / "artifacts"
            (arts / "grid_metrics").mkdir(parents=True)
            rng = np.random.default_rng(100 + s)
            rows = []
            for _ in range(40):
                c, r = int(rng.integers(0, 10)), int(rng.integers(0, 10))
                rows.append({
                    "cell_col": c, "cell_row": r, "gid": f"{c}_{r}",
                    "n": int(rng.integers(1, 30)),
                    "z_sum": float(rng.random() * 100),
                    "z_sumsq": float(rng.random() * 1000),
                    "z_min": float(rng.random() * 10),
                    "z_max": float(rng.random() * 10 + 10),
                    "ground": int(rng.integers(0, 20)),
                    "noise": int(rng.integers(0, 5)),
                    "i_sum": float(rng.random() * 500),
                    "i_sumsq": float(rng.random() * 5000),
                    "i_n": int(rng.integers(0, 30)),
                    "first_ret": int(rng.integers(0, 20)),
                    "last_ret": int(rng.integers(0, 20)),
                    "i_min": float(rng.random() * 50),
                    "i_max": float(rng.random() * 50 + 50),
                    **{f"class_{v}": int(rng.integers(0, 10))
                       for v in (0, 1, 2, 3, 4, 5, 6, 7, 9, 17, 18)},
                })
            frame = pd.DataFrame(rows)
            gdf = gpd.GeoDataFrame(
                frame,
                geometry=gpd.GeoSeries.from_xy(
                    frame["cell_col"] + 0.5, frame["cell_row"] + 0.5,
                    crs="EPSG:32621"),
                crs="EPSG:32621",
            )
            gdf.to_file(arts / "grid_metrics" / f"t{s}.gpkg", driver="GPKG")
            ctx = {
                "session_dir": str(arts),
                "node_iid": "g1",
                "crs": "EPSG:32621",
                "resolution": 1.0,
            }
            payload = barrier_grid_metrics(ctx)
            self.segment_files.append(payload["file"])

    def _run(self, budget, tag):
        from lynceus.nodes.flow.consolidate_grid_metrics import (
            barrier_consolidate_grid_metrics,
        )

        final = self.tmp / tag / "final"
        final.mkdir(parents=True)
        ctx = {
            "node_iid": "c1",
            "module_id": "lynceus.nodes.flow.consolidate",
            "session_dir": str(final),
            "segment_outputs": [
                {"gm": {"file": str(self.segment_files[0])}},
                {"gm": {"file": str(self.segment_files[1])}},
            ],
            "segment_providers_by_port": {
                "c1": {"grid_metrics": ["gm"]}
            },
            "_raster_grid_memory_budget_bytes": budget,
        }
        return barrier_consolidate_grid_metrics(ctx)

    def test_stream_matches_direct(self) -> None:
        direct = gpd.read_file(self._run(10 ** 15, "d")["file"])
        stream = gpd.read_file(self._run(1, "s")["file"])
        direct = direct.sort_values("gid").reset_index(drop=True)
        stream = stream.sort_values("gid").reset_index(drop=True)
        self.assertEqual(list(direct.columns), list(stream.columns))
        self.assertEqual(direct["gid"].tolist(), stream["gid"].tolist())
        self.assertTrue(bool((direct.geometry == stream.geometry).all()))
        for col in direct.columns:
            if col in ("gid", "geometry"):
                continue
            a = direct[col].to_numpy()
            b = stream[col].to_numpy()
            if np.issubdtype(a.dtype, np.integer):
                self.assertTrue(np.array_equal(a, b), col)
            else:
                self.assertTrue(
                    np.allclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True),
                    col)


class VegetationParityTests(unittest.TestCase):
    """Vegetation merges: direct vs stream, same histograms and finals."""

    def setUp(self) -> None:
        import shutil

        self.tmp = Path(tempfile.mkdtemp(prefix="vegpar_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _partials(self, root: Path, seed: int, n_files: int = 3):
        from lynceus.nodes.lidar.analysis.grid_vegetation import HIST_BINS

        arts = root / "grid_vegetation"
        arts.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed)
        for t in range(n_files):
            rows = []
            for _ in range(30):
                c, r = int(rng.integers(0, 8)), int(rng.integers(0, 8))
                rows.append({
                    "cell_col": c, "cell_row": r, "gid": f"{c}_{r}",
                    **{f"h{b}": int(rng.integers(0, 20))
                       for b in range(HIST_BINS)},
                })
            frame = pd.DataFrame(rows)
            gdf = gpd.GeoDataFrame(
                frame,
                geometry=gpd.GeoSeries.from_xy(
                    frame["cell_col"] + 0.5, frame["cell_row"] + 0.5,
                    crs="EPSG:32621"),
                crs="EPSG:32621",
            )
            gdf.to_file(arts / f"t{t}.gpkg", driver="GPKG")
        return arts

    def _frame(self, payload):
        frame = gpd.read_file(payload["file"]).sort_values("gid")
        return frame.reset_index(drop=True)

    def _check_equal(self, direct, stream) -> None:
        self.assertEqual(list(direct.columns), list(stream.columns))
        self.assertEqual(direct["gid"].tolist(), stream["gid"].tolist())
        self.assertTrue(bool((direct.geometry == stream.geometry).all()))
        for col in direct.columns:
            if col in ("gid", "geometry"):
                continue
            a = direct[col].to_numpy()
            b = stream[col].to_numpy()
            if np.issubdtype(a.dtype, np.integer):
                self.assertTrue(np.array_equal(a, b), col)
            else:
                self.assertTrue(
                    np.allclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True),
                    col)

    def test_barrier_matches(self) -> None:
        from lynceus.nodes.lidar.analysis.grid_vegetation import (
            barrier_grid_vegetation,
        )

        arts = self.tmp / "arts"
        self._partials(arts, seed=11)
        base = {
            "session_dir": str(arts),
            "node_iid": "v1",
            "crs": "EPSG:32621",
            "resolution": 1.0,
        }
        direct_ctx = dict(base, _raster_grid_memory_budget_bytes=10 ** 15)
        stream_ctx = dict(base, _raster_grid_memory_budget_bytes=1)
        direct = self._frame(barrier_grid_vegetation(direct_ctx))
        stream = self._frame(barrier_grid_vegetation(stream_ctx))
        self._check_equal(direct, stream)

    def test_consolidate_matches(self) -> None:
        from lynceus.nodes.flow.consolidate_vegetation import (
            barrier_consolidate_vegetation,
        )
        from lynceus.nodes.lidar.analysis.grid_vegetation import (
            barrier_grid_vegetation,
        )

        seg_files = []
        for s in range(2):
            arts = self.tmp / f"seg{s}" / "artifacts"
            self._partials(arts, seed=21 + s, n_files=2)
            ctx = {
                "session_dir": str(arts),
                "node_iid": "v1",
                "crs": "EPSG:32621",
                "resolution": 1.0,
            }
            seg_files.append(barrier_grid_vegetation(ctx)["file"])
        final = self.tmp / "final"
        final.mkdir(parents=True)
        base = {
            "node_iid": "c1",
            "module_id": "lynceus.nodes.flow.consolidate",
            "session_dir": str(final),
            "segment_outputs": [
                {"gm": {"file": str(seg_files[0])}},
                {"gm": {"file": str(seg_files[1])}},
            ],
            "segment_providers_by_port": {
                "c1": {"grid_vegetation": ["gm"]}
            },
        }
        direct = self._frame(barrier_consolidate_vegetation(dict(
            base, _raster_grid_memory_budget_bytes=10 ** 15)))
        stream = self._frame(barrier_consolidate_vegetation(dict(
            base, _raster_grid_memory_budget_bytes=1)))
        self._check_equal(direct, stream)


if __name__ == "__main__":
    unittest.main()
