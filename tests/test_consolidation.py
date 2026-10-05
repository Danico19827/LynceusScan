"""Unit tests for consolidator variants: discovery, deferred reachability,
chained barrier deps, per-segment fingerprint salting, per-product merges,
keyed grid merge (no segment seams) and segment tagging in the gallery.

Stdlib + engine modules only (no Qt, no real runs).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lynceus.nodes._variants import resolved_ports
from lynceus.nodes.ports import PortType, port_metadata
from lynceus.plugins.registry import manager
from lynceus.processing.controller import PipelineController
from lynceus.processing.steps import (
    build_dag,
    discover_node_capabilities,
    effective_node_caps,
)

LOAD = "lynceus.nodes.lidar.source.load_las_laz"
ELEV = "lynceus.nodes.lidar.clean.elevation_range"
DTM = "lynceus.nodes.lidar.terrain.generate_dtm"
CONS = "lynceus.nodes.flow.consolidate"
PC = PortType.POINT_CLOUD
DTM_MOSAIC = PortType.DTM_MOSAIC

DTM_CFG = {"strategy": "dtm"}
CHM_CFG = {"strategy": "chm"}

NODES = {
    "loader": LOAD,
    "elev": ELEV,
    "dtm": DTM,
    "c1": CONS,
    "c2": CONS,
}
EDGES = [
    ("loader", "elev", PC, PC),
    ("elev", "dtm", PC, PC),
    ("dtm", "c1", DTM_MOSAIC, DTM_MOSAIC),
    ("c1", "c2", DTM_MOSAIC, DTM_MOSAIC),
]
CONFIGS = {"c1": dict(DTM_CFG), "c2": dict(DTM_CFG)}
TILES = [{"tile_id": "tag_c0000_r0000", "file": "x.laz", "src_iid": "loader",
          "point_count": 10}]


class ConsolidateVariantTests(unittest.TestCase):
    def test_base_is_inert_without_strategy(self) -> None:
        manager.discover()
        info = manager.node_info(CONS)
        self.assertIsNotNone(info)
        self.assertEqual(info.category, "Flow")
        caps = discover_node_capabilities(CONS)
        self.assertFalse(caps.get("consolidator"))
        self.assertIsNone(caps.get("consolidate_task"))
        inputs, outputs = resolved_ports(CONS, {})
        self.assertEqual((inputs, outputs), ((), ()))

    def test_seven_variants_discovered(self) -> None:
        manager.discover()
        keys = sorted(v["key"] for v in manager.list_variants(CONS))
        self.assertEqual(
            keys, ["chm", "dsm", "dtm", "grid_metrics", "table", "vector",
                   "vegetation"]
        )

    def test_variant_caps_carry_consolidator(self) -> None:
        for key, task in (
            ("chm", "barrier_consolidate_chm"),
            ("dsm", "barrier_consolidate_dsm"),
            ("dtm", "barrier_consolidate_dtm"),
            ("grid_metrics", "barrier_consolidate_grid_metrics"),
            ("vegetation", "barrier_consolidate_vegetation"),
            ("vector", "barrier_consolidate_vector"),
            ("table", "barrier_consolidate_table"),
        ):
            with self.subTest(variant=key):
                caps = effective_node_caps(CONS, {"strategy": key})
                self.assertTrue(caps["consolidator"])
                self.assertEqual(caps["consolidate_task"].__name__, task)

    def test_variant_ports_are_single_required(self) -> None:
        inputs, outputs = resolved_ports(CONS, dict(DTM_CFG))
        self.assertEqual(len(inputs), 1)
        self.assertEqual(len(outputs), 1)
        self.assertTrue(port_metadata(inputs[0]).required)
        self.assertEqual(port_metadata(inputs[0]).port_type.value, "dtm_mosaic")
        self.assertEqual(outputs[0].value, "dtm_mosaic")

    def test_strategies_hash_differently(self) -> None:
        ctrl = PipelineController()
        ctrl._modules = {"c1": CONS}
        fps = ctrl._node_fingerprints(
            ["c1"], [], {"c1": {"strategy": "dtm"}},
            {}, scope="seg0000:abc",
        )
        other = ctrl._node_fingerprints(
            ["c1"], [], {"c1": {"strategy": "chm"}},
            {}, scope="seg0000:abc",
        )
        self.assertNotEqual(fps["c1"], other["c1"])


class DeferredTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ctrl = PipelineController()
        self.ctrl._modules = dict(NODES)

    def test_reachability_without_consolidators_is_empty(self) -> None:
        self.assertEqual(self.ctrl._deferred_nodes(["a", "b"], [], {}), set())

    def test_consolidator_and_downstream_are_deferred(self) -> None:
        deferred = self.ctrl._deferred_nodes(list(NODES), EDGES, dict(CONFIGS))
        self.assertIn("c1", deferred)
        self.assertIn("c2", deferred)
        self.assertNotIn("dtm", deferred)
        self.assertNotIn("loader", deferred)

    def test_inert_base_without_strategy_is_not_deferred(self) -> None:
        deferred = self.ctrl._deferred_nodes(list(NODES), EDGES, {})
        self.assertNotIn("c1", deferred)
        self.assertNotIn("c2", deferred)

    def test_segment_session_edges_exclude_deferred(self) -> None:
        deferred = self.ctrl._deferred_nodes(list(NODES), EDGES, dict(CONFIGS))
        segment_edges = [
            e for e in EDGES if e[0] not in deferred and e[1] not in deferred
        ]
        self.assertEqual(
            segment_edges,
            [
                ("loader", "elev", PC, PC),
                ("elev", "dtm", PC, PC),
            ],
        )


class DagTests(unittest.TestCase):
    def test_deferred_subgraph_builds_consolidate_barriers(self) -> None:
        tasks = build_dag(
            ["c1", "c2"],
            [("c1", "c2", DTM_MOSAIC, DTM_MOSAIC)],
            [],
            {"session_dir": "unused"},
            modules={"c1": CONS, "c2": CONS},
            configs=dict(CONFIGS),
        )
        ids = sorted(t.task_id for t in tasks)
        self.assertEqual(ids, ["c1|merge", "c2|merge"])
        c2 = next(t for t in tasks if t.task_id == "c2|merge")
        self.assertEqual(c2.deps, {"c1|merge"})

    def test_chained_consolidator_waits_for_provider_merge(self) -> None:
        tasks = build_dag(
            ["dtm", "c1", "c2"],
            [
                ("dtm", "c1", DTM_MOSAIC, DTM_MOSAIC),
                ("c1", "c2", DTM_MOSAIC, DTM_MOSAIC),
            ],
            [],
            {"session_dir": "unused"},
            modules={"dtm": DTM, "c1": CONS, "c2": CONS},
            configs=dict(CONFIGS),
        )
        c2 = next(t for t in tasks if t.task_id == "c2|merge")
        self.assertIn("c1|merge", c2.deps)
        self.assertNotIn("dtm|merge", c2.deps)


class FingerprintTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ctrl = PipelineController()
        self.ctrl._modules = {"c1": CONS}

    def test_seg_sig_salts_the_consolidator_fingerprint(self) -> None:
        fp1 = self.ctrl._node_fingerprints(
            ["c1"], [], {}, {}, seg_sig="segs:aaa"
        )
        fp2 = self.ctrl._node_fingerprints(
            ["c1"], [], {}, {}, seg_sig="segs:bbb"
        )
        fp3 = self.ctrl._node_fingerprints(
            ["c1"], [], {}, {}, seg_sig=""
        )
        self.assertNotEqual(fp1["c1"], fp2["c1"])
        self.assertNotEqual(fp3["c1"], fp1["c1"])

    def test_operator_salts_fingerprints(self) -> None:
        base = self.ctrl._node_fingerprints(["c1"], [], {}, {})
        self.ctrl._run_operator = {"name": "Nico", "org": ""}
        salted = self.ctrl._node_fingerprints(["c1"], [], {}, {})
        self.assertNotEqual(base["c1"], salted["c1"])
        self.ctrl._run_operator = None
        again = self.ctrl._node_fingerprints(["c1"], [], {}, {})
        self.assertEqual(base["c1"], again["c1"])


class ConsolidateHelpersTests(unittest.TestCase):
    def test_segment_files_reads_deferred_provider_via_ctx_paths(self) -> None:
        import tempfile
        from pathlib import Path

        from lynceus.nodes.flow._consolidate_base import _segment_files

        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp) / "final" / "artifacts"
            session.mkdir(parents=True)
            merged = session / "c1" / "dtm_mosaic.tif"
            merged.parent.mkdir()
            merged.write_bytes(b"fake")
            files = _segment_files(
                {
                    "node_iid": "c2",
                    "module_id": CONS,
                    "session_dir": str(session),
                    "segment_outputs": [],
                    "segment_providers_by_port": {},
                    "dtm_mosaic_path": str(merged),
                    "in0_path": str(merged),
                },
                "dtm_mosaic",
            )
            self.assertEqual(files, [str(merged)])

    def test_segment_files_matches_provider_by_input_type(self) -> None:
        import tempfile
        from pathlib import Path

        from lynceus.nodes.flow._consolidate_base import _segment_files

        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp) / "artifacts"
            session.mkdir(parents=True)
            provider_dir = session / "dtm"
            provider_dir.mkdir()
            seg_file = provider_dir / "dtm_mosaic.tif"
            seg_file.write_bytes(b"fake")
            seg_out = {"dtm": {"file": str(seg_file)}}
            files = _segment_files(
                {
                    "node_iid": "c1",
                    "module_id": CONS,
                    "session_dir": str(Path(tmp) / "final"),
                    "segment_outputs": [seg_out],
                    "segment_providers_by_port": {"c1": {
                        "dtm_mosaic": ["dtm"],
                        "vector": ["grid"],
                    }},
                },
                "dtm_mosaic",
            )
            self.assertEqual(files, [str(seg_file)])

    def _write_tif(self, path: Path, value: float) -> None:
        import numpy as np
        import rasterio

        profile = {
            "driver": "GTiff",
            "height": 2,
            "width": 2,
            "count": 1,
            "dtype": "float32",
            "crs": "EPSG:4326",
            "transform": rasterio.Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
            "nodata": -9999.0,
        }
        with rasterio.open(str(path), "w", **profile) as ds:
            ds.write(np.full((1, 2, 2), value, dtype="float32"))

    def test_segment_files_without_providers_raises(self) -> None:
        from lynceus.nodes.flow._consolidate_base import _segment_files

        with self.assertRaises(RuntimeError):
            _segment_files(
                {
                    "node_iid": "c1",
                    "module_id": CONS,
                    "session_dir": "/tmp/final/artifacts",
                    "segment_outputs": [{}],
                    "segment_providers_by_port": {
                        "c1": {"dtm_mosaic": ["dtm"]}
                    },
                },
                "dtm_mosaic",
            )

    def test_dtm_variant_merges_only_its_type(self) -> None:
        import tempfile
        from pathlib import Path

        from lynceus.nodes.flow.consolidate_dtm import barrier_consolidate_dtm

        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp) / "artifacts"
            session.mkdir(parents=True)
            (session / "dtm").mkdir()
            dtm_tif = session / "dtm" / "dtm_mosaic.tif"
            self._write_tif(dtm_tif, 10.0)
            seg_out = {"dtm": {"file": str(dtm_tif)}}
            ctx = {
                "node_iid": "c1",
                "module_id": CONS,
                "session_dir": str(Path(tmp) / "final"),
                "crs": "EPSG:4326",
                "segment_outputs": [seg_out],
                "segment_providers_by_port": {
                    "c1": {"dtm_mosaic": ["dtm"]}
                },
            }
            payload = barrier_consolidate_dtm(ctx)
            self.assertTrue(Path(payload["file"]).is_file())
            self.assertEqual(payload["display_name"], "DTM")

    def test_variant_without_providers_raises(self) -> None:
        from lynceus.nodes.flow.consolidate_dtm import barrier_consolidate_dtm

        with self.assertRaises(RuntimeError):
            barrier_consolidate_dtm(
                {
                    "node_iid": "c1",
                    "module_id": CONS,
                    "session_dir": "/tmp/final",
                    "segment_outputs": [],
                    "segment_providers_by_port": {},
                }
            )

    def _write_grid_final(
        self, path: Path, rows: list[tuple],
    ) -> None:
        import geopandas as gpd
        from shapely.geometry import box

        gdf = gpd.GeoDataFrame(
            {
                "gid": [r[0] for r in rows],
                "cell_col": [r[1] for r in rows],
                "cell_row": [r[2] for r in rows],
                "point_count": [r[3] for r in rows],
                "density": [r[3] / 100.0 for r in rows],
                "z_min": [r[4] for r in rows],
                "z_max": [r[5] for r in rows],
                "z_mean": [r[6] for r in rows],
                "z_std": [0.0 for r in rows],
                "ground_count": [r[7] for r in rows],
                "ground_cover": [r[7] / r[3] for r in rows],
                "intensity_mean": [r[8] for r in rows],
                "intensity_std": [0.0 for r in rows],
                "noise_count": [0 for r in rows],
                "z_sumsq": [r[6] ** 2 * r[3] for r in rows],
                "i_sum": [r[8] * r[3] for r in rows],
                "i_sumsq": [r[8] ** 2 * r[3] for r in rows],
                "i_n": [r[3] for r in rows],
                "first_ret": [r[3] for r in rows],
                "last_ret": [r[3] for r in rows],
                "i_min": [r[8] for r in rows],
                "i_max": [r[8] for r in rows],
                "dominant_class": [2 for r in rows],
                "intensity_min": [r[8] for r in rows],
                "intensity_max": [r[8] for r in rows],
                "z_range": [r[5] - r[4] for r in rows],
                "first_return_cover": [1.0 for r in rows],
                "last_return_cover": [1.0 for r in rows],
                "veg_cover": [0.0 for r in rows],
                "building_cover": [0.0 for r in rows],
                "water_cover": [0.0 for r in rows],
                **{f"class_{v}": [0 for r in rows]
                   for v in (0, 1, 2, 3, 4, 5, 6, 7, 9, 17, 18)},
            },
            geometry=[box(r[1] * 10.0, r[2] * 10.0,
                          (r[1] + 1) * 10.0, (r[2] + 1) * 10.0) for r in rows],
            crs="EPSG:32621",
        )
        gdf.to_file(str(path), driver="GPKG")

    def test_grid_variant_merges_shared_cells_exactly(self) -> None:
        """No seams: a cell split across segments merges arithmetically."""
        import tempfile
        from pathlib import Path

        from lynceus.nodes.flow.consolidate_grid_metrics import (
            barrier_consolidate_grid_metrics,
        )

        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp)
            # Same global cell from two segments: partials, not finals.
            s1 = session / "s1.gpkg"
            s2 = session / "s2.gpkg"
            # gid, col, row, n, zmin, zmax, zmean, ground, imean
            self._write_grid_final(s1, [("0_0", 0, 0, 6, 400.0, 402.0, 401.0, 4, 100.0)])
            self._write_grid_final(s2, [("0_0", 0, 0, 4, 401.0, 403.0, 402.0, 1, 300.0),
                                        ("1_0", 1, 0, 2, 410.0, 411.0, 410.5, 0, 50.0)])
            ctx = {
                "node_iid": "c1",
                "module_id": CONS,
                "session_dir": str(session / "final"),
                "segment_outputs": [
                    {"gm": {"file": str(s1)}},
                    {"gm": {"file": str(s2)}},
                ],
                "segment_providers_by_port": {
                    "c1": {"grid_metrics": ["gm"]}
                },
            }
            payload = barrier_consolidate_grid_metrics(ctx)
            self.assertEqual(payload["display_name"], "Grid Metrics")
            import geopandas as gpd

            frame = gpd.read_file(payload["file"])
            self.assertEqual(len(frame), 2)
            row = frame.set_index("gid").to_dict("index")["0_0"]
            self.assertEqual(row["point_count"], 10)
            self.assertAlmostEqual(row["z_mean"], 401.4, places=4)
            self.assertAlmostEqual(row["z_min"], 400.0, places=4)
            self.assertAlmostEqual(row["z_max"], 403.0, places=4)
            self.assertAlmostEqual(row["ground_cover"], 0.5, places=6)
            self.assertAlmostEqual(row["intensity_mean"], 180.0, places=4)
            self.assertEqual(row["cell_col"], 0)


class GalleryTagTests(unittest.TestCase):
    def _touch(self, tmp: str, name: str) -> str:
        path = Path(tmp) / name
        path.write_bytes(b"x")
        return str(path)

    def test_flatten_outputs_tags_segment_products(self) -> None:
        from lynceus.ui.outputs_model import flatten_outputs

        with tempfile.TemporaryDirectory() as tmp:
            products = flatten_outputs(
                {"dtm": {"file": self._touch(tmp, "dtm.tif")}},
                {"dtm": "DTM"},
                segment="Segment 2/5",
                segment_index=2,
            )
        self.assertEqual(len(products), 1)
        self.assertEqual(products[0].segment, "Segment 2/5")
        self.assertEqual(products[0].segment_index, 2)

    def test_flatten_outputs_without_segment_is_untagged(self) -> None:
        from lynceus.ui.outputs_model import flatten_outputs

        with tempfile.TemporaryDirectory() as tmp:
            products = flatten_outputs(
                {"dtm": {"file": self._touch(tmp, "dtm.tif")}},
                {"dtm": "DTM"},
            )
        self.assertIsNone(products[0].segment)

    def test_flatten_outputs_uses_display_name(self) -> None:
        from lynceus.ui.outputs_model import flatten_outputs

        with tempfile.TemporaryDirectory() as tmp:
            products = flatten_outputs(
                {
                    "gm": {
                        "file": self._touch(tmp, "grid.gpkg"),
                        "display_name": "Grid Metrics",
                    }
                },
                {"gm": "Consolidate"},
            )
        self.assertEqual(products[0].name, "Grid Metrics")

    def test_flatten_outputs_skips_missing_files(self) -> None:
        from lynceus.ui.outputs_model import flatten_outputs

        with tempfile.TemporaryDirectory() as tmp:
            missing = str(Path(tmp) / "ghost.tif")
            products = flatten_outputs(
                {"dtm": {"file": missing}}, {"dtm": "DTM"}
            )
        self.assertEqual(products, [])


class ConsolidateProvenanceTests(unittest.TestCase):
    """Merged products carry provenance stamps (tags/sidecars/metadata)."""

    def _ctx(self, session, extra: dict) -> dict:
        ctx = {
            "node_iid": "c1",
            "module_id": "lynceus.nodes.flow.consolidate",
            "session_dir": str(session),
        }
        ctx.update(extra)
        return ctx

    def _write_tif(self, path: Path, value: float) -> None:
        import numpy as np
        import rasterio
        from affine import Affine

        arr = np.full((8, 8), value, dtype=np.float32)
        with rasterio.open(
            str(path), "w", driver="GTiff", height=8, width=8, count=1,
            dtype="float32", crs="EPSG:32621",
            transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 8.0),
            nodata=-9999.0,
        ) as dst:
            dst.write(arr, 1)

    def test_consolidate_mosaic_embeds_tag(self) -> None:
        import json
        import tempfile

        import rasterio
        from lynceus.nodes.flow._consolidate_base import consolidate_mosaic

        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp)
            f1 = session / "s1.tif"
            f2 = session / "s2.tif"
            self._write_tif(f1, 10.0)
            self._write_tif(f2, 20.0)
            ctx = self._ctx(session, {
                "chm_mosaic_path": str(f1),
                "in0_path": str(f2),
                "provenance": {
                    "operator": {"name": "Nico", "org": ""},
                },
            })
            out = consolidate_mosaic(
                ctx, session_file="chm_mosaic.tif", input_type="chm_mosaic"
            )
            with rasterio.open(out["file"]) as src:
                tag = src.tags().get("lynceus_provenance", "")
            self.assertTrue(tag)
            doc = json.loads(tag)
            self.assertEqual(doc["schema"], "lynceus-provenance")
            self.assertEqual(doc["operator"], {"name": "Nico", "org": ""})
            self.assertEqual(doc["product"], {"name": "chm_mosaic.tif"})

    def test_consolidate_vectors_embed_and_carry_sidecar(self) -> None:
        import json
        import tempfile

        import geopandas as gpd
        from shapely.geometry import box

        from lynceus.nodes.flow._consolidate_base import consolidate_vectors

        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp)
            frames = []
            for i, name in enumerate(("s1.gpkg", "s2.gpkg")):
                path = session / name
                gdf = gpd.GeoDataFrame(
                    {"h_mean": [float(i + 1)], "cell_col": [i]},
                    geometry=[box(i * 20.0, 0.0, (i + 1) * 20.0, 20.0)],
                    crs="EPSG:32621",
                )
                gdf.to_file(str(path), driver="GPKG")
                frames.append(path)
            frames[0].with_suffix(".meta.json").write_text(
                json.dumps({
                    "metrics_computed": ["h_mean"],
                    "bookkeeping_cols": ["cell_col"],
                    "default_metric": "h_mean",
                }),
                encoding="utf-8",
            )
            ctx = self._ctx(session, {
                "segment_providers_by_port": {
                    "c1": {"grid_metrics": ["p1", "p2"]},
                },
                "segment_outputs": [
                    {"p1": {"file": str(frames[0])},
                     "p2": {"file": str(frames[1])}},
                ],
            })
            out = consolidate_vectors(
                ctx, session_file="grid_metrics.gpkg",
                input_type="grid_metrics",
            )
            self.assertTrue(Path(out["file"]).exists())
            import sqlite3

            con = sqlite3.connect(out["file"])
            try:
                rows = con.execute(
                    "SELECT metadata FROM gpkg_metadata"
                ).fetchall()
            finally:
                con.close()
            self.assertEqual(len(rows), 1)
            self.assertEqual(
                json.loads(rows[0][0])["schema"], "lynceus-provenance"
            )
            sidecar = json.loads(
                Path(out["file"]).with_suffix(".meta.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(sidecar["metrics_computed"], ["h_mean"])
            self.assertEqual(sidecar["bookkeeping_cols"], ["cell_col"])
            self.assertEqual(sidecar["default_metric"], "h_mean")
            self.assertEqual(
                sidecar["provenance"]["schema"], "lynceus-provenance"
            )

    def test_consolidate_tables_writes_sidecar(self) -> None:
        import json
        import tempfile

        from lynceus.nodes.flow._consolidate_base import consolidate_tables

        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp)
            f1 = session / "a.csv"
            f2 = session / "b.csv"
            f1.write_text("x\n1\n", encoding="utf-8")
            f2.write_text("x\n2\n", encoding="utf-8")
            ctx = self._ctx(session, {"table_path": str(f1),
                                      "in0_path": str(f2)})
            out = consolidate_tables(
                ctx, session_file="consolidated.csv", input_type="table"
            )
            sidecar = json.loads(
                Path(out["file"]).with_suffix(".meta.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(sidecar["delimiter"], ",")
            self.assertEqual(
                sidecar["provenance"]["schema"], "lynceus-provenance"
            )

    def test_consolidate_grid_carries_sidecar(self) -> None:
        import json
        import tempfile

        import geopandas as gpd
        from shapely.geometry import box

        from lynceus.nodes.flow._consolidate_base import consolidate_grid_metrics

        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp)
            path = session / "seg.gpkg"
            gdf = gpd.GeoDataFrame(
                {
                    "gid": ["0_0"],
                    "cell_col": [0],
                    "cell_row": [0],
                    "point_count": [6],
                    "density": [0.06],
                    "z_min": [400.0],
                    "z_max": [402.0],
                    "z_mean": [401.0],
                    "z_std": [0.8],
                    "ground_count": [4],
                    "ground_cover": [0.5],
                    "intensity_mean": [100.0],
                    "intensity_std": [10.0],
                    "noise_count": [0],
                "z_sumsq": [6 * 401.0**2 + 6 * 0.8**2],
                "i_sum": [600.0],
                "i_sumsq": [600.0**2 / 6 + 6 * 10.0**2],
                "i_n": [6],
                },
                geometry=[box(0, 0, 10, 10)],
                crs="EPSG:32621",
            )
            gdf.to_file(str(path), driver="GPKG")
            path.with_suffix(".meta.json").write_text(
                json.dumps({
                    "metrics_computed": ["density"],
                    "bookkeeping_cols": ["cell_col", "cell_row", "gid"],
                    "default_metric": "density",
                }),
                encoding="utf-8",
            )
            ctx = self._ctx(session, {
                "segment_providers_by_port": {
                    "c1": {"grid_metrics": ["gm"]}
                },
                "segment_outputs": [{"gm": {"file": str(path)}}],
            })
            out = consolidate_grid_metrics(
                ctx, session_file="grid_metrics.gpkg", input_type="grid_metrics"
            )
            sidecar = json.loads(
                Path(out["file"]).with_suffix(".meta.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(sidecar["metrics_computed"], ["density"])


if __name__ == "__main__":
    unittest.main()
