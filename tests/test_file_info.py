# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Tests for the File Info inspector node and the JSON tree viewer."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import laspy
import numpy as np

from lynceus.nodes.lidar.source.file_info import (
    NODE_ID,
    barrier_file_info,
)


def _wkt() -> str:
    from rasterio.crs import CRS

    return CRS.from_epsg(25831).to_wkt()


def _write_cloud(path: Path, point_format: int = 6, n: int = 12,
                 with_vlrs: bool = True) -> Path:
    header = laspy.LasHeader(point_format=point_format, version="1.4")
    header.scales = np.array([0.01, 0.01, 0.01])
    header.offsets = np.array([100.0, 200.0, 0.0])
    header.system_identifier = "TEST-SYSTEM"
    header.generating_software = "TEST-SOFT 1.0"
    if with_vlrs:
        header.vlrs.append(laspy.VLR(
            user_id="LASF_Projection", record_id=2112,
            description="OGC WKT", record_data=_wkt().encode("utf-8"),
        ))
        header.vlrs.append(laspy.VLR(
            user_id="copc", record_id=1, description="",
            record_data=b"\x00",
        ))
    record = laspy.ScaleAwarePointRecord.zeros(
        n, point_format=header.point_format,
        scales=header.scales, offsets=header.offsets,
    )
    record.x = np.linspace(100.0, 199.0, n)
    record.y = np.linspace(200.0, 299.0, n)
    record.z = np.linspace(400.0, 411.0, n)
    record.classification = np.array(
        [2, 2, 5, 5, 5, 7, 12, 18, 2, 5, 3, 4], dtype=np.uint8
    )
    record.intensity = np.array(
        [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200],
        dtype=np.uint16,
    )
    record.return_number = np.array(
        [1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1], dtype=np.uint8
    )
    record.number_of_returns = np.array(
        [1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1], dtype=np.uint8
    )
    record.withheld = np.array([False] * 11 + [True])
    record.overlap = np.array([False] * 10 + [True, False])
    record.key_point = np.array([True] + [False] * 11)
    record.synthetic = np.array([False] * 9 + [True, False, False])
    record.scanner_channel = np.array(
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=np.uint8
    )
    record.scan_angle = np.array(
        [0, 1000, -2000, 3000, -4000, 5000, -6000, 7000, -800, 900, 0, 0],
        dtype=np.int16,
    )
    record.gps_time = np.linspace(1000.0, 1011.0, n)
    with laspy.open(str(path), mode="w", header=header) as writer:
        writer.write_points(record)
    return path


def _ctx(session: Path, sources: list) -> dict:
    return {
        "session_dir": str(session),
        "node_iid": "fi",
        "provenance": {"sources": sources},
    }


def _src_entry(iid: str, path: Path, tag: str = "abc123") -> dict:
    return {"iid": iid, "file": str(path), "source_fp": "x", "tag": tag}


class FileInfoTests(unittest.TestCase):
    def test_full_report_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = _write_cloud(root / "s.laz")
            out = barrier_file_info(
                _ctx(root / "s", [_src_entry("load", src)])
            )
            doc = json.loads(Path(out["file"]).read_text(encoding="utf-8"))
            self.assertEqual(
                sorted(doc),
                ["channel_attributes", "classification_flags",
                 "derived_metrics", "file_info", "qa_qc_checks",
                 "spatial_crs", "vlr_metadata"],
            )
            info = doc["file_info"]
            self.assertEqual(info["las_version"], "1.4")
            self.assertEqual(info["pdrf_id"], 6)
            self.assertEqual(info["total_points"], 12)
            self.assertEqual(info["generator_software"], "TEST-SOFT 1.0")
            self.assertEqual(
                info["raw_size_bytes"], 12 * info["record_length_bytes"]
            )
            self.assertEqual(
                info["processing_ram_estimate_bytes"],
                4 * info["raw_size_bytes"],
            )
            crs = doc["spatial_crs"]
            self.assertEqual(crs["epsg_code"], 25831)
            self.assertIsInstance(crs["crs_wkt"], str)
            self.assertIn("25831", crs["crs_wkt"])
            self.assertEqual(crs["bounding_box"]["x_min"], 100.0)
            self.assertEqual(crs["scale"]["scale_z"], 0.01)
            ch = doc["channel_attributes"]
            self.assertTrue(ch["gps_time"]["present"])
            self.assertEqual(ch["gps_time"]["type"], "week_time")
            self.assertEqual(ch["gps_time"]["min"], 1000.0)
            self.assertEqual(ch["intensity"]["bit_depth"], 16)
            self.assertEqual(
                (ch["intensity"]["min"], ch["intensity"]["max"]), (100, 1200)
            )
            self.assertIsNone(ch["intensity"]["linearly_normalized"])
            self.assertFalse(ch["color_rgb"])
            self.assertFalse(ch["infrared_nir"])
            self.assertFalse(ch["waveform"]["present"])
            self.assertEqual(
                ch["scanner_channel"]["distribution"],
                {"0": 3, "1": 3, "2": 3, "3": 3},
            )
            # Raw angles x0.006: 0,6,-12,18,-24,30,-36,42,-4.8,5.4,0,0.
            ang = ch["scan_angle"]
            self.assertTrue(ang["available"])
            self.assertAlmostEqual(ang["min"], -36.0, places=2)
            self.assertAlmostEqual(ang["max"], 42.0, places=2)
            self.assertAlmostEqual(ang["mean"], 2.05, places=2)
            self.assertAlmostEqual(ang["extreme_pct"], 200 / 12, places=2)
            cls = doc["classification_flags"]
            self.assertEqual(
                cls["class_histogram"],
                {"2": 3, "3": 1, "4": 1, "5": 4, "7": 1, "12": 1, "18": 1},
            )
            self.assertEqual(len(cls["returns"]["counts_1_to_15"]), 15)
            self.assertEqual(sum(cls["returns"]["counts_1_to_15"]), 12)
            self.assertEqual(cls["returns"]["max_allowed"], 15)
            # Pulses ~= first returns (10): 2 multi-return points.
            self.assertAlmostEqual(
                cls["returns"]["multi_return_pct"], 20.0, places=2
            )
            qa = cls["qa_flags"]
            self.assertAlmostEqual(qa["pct_withheld"], 100 / 12, places=2)
            self.assertAlmostEqual(qa["pct_overlap"], 100 / 12, places=2)
            self.assertAlmostEqual(qa["pct_keypoint"], 100 / 12, places=2)
            self.assertAlmostEqual(qa["pct_synthetic"], 100 / 12, places=2)
            self.assertEqual(cls["flight_lines"]["total_unique"], 1)
            meta = doc["vlr_metadata"]
            self.assertGreaterEqual(meta["vlr_count"], 2)
            self.assertTrue(meta["copc_ready"])
            der = doc["derived_metrics"]
            self.assertAlmostEqual(der["coverage_area_m2"], 99.0 * 99.0)
            self.assertAlmostEqual(der["elevation_range_m"], 11.0)
            self.assertEqual(der["area_aspect"], "BLOCK_POLYGON")
            self.assertAlmostEqual(der["flight_duration_s"], 11.0)
            self.assertAlmostEqual(
                der["acquisition_rate_pts_s"], 12 / 11.0, places=4
            )
            self.assertAlmostEqual(
                der["mean_density_pts_m2"], 12 / (99.0 * 99.0), places=6
            )
            # Clean first returns: 10 firsts minus the overlapped idx 10
            # and the withheld idx 11.
            self.assertAlmostEqual(der["usgs_npd_pts_m2"], 8 / 9801.0, places=6)
            self.assertAlmostEqual(
                der["usgs_nps_m"], (9801.0 / 8) ** 0.5, places=3
            )
            self.assertAlmostEqual(
                der["ground_class2_density_pts_m2"], 3 / 9801.0, places=6
            )
            checks = doc["qa_qc_checks"]
            self.assertTrue(checks["crs_wkt_valid"])
            self.assertFalse(checks["gps_time_adjusted"])
            self.assertTrue(checks["reserved_classes_alert"])
            self.assertTrue(checks["high_noise_withheld_alert"])
            self.assertFalse(checks["usgs_nps_compliant"])

    def test_pdrf8_rgb_nir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            header = laspy.LasHeader(point_format=8, version="1.4")
            header.scales = np.array([0.01, 0.01, 0.01])
            header.offsets = np.zeros(3)
            rec = laspy.ScaleAwarePointRecord.zeros(
                4, point_format=header.point_format,
                scales=header.scales, offsets=header.offsets,
            )
            rec.x = [0.0, 1.0, 2.0, 3.0]
            rec.y = [0.0, 0.0, 0.0, 0.0]
            rec.z = [10.0, 11.0, 12.0, 13.0]
            with laspy.open(str(root / "m.laz"), mode="w",
                            header=header) as writer:
                writer.write_points(rec)
            out = barrier_file_info(
                _ctx(root / "s", [_src_entry("load", root / "m.laz")])
            )
            doc = json.loads(Path(out["file"]).read_text(encoding="utf-8"))
            ch = doc["channel_attributes"]
            self.assertTrue(ch["color_rgb"])
            self.assertTrue(ch["infrared_nir"])
            self.assertEqual(doc["file_info"]["pdrf_id"], 8)

    def test_deterministic_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = _write_cloud(root / "s.laz")
            ctx = _ctx(root / "s", [_src_entry("load", src)])
            first = Path(barrier_file_info(ctx)["file"]).read_bytes()
            second = Path(barrier_file_info(ctx)["file"]).read_bytes()
            self.assertEqual(first, second)

    def test_multi_loader_two_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = _write_cloud(root / "a.laz")
            second = _write_cloud(root / "b.laz")
            out = barrier_file_info(_ctx(root / "s", [
                _src_entry("l2", second, tag="t222"),
                _src_entry("l1", first, tag="t111"),
            ]))
            self.assertTrue(out["file"].endswith("t111_file_info.json"))
            self.assertTrue(out["file_2"].endswith("t222_file_info.json"))

    def test_no_sources_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                barrier_file_info(_ctx(Path(tmp) / "s", []))

    def test_missing_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(RuntimeError):
                barrier_file_info(_ctx(
                    root / "s", [_src_entry("load", root / "gone.laz")]
                ))

    def test_node_is_registered(self) -> None:
        from lynceus.plugins.registry import manager
        from lynceus.processing.steps import discover_node_capabilities

        manager.discover()
        self.assertIsNotNone(manager.node_info(NODE_ID))
        caps = discover_node_capabilities(NODE_ID)
        self.assertEqual(
            getattr(caps.get("barrier_task"), "__name__", None),
            "barrier_file_info",
        )

    def test_json_maps_to_json_viewer(self) -> None:
        from lynceus.ui.outputs_model import flatten_outputs

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "x_file_info.json"
            target.write_text("{}", encoding="utf-8")
            products = flatten_outputs(
                {"fi": {"file": str(target)}}, {"fi": "File Info"}
            )
            self.assertEqual(len(products), 1)
            self.assertEqual(products[0].viewer, "json")


def _wait_for_json_population(viewer) -> None:
    from PySide6.QtCore import QEventLoop, QTimer

    loop = QEventLoop()
    deadline = QTimer()
    deadline.setSingleShot(True)
    deadline.timeout.connect(loop.quit)
    poll = QTimer()
    poll.timeout.connect(
        lambda: loop.quit()
        if not viewer._progress.isVisible() and viewer._tree.isVisible()
        else None
    )
    poll.start(5)
    deadline.start(5000)
    loop.exec()
    poll.stop()
    if not (viewer._tree.isVisible() and viewer._progress.isHidden()):
        raise AssertionError("JSON preview did not finish loading")


class JsonViewerTests(unittest.TestCase):
    @staticmethod
    def _app():
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        return QApplication.instance() or QApplication([])

    def test_populates_nested_and_caps(self) -> None:
        from lynceus.ui.viewers.json_tree import JsonTreeViewer

        self._app()
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "doc.json"
            target.write_text(
                json.dumps({
                    "b": {"leaf": True, "n": None,
                          "long": "x" * 200},
                    "a": {"m": 1},
                    "big": {f"k{i}": i for i in range(600)},
                }),
                encoding="utf-8",
            )
            viewer = JsonTreeViewer("json")
            viewer.set_product_title("File Info", "doc.json")
            viewer.show()
            viewer.set_payload({"file": str(target)})
            _wait_for_json_population(viewer)
            self.assertIn("doc.json", viewer.windowTitle())
            top = [
                viewer._tree.topLevelItem(i).text(0)
                for i in range(viewer._tree.topLevelItemCount())
            ]
            # Sorted top-level keys, capped children holder present.
            self.assertEqual(top[0], "a")
            big = viewer._tree.topLevelItem(2)
            texts = [
                big.child(j).text(0) for j in range(big.childCount())
            ]
            self.assertTrue(any("more" in text for text in texts))
            viewer.close()

    def test_bad_file_placeholder(self) -> None:
        from lynceus.ui.viewers.json_tree import JsonTreeViewer

        self._app()
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "bad.json"
            target.write_text("{oops", encoding="utf-8")
            viewer = JsonTreeViewer("json")
            viewer.show()
            viewer.set_payload({"file": str(target)})
            self.assertTrue(viewer._placeholder.isVisible())
            viewer.close()


class GuideTests(unittest.TestCase):
    def _doc(self, root: Path) -> dict:
        header = laspy.LasHeader(point_format=6, version="1.4")
        header.scales = np.array([0.01, 0.01, 0.01])
        header.offsets = np.zeros(3)
        rec = laspy.ScaleAwarePointRecord.zeros(
            4, point_format=header.point_format,
            scales=header.scales, offsets=header.offsets,
        )
        rec.x = [0.0, 1.0, 2.0, 3.0]
        rec.y = [0.0, 0.0, 0.0, 0.0]
        rec.z = [10.0, 11.0, 12.0, 13.0]
        with laspy.open(str(root / "g.laz"), mode="w",
                        header=header) as writer:
            writer.write_points(rec)
        out = barrier_file_info(
            _ctx(root / "s", [_src_entry("load", root / "g.laz")])
        )
        return json.loads(Path(out["file"]).read_text(encoding="utf-8"))

    @staticmethod
    def _paths(doc: dict, prefix: str = "") -> set[str]:
        found = set()
        if isinstance(doc, dict):
            for key, value in doc.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                found.add(path)
                found |= GuideTests._paths(value, path)
        return found

    def test_every_guide_path_exists_in_real_doc(self) -> None:
        from lynceus.ui.viewers import json_tree

        with tempfile.TemporaryDirectory() as tmp:
            paths = self._paths(self._doc(Path(tmp)))
            # Every curated dot-path in the viewer must resolve in the
            # emitted document (guards stale/typo'd guide keys).
            import re as _re

            src = Path(json_tree.__file__).read_text(encoding="utf-8")
            found = sorted(set(
                m for m in _re.findall(r'"([a-z_]+(?:\.[a-z_0-9]+)+)"', src)
                if not m.startswith("lynceus")
            ))
            self.assertGreater(len(found), 20)
            for match in found:
                self.assertIn(match, paths, f"stale guide path: {match}")

    def test_selection_shows_guide_or_value(self) -> None:
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        from lynceus.ui.viewers.json_tree import JsonTreeViewer

        QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "doc.json"
            target.write_text(
                json.dumps({"derived_metrics": {"usgs_npd_pts_m2": 8.5},
                            "file_info": {"pdrf_id": 6}}),
                encoding="utf-8",
            )
            viewer = JsonTreeViewer("json")
            viewer.show()
            viewer.set_payload({"file": str(target)})
            _wait_for_json_population(viewer)

            def _find(texts):
                found = []

                def _walk(item):
                    for j in range(item.childCount()):
                        child = item.child(j)
                        found.append(child)
                        _walk(child)

                for i in range(viewer._tree.topLevelItemCount()):
                    top = viewer._tree.topLevelItem(i)
                    found.append(top)
                    _walk(top)
                for item in found:
                    if any(text in item.text(0) for text in texts):
                        return item
                raise AssertionError(f"missing row: {texts}")

            viewer._tree.setCurrentItem(_find(["usgs_npd_pts_m2"]))
            self.assertIn("USGS", viewer._guide.text())
            viewer._tree.setCurrentItem(_find(["pdrf_id: 6"]))
            self.assertIn("pdrf_id: 6", viewer._guide.text())
            viewer.close()


class StripConsistencyTests(unittest.TestCase):
    def _strip_cloud(self, path: Path, offset: float, lines=(0, 1)) -> Path:
        header = laspy.LasHeader(point_format=6, version="1.4")
        header.scales = np.array([0.01, 0.01, 0.01])
        header.offsets = np.zeros(3)
        xs, ys, zs, ps = [], [], [], []
        for line in lines:
            for cx in range(3):
                for cy in range(3):
                    for k in range(6):
                        xs.append(cx * 5.0 + 1.0 + k * 0.1)
                        ys.append(cy * 5.0 + 1.0)
                        zs.append(100.0 + (offset if line else 0.0))
                        ps.append(line)
        n = len(xs)
        rec = laspy.ScaleAwarePointRecord.zeros(
            n, point_format=header.point_format,
            scales=header.scales, offsets=header.offsets,
        )
        rec.x = np.asarray(xs)
        rec.y = np.asarray(ys)
        rec.z = np.asarray(zs, dtype=float)
        rec.point_source_id = np.asarray(ps, dtype=np.uint16)
        with laspy.open(str(path), mode="w", header=header) as writer:
            writer.write_points(rec)
        return path

    def _checks(self, root: Path, name: str, **kw) -> dict:
        src = self._strip_cloud(root / name, **kw)
        out = barrier_file_info(
            _ctx(root / "s", [_src_entry("load", src)])
        )
        doc = json.loads(Path(out["file"]).read_text(encoding="utf-8"))
        return doc["qa_qc_checks"]

    def test_flight_line_histogram_keeps_sparse_ids_and_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = self._strip_cloud(root / "lines.las", offset=0.05)
            out = barrier_file_info(
                _ctx(root / "s", [_src_entry("load", src)])
            )
            doc = json.loads(Path(out["file"]).read_text(encoding="utf-8"))

        lines = doc["classification_flags"]["flight_lines"]
        self.assertEqual(lines["total_unique"], 2)
        self.assertEqual(lines["points_per_line"], {"0": 54, "1": 54})

    def test_aligned_strips_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checks = self._checks(Path(tmp), "a.laz", offset=0.05)
            self.assertAlmostEqual(checks["strip_rmsdz_m"], 0.05, places=3)
            self.assertEqual(checks["strip_pairs_evaluated"], 9)
            self.assertIs(checks["strip_consistency_ok"], True)

    def test_shifted_strips_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checks = self._checks(Path(tmp), "b.laz", offset=0.30)
            self.assertAlmostEqual(checks["strip_rmsdz_m"], 0.30, places=3)
            self.assertIs(checks["strip_consistency_ok"], False)

    def test_single_line_unevaluable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checks = self._checks(
                Path(tmp), "c.laz", offset=0.05, lines=(0,)
            )
            self.assertIsNone(checks["strip_rmsdz_m"])
            self.assertEqual(checks["strip_pairs_evaluated"], 0)
            self.assertIsNone(checks["strip_consistency_ok"])


if __name__ == "__main__":
    unittest.main()
