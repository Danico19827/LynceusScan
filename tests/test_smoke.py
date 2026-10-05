"""Smoke tests for the node registry, DAG compiler, export node, and outputs catalog."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import laspy
import numpy as np

from lynceus.nodes.ports import PortType
from lynceus.processing.controller import default_output_root
from lynceus.processing.steps import build_dag, discover_node_capabilities
from lynceus.ui.outputs_model import flatten_outputs


LOAD_NODE = "lynceus.nodes.lidar.source.load_las_laz"
CLASSIFY_NODE = "lynceus.nodes.lidar.terrain.classify_ground"
EXPORT_NODE = "lynceus.nodes.lidar.output.export_point_cloud"


class SmokeTests(unittest.TestCase):
    def test_export_node_is_discoverable(self) -> None:
        capabilities = discover_node_capabilities(EXPORT_NODE)

        self.assertTrue(capabilities["point_cloud_export"])
        self.assertIsNotNone(capabilities["barrier_task"])

    def test_export_barrier_waits_for_upstream_tiles(self) -> None:
        tiles = [
            {"tile_id": "source_c0000_r0000", "file": "a.laz"},
            {"tile_id": "source_c0001_r0000", "file": "b.laz"},
        ]
        tasks = build_dag(
            ["classify", "export"],
            [("classify", "export", PortType.POINT_CLOUD, PortType.POINT_CLOUD)],
            tiles,
            {"session_dir": "unused"},
            modules={"classify": CLASSIFY_NODE, "export": EXPORT_NODE},
            configs={"classify": {"strategy": "pmf"}},
        )

        export_task = next(task for task in tasks if task.task_id == "export|merge")
        self.assertEqual(
            export_task.deps,
            {"classify|source_c0000_r0000", "classify|source_c0001_r0000"},
        )

    def test_export_crops_tile_buffers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = self._write_tile(root / "first.laz", [1, 9, 10])
            second = self._write_tile(root / "second.laz", [9, 10, 15])

            export_module = __import__(
                EXPORT_NODE, fromlist=["barrier_export_point_cloud"]
            )
            result = export_module.barrier_export_point_cloud(
                {
                    "session_dir": str(root),
                    "node_iid": "classified_export",
                    "_point_cloud_sources": [
                        {
                            "point_cloud_dir": "",
                            "tiles": [
                                self._tile_meta(first, "first", 0, 10),
                                self._tile_meta(second, "second", 10, 20),
                            ],
                        }
                    ],
                }
            )

            self.assertEqual(result["count"], 4)
            with laspy.open(result["file"]) as reader:
                values = []
                for chunk in reader.chunk_iterator(100):
                    values.extend(np.asarray(chunk.x).tolist())
            self.assertEqual(sorted(round(value, 2) for value in values), [1, 9, 10, 15])

    def test_outputs_catalog_lists_laz_and_tiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cloud = str(Path(tmp, "cloud.laz"))
            tile = str(Path(tmp, "tile.laz"))
            Path(cloud).write_bytes(b"x")
            Path(tile).write_bytes(b"x")
            products = flatten_outputs(
                {
                    "export": {
                        "file": cloud,
                        "display_name": "Classified",
                        "tiles": [{"file": tile}],
                    }
                },
                {"export": "Export Point Cloud"},
            )

        self.assertEqual(sum(product.viewer == "point_cloud" for product in products), 2)
        self.assertEqual(next(product for product in products if product.path).name, "Classified")

    @staticmethod
    def _write_tile(path: Path, x_values: list[float]) -> Path:
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.scales = np.array([0.01, 0.01, 0.01])
        header.offsets = np.zeros(3)
        record = laspy.ScaleAwarePointRecord.zeros(
            len(x_values),
            point_format=header.point_format,
            scales=header.scales,
            offsets=header.offsets,
        )
        record.x = x_values
        record.y = 1.0
        record.z = np.arange(len(x_values), dtype=float)
        with laspy.open(str(path), mode="w", header=header) as writer:
            writer.write_points(record)
        return path

    @staticmethod
    def _tile_meta(path: Path, tile_id: str, x_min: float, x_max: float) -> dict:
        return {
            "tile_id": tile_id,
            "file": str(path),
            "core_x_min": x_min,
            "core_y_min": 0,
            "core_x_max": x_max,
            "core_y_max": 10,
        }

    def test_default_output_root_is_absolute_and_cwd_independent(self) -> None:
        """Sessions must never scatter across working directories.

        The panel, the controller and session deletion agree on this absolute
        root; a CWD-relative ``output`` made sessions unreachable for reuse
        after a relaunch and let the panel appear to delete folders that
        Explorer still showed.
        """
        root = default_output_root()
        self.assertTrue(root.is_absolute())
        self.assertEqual(root.name, "output")
        self.assertTrue(Path(__file__).resolve().parent.parent in root.parents)


if __name__ == "__main__":
    unittest.main()
