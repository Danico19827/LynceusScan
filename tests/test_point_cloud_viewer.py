# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Tests for bounded point-cloud preview loading."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from threading import Event
from unittest.mock import MagicMock, patch

import laspy
import numpy as np

from lynceus.ui.viewers.point_cloud_3d import (
    _PointCloudLoader,
    _bounded_sample_merge,
    _localize_coordinates,
    classification_colors,
)


class PointCloudViewerTests(unittest.TestCase):
    def test_classification_lut_preserves_known_and_unknown_colors(self) -> None:
        classes = np.array([0, 2, 18, 19, 255], dtype=np.uint8)

        colors = classification_colors(classes)

        np.testing.assert_allclose(colors[0], (0.55, 0.55, 0.55))
        np.testing.assert_allclose(colors[1], (0.32, 0.70, 0.24))
        np.testing.assert_allclose(colors[2], (0.98, 0.05, 0.05))
        np.testing.assert_allclose(colors[3:], ((0.65, 0.65, 0.65),) * 2)

    def test_local_coordinates_preserve_small_offsets_at_utm_scale(self) -> None:
        origin = np.array([500_000.0, 6_200_000.0, 120.0])

        positions = _localize_coordinates(
            origin[0] + np.array([0.0, 0.01]),
            origin[1] + np.array([0.0, 0.01]),
            origin[2] + np.array([0.0, 0.01]),
            origin,
        )

        np.testing.assert_allclose(positions[1], [0.01, 0.01, 0.01], atol=1e-6)

    def test_priority_sampler_never_exceeds_budget_and_keeps_attributes_aligned(self) -> None:
        scores = np.arange(100, dtype=np.float64)
        positions = np.column_stack((scores, scores, scores)).astype(np.float32)
        rgb = np.column_stack((scores, scores, scores)).astype(np.float32)
        classes = scores.astype(np.uint8)

        sampled = _bounded_sample_merge(
            None, scores, positions, rgb, classes, limit=8
        )

        self.assertEqual(sampled[0].size, 8)
        np.testing.assert_array_equal(sampled[1][:, 0], sampled[0])
        np.testing.assert_array_equal(sampled[2][:, 0], sampled[0])
        np.testing.assert_array_equal(sampled[3], sampled[0].astype(np.uint8))

        next_scores = np.linspace(0.0, 0.49, 50)
        identifiers = np.arange(50, dtype=np.uint8) + 100
        next_positions = np.column_stack(
            (identifiers, identifiers, identifiers)
        ).astype(np.float32)
        updated = _bounded_sample_merge(
            sampled,
            next_scores,
            next_positions,
            next_positions.copy(),
            identifiers,
            limit=8,
        )
        self.assertEqual(updated[0].size, 8)
        np.testing.assert_array_equal(updated[1][:, 0], updated[2][:, 0])
        np.testing.assert_array_equal(updated[1][:, 0], updated[3].astype(np.float32))

    def test_las_loader_caps_points_and_keeps_rgb_and_classification_aligned(self) -> None:
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.scales = np.array([0.001, 0.001, 0.001])
        header.offsets = np.array([500_000.0, 6_200_000.0, 100.0])
        cloud = laspy.LasData(header)
        offsets = np.arange(100, dtype=np.float64) * 0.01
        cloud.x = 500_000.0 + offsets
        cloud.y = 6_200_000.0 + offsets
        cloud.z = 100.0 + offsets
        cloud.red = np.full(100, 65535, dtype=np.uint16)
        cloud.green = np.zeros(100, dtype=np.uint16)
        cloud.blue = np.zeros(100, dtype=np.uint16)
        cloud.classification = np.arange(100, dtype=np.uint8) % 5

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "utm.las"
            cloud.write(path)
            worker = _PointCloudLoader([])
            positions, rgb, classes, has_rgb, has_classes = worker._load_tile(
                {"file": str(path)},
                8,
                np.array([500_000.0, 6_200_000.0, 100.0]),
                Event(),
            )

        self.assertEqual(positions.shape, (8, 3))
        self.assertEqual(rgb.shape, (8, 3))
        self.assertEqual(classes.shape, (8,))
        self.assertTrue(has_rgb)
        self.assertTrue(has_classes)
        np.testing.assert_allclose(positions[:, 0], positions[:, 1], atol=1e-6)
        np.testing.assert_allclose(positions[:, 1], positions[:, 2], atol=1e-6)
        np.testing.assert_allclose(rgb[:, 0], 1.0)

    def test_las_14_point_format_without_rgb_uses_classification_only(self) -> None:
        header = laspy.LasHeader(point_format=6, version="1.4")
        cloud = laspy.LasData(header)
        cloud.x = np.array([10.0, 11.0])
        cloud.y = np.array([20.0, 21.0])
        cloud.z = np.array([30.0, 31.0])
        cloud.classification = np.array([2, 5], dtype=np.uint8)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "las14.las"
            cloud.write(path)
            result = _PointCloudLoader([])._load_tile(
                {"file": str(path)}, 8, np.array([10.0, 20.0, 30.0]), Event()
            )

        positions, rgb, classes, has_rgb, has_classes = result
        self.assertEqual(positions.shape, (2, 3))
        self.assertIsNone(rgb)
        np.testing.assert_array_equal(np.sort(classes), [2, 5])
        self.assertFalse(has_rgb)
        self.assertTrue(has_classes)

    def test_copc_uses_spatial_query_when_reader_supports_it(self) -> None:
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.vlrs.append(laspy.VLR("copc", 1, record_data=b""))
        header.mins = np.array([500_000.0, 6_200_000.0, 100.0])
        header.maxs = np.array([500_010.0, 6_200_010.0, 110.0])
        points = MagicMock()
        points.x = np.array([500_001.0, 500_002.0])
        points.y = np.array([6_200_001.0, 6_200_002.0])
        points.z = np.array([101.0, 102.0])
        points.red = np.array([65535, 65535], dtype=np.uint16)
        points.green = np.array([0, 0], dtype=np.uint16)
        points.blue = np.array([0, 0], dtype=np.uint16)
        points.classification = np.array([2, 5], dtype=np.uint8)
        metadata_context = MagicMock()
        metadata_context.__enter__.return_value.header = header
        copc_context = MagicMock()
        copc_context.__enter__.return_value.query.return_value = points

        with (
            patch("laspy.open", return_value=metadata_context) as las_open,
            patch("laspy.copc.CopcReader.open", return_value=copc_context) as copc_open,
        ):
            result = _PointCloudLoader([])._load_tile(
                {"file": "sample.laz"},
                1,
                np.array([500_000.0, 6_200_000.0, 100.0]),
                Event(),
            )

        self.assertEqual(result[0].shape, (1, 3))
        self.assertEqual(result[1].shape, (1, 3))
        self.assertEqual(result[2].shape, (1,))
        copc_open.assert_called_once_with("sample.laz")
        copc_context.__enter__.return_value.query.assert_called_once()
        query_kwargs = copc_context.__enter__.return_value.query.call_args.kwargs
        self.assertIn("bounds", query_kwargs)
        self.assertGreater(query_kwargs["resolution"], 0)
        las_open.assert_called_once_with("sample.laz")

    def test_copc_without_query_capability_falls_back_to_chunk_reader(self) -> None:
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.vlrs.append(laspy.VLR("copc", 1, record_data=b""))
        points = MagicMock()
        points.x = np.array([500_001.0])
        points.y = np.array([6_200_001.0])
        points.z = np.array([101.0])
        points.red = np.array([65535], dtype=np.uint16)
        points.green = np.array([0], dtype=np.uint16)
        points.blue = np.array([0], dtype=np.uint16)
        points.classification = np.array([2], dtype=np.uint8)
        metadata_context = MagicMock()
        metadata_context.__enter__.return_value.header = header
        stream_context = MagicMock()
        stream_context.__enter__.return_value.chunk_iterator.return_value = [points]
        copc_context = MagicMock()
        copc_context.__enter__.return_value.query = None

        with (
            patch(
                "laspy.open", side_effect=[metadata_context, stream_context]
            ) as las_open,
            patch("laspy.copc.CopcReader.open", return_value=copc_context) as copc_open,
        ):
            result = _PointCloudLoader([])._load_tile(
                {"file": "unsupported-copc.laz"},
                4,
                np.array([500_000.0, 6_200_000.0, 100.0]),
                Event(),
            )

        self.assertEqual(result[0].shape, (1, 3))
        self.assertEqual(las_open.call_count, 2)
        copc_open.assert_called_once_with("unsupported-copc.laz")
        stream_context.__enter__.return_value.chunk_iterator.assert_called_once()


if __name__ == "__main__":
    unittest.main()