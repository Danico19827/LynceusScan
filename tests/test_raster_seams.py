# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Frontier-cell seams between tiles (snapped grids vs strict cores).

``tile_transform`` snaps grid bounds outward to the tiling lattice so
neighboring fragments merge exactly. Masking points by the strict core cut
the frontier cell shared with the neighbor; a first-wins mosaic kept one
sliver, which read as a one-pixel under-dense line at 100 m borders
(Flightline Coverage). Writers that mosaic partials now mask by their own
raster extent, so every cell matches a global reference at any resolution
while dividing resolutions stay bit-identical (extent == core).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

OX, OY = 643000.0, 6851000.0
TILE = 100.0
BUF = 10.0


def _write_laz(path: Path, x, y, z) -> Path:
    import laspy

    from lynceus.processing.tiler import _laz_backend

    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = np.array([0.01, 0.01, 0.01])
    header.offsets = np.zeros(3)
    rec = laspy.ScaleAwarePointRecord.zeros(
        len(x),
        point_format=header.point_format,
        scales=header.scales,
        offsets=header.offsets,
    )
    rec.x = np.asarray(x, dtype=float)
    rec.y = np.asarray(y, dtype=float)
    rec.z = np.asarray(z, dtype=float)
    with laspy.open(
        str(path), mode="w", header=header, laz_backend=_laz_backend()
    ) as writer:
        writer.write_points(rec)
    return path


def _tile(col: int, path: Path | str = "") -> dict:
    return {
        "tile_id": f"t_c{col:04d}_r0000",
        "file": str(path),
        "core_x_min": OX + col * TILE,
        "core_x_max": OX + (col + 1) * TILE,
        "core_y_min": OY,
        "core_y_max": OY + TILE,
        "grid_ox": OX,
        "grid_oy": OY,
    }


class _Scene:
    """One synthetic cloud over two adjacent tiles (files carry buffers)."""

    def __init__(self) -> None:
        xs = OX + np.arange(-BUF + 0.37, 2 * TILE + BUF - 0.37, 0.1)
        ys = OY + np.arange(0.11, 20.0, 0.1)
        gx, gy = np.meshgrid(xs, ys)
        self.x = gx.ravel()
        self.y = gy.ravel()
        self.z = 100.0 + (self.x - OX) * 0.01

    def file_for(self, tmp: Path, col: int) -> Path:
        x0 = OX + col * TILE
        x1 = x0 + TILE
        keep = (self.x >= x0 - BUF) & (self.x < x1 + BUF)
        return _write_laz(
            tmp / f"c{col}.laz", self.x[keep], self.y[keep], self.z[keep]
        )


class ExtentMaskTests(unittest.TestCase):
    """The mask matches the strict core only when the snap is an identity."""

    def _points(self):
        xs = np.linspace(OX - 1.0, OX + TILE + 1.0, 401)
        ys = np.linspace(OY - 1.0, OY + TILE + 1.0, 403)
        gx, gy = np.meshgrid(xs, ys)
        return gx.ravel(), gy.ravel()

    def test_dividing_resolutions_equal_core_mask(self) -> None:
        from lynceus.processing.raster import CountAccumulator, core_mask

        x, y = self._points()
        tile = _tile(0)
        for res in (0.5, 1.0, 2.0, 5.0):
            with self.subTest(res=res):
                acc = CountAccumulator(tile, res)
                np.testing.assert_array_equal(
                    acc.extent_mask(x, y), core_mask(x, y, tile)
                )

    def test_fractional_resolution_widens_to_the_grid(self) -> None:
        from lynceus.processing.raster import CountAccumulator, TileAccumulator

        x, y = self._points()
        tile = _tile(0)
        core = (
            (x >= tile["core_x_min"])
            & (x < tile["core_x_max"])
            & (y >= tile["core_y_min"])
            & (y < tile["core_y_max"])
        )
        for acc in (CountAccumulator(tile, 0.3), TileAccumulator(tile, 0.3)):
            mask = acc.extent_mask(x, y)
            # Never drops a core point (coverage can only grow).
            self.assertFalse(bool(np.any(core & ~mask)))
            # Keeps the shared frontier column whole: the far sliver is
            # in the snapped grid, out of the strict core.
            self.assertTrue(
                bool(acc.extent_mask(
                    np.array([tile["core_x_max"] + 0.1]),
                    np.array([tile["core_y_min"] + 1.0]),
                )[0])
            )

    def test_far_limits_belong_to_the_next_grid(self) -> None:
        from lynceus.processing.raster import CountAccumulator

        acc = CountAccumulator(_tile(0), 0.3)
        east = float(acc._transform.c) + acc._cols * 0.3
        mask = acc.extent_mask(
            np.array([east + 0.05, east - 0.05]),
            np.array([acc._transform.f - 1.0, acc._transform.f - 1.0]),
        )
        self.assertFalse(bool(mask[0]))
        self.assertTrue(bool(mask[1]))


class SnapOvershootTests(unittest.TestCase):
    """Snapped grids overshoot the core by at most one cell, never more."""

    def test_overshoot_is_at_most_one_cell(self) -> None:
        from lynceus.processing.raster import CountAccumulator, TileAccumulator

        for res in (0.3, 0.7, 3.0, 7.0):
            with self.subTest(res=res):
                for acc in (
                    CountAccumulator(_tile(1), res),
                    TileAccumulator(_tile(1), res),
                ):
                    west = float(acc._transform.c)
                    east = west + acc._cols * res
                    north = float(acc._transform.f)
                    south = north - acc._rows * res
                    self.assertGreaterEqual(west, OX + TILE - res - 1e-9)
                    self.assertLessEqual(west, OX + TILE + 1e-9)
                    self.assertGreaterEqual(east, OX + 2 * TILE - 1e-9)
                    self.assertLessEqual(east, OX + 2 * TILE + res + 1e-9)
                    self.assertGreaterEqual(south, OY - res - 1e-9)
                    self.assertLessEqual(south, OY + 1e-9)
                    self.assertGreaterEqual(north, OY + TILE - 1e-9)
                    self.assertLessEqual(north, OY + TILE + res + 1e-9)


class BoundaryDustTests(unittest.TestCase):
    """Core edges keep their semi-open convention despite float dust."""

    TILE = {
        "tile_id": "t_big",
        "core_x_min": 4345749.416,
        "core_x_max": 4345849.416,
        "core_y_min": 7391472.637,
        "core_y_max": 7391572.637,
        "grid_ox": 4345749.416,
        "grid_oy": 7391872.837,
    }

    def test_inclusive_south_west_edges_survive_float_dust(self) -> None:
        from lynceus.processing.raster import CountAccumulator

        tile = self.TILE
        acc = CountAccumulator(tile, 0.3)
        west = float(acc._transform.c)
        east = west + acc._cols * 0.3
        north = float(acc._transform.f)
        mask = acc.extent_mask(
            np.array([
                tile["core_x_min"],  # west core line, inclusive
                tile["core_x_min"] + 50.0,
                tile["core_x_min"] + 50.0,
                tile["core_x_min"] + 50.0,
                east - 0.05,
                east + 0.05,
            ]),
            np.array([
                tile["core_y_min"] + 50.0,
                tile["core_y_min"],  # south line: index lands at rows + dust
                tile["core_y_max"],  # inside the overshoot (frontier) row
                north - 0.05,
                tile["core_y_min"] + 50.0,
                tile["core_y_min"] + 50.0,
            ]),
        )
        np.testing.assert_array_equal(
            mask, [True, True, True, True, True, False]
        )
        # The inclusive corner clips into the last row's first cell.
        flat = acc.flat(
            np.array([tile["core_x_min"]]),
            np.array([tile["core_y_min"]]),
        )
        self.assertEqual(int(flat[0]), (acc._rows - 1) * acc._cols)

    def test_far_core_limits_leave_at_dividing_res(self) -> None:
        from lynceus.processing.raster import CountAccumulator

        # Lattice-aligned cores (dividing resolution -> identity snap).
        acc = CountAccumulator(_tile(0), 1.0)
        mask = acc.extent_mask(
            np.array([OX, OX + TILE, OX + TILE - 0.01, OX + 50.0]),
            np.array([OY + 50.0, OY + 50.0, OY + 50.0, OY + TILE]),
        )
        np.testing.assert_array_equal(mask, [True, False, True, False])
        mask = acc.extent_mask(
            np.array([OX + 50.0, OX + 50.0]),
            np.array([OY, OY + TILE - 0.01]),
        )
        np.testing.assert_array_equal(mask, [True, True])


class AccumulatorFrontierTests(unittest.TestCase):
    """The frontier cell aggregates every point the grid covers."""

    def _local_tile(self) -> dict:
        return {
            "tile_id": "t",
            "core_x_min": 0.0,
            "core_x_max": 100.0,
            "core_y_min": 0.0,
            "core_y_max": 100.0,
            "grid_ox": 0.0,
            "grid_oy": 0.0,
        }

    def test_min_reducer_sees_the_far_sliver(self) -> None:
        from lynceus.processing.raster import (
            TileAccumulator,
            core_mask,
        )

        tile = self._local_tile()
        acc = TileAccumulator(tile, 0.3, "min")
        x = np.array([99.95, 100.05, 100.4])
        y = np.array([50.0, 50.0, 50.0])
        z = np.array([10.0, 5.0, 1.0])
        mask = acc.extent_mask(x, y)
        np.testing.assert_array_equal(mask, [True, True, False])
        # The strict core would have dropped the lowest point: the
        # frontier cell kept only this tile's sliver.
        self.assertFalse(bool(core_mask(x[1:2], y[1:2], tile)[0]))
        acc.add(x[mask], y[mask], z[mask])
        grid, transform = acc.result()
        col = int(np.floor((100.05 - transform.c) / 0.3))
        row = int(np.floor((transform.f - 50.0) / 0.3))
        self.assertAlmostEqual(float(grid[row, col]), 5.0)


class CoverageSeamTests(unittest.TestCase):
    """Per-tile rasters mosaicked first-wins match a global reference."""

    def _reference(self, scene: _Scene, transform, rows: int, cols: int):
        res = float(transform.a)
        c = np.floor((scene.x - transform.c) / res).astype(np.int64)
        r = np.floor((transform.f - scene.y) / res).astype(np.int64)
        inside = (c >= 0) & (c < cols) & (r >= 0) & (r < rows)
        counts = np.bincount(
            r[inside] * cols + c[inside],
            minlength=rows * cols,
        ).reshape(rows, cols)
        return counts

    def test_coverage_mosaic_equals_global_counts(self) -> None:
        from lynceus.nodes.lidar.analysis.flightline_coverage import (
            tile_flightline_coverage,
        )
        from lynceus.processing.raster import finish_mosaic, read_geotiff

        scene = _Scene()
        for res in (0.3, 0.5, 1.0):
            with self.subTest(res=res), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                session = root / "session"
                session.mkdir()
                paths = []
                for col in (0, 1):
                    tile = _tile(col, scene.file_for(root, col))
                    ctx = {
                        "session_dir": str(session),
                        "resolution": res,
                        "mode": "density",
                    }
                    out = tile_flightline_coverage(tile, ctx)["output"]
                    paths.append(out)
                mosaic = root / "coverage_mosaic.tif"
                finish_mosaic(
                    paths,
                    mosaic,
                    nodata=-9999.0,
                    crop_to_valid=False,
                )
                arr, transform, _ = read_geotiff(mosaic)
                rows, cols = arr.shape
                reference = self._reference(scene, transform, rows, cols)
                valid = reference > 0
                # Every covered cell carries the full global count; a
                # half-masked frontier cell used to keep only one sliver.
                np.testing.assert_array_equal(
                    arr[valid].astype(np.int64), reference[valid]
                )
                self.assertTrue(
                    bool(np.all(arr[~valid] == -9999.0)),
                    "empty cells must stay NODATA",
                )


if __name__ == "__main__":
    unittest.main()
