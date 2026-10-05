# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Unit tests for gallery source-file products (no Qt).

File-backed loader/input instances are also emitted as previewable
``source`` products so the gallery can open the original file (the
point-cloud viewer resolves COPC hierarchies itself).
"""

import json
import tempfile
import unittest
from pathlib import Path

from lynceus.ui.outputs_model import (
    flatten_outputs,
    product_for_path,
    products_from_session,
)


def _touch(root: Path, name: str) -> str:
    path = root / name
    path.write_bytes(b"")
    return str(path)


class SourceProductTests(unittest.TestCase):
    def test_laz_source_emits_point_cloud_item(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = _touch(Path(tmp), "flight.laz")
            products = flatten_outputs(
                {"abc": {"file": src}},
                {"abc": "Load LAS/LAZ"},
                source_files={"abc": "flight.laz"},
                source_paths={"abc": src},
            )
            sources = [p for p in products if p.payload_key == "source"]
            self.assertEqual(len(sources), 1)
            item = sources[0]
            self.assertEqual(item.viewer, "point_cloud")
            self.assertEqual(item.path, src)
            self.assertEqual(item.name, "flight.laz")
            self.assertEqual(item.node_name, "Load LAS/LAZ")

    def test_missing_source_is_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gone = str(Path(tmp) / "gone.laz")
            products = flatten_outputs(
                {}, {"abc": "Load LAS/LAZ"},
                source_paths={"abc": gone},
            )
            self.assertEqual(
                [p for p in products if p.payload_key == "source"], []
            )

    def test_source_without_viewer_is_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = _touch(Path(tmp), "notes.xyz")
            products = flatten_outputs(
                {}, {"abc": "Load LAS/LAZ"},
                source_paths={"abc": src},
            )
            self.assertEqual(
                [p for p in products if p.payload_key == "source"], []
            )

    def test_source_sorts_after_primary_product(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = _touch(Path(tmp), "flight.laz")
            out = _touch(Path(tmp), "dtm_mosaic.tif")
            products = flatten_outputs(
                {"abc": {"file": out}},
                {"abc": "Generate DTM"},
                source_paths={"abc": src},
            )
            keys = [p.payload_key for p in products]
            self.assertEqual(keys, ["file", "source"])

    def test_source_carries_lineage_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = _touch(Path(tmp), "flight.laz")
            products = flatten_outputs(
                {}, {"abc": "Load LAS/LAZ", "def": "Generate DTM"},
                edges=[("abc", "def")],
                source_files={"abc": "flight.laz"},
                source_paths={"abc": src},
            )
            sources = [p for p in products if p.payload_key == "source"]
            self.assertEqual(len(sources), 1)
            self.assertTrue(any(seg[0] == "def" for seg in sources[0].flow))


class ProductForPathTests(unittest.TestCase):
    """Ad-hoc products for File → Open Preview (bridge mode)."""

    def test_known_extension_builds_product(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = _touch(Path(tmp), "flight.laz")
            product = product_for_path(src, "External file")
            self.assertIsNotNone(product)
            assert product is not None
            self.assertEqual(product.viewer, "point_cloud")
            self.assertEqual(product.path, src)
            self.assertEqual(product.payload_key, "external")

    def test_missing_file_yields_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(
                product_for_path(str(Path(tmp) / "gone.laz"), "External file")
            )
            self.assertIsNone(product_for_path("", "External file"))
            self.assertIsNone(product_for_path(None, "External file"))

    def test_unknown_extension_yields_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = _touch(Path(tmp), "notes.xyz")
            self.assertIsNone(product_for_path(src, "External file"))


def _write_session(
    root: Path, outputs: dict, modules: dict, sources: list
) -> Path:
    sess = root / "20260101_000000"
    arts = sess / "artifacts"
    arts.mkdir(parents=True)
    (sess / "session_state.json").write_text(
        json.dumps({"node_outputs": outputs, "node_modules": modules}),
        encoding="utf-8",
    )
    (sess / "session_meta.json").write_text(
        json.dumps({"sources": sources}), encoding="utf-8"
    )
    return sess


class SessionProductsTests(unittest.TestCase):
    """Past-session browsing: read-only reconstruction, no run."""

    def test_rebuilds_products_and_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mosaic = root / "20260101_000000" / "artifacts" / "dtm_mosaic.tif"
            src = root / "flight.laz"
            src.write_bytes(b"")
            sess = _write_session(
                root,
                {
                    "a": {"file": str(mosaic)},
                    "ghost": {"file": str(root / "gone.tif")},
                },
                {"a": "lynceus.nodes.lidar.terrain.generate_dtm"},
                [{"iid": "load", "file": str(src)}],
            )
            mosaic.write_bytes(b"")
            products = products_from_session(sess)
            by_key = {}
            for product in products:
                by_key.setdefault(product.payload_key, []).append(product)
            self.assertEqual([p.name for p in by_key["file"]], ["dtm_mosaic.tif"])
            self.assertEqual(len(by_key["source"]), 1)
            self.assertEqual(by_key["source"][0].viewer, "point_cloud")
            self.assertNotIn("gone.tif", [p.name for p in products])

    def test_missing_or_corrupt_state_yields_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(products_from_session(root / "nope"), [])
            bad = root / "bad"
            bad.mkdir()
            (bad / "session_state.json").write_text("not json", encoding="utf-8")
            self.assertEqual(products_from_session(bad), [])


if __name__ == "__main__":
    unittest.main()
