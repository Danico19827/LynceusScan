# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Pure tests for the subgraph clipboard / file-assignment helpers."""

import os
import tempfile
import unittest

from lynceus.graph_clipboard import (
    assign_ordered,
    collect_files,
    decode_subgraph,
    encode_subgraph,
    extract_subgraph,
    pair_files,
    parse_filters_extensions,
)


def _node(iid, nid="mod", x=0.0, y=0.0):
    return {"iid": iid, "id": nid, "x": x, "y": y, "label": "", "config": {}}


def _edge(src, dst, so=0, di=0):
    return {
        "src": src, "src_out": so, "dst": dst, "dst_in": di,
        "out_type": "point_cloud", "in_type": "point_cloud",
    }


class ExtractSubgraphTests(unittest.TestCase):
    def test_internal_and_boundary_edges(self) -> None:
        nodes = [_node("a"), _node("b"), _node("c")]
        edges = [_edge("a", "b"), _edge("b", "c"), _edge("c", "a")]
        sub = extract_subgraph(nodes, edges, {"a", "b"})
        self.assertEqual([n["iid"] for n in sub["nodes"]], ["a", "b"])
        self.assertEqual(len(sub["edges"]), 1)  # a -> b
        self.assertEqual(len(sub["boundary"]), 2)  # b -> c, c -> a
        b_to_c = next(e for e in sub["boundary"] if e["dst"] == "c")
        self.assertTrue(b_to_c["src_selected"])
        self.assertFalse(b_to_c["dst_selected"])
        c_to_a = next(e for e in sub["boundary"] if e["src"] == "c")
        self.assertTrue(c_to_a["dst_selected"])
        self.assertFalse(c_to_a["src_selected"])

    def test_only_selected_nodes(self) -> None:
        nodes = [_node("a"), _node("b")]
        sub = extract_subgraph(nodes, [], {"a"})
        self.assertEqual(len(sub["nodes"]), 1)
        self.assertEqual(sub["nodes"][0]["iid"], "a")


class EncodeDecodeTests(unittest.TestCase):
    def test_roundtrip(self) -> None:
        payload = {
            "nodes": [_node("a")],
            "edges": [],
            "boundary": [_edge("a", "b")],
        }
        text = encode_subgraph(payload)
        decoded = decode_subgraph(text)
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded["nodes"][0]["iid"], "a")
        self.assertEqual(len(decoded["boundary"]), 1)

    def test_rejects_foreign_payload(self) -> None:
        self.assertIsNone(decode_subgraph("hello"))
        self.assertIsNone(decode_subgraph("{}"))
        self.assertIsNone(decode_subgraph('{"format": "other", "version": 1}'))

    def test_rejects_node_without_id(self) -> None:
        text = (
            '{"format": "lynceus-subgraph", "version": 1, '
            '"nodes": [{"iid": "a"}]}'
        )
        self.assertIsNone(decode_subgraph(text))


class FilterExtensionTests(unittest.TestCase):
    def test_parses_patterns_and_dedupes(self) -> None:
        filters = "Raster files (*.tif *.tiff *.TIF);;All files (*)"
        self.assertEqual(parse_filters_extensions(filters), ["tif", "tiff"])

    def test_no_patterns(self) -> None:
        self.assertEqual(parse_filters_extensions("All files (*)"), [])
        self.assertEqual(parse_filters_extensions(""), [])


class CollectFilesTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        os.makedirs(os.path.join(self.root, "nested"), exist_ok=True)
        self._write("b.LAZ")
        self._write("a.las")
        self._write(os.path.join("nested", "c.laz"))
        self._write("note.txt")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, rel) -> None:
        with open(os.path.join(self.root, rel), "w", encoding="utf-8") as fh:
            fh.write("x")

    def test_recursive_filtered_sorted(self) -> None:
        found = collect_files(self.root, ["las", "laz"])
        names = [os.path.basename(p) for p in found]
        self.assertEqual(names, ["a.las", "b.LAZ", "c.laz"])

    def test_non_recursive(self) -> None:
        found = collect_files(self.root, ["laz"], recursive=False)
        self.assertEqual([os.path.basename(p) for p in found], ["b.LAZ"])

    def test_all_files_without_extensions(self) -> None:
        found = collect_files(self.root)
        self.assertEqual(len(found), 4)

    def test_missing_directory(self) -> None:
        self.assertEqual(collect_files(os.path.join(self.root, "nope")), [])


class PairingTests(unittest.TestCase):
    def test_pair_files_min_length(self) -> None:
        self.assertEqual(
            pair_files(["n1", "n2", "n3"], ["f1", "f2"]),
            [("n1", "f1"), ("n2", "f2")],
        )

    def test_assign_ordered_maps_first_k(self) -> None:
        self.assertEqual(
            assign_ordered(["n1", "n2"], ["f1", "f2"]),
            {"n1": "f1", "n2": "f2"},
        )


if __name__ == "__main__":
    unittest.main()
