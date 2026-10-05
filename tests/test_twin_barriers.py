# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Twin barrier instances + labeled custom ports (no collisions).

Two instances of one barrier module writing the same basename are scoped
apart by the engine ({iid}/), and a combiner with labeled same-type
inputs (PortDef with a custom id) reads each side positionally. All with
synthetic modules: no third-party files involved.
"""

import tempfile
import unittest
from pathlib import Path

TWIN_A = "t2.twin"
TWIN_B_PORT = "t2.image"


def _caps(module_id: str, specs: dict):
    return module_id, specs


class TwinBarrierTests(unittest.TestCase):
    CHM_DIFF = "lynceus.nodes.raster.ops.chm_difference"

    def test_same_module_twins_scope_apart(self) -> None:
        from lynceus.processing.steps import _branch_scope_map

        # Two instances of one barrier module writing the same basename
        # (the real CHM Difference: chm_difference.tif) with no chaining
        # edge between them must both be scoped apart.
        order = ["c1", "c2"]
        modules = {"c1": self.CHM_DIFF, "c2": self.CHM_DIFF}
        scope_map = _branch_scope_map(
            order, [], [], modules, {}, lambda iid: []
        )
        self.assertEqual(scope_map, {"c1": "c1", "c2": "c2"})

    def test_chained_same_module_instances_share(self) -> None:
        from lynceus.processing.steps import _branch_scope_map

        # Chained by design (c1 -> c2): second consumes the first, so they
        # share the legacy layout instead of scoping apart.
        order = ["c1", "c2"]
        modules = {"c1": self.CHM_DIFF, "c2": self.CHM_DIFF}
        edges = [("c1", "c2", "raster", "raster")]
        scope_map = _branch_scope_map(
            order, edges, [], modules, {}, lambda iid: []
        )
        self.assertEqual(scope_map, {})

    def test_twins_write_apart_and_combiner_reads_both(self) -> None:
        from lynceus.nodes._paths import scoped_file
        from lynceus.processing.steps import (
            _branch_scope_map,
            _inject_input_paths,
            effective_node_caps,
        )

        concat = "lynceus.nodes.table.concatenate_tables"
        with tempfile.TemporaryDirectory() as tmpdir:
            arts = Path(tmpdir) / "artifacts"
            order = ["p1", "p2", "b"]
            modules = {"p1": concat, "p2": concat, "b": concat}
            configs: dict = {}
            edges = [
                ("p1", "b", "table_csv", "table"),
                ("p2", "b", "table_csv", "table"),
            ]
            scope_map = _branch_scope_map(
                order, edges, [], modules, configs, lambda iid: []
            )
            self.assertEqual(scope_map, {"p1": "p1", "p2": "p2"})

            def caps_of(iid: str):
                return modules[iid], effective_node_caps(
                    modules[iid], configs.get(iid)
                )

            written = {}
            for iid in ("p1", "p2"):
                ctx = {
                    "session_dir": str(arts),
                    "node_iid": iid,
                    "tile_scope": scope_map[iid],
                }
                out = Path(scoped_file(ctx, "concatenated.csv"))
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(b"rows-from-" + iid.encode())
                written[iid] = out
            self.assertNotEqual(written["p1"], written["p2"])
            self.assertTrue(written["p1"].is_file())
            self.assertTrue(written["p2"].is_file())

            # The combiner (Table A / Table B labels) resolves each side
            # positionally to the scoped product of its own provider.
            bctx: dict = {"session_dir": str(arts), "node_iid": "b"}
            _inject_input_paths(
                bctx, concat,
                [("p1", "b", "table_csv", "table"),
                 ("p2", "b", "table_csv", "table")],
                caps_of, None, scope_map,
            )
            self.assertEqual(
                bctx.get("in0_path"), str(arts / "p1" / "concatenated.csv")
            )
            self.assertEqual(
                bctx.get("in1_path"), str(arts / "p2" / "concatenated.csv")
            )


class LabeledCustomPortTests(unittest.TestCase):
    def test_portdef_custom_id_with_label(self) -> None:
        from lynceus.nodes.ports import PortDef, port_metadata

        first = port_metadata(PortDef("t2.image", name="Base"))
        second = port_metadata(PortDef("t2.image", name="Overlay"))
        self.assertEqual(first.port_type, "t2.image")
        self.assertEqual(first.name, "Base")
        self.assertEqual(second.name, "Overlay")
        self.assertTrue(first.required)

    def test_canvas_item_keeps_custom_labels(self) -> None:
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        from lynceus.nodes.ports import PortDef
        from lynceus.ui.nodes.node_item import NodeItem

        QApplication.instance() or QApplication([])
        item = NodeItem(
            "t2.blend",
            "Blend",
            "Image",
            [PortDef("t2.image", name="Base"), PortDef("t2.image", name="Overlay")],
            ["t2.image"],
        )
        try:
            names = [p.name for p in item.inputs()]
            self.assertEqual(names, ["Base", "Overlay"])
        finally:
            item.deleteLater()


if __name__ == "__main__":
    unittest.main()
