"""Tests for the provenance document builder (stdlib-only).

The smoke suite stays on the standard library: provenance building uses only
JSON/datetime + the node registry (the heavy raster/LAS readers are exercised
by the real app, not by these tests).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lynceus.plugins.registry import NodeInfo, PluginManager, manager
from lynceus.processing import provenance

SAMPLE_DISCLAIMER = (
    "Algorithms generated with AI assistance; precision unverified; "
    "no liability for inaccuracy."
)


class ProvenanceTests(unittest.TestCase):
    def test_document_carries_system_identity_and_disclaimer(self) -> None:
        doc = provenance.build_provenance(
            {"node_iid": "n1", "node_fps": {"n1": "fp1"}}
        )

        self.assertEqual(doc["schema"], "lynceus-provenance")
        self.assertEqual(doc["schema_version"], 1)
        self.assertEqual(doc["system"], "LynceusScan")
        self.assertIn("system_version", doc)
        self.assertEqual(doc["license"], "GPL-3.0-or-later")
        self.assertIn("disclaimer", doc)
        self.assertIn("as is", doc["disclaimer"])
        self.assertEqual(doc["node"]["fingerprint"], "fp1")

    def test_parameters_include_config_but_not_paths_or_internals(self) -> None:
        doc = provenance.build_provenance(
            {
                "module_id": "x",
                "node_iid": "n1",
                "session_dir": "s",
                "crs": "EPSG:4326",
                "has_ground": True,
                "classified": True,
                "chm_mosaic_path": "s/chm.tif",
                "delimiter": ";",
                "acceleration": {"device": "cpu"},
                "provenance": {},
            }
        )
        params = doc["node"]["parameters"]
        self.assertEqual(params.get("delimiter"), ";")
        self.assertEqual(params.get("has_ground"), True)
        self.assertNotIn("chm_mosaic_path", params)
        self.assertNotIn("session_dir", params)
        self.assertNotIn("acceleration", params)

    def test_session_blueprint_and_product(self) -> None:
        blueprint = provenance.session_blueprint(
            "out/sessionA", [{"iid": "l1", "file": "a.laz"}]
        )
        self.assertEqual(blueprint["session"]["root"], "sessionA")
        self.assertNotIn("operator", blueprint)

        doc = provenance.build_provenance(
            {"provenance": blueprint, "module_id": "m", "node_iid": "n1"},
            {"tile_id": "t1"},
        )
        self.assertEqual(doc["product"]["tile_id"], "t1")
        self.assertEqual(doc["sources"][0]["file"], "a.laz")

    def test_session_blueprint_embeds_operator(self) -> None:
        blueprint = provenance.session_blueprint(
            "out/sessionA", [], {"name": "Nico", "org": "INTA"}
        )
        self.assertEqual(
            blueprint["operator"], {"name": "Nico", "org": "INTA"}
        )
        empty = provenance.session_blueprint(
            "out/sessionA", [], {"name": "", "org": ""}
        )
        self.assertNotIn("operator", empty)

    def test_build_provenance_carries_operator(self) -> None:
        blueprint = provenance.session_blueprint(
            "out/sessionA", [], {"name": "Nico", "org": ""}
        )
        doc = provenance.build_provenance(
            {"provenance": blueprint, "module_id": "m", "node_iid": "n1"},
            "a.tif",
        )
        self.assertEqual(doc["operator"], {"name": "Nico", "org": ""})
        plain = provenance.build_provenance(
            {"module_id": "m", "node_iid": "n1"}, "a.tif"
        )
        self.assertNotIn("operator", plain)

    def test_resolved_parameters_sorted(self) -> None:
        params = provenance._resolved_parameters(
            {"zebra": 1, "apple": 2, "session_dir": "x", "mango": 3}
        )
        self.assertEqual(list(params), ["apple", "mango", "zebra"])

    def test_node_disclaimer_parsed_from_standalone_extension(self) -> None:
        node_source = (
            'NODE_ID = "test.disc"\n'
            'NODE_NAME = "AI Node"\n'
            'NODE_AUTHOR = "Jane"\n'
            f"NODE_DISCLAIMER = {json.dumps(SAMPLE_DISCLAIMER)}\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            ext = Path(temp_dir)
            (ext / "ai_node.py").write_text(node_source, encoding="utf-8")
            temp_manager = PluginManager(extensions_dir=ext)
            temp_manager.discover()
            info = temp_manager.node_info("test.disc")
            self.assertIsNotNone(info)
            self.assertEqual(info.disclaimer, SAMPLE_DISCLAIMER)

    def test_node_disclaimer_embedded_when_present(self) -> None:
        manager.discover()
        manager._nodes["test.disc"] = NodeInfo(
            node_id="test.disc",
            name="AI Node",
            category="Test",
            description="",
            inputs=(),
            outputs=(),
            extension_id="test",
            source="extension",
            file_path="",
            author="Jane",
            disclaimer=SAMPLE_DISCLAIMER,
        )
        try:
            doc = provenance.build_provenance(
                {"module_id": "test.disc", "node_iid": "n1"}
            )
            self.assertEqual(doc["node_disclaimer"], SAMPLE_DISCLAIMER)
            self.assertEqual(doc["node"]["author"], "Jane")
        finally:
            manager._nodes.pop("test.disc", None)

    def test_write_sidecar_merges_node_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "out.csv"
            path.write_text("a,b\n1,2\n", encoding="utf-8")
            sidecar = provenance.write_sidecar(
                path,
                {"schema": "lynceus-provenance", "system": "LynceusScan"},
                {"metrics_computed": ["h_mean"]},
            )
            data = json.loads(Path(sidecar).read_text(encoding="utf-8"))
            self.assertEqual(data["metrics_computed"], ["h_mean"])
            self.assertEqual(data["provenance"]["system"], "LynceusScan")
            self.assertEqual(Path(sidecar).name, "out.meta.json")


if __name__ == "__main__":
    unittest.main()