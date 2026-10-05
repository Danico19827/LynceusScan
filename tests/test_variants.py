"""Tests for the strategy-node contract (family + variants).

Covers the per-instance resolution shared by the engine and the UI: variant
discovery, port resolution, effective capabilities, provenance fingerprints,
and DAG wiring. Stdlib + engine/domain modules only (no Qt, no real runs).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lynceus.nodes._variants import (
    current_variant_key,
    resolved_ports,
    strategy_options,
    variant_meta,
)
from lynceus.nodes.ports import PortType
from lynceus.plugins.registry import PluginManager, manager
from lynceus.processing.controller import PipelineController
from lynceus.processing.steps import build_dag, effective_node_caps

FAMILY = "lynceus.nodes.raster.input_raster"
STATIC_TABLE = "lynceus.nodes.table.input_table"
DTM_TASK = "barrier_input_raster_dtm"
CHM_TASK = "barrier_input_raster_chm"
DTM_MOSAIC = PortType.DTM_MOSAIC
CHM_MOSAIC = PortType.CHM_MOSAIC
TABLE_CSV = PortType.TABLE_CSV


class VariantDiscoveryTests(unittest.TestCase):
    def test_family_registered_without_ports(self) -> None:
        manager.discover()
        info = manager.node_info(FAMILY)
        self.assertIsNotNone(info)
        inputs, outputs = manager.load_ports(FAMILY)
        self.assertEqual(inputs, ())
        self.assertEqual(outputs, ())

    def test_variants_are_discovered_by_key(self) -> None:
        keys = [v["key"] for v in manager.list_variants(FAMILY)]
        self.assertEqual(keys, ["chm", "dsm", "dtm", "multispectral"])
        opts = strategy_options(FAMILY)
        self.assertEqual([k for k, _label in opts], keys)

    def test_variant_metadata_includes_target_and_filters(self) -> None:
        meta = variant_meta(FAMILY, "dtm")
        self.assertEqual(meta.get("session_file"), "dtm_mosaic.tif")
        self.assertIn("*.tif", meta.get("file_filters", ""))

    def test_unknown_variant_is_empty(self) -> None:
        self.assertEqual(list(manager.list_variants("no.such.family")), [])
        self.assertEqual(variant_meta(FAMILY, "nope"), {})


class ResolutionTests(unittest.TestCase):
    def test_current_variant_key_requires_clean_string(self) -> None:
        self.assertIsNone(current_variant_key(FAMILY, {}))
        self.assertIsNone(current_variant_key(FAMILY, None))
        self.assertIsNone(current_variant_key(FAMILY, {"strategy": "  "}))
        self.assertIsNone(current_variant_key(FAMILY, {"strategy": 7}))
        self.assertEqual(current_variant_key(FAMILY, {"strategy": "dsm"}), "dsm")

    def test_resolved_ports_follow_strategy(self) -> None:
        self.assertEqual(resolved_ports(FAMILY, None), ((), ()))
        self.assertEqual(resolved_ports(FAMILY, {"strategy": "dtm"}), ((), (DTM_MOSAIC,)))
        self.assertEqual(resolved_ports(FAMILY, {"strategy": "chm"}), ((), (CHM_MOSAIC,)))

    def test_non_variant_module_keeps_static_ports(self) -> None:
        inputs, outputs = resolved_ports(STATIC_TABLE, None)
        self.assertEqual(inputs, ())
        self.assertEqual(outputs, (TABLE_CSV,))

    def test_effective_caps_switch_with_strategy(self) -> None:
        base_caps = effective_node_caps(FAMILY, None)
        self.assertIsNone(base_caps.get("barrier_task"))
        dtm_caps = effective_node_caps(FAMILY, {"strategy": "dtm"})
        self.assertEqual(dtm_caps.get("barrier_task").__name__, DTM_TASK)
        self.assertEqual(dtm_caps.get("session_file"), "dtm_mosaic.tif")
        self.assertEqual(dtm_caps.get("qml"), {"dtm_mosaic.tif": "dtm"})
        chm_caps = effective_node_caps(FAMILY, {"strategy": "chm"})
        self.assertEqual(chm_caps.get("barrier_task").__name__, CHM_TASK)
        self.assertEqual(chm_caps.get("qml"), {"chm_mosaic.tif": "chm"})

    def test_unrelated_node_keeps_static_caps(self) -> None:
        caps = effective_node_caps(STATIC_TABLE, None)
        self.assertEqual(caps.get("barrier_task").__name__, "barrier_input_table")


class FingerprintTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ctrl = PipelineController()
        self.ctrl._modules = {"a": FAMILY, "b": FAMILY}

    def test_strategies_define_different_fingerprints(self) -> None:
        dtm_fps = self.ctrl._node_fingerprints(
            ["a", "b"], [], {"a": {"strategy": "dtm"}, "b": {"strategy": "dtm"}}, {}
        )
        mixed_fps = self.ctrl._node_fingerprints(
            ["a", "b"], [], {"a": {"strategy": "dtm"}, "b": {"strategy": "chm"}}, {}
        )
        self.assertEqual(dtm_fps["a"], dtm_fps["b"])
        self.assertNotEqual(mixed_fps["a"], mixed_fps["b"])

    def test_module_bundle_globs_follow_variant(self) -> None:
        self.assertEqual(
            self.ctrl._module_bundle_globs(FAMILY, {"strategy": "dtm"}),
            ("*/dtm_mosaic.tif",),
        )
        self.assertEqual(
            self.ctrl._module_bundle_globs(FAMILY, {"strategy": "chm"}),
            ("*/chm_mosaic.tif",),
        )
        self.assertEqual(self.ctrl._module_bundle_globs(FAMILY, None), ())


class DagTests(unittest.TestCase):
    def test_chained_strategy_instances_serialize_merges(self) -> None:
        tasks = build_dag(
            ["a", "b", "chm"],
            [
                ("a", "b", DTM_MOSAIC, DTM_MOSAIC),
                ("b", "chm", DTM_MOSAIC, DTM_MOSAIC),
            ],
            [],
            {"session_dir": "unused"},
            modules={
                "a": FAMILY,
                "b": FAMILY,
                "chm": "lynceus.nodes.lidar.terrain.generate_chm",
            },
            configs={
                "a": {"strategy": "dtm"},
                "b": {"strategy": "dtm"},
            },
        )
        by_id = {t.task_id: t for t in tasks}
        self.assertEqual(by_id["chm|merge"].deps, {"b|merge"})
        self.assertEqual(by_id["b|merge"].deps, {"a|merge"})
        self.assertEqual(by_id["a|merge"].deps, set())

    def test_consumer_injects_provider_path_by_port(self) -> None:
        tasks = build_dag(
            ["in", "chm"],
            [("in", "chm", CHM_MOSAIC, CHM_MOSAIC)],
            [],
            {"session_dir": "unused"},
            modules={"in": FAMILY, "chm": "lynceus.nodes.lidar.terrain.generate_chm"},
            configs={"in": {"strategy": "chm"}},
        )
        chm_task = next(t for t in tasks if t.task_id == "chm|merge")
        ctx = chm_task.args[0]
        self.assertEqual(
            ctx.get("chm_mosaic_path"),
            r"unused\in\chm_mosaic.tif",
        )


class RegistryHardeningTests(unittest.TestCase):
    """load_ports must keep AST-read fields; disabled families skip variants."""

    NODE_SRC = (
        'NODE_ID = "test.subcat"\n'
        'NODE_NAME = "Subcat Node"\n'
        'NODE_CATEGORY = "Test"\n'
        'NODE_SUBCATEGORY = "Subgroup"\n'
        "INPUTS = ()\n"
        "def _outs():\n"
        "    from lynceus.nodes.ports import PortType\n"
        "    return (PortType.TABLE_CSV,)\n"
        "OUTPUTS = _outs()\n"
    )
    VARIANT_SRC = (
        'VARIANT_OF = "test.family"\n'
        'VARIANT_KEY = "k1"\n'
        'VARIANT_LABEL = "K One"\n'
    )

    def test_load_ports_keeps_subcategory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "node_subcat.py").write_text(self.NODE_SRC, encoding="utf-8")
            mgr = PluginManager(extensions_dir=tmp)
            mgr.discover()
            before = mgr.node_info("test.subcat")
            self.assertIsNotNone(before)
            self.assertIsNone(before.outputs)  # dynamic: AST cannot read it
            inputs, outputs = mgr.load_ports("test.subcat")
            self.assertEqual(inputs, ())
            self.assertEqual(outputs, (TABLE_CSV,))
            after = mgr.node_info("test.subcat")
            self.assertEqual(after.subcategory, "Subgroup")
            self.assertEqual(after.category, "Test")

    def test_disabled_family_skips_extension_variant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "variant_k1.py").write_text(self.VARIANT_SRC, encoding="utf-8")
            enabled = PluginManager(extensions_dir=tmp)
            enabled.discover()
            self.assertEqual(
                [v["key"] for v in enabled.list_variants("test.family")], ["k1"]
            )
            disabled = PluginManager(
                extensions_dir=tmp, disabled_provider=lambda: {"test.family"}
            )
            disabled.discover()
            self.assertEqual(list(disabled.list_variants("test.family")), [])


if __name__ == "__main__":
    unittest.main()