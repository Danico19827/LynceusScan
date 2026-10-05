# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Custom port types (PORT_TYPE_DEFS): registry contract end to end.

A third-party file declares its own port ids; the registry seeds them at
discovery (AST, no execution) and the UI/DAG share the same contract:
display name, cable color, compatibility and viewer reuse.
"""

import tempfile
import unittest
from pathlib import Path

PORT_NODE = '''
NODE_ID = "acme.ports"
NODE_NAME = "Ports"
NODE_CATEGORY = "Image"
NODE_AUTHOR = "Acme"

from lynceus.nodes.ports import PortType

INPUTS = ("acme.image",)
OUTPUTS = ("acme.image",)

PORT_TYPE_DEFS = [
    {"id": "acme.image", "display_name": "Image", "color": "#7dd3a8",
     "compatible": ("acme.image", "raster"), "viewer_kind": "raster"},
]

PROCESSING_SPECS = {
    "barrier_task": "run_ports",
}
'''


class CustomPortContractTests(unittest.TestCase):
    def setUp(self) -> None:
        from lynceus.plugins.registry import PluginManager

        self._tmp = tempfile.TemporaryDirectory()
        node = Path(self._tmp.name) / "ports.py"
        node.write_text(PORT_NODE, encoding="utf-8")
        self._mgr = PluginManager(
            extensions_dir=self._tmp.name
        )
        self._mgr.discover()

    def tearDown(self) -> None:
        # Custom specs live in the global registry: remove ours so later
        # suites (i18n coverage over all_specs) never see test ids.
        from lynceus.nodes.ports import _PORT_SPECS

        _PORT_SPECS.pop("acme.image", None)
        self._tmp.cleanup()

    def test_spec_seeded_at_discovery(self) -> None:
        from lynceus.nodes.ports import get_spec

        spec = get_spec("acme.image")
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertEqual(spec.display_name, "Image")
        self.assertEqual(spec.color, "#7dd3a8")
        self.assertEqual(spec.viewer_kind, "raster")

    def test_compatibility_uses_declared_set(self) -> None:
        from lynceus.nodes.ports import can_connect

        self.assertTrue(can_connect("acme.image", "acme.image"))
        self.assertTrue(can_connect("acme.image", "raster"))
        self.assertFalse(can_connect("acme.image", "vector"))
        self.assertFalse(can_connect("vector", "acme.image"))

    def test_display_and_color_fall_back_sanely(self) -> None:
        from lynceus.nodes.ports import get_display_name, get_port_color

        self.assertEqual(get_display_name("acme.image"), "Image")
        self.assertEqual(get_port_color("acme.image"), "#7dd3a8")
        self.assertEqual(get_port_color("nope.unknown"), "#888888")
        self.assertEqual(get_display_name("nope.unknown"), "Nope.Unknown")

    def test_validate_warns_on_unknown_compatible(self) -> None:
        import io
        from contextlib import redirect_stdout

        from lynceus.tools import extension_tool

        with tempfile.TemporaryDirectory() as tmp:
            node = Path(tmp) / "ports.py"
            node.write_text(
                PORT_NODE.replace('"raster"', '"rastre"'),
                encoding="utf-8",
            )
            buf = io.StringIO()
            with redirect_stdout(buf):
                extension_tool.main(["validate", str(node)])
            self.assertIn("rastre", buf.getvalue())


def tearDownModule() -> None:
    # Belt and suspenders alongside the per-test pop: no test port id may
    # leak into the global registry for later suites.
    from lynceus.nodes.ports import _PORT_SPECS

    for key in [k for k in _PORT_SPECS if k.startswith("acme.")]:
        _PORT_SPECS.pop(key, None)


if __name__ == "__main__":
    unittest.main()
