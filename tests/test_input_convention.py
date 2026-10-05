# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Third-party file inputs by convention (no core registration).

A node declaring ``input_file_key`` in PROCESSING_SPECS gets the Browse +
file-label item automatically (``file_filters`` from the same specs), and
its barrier can reuse ``barrier_input_file`` with its own target dict.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent

THIRD_PARTY_INPUT = '''
NODE_ID = "acme.input_image"
NODE_NAME = "Input Image"
NODE_CATEGORY = "Image"
NODE_AUTHOR = "Acme"
NODE_FOLDER = "acme"

from lynceus.nodes.ports import PortType

INPUTS = ()
OUTPUTS = (PortType.RASTER,)

PROCESSING_SPECS = {
    "barrier_task": "barrier_input_image",
    "input_file_key": "file_path",
    "file_filters": "Images (*.png *.jpg)",
    "output_port": "raster",
    "session_file": "image.png",
    "kind_group": "none",
    "kind_label": "Image",
}

PORT_TYPE_DEFS = [
    {"id": "acme.image", "display_name": "Image", "color": "#7dd3a8",
     "compatible": ("acme.image",)},
]
'''

PLAIN_NODE = '''
NODE_ID = "acme.plain"
NODE_NAME = "Plain"
NODE_CATEGORY = "Image"
NODE_AUTHOR = "Acme"

from lynceus.nodes.ports import PortType

INPUTS = (PortType.RASTER,)
OUTPUTS = (PortType.RASTER,)

PROCESSING_SPECS = {
    "barrier_task": "run_plain",
}
'''


class NodeSpecsTests(unittest.TestCase):
    def test_builtin_specs_without_import(self) -> None:
        from lynceus.plugins.registry import manager

        specs = manager.node_specs("lynceus.nodes.table.input_table")
        self.assertEqual(specs.get("input_file_key"), "file_path")
        self.assertEqual(specs.get("session_file"), "table.csv")
        # Legacy builtins keep dialog filters in code registries, not specs.
        self.assertNotIn("file_filters", specs)

    def test_third_party_specs_ast_only(self) -> None:
        from lynceus.plugins.registry import PluginManager

        with tempfile.TemporaryDirectory() as tmp:
            node = Path(tmp) / "input_image.py"
            node.write_text(THIRD_PARTY_INPUT, encoding="utf-8")
            mgr = PluginManager(extensions_dir=tmp)
            mgr.discover()
            before = set(sys.modules)
            specs = mgr.node_specs("acme.input_image")
            self.assertEqual(specs.get("input_file_key"), "file_path")
            self.assertEqual(
                specs.get("file_filters"), "Images (*.png *.jpg)"
            )
            self.assertEqual(specs.get("kind_group"), "none")
            # AST only: the module was never imported.
            self.assertEqual(set(sys.modules), before)
            self.assertNotIn("acme.input_image", sys.modules)


class FactoryConventionTests(unittest.TestCase):
    def test_builtin_inputs_resolve_by_convention(self) -> None:
        from PySide6.QtWidgets import QApplication

        from lynceus.plugins.registry import manager
        from lynceus.ui.nodes.widget_factory import create_node_item
        from lynceus.ui.nodes.widgets.input_file import InputFileNodeItem

        QApplication.instance() or QApplication([])
        for node_id in (
            "lynceus.nodes.table.input_table",
            "lynceus.nodes.table.input_gpkg",
        ):
            info = manager.node_info(node_id)
            assert info is not None
            item = create_node_item(
                node_id,
                info.name,
                info.category,
                (),
                ("table",),
                specs=manager.node_specs(node_id),
            )
            self.assertIsInstance(item, InputFileNodeItem)
            item.set_file_filters(None)
            self.assertIsNone(item._file_filters)

    def test_input_convention_picks_file_widget(self) -> None:
        from PySide6.QtWidgets import QApplication

        from lynceus.plugins.registry import PluginManager
        from lynceus.ui.nodes.widget_factory import create_node_item
        from lynceus.ui.nodes.widgets.input_file import InputFileNodeItem
        from lynceus.ui.nodes.node_item import NodeItem

        QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as tmp:
            node = Path(tmp) / "input_image.py"
            node.write_text(THIRD_PARTY_INPUT, encoding="utf-8")
            plain = Path(tmp) / "plain.py"
            plain.write_text(PLAIN_NODE, encoding="utf-8")
            mgr = PluginManager(extensions_dir=tmp)
            mgr.discover()
            item = create_node_item(
                "acme.input_image",
                "Input Image",
                "Image",
                (),
                ("raster",),
                specs=mgr.node_specs("acme.input_image"),
            )
            self.assertIsInstance(item, InputFileNodeItem)
            self.assertEqual(
                item._file_filters, "Images (*.png *.jpg)"
            )
            generic = create_node_item(
                "acme.plain",
                "Plain",
                "Image",
                ("raster",),
                ("raster",),
                specs=mgr.node_specs("acme.plain"),
            )
            self.assertIsInstance(generic, NodeItem)
            self.assertNotIsInstance(generic, InputFileNodeItem)


class BarrierNoneBranchTests(unittest.TestCase):
    def test_custom_kind_copies_without_opinions(self) -> None:
        from lynceus.nodes._product_input import barrier_input_file

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "photo.png"
            src.write_bytes(b"\x89PNG\r\n\x1a\nnot-a-real-image")
            session = Path(tmp) / "session"
            ctx = {
                "session_dir": str(session),
                "file_path": str(src),
                "node_iid": "img1",
                "module_id": "acme.input_image",
            }
            payload = barrier_input_file(
                ctx,
                {
                    "node_id": "acme.input_image",
                    "node_name": "Input Image",
                    "output_port": "raster",
                    "session_file": "image.png",
                    "kind_label": "Image",
                    "kind_group": "none",
                },
            )
            self.assertTrue(Path(payload["file"]).is_file())
            self.assertEqual(payload["kind"], "Image")
            self.assertNotIn("warnings", payload)
            self.assertTrue(
                Path(payload["file"]).with_suffix(".meta.json").is_file()
            )

    def test_missing_file_still_fails_loudly(self) -> None:
        from lynceus.nodes._product_input import barrier_input_file

        with tempfile.TemporaryDirectory() as tmp:
            ctx = {
                "session_dir": str(Path(tmp) / "session"),
                "file_path": str(Path(tmp) / "ghost.png"),
                "node_iid": "img1",
                "module_id": "acme.input_image",
            }
            with self.assertRaises(RuntimeError):
                barrier_input_file(
                    ctx,
                    {
                        "node_id": "acme.input_image",
                        "node_name": "Input Image",
                        "session_file": "image.png",
                        "kind_group": "none",
                    },
                )


class ValidateCommandTests(unittest.TestCase):
    def _run(self, *argv: str) -> tuple[int, str]:
        import io
        from contextlib import redirect_stdout

        from lynceus.tools import extension_tool

        buf = io.StringIO()
        code = 0
        try:
            with redirect_stdout(buf):
                extension_tool.main(["validate", *argv])
        except SystemExit as exc:
            code = int(exc.code or 0) if str(exc.code).isdigit() else 1
        return code, buf.getvalue()

    def test_valid_third_party_input_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            node = Path(tmp) / "input_image.py"
            node.write_text(THIRD_PARTY_INPUT, encoding="utf-8")
            code, out = self._run(str(node))
            self.assertEqual(code, 0)
            self.assertIn("OK", out)

    def test_missing_author_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            node = Path(tmp) / "noname.py"
            node.write_text(
                'NODE_ID = "acme.noname"\n'
                "INPUTS = ()\n"
                "OUTPUTS = ()\n",
                encoding="utf-8",
            )
            code, out = self._run(str(node))
            self.assertNotEqual(code, 0)
            self.assertIn("NODE_AUTHOR", out)

    def test_theme_warnings_do_not_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            palette = Path(tmp) / "theme.json"
            palette.write_text(
                '{"background": "#000000", "watermark": "#ffffff"}',
                encoding="utf-8",
            )
            code, out = self._run(str(palette))
            self.assertEqual(code, 0)
            self.assertIn("watermark", out)

    def test_finished_package_validates(self) -> None:
        import zipfile

        from lynceus.tools import extension_tool

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "mynode.py"
            src.write_text(
                'NODE_ID = "acme.mynode"\n'
                'NODE_NAME = "My Node"\n'
                'NODE_AUTHOR = "Acme"\n'
                "INPUTS = ()\n"
                "OUTPUTS = ()\n",
                encoding="utf-8",
            )
            out = str(Path(tmp) / "mynode.lxpkg")
            extension_tool.main(["pack", str(src), "-o", out])
            self.assertEqual(
                sorted(zipfile.ZipFile(out).namelist()), ["mynode.py"]
            )
            code, result = self._run(out)
            self.assertEqual(code, 0)
            self.assertIn("OK", result)

    def test_placeholder_author_warns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            node = Path(tmp) / "anon.py"
            node.write_text(
                'NODE_ID = "acme.anon"\n'
                'NODE_NAME = "Anon"\n'
                'NODE_AUTHOR = "Test User"\n'
                "INPUTS = ()\n"
                "OUTPUTS = ()\n",
                encoding="utf-8",
            )
            code, out = self._run(str(node))
            self.assertEqual(code, 0)
            self.assertIn("placeholder", out)

    def test_barrier_outputs_without_declaration_warns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            node = Path(tmp) / "loose.py"
            node.write_text(
                'NODE_ID = "acme.loose"\n'
                'NODE_NAME = "Loose"\n'
                'NODE_AUTHOR = "Acme"\n'
                'from lynceus.nodes.ports import PortType\n'
                "INPUTS = ()\n"
                "OUTPUTS = (PortType.TABLE_CSV,)\n"
                "PROCESSING_SPECS = {\n"
                '    "barrier_task": "run_loose",\n'
                "}\n",
                encoding="utf-8",
            )
            code, out = self._run(str(node))
            self.assertEqual(code, 0)
            self.assertIn("session_file", out)

    def test_conflicting_port_defs_fail_in_pack(self) -> None:
        import json

        with tempfile.TemporaryDirectory() as tmp:
            pack = Path(tmp) / "duo"
            pack.mkdir()
            for stem, color in (("a", "#111111"), ("b", "#222222")):
                (pack / f"{stem}.py").write_text(
                    f'NODE_ID = "acme.{stem}"\n'
                    f'NODE_NAME = "{stem.upper()}"\n'
                    'NODE_AUTHOR = "Acme"\n'
                    "INPUTS = ()\n"
                    "OUTPUTS = ()\n"
                    "PORT_TYPE_DEFS = [\n"
                    '    {"id": "acme.shared", "display_name": "Shared",\n'
                    f'     "color": "{color}", "compatible": ("acme.shared",)}},\n'
                    "]\n",
                    encoding="utf-8",
                )
            (pack / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "id": "acme",
                        "version": "1.0.0",
                        "kind": "node",
                        "display_name": "Duo",
                        "author": "Acme",
                        "payload": {
                            "nodes": [
                                {"id": "acme.a", "file": "a.py"},
                                {"id": "acme.b", "file": "b.py"},
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )
            code, out = self._run(str(pack))
            self.assertNotEqual(code, 0)
            self.assertIn("Conflicting", out)

    def test_identical_port_defs_only_warn(self) -> None:
        import json

        with tempfile.TemporaryDirectory() as tmp:
            pack = Path(tmp) / "duo"
            pack.mkdir()
            defs = (
                "PORT_TYPE_DEFS = [\n"
                '    {"id": "acme.shared", "display_name": "Shared",\n'
                '     "color": "#111111", "compatible": ("acme.shared",)},\n'
                "]\n"
            )
            for stem in ("a", "b"):
                (pack / f"{stem}.py").write_text(
                    f'NODE_ID = "acme.{stem}"\n'
                    f'NODE_NAME = "{stem.upper()}"\n'
                    'NODE_AUTHOR = "Acme"\n'
                    "INPUTS = ()\n"
                    "OUTPUTS = ()\n" + defs,
                    encoding="utf-8",
                )
            (pack / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "id": "acme",
                        "version": "1.0.0",
                        "kind": "node",
                        "display_name": "Duo",
                        "author": "Acme",
                        "payload": {
                            "nodes": [
                                {"id": "acme.a", "file": "a.py"},
                                {"id": "acme.b", "file": "b.py"},
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )
            code, out = self._run(str(pack))
            self.assertEqual(code, 0)
            self.assertIn("repeated identically", out)


def tearDownModule() -> None:
    # The example node declares acme.image: keep it out of the global
    # registry for later suites (i18n coverage over all_specs).
    from lynceus.nodes.ports import _PORT_SPECS

    for key in [k for k in _PORT_SPECS if k.startswith("acme.")]:
        _PORT_SPECS.pop(key, None)


if __name__ == "__main__":
    unittest.main()
