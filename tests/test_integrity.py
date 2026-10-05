# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Execution integrity guards (escalabilidad P1).

C1: session_file + scoped_file in one barrier fails validate.
B4: declared output globs matching nothing warn at provenance time.
A6: input copies are atomic (no .tmp leftovers).
A1: oversized single raster reads fail fast with a legible error.
D1: import enforces folder==prefix and rejects taken ids.
"""

import tempfile
import unittest
from pathlib import Path


def _run_validate(path: str) -> tuple[int, str]:
    import io
    from contextlib import redirect_stdout

    from lynceus.tools import extension_tool

    buf = io.StringIO()
    code = 0
    try:
        with redirect_stdout(buf):
            extension_tool.main(["validate", path])
    except SystemExit as exc:
        code = int(exc.code or 0) if str(exc.code).isdigit() else 1
    return code, buf.getvalue()


class ScopedSessionMismatchTests(unittest.TestCase):
    def test_session_file_plus_scoped_file_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            node = Path(tmp) / "bad.py"
            node.write_text(
                'NODE_ID = "acme.bad"\n'
                'NODE_NAME = "Bad"\n'
                'NODE_AUTHOR = "Acme"\n'
                "INPUTS = ()\n"
                "OUTPUTS = ()\n"
                "PROCESSING_SPECS = {\n"
                '    "barrier_task": "run_bad",\n'
                '    "session_file": "out.bin",\n'
                "}\n"
                "def run_bad(ctx):\n"
                "    from lynceus.nodes._paths import scoped_file\n"
                '    return {"file": scoped_file(ctx, "out.bin")}\n',
                encoding="utf-8",
            )
            code, out = _run_validate(str(node))
            self.assertNotEqual(code, 0)
            self.assertIn("scoped_file", out)


class GlobMismatchWarningTests(unittest.TestCase):
    def _controller(self, messages: list):
        from lynceus.processing.controller import PipelineController

        ctrl = PipelineController()
        ctrl._modules = {
            "n1": "lynceus.nodes.table.concatenate_tables",
        }
        ctrl._run_configs = {}
        ctrl.on_message = (
            lambda text, kind="info": messages.append((text, kind))
        )
        return ctrl

    def _session(self, tmp: Path, product: str | None):
        session = Path(tmp) / "session"
        arts = session / "artifacts"
        if product is not None:
            target = arts / product
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"x")
        return session, arts

    def test_unmatched_globs_warn(self) -> None:
        from lynceus.processing.controller import PipelineController

        _ = PipelineController
        with tempfile.TemporaryDirectory() as tmp:
            session, arts = self._session(Path(tmp), "other.csv")
            messages: list = []
            ctrl = self._controller(messages)
            nodes = ctrl._save_provenance(
                session,
                {},
                Path(tmp) / "tiles",
                {"n1": "fp1"},
                ["n1"],
                [],
                {"n1": {"file": str(arts / "other.csv")}},
                {},
            )
            self.assertIn("n1", nodes)
            self.assertEqual(nodes["n1"]["files"], [])
            warnings = [t for t, k in messages if k == "warning"]
            self.assertTrue(
                any("output_globs" in t for t in warnings),
                warnings,
            )

    def test_matched_globs_stay_silent(self) -> None:
        from lynceus.processing.controller import PipelineController

        _ = PipelineController
        with tempfile.TemporaryDirectory() as tmp:
            session, arts = self._session(Path(tmp), "concatenated.csv")
            messages: list = []
            ctrl = self._controller(messages)
            nodes = ctrl._save_provenance(
                session,
                {},
                Path(tmp) / "tiles",
                {"n1": "fp1"},
                ["n1"],
                [],
                {"n1": {"file": str(arts / "concatenated.csv")}},
                {},
            )
            self.assertIn("concatenated.csv", nodes["n1"]["files"])
            self.assertFalse(
                [t for t, k in messages if k == "warning"]
            )


class AtomicCopyTests(unittest.TestCase):
    def test_no_tmp_leftovers(self) -> None:
        from lynceus.nodes._product_input import barrier_input_file

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "a.csv"
            src.write_text("h\n1\n", encoding="utf-8")
            session = Path(tmp) / "session"
            ctx = {
                "session_dir": str(session),
                "file_path": str(src),
                "node_iid": "t1",
                "module_id": "lynceus.nodes.table.input_table",
            }
            payload = barrier_input_file(
                ctx,
                {
                    "node_id": "lynceus.nodes.table.input_table",
                    "node_name": "Input Table",
                    "output_port": "table_csv",
                    "session_file": "table.csv",
                    "kind_label": "Table",
                    "kind_group": "table_csv",
                },
            )
            self.assertTrue(Path(payload["file"]).is_file())
            leftovers = list(session.rglob("*.tmp"))
            self.assertEqual(leftovers, [])


class RasterGuardTests(unittest.TestCase):
    def test_huge_read_fails_legibly(self) -> None:
        from lynceus.processing.raster import _guard_raster_alloc

        with self.assertRaises(RuntimeError) as raised:
            _guard_raster_alloc(1_000_000, 1_000_000, 1, "huge.tif")
        self.assertIn("windowed", str(raised.exception))

    def test_normal_read_passes(self) -> None:
        from lynceus.processing.raster import _guard_raster_alloc

        _guard_raster_alloc(100, 100, 3, "small.tif")


class ImportEnforcementTests(unittest.TestCase):
    def _node(self, node_id: str, folder: str | None, author="Acme") -> str:
        lines = [
            f'NODE_ID = "{node_id}"',
            'NODE_NAME = "X"',
            f'NODE_AUTHOR = "{author}"',
            "INPUTS = ()",
            "OUTPUTS = ()",
        ]
        if folder is not None:
            lines.append(f'NODE_FOLDER = "{folder}"')
        return "\n".join(lines) + "\n"

    def test_folder_must_match_prefix(self) -> None:
        from lynceus.plugins.importer import (
            ExtensionImportError,
            import_extension,
        )

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src" / "x.py"
            src.parent.mkdir(parents=True)
            src.write_text(
                self._node("acme.x", "other"), encoding="utf-8"
            )
            ext = Path(tmp) / "extensions"
            ext.mkdir()
            with self.assertRaises(ExtensionImportError) as raised:
                import_extension(src, ext)
            self.assertIn("must match", str(raised.exception))

    def test_taken_id_from_another_file_fails(self) -> None:
        from lynceus.plugins.importer import (
            ExtensionImportError,
            import_extension,
        )

        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "one" / "x.py"
            first.parent.mkdir(parents=True)
            first.write_text(
                self._node("acme.x", "acme"), encoding="utf-8"
            )
            ext = Path(tmp) / "extensions"
            ext.mkdir()
            import_extension(first, ext)
            second = Path(tmp) / "two" / "x.py"
            second.parent.mkdir(parents=True)
            second.write_text(
                self._node("acme.x", "acme").replace(
                    'NODE_NAME = "X"', 'NODE_NAME = "Y"'
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ExtensionImportError) as raised:
                import_extension(second, ext)
            self.assertIn("already installed", str(raised.exception))

    def test_same_content_idempotent(self) -> None:
        from lynceus.plugins.importer import import_extension

        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "one" / "x.py"
            first.parent.mkdir(parents=True)
            first.write_text(
                self._node("acme.x", "acme"), encoding="utf-8"
            )
            ext = Path(tmp) / "extensions"
            ext.mkdir()
            import_extension(first, ext)
            clone = Path(tmp) / "two" / "x.py"
            clone.parent.mkdir(parents=True)
            clone.write_text(
                self._node("acme.x", "acme"), encoding="utf-8"
            )
            result = import_extension(clone, ext)
            self.assertFalse(result.created)


class SourceHashTests(unittest.TestCase):
    def test_module_edits_change_source_hash(self) -> None:
        from lynceus.plugins.registry import PluginManager
        from lynceus.processing.controller import _module_source_files

        with tempfile.TemporaryDirectory() as tmp:
            node = Path(tmp) / "n.py"
            node.write_text(
                'NODE_ID = "acme.n"\n'
                'NODE_NAME = "N"\n'
                'NODE_AUTHOR = "Acme"\n'
                "INPUTS = ()\n"
                "OUTPUTS = ()\n",
                encoding="utf-8",
            )
            mgr = PluginManager(extensions_dir=tmp)
            mgr.discover()
            before = _module_source_files("acme.n", {}, registry=mgr)
            self.assertEqual(len(before), 1)
            node.write_text(
                node.read_text(encoding="utf-8") + "# fix\n",
                encoding="utf-8",
            )
            after = _module_source_files("acme.n", {}, registry=mgr)
            self.assertEqual(len(after), 1)
            self.assertNotEqual(before, after)

    def test_unresolvable_modules_contribute_nothing(self) -> None:
        from lynceus.processing.controller import _module_source_files

        self.assertEqual(_module_source_files("nope.missing", {}), [])

    def test_fingerprint_moves_with_source_edits(self) -> None:
        from lynceus.plugins.registry import PluginManager
        from lynceus.processing.controller import PipelineController

        with tempfile.TemporaryDirectory() as tmp:
            node = Path(tmp) / "n.py"
            node.write_text(
                'NODE_ID = "acme.n"\n'
                'NODE_NAME = "N"\n'
                'NODE_AUTHOR = "Acme"\n'
                "INPUTS = ()\n"
                "OUTPUTS = ()\n",
                encoding="utf-8",
            )
            mgr = PluginManager(extensions_dir=tmp)
            mgr.discover()
            ctrl = PipelineController()
            ctrl._modules = {"n1": "acme.n"}
            ctrl._run_configs = {}
            ctrl._registry = mgr
            fp1 = ctrl._node_fingerprints(["n1"], [], {}, {})["n1"]
            node.write_text(
                node.read_text(encoding="utf-8") + "# fix\n",
                encoding="utf-8",
            )
            fp2 = ctrl._node_fingerprints(["n1"], [], {}, {})["n1"]
            self.assertNotEqual(fp1, fp2)


if __name__ == "__main__":
    unittest.main()
