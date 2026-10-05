# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Qt-free tests for template discovery and metadata (lynceus/templates.py)."""

import json
import tempfile
import unittest
from pathlib import Path

from lynceus import project as project_io
from lynceus import templates as tmpl


def _write_lynx(path: Path, nodes, template_meta=None, name="doc") -> None:
    doc = {
        "format": project_io.PROJECT_FORMAT,
        "version": project_io.PROJECT_VERSION,
        "name": name,
        "nodes": nodes,
        "edges": [],
    }
    if template_meta is not None:
        doc[tmpl.TEMPLATE_META_KEY] = template_meta
    path.write_text(json.dumps(doc), encoding="utf-8")


LOAD = "lynceus.nodes.lidar.source.load_las_laz"
DTM = "lynceus.nodes.lidar.terrain.generate_dtm"


class TemplateDiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.bundled = self.root / "bundled"
        self.user = self.root / "user"
        self.bundled.mkdir()
        self.user.mkdir()

    def _list(self):
        return tmpl.list_templates(
            user_dir=self.user, bundled_dir=self.bundled
        )

    def test_valid_template_with_metadata(self) -> None:
        _write_lynx(
            self.user / "chm_basic.lynx",
            [{"id": LOAD, "iid": "a", "x": 0, "y": 0, "config": {}}],
            template_meta={
                "name": "CHM Básico",
                "description": "Flujo mínimo de CHM.",
                "author": "Nico",
                "version": "1.0",
                "tags": ["chm"],
            },
        )
        infos = self._list()
        self.assertEqual(len(infos), 1)
        info = infos[0]
        self.assertTrue(info.valid)
        self.assertEqual(info.name, "CHM Básico")
        self.assertEqual(info.description, "Flujo mínimo de CHM.")
        self.assertEqual(info.author, "Nico")
        self.assertEqual(info.version, "1.0")
        self.assertEqual(info.tags, ("chm",))
        self.assertEqual(info.source, "user")
        self.assertEqual(info.node_count, 1)

    def test_metadata_defaults_without_block(self) -> None:
        _write_lynx(
            self.user / "plain.lynx",
            [{"id": DTM, "iid": "a", "x": 0, "y": 0, "config": {}}],
            name="Plain Doc",
        )
        (info,) = self._list()
        self.assertTrue(info.valid)
        self.assertEqual(info.name, "Plain Doc")
        self.assertEqual(info.description, "")

    def test_bundled_before_user_and_sources(self) -> None:
        _write_lynx(self.bundled / "b.lynx", [], name="B")
        _write_lynx(self.user / "u.lynx", [], name="U")
        infos = self._list()
        self.assertEqual([i.source for i in infos], ["bundled", "user"])

    def test_invalid_document_reported_not_raised(self) -> None:
        (self.user / "broken.lynx").write_text("{no json", encoding="utf-8")
        (info,) = self._list()
        self.assertFalse(info.valid)
        self.assertTrue(info.error)
        self.assertEqual(info.name, "broken")

    def test_missing_nodes_flagged(self) -> None:
        _write_lynx(
            self.user / "ghost.lynx",
            [{"id": "nope.missing_node", "iid": "a", "x": 0, "y": 0}],
        )
        (info,) = self._list()
        self.assertFalse(info.valid)
        self.assertIn("nope.missing_node", info.error)

    def test_user_dir_created_on_demand(self) -> None:
        missing = self.root / "fresh"
        infos = tmpl.list_templates(
            user_dir=missing, bundled_dir=self.bundled
        )
        self.assertTrue(missing.is_dir())
        self.assertEqual(infos, [])

    def test_save_target_prefers_writable_bundled(self) -> None:
        bundled = self.root / "repo_templates"
        user = self.root / "user_templates"
        target = tmpl.save_target_dir(
            bundled_dir=bundled, user_dir=user
        )
        self.assertEqual(target, bundled)
        self.assertTrue(bundled.is_dir())

    def test_save_target_falls_back_to_user(self) -> None:
        blocker = self.root / "blocker"
        blocker.write_text("x", encoding="utf-8")
        user = self.root / "user_templates"
        target = tmpl.save_target_dir(
            bundled_dir=blocker, user_dir=user
        )
        self.assertEqual(target, user)
        self.assertTrue(user.is_dir())

    def test_build_template_doc_and_file_name(self) -> None:
        doc = tmpl.build_template_doc(
            {"nodes": [], "edges": [], "view": {}},
            {
                "name": "Mi Plantilla!",
                "description": "d",
                "author": "a",
                "version": "1",
                "tags": "x, y",
            },
        )
        self.assertEqual(doc["name"], "Mi Plantilla!")
        meta = doc[tmpl.TEMPLATE_META_KEY]
        self.assertEqual(meta["tags"], ["x", "y"])
        self.assertEqual(
            tmpl.template_file_name("Mi Plantilla!"), "Mi_Plantilla.lynx"
        )
        self.assertEqual(tmpl.template_file_name("   "), "template.lynx")


if __name__ == "__main__":
    unittest.main()
