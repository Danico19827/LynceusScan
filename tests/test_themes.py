# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Theme packs (kind="theme"): manifest, import routing and palettes.

Also covers pack *packages* (.lxpkg with a manifest): theme/locale kinds
extract into their kind folder, node kinds stay legacy copies.
"""

import json
import tempfile
import unittest
from pathlib import Path

from lynceus.plugins.importer import import_extension
from lynceus.plugins.manifest import (
    ExtensionManifest,
    ManifestError,
)
from lynceus.plugins.registry import PluginManager
from lynceus.plugins.theme import (
    DEFAULT_THEME_ID,
    THEME_ROLES,
    ThemeManager,
)

THEME_MANIFEST = {
    "schema_version": 2,
    "id": "ocean",
    "version": "1.0.0",
    "kind": "theme",
    "display_name": "Ocean",
    "description": "Test theme.",
    "author": "Test Author",
    "payload": {"palette": "theme.json"},
}


def _write_pack(root: Path, palette: object) -> Path:
    (root).mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps(THEME_MANIFEST), encoding="utf-8"
    )
    if isinstance(palette, str):
        (root / "theme.json").write_text(palette, encoding="utf-8")
    else:
        (root / "theme.json").write_text(
            json.dumps(palette), encoding="utf-8"
        )
    return root


class ThemeManifestTests(unittest.TestCase):
    def test_valid_theme_parses_with_empty_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_pack(Path(tmp) / "ocean", dict(THEME_ROLES))
            manifest = ExtensionManifest.load(root / "manifest.json")
            self.assertEqual(manifest.kind, "theme")
            self.assertEqual(manifest.palette, "theme.json")
            self.assertEqual(manifest.nodes, ())
            self.assertTrue(manifest.palette_file().exists())

    def test_theme_without_palette_rejected(self) -> None:
        bad = dict(THEME_MANIFEST)
        bad["payload"] = {}
        with self.assertRaises(ManifestError):
            ExtensionManifest.from_dict(bad, root=".")

    def test_unknown_kind_still_rejected(self) -> None:
        bad = dict(THEME_MANIFEST)
        bad["kind"] = "wallpaper"
        with self.assertRaises(ManifestError):
            ExtensionManifest.from_dict(bad, root=".")

    def test_palette_escaping_root_rejected(self) -> None:
        bad = dict(THEME_MANIFEST)
        bad["payload"] = {"palette": "../evil.json"}
        with self.assertRaises(ManifestError):
            ExtensionManifest.from_dict(bad, root=".")


class ThemeImportTests(unittest.TestCase):
    def test_theme_pack_lands_in_themes_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = _write_pack(Path(tmp) / "src", {"accent": "#3f6fb4"})
            ext = Path(tmp) / "extensions"
            ext.mkdir()
            result = import_extension(src, ext)
            self.assertEqual(result.kind, "pack")
            self.assertEqual(
                result.destination, (ext / "themes" / "ocean").resolve()
            )
            self.assertTrue(
                (ext / "themes" / "ocean" / "manifest.json").exists()
            )


class ThemePackageTests(unittest.TestCase):
    def _make_lxpkg(self, directory: Path, name: str = "ocean.lxpkg") -> Path:
        import zipfile

        pack = directory / name
        with zipfile.ZipFile(pack, "w") as archive:
            archive.writestr("manifest.json", json.dumps(THEME_MANIFEST))
            archive.writestr(
                "theme.json", json.dumps({"accent": "#3f6fb4"})
            )
        return pack

    def test_theme_lxpkg_extracts_to_themes_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = self._make_lxpkg(Path(tmp))
            ext = Path(tmp) / "extensions"
            ext.mkdir()
            result = import_extension(src, ext)
            self.assertEqual(result.kind, "pack")
            self.assertEqual(
                result.destination, (ext / "themes" / "ocean").resolve()
            )
            self.assertTrue(
                (ext / "themes" / "ocean" / "theme.json").exists()
            )
            # No legacy copy: the zip is not kept, nothing materializes.
            self.assertFalse((ext / "ocean.lxpkg").exists())
            manifest = ExtensionManifest.load(
                ext / "themes" / "ocean" / "manifest.json"
            )
            self.assertEqual(manifest.kind, "theme")

    def test_theme_lxpkg_reimport_needs_force(self) -> None:
        from lynceus.plugins.importer import ExtensionImportError

        with tempfile.TemporaryDirectory() as tmp:
            src = self._make_lxpkg(Path(tmp))
            ext = Path(tmp) / "extensions"
            ext.mkdir()
            import_extension(src, ext)
            with self.assertRaises(ExtensionImportError):
                import_extension(src, ext)
            result = import_extension(src, ext, force=True)
            self.assertTrue(result.created)
            self.assertTrue(
                (ext / "themes" / "ocean" / "theme.json").exists()
            )

    def test_node_pack_lxpkg_stays_legacy_copy(self) -> None:
        import zipfile

        node_manifest = {
            "schema_version": 2,
            "id": "demo-pack",
            "version": "1.0.0",
            "kind": "node",
            "display_name": "Demo",
            "description": "Legacy node pack.",
            "author": "Test Author",
            "payload": {
                "nodes": [{"id": "demo-pack.x", "file": "x.py"}]
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            pack = Path(tmp) / "demo-pack.lxpkg"
            with zipfile.ZipFile(pack, "w") as archive:
                archive.writestr(
                    "manifest.json", json.dumps(node_manifest)
                )
            ext = Path(tmp) / "extensions"
            ext.mkdir()
            result = import_extension(pack, ext)
            self.assertEqual(
                result.destination, (ext / "demo-pack.lxpkg").resolve()
            )
            self.assertTrue((ext / "demo-pack.lxpkg").exists())


class LocalePackPackageTests(unittest.TestCase):
    def test_locale_lxpkg_extracts_to_locales_dir(self) -> None:
        import zipfile

        from lynceus.plugins.registry import PluginManager

        locale_manifest = {
            "schema_version": 2,
            "id": "xx",
            "version": "1.0.0",
            "kind": "locale",
            "display_name": "Test",
            "description": "Test locale pack.",
            "author": "Test Author",
            "payload": {"catalog": "xx.json"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            pack = Path(tmp) / "xx.lxpkg"
            with zipfile.ZipFile(pack, "w") as archive:
                archive.writestr(
                    "manifest.json", json.dumps(locale_manifest)
                )
                archive.writestr(
                    "xx.json", json.dumps({"xx": {"OK": "OK"}})
                )
            ext = Path(tmp) / "extensions"
            ext.mkdir()
            result = import_extension(pack, ext)
            self.assertEqual(result.kind, "pack")
            self.assertEqual(
                result.destination, (ext / "locales" / "xx").resolve()
            )
            self.assertTrue((ext / "locales" / "xx" / "xx.json").exists())
            self.assertFalse((ext / "xx.lxpkg").exists())
            # Discovery registers the extracted folder pack directly.
            registry = PluginManager(extensions_dir=str(ext))
            registry.discover()
            manifests = registry.extension_manifests()
            self.assertIn("xx", manifests)
            self.assertEqual(manifests["xx"].kind, "locale")


class ThemeManagerTests(unittest.TestCase):
    def _manager(self, tmp: str) -> ThemeManager:
        _write_pack(
            Path(tmp) / "themes" / "ocean",
            {"accent": "#3f6fb4", "background": "#0b1a30"},
        )
        registry = PluginManager(extensions_dir=tmp)
        registry.discover()
        return ThemeManager(registry=registry)

    def test_available_lists_default_and_pack(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mgr = self._manager(tmp)
            self.assertEqual(mgr.available(), [DEFAULT_THEME_ID, "ocean"])
            self.assertEqual(mgr.display_name("ocean"), "Ocean")
            self.assertEqual(mgr.display_name("default"), "Default")

    def test_palette_merges_pack_over_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mgr = self._manager(tmp)
            pal = mgr.palette("ocean")
            self.assertEqual(pal["accent"], "#3f6fb4")
            self.assertEqual(pal["background"], "#0b1a30")
            # Untouched roles keep defaults.
            self.assertEqual(pal["danger"], THEME_ROLES["danger"])
            self.assertEqual(set(pal), set(THEME_ROLES))

    def test_unknown_theme_falls_back_to_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mgr = self._manager(tmp)
            self.assertEqual(mgr.palette("nope"), dict(THEME_ROLES))
            mgr.set_theme("nope")
            self.assertEqual(mgr.theme, DEFAULT_THEME_ID)

    def test_corrupt_palette_degrades_to_defaults(self) -> None:
        # A corrupt palette never enables the theme (same rule as locale
        # catalogs); resolving it still returns the defaults.
        with tempfile.TemporaryDirectory() as tmp:
            _write_pack(Path(tmp) / "themes" / "ocean", "{not json")
            registry = PluginManager(extensions_dir=tmp)
            registry.discover()
            mgr = ThemeManager(registry=registry)
            self.assertNotIn("ocean", mgr.available())
            self.assertEqual(mgr.palette("ocean"), dict(THEME_ROLES))

    def test_unknown_roles_and_bad_hex_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _write_pack(
                Path(tmp) / "themes" / "ocean",
                {
                    "accent": "#3f6fb4",
                    "watermark": "#ffffff",
                    "danger": "red",
                    "amber": "#12345",
                },
            )
            registry = PluginManager(extensions_dir=tmp)
            registry.discover()
            mgr = ThemeManager(registry=registry)
            pal = mgr.palette("ocean")
            self.assertEqual(pal["accent"], "#3f6fb4")
            self.assertNotIn("watermark", pal)
            self.assertEqual(pal["danger"], THEME_ROLES["danger"])
            self.assertEqual(pal["amber"], THEME_ROLES["amber"])


class ThemeScaffoldTests(unittest.TestCase):
    def test_scaffold_theme_creates_valid_pack(self) -> None:
        from lynceus.tools import extension_tool

        with tempfile.TemporaryDirectory() as tmp:
            target = str(Path(tmp) / "my-theme")
            extension_tool.main(
                [
                    "scaffold",
                    target,
                    "--kind",
                    "theme",
                    "--author",
                    "Test Author",
                ]
            )
            manifest = ExtensionManifest.load(
                Path(target) / "manifest.json"
            )
            self.assertEqual(manifest.kind, "theme")
            palette = json.loads(
                (Path(target) / "theme.json").read_text(encoding="utf-8")
            )
            self.assertEqual(palette, dict(THEME_ROLES))


class ShippedPacksTests(unittest.TestCase):
    def test_every_shipped_pack_validates(self) -> None:
        from lynceus.plugins.theme import _validated_roles

        themes_dir = (
            Path(__file__).resolve().parent.parent / "extensions" / "themes"
        )
        packs = sorted(p for p in themes_dir.iterdir() if p.is_dir())
        self.assertGreaterEqual(len(packs), 22)
        for pack_dir in packs:
            with self.subTest(pack=pack_dir.name):
                manifest = ExtensionManifest.load(
                    pack_dir / "manifest.json"
                )
                self.assertEqual(manifest.kind, "theme")
                self.assertEqual(manifest.nodes, ())
                palette = json.loads(
                    manifest.palette_file().read_text(encoding="utf-8")
                )
                roles = _validated_roles(manifest.id, palette)
                self.assertEqual(set(roles), set(THEME_ROLES))


if __name__ == "__main__":
    unittest.main()
