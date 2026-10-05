# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Installer readiness: spec bundle contents and frozen seed behavior."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class SpecBundleTests(unittest.TestCase):
    def test_spec_ships_themes_not_locales(self) -> None:
        spec = (ROOT / "build" / "LynceusScan.spec").read_text(
            encoding="utf-8"
        )
        self.assertIn('"extensions/themes"', spec)
        self.assertNotIn('"extensions/locales"', spec)
        self.assertIn('"lynceus/nodes"', spec)
        self.assertIn("console=False", spec)
        self.assertIn("version=", spec)

    def test_version_info_matches_app_version(self) -> None:
        sys.path.insert(0, str(ROOT / "build"))
        try:
            from version_info import app_version, version_info_txt, version_tuple
        finally:
            sys.path.remove(str(ROOT / "build"))
        from lynceus import __version__

        self.assertEqual(app_version(str(ROOT)), __version__)
        numbers = version_tuple(__version__)
        self.assertEqual(len(numbers), 4)
        self.assertTrue(all(isinstance(n, int) for n in numbers))
        txt = version_info_txt(str(ROOT))
        self.assertIn("VSVersionInfo(", txt)
        self.assertIn(__version__, txt)
        with self.assertRaises(ValueError):
            version_tuple("nope")

    def test_source_extensions_dir_is_repo_folder(self) -> None:
        self.assertFalse(getattr(sys, "frozen", False))
        from lynceus.plugins.registry import DEFAULT_EXTENSIONS_DIR

        self.assertEqual(
            DEFAULT_EXTENSIONS_DIR.resolve(), (ROOT / "extensions").resolve()
        )

    def test_guaranteed_third_party_imports(self) -> None:
        # The documented extension dependency contract: everything an
        # extension may import without vendoring or try/except.
        import importlib

        guaranteed = {
            "PySide6": "PySide6",
            "numpy": "numpy",
            "laspy": "laspy",
            "PyOpenGL": "OpenGL",
            "rasterio": "rasterio",
            "scipy": "scipy",
            "shapely": "shapely",
            "geopandas": "geopandas",
            "psutil": "psutil",
            "Pillow": "PIL",
        }
        for label, module in sorted(guaranteed.items()):
            with self.subTest(dep=label):
                self.assertIsNotNone(importlib.import_module(module))


class SeedTests(unittest.TestCase):
    def _bundled(self, tmp: Path) -> Path:
        src = tmp / "bundled" / "extensions"
        (src / "themes" / "ocean").mkdir(parents=True)
        (src / "themes" / "ocean" / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "id": "ocean",
                    "version": "1.0.0",
                    "kind": "theme",
                    "display_name": "Ocean",
                    "author": "Test",
                    "payload": {"palette": "theme.json"},
                }
            ),
            encoding="utf-8",
        )
        (src / "themes" / "ocean" / "theme.json").write_text(
            json.dumps({"accent": "#3f6fb4"}), encoding="utf-8"
        )
        (src / "nodes").mkdir(parents=True, exist_ok=True)
        (src / "nodes" / "helper.py").write_text("# helper\n", encoding="utf-8")
        return src

    def test_seed_copies_missing_entries_frozen(self) -> None:
        from lynceus.plugins.importer import seed_bundled_extensions

        with tempfile.TemporaryDirectory() as tmp:
            src = self._bundled(Path(tmp))
            dest = Path(tmp) / "user" / "extensions"
            sys.frozen = True  # type: ignore[attr-defined]
            try:
                seeded = seed_bundled_extensions(
                    target=dest, bundled=src
                )
            finally:
                del sys.frozen  # type: ignore[attr-defined]
            self.assertEqual(
                sorted(seeded), ["nodes/helper.py", "themes/ocean"]
            )
            self.assertTrue(
                (dest / "themes" / "ocean" / "manifest.json").exists()
            )
            self.assertTrue((dest / "nodes" / "helper.py").exists())

    def test_seed_never_overwrites_user_copies(self) -> None:
        from lynceus.plugins.importer import seed_bundled_extensions

        with tempfile.TemporaryDirectory() as tmp:
            src = self._bundled(Path(tmp))
            dest = Path(tmp) / "user" / "extensions"
            mine = dest / "themes" / "ocean"
            mine.mkdir(parents=True)
            (mine / "manifest.json").write_text("mine", encoding="utf-8")
            sys.frozen = True  # type: ignore[attr-defined]
            try:
                seeded = seed_bundled_extensions(
                    target=dest, bundled=src
                )
            finally:
                del sys.frozen  # type: ignore[attr-defined]
            self.assertEqual(seeded, ["nodes/helper.py"])
            self.assertEqual(
                (mine / "manifest.json").read_text(encoding="utf-8"), "mine"
            )

    def test_seed_noop_in_source(self) -> None:
        from lynceus.plugins.importer import seed_bundled_extensions

        with tempfile.TemporaryDirectory() as tmp:
            src = self._bundled(Path(tmp))
            dest = Path(tmp) / "user" / "extensions"
            self.assertEqual(
                seed_bundled_extensions(target=dest, bundled=src), []
            )
            self.assertFalse(dest.exists())

    def test_seeded_packs_enable_after_discovery(self) -> None:
        # Regression: on frozen first runs discovery ran BEFORE the seed
        # (theme setup precedes it in main), caching an empty registry
        # while the dialog still listed the packs from disk. main() now
        # seeds first: seed -> discover must expose the pack.
        from lynceus.plugins.importer import seed_bundled_extensions
        from lynceus.plugins.registry import PluginManager
        from lynceus.plugins.theme import ThemeManager

        with tempfile.TemporaryDirectory() as tmp:
            src = self._bundled(Path(tmp))
            dest = Path(tmp) / "user" / "extensions"
            sys.frozen = True  # type: ignore[attr-defined]
            try:
                seeded = seed_bundled_extensions(target=dest, bundled=src)
            finally:
                del sys.frozen  # type: ignore[attr-defined]
            self.assertEqual(
                sorted(seeded), ["nodes/helper.py", "themes/ocean"]
            )
            registry = PluginManager(extensions_dir=str(dest))
            registry.discover()
            themes = ThemeManager(registry=registry)
            self.assertIn("ocean", themes.available())
            self.assertEqual(
                themes.palette("ocean")["accent"], "#3f6fb4"
            )


class FrozenOutputRootTests(unittest.TestCase):
    def test_frozen_sessions_leave_the_install_dir(self) -> None:
        import os

        from lynceus.processing.controller import default_output_root

        with tempfile.TemporaryDirectory() as local:
            old = os.environ.get("LOCALAPPDATA")
            os.environ["LOCALAPPDATA"] = local
            sys.frozen = True  # type: ignore[attr-defined]
            try:
                root = default_output_root()
            finally:
                del sys.frozen  # type: ignore[attr-defined]
                if old is None:
                    del os.environ["LOCALAPPDATA"]
                else:
                    os.environ["LOCALAPPDATA"] = old
            self.assertEqual(
                root, Path(local) / "LynceusScan" / "output"
            )

    def test_source_output_root_is_repo_folder(self) -> None:
        from lynceus.processing.controller import default_output_root

        self.assertEqual(
            default_output_root(), ROOT / "output"
        )


if __name__ == "__main__":
    unittest.main()
