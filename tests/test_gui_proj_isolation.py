# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""The GUI process must not load the geopandas/pyproj/pyogrio stack.

rasterio bundles one PROJ build; geopandas/pyproj/pyogrio bundle another.
Mixing both in one process crashes natively (access violation inside
``proj_9`` while previewing vectors after a run), so every GUI-process
path (viewers, QML styling, vector reading) reads GeoPackages with
sqlite3 + shapely instead.

``pyproj`` still gets imported by ``laspy`` at viewer discovery time
(its GeoTIFF VLR module) but the app never calls into it; the hard
invariant checked here is that geopandas/pyogrio never load and that no
GUI-process module imports any of the three.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_FORBIDDEN = ("geopandas", "pyproj", "pyogrio")

_GUI_MODULES = (
    "lynceus.processing.vector_table",
    "lynceus.processing.qml_style",
    "lynceus.ui.viewers.registry",  # discovers every viewer module
)


class GuiProcessStackTests(unittest.TestCase):
    def test_viewer_discovery_does_not_load_geopandas_or_pyogrio(self) -> None:
        code = textwrap.dedent(
            """
            import sys
            for module in %r:
                __import__(module)
            loaded = sorted(
                name for name in ("geopandas", "pyogrio") if name in sys.modules
            )
            print("|".join(loaded))
            """
            % (_GUI_MODULES,)
        )
        env = dict(os.environ)
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["PYTHONPATH"] = str(ROOT)
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=ROOT,
            env=env,
            timeout=180,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.strip(),
            "",
            f"GUI-process imports pulled in: {result.stdout.strip()}",
        )

    def test_gui_sources_never_import_the_other_proj_stack(self) -> None:
        targets = list((ROOT / "lynceus" / "ui").rglob("*.py"))
        targets.append(ROOT / "lynceus" / "processing" / "qml_style.py")
        targets.append(ROOT / "lynceus" / "processing" / "vector_table.py")
        offenders = []
        for path in targets:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    if name.split(".")[0] in _FORBIDDEN:
                        offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
        self.assertEqual(offenders, [], f"forbidden imports: {offenders}")


if __name__ == "__main__":
    unittest.main()
