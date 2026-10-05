# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Frozen-bundle discovery fallback (stdlib only, no Qt).

The AST node discovery requires `.py` sources. Frozen imports resolve
from the archive (no `.py` origin), so `_module_spec_origin` falls back
to the node sources shipped as data (see `build/LynceusScan.spec`).
These tests simulate a frozen origin and assert the fallback resolves —
and stays scoped to `lynceus.nodes.*`.
"""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from lynceus.plugins import registry


def _node_source(*parts: str) -> Path:
    return (
        Path(registry.__file__).resolve().parent.parent / "nodes" / Path(*parts)
    )


class FrozenDiscoveryTests(unittest.TestCase):
    def test_pyc_origin_falls_back_to_shipped_sources(self) -> None:
        target = "lynceus.nodes.lidar.terrain.generate_dtm"
        real = _node_source("lidar", "terrain", "generate_dtm.py")
        self.assertTrue(real.is_file())
        real_find_spec = importlib.util.find_spec

        def fake_find_spec(name: str):
            if name == target:
                return SimpleNamespace(origin=str(real.with_suffix(".pyc")))
            return real_find_spec(name)

        with mock.patch.object(
            importlib.util, "find_spec", side_effect=fake_find_spec
        ):
            self.assertEqual(registry._module_spec_origin(target), real)

    def test_non_node_modules_have_no_fallback(self) -> None:
        fake = SimpleNamespace(origin="C:\\bundle\\controller.pyc")
        with mock.patch.object(
            importlib.util, "find_spec", return_value=fake
        ):
            self.assertIsNone(
                registry._module_spec_origin("lynceus.processing.controller")
            )

    def test_missing_spec_stays_none(self) -> None:
        with mock.patch.object(
            importlib.util, "find_spec", return_value=None
        ):
            self.assertIsNone(
                registry._module_spec_origin("lynceus.nodes.nope.missing")
            )


if __name__ == "__main__":
    unittest.main()
