# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Derive PyInstaller hidden imports from the source tree (stdlib only).

The node/viewer registries resolve modules by dotted name at runtime
(`importlib.import_module`, `pkgutil.walk_packages`, `__import__`), which
PyInstaller's static analysis cannot see. This walks the package
directories and returns every dotted module name so the spec can list
them explicitly. Pure path walk: imports nothing (safe to run at spec
time, no Qt, no third-party deps).
"""

from __future__ import annotations

import os

# Dotted roots whose modules are resolved dynamically at runtime
# (node ids, viewer registry, .ui promoted widgets via ui_loader, ...).
# The whole package is covered: anything importable by string must ship.
_DYNAMIC_ROOTS = ("lynceus",)


def collect_hiddenimports(repo_root: str | os.PathLike) -> list[str]:
    """Every dotted module under the dynamically-resolved roots."""
    root = os.path.abspath(repo_root)
    found: set[str] = set()
    for dotted in _DYNAMIC_ROOTS:
        package_dir = os.path.join(root, *dotted.split("."))
        if not os.path.isdir(package_dir):
            continue
        for dirpath, _dirnames, filenames in os.walk(package_dir):
            for filename in filenames:
                if not filename.endswith(".py"):
                    continue
                full = os.path.join(dirpath, filename)
                rel = os.path.relpath(full, root)
                mod = rel[:-3].replace(os.sep, ".")
                if mod.endswith(".__init__"):
                    mod = mod[: -len(".__init__")]
                found.add(mod)
    return sorted(found)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    for name in collect_hiddenimports(os.path.join(here, os.pardir)):
        print(name)
