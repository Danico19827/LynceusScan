# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Windows VersionInfo for the PyInstaller bundle (stdlib only).

Generated from ``lynceus.__version__`` at spec time so the Explorer
details tab and the installer always match the app version. Safe to run
at spec time: parses the version literal via AST, never imports Qt.
"""

from __future__ import annotations

import ast
import re

_VERSION_RE = re.compile(r"^(\d+)\.(\d+)(?:\.(\d+))?(?:[.-]?(?:b|beta|rc|alpha)\d*)?$")


def app_version(repo_root: str) -> str:
    """Read ``__version__`` from lynceus/__init__.py (AST, no import)."""
    import os

    init_path = os.path.join(repo_root, "lynceus", "__init__.py")
    with open(init_path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "__version__"
            and isinstance(node.value, ast.Constant)
        ):
            return str(node.value.value)
    raise ValueError("No __version__ literal in lynceus/__init__.py")


def version_tuple(version: str) -> tuple[int, int, int, int]:
    """'0.1.0' -> (0, 1, 0, 0); a pre-release suffix ('0.1.0b1') keeps the
    numeric tuple — the full string still lands in FileVersion strings."""
    match = _VERSION_RE.match(version.strip())
    if not match:
        raise ValueError(f"Invalid version {version!r} (expected X.Y[.Z][suffix])")
    major, minor, patch = match.groups()
    return (int(major), int(minor), int(patch or 0), 0)


def version_info_txt(repo_root: str) -> str:
    """VSVersionInfo text for PyInstaller's ``version=`` EXE argument."""
    version = app_version(repo_root)
    numbers = ", ".join(str(n) for n in version_tuple(version))
    return f"""VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=({numbers}),
    prodvers=({numbers}),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo(
      [
        StringTable(
          u'040904B0',
          [
            StringStruct(u'CompanyName', u'Taritolay, Nicolas Daniel'),
            StringStruct(u'FileDescription', u'LynceusScan LiDAR processing'),
            StringStruct(u'FileVersion', u'{version}'),
            StringStruct(u'InternalName', u'LynceusScan'),
            StringStruct(u'LegalCopyright', u'GPL-3.0-or-later'),
            StringStruct(u'OriginalFilename', u'LynceusScan.exe'),
            StringStruct(u'ProductName', u'LynceusScan'),
            StringStruct(u'ProductVersion', u'{version}')
          ]
        )
      ]
    ),
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)
"""
