# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Install/source-agnostic resource paths (domain, no Qt).

Asset loads (QSS, `.ui` files, bundled templates) must not depend on the
current working directory: the app also runs from an installed executable
(PyInstaller/Nuitka bundle) and from `python <path>/main.py` launched
anywhere. Everything resolves through this module:

- frozen bundle (``sys.frozen`` + ``sys._MEIPASS``) → the bundle dir;
- source checkout → the repository root (parent of this package).

Rule: never `open()` a repo-relative asset path directly; use
``resource_path()``.
"""

from __future__ import annotations

import sys
from pathlib import Path


def project_root() -> Path:
    """Base dir for bundled/source assets (bundle dir or repo root)."""
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
    return Path(__file__).resolve().parent.parent


def resource_path(relative: str | Path) -> Path:
    """Absolute path of a repo/bundle-relative asset (QSS, `.ui`, ...)."""
    return project_root() / Path(relative)
