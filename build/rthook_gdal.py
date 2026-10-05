# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""PyInstaller runtime hook: point GDAL/PROJ at the collected data dirs.

The frozen `rasterio` package has no real `__file__`, so its package-
relative `gdal_data`/`proj_data` lookup fails. The spec collects those
dirs as data; this hook exports them before `rasterio` is imported.
Only fills unset variables and existing directories (never overrides).
"""

import os
import sys


def _data_root() -> str:
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return str(meipass)
    return os.path.abspath(os.path.dirname(sys.executable))


_ROOT = _data_root()
for _var, _sub in (
    ("GDAL_DATA", os.path.join("rasterio", "gdal_data")),
    ("PROJ_DATA", os.path.join("rasterio", "proj_data")),
    ("PROJ_LIB", os.path.join("rasterio", "proj_data")),
):
    _path = os.path.join(_ROOT, _sub)
    if _var not in os.environ and os.path.isdir(_path):
        os.environ[_var] = _path
