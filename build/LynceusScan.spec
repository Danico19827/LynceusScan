# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""PyInstaller bundle spec (one-dir): LynceusScan.exe + dependency folder.

Build from the repo root::

    .venv\\Scripts\\pyinstaller build\\LynceusScan.spec

Notes:
- one-dir (not one-file): fewer antivirus false positives, and the AST
  node discovery reads the shipped ``lynceus/nodes/**/*.py`` sources.
  Nothing is obfuscated: the core is GPL-3.0-or-later, sources ship openly.
- ``console=False``: no console window in the release build (flip to
  ``True`` temporarily for debugging frozen startup issues).
- ``extensions/themes`` ships the public theme packs when present
  (seeded into %APPDATA% on first frozen run). Locale packs are NOT
  bundled (English base; locales arrive as downloaded extensions);
  private/commercial extensions are added at packaging time, never here.
"""

import os

SPEC_DIR = os.path.dirname(os.path.abspath(SPEC))
ROOT = os.path.abspath(os.path.join(SPEC_DIR, os.pardir))

exec(open(os.path.join(SPEC_DIR, "collect_hiddenimports.py"), encoding="utf-8").read())
hiddenimports = collect_hiddenimports(ROOT)

exec(open(os.path.join(SPEC_DIR, "version_info.py"), encoding="utf-8").read())
_version_file = os.path.join(SPEC_DIR, "version_info.txt")
with open(_version_file, "w", encoding="utf-8") as _vf:
    _vf.write(version_info_txt(ROOT))

# rasterio resolves Cython submodules and GDAL/PROJ data dynamically at
# runtime (invisible to static analysis); collect them explicitly.
from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_submodules,
)

hiddenimports += collect_submodules("rasterio")

# pyogrio is geopandas' default file engine and ships compiled extensions
# plus its own GDAL/PROJ data. Static analysis misses pyogrio._geometry
# (loaded while importing the package) and the data folders, so frozen
# runs failed with "to_file requires pyogrio". collect_all brings the
# submodules, the .pyd files and gdal_data/proj_data.
pyogrio_datas, pyogrio_binaries, pyogrio_hidden = collect_all("pyogrio")
hiddenimports += pyogrio_hidden

datas = [
    ("lynceus/ui/styles/app.qss", "lynceus/ui/styles"),
    ("lynceus/ui/mainwindow.ui", "lynceus/ui"),
    ("templates", "templates"),
    ("assets", "assets"),  # brand: SVGs (window/menu icon), splash lockup
    # PNG, icon.ico (frozen windows), wizard PNGs (installer)
    ("licenses", "licenses"),  # LGPL-3.0 text + third-party attributions
    # .py sources: the AST node discovery reads them at runtime
    # (frozen imports resolve from the bundle, sources stay for metadata).
    ("lynceus/nodes", "lynceus/nodes"),
]
datas += collect_data_files("rasterio", subdir="gdal_data")
datas += collect_data_files("rasterio", subdir="proj_data")
datas += pyogrio_datas
# NOTE: locale packs are NOT bundled: English is the base language and
# locales arrive as downloaded extensions (.lxpkg). Only theme packs ship.
if os.path.isdir(os.path.join(ROOT, "extensions", "themes")):
    datas.append(("extensions/themes", "extensions/themes"))

a = Analysis(
    [os.path.join(ROOT, "main.py")],
    pathex=[ROOT],
    binaries=pyogrio_binaries,
    datas=[(os.path.join(ROOT, src), dst) for src, dst in datas],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[os.path.join(SPEC_DIR, "rthook_gdal.py")],
    excludes=["tests", "matplotlib"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="LynceusScan",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(ROOT, "assets", "icon.ico"),
    version=_version_file,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="LynceusScan",
)
