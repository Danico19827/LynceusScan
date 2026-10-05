# Third-party software (INTERNAL build note, not shipped verbatim)

LynceusScan core is GPL-3.0-or-later (`LICENSE`). It links the following
third-party libraries (dynamic linking, unmodified). Their full license
texts ship with the installer under `licenses/` (LGPL text verbatim;
permissive texts are referenced by SPDX with their canonical sources).

| Library | Version (pinned) | License (SPDX) | Used for |
|---|---|---|---|
| Qt for Python (PySide6) | 6.11.2 | LGPL-3.0-only (`licenses/LGPL-3.0-only.txt`) | GUI binding (all of `lynceus/ui/`) |
| numpy | 2.5.2 | BSD-3-Clause | numeric arrays |
| laspy (+lazrs) | 2.7.0 | BSD-2-Clause | LAS/LAZ I/O |
| PyOpenGL | 3.1.10 | BSD-3-Clause (SGI-B-2.0 style) | 3D point-cloud viewer |
| rasterio | 1.5.1 | BSD-3-Clause | GeoTIFF I/O |
| scipy | 1.18.1 | BSD-3-Clause | spatial indexes (SOR/ROR denoise) |
| shapely | 2.1.2 | BSD-3-Clause | vector geometry |
| geopandas | 1.1.4 | BSD-3-Clause | GeoPackage/vector I/O |
| psutil | 7.2.2 | BSD-3-Clause | RAM-adaptive budgets |
| Pillow | 12.3.0 | HPND (MIT-like) | icon rasterization |

Prominent notice per LGPL-3.0 §4a lives in the app itself:
Tools → Preferences → About → "Built with: Qt for Python (PySide6) · LGPL-3.0".
