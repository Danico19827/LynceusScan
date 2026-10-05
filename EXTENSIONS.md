# LynceusScan Extension Authoring Guide (provisional)

> **Versión en español:** [EXTENSIONES.es.md](EXTENSIONES.es.md)

> **Status:** provisional authoring guide. It is complete enough to build
> working extensions with no guesswork, but terse: a polished official
> manual with compatibility standards comes later. When in doubt, the code
> is the contract — every section cites the source file that enforces it.

> **Public beta:** LynceusScan is in public beta, so node contracts,
> session layouts and project files may still evolve before 1.0. Build
> against the contracts in this guide (literal `NODE_*` metadata, typed
> ports, `PROCESSING_SPECS`) and your extension will track the core.
> This system was designed with the help of artificial intelligence and
> supervised across several dimensions (LiDAR domain rules, engine
> architecture, translation parity, installer delivery, ship audit) —
> if a section below disagrees with the cited source file, the file wins
> and deserves a bug report. New extensions are welcome: shape the tool
> around your workflow and share them through the [extensions
> repository](https://github.com/Danico19827/LynceusScan-Extensions) so
> others can use them too.

## 0. What an extension is

LynceusScan processes LiDAR (and related) data as a graph of **nodes**:
boxes on a canvas wired by typed ports. An **extension** adds new
capabilities without touching the core, in one of these kinds:

| Kind | What it is | Unit of distribution |
|---|---|---|
| `node` (standalone) | One `.py` file: a processing node, a file input, or a node with custom ports | `.py` file, or `.lxpkg` bundle |
| `locale` (pack) | A translation catalog enabling one UI language | Folder with `manifest.json`, or `.lxpkg` |
| `theme` (pack) | A 12-color palette restyling the whole app | Folder with `manifest.json`, or `.lxpkg` |
| `font` (future) | Not a pack kind yet. Interface fonts are picked from system families in Preferences → General; packs bringing TTF files will plug into the same selector later (`lynceus/ui/fonts.py`) | — |

**Ground rules (no exceptions):**
- Extensions load **in-process** and are derivative works: they inherit
  **GPL-3.0-or-later**. Mark `license` metadata accordingly (omit it and
  GPL-3.0-or-later applies by default).
- There is **no license machinery**: no `.lic` files, accounts,
  signatures, or payments. The only gate is an optional per-node **EULA**
  (consent text shown once before first use).
- Node files are read by **AST first, executed only when run**. Anything
  the app needs without running your code (ids, ports, schemas, specs)
  **must be a plain literal** at module level — no computed values, no
  function calls, no imports needed to read them.

## 1. What you need

- A LynceusScan **source checkout** to run the authoring CLI
  (`python -m lynceus.tools.extension_tool ...` from the repo root).
  Without it you can still hand-write every file below (they are plain
  text/JSON) and let the app's import dialog check them on entry.
- A text editor. No SDK, no compiler, no registration anywhere.

CLI commands (run from the repo root):

```
python -m lynceus.tools.extension_tool scaffold <dir> --kind node|locale|theme --author "Your Name"
python -m lynceus.tools.extension_tool pack <files...> -o out.lxpkg
python -m lynceus.tools.extension_tool validate <path>
```

## 1b. Guaranteed dependencies (what your code may import)

Inside tile/barrier tasks (workers included, installed app included) you
can rely on the **standard library** plus exactly these pinned packages:

| Import | Package | Use it for |
|---|---|---|
| `numpy` | numpy | arrays, histograms, vectorized math |
| `laspy` | laspy | LAS/LAZ point clouds |
| `rasterio` | rasterio | GeoTIFF read/write, transforms, CRS |
| `scipy` | scipy | KD-trees, filters, stats |
| `shapely` | shapely | geometries |
| `geopandas` | geopandas | vector tables (slow; prefer `sqlite3` for GPKG reads) |
| `PIL` | Pillow | image encode/decode/resize |
| `psutil` | psutil | memory/process introspection |
| `PySide6` | PySide6 | Qt (nodes rarely need it; widgets do) |
| `OpenGL` | PyOpenGL | raw GL (3D viewer internals) |

Anything else is **not guaranteed** (it may exist in your dev venv by
accident via another package, then vanish in the installed app). Either
avoid it, vendor the file next to your node, or fail loudly with a
readable message:

```python
try:
    import cv2
except ImportError:
    raise RuntimeError(
        "Image Filter needs OpenCV, which is not bundled: "
        "pip install opencv-python-headless or vendor cv2 next to this file"
    )
```

## 2. Your first node in 10 minutes (barrier processing node)

A **barrier** node runs once per run (no tiling): read inputs, write
products, return a payload. Copy this skeleton — it is complete and runs:

```python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Jane Doe <jane@example.com>
"""Node - Uppercase CSV (stack example)."""

from lynceus.nodes.ports import PortType

NODE_ID = "acme.uppercase"
NODE_NAME = "Uppercase CSV"
NODE_CATEGORY = "Table"
NODE_AUTHOR = "Jane Doe"
NODE_FOLDER = "acme"
NODE_LICENSE = "GPL-3.0-or-later"

INPUTS = (PortType.TABLE_CSV,)
OUTPUTS = (PortType.TABLE_CSV,)

PROCESSING_SPECS = {
    "barrier_task": "barrier_uppercase",
    "input_ports": ("table_csv",),
    "output_files": {"table_csv": "upper.csv"},
    "output_port": "table_csv",
    "output_globs": ("upper.csv",),
    "config_schema": {
        "column": {
            "type": "str",
            "default": "name",
            "description": "Column to uppercase.",
            "impact": "Only this column is rewritten; the rest passes through.",
            "group": "Options",
        },
    },
}


def get_config_defaults():
    return {"column": "name"}


def barrier_uppercase(ctx):
    import csv
    from pathlib import Path

    from lynceus.nodes._paths import scoped_file
    from lynceus.processing import provenance

    src = Path(ctx.get("table_csv_path", ""))
    if not src.is_file():
        raise RuntimeError("Uppercase CSV needs a table on its input")
    column = ctx.get("column", "name")
    out_path = Path(scoped_file(ctx, "upper.csv"))
    with open(src, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if column in row and row[column]:
            row[column] = row[column].upper()
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    prov_doc = provenance.build_provenance(ctx, "upper.csv")
    provenance.write_sidecar(out_path, prov_doc, {"column": column})
    return {
        "file": str(out_path),
        "kind": "Table",
        "node": NODE_ID,
        "rows": len(rows),
    }
```

Then:

```
python -m lynceus.tools.extension_tool validate acme_uppercase.py
python -m lynceus.tools.extension_tool pack acme_uppercase.py -o uppercase.lxpkg
```

Import `uppercase.lxpkg` via File → Extensions → *Import extension...*
(or drag & drop). It lands in `extensions/nodes/acme/`, appears in the
node library under Table, wires to any table output, and its CSV opens in
the gallery's table viewer.

## 3. Node metadata reference

Module-level literals (AST-read; `lynceus/plugins/registry.py`):

| Constant | Required | Meaning |
|---|---|---|
| `NODE_ID` | **yes** | Unique stable id, convention `<folder>.<node>`. Renaming it orphans saved projects. |
| `NODE_NAME` | **yes** | Visible label (translatable, §9). |
| `NODE_AUTHOR` | **yes** | Attribution. Import refuses files without it. Must be a real name — placeholders (`Test`, `todo`, `xxx`, `your name`…) fail review and `validate` warns about them. |
| `NODE_CATEGORY` | no | Library top level (`"LiDAR"`, `"Raster"`, `"Table"`, `"Flow"`, or yours). |
| `NODE_SUBCATEGORY` | no | Second library level. |
| `NODE_DESCRIPTION` | no | HTML shown in the library (translatable, keep `<b>/<i>/<code>` markup). |
| `NODE_FOLDER` | no | Install subfolder; defaults to the `NODE_ID` prefix. |
| `NODE_LICENSE` | no | SPDX id; omitted ⇒ `GPL-3.0-or-later`. |
| `NODE_EULA` | no | Consent text shown once before first use (stays English). |
| `NODE_DISCLAIMER` | no | Short text stamped into every product's provenance (§6). Stays English. |
| `NODE_HOMEPAGE` / `NODE_REPOSITORY` / `NODE_ISSUES` / `NODE_DOCUMENTATION` / `NODE_DONATE` / `NODE_COPYRIGHT` / `NODE_CONTRIBUTORS` / `NODE_TAGS` / `NODE_CHANGELOG` | no | Soft trust/attribution metadata shown in the Extensions dialog. |
| `NODE_TRANSLATIONS` | no | `{"lang": {"source": "target"}}` fallback translations (§9). |
| `INPUTS` / `OUTPUTS` | — | Tuples of `PortType`, custom-id strings, `(PortType, "label")` or `PortDef(...)` (§4). `INPUTS = ()` means a source. |
| `PORT_TYPE_DEFS` | no | Custom port types (§4). |
| `PROCESSING_SPECS` | — | Dict below; only literal values are UI-visible. |
| `get_config_defaults()` | no | Returns `{key: default}`; engine defaults come from the schema itself. |

## 4. Ports

Builtin port ids live in the `PortType` enum
(`lynceus/nodes/ports.py`): `point_cloud`, `tiles`, `dtm_tile`,
`dsm_tile`, `dtm_mosaic`, `dsm_mosaic`, `chm_mosaic`, `grid_metrics`,
`grid_vegetation`, `canopy_penetration`, `coverage_raster`,
`intensity_ortho`, `strata_raster`, `vector`, `table_csv`, `table_gpkg`,
`table`, `raster` (plus `classified_point_cloud` kept for compatibility).

Declaring an end:

```python
from lynceus.nodes.ports import PortDef, PortType

INPUTS = (
    PortType.CHM_MOSAIC,                            # required input
    PortDef(PortType.POINT_CLOUD, required=False),  # optional: silence if loose
    PortDef(PortType.TABLE, name="Table A"),        # labeled (translatable)
    PortDef(PortType.TABLE, name="Table B", group="tables"),  # exclusive group
)
```

- `required=False` makes the port optional (a missing cable is silence,
  never an error). Required inputs block the run with a readable message.
- `group="..."` declares mutual exclusion: only one port of the group
  may be wired at a time (the canvas refuses the second cable).
- Two inputs of the **same** type arrive positionally as
  `ctx["in0_path"]`, `ctx["in1_path"]`, ... A single input of a type
  arrives as `ctx["<port_type>_path"]` (e.g. `ctx["table_csv_path"]`).
  Custom ids keep their dots verbatim: port `demo.image` arrives as
  `ctx["demo.image_path"]`. When several providers could match, the first
  one wins — read the key, with `in0_path` as the positional fallback.

Custom port types (no core changes needed):

```python
PORT_TYPE_DEFS = [
    {"id": "acme.image", "display_name": "Image", "color": "#7dd3a8",
     "compatible": ("acme.image", "raster"), "viewer_kind": "raster"},
]
```

- `id`: unique string. `display_name`: English label (translatable key,
  §9). `color`: `#rrggbb` cable/plug color. `compatible`: tuple of input
  ids this output may feed (builtin ids or ids from the same file).
- `viewer_kind` (optional): reuse an existing viewer — `raster` (GeoTIFFs
  with CRS), `image` (plain PNG/JPG/BMP/GIF/WEBP, no georeference),
  `vector`, `table_csv`, `point_cloud`, `json`. There is **no way to add
  a new viewer kind** from an extension today; pick the closest existing
  one (a PNG with `viewer_kind: "raster"` previews as a raster).
- Unknown ids degrade gracefully (grey `#888888`, no connections).
- `validate` checks entry shape, `#rrggbb` form, and warns on
  `compatible` ids that match neither builtins nor the same file (typos).
- Declare each custom id **once** across your files: registration is
  last-wins in discovery order, so a *conflicting* duplicate silently
  rewires cables (`validate` fails it across packs and bundles, and warns
  on identical repeats). Producer + consumer sharing one type is the
  normal case — duplicate only the agreement, never divergent copies.
  Validating one file alone warns about ids declared in its siblings;
  that warning disappears when you validate the folder, pack or bundle
  together.

## 5. Processing contracts

### 5a. Barrier tasks (start here — covers inputs, combiners, exports)

Signature: `def barrier_<name>(ctx: dict) -> dict`. Runs **once per run**,
after all tile work it depends on. Use it for anything that is not
per-tile parallel work.

`ctx` keys you can rely on:

| Key | Meaning |
|---|---|
| `session_dir` | Absolute session artifacts dir. **Write only under it** (via `scoped_file`, never `session/<name>` hardcoded). |
| `node_iid` | This instance's id (branch scoping, per-instance subfolders). |
| `module_id` | Your `NODE_ID`. |
| `file_path` | File picked in a file-input widget (§7), or `""`. |
| `<port_type>_path` | Provider output for a singly-typed input. |
| `in0_path`, `in1_path`, ... | Provider outputs for repeated same-type inputs, in declaration order. |
| `<config_key>` | Your `config_schema` values, flat (schema defaults pre-applied). |
| `tile_scope` | Branch scope string; pass through `scoped_file`, never into fingerprints. |

Return payload (gallery + reuse contract):

```python
{
    "file": str(out_path),   # MUST exist: advertised-but-missing files
                             # are a hard error, never a silent ghost.
    "kind": "Table",         # human kind label (gallery grouping).
    "node": NODE_ID,
    "warnings": [...],       # optional; shown amber, stay English.
    "display_name": ...,     # optional leaf label; defaults to basename.
}
```

- Extra `file_2`, `csv`, ... string keys pointing at existing files become
  extra gallery products. `node`/`kind`/`source_file`/`warnings`/`count`
  and friends are metadata, never products.
- Errors: raise `RuntimeError("...")` with a plain-English message (the UI
  shows it on the node). Anything the user can tune around goes to
  `warnings` instead — warnings **stay in English by design**.
- Determinism is a correctness property: same inputs + same config must
  produce byte-identical outputs (no timestamps, no randomness, sorted
  structures). Reuse, provenance and the gallery all assume it. (Compressed
  formats may differ byte-wise across OS/zlib versions — reuse keys on
  fingerprints, never on byte comparison — but keep your own outputs
  deterministic all the same.)

Where to write (this rule is load-bearing — getting it wrong breaks
downstream silently):

- **Processing/combiner barrier** (mosaic-style shared final): declare
  `output_files: {"<port>": "<basename>"}` and write exactly that
  basename via `scoped_file(ctx, "<basename>")`. Downstream resolves
  `session/<basename>` (or the branch scope dir).
- **File-input barrier** (per-instance product, §7): declare
  `session_file: "<basename>"` (fixed name, convert formats when
  needed) and write exactly
  `Path(ctx["session_dir"]) / ctx["node_iid"] / "<basename>"`.
  Downstream resolves `session/<src_iid>/<basename>`, so two inputs of
  the same kind coexist. `scoped_file` is **wrong here**: it writes the
  session root while the engine looks inside the instance folder, and
  the consumer receives a dangling path.
- Never hardcode `session/<name>` paths by hand in either case; never
  write outside `session_dir` at all.

### 5b. Tile tasks (per-tile parallel work)

Signature: `def tile_<name>(tile: dict, ctx: dict) -> dict`. Only for work
that splits spatially (point clouds, rasters). If your algorithm reads
whole files and writes one product, use a barrier (§5a) instead — tiling
is LiDAR-file-based; custom tiled sources must resolve to point-cloud
files the engine can tile.

Declare `"tile_task": "tile_<name>"` in `PROCESSING_SPECS` and read tile
inputs the same way (`ctx` paths). Partial outputs merge in your
`barrier_task` or a `consolidator`.

### 5c. `config_schema` (Inspector form)

```python
"res_m": {
    "type": "float",        # str | int | float | bool | str+options
    "default": 0.5,         # must match the type; engine default source
    "minimum": 0.05, "maximum": 100.0,   # numeric only (int/float)
    "options": [("Label", "value"), ...],# or "enum": [...]; label/data pairs
    "description": "...",   # what it does (translatable, §9)
    "impact": "...",        # when to raise/lower + side effects (translatable)
    "group": "Grid",        # free-form section header (translatable)
    "advanced": True,       # hides the inline impact line (tooltip stays)
},
```

- Types: `str` (also the fallback for missing/unknown types), `int`
  (±2³¹), `float` (3 decimals in UI — never rely on finer user-edited
  precision), `bool`, select via `options`/`enum` (data values must be
  `str`/`int`/`float`).
- Labels render as `key.replace("_", " ")`; `description` + `impact`
  become tooltips. Groups are free-form; reuse sibling names (`Grid`,
  `Output`, `Advanced`, `Behavior`...).
- Constraints: values must be **JSON-serializable** (`str/int/float/bool`
  and plain lists) — fingerprints `json.dumps` the config with no
  fallback. The key `strategy` is reserved for strategy families (§8).

### 5d. Reuse (why `output_files` / `output_globs` matter)

```python
"output_files": {"table_csv": "upper.csv"},  # {out_port_id: basename}
"output_port": "table_csv",
"output_globs": ("upper.csv",),
```

These declare the session layout so re-runs with identical fingerprints
hardlink your products instead of recomputing. Rules: only announce files
you actually write; keep basenames stable across configs; never write
outside the session dir.

Variable extensions (`.jpg` today, `.png` tomorrow): do NOT preserve
the source extension. Normalize every run to one fixed `session_file`
(convert the format when needed) and declare literal globs covering the
product **and** its sidecar, e.g. `"session_file": "image.png"` plus
`output_globs = ("image.png", "image.meta.json")`. Rationale, verified
against the engine:
- Without `output_files`/`session_file`, downstream nodes receive **no
  path at all** (the injector skips providers it cannot locate) —
  `validate` warns about this.
- Reuse matches globs against real basenames: a glob that matches
  nothing (e.g. `input_image.*` for a product saved as `photo.jpg`)
  silently disables reuse for that node, forever recomputing.
- Sidecars travel only when a glob matches their basename
  (`image.*` covers `image.meta.json`; otherwise list it explicitly).

What the fingerprint covers (so you can reason about reuse): the module
id string, the **bytes of the module source file** (editing your `.py` —
or the active strategy variant file — always recomputes), your instance
config as JSON (sorted keys), upstream fingerprints, the batch scope,
and the operator identity when set. It does **not** include sidecars:
`generated_at` timestamps legitimately differ between runs, so compare
product bytes for determinism, never sidecars. Keep basenames stable and
outputs deterministic.

### 5e. Config lifecycle (renames, removals, versions)

- Saved projects store your config as plain `{key: value}`. On restore,
  unknown keys are **ignored by the UI** (only current schema keys
  render) but travel harmlessly; a renamed key changes the fingerprint,
  so the node safely recomputes instead of reusing stale products.
- Rules: never reuse a removed key name with a different meaning or
  type; keep defaults type-stable; `NODE_VERSION` is reserved for future
  migrations (declare it today if you plan breaking changes, the
  machinery will honor it later).
- Strategy families: the `strategy` value travels inside the config, so
  switching product/method always recomputes — variants never share
  reuse across keys.

## 6. Provenance and styles (do the standard thing)

```python
from lynceus.processing import provenance

prov_doc = provenance.build_provenance(ctx, "upper.csv")
provenance.write_sidecar(out_path, prov_doc, {"column": column})
```

- `build_provenance` tolerates missing `ctx` keys (falls back to system
  constants; operator only when configured). `write_sidecar` merges into
  `<stem>.meta.json` — the product path with its extension **replaced**
  (`Path("image.png").with_suffix(".meta.json")` → `image.meta.json`,
  never `image.png.meta.json`) — without clobbering node keys.
- GeoTIFF/GPKG/LAZ get embedded tags automatically **only** on the core
  validation branches; custom kinds keep the sidecar (enough for gallery
  + QGIS sidecar reading).
- `NODE_DISCLAIMER` (if you set one) is embedded as `node_disclaimer` in
  every product: keep it short, English, factual.
- QGIS styles: `PROCESSING_SPECS["qml"] = {"upper.csv": ...}` — raster
  ramp id (`"dtm"`, `"chm"`, `"ndvi"`, `"generic"`...), vector field name,
  or explicit `{"field", "classes": [...]}` / `{"name", "stops": [...]}`;
  `{"self_styled": True}` when you write the `.qml` yourself. No valid
  data ⇒ silently skipped, never an error.

## 7. File-input nodes by convention (no core edits)

Any node with `input_file_key` in `PROCESSING_SPECS` automatically gets
the Browse + file-label item (dialog filters from `file_filters`; the
picked path always arrives as `ctx["file_path"]`). The barrier reuses the
shared framework with **its own** target dict — no registry rows needed.
`barrier_input_file` writes `session/<node_iid>/<session_file>` internally,
so using it complies with the §5a layout rule automatically; only
hand-rolled barriers must replicate that layout themselves:

```python
PROCESSING_SPECS = {
    "barrier_task": "barrier_input_txt",
    "input_file_key": "file_path",
    "file_filters": "Text files (*.txt)",
    "output_port": "table_csv",
    "session_file": "words.csv",
    "kind_group": "none",     # no core validation branch: copy faithfully
    "kind_label": "Text",
}

def barrier_input_txt(ctx):
    from lynceus.nodes._product_input import barrier_input_file
    return barrier_input_file(ctx, {
        "node_id": NODE_ID, "node_name": NODE_NAME,
        "output_port": "table_csv", "session_file": "words.csv",
        "kind_label": "Text", "kind_group": "none",
    })
```

- `kind_group` picks validation: `"raster"` (single-band notice),
  `"multispectral"` (multiband check), `"table_gpkg"` (SQLite),
  anything else (CSV branch), `"none"`/`"custom"`/`""` (no opinions —
  for product kinds the core has no branch for).
- Without `input_file_key`, a barrier runs headless on
  `ctx["file_path"]` (generic widget, no Browse). Strategy families with
  file inputs keep explicit widget registration (core edit) — plain nodes
  never need it.

## 8. Strategy families (variants)

For one node with switchable products/methods (like Input Raster's
`strategy=dtm|dsm|chm` or Classify Ground's `pmf|csf|smrf`):

- **Base**: normal node with `NODE_ID`, `INPUTS = ()`, `OUTPUTS = ()`,
  and a `strategy` field in `config_schema` (default `""`) plus
  `get_config_defaults() -> {"strategy": ""}`. With no strategy selected
  the node is **inert** (0 ports) and wired runs block with
  *"Select a product strategy in node(s): …"*.
- **Variant**: a module with **no `NODE_ID`** declaring `VARIANT_OF`
  (= base id), `VARIANT_KEY`, `VARIANT_LABEL`, plus its own
  `INPUTS`/`OUTPUTS`/`PROCESSING_SPECS`/task functions. Variant `requires`/
  `provides`/ports replace the base's; only `config_schema` merges (the
  base keeps `strategy` for the dropdown). The `strategy` value travels in
  the fingerprint, so variants never share reuse.
- Limitation (honest): the Product/Method/Merge selector widgets are
  explicitly registered in the core — a third-party family renders with
  the generic item unless the core maps it. Standalone inputs (§7) need
  no registration at all.

## 9. Translations (making your node multilingual)

The app translates by **exact English match**; missing keys fall back to
English, so partial translations ship safely. Only languages enabled by
an installed `locale` pack are offered — your translations can never
half-translate the app alone.

What is translatable in your file (all read by AST):

- `NODE_NAME`, `NODE_CATEGORY`, `NODE_SUBCATEGORY`, `NODE_DESCRIPTION`
  (keep `\n` and `<b>/<i>/<code>` markup intact in translations).
- `config_schema`: labels (`param_name` with `_` → space), `description`,
  `impact`, `group`, and `options` labels (data values stay).
- Port labels: explicit `PortDef(..., name="Table A")` or
  `(PortType, "label")` tuples — and `PORT_TYPE_DEFS` `display_name`s.
- `VARIANT_LABEL`s.

Provide them via a sidecar next to your file
(`<node>.i18n.json`, rides along in `.lxpkg` automatically, wins over
everything):

```json
{"es": {"Uppercase CSV": "CSV en mayúsculas", "Table A": "Tabla A"}}
```

or via a `NODE_TRANSLATIONS = {"es": {...}}` dict in the `.py`
(precedence: pack catalog < `NODE_TRANSLATIONS` < sidecar).

Stays in English by design (never translate): `NODE_ID`, warnings,
`NODE_EULA`/`NODE_DISCLAIMER`, and technical tokens (`CHM`, `NODATA`,
`LAI`, units, `Ctrl+…`, `JSON`/`CSV`…). Translate meaning, not words:
impact texts must explain what to do and what breaks, in natural
language — a word-by-word translation users can't act on is worse than
English.

## 10. Locale packs (new languages)

```
my-lang/
  manifest.json   {"schema_version": 2, "id": "fr", "version": "0.1.0",
                   "kind": "locale", "display_name": "...",
                   "author": "...", "payload": {"catalog": "fr.json"}}
  fr.json         {"fr": {"Source English string": "Translation", ...}}
```

- The catalog key is the language code; the **key set should mirror the
  reference catalog** (copy one and translate values — `validate` warns on
  empty values and placeholder (`{x}`) mismatches).
- Importing the pack only **enables** the language (Preferences →
  General); it never activates it alone. English is always the fallback.

## 11. Theme packs

```
my-theme/
  manifest.json   {... "kind": "theme", "payload": {"palette": "theme.json"}}
  theme.json      {"background": "#...", ...}   // 12 fixed roles below
```

| Role | Default | Used for |
|---|---|---|
| `background` | `#0b0f1a` | Window, canvas, splash |
| `surface` | `#1c2333` | Panels, menus, node fill |
| `border` | `#2f3a4f` | Borders, node border |
| `border_strong` | `#3a465c` | Hovers, selection, tree selection |
| `text` | `#e6e9f0` | Primary text, titles, splash lockup |
| `text_body` | `#aeb7c6` | Body/label text |
| `muted` | `#8a93a6` | Secondary text, wires, categories |
| `accent` | `#7d9fd4` | Selection, focus, progress, links |
| `danger` | `#e08a8a` | Close hover, errors |
| `amber` | `#c9a86a` | Modified values, warnings |
| `canvas_bg` | `#0b0f1a` | Node canvas background |
| `wire` | `#8a93a6` | Connection wires |

- Any subset is valid (missing roles fall back to defaults); unknown
  roles and malformed `#rrggbb` are ignored with a warning and never
  break startup. Semantic colors (status greens/reds, notice chips, 3D
  viewport) stay fixed by design.
- Importing enables the theme in Preferences → General (applies live);
  it never activates it alone.

## 12. Fonts (setting today, packs tomorrow)

Interface fonts are picked from **system families** in Preferences →
General (default = system default); every `QFont()` in widgets and
painted canvas items follows the application font, so one setting
re-skins all text. Only smoothly-scalable Latin-covering faces are
offered (bitmap/symbol/display faces fail or break layouts). Never
hardcode a family in a node or widget. Font packs (`kind=font` with TTF
files) will register families into the same selector later — write
nothing font-specific in extensions today.

## 13. Trust, license, experimental nodes

- `NODE_LICENSE`: SPDX id, omitted ⇒ `GPL-3.0-or-later`. Extensions load
  in-process (derivative work) and inherit GPL — the law is the limit,
  not the code.
- `NODE_EULA`: consent text shown **once** before first use (stays
  English). Use it for anything the user must actively accept.
- `NODE_DISCLAIMER`: short text **stamped into every product** (stays
  English). Recommended shape for unverified/experimental work
  (AI-assisted algorithms, unchecked precision):
  experimental status + professional-validation requirement + user
  responsibility, e.g. the forestry nodes' gate. Never promise precision
  you have not measured.

## 14. Tooling reference

```
# New pack skeleton (node|locale|theme; --author required)
python -m lynceus.tools.extension_tool scaffold my-pack --kind node --author "Jane Doe"
# Distribute: one .lxpkg per node (.py + sidecar ride together)
python -m lynceus.tools.extension_tool pack mynode.py -o mynode.lxpkg
# Check without installing (errors fail, warnings advise)
python -m lynceus.tools.extension_tool validate mynode.py
python -m lynceus.tools.extension_tool validate mynode.lxpkg
python -m lynceus.tools.extension_tool validate my-pack/
```

In-app: File → Extensions → *Import extension...* (opens on Package
filter), *Import pack folder...*, or drag & drop files onto the dialog.
Import one file at a time: with two `.py` files (e.g. an input plus its
processing node), import the first, then the second — each lands next to
its siblings by `NODE_FOLDER` and both register on the next discovery.
Shared `PORT_TYPE_DEFS` ids across your files are fine as long as they
are identical (declare once if you can; conflicts fail `validate`).
A pack `.lxpkg` (with `manifest.json`) of kind `theme`/`locale`
extracts to `extensions/themes/<id>/` / `extensions/locales/<id>/` —
the same layout as importing the folder; `node`-kind packs are still
copied as-is (legacy). Re-importing an extracted pack without `force`
is refused; `force=True` re-extracts over it.
Import is idempotent; overwriting different content is refused with an
error unless imported with `force=True` via the API. Per-row *Disable*
(stays listed, out of library/pipeline), *Remove* (deletes files, asks
first), *Open folder*. Disabled packs never register — disabling the
active language/theme falls back to English / default.

### 14b. Without the CLI (no source checkout)

Every artifact above is hand-writable; the CLI only speeds things up:

- **Node**: copy the §2 skeleton into a `.py` file. No manifest needed:
  the import dialog accepts a bare `.py` directly (it checks `NODE_ID` +
  `NODE_AUTHOR` on entry and reports errors in a message box).
- **`.lxpkg` is a ZIP**: pack = files at the archive root. A bundle has
  no `manifest.json` (node `.py` files + `<node>.i18n.json` sidecars); a
  full pack has `manifest.json` + payload at root. Any zip tool works —
  rename `.zip` to `.lxpkg`. Unsafe entries (`..`, absolute paths) are
  rejected on import.
- **Locale/theme**: hand-write the JSON shapes from §10/§11; import the
  folder via *Import pack folder...* or zip it as `.lxpkg`.
- **`validate` unavailable**: rely on the import dialog messages (the
  same checks run on entry) plus the checklist in §15.

## 15. Testing checklist (before you distribute)

1. `validate` on the `.py` **and** on the finished `.lxpkg`: zero errors.
2. Import into the app, wire the node, run: products appear in the
   gallery with the right viewer; re-run reuses (fingerprint stable).
3. Disable/re-enable from the dialog; remove and re-import cleanly.
4. If multilingual: sidecar keys match the English sources exactly
   (placeholders intact).
5. Determinism: run twice, `diff` the products (bytes must match;
   sidecars may differ in `generated_at`, products must not).
6. Verify the real on-disk path of every product after the first run
   (it must be where §5a says downstream will look for it).
7. Run twice and compare sidecar hashes: identical except timestamps.

## 16. Distribution

- One `.lxpkg` per node (`.py` + `<node>.i18n.json` sidecar, no bundle),
  or a pack folder with `manifest.json` for locale/theme packs.
- Version in `manifest.json` (`version` + `compatibility` with tested app
  versions); `author` required, `homepage`/`repository`/`issues`/
  `documentation`/`donate`/`changelog`/`tags`/`contributors`/`copyright`
  optional but recommended. `NODE_ID`s are forever: renaming orphans
  saved projects and voids reuse.
- Share via Releases/drive/web: users import with *Import extension...*.

## 17. Troubleshooting

| Symptom | Cause → fix |
|---|---|
| Import says "no NODE_ID / no NODE_AUTHOR" | Literals missing or computed — declare both as plain strings. |
| Node never appears in the library | Disabled pack; `INPUTS`/`OUTPUTS` unparsable; file without `NODE_ID` is ignored (helpers). Run `validate`. |
| Strategy node inert (0 ports), run blocked | No strategy selected: pick Product/Method/Merge on the node. |
| Barrier ghost error (`file` not written) | The barrier announced a product it never wrote — write every announced file, always. |
| Ports won't connect | `compatible` mismatch (check ids both sides) or destination `group` conflict (one cable per exclusive group). |
| Product has no preview | Unknown file extension (gallery routes: `.tif` raster, images `.png/.jpg/.jpeg/.bmp/.gif/.webp`, `.gpkg` vector, `.csv/.txt` table, `.las/.laz` cloud, `.json` tree) or `viewer_kind` missing on a custom port. |
| Language/theme not offered | No enabling pack installed (translations alone never enable); disabled pack. |
| Re-run recomputes everything | `output_files`/`output_globs` wrong or unstable basenames; non-JSON config; nondeterministic bytes. |
| UI text stuck in English | Key mismatch with the English source (exact match incl. `\n`/markup) or language without enabling pack. |
| `validate` warns `input_file_key` without `barrier_task` | Browse would show but nothing runs — wire the task or drop the key. |
| Downstream reads empty/missing paths | Producer omitted `output_files`/`session_file`: no path is injected (warns in `validate`). Normalize to a fixed `session_file`. |
| Downstream empty ALTHOUGH `session_file` is declared | The producer wrote somewhere else: inputs must write exactly `session/<own_iid>/<session_file>` (a bare `scoped_file` lands in the session root, where nobody looks). Check on disk that the file exists at the injected path. |
| `validate` fails on conflicting port ids | Two files declare the same port id differently — unify to one definition; last-wins would silently rewire cables. |

## 18. Execution model (design heavy nodes around this)

- **Workers, not the app**: tile and barrier tasks run in separate
  `spawn` worker processes (pool sized to CPU count), never in the GUI
  process. Keep tasks self-contained functions of `(tile, ctx)` /
  `(ctx)`: **everything they touch must be picklable** (plain
  dicts/lists/strings/numbers/paths-as-strings — no QObjects, no open
  file handles, no lambdas, no module-level mutable caches shared with
  the UI).
- **Memory envelope**: the engine throttles dispatch predictively
  (~70% of RAM over in-flight tiles) and reactively (RSS sampling
  shrinking workers); the tiler pages per bucket under a byte budget.
  Design for it anyway: stream files (never `read()` a whole raster
  bigger than RAM — windowed reads arrive post-MVP; today, oversized
  rasters fail the node with a legible error instead of swapping the
  machine), prefer generators over materialized lists, close files
  explicitly.
- **Failures**: any exception marks the task failed and cascades to
  dependents; the message reaches the node in the canvas. `RuntimeError`
  with a plain message is the convention; `MemoryError`/`KeyError`/
  native crashes behave the same way (failed, never zombi) but test
  them — a segfault kills only its worker.
- **Cancel/pause**: the engine polls cancel/pause flags between tasks;
  long barriers should checkpoint to disk periodically so a cancelled
  run resumes cheaply on re-run (reuse keys on finished files).
- **Discovery order**: `extensions/` walks sorted and recursively,
  deterministically — last registration wins for duplicate port ids
  (hence §4's single-definition rule).
- **Large raster policy (v1)**: tiling covers point clouds; raster
  barriers that cannot fit RAM must fail fast with a legible error
  naming the shortfall. Windowed processing is roadmap, not v1.

## 19. Sandbox and capabilities (the honest policy)

- There is **no sandbox**: extensions load in-process under GPL
  (derivative work) with the same OS privileges as the app — files,
  network, subprocesses. Design your node accordingly and say what it
  does: if it touches the network, spawns processes, or reads outside
  the session dir + picked files, state it in `NODE_DESCRIPTION` and
  gate anything irreversible behind the `NODE_EULA`.
- Never write outside `session_dir` except the picked-source read;
  never phone home silently; never block the worker on interactive
  input. Users grant trust per pack/node at import + EULA time — keep
  it auditable: small files, literal metadata, no obfuscation.
