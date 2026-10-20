# LynceusScan — public beta (0.1.0b5)

> **Versión en español:** [README.es.md](README.es.md)

Visual workflow tool for geospatial data — LiDAR first.

LynceusScan is a visual workflow editor: connect processing steps on a
canvas and run them in parallel over a survey split into grids, with 2D/3D
previews of every product. It ships with a LiDAR core (load, clean,
classify, terrain, metrics, exports) and accepts extensions — extra nodes,
language packs and themes — as single importable files.

- Website & docs: https://danico19827.github.io/LynceusScan-Web/
- Extensions: https://github.com/Danico19827/LynceusScan-Extensions
- Issues: https://github.com/Danico19827/LynceusScan/issues
- Contact: lynceusscan@gmail.com

## Public beta

This is a **public beta**: it runs real LiDAR workflows end to end, but
expect breaking changes — node contracts, session layouts and project
files may still evolve before 1.0. Bug reports, improvement suggestions
and new extensions are all welcome (see Contributing below).

## What's new in Beta 5

Beta 5 is the scale-and-trust release: the engine now finishes professional
datasets end to end on ordinary machines, and when something cannot fit it
says so instead of silently lowering quality.

- **Large datasets, same results** — mosaics, terrain derivatives and table
  merges are computed in pieces that fit your memory, with the same result
  as doing everything at once.
- **Seam-free mosaics** — DTM, DSM, intensity, canopy penetration and
  flightline coverage keep the whole border cell shared by neighboring
  grids; results match a single global pass at 0.3/0.5/1 m, with no
  one-pixel lines.
- **Runs that survive real data** — sparse neighboring cells merge into
  larger files (up to ~19× fewer files and ~4× faster grid preparation in
  our billion-point benchmark), huge sessions split into independent
  segments, and virtual mosaics join segments without duplicating data.
- **Honest failures** — the reason for every failed node stays in the
  session, and a persistent report lists the failed nodes, their cause and
  the session folder when a run ends with errors.
- **Preflight advisor** — estimates RAM from the LAS/LAZ headers before
  running and offers one-click fixes (keep intermediate files for this run,
  more memory, merge sparse cells, split into segments); the output folder
  is configurable.
- **File queues** — queue several files on any input and run them
  unattended, one clean session per file, with an OK/failed summary at the
  end.
- **Command line (CLI)** — run projects from a script, reproducibly, with a
  machine-readable report and exit codes (`run`, `validate`, `nodes list`
  included).
- **New nodes** — Merge Rasters (DTM/DSM/CHM editions, one shared legend)
  and Filter by QA Flags (remove or keep overlap and synthetic returns).
- **Canvas operations** — multi-select, copy/paste/duplicate groups of
  nodes, undo/redo and batch "Assign files...".
- **Interface** — searchable node library, a rebuilt extensions panel,
  lightweight one-line runs, a product gallery that fills while the run
  progresses, localized dialog buttons, a configurable output folder and a
  refreshed look (floating toolbar, colored node headers).
- **Viewer upgrades** — rasters use the same color ramp and range as QGIS
  (with a gradient legend), external NODATA is respected, zoom is smoother,
  and 3D points respect depth (the reference grid no longer draws over the
  cloud).
- **Templates that travel** — opening a template copied from another
  machine keeps the files that exist and, once, lists the ones to re-pick.

## Designed with AI, supervised throughout

This system was designed with the help of artificial intelligence, and
supervised across several dimensions to guarantee it works well: LiDAR
domain rules (ASPRS/USGS), engine architecture and data correctness,
interface translations (full parity in 9 languages), installer delivery
(verified by launching the packaged app), and a folder-by-folder audit of
everything that ships. The `/docs` of the development repository hold the normative
domain report; the code is the contract wherever this README stays terse.

## Install

**Windows installer** (no terminal required): download the setup from
[Releases](https://github.com/Danico19827/LynceusScan/releases). It
installs per-user (no admin rights), associates `.lynx` project files,
and keeps your data on uninstall by default. Beta builds are unsigned,
so Windows SmartScreen will ask for confirmation on first run.

**Run from source** — requirements: Windows 10/11, Python 3.13.

```powershell
git clone https://github.com/Danico19827/LynceusScan.git
cd LynceusScan
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python main.py
```

Open a project directly, or run it headless:

```powershell
.venv\Scripts\python main.py path\to\project.lynx
.venv\Scripts\python -m lynceus.cli run path\to\project.lynx   # headless
```

## What's included (and what's not)

The core runs in English and ships the LiDAR node set, the processing
engine, 2D/3D viewers, project templates, 22 interface themes and the
extension framework. **Not** included, by design — download them from the
[extensions repository](https://github.com/Danico19827/LynceusScan-Extensions):

- **Language packs** (Spanish, Russian, Italian, Brazilian Portuguese, Simplified Chinese, Hindi, Arabic, French, Bengali) —
  import the `.lxpkg` with File → Extensions → Import extension (or drag &
  drop the file onto the canvas). The new language appears under Tools →
  Preferences → General.
- **Extra nodes** — five experimental forestry nodes (area-based, canopy
  gaps, height strata, treetops, tree crowns) and Open-Meteo weather
  queries; same import flow, landing in the node library under their own
  category.

Bundled **project templates** (`templates/`, File → New From Template):
eight ready-to-use workflows in Spanish — seven core (file inspection,
cleaning, terrain, vegetation indices, segmented large datasets, two-CHM
change comparison and flight QA) plus a forestry inventory chain that
requires the forestry extension packs.

## Scope and limits (read this before trusting outputs)

No silent magic: every algorithm documents what it can and cannot do.

- **Classify Ground** ships five published methods, each with a known
  scope. PMF keeps objects wider than its max window by design; SMRF is
  flat-to-mild terrain by design and strips steep relief; CSF needs its
  iterations to cover the tile's relief (defaults cover 50 m) and has no
  slope post-processing; MCC is best under dense canopy without ground
  returns; ATIN densifies an adaptive TIN from seed minima and follows
  mixed relief (its seed spacing must exceed your largest building).
  Outside its scope a method degrades gracefully and warns — it never
  pretends certainty.
- **Validation is fixture-based**: analytic synthetic scenes with known
  answers plus a Type I/II benchmark, all in the open test suite. That
  proves fidelity to the published papers, not accuracy on your survey.
  Field accuracy is the operator's QA task — check against control
  points before operational use, exactly as the ASPRS/USGS standards
  require.

## Make it yours: write an extension

LynceusScan is built to be customized: a new processing node is a single
`.py` file, a language or theme pack is a folder with a `manifest.json`.
The [authoring guide](EXTENSIONS.md) ([en español](EXTENSIONES.es.md))
takes you from zero to an installable `.lxpkg` with no guesswork — no
SDK, no compiler, no registration. If you shape the tool around your own
workflow, consider sharing the extension so others can use it too.

## Contributing

- **Bug reports**: open an issue with steps to reproduce, ideally with a
  minimal `.lynx` project.
- **Improvement suggestions**: open an issue describing the workflow the
  change would unlock.
- **New extensions**: build it with the authoring guide and share it via
  the extensions repository.
- Code contributions are governed by `CLA.md`.

## Layout

- `main.py` — application entry point.
- `lynceus/cli.py` — command-line interface (`run`, `validate`, `nodes list`).
- `lynceus/nodes/` — built-in processing nodes (LiDAR, Raster, Table, Flow).
- `lynceus/processing/` — grid engine, execution engine, sessions, provenance.
- `lynceus/ui/` — canvas, inspector, viewers, dialogs.
- `lynceus/plugins/` — extensions (discovery, importer, locales, terms).
- `lynceus/tools/` — vendor CLI (`extension_tool`: scaffold + pack).
- `templates/` — bundled project templates.
- `tests/` — stdlib test suite (`python -m unittest discover -s tests`).

## License

GPL-3.0-or-later (see `LICENSE`). Extensions load in-process and inherit
the GPL. Third-party contributions are governed by `CLA.md`. If you use
this software in research, please cite it (see `CITATION.cff`).
