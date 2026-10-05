# LynceusScan — public beta (0.1.0b1)

> **Versión en español:** [README.es.md](README.es.md)

Agnostic visual ETL for geospatial data — LiDAR first.

LynceusScan is a node-based pipeline editor: connect processing nodes on a
canvas and run them over tiled data in parallel, with 2D/3D previews of
every product. It ships with a LiDAR core (load, clean, classify, terrain,
raster metrics, exports) and accepts extensions — extra nodes, language
packs and themes — as single importable files.

- Website & docs: https://danico19827.github.io/LynceusScan-Web/
- Extensions: https://github.com/Danico19827/LynceusScan-Extensions
- Issues: https://github.com/Danico19827/LynceusScan/issues
- Contact: lynceusscan@gmail.com

## Public beta

This is a **public beta**: it runs real LiDAR workflows end to end, but
expect breaking changes — node contracts, session layouts and project
files may still evolve before 1.0. Bug reports, improvement suggestions
and new extensions are all welcome (see Contributing below).

## Designed with AI, supervised throughout

This system was designed with the help of artificial intelligence, and
supervised across several dimensions to guarantee it works well: LiDAR
domain rules (ASPRS/USGS), engine architecture and data correctness,
interface translations (full parity in 4 languages), installer delivery
(signed-off with a smoke test), and a folder-by-folder audit of everything
that ships. The `/docs` of the development repository hold the normative
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

Open a project directly:

```powershell
.venv\Scripts\python main.py path\to\project.lynx
```

## What's included (and what's not)

The core runs in English and ships the LiDAR node set, the processing
engine, 2D/3D viewers, project templates, 22 interface themes and the
extension framework. **Not** included, by design — download them from the
[extensions repository](https://github.com/Danico19827/LynceusScan-Extensions):

- **Language packs** (Spanish, Russian, Italian, Brazilian Portuguese) —
  import the `.lxpkg` with File → Extensions → Import extension (or drag &
  drop the file onto the canvas). The new language appears under Tools →
  Preferences → General.
- **Extra nodes** (forestry metrics, weather queries, ...) — same import
  flow; they land in the node library under their own category.

Bundled **project templates** (`templates/`, File → New From Template):
a canopy-height-model workflow and an SfM multispectral NDVI flight —
both core-only, ready to adapt to your own files.

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
- `lynceus/nodes/` — built-in processing nodes (LiDAR, Raster, Table, Flow).
- `lynceus/processing/` — tiling engine, DAG executor, sessions, provenance.
- `lynceus/ui/` — canvas, inspector, viewers, dialogs.
- `lynceus/plugins/` — extensions (AST discovery, importer, locales, terms).
- `lynceus/tools/` — vendor CLI (`extension_tool`: scaffold + pack).
- `templates/` — bundled project templates.
- `tests/` — stdlib test suite (`python -m unittest discover -s tests`).

## License

GPL-3.0-or-later (see `LICENSE`). Extensions load in-process and inherit
the GPL. Third-party contributions are governed by `CLA.md`. If you use
this software in research, please cite it (see `CITATION.cff`).
