# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Headless command line over the Qt-free pipeline domain.

``python -m lynceus.cli run proyecto.lynx`` executes the same
``PipelineController`` run the GUI canvas would start: same sessions,
fingerprints, reuse and products — only the Qt widgets are missing.
Progress goes to stderr; ``--json`` prints a machine-readable report on
stdout (the substrate a future MCP server can reuse).

Exit codes: 0 ran clean; 1 usage/project/validation errors; 2 blocked
(terms, EULAs, missing strategy/inputs/sources); 3 finished with failures,
cancelled, or errored. Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from pathlib import Path

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_BLOCKED = 2
EXIT_FAILED = 3

LOAD_MODULE = "lynceus.nodes.lidar.source.load_las_laz"

_TERMS_HINT = (
    "Terms & Conditions are not accepted on this machine. "
    "Open LynceusScan once and accept them in the first-launch dialog, "
    "then retry."
)
_EULA_HINT = (
    "Extension EULA(s) pending: {names}. Open LynceusScan once and accept "
    "them in File → Extensions, then retry."
)


def _err(text: str) -> None:
    print(f"lynceus: error: {text}", file=sys.stderr)


def _say(text: str) -> None:
    print(text, file=sys.stderr)


def _bootstrap():
    """Discover nodes/extensions and wire the shared consent store."""
    from lynceus.plugins.registry import manager
    from lynceus.plugins.store import ConsentStore

    store = ConsentStore()
    manager.discover()
    manager.disabled_provider = lambda: store.disabled_ids()
    return store


def _check_terms(store) -> str | None:
    """Refuse headless runs until the app Terms are accepted (no prompts)."""
    from lynceus.plugins import terms

    if store.terms_accepted(terms.TERMS_VERSION, terms.terms_blob()):
        return None
    return _TERMS_HINT


def _node_block_reason(store, module_id: str) -> str | None:
    """Worker-safe EULA gate mirroring the GUI worker gate (no Qt)."""
    from lynceus.plugins.registry import manager

    info = manager.node_info(module_id)
    if info is None or info.source == "builtin":
        return None
    try:
        if info.manifest is not None:
            return store.block_reason(info.manifest)
        return store.node_block_reason(module_id, info.eula)
    except Exception as exc:
        return f"Cannot check extension: {exc}"


def _display_name(entry: dict, module_id: str) -> str:
    from lynceus.plugins.registry import manager

    label = str(entry.get("label") or "").strip()
    if label:
        return label
    info = manager.node_info(module_id)
    if info is not None and info.name:
        return info.name
    return module_id


def _port(value):
    """File edge types are strings; normalize known ones to PortType."""
    from lynceus.nodes.ports import PortType

    try:
        return PortType(value)
    except ValueError:
        return value


def _load_document(path_str: str):
    """Load a .lynx project or return an error string (never raises)."""
    from lynceus import project as project_io

    path = Path(path_str)
    if not path.is_file():
        return None, f"Project not found: {path_str}"
    try:
        return project_io.load_project(path), None
    except Exception as exc:
        return None, f"Cannot load project {path_str}: {exc}"


def _pipeline_config(entry: dict) -> dict:
    config = entry.get("config") or {}
    pipeline = config.get("pipeline", {})
    return pipeline if isinstance(pipeline, dict) else {}


def _file_config(entry: dict) -> str:
    config = entry.get("config") or {}
    return str(config.get("file_path", "") or "")


def _is_strategy_family(module_id: str) -> bool:
    from lynceus.plugins.registry import manager

    try:
        return bool(manager.list_variants(module_id))
    except Exception:
        return False


def _effective_specs(module_id: str, config: dict) -> dict:
    from lynceus.nodes._variants import effective_specs

    try:
        specs = effective_specs(module_id, config)
    except Exception:
        return {}
    return specs if isinstance(specs, dict) else {}


def _instance_ports(module_id: str, config: dict):
    """Per-instance (input, output) ports, mirroring the engine gate."""
    from lynceus.nodes._variants import resolved_ports
    from lynceus.plugins.registry import manager

    try:
        return resolved_ports(module_id, config)
    except Exception:
        pass
    try:
        return manager.load_ports(module_id)
    except Exception:
        return ((object(),), ())


def _orphan_input_empty(module_id: str, pipeline: dict, entry: dict) -> bool:
    """An unwired file-backed node with nothing usable selected."""
    if _is_strategy_family(module_id) and not str(
            pipeline.get("strategy", "") or "").strip():
        return True
    fkey = _effective_specs(module_id, pipeline).get("input_file_key")
    if fkey:
        # Product inputs persist the file inside config.pipeline (the Load
        # node keeps it flat in config); accept both shapes.
        path = pipeline.get(fkey) or _file_config(entry)
        return not path or not os.path.isfile(path)
    if module_id == LOAD_MODULE:
        path = _file_config(entry)
        return not path or not os.path.isfile(path)
    return False


def _build_run(doc: dict, source_overrides: dict[str, str]):
    """Translate a project document into engine run arguments.

    Mirrors the canvas gates (empty graph, unavailable nodes, strategy
    selection, scope, wired-loader files) with plain-English problems.
    Returns (spec, problems, blocks) where spec holds nodes/edges/configs/
    source_paths/node_info when usable, else None.
    """
    from lynceus.plugins.registry import manager
    from lynceus.plugins.store import ConsentStore

    entries = doc.get("nodes", [])
    raw_edges = doc.get("edges", [])
    if not entries:
        return None, ["Add at least one node to run a pipeline"], []

    _apply_source_overrides(entries, source_overrides)

    disabled = ConsentStore().disabled_ids()
    by_iid: dict[str, tuple[dict, str]] = {}
    problems: list[str] = []
    blocks: list[str] = []
    for entry in entries:
        module_id = entry.get("id", "")
        iid = entry.get("iid") or module_id
        by_iid[iid] = (entry, module_id)
        if not manager.contains(module_id):
            problems.append(f"Unknown node: {module_id}")
        elif module_id in disabled:
            blocks.append(
                "Node uses a disabled extension: "
                f"{_display_name(entry, module_id)}"
            )
    if problems:
        return None, problems, blocks

    typed_edges = []
    for edge in raw_edges:
        try:
            typed_edges.append((
                edge["src"], edge["dst"],
                _port(edge.get("out_type", "")),
                _port(edge.get("in_type", "")),
            ))
        except KeyError:
            problems.append(f"Malformed edge (needs src/dst): {edge!r}")
    if problems:
        return None, problems, blocks
    wired: set[str] = set()
    for src, dst, _o, _i in typed_edges:
        wired.add(src)
        wired.add(dst)

    unselected = sorted(
        _display_name(entry, module_id)
        for iid, (entry, module_id) in by_iid.items()
        if iid in wired and _is_strategy_family(module_id)
        and not str(_pipeline_config(entry).get("strategy", "") or "").strip()
    )
    if unselected:
        blocks.append(
            "Select a product strategy in node(s): " + ", ".join(unselected)
        )

    scope: list[str] = []
    for iid, (entry, module_id) in by_iid.items():
        if iid in wired:
            scope.append(iid)
            continue
        try:
            inputs, _outputs = _instance_ports(
                module_id, _pipeline_config(entry))
        except Exception:
            continue
        if inputs:
            continue  # orphan draft with inputs: ignored, like the canvas
        if _orphan_input_empty(module_id, _pipeline_config(entry), entry):
            continue
        scope.append(iid)
    if not scope:
        return None, [
            "No connected pipeline to run — connect nodes or add a source",
        ], blocks

    configs = {}
    for iid in scope:
        entry, _module_id = by_iid[iid]
        pipeline = _pipeline_config(entry)
        if pipeline:
            configs[iid] = pipeline

    source_paths: dict[str, str] = {}
    for iid in scope:
        entry, module_id = by_iid[iid]
        if module_id != LOAD_MODULE:
            continue
        path = source_overrides.get(iid) or _file_config(entry)
        if not path:
            if iid not in wired:
                continue
            blocks.append("Set a LiDAR file in every Load node")
            continue
        if not os.path.isfile(path):
            if iid not in wired:
                continue
            blocks.append(f"Source file not found: {path}")
            continue
        source_paths[iid] = path

    node_info = {iid: _display_name(entry, module_id)
                 for iid, (entry, module_id) in by_iid.items()
                 if iid in scope}
    spec = {
        "nodes": [(iid, by_iid[iid][1]) for iid in scope],
        "edges": typed_edges,
        "configs": configs,
        "source_paths": source_paths,
        "node_info": node_info,
    }
    return spec, problems, blocks


def _pipeline_config(entry: dict) -> dict:
    config = entry.get("config") or {}
    pipeline = config.get("pipeline", {})
    return pipeline if isinstance(pipeline, dict) else {}


def _file_input_key(module_id: str, pipeline: dict) -> str:
    """Config key holding this input's file ("" when not file-backed)."""
    if module_id == LOAD_MODULE:
        return "file_path"
    fkey = _effective_specs(module_id, pipeline).get("input_file_key")
    return str(fkey or "")


def _apply_source_overrides(entries: list,
                             source_overrides: dict[str, str]) -> None:
    """Pre-merge --source files into the project entries (in place).

    Mirrors picking the file in the GUI: the loader keeps it flat in
    ``config.file_path`` while product inputs keep it in
    ``config.pipeline[<input_file_key>]``. Gates below then see the
    effective file with no special cases.
    """
    by_iid = {str(e.get("iid") or e.get("id", "")): e for e in entries}
    for iid, path in source_overrides.items():
        entry = by_iid.get(iid)
        if entry is None:
            continue
        module_id = str(entry.get("id", ""))
        config = entry.get("config")
        if not isinstance(config, dict):
            config = {}
            entry["config"] = config
        if module_id == LOAD_MODULE:
            config["file_path"] = path
        else:
            fkey = _file_input_key(
                module_id, _pipeline_config(entry)) or "file_path"
            pipeline = config.get("pipeline")
            if not isinstance(pipeline, dict):
                pipeline = {}
                config["pipeline"] = pipeline
            pipeline[fkey] = path


def _parse_sources(values: list[str], file_iids: list[str]):
    """Parse --source flags into {iid: path} (bare path needs one input)."""
    overrides: dict[str, str] = {}
    for value in values or []:
        if "=" in value:
            iid, _, path = value.partition("=")
            iid, path = iid.strip(), path.strip()
            if not iid or not path:
                return None, f"Bad --source (want iid=path): {value!r}"
            if iid not in file_iids:
                return None, (
                    f"--source iid unknown or not a file input: {iid!r}")
            overrides[iid] = path
        else:
            if len(file_iids) != 1:
                return None, (
                    "--source needs iid=path when the project has "
                    f"{len(file_iids)} file inputs"
                )
            overrides[file_iids[0]] = value
    for iid, path in overrides.items():
        if not os.path.isfile(path):
            return None, f"Source file not found: {path}"
    return overrides, None


def _collect_products(outputs: dict, node_info: dict, edges: list,
                      source_paths: dict) -> list[dict]:
    from dataclasses import asdict

    from lynceus.ui.outputs_model import flatten_outputs

    products = flatten_outputs(
        outputs, node_info, edges=edges, source_paths=source_paths)
    return [asdict(product) for product in products]


def _cmd_run(args) -> int:
    from lynceus.processing.controller import (
        PipelineCallbacks, PipelineController,
    )

    store = _bootstrap()
    problem = _check_terms(store)
    if problem:
        _err(problem)
        return EXIT_BLOCKED
    doc, error = _load_document(args.project)
    if error:
        _err(error)
        return EXIT_USAGE
    from lynceus import project as project_io

    missing = project_io.validate_nodes(doc)
    if missing:
        _err("Unknown node(s): " + ", ".join(missing[:5])
             + ("..." if len(missing) > 5 else ""))
        return EXIT_USAGE

    entries = {(e.get("iid") or e.get("id", "")): e
               for e in doc.get("nodes", [])}
    file_iids = sorted(
        iid for iid, e in entries.items()
        if _file_input_key(str(e.get("id", "")),
                           _pipeline_config(e)))
    overrides, error = _parse_sources(args.source, file_iids)
    if error:
        _err(error)
        return EXIT_USAGE

    spec, problems, blocks = _build_run(doc, overrides)
    for problem in problems:
        _err(problem)
    if problems:
        return EXIT_USAGE
    for block in blocks:
        _say(block)
    if blocks:
        return EXIT_BLOCKED

    max_workers = args.workers
    if max_workers is not None and max_workers < 1:
        _err("--workers needs a positive integer")
        return EXIT_USAGE
    memory_budget = None
    if args.memory_gb is not None:
        if args.memory_gb <= 0:
            _err("--memory-gb needs a positive number")
            return EXIT_USAGE
        memory_budget = int(args.memory_gb * 1024 ** 3)
    operator = None
    if args.operator_name or args.operator_org:
        operator = {"name": args.operator_name or "",
                    "org": args.operator_org or ""}

    controller = PipelineController()
    controller.eula_checker = (
        lambda module_id: _node_block_reason(store, module_id))

    done = threading.Event()
    state: dict = {"cancelled": False, "messages": []}

    def on_message(text: str, kind: str = "info") -> None:
        state["messages"].append({"text": text, "kind": kind})
        if args.quiet and kind in ("info", "success", "action"):
            return
        print(f"[{kind}] {text}", file=sys.stderr)

    def on_finished(outputs, _results=None, failed: int = 0,
                    tiles=None, extra=None) -> None:
        state["outputs"] = outputs
        state["failed"] = int(failed or 0)
        state["tiles"] = list(tiles) if isinstance(tiles, list) else []
        state["extra"] = extra if isinstance(extra, dict) else {}
        done.set()

    def on_cancelled() -> None:
        state["cancelled"] = True
        done.set()

    def on_progress(finished: int, total: int) -> None:
        if not args.quiet:
            print(f"[progress] {finished}/{total}", file=sys.stderr)

    callbacks = PipelineCallbacks(
        on_message=on_message, on_finished=on_finished,
        on_cancelled=on_cancelled, on_progress=on_progress)

    try:
        controller.run(
            spec["nodes"], spec["edges"], spec["source_paths"], callbacks,
            session_root=args.output, configs=spec["configs"],
            max_workers=max_workers, memory_budget_bytes=memory_budget,
            operator=operator, keep_intermediates=args.keep_intermediates,
        )
    except Exception as exc:
        _err(f"Could not start the run: {exc}")
        return EXIT_FAILED
    try:
        done.wait()
    except KeyboardInterrupt:
        controller.cancel()
        print("[info] Cancelling…", file=sys.stderr)
        done.wait(timeout=120)
        state["cancelled"] = True

    if state.get("cancelled"):
        _report(args, spec, state, controller, status="cancelled")
        return EXIT_FAILED
    extra = state.get("extra") or {}
    if any(str(key).startswith("blocked_by_") for key in extra):
        _report(args, spec, state, controller, status="blocked")
        return EXIT_BLOCKED
    failed = int(state.get("failed") or 0)
    status = "ok" if failed == 0 else "failed"
    _report(args, spec, state, controller, status=status)
    return EXIT_OK if failed == 0 else EXIT_FAILED


def _report(args, spec, state: dict, controller, status: str) -> None:
    outputs = state.get("outputs") or {}
    products = _collect_products(outputs, spec["node_info"], spec["edges"],
                                 spec["source_paths"])
    warnings = [m["text"] for m in state["messages"] if m["kind"] == "warning"]
    errors = [m["text"] for m in state["messages"] if m["kind"] == "error"]
    report = {
        "status": status,
        "project": str(args.project),
        "session_dir": controller.session_dir,
        "tiles_processed": len(state.get("tiles") or []),
        "failed_tasks": int(state.get("failed") or 0),
        "products": products,
        "warnings": warnings,
        "errors": errors,
    }
    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return
    print(f"status: {status}")
    if report["session_dir"]:
        print(f"session: {report['session_dir']}")
    print(f"tiles: {report['tiles_processed']}  "
          f"failed tasks: {report['failed_tasks']}  "
          f"products: {len(products)}")
    for product in products:
        line = f"  - {product.get('node_name')}: {product.get('name')}"
        if product.get("path"):
            line += f"  [{product['path']}]"
        print(line)
    for text in warnings:
        print(f"  warning: {text}")
    for text in errors:
        print(f"  error: {text}")


def _cmd_nodes(args) -> int:
    from lynceus.plugins.registry import manager

    _bootstrap()
    rows = []
    for info in sorted(manager.list_nodes(),
                       key=lambda i: (i.category, i.name)):
        try:
            variants = [v["key"] for v in manager.list_variants(info.node_id)]
        except Exception:
            variants = []
        rows.append({
            "id": info.node_id,
            "name": info.name,
            "category": info.category,
            "subcategory": info.subcategory or "",
            "source": info.source,
            "strategies": variants,
        })
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        for row in rows:
            line = f"{row['id']}  |  {row['name']}  [{row['category']}]"
            if row["strategies"]:
                line += f"  (strategies: {', '.join(row['strategies'])})"
            print(line)
    return EXIT_OK


def _cmd_validate(args) -> int:
    _bootstrap()
    doc, error = _load_document(args.project)
    if error:
        print(json.dumps({"valid": False, "errors": [error], "blocks": []},
                         indent=2))
        return EXIT_USAGE
    from lynceus import project as project_io

    errors: list[str] = []
    missing = project_io.validate_nodes(doc)
    if missing:
        errors.append("Unknown node(s): " + ", ".join(missing))
    _spec, problems, blocks = _build_run(doc, {})
    errors.extend(problems)
    print(json.dumps({
        "valid": not errors and not blocks,
        "project": str(args.project),
        "nodes": len(doc.get("nodes", [])),
        "edges": len(doc.get("edges", [])),
        "errors": errors,
        "blocks": blocks,
    }, indent=2))
    if errors:
        return EXIT_USAGE
    if blocks:
        return EXIT_BLOCKED
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lynceus",
        description="Headless LynceusScan pipelines (same engine as the GUI).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run a .lynx project headless.")
    run.add_argument("project", help="Project file (.lynx).")
    run.add_argument("--source", action="append", default=[],
                     help="Override an input file: PATH (single file input) "
                          "or iid=PATH (repeatable, any file-backed input).")
    run.add_argument("--output",
                     help="Session root (default: the app output root).")
    run.add_argument("--workers", type=int, default=None,
                     help="Pool workers (default: engine choice).")
    run.add_argument("--memory-gb", type=float, default=None,
                     help="Memory budget in GiB (default: engine choice).")
    run.add_argument("--keep-intermediates", action="store_true",
                     help="Keep per-tile intermediates and the tile cache.")
    run.add_argument("--operator-name", default="",
                     help="Operator name for provenance.")
    run.add_argument("--operator-org", default="",
                     help="Operator organization for provenance.")
    run.add_argument("--json", action="store_true",
                     help="Print the machine-readable report on stdout.")
    run.add_argument("--quiet", action="store_true",
                     help="Only warnings/errors to stderr.")
    run.set_defaults(func=_cmd_run)

    nodes = sub.add_parser("nodes", help="Inspect the node registry.")
    nodes_sub = nodes.add_subparsers(dest="nodes_command", required=True)
    nodes_list = nodes_sub.add_parser("list", help="List nodes.")
    nodes_list.add_argument("--json", action="store_true")
    nodes_list.set_defaults(func=_cmd_nodes)

    validate = sub.add_parser("validate", help="Validate a .lynx project.")
    validate.add_argument("project", help="Project file (.lynx).")
    validate.set_defaults(func=_cmd_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    sys.exit(main())
