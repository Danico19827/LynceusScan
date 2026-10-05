# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Discover node capabilities and build the processing DAG without Qt.

Each node in lynceus/nodes/* exposes PROCESSING_SPECS:
- tile_task / barrier_task / visual_task / consolidate_task: function name
- triggers_tiling: bool for source nodes such as Load LAS/LAZ
- consolidator: bool for merge nodes deferred to the final session
- provides: metadata supplied by the node
- requires: metadata expected from its inputs
- config_schema: per-node configuration defaults
- point_cloud_dir / passes_flags / input_file_key / output_files /
  output_globs / output_port / point_cloud_export / qml: artifact, reuse,
  viewer and export contracts

Strategy families resolve per instance: ``effective_node_caps`` merges the
selected variant over the static capabilities (see ``_variants``).

The DAG is built from the canvas edges typed by the domain PortType contract.
Dependencies come from actual connections. ``requires`` is informational:
flags are resolved from real inputs and an unsatisfied requirement degrades
the node instead of blocking execution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from lynceus.nodes._paths import scope_for
from lynceus.nodes._variants import (
    current_variant_key,
    resolved_ports,
    variant_module,
)
from lynceus.nodes.ports import PortType, port_metadata
from lynceus.plugins.registry import NodeLoadError, manager, run_file_node_task

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dynamic capability discovery through PROCESSING_SPECS.
# ---------------------------------------------------------------------------

_CAPABILITY_CACHE: dict[str, dict] = {}


def discover_node_capabilities(node_id: str) -> dict:
    """Import a node module and resolve its PROCESSING_SPECS capabilities.

    Callable task entries are resolved lazily and missing declarations are
    retained as ``None`` so discovery failures remain diagnosable.
    """
    if node_id in _CAPABILITY_CACHE:
        return _CAPABILITY_CACHE[node_id]

    result = {
        "tile_task": None,
        "barrier_task": None,
        "visual_task": None,
        "consolidate_task": None,
        "triggers_tiling": False,
        "consolidator": False,
        "provides": {},
        "requires": {},
        "passes_flags": (),
        "point_cloud_dir": "",
        "input_file_key": "",
        "session_file": "",
        "table_output_file": "",
        "output_files": {},
        "point_cloud_export": False,
        "config_schema": {},
        "qml": {},
        "description": "",
    }
    try:
        mod = manager.import_node(node_id)
    except NodeLoadError as exc:
        logger.warning(f"Node module {node_id} not found: {exc}")
        _CAPABILITY_CACHE[node_id] = result
        return result

    specs = getattr(mod, "PROCESSING_SPECS", {})

    for kind in ("tile_task", "barrier_task", "visual_task", "consolidate_task"):
        fn_name = specs.get(kind)
        if fn_name:
            fn = getattr(mod, fn_name, None)
            if fn is None:
                logger.warning(
                    f"Node {node_id} declares {kind} '{fn_name}' "
                    "but function not found"
                )
            result[kind] = fn

    result["triggers_tiling"] = specs.get("triggers_tiling", False)
    result["consolidator"] = specs.get("consolidator", False)
    result["provides"] = specs.get("provides", {})
    result["requires"] = specs.get("requires", {})
    result["passes_flags"] = tuple(specs.get("passes_flags") or ())
    result["point_cloud_dir"] = specs.get("point_cloud_dir", "")
    result["input_file_key"] = specs.get("input_file_key", "")
    result["session_file"] = specs.get("session_file", "")
    result["table_output_file"] = specs.get("table_output_file", "")
    result["output_files"] = specs.get("output_files", {})
    result["output_globs"] = specs.get("output_globs", ())
    result["output_port"] = specs.get("output_port", "")
    result["point_cloud_export"] = specs.get("point_cloud_export", False)
    result["config_schema"] = specs.get("config_schema", {})
    result["qml"] = specs.get("qml", {})
    result["description"] = getattr(mod, "NODE_DESCRIPTION", "")

    _CAPABILITY_CACHE[node_id] = result
    return result


def _task_exec(module_id: str, fn: Callable, args: tuple):
    """Return a (fn, args) pair that the `spawn` pool can pickle.

    Extension modules load from file under a synthetic name that only exists
    in the parent process; pickling their functions directly fails in the
    workers. For them, dispatch through the module-level `run_file_node_task`
    wrapper, which is importable by path from any process.
    """
    info = manager.node_info(module_id)
    if info is not None and info.source == "extension":
        return run_file_node_task, (module_id, fn.__name__, args)
    return fn, args


def _resolve_spec_overlay(module, base: dict) -> dict:
    """Merge a variant module's PROCESSING_SPECS over static capabilities."""
    specs = getattr(module, "PROCESSING_SPECS", {}) or {}
    merged = dict(base)
    for kind in ("tile_task", "barrier_task", "visual_task", "consolidate_task"):
        fn_name = specs.get(kind)
        if not fn_name:
            continue
        fn = getattr(module, fn_name, None)
        if fn is None:
            logger.warning(
                f"Variant {getattr(module, '__name__', '?')} declares "
                f"{kind} '{fn_name}' but function not found"
            )
        merged[kind] = fn
    for key in (
        "triggers_tiling",
        "consolidator",
        "provides",
        "requires",
        "passes_flags",
        "point_cloud_dir",
        "input_file_key",
        "session_file",
        "table_output_file",
        "output_files",
        "output_globs",
        "output_port",
        "point_cloud_export",
        "qml",
    ):
        if key in specs:
            merged[key] = specs[key]
    # config_schema merges per-key (variant params extend the base): the
    # base keeps its strategy field so the Inspector dropdown survives.
    if "config_schema" in specs:
        merged_schema = dict(merged.get("config_schema") or {})
        merged_schema.update(specs["config_schema"] or {})
        merged["config_schema"] = merged_schema
    return merged


def effective_node_caps(node_id: str, config: dict | None = None) -> dict:
    """Per-instance capabilities for a strategy node.

    Non-variant nodes resolve to their static capabilities (the common case,
    unchanged); a strategy node merges the selected variant's
    PROCESSING_SPECS over the base, so its task/session/output contract
    follows the configured variant.
    """
    caps = discover_node_capabilities(node_id)
    key = current_variant_key(node_id, config)
    if key is None:
        return caps
    try:
        variant_module_obj = variant_module(node_id, key)
    except (ImportError, AttributeError):
        # variant_module already degrades to None on failure; anything else
        # is unexpected, so fall back to static capabilities loudly.
        logger.warning(f"Strategy variant {node_id}.{key} failed to load")
        return caps
    if variant_module_obj is None:
        return caps
    return _resolve_spec_overlay(variant_module_obj, caps)


def get_node_metadata(node_id: str) -> dict:
    """Return typed ports, capabilities, configuration, and description metadata."""
    try:
        mod = manager.import_node(node_id)
    except NodeLoadError:
        return {}
    specs = getattr(mod, "PROCESSING_SPECS", {})
    return {
        "inputs": tuple(getattr(mod, "INPUTS", ())),
        "outputs": tuple(getattr(mod, "OUTPUTS", ())),
        "capabilities": list(specs.keys()),
        "config_schema": specs.get("config_schema", {}),
        "config_defaults": getattr(mod, "get_config_defaults", lambda: {})(),
        "description": getattr(mod, "NODE_DESCRIPTION", ""),
    }


# ---------------------------------------------------------------------------
# Graph and DAG
#
# Instance identity: canvas nodes are ``(iid, module_id)`` pairs. The iid is
# unique even when several instances share a module; static capabilities come
# from the module's PROCESSING_SPECS, per-instance ones from
# ``effective_node_caps(module, config)`` (strategy overlay).
# ---------------------------------------------------------------------------

Edge = tuple[str, str, PortType, PortType]  # (src_iid, dst_iid, out_type, in_type)
NodeSpec = tuple[str, str]  # (iid, module_id)


@dataclass
class Task:
    """Atomic DAG task dispatched after all dependency tasks complete."""

    task_id: str
    iid: str  # Owning instance ID for status and session tracking.
    module_id: str  # Node module ID for capabilities and labels.
    kind: str  # "tile" | "barrier"
    fn: Callable
    args: tuple
    deps: set[str] = field(default_factory=set)
    tile_id: str | None = None
    weight: int = 0  # Tile point count: queue priority and RAM estimate.


def compile_pipeline(nodes, edges) -> list[str]:
    """Return canvas instance IDs in dependency-first topological order.

    ``nodes`` may contain instance IDs or ``(instance_id, module_id)`` pairs.
    No cycle validation here: a cycle yields a silently wrong order and the
    executor raises once dispatch stalls.
    """
    ids = [n[0] if isinstance(n, tuple) else n for n in nodes]
    order: list[str] = []
    visited: set[str] = set()

    def visit(iid: str) -> None:
        if iid in visited:
            return
        visited.add(iid)
        for edge in edges:
            if edge[1] == iid:
                visit(edge[0])
        order.append(iid)

    for iid in ids:
        visit(iid)
    return order


def _config_defaults(caps: dict) -> dict:
    """Extract default values from a node configuration schema."""
    defaults = {}
    for key, value in caps.get("config_schema", {}).items():
        if isinstance(value, dict) and "default" in value:
            defaults[key] = value["default"]
    return defaults


def _validate_requires(
    iid: str,
    module_id: str,
    caps: dict,
    input_edges: list[Edge],
    modules: dict[str, str],
    resolved_flags: dict[str, dict],
) -> None:
    """Log unsatisfied informational requirements before task creation.

    Requirements are advisory: an unsatisfied flag produces a warning and the
    node runs with degraded behavior instead of blocking the pipeline.
    """
    requires = caps.get("requires", {})
    if not requires:
        return

    def source_satisfies(src_iid: str, key: str, value) -> bool:
        return resolved_flags.get(src_iid, {}).get(key) == value

    for key, value in requires.items():
        satisfied = any(
            source_satisfies(src, key, value)
            for src, _dst, _out, _in in input_edges
        )
        if not satisfied:
            providers = sorted(
                {modules[src] for src, _d, _o, _i in input_edges if src in modules}
            )
            logger.warning(
                f"{module_id} (instance {iid}) prefers {key}={value}, "
                f"but its inputs ({', '.join(providers) or 'none'}) do not "
                "provide it. Running with degraded behavior "
                f"(flag {key} resolves to False)."
            )


def _resolve_node_flags(
    module_id: str,
    input_edges: list[Edge],
    modules: dict[str, str],
    provider_flags: dict[str, dict] | None = None,
    caps: dict | None = None,
) -> dict:
    """Resolve contract flags from the node's actual provider edges.

    ``provider_flags`` is resolved in topological order. Nodes inherit the
    resolved provider values rather than static declarations, allowing a
    cleaning chain to preserve flags such as ``classified``. ``caps`` may be a
    per-instance (strategy) capability dict; otherwise static caps are used.
    """
    caps = caps if caps is not None else discover_node_capabilities(module_id)
    flags: dict = {}
    for key, value in caps.get("provides", {}).items():
        flags[key] = value

    provider_flags = provider_flags or {}

    # Inherit requirements satisfied by at least one input provider.
    requires = caps.get("requires", {})
    for key, value in requires.items():
        flags[key] = any(
            provider_flags.get(src, {}).get(key) == value
            for src, _dst, _out, _in in input_edges
        )

    # Pass through explicitly declared signals, such as classified.
    for key in caps.get("passes_flags", ()):
        flags[key] = any(
            provider_flags.get(src, {}).get(key)
            for src, _dst, _out, _in in input_edges
        )
    return flags


def build_ctx(
    session_dir: str,
    crs: str,
    cell_size: float = 1.0,
    source_meta: dict | None = None,
    acceleration: dict | None = None,
) -> dict:
    """Build the shared session context without global contract state."""
    return {
        "session_dir": session_dir,
        "crs": crs,
        "cell_size": cell_size,
        "meta": source_meta or {},
        "acceleration": acceleration or {"device": "cpu", "backend": "none", "available": False},
    }


def _task_ctx(
    base_ctx: dict,
    caps: dict,
    flags: dict,
    overrides: dict | None = None,
    node_iid: str | None = None,
    module_id: str | None = None,
) -> dict:
    """Build a node context from base values, defaults, overrides, and flags.

    Inspector overrides replace defaults. Contract flags are applied last and
    are not user-configurable. ``node_iid`` scopes per-instance artifacts and
    ``module_id`` tags the ctx with the node's module (used for provenance).
    """
    ctx = dict(base_ctx)
    ctx.update(_config_defaults(caps))
    if overrides:
        ctx.update(overrides)
    if node_iid is not None:
        ctx["node_iid"] = node_iid
    if module_id is not None:
        ctx["module_id"] = module_id
    ctx.update(flags)
    return ctx


def _port_id(port) -> str:
    """Normalize a port value to its canonical string ID."""
    if hasattr(port, "value"):
        return port.value
    return str(port)


def _inject_input_paths(
    ctx: dict,
    module_id: str,
    input_edges: list[Edge],
    caps_of: Callable[[str], tuple[str, dict]],
    config: dict | None = None,
    scope_map: dict | None = None,
) -> None:
    """Inject provider artifact paths into a consumer context.

    Product inputs are copied to ``session_dir/<src_iid>/<session_file>``
    (per-instance by design, F2 path-by-port). Core mosaic producers
    (DTM/DSM/CHM/...) declare ``output_files`` and write the session root
    in legacy graphs, or ``session_dir/<scope>/<file>`` when the provider
    instance is branch-scoped. Consumers receive both a type-based key
    (for example ``chm_mosaic_path``) and a position-based ``in<pos>_path``
    key so combiners can distinguish same-type inputs.
    """
    try:
        input_ports, _ = resolved_ports(module_id, config)
    except Exception:
        input_ports = ()
    if not input_ports:  # non-variant static fallback
        try:
            input_ports, _ = manager.load_ports(module_id)
        except Exception:
            input_ports = ()
    used_positions: set[int] = set()
    for src_iid, _dst, out_type, in_type in input_edges:
        _sid, source_caps = caps_of(src_iid)
        source_files = source_caps.get("output_files") or {}
        mosaic_file = source_files.get(_port_id(out_type))
        if mosaic_file:
            # Mosaic producer: session root, or the provider's scope dir.
            scope = (scope_map or {}).get(src_iid, "")
            provider_path = str(Path(ctx.get("session_dir", "")) / scope / mosaic_file) if scope else str(
                Path(ctx.get("session_dir", "")) / mosaic_file
            )
        else:
            session_file = source_caps.get("session_file") or ""
            if session_file:
                provider_path = str(
                    Path(ctx.get("session_dir", "")) / src_iid / session_file
                )
            else:
                # Legacy forestry-table contract (table_output_file without an
                # instance dir); only the two forestry table nodes declare it.
                table_file = source_caps.get("table_output_file")
                if not table_file:
                    continue
                provider_path = str(Path(ctx.get("session_dir", "")) / table_file)
        ctx.setdefault(f"{_port_id(out_type)}_path", provider_path)
        in_id = _port_id(in_type)
        seen_same_type = False
        for pos, item in enumerate(input_ports):
            if pos in used_positions:
                continue
            meta = port_metadata(item)
            if _port_id(meta.port_type) == in_id:
                used_positions.add(pos)
                ctx[f"in{pos}_path"] = provider_path
                seen_same_type = True
                break
        if not seen_same_type and input_ports:
            # Fall back to edge order when AST metadata has no matching port.
            for pos, item in enumerate(input_ports):
                if pos in used_positions:
                    continue
                used_positions.add(pos)
                ctx[f"in{pos}_path"] = provider_path
                break


def build_dag(
    order: list[str],
    edges: list[Edge],
    tiles: list[dict],
    ctx: dict,
    modules: dict[str, str] | None = None,
    configs: dict[str, dict] | None = None,
) -> list[Task]:
    """Compile the canvas graph into edge-dependent executable tasks.

    ``order`` is topological. ``modules`` maps instance IDs to module IDs and
    defaults to the instance ID for compatibility. Task identity remains per
    instance, so same-type nodes cannot cross-contaminate.

        Tile tasks depend on connected upstream tile tasks. Barrier tasks depend
        on their own tiles and connected upstream barriers. Informational requires
        are validated before task creation but never block execution.
    """
    modules = modules or {}
    configs = configs or {}

    def caps_of(iid: str) -> tuple[str, dict]:
        module_id = modules.get(iid, iid)
        return module_id, effective_node_caps(module_id, configs.get(iid))

    input_edges_by_node: dict[str, list[Edge]] = {}
    for edge in edges:
        input_edges_by_node.setdefault(edge[1], []).append(edge)

    # Resolve contract flags in topological order. A node inherits resolved
    # provider flags so cleaning chains can pass signals such as classified.
    resolved: dict[str, dict] = {}
    for iid in order:
        module_id, caps = caps_of(iid)
        resolved[iid] = _resolve_node_flags(
            module_id,
            input_edges_by_node.get(iid, []),
            modules,
            resolved,
            caps=caps,
        )

    # Tile domain per node: the loader (LiDAR source) feeding a subgraph.
    # Tile tasks are restricted to their own domain; barriers/mosaics aggregate
    # every tile.
    # - a loader is its own domain
    # - a node inherits the domain of its immediate tile-task provider
    # - no loader upstream (barrier-only) => None => all tiles
    domain_by_node: dict[str, str | None] = {}
    for iid in order:
        _mid, caps = caps_of(iid)
        if caps.get("triggers_tiling"):
            domain_by_node[iid] = iid
            continue
        srcs = []
        for source_iid, *_edge in input_edges_by_node.get(iid, []):
            _sid, s_caps = caps_of(source_iid)
            if s_caps.get("triggers_tiling"):
                srcs.append(source_iid)
            elif source_iid in domain_by_node and domain_by_node[source_iid]:
                srcs.append(domain_by_node[source_iid])
        domain_by_node[iid] = srcs[0] if srcs else None

    def _domain_tiles(iid: str) -> list:
        """Tiles in the node's domain (all tiles when there is no upstream loader)."""
        dom = domain_by_node.get(iid)
        if not dom:
            return tiles
        return [t for t in tiles if t.get("src_iid") == dom]

    # Branch isolation (F4.2): same-module instances over overlapping
    # tiles write to instance-scoped directories; disjoint domains keep
    # the shared layout (multi-source aggregation), chained instances
    # keep sharing by design, per-instance writers never collide.
    scope_map = _branch_scope_map(
        order, edges, tiles, modules, configs, _domain_tiles
    )
    ctx = dict(ctx)
    ctx["_scope_map"] = scope_map

    tasks: dict[str, Task] = {}
    # Mosaic barriers for same-module instances wait for all same-module tile
    # tasks because those instances share the output directory.
    mosaic_deps: dict[str, set[str]] = {}

    # First: tile tasks
    for iid in order:
        module_id, caps = caps_of(iid)
        tile_fn = caps.get("tile_task")
        if tile_fn is None:
            continue

        input_edges = input_edges_by_node.get(iid, [])
        _validate_requires(iid, module_id, caps, input_edges, modules, resolved)
        flags = resolved[iid]
        task_ctx = _task_ctx(
            ctx, caps, flags, configs.get(iid), node_iid=iid, module_id=module_id
        )
        task_ctx["tile_scope"] = scope_map.get(iid, "")
        _inject_input_paths(
            task_ctx, module_id, input_edges, caps_of, configs.get(iid),
            scope_map,
        )

        # The immediate provider selects the point-cloud directory. Without a
        # provider directory, point_src falls back to classified/ or raw tiles.
        point_source_dir = ""
        for source_iid, _dst, _out, _in in input_edges:
            _sid, source_caps = caps_of(source_iid)
            source_dir = source_caps.get("point_cloud_dir")
            if source_dir:
                point_source_dir = scope_for(
                    source_dir, scope_map.get(source_iid, "")
                )
                break
        if point_source_dir:
            task_ctx["point_source_dir"] = point_source_dir

        produces_mosaic = caps.get("barrier_task") is not None
        for tile in _domain_tiles(iid):
            tile_id = tile["tile_id"]
            task_id = f"{iid}|{tile_id}"
            deps = set()
            for source_iid, _dst, out_type, _in in input_edges:
                _src_mod, source_caps = caps_of(source_iid)
                if source_caps.get("triggers_tiling"):
                    continue
                out_id = _port_id(out_type)
                source_files = source_caps.get("output_files") or {}
                if (
                    out_id in source_files
                    and source_caps.get("barrier_task") is not None
                ):
                    # Barrier final (mosaic/merge product): only stable
                    # after the provider merge, so the tile waits for it
                    # instead of racing the provider tiles (a tile
                    # sampling a mosaic being written reads partial or
                    # missing data). Consolidate finals keep the legacy
                    # behavior (different session scope).
                    deps.add(f"{source_iid}|merge")
                elif source_caps.get("tile_task") is not None:
                    deps.add(f"{source_iid}|{tile_id}")
            exec_fn, exec_args = _task_exec(module_id, tile_fn, (tile, task_ctx))
            tasks[task_id] = Task(
                task_id, iid, module_id, "tile", exec_fn, exec_args,
                deps, tile_id, tile.get("point_count", 0),
            )
            if produces_mosaic:
                mosaic_deps.setdefault(module_id, set()).add(task_id)

    # Second: barrier tasks
    # Same-module mosaic barriers (same output folder: dsm/, dtm/, ...)
    # write the same mosaic file; they are serialized in topological
    # order (each waits for the merge of the previous sibling) so
    # two processes never write the output at the same time. The
    # last one leaves the complete mosaic.
    mosaic_siblings: dict[str, list[str]] = {}
    for iid in order:
        _mid, caps = caps_of(iid)
        if caps.get("tile_task") is not None and caps.get("barrier_task"):
            mosaic_siblings.setdefault(_mid, []).append(iid)

    for iid in order:
        module_id, caps = caps_of(iid)
        # Consolidators reuse the same barriers: they are barrier-only tasks
        # that merge per-segment provider products into a single final one.
        barrier_fn = caps.get("consolidate_task") or caps.get("barrier_task")
        if barrier_fn is None:
            continue

        input_edges = input_edges_by_node.get(iid, [])
        _validate_requires(iid, module_id, caps, input_edges, modules, resolved)
        flags = resolved[iid]
        task_ctx = _task_ctx(
            ctx, caps, flags, configs.get(iid), node_iid=iid, module_id=module_id
        )
        task_ctx["tile_scope"] = scope_map.get(iid, "")
        _inject_input_paths(
            task_ctx, module_id, input_edges, caps_of, configs.get(iid),
            scope_map,
        )

        task_id = f"{iid}|merge"
        deps: set[str] = set()
        if caps.get("tile_task") is not None:
            same_mod = mosaic_deps.get(module_id)
            deps = set(same_mod) if same_mod else {
                f"{iid}|{tile['tile_id']}" for tile in _domain_tiles(iid)
            }
            # Mosaic serialization: wait for merge of siblings
            # earlier in topological order.
            for sib in mosaic_siblings.get(module_id, []):
                if sib == iid:
                    break
                deps.add(f"{sib}|merge")
        for source_iid, _dst, _out, _in in input_edges:
            _src_mod, source_caps = caps_of(source_iid)
            if source_caps.get("triggers_tiling"):
                continue
            if (
                source_caps.get("barrier_task") is not None
                or source_caps.get("consolidate_task") is not None
            ):
                deps.add(f"{source_iid}|merge")

        if caps.get("point_cloud_export"):
            point_cloud_sources = []
            for source_iid, _dst, _out, _in in input_edges:
                source_module, source_caps = caps_of(source_iid)
                if not (
                    source_caps.get("tile_task")
                    or source_caps.get("triggers_tiling")
                ):
                    continue
                source_tiles = _domain_tiles(source_iid)
                deps.update(
                    f"{source_iid}|{tile['tile_id']}"
                    for tile in source_tiles
                    if source_caps.get("tile_task") is not None
                )
                point_cloud_sources.append(
                    {
                        "iid": source_iid,
                        "module": source_module,
                        "point_cloud_dir": scope_for(
                            source_caps.get("point_cloud_dir", ""),
                            scope_map.get(source_iid, ""),
                        ),
                        "tiles": source_tiles,
                    }
                )
            task_ctx["_point_cloud_sources"] = point_cloud_sources

        exec_fn, exec_args = _task_exec(module_id, barrier_fn, (task_ctx,))
        tasks[task_id] = Task(
            task_id, iid, module_id, "barrier", exec_fn, exec_args, deps
        )

    return list(tasks.values())


def _barrier_output_names(caps: dict) -> set | None:
    """Output basenames a barrier writes, or None when not statically known.

    Inputs and exports write per-instance paths (out of scope here).
    Wildcard globs and code-fixed mosaics without declarations are
    unknown: callers treat unknown as colliding (safe direction).
    """
    names = set((caps.get("output_files") or {}).values())
    if caps.get("session_file"):
        names.add(caps["session_file"])
    for glob in caps.get("output_globs") or ():
        base = str(glob).split("/")[-1]
        if "*" in base or not base:
            return None
        names.add(base)
    return names or None


def _branch_scope_map(order, edges, tiles, modules, configs, tiles_of) -> dict:
    """Instance scopes for same-module branches sharing tiles or finals.

    Returns ``{iid: iid}`` for instances that must write apart; everyone
    else keeps the legacy shared layout (single instances, disjoint
    multi-source domains, chained instances sharing by design, per-instance
    writers like product inputs and exports). Tile tasks scope on shared
    tile ids (same output paths, always); barrier-only instances scope on
    shared output basenames without a chaining edge. Deterministic per
    graph+tiles, so identical re-runs reuse.
    """
    modules = modules or {}
    configs = configs or {}

    def _caps(iid: str) -> dict:
        try:
            return effective_node_caps(
                modules.get(iid, iid), configs.get(iid)
            )
        except Exception:
            return {}

    def _safe(iid: str) -> bool:
        caps = _caps(iid)
        return bool(caps.get("input_file_key") or caps.get("point_cloud_export"))

    chained: set[tuple[str, str]] = set()
    for edge in edges:
        chained.add((edge[0], edge[1]))
        chained.add((edge[1], edge[0]))

    scoped: set[str] = set()
    by_module: dict[str, list[str]] = {}
    for iid in order:
        if _caps(iid).get("tile_task"):
            by_module.setdefault(modules.get(iid, iid), []).append(iid)
    for module_id, iids in by_module.items():
        if len(iids) < 2 or all(_safe(i) for i in iids):
            continue
        seen: dict[str, str] = {}
        for iid in iids:
            for tile in tiles_of(iid):
                tile_id = tile.get("tile_id")
                if tile_id in seen and seen[tile_id] != iid:
                    scoped.add(iid)
                    scoped.add(seen[tile_id])
                seen.setdefault(tile_id, iid)

    cousins: dict[str, list[str]] = {}
    for iid in order:
        caps = _caps(iid)
        if (
            (caps.get("barrier_task") or caps.get("consolidate_task"))
            and not caps.get("tile_task")
            and not _safe(iid)
        ):
            cousins.setdefault(modules.get(iid, iid), []).append(iid)
    for module_id, iids in cousins.items():
        if len(iids) < 2:
            continue
        names = {iid: _barrier_output_names(_caps(iid)) for iid in iids}
        for pos, left in enumerate(iids):
            for right in iids[pos + 1:]:
                if (left, right) in chained:
                    continue
                left_names, right_names = names[left], names[right]
                if (
                    left_names is not None
                    and right_names is not None
                    and left_names.isdisjoint(right_names)
                ):
                    continue
                scoped.add(left)
                scoped.add(right)
    return {iid: iid for iid in scoped}


__all__ = [
    "Task",
    "Edge",
    "discover_node_capabilities",
    "effective_node_caps",
    "get_node_metadata",
    "compile_pipeline",
    "build_dag",
    "build_ctx",
]
