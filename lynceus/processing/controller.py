# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Pipeline orchestration, session management, and execution control.

This module is Qt-free. It receives the typed node graph and source paths,
then reports events through callbacks invoked by the worker thread. The canvas
adapts those callbacks to queued Qt signals.

Responsibilities:
- complete execution: tiling followed by the per-tile DAG
- cancellation, pause, and resume
- session persistence for tiles, completed nodes, and barrier outputs
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable

from lynceus.nodes.ports import get_display_name, port_metadata
from lynceus.nodes._variants import effective_specs, resolved_ports
from lynceus.plugins.locale import t as translate
from lynceus.plugins.registry import manager
from lynceus.processing.acceleration import (
    VALID_BACKENDS,
    VALID_MODES,
    resolve_acceleration,
)
from lynceus.processing.executor import create_pool, run_dag
from lynceus.processing import provenance as prov
from lynceus.processing.raster import RASTER_CELL_SIZE
from lynceus.processing.steps import (
    _port_id,
    build_ctx,
    build_dag,
    compile_pipeline,
    discover_node_capabilities,
    effective_node_caps,
)
from lynceus.processing.tiler import LiDARTiler, default_memory_budget_bytes

logger = logging.getLogger(__name__)


def _module_source_files(module_id: str, config, registry=None) -> list[str]:
    """Short hashes of the node source file(s) for fingerprints.

    Editing a `.py` changes the fingerprint, so fixed code never reuses
    products computed by older code. Variant strategies resolve their own
    file too. Unresolvable modules (synthetic test ids) contribute
    nothing, preserving legacy fingerprints exactly.
    """
    from lynceus.plugins.registry import _module_spec_origin, manager

    registry = registry if registry is not None else manager
    paths: list[Path] = []
    try:
        info = registry.node_info(module_id)
    except Exception:
        info = None
    if info is not None and info.file_path:
        paths.append(Path(info.file_path))
    else:
        origin = _module_spec_origin(module_id)
        if origin is not None:
            paths.append(origin)
    if isinstance(config, dict):
        strategy = config.get("strategy")
        if isinstance(strategy, str) and strategy:
            try:
                variant = registry.variant_info(module_id, strategy)
            except Exception:
                variant = None
            variant_path = (variant or {}).get("path")
            if variant_path:
                paths.append(Path(variant_path))
    digests: list[str] = []
    for path in paths:
        try:
            digest = hashlib.sha1(path.read_bytes()).hexdigest()[:12]
        except OSError:
            continue
        if digest not in digests:
            digests.append(digest)
    return sorted(digests)


def default_output_root() -> Path:
    """Return the absolute, CWD-independent session root.

    The panel, the controller and session deletion must agree on a single
    location even when the app is launched from a different working directory;
    a CWD-relative ``output`` made sessions silently unreachable across
    launches (no reuse) and let the panel "delete" a folder that was really a
    different one under Explorer. Anchor it to the package root.

    Frozen (installer) builds anchor it to the non-roaming user data dir
    instead: sessions must never land inside the install dir (read-only
    under Program Files, wiped on update, and left behind on uninstall).
    """
    import sys

    if getattr(sys, "frozen", False):
        from lynceus.plugins.store import local_data_root

        return local_data_root() / "output"
    return Path(__file__).resolve().parents[2] / "output"


def _noop_barrier(payload: dict) -> dict:
    """Return a cached barrier payload without recomputing the node.

    Cached files have already been materialized in the current session, so the
    replaced barrier only republishes its previous payload.
    """
    return payload


def _safe_dir_name(name: str) -> str:
    """Build a filesystem-safe directory name from a source filename.

    Separators and invalid characters are replaced while preserving readable
    names, for example ``My File (1).laz -> my_file_1``.
    """
    name = os.path.splitext(name)[0]
    cleaned = re.sub(r"[^A-Za-z0-9_.\-]+", "_", name)
    cleaned = re.sub(r"_{2,}", "_", cleaned).strip("._")
    return cleaned.lower() or "source"


class PipelineCallbacks:
    """Optional callbacks invoked from the worker thread.

    ``on_message(text, kind)`` reports progress, state, warnings, or errors.
    ``kind`` is one of ``info``, ``action``, ``success``, ``warning``, or
    ``error``. Other callbacks report node status, tile counts, completion,
    progress, cancellation, per-segment starts/outputs and the final
    consolidated outputs.
    """

    def __init__(
        self,
        on_message: Callable[[str, str], None] | None = None,
        on_node_status: Callable[[str, str], None] | None = None,
        on_tile_count: Callable[[int], None] | None = None,
        on_finished: Callable[[dict, list, int, list, dict], None] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
        on_cancelled: Callable[[], None] | None = None,
        on_segment_started: Callable[[int, int, int], None] | None = None,
        on_segment_outputs: Callable[[int, int, dict], None] | None = None,
        on_consolidated_outputs: Callable[[dict], None] | None = None,
    ):
        self.on_message = on_message
        self.on_node_status = on_node_status
        self.on_tile_count = on_tile_count
        self.on_cancelled = on_cancelled
        self.on_finished = on_finished
        self.on_progress = on_progress
        self.on_segment_started = on_segment_started
        self.on_segment_outputs = on_segment_outputs
        self.on_consolidated_outputs = on_consolidated_outputs


class PipelineController:
    """Qt-free pipeline orchestrator. Construct it on the GUI thread."""

    def __init__(self):
        self._pool = None
        self._state = "idle"  # idle | running | paused | cancelling
        self._cancel = threading.Event()
        self._pause = threading.Event()
        self._pause.set()
        # Optional gate called from the WORKER thread for each module id;
        # must NOT touch Qt. Returns None to allow, else the reason text.
        self.eula_checker: Callable[[str], str | None] | None = None

        self._session_dir: Path | None = None
        self._session_state: dict = {}
        self._tiles_seen = 0
        # Instance ID to module ID for the active run/session.
        self._modules: dict[str, str] = {}

        # Progress state for the active DAG.
        self._last_tasks = []
        self._node_task_total: dict[str, int] = {}
        self._node_task_done: dict[str, int] = {}
        self._stage_total: dict[str, int] = {}
        self._stage_done: dict[str, int] = {}
        self._node_failed: set[str] = set()
        # Metrics from the most recent run.
        self._run_start_iso: str = ""
        self.last_metrics: dict = {}

        # Nodes reused in the active run from previous-session artifacts.
        # Any role reused (tile and/or barrier). Role-specific sets below.
        self._reused_iids: set[str] = set()
        self._reused_outputs: dict[str, dict] = {}
        self._reused_tiles: set[str] = set()
        self._reused_barriers: set[str] = set()
        self._reused_prev: dict[str, tuple] = {}
        # Intermediate point-cloud extension for the active run (P2a).
        self._run_intermediate_ext: str = "laz"
        # Performance preferences for the active run (Preferences >
        # Performance). Never part of node fingerprints (same outputs).
        self._run_max_workers: int | None = None
        self._run_memory_budget_bytes: int | None = None
        # Operator attribution for the active run (Preferences > Operator).
        # Unlike perf settings this DOES salt fingerprints (output bytes).
        self._run_operator: dict | None = None
        # Public policy for the future UI toggle: auto | on | off.
        self.acceleration_mode = "auto"
        self.acceleration_backend = "auto"
        self.acceleration_device_id = ""
        self._acceleration = resolve_acceleration(
            self.acceleration_mode,
            self.acceleration_backend,
            self.acceleration_device_id,
        )

    # ------------------------------------------------------------------
    # Public API (GUI thread).
    # ------------------------------------------------------------------

    @property
    def state(self) -> str:
        return self._state

    @property
    def session_dir(self) -> Path | None:
        return self._session_dir

    @property
    def session_state(self) -> dict:
        return self._session_state

    def set_acceleration_mode(self, mode: str) -> None:
        """Set the optional numerical acceleration policy for the next run."""
        if mode not in VALID_MODES:
            raise ValueError("acceleration mode must be 'auto', 'on' or 'off'")
        self.acceleration_mode = mode

    def set_acceleration_backend(
        self, backend: str = "auto", device_id: str = ""
    ) -> None:
        """Select a backend/device for the next run (Preferences UI)."""
        if backend not in VALID_BACKENDS:
            raise ValueError(
                "acceleration backend must be auto, cpu, cuda, opencl or directml"
            )
        self.acceleration_backend = backend
        self.acceleration_device_id = device_id or ""

    def run(
        self,
        nodes: list,
        edges: list,
        source_paths: dict[str, str],
        cb: PipelineCallbacks,
        session_root=None,
        configs: dict | None = None,
        max_workers: int | None = None,
        memory_budget_bytes: int | None = None,
        operator: dict | None = None,
        keep_intermediates: bool = False,
    ) -> None:
        """Start a complete pipeline run in a background worker.

        ``nodes`` contains ``(instance_id, module_id)`` pairs. ``source_paths``
        maps every LiDAR loader instance to its file; an empty mapping denotes
        a barrier-only graph. ``configs`` contains per-instance overrides, and
        ``session_root`` selects the parent directory for this run.
        ``max_workers``/``memory_budget_bytes`` are performance preferences
        (Preferences > Performance): they never enter node fingerprints.
        ``operator`` (``{"name", "org"}``) is embedded in provenance sidecars
        and salts fingerprints (attribution changes recompute).
        ``keep_intermediates`` (Preferences, default off) keeps per-tile
        intermediates and the shared tile cache so identical re-runs reuse
        previous sessions. Off: the session keeps only final products and
        every run re-tiles and recomputes from scratch. The flag never
        enters fingerprints (it changes which files remain, not bytes).
        """
        self._run_max_workers = max_workers
        self._run_memory_budget_bytes = memory_budget_bytes
        self._run_operator = operator
        self._run_keep_intermediates = bool(keep_intermediates)
        self._begin_run()
        threading.Thread(
            target=self._run_worker,
            args=(
                nodes, edges, source_paths, cb, session_root, configs,
                max_workers, memory_budget_bytes, keep_intermediates,
            ),
            daemon=True,
        ).start()

    def cancel(self) -> None:
        if self._state not in ("running", "paused"):
            return
        self._state = "cancelling"
        self._cancel.set()
        self._pause.set()  # Release a paused worker.

    def pause(self) -> None:
        if self._state != "running":
            return
        self._state = "paused"
        self._pause.clear()

    def resume(self) -> None:
        if self._state != "paused":
            return
        self._state = "running"
        self._pause.set()

    # ------------------------------------------------------------------
    # Startup / cleanup
    # ------------------------------------------------------------------

    def _begin_run(self) -> None:
        self._state = "running"
        self._cancel.clear()
        self._pause.set()
        self._run_start_iso = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        self._tilings = {}
        self._run_flag_counts = {}
        self._run_density_counts = {}
        self._pool = create_pool(self._run_max_workers)  # Multiprocessing pools belong on the main thread.

    def _cleanup(self, cb: PipelineCallbacks) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None
        self._cancel.clear()
        self._pause.set()
        if self._state == "cancelling":
            self._state = "idle"
            if cb.on_message:
                cb.on_message("Pipeline cancelled", "success")
            if cb.on_cancelled:
                cb.on_cancelled()
        else:
            self._state = "idle"

    def _module_label(self, iid: str) -> str:
        """Return a readable instance label using the module short name."""
        return self._modules.get(iid, iid).rsplit(".", 1)[-1]

    def _node_display_name(self, iid: str) -> str:
        """Return NODE_NAME with a module-name fallback."""
        module_id = self._modules.get(iid, iid)
        try:
            info = manager.node_info(module_id)
            if info is not None and info.name:
                return info.name
        except Exception:
            pass
        return self._module_label(iid)

    def _fail_all(self, cb: PipelineCallbacks) -> None:
        """Mark nodes with pending work as failed after a pipeline error."""
        if cb.on_node_status:
            for node_id in list(self._node_task_total):
                if node_id not in self._node_failed:
                    cb.on_node_status(node_id, "error")

    # ------------------------------------------------------------------
    # Session management.
    # ------------------------------------------------------------------

    def _new_session_dir(self, session_root=None) -> Path:
        base = Path(session_root) if session_root else default_output_root()
        return base / datetime.now().strftime("%Y%m%d_%H%M%S")

    def _session_root_name(self, trigger_iids, source_paths) -> str:
        """Return a stable session root name for the run sources.

        Single-source runs retain the readable filename for cache compatibility.
        Multi-source runs use stable source tags: ``multi_<tag1>_<tag2>``.
        """
        if len(trigger_iids) == 1:
            return _safe_dir_name(Path(source_paths[trigger_iids[0]]).name)
        tags = []
        for iid in trigger_iids:
            fp = self._source_fingerprint(source_paths[iid])
            tags.append(hashlib.sha1(fp.encode()).hexdigest()[:8])
        return "multi_" + "_".join(tags)

    # ------------------------------------------------------------------
    # Session cache: per-run artifacts and cross-session reuse.
    #
    # Each run writes to its own artifact directory. Node fingerprints include
    # upstream fingerprints, so unchanged products can be hard-linked from a
    # previous session without overwriting either run.
    #
    # Tiled source data is shared through cache/tiles/<src_fp>/ while unchanged.
    # ------------------------------------------------------------------

    def _source_fingerprint(self, source_path: str) -> str:
        """Return a stable source fingerprint based on size and mtime."""
        st = Path(source_path).stat()
        return hashlib.sha1(
            f"{st.st_size}:{st.st_mtime}".encode()
        ).hexdigest()[:16]

    def _choose_intermediate_format(self, source_paths: dict | None) -> str:
        """Resolve the intermediate point-cloud extension for this run (P2a)."""
        from lynceus.processing.stream_format import (
            choose_intermediate_format,
            format_enabled,
        )

        override = format_enabled()
        if override:
            logger.info(f"[Controller] Intermediate format forced to '{override}'")
            return override
        total_points = 0
        import laspy

        for iid, path in (source_paths or {}).items():
            if not path:
                continue
            try:
                with laspy.open(str(path)) as reader:
                    total_points += int(reader.header.point_count)
            except Exception:
                continue
        fmt = choose_intermediate_format(total_points=total_points)
        logger.info(
            f"[Controller] Intermediate format '{fmt}' "
            f"({total_points:,} source points)"
        )
        return fmt

    def _cache_root(self, session_root=None) -> Path:
        base = Path(session_root) if session_root else default_output_root()
        return base / "cache"

    def _run_tiles_dir_for(
        self, session_root, source_fp: str, fmt: str = "laz"
    ) -> Path:
        """Per-source tile directory for this run.

        With intermediates kept, tiles live in the shared ``cache/tiles/``
        tree so identical re-runs skip re-tiling. Otherwise they go to a
        per-run temp dir (deleted by the end-of-run sweep, always): every
        run re-tiles from the source, and no tile cache accumulates.
        """
        if getattr(self, "_run_keep_intermediates", True):
            tiles_dir = self._tiles_dir(session_root, source_fp, fmt)
        else:
            if self._run_tiles_tmp is None:
                base = (
                    Path(session_root)
                    if session_root
                    else default_output_root()
                )
                base.mkdir(parents=True, exist_ok=True)
                self._run_tiles_tmp = Path(
                    tempfile.mkdtemp(prefix="_tiles_tmp_", dir=str(base))
                )
            leaf = (
                source_fp
                if not fmt or fmt == "laz"
                else f"{source_fp}_{fmt}"
            )
            tiles_dir = self._run_tiles_tmp / leaf
        tiles_dir.mkdir(parents=True, exist_ok=True)
        return tiles_dir

    def _tiles_dir(self, session_root, source_fp: str, fmt: str = "laz") -> Path:
        base = self._cache_root(session_root) / "tiles" / source_fp
        # Legacy tiled caches are LAZ under <fp>/; LAS runs get their own
        # <fp>_las/ so both formats coexist and never cross-contaminate.
        if fmt and fmt != "laz":
            return self._cache_root(session_root) / "tiles" / f"{source_fp}_{fmt}"
        return base

    def _artifacts_root(self, session_dir: Path) -> Path:
        """Return the self-contained artifact root for one session."""
        return session_dir / "artifacts"

    def _provenance_path(self, session_dir: Path) -> Path:
        return session_dir / "provenance.json"

    def _node_caps(self, iid: str, configs: dict | None = None) -> dict:
        """Per-instance capabilities (strategy-aware) for a canvas node."""
        module_id = self._modules.get(iid, iid)
        return effective_node_caps(module_id, (configs or {}).get(iid))

    def _module_bundle_globs(self, module_id: str, config: dict | None = None):
        """Return artifact-relative output globs declared by a node.

        Built-ins use historical fallbacks; extensions may declare
        ``PROCESSING_SPECS["output_globs"]`` for session reuse. Intermediate
        point-cloud streams (``/**/*.laz``) switch to the run's format so
        provenance reuse tracks the tiles that were actually written.
        """

        def _normalize(globs: tuple) -> tuple:
            fmt = getattr(self, "_run_intermediate_ext", "laz")
            if fmt != "las":
                return globs
            return tuple(
                g.replace("/**/*.laz", "/**/*.las")
                for g in globs
            )

        try:
            specs = effective_specs(module_id, config)
        except Exception:
            specs = {}
        globs = specs.get("output_globs")
        if isinstance(globs, (list, tuple)) and globs:
            return _normalize(tuple(globs))
        name = module_id.rsplit(".", 1)[-1]
        if name == "classify_ground":
            return _normalize(("classified/**/*.laz",))
        if name == "generate_dtm":
            return ("dtm/**/*.tif", "dtm_mosaic.tif")
        if name == "generate_dsm":
            return ("dsm/**/*.tif", "dsm_mosaic.tif")
        if name == "generate_chm":
            return ("chm_mosaic.tif",)
        # Forestry extensions declare their own output globs.
        return ()

    def _node_fingerprints(
        self,
        iids: list[str],
        edges: list,
        configs: dict | None,
        source_fps: dict[str, str],
        scope: str = "",
        seg_sig: str = "",
    ) -> dict[str, str]:
        """Compute per-node fingerprints in topological order.

        Each fingerprint hashes the module, instance configuration, and
        upstream fingerprints. Direct loader inputs add their source
        fingerprint, so upstream changes invalidate downstream reuse.
        Module source bytes hash in too (`modsrc:`): editing a node's
        `.py` (or its active strategy variant file) always recomputes
        instead of reusing products from older code.
        ``scope`` (e.g. ``seg0000:ab12cd34``) salts every fingerprint with the
        per-batch tile content: two batches of a segmented run hash differently
        unless their tile sets match exactly, so provenance reuse never
        cross-links batches with different tiles.

        ``seg_sig`` salts the final consolidation session: a consolidator's
        fingerprint must change when the underlying segments change, so a
        stale merged mosaic is never reused across runs with different batches.

        A non-empty operator identity (Preferences > Operator) salts every
        fingerprint as well: attribution lands in output bytes, so changing
        it recomputes instead of reusing sidecars stamped by someone else.
        """
        order = compile_pipeline(iids, edges)
        input_edges_by_node: dict[str, list] = {}
        for edge in edges:
            input_edges_by_node.setdefault(edge[1], []).append(edge)

        fps: dict[str, str] = {}
        for iid in order:
            module_id = self._modules.get(iid, iid)
            comp = [
                module_id,
                json.dumps(configs.get(iid, {}) if configs else {}, sort_keys=True),
            ]
            for src_file in _module_source_files(
                module_id,
                (configs or {}).get(iid),
                getattr(self, "_registry", None),
            ):
                comp.append(f"modsrc:{src_file}")
            for src, _dst, _out, _in in input_edges_by_node.get(iid, []):
                if src in source_fps:
                    comp.append(f"src:{source_fps[src]}")
                elif src in fps:
                    comp.append(f"{src}:{fps[src]}")
            if scope:
                comp.append(f"scope:{scope}")
            if seg_sig:
                comp.append(f"segs:{seg_sig}")
            operator = getattr(self, "_run_operator", None) or {}
            op_sig = f"{operator.get('name', '')}|{operator.get('org', '')}"
            if op_sig.strip("|"):
                comp.append(f"operator:{op_sig}")
            raw = "\n".join(comp)
            fps[iid] = hashlib.sha1(raw.encode()).hexdigest()[:16]
        return fps

    def _scan_prov_dir(self, entry: Path, results: list) -> None:
        """Append ``(dir, provenance)`` for a completed mini-session dir."""
        meta_file = entry / "session_meta.json"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                status = meta.get("pipeline_status")
            except Exception:
                status = None
            if status not in (None, "completed"):
                return
        prov_file = self._provenance_path(entry)
        if not prov_file.exists():
            return
        try:
            data = json.loads(prov_file.read_text(encoding="utf-8"))
        except Exception:
            return
        if isinstance(data, dict) and isinstance(data.get("nodes"), dict):
            results.append((entry, data))

    def _prior_provenances(self, session_root, exclude_session: Path):
        """Return previous-session provenance records, newest first.

        Each session root holds one directory per run; a segmented run nests
        one directory per batch under ``lotes/`` (each is an independent
        mini-session with its own provenance). Every fully completed run or
        batch is a reuse candidate. Sessions whose session_meta.json marks a
        cancelled/failed/partial run are skipped so a broken run never feeds
        the cascade cache.
        """
        base = Path(session_root) if session_root else default_output_root()
        results = []
        if not base.exists():
            return results
        for entry in base.iterdir():
            if not entry.is_dir() or entry == exclude_session:
                continue
            if not entry.name[:1].isdigit():
                continue
            self._scan_prov_dir(entry, results)
            lotes = entry / "lotes"
            if lotes.is_dir():
                for seg in sorted(lotes.iterdir()):
                    if seg.is_dir():
                        self._scan_prov_dir(seg, results)
            final = entry / "final"
            if final.is_dir():
                self._scan_prov_dir(final, results)
        results.sort(key=lambda kv: kv[0].name, reverse=True)
        return results

    def _find_reusable(
        self,
        candidates: list,
        consumed: set[int],
        module_id: str,
        fp: str,
    ):
        """Find a previous node with the same module and fingerprint.

        ``consumed`` prevents two same-fingerprint instances from claiming the
        same previous entry. Completeness is NOT checked here: the reverse
        fixpoint (``_effective_reuse``) decides per role whether missing
        files matter. Return ``(session_dir, entry)`` or ``(None, None)``.
        """
        for key, (prev_dir, entry) in enumerate(candidates):
            if key in consumed:
                continue
            if isinstance(entry, dict) and entry.get("module") == module_id:
                if entry.get("fp") == fp:
                    consumed.add(key)
                    return prev_dir, entry
        return None, None

    @staticmethod
    def _bundle_complete(prev_dir, entry: dict) -> bool:
        """True when every recorded bundle file still exists upstream.

        Entries with no recorded files (metadata-only) count as complete.
        """
        files = entry.get("files", []) if isinstance(entry, dict) else []
        if not files:
            return True
        prev_arts = Path(prev_dir) / "artifacts"
        return all((prev_arts / str(rel)).is_file() for rel in files)

    @staticmethod
    def _rebase_payload(payload: dict, prev_dir, artifacts: Path) -> dict:
        """Rewrite republished absolute paths to the current session.

        Reused payloads were recorded with the previous session's absolute
        paths; without rebasing, the gallery, QML styling and downstream
        readers would touch the old session (or a deleted one). Only
        paths under the previous artifacts root are rewritten; tile-cache
        and external paths are shared by design and stay as-is.
        """
        prev_root = str(Path(prev_dir) / "artifacts")
        new_root = str(artifacts)

        def _rebase(value):
            if isinstance(value, str):
                if value == prev_root or value.startswith(prev_root + "/") or value.startswith(prev_root + "\\"):
                    return new_root + value[len(prev_root):]
                return value
            if isinstance(value, list):
                return [_rebase(item) for item in value]
            if isinstance(value, tuple):
                return tuple(_rebase(item) for item in value)
            if isinstance(value, dict):
                return {key: _rebase(item) for key, item in value.items()}
            return value

        if not isinstance(payload, dict):
            return payload
        return {key: _rebase(value) for key, value in payload.items()}

    # Barrier input types that read provider tile streams (not merged
    # finals). Unknown/custom types default to merged: declared custom
    # ports carry viewer kinds, i.e. products, in practice.
    _STREAM_IN_TYPES = frozenset({
        "point_cloud",
        "classified_point_cloud",
        "tiles",
        "dtm_tile",
        "dsm_tile",
    })

    @staticmethod
    def _entry_role_files(caps: dict, files: list) -> tuple[list, list]:
        """Split recorded bundle files into barrier-need vs tile-need.

        Barrier/merge finals live at the artifacts root (depth 1);
        tile-task streams and per-tile outputs live deeper. Pure roles
        take the whole bundle on their side.
        """
        files = [str(rel) for rel in files or []]
        has_tile = bool((caps or {}).get("tile_task"))
        has_final = bool(
            (caps or {}).get("barrier_task")
            or (caps or {}).get("consolidate_task")
            or (caps or {}).get("input_file_key")
        )
        if has_final and not has_tile:
            return files, []
        if has_tile and not has_final:
            return [], files
        if has_tile and has_final:
            finals = set((caps.get("output_files") or {}).values())
            if caps.get("session_file"):
                finals.add(caps["session_file"])
            barrier = [
                rel for rel in files
                if len(Path(rel).parts) <= 1
                or (
                    len(Path(rel).parts) == 2
                    and Path(rel).parts[1] in finals
                )
            ]
            tile = [rel for rel in files if rel not in barrier]
            return barrier, tile
        return [], []

    def _effective_reuse(
        self, order: list, edges: list, matched: dict, configs: dict | None
    ) -> tuple[set, set]:
        """Decide effective reuse per role, consumers-first (reverse topo).

        A barrier role reuses iff its finals exist upstream (the gallery
        needs real files). A tile role reuses iff its files exist upstream
        or every role reading them reuses: its own barrier (merges its
        partials), downstream tile roles (read provider streams) and
        downstream barrier roles over stream inputs. Everything else
        recomputes. Returns ``(tile_reuse, barrier_reuse)`` iid sets.
        """
        configs = configs or {}
        has_tile: dict[str, bool] = {}
        has_barrier: dict[str, bool] = {}
        barrier_need: dict[str, list] = {}
        tile_need: dict[str, list] = {}
        prev_of: dict[str, object] = {}
        for iid, (prev_dir, entry) in matched.items():
            try:
                caps = self._node_caps(iid, configs)
            except Exception:
                caps = {}
            needs = self._entry_role_files(
                caps,
                entry.get("files", []) if isinstance(entry, dict) else [],
            )
            has_barrier[iid] = bool(
                caps.get("barrier_task") or caps.get("consolidate_task")
            )
            has_tile[iid] = bool(caps.get("tile_task"))
            barrier_need[iid], tile_need[iid] = needs
            prev_of[iid] = prev_dir

        def _present(iid: str, rels: list) -> bool:
            arts = Path(prev_of[iid]) / "artifacts"
            return all((arts / rel).is_file() for rel in rels)

        def _barrier_present(iid: str) -> bool:
            """Barrier finals exist upstream, recorded or advertised.

            Recorded files must all be there; additionally, any payload
            path advertised under the previous artifacts root must exist
            (an unrecorded product the barrier claims still has to be
            real, or the republish would be a ghost).
            """
            if not _present(iid, barrier_need[iid]):
                return False
            _prev_dir, entry = matched[iid]
            output = entry.get("output") if isinstance(entry, dict) else None
            if not isinstance(output, dict):
                return True
            prev_arts = Path(_prev_dir) / "artifacts"
            for value in output.values():
                if not isinstance(value, str) or not value:
                    continue
                try:
                    rel = Path(value).relative_to(prev_arts)
                except ValueError:
                    continue
                if not (prev_arts / rel).is_file():
                    return False
            return True

        barrier_reuse = {
            iid
            for iid in matched
            if has_barrier[iid] and _barrier_present(iid)
        }

        downstream: dict[str, list] = {}
        for src, dst, _out, in_t in edges:
            try:
                in_id = _port_id(in_t)
            except Exception:
                in_id = str(in_t)
            downstream.setdefault(src, []).append((dst, in_id))

        tile_reuse: set[str] = set()
        for iid in reversed(order):
            if iid not in matched or not has_tile[iid]:
                continue
            if _present(iid, tile_need[iid]):
                tile_reuse.add(iid)
                continue
            readers_ok = True
            if has_barrier[iid] and iid not in barrier_reuse:
                readers_ok = False
            if readers_ok:
                for dst, in_id in downstream.get(iid, []):
                    if dst not in matched:
                        readers_ok = False
                        break
                    if has_tile.get(dst) and dst not in tile_reuse:
                        readers_ok = False
                        break
                    if (
                        has_barrier.get(dst)
                        and dst not in barrier_reuse
                        and in_id in self._STREAM_IN_TYPES
                    ):
                        readers_ok = False
                        break
            if readers_ok:
                tile_reuse.add(iid)
        return tile_reuse, barrier_reuse

    def _materialize_bundle(
        self, prev_dir: Path, artifacts: Path, entry: dict
    ) -> list[str]:
        """Materialize a node's products from a previous session.

        Return the relative paths materialized under ``artifacts``.
        """
        rel_paths = entry.get("files", [])
        if not rel_paths:
            return []
        prev_arts = prev_dir / "artifacts"
        done = []
        for rel in rel_paths:
            src = prev_arts / rel
            dst = artifacts / rel
            if not src.is_file():
                continue
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    dst.unlink()
                os.link(str(src), str(dst))
                done.append(rel)
            except OSError:
                try:
                    if not dst.exists():
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        dst.write_bytes(src.read_bytes())
                        done.append(rel)
                except OSError:
                    pass
        return done

    def _provenance_bundle_files(
        self,
        artifacts: Path,
        module_id: str,
        config: dict | None,
        iid: str,
        exclude_iids: set[str] | frozenset[str] = frozenset(),
    ) -> list[str]:
        """Artifact files recorded for a node in provenance.json.

        Skipped (effectively reused) roles write nothing new, so the disk
        glob would record an empty bundle and break future reuse chains;
        carry over the previous entry's file list instead (future
        completeness checks fall through correctly when files are absent
        here too).
        """
        files = self._module_bundle_files(
            artifacts, module_id, config, iid, exclude_iids
        )
        if not files and iid in getattr(self, "_reused_iids", set()):
            prev = (getattr(self, "_reused_prev", None) or {}).get(iid)
            if prev:
                _prev_dir, prev_entry = prev
                files = [
                    str(rel)
                    for rel in (prev_entry.get("files", []) or [])
                ]
        return files

    def _module_bundle_files(
        self, artifacts: Path, module_id: str, config: dict | None = None,
        iid: str | None = None,
        exclude_iids: set[str] | frozenset[str] = frozenset(),
    ) -> list[str]:
        """List artifact files emitted by a node for the provenance manifest.

        Branch-scoped instances (F4.2) write tile outputs under
        ``<kind>/<scope>/`` and barrier finals under ``<scope>/`` with
        ``scope`` equal to their own instance id, so the declared globs are
        also tried with the scope inserted (after the first segment for
        tile globs, as a prefix for bare finals). Unscoped instances never
        create ``<iid>/`` dirs (except product inputs, excluded from reuse),
        so the extra patterns match nothing for them.

        Files attributable to a *foreign* instance scope (``exclude_iids``)
        are dropped: materializing another instance's paths would trip the
        writers' exists-skip and freeze stale products on recompute.
        """
        rels = []
        globs = list(self._module_bundle_globs(module_id, config))
        if iid:
            extra = []
            for glob in globs:
                if "/" in glob:
                    head, tail = glob.split("/", 1)
                    extra.append(f"{head}/{iid}/{tail}")
                else:
                    extra.append(f"{iid}/{glob}")
            globs.extend(extra)
        for glob in globs:
            for p in artifacts.glob(glob):
                if p.is_file():
                    rels.append(str(p.relative_to(artifacts)).replace("\\", "/"))
        foreign = set(exclude_iids or ()) - ({iid} if iid else set())
        if foreign:
            kept = []
            for rel in rels:
                parts = Path(rel).parts
                if parts[0] in foreign:
                    continue
                if len(parts) > 2 and parts[1] in foreign:
                    continue
                kept.append(rel)
            rels = kept
        return sorted(set(rels))

    def _save_provenance(
        self,
        session_dir: Path,
        source_fps: dict[str, str] | str,
        tiles_dir: Path,
        fps: dict[str, str],
        order: list[str],
        tiles: list,
        node_outputs: dict | None = None,
        configs: dict | None = None,
    ) -> dict[str, dict]:
        """Persists provenance.json with the fingerprint and products of each node."""
        node_outputs = node_outputs or {}
        configs = configs or {}
        artifacts = self._artifacts_root(session_dir)
        nodes = {}
        for iid in order:
            module_id = self._modules.get(iid, iid)
            caps = self._node_caps(iid, configs)
            if not (
                caps.get("tile_task")
                or caps.get("barrier_task")
                or caps.get("consolidate_task")
            ):
                continue
            nodes[iid] = {
                "module": module_id,
                "fp": fps.get(iid, ""),
                "files": self._provenance_bundle_files(
                    artifacts, module_id, configs.get(iid), iid,
                    set(order),
                ),
                "output": node_outputs.get(iid),
            }
            if (
                nodes[iid]["output"] is not None
                and not nodes[iid]["files"]
                and caps.get("output_globs")
                and not caps.get("tile_task")
                and (
                    caps.get("barrier_task")
                    or caps.get("consolidate_task")
                    or caps.get("input_file_key")
                )
            ):
                self._warn_glob_mismatch(module_id, caps["output_globs"])
        src_fps = source_fps if isinstance(source_fps, dict) else {"": source_fps}
        data = {
            "source_fp": next(iter(src_fps.values()), ""),
            "source_fps": src_fps,
            "tiles_dir": str(tiles_dir),
            "tiles": [dict(t) for t in tiles],
            "nodes": nodes,
        }
        session_dir.mkdir(parents=True, exist_ok=True)
        prov_file = self._provenance_path(session_dir)
        prov_file.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return nodes

    def _warn_glob_mismatch(self, module_id: str, globs) -> None:
        """Warn when declared output globs matched nothing written.

        A glob that matches no file silently disables reuse for the node
        (usually a basename typo or a product saved under another name).
        Informational only: never fails the run.
        """
        callback = getattr(self, "on_message", None)
        if not callable(callback):
            return
        callback(
            f"{module_id} declares output_globs that matched no written "
            "files; reuse stays disabled for it (check product basenames)",
            "warning",
        )

    def _save_session_state(self) -> None:
        if not self._session_dir:
            return
        self._session_state.setdefault("tiles", [])
        self._session_state.setdefault("completed_nodes", [])
        self._session_state.setdefault("failed_nodes", [])
        self._session_state.setdefault("node_outputs", {})
        self._session_state.setdefault("pipeline_status", "running")
        self._session_dir.mkdir(parents=True, exist_ok=True)
        (self._session_dir / "session_state.json").write_text(
            json.dumps(self._session_state, indent=2), encoding="utf-8"
        )

    def _write_results(self, tile_results: list) -> None:
        """Persist per-tile results ledger (work actually executed).

        Skipped tile tasks (effectively reused roles) are omitted by
        design; reuse is counted in session state and the QA report.
        """
        if not self._session_dir:
            return
        output_path = self._session_dir / "results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(tile_results, indent=2), encoding="utf-8"
        )

    def _save_run_metrics(self, metrics: dict) -> None:
        """Appends run_metrics to the existing session_meta.json."""
        if not self._session_dir:
            return
        meta_file = self._session_dir / "session_meta.json"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        else:
            meta = {}
        meta["run_metrics"] = metrics
        meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Finals-only sweep (intermediates off).
    # ------------------------------------------------------------------

    def _sweep_intermediates(self) -> None:
        """Delete per-run intermediates, keeping only final products.

        The temp tile dir always goes (even on cancel/error, even when
        intermediates are kept: it is per-run by construction). With
        intermediates kept this is all it does. Otherwise every
        mini-session of this run (main + segments) loses its tile-task
        files; barrier/merge products, their sidecars and the session
        JSONs stay. Conservative: a mini-session without provenance.json
        is left untouched, and failures only log (the session stays
        usable, just heavier).
        """
        tmp = getattr(self, "_run_tiles_tmp", None)
        if tmp is not None:
            self._run_tiles_tmp = None
            shutil.rmtree(tmp, ignore_errors=True)
        if getattr(self, "_run_keep_intermediates", True):
            return
        session_dir = getattr(self, "_run_main_session_dir", None)
        if session_dir is None:
            return
        session_dir = Path(session_dir)
        targets = [session_dir]
        lotes = session_dir / "lotes"
        if lotes.is_dir():
            targets.extend(sorted(p for p in lotes.iterdir() if p.is_dir()))
        for target in targets:
            try:
                self._sweep_session(target)
            except Exception as exc:
                logger.warning(
                    "Intermediate sweep skipped %s: %s", target, exc
                )

    def _sweep_session(self, session_dir: Path) -> int:
        """Delete intermediate files of one mini-session; return the count.

        Finals rule (location + capabilities, belt and braces):
        - barrier/merge products live at the artifacts root (depth 1):
          mosaics, merged vectors/tables, CSV twins and their sidecars;
        - input copies live one level down under their instance id
          (``artifacts/<iid>/<file>``);
        - files recorded for pure barrier/merge/input entries are kept
          wherever they are (defensive: a barrier final nested deeper,
          e.g. the export LAZ under ``point_cloud_exports/<iid>/``).
        Everything else (tile-task streams, per-tile outputs, undeclared
        intermediates in subdirectories) is deleted. Session JSONs are
        never touched.
        """
        try:
            data = json.loads(
                (session_dir / "provenance.json").read_text(encoding="utf-8")
            )
            entries = data.get("nodes", {})
        except Exception:
            return 0
        if not isinstance(entries, dict):
            return 0
        configs = getattr(self, "_run_configs", {}) or {}
        entry_iids = set(entries)
        keep_rel: set[str] = set()
        for iid, entry in entries.items():
            if not isinstance(entry, dict):
                continue
            try:
                caps = effective_node_caps(
                    entry.get("module", ""), configs.get(iid)
                )
            except Exception:
                caps = {}
            if (
                caps.get("barrier_task")
                or caps.get("consolidate_task")
                or caps.get("input_file_key")
            ) and not caps.get("tile_task"):
                for rel in entry.get("files", []) or []:
                    keep_rel.add(str(rel))
        artifacts = session_dir / "artifacts"
        removed = 0
        if artifacts.is_dir():
            for path in sorted(artifacts.rglob("*")):
                if not path.is_file():
                    continue
                rel = path.relative_to(artifacts).as_posix()
                if rel in keep_rel:
                    continue  # pure barrier/merge/input product, anywhere
                parts = Path(rel).parts
                if len(parts) == 1:
                    continue  # barrier finals live at the artifacts root
                if len(parts) == 2 and parts[0] in entry_iids:
                    continue  # input copies live under their instance id
                try:
                    path.unlink()
                    removed += 1
                except OSError:
                    pass
            # Drop dirs left empty by the sweep (bottom-up, best-effort).
            for path in sorted(
                (p for p in artifacts.rglob("*") if p.is_dir()),
                key=lambda p: len(p.parts),
                reverse=True,
            ):
                try:
                    path.rmdir()
                except OSError:
                    pass
        if removed:
            logger.info(
                "Intermediate sweep: %d file(s) removed in %s",
                removed,
                session_dir,
            )
        return removed

    def _update_session_meta(self, status: str) -> None:
        """Record how the run ended in session_meta.json.

        Reuse excludes sessions that did not complete cleanly: the status
        written here ("completed" only when every node finished) is the gate
        ``_prior_provenances`` checks before offering a session to the cache.
        """
        if not self._session_dir:
            return
        meta_file = self._session_dir / "session_meta.json"
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        meta["pipeline_status"] = status
        meta_file.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    # ------------------------------------------------------------------
    # Full run
    # ------------------------------------------------------------------

    def _run_worker(
        self,
        nodes: list,
        edges: list,
        source_paths: dict[str, str],
        cb: PipelineCallbacks,
        session_root=None,
        configs: dict | None = None,
        max_workers: int | None = None,
        memory_budget_bytes: int | None = None,
        keep_intermediates: bool = False,
    ) -> None:
        try:
            self._tiles_seen = 0
            self._modules = dict(nodes)
            self._run_max_workers = max_workers
            self._run_memory_budget_bytes = memory_budget_bytes
            self._run_keep_intermediates = bool(keep_intermediates)
            self._run_configs = dict(configs or {})
            self._run_tiles_tmp: Path | None = None
            session_dir = None
            self._acceleration = resolve_acceleration(
                self.acceleration_mode,
                self.acceleration_backend,
                self.acceleration_device_id,
            )
            iids = list(self._modules)
            edges = list(edges)
            configs = dict(configs or {})

            # Compatibility: legacy scripts pass source_path (str) instead of
            # the dict. Kept (not dead): external callers/tests may use it.
            if isinstance(source_paths, str):
                _loader_iids = [
                    iid
                    for iid, mid in nodes
                    if discover_node_capabilities(mid).get("triggers_tiling")
                ]
                source_paths = (
                    {_loader_iids[0]: source_paths}
                    if _loader_iids
                    else {}
                )

            # Product inputs contribute external files as seeds. Store a
            # size+mtime fingerprint so reuse detects content changes even when
            # the selected path remains unchanged.
            for iid in iids:
                caps = self._node_caps(iid, configs)
                fkey = caps.get("input_file_key")
                if not fkey:
                    continue
                cfg = dict(configs.get(iid, {}))
                path = cfg.get(fkey) or ""
                src_fp = ""
                if path and os.path.isfile(path):
                    st = os.stat(path)
                    src_fp = hashlib.sha1(
                        f"{st.st_size}:{st.st_mtime}".encode()
                    ).hexdigest()[:16]
                cfg[f"__src_fp__:{fkey}"] = src_fp
                configs[iid] = cfg

            if not self._check_consent(cb):
                return

            if not self._check_available(cb):
                return

            if not self._check_required_ports(iids, edges, cb, configs):
                return

            tiling_plan = self._detect_tiling_plan(iids, edges, configs)
            tiling_triggers = list(tiling_plan)

            missing_source = [
                iid for iid in tiling_triggers if not source_paths.get(iid)
            ]
            if missing_source:
                names = ", ".join(
                    sorted(self._node_display_name(iid) for iid in missing_source)
                )
                if cb.on_node_status:
                    for iid in missing_source:
                        cb.on_node_status(iid, "error")
                if cb.on_message:
                    cb.on_message(
                        "LiDAR source file missing for: " + names + " — set "
                        "the file in each Load node before running",
                        "error",
                    )
                if cb.on_finished:
                    cb.on_finished({}, [], 0, [], {"blocked_by_source": True})
                self._state = "idle"
                return

            # Barrier-only runs use a stable _no_source root so reuse works
            # without a LiDAR file. Source runs use one session root per source.
            base = Path(session_root) if session_root else (
                default_output_root() / "_default_project"
            )

            if not tiling_triggers:
                # ---- Barrier-only branch: no tiling and no LiDAR source. ----
                session_crs = self._resolve_input_crs(configs, iids)
                self._run_crs_set = None
                session_root = base / "_no_source"
                source_fps: dict[str, str] = {}
                tiles_dir = None
                tiling = None
                tiles = []
                self._prov_blueprint = prov.session_blueprint(
                    session_root, [], self._run_operator
                )
                if cb.on_message:
                    cb.on_message("Importing inputs...", "info")
            else:
                # ---- Source branch: one tiling domain per loader. ----
                # Each source has its own tag, tile directory, and tile IDs;
                # the DAG distinguishes domains through the immediate provider.
                session_root = base / self._session_root_name(tiling_triggers, source_paths)

                # Progress is reported per read chunk (millions of messages on
                # a billion-point source); throttle it to integer-% changes so
                # the GUI is not flooded with layout work.
                last_tiling_pct: dict[str, int] = {}
                for iid in tiling_triggers:
                    if cb.on_node_status:
                        cb.on_node_status(iid, "processing")
                if cb.on_message:
                    cb.on_message("Tiling in progress...", "info")

                source_fps = {
                    iid: self._source_fingerprint(source_paths[iid])
                    for iid in tiling_triggers
                    if source_paths.get(iid)
                }
                self._prov_blueprint = prov.session_blueprint(
                    session_root,
                    [
                        {
                            "iid": iid,
                            "file": source_paths.get(iid, ""),
                            "source_fp": source_fps.get(iid, ""),
                            "tag": (
                                hashlib.sha1(
                                    source_fps[iid].encode()
                                ).hexdigest()[:8]
                                if iid in source_fps
                                else ""
                            ),
                        }
                        for iid in tiling_triggers
                    ],
                    self._run_operator,
                )

                all_tiles: list[dict] = []
                first_crs = None
                intermediate_fmt = self._choose_intermediate_format(
                    {iid: source_paths.get(iid) for iid in tiling_triggers}
                )
                self._run_intermediate_ext = intermediate_fmt
                for iid in tiling_triggers:
                    source_path = source_paths.get(iid)
                    if not source_path:
                        if cb.on_node_status:
                            cb.on_node_status(iid, "error")
                        if cb.on_message:
                            cb.on_message("No source file set for LiDAR input", "error")
                        return

                    # Tile data is shared between sessions while the source
                    # remains unchanged; LiDARTiler validates mtime metadata.
                    # Without intermediates every run re-tiles into a temp
                    # dir that the end-of-run sweep deletes.
                    source_fp = source_fps[iid]
                    source_tag = hashlib.sha1(source_fp.encode()).hexdigest()[:8]
                    tiles_dir = self._run_tiles_dir_for(
                        session_root, source_fp, intermediate_fmt
                    )
                    def tiling_progress(p: int, t: int) -> None:
                        pct = int(p / t * 100) if t else 0
                        if last_tiling_pct.get(iid) != pct:
                            last_tiling_pct[iid] = pct
                            cb.on_message(f"Tiling... {pct}%", "info")

                    tiler = LiDARTiler(
                        source_path,
                        tiles_dir,
                        tile_size=float(tiling_plan.get(iid, {}).get("tile_size_m") or 100.0),
                        buffer_m=float(tiling_plan.get(iid, {}).get("buffer_m") or 10.0),
                        max_points_per_tile=int(
                            tiling_plan.get(iid, {}).get("max_points_per_tile") or 0
                        ),
                        output_format=intermediate_fmt,
                        memory_budget_bytes=self._run_memory_budget_bytes,
                        progress_callback=tiling_progress if cb.on_message else None,
                        tile_callback=self._tile_seen_cb if cb.on_tile_count else None,
                        tag=source_tag,
                        acceleration=self._acceleration,
                        provenance=prov.build_provenance(
                            {
                                "module_id": self._modules.get(iid, iid),
                                "node_iid": iid,
                                "node_fps": {},
                                "provenance": self._prov_blueprint,
                            }
                        ),
                    )
                    tiling = tiler.run(
                        cancel_flag=lambda: self._cancel.is_set()
                    )
                    self._tilings[iid] = tiling
                    src_tiles = tiling.tiles_generated
                    if not src_tiles:
                        if self._cancel.is_set():
                            return  # cancelled; _cleanup notifies
                        if cb.on_node_status:
                            cb.on_node_status(iid, "error")
                        if cb.on_message:
                            cb.on_message(
                                f"No tiles generated — check the source file ({Path(source_path).name})",
                                "error",
                            )
                        return
                    if cb.on_node_status:
                        cb.on_node_status(iid, "done")
                    if cb.on_tile_count:
                        cb.on_tile_count(len(src_tiles))
                    if first_crs is None:
                        first_crs = tiling.crs
                    for t in src_tiles:
                        t["src_iid"] = iid
                    all_tiles.extend(src_tiles)

                tiles_dir = self._run_tiles_dir_for(
                    session_root,
                    source_fps[tiling_triggers[0]],
                    self._run_intermediate_ext,
                )
                tiles = all_tiles
                session_crs = first_crs

                # Heterogeneous source CRS is a warning, not a hard failure.
                # Each domain keeps its CRS; merged products may degrade.
                crss = {self._tilings[i].crs for i in tiling_triggers if i in self._tilings}
                self._run_crs_set = set(crss)
                if len(crss) > 1 and cb.on_message:
                    cb.on_message(
                        "Sources have different CRS (" + ", ".join(sorted(crss)) +
                        ") — continuing; check that products align.",
                        "warning",
                    )

                # Vendor QA flags censused during tiling (withheld/overlap):
                # snapshot for the quality report + actionable warnings.
                # Cache hits from old manifests may report nothing.
                # Density census (first returns, histograms) rides along.
                self._run_flag_counts = {}
                self._run_density_counts = {}
                for iid in tiling_triggers:
                    tiling = self._tilings.get(iid)
                    if tiling is None:
                        continue
                    counts = dict(getattr(tiling, "flag_counts", None) or {})
                    if counts:
                        total = int(getattr(tiling, "total_points", 0) or 0)
                        self._run_flag_counts[iid] = {
                            "total_points": total, "flags": counts
                        }
                    density = dict(
                        getattr(tiling, "density_counts", None) or {}
                    )
                    if density:
                        self._run_density_counts[iid] = density
                    if cb.on_message:
                        name = Path(
                            (source_paths or {}).get(iid, "")
                        ).name or iid
                        total = int(getattr(tiling, "total_points", 0) or 0)
                        meta = getattr(tiling, "source_meta", None) or {}
                        for text in self._flag_warnings(
                            name, total, counts
                        ):
                            cb.on_message(text, "warning")
                        for text in self._ingest_warnings(
                            name, getattr(tiling, "crs", None),
                            meta.get("point_format_id"),
                            meta.get("version"),
                        ):
                            cb.on_message(text, "warning")
                        for text in self._crs_origin_warnings(
                            name, (source_paths or {}).get(iid, "")
                        ):
                            cb.on_message(text, "warning")
                        try:
                            from lynceus.processing.tiler import _has_copc_vlr

                            is_copc = _has_copc_vlr(meta.get("vlrs"))
                        except Exception:
                            is_copc = False
                        if is_copc:
                            cb.on_message(
                                f"{name}: COPC source tiled as plain LAZ "
                                "(octree streaming not preserved); point "
                                "data intact.",
                                "warning",
                            )

            session_dir = self._new_session_dir(session_root)
            session_dir.mkdir(parents=True, exist_ok=True)
            self._run_main_session_dir = session_dir

            # Run context shared with every batch mini-session.
            self._run_source_fps = source_fps
            self._run_tiles_dir = tiles_dir
            self._run_tiling = tiling
            self._run_source_paths = source_paths
            self._run_session_crs = session_crs
            self._run_tiling_triggers = tiling_triggers

            # Segmentation: run the pipeline as sequential mini-sessions over
            # batches of whole tiles (no re-tiling, no sub-files). The cap
            # comes from the loader config (0 = single pass). v1 limits
            # segmentation to a single source; multi-source runs ignore it.
            seg_cap = self._resolve_segment_cap(
                configs, tiling_triggers[0]
            ) if tiling_triggers and tiles else 0
            if seg_cap > 0 and len(tiling_triggers) > 1:
                if cb.on_message:
                    cb.on_message(
                        "Segmentation only applies to a single source; "
                        "ignoring it for multi-source runs.",
                        "warning",
                    )
                seg_cap = 0
            segments = self._build_batches(tiles, seg_cap) if seg_cap > 0 else None

            if segments is None:
                batches = [(session_dir, tiles)]
            else:
                batches = [
                    (session_dir / "lotes" / f"segment_{k:04d}", batch_tiles)
                    for k, batch_tiles in enumerate(segments)
                ]

            outputs: dict = {}
            tile_results: list = []
            failed = 0
            run_metrics: dict = {}
            cancelled = False
            segment_output_payloads: list[dict] = []
            batch_verdicts: list = []

            # Consolidators merge every segment's provider product, so neither
            # they nor anything reachable from them can run inside a segment:
            # those nodes are deferred to a final session over all segment
            # outputs. Segment sessions run the remaining subgraph only.
            deferred = self._deferred_nodes(iids, edges, configs)
            segment_iids_run = [i for i in iids if i not in deferred]
            segment_edges_run = [
                e for e in edges if e[0] not in deferred and e[1] not in deferred
            ]
            providers_of: dict[str, list[str]] = {}
            providers_by_port: dict[str, dict[str, list[str]]] = {}
            if deferred:
                for s, d, out, _in in edges:
                    if d in deferred and s not in deferred:
                        providers_of.setdefault(d, []).append(s)
                        port_id = _port_id(out)
                        providers_by_port.setdefault(d, {}).setdefault(
                            port_id, []
                        ).append(s)
                # Salt the final session with every segment's scope AND the
                # fingerprint of every node that ran inside each segment: a
                # consolidator fingerprint must change when any upstream change
                # alters the segment products, so stale merged outputs are
                # never reused across different runs.
                seg_sig_parts = []
                for batch_index, (_batch_dir, batch_tiles) in enumerate(batches):
                    scope = self._batch_scope(batch_tiles, batch_index)
                    seg_fps = self._node_fingerprints(
                        segment_iids_run, segment_edges_run, configs,
                        self._run_source_fps, scope=scope,
                    )
                    seg_sig_parts.append(
                        scope + "|" + "|".join(seg_fps[i] for i in segment_iids_run)
                    )
                seg_sig = hashlib.sha1(
                    "\n".join(seg_sig_parts).encode()
                ).hexdigest()[:16]
            else:
                seg_sig = ""

            for batch_index, (batch_dir, batch_tiles) in enumerate(batches):
                if self._cancel.is_set():
                    cancelled = True
                    break
                if segments is not None and cb.on_segment_started:
                    cb.on_segment_started(
                        batch_index + 1,
                        len(batches),
                        int(sum(t.get("point_count", 0) for t in batch_tiles)),
                    )
                if len(batches) > 1 and cb.on_message:
                    cb.on_message(
                        f"Segment {batch_index + 1}/{len(batches)} "
                        f"({int(sum(t.get('point_count', 0) for t in batch_tiles))} "
                        "points)...",
                        "info",
                    )
                batch_out, batch_results, batch_failed, _, batch_metrics = (
                    self._run_batch(
                        segment_iids_run, segment_edges_run, configs, cb,
                        session_root, session_dir,
                        batch_dir, batch_tiles, batch_index, len(batches),
                    )
                )
                batch_verdicts.append(
                    getattr(self, "_last_batch_verdict", None)
                )
                if self._session_state.get("pipeline_status") == "cancelled":
                    cancelled = True
                    break
                if segments is not None:
                    # Publish this segment's products as soon as they finish so
                    # the gallery shows every segment incrementally.
                    segment_output_payloads.append(dict(batch_out))
                    if cb.on_segment_outputs:
                        cb.on_segment_outputs(
                            batch_index + 1, len(batches),
                            self._publishable_outputs(dict(batch_out)),
                        )
                if batch_out:
                    outputs.update(batch_out)
                if batch_results:
                    tile_results.extend(batch_results)
                failed += batch_failed
                if batch_metrics:
                    run_metrics = batch_metrics

            consolidated_outputs: dict = {}
            if not cancelled and deferred:
                if cb.on_message:
                    cb.on_message(
                        f"Consolidating {len(deferred)} deferred node(s) from "
                        f"{len(segment_output_payloads) or 1} segment "
                        "product(s)...",
                        "info",
                    )
                final_dir = session_dir / "final"
                final_iids = [i for i in iids if i in deferred]
                final_edges = [
                    e for e in edges if e[0] in deferred and e[1] in deferred
                ]
                final_out, final_results, final_failed, _, final_metrics = (
                    self._run_batch(
                        final_iids, final_edges, configs, cb, session_root,
                        session_dir, final_dir, [],
                        batch_index=len(batches), batch_count=len(batches),
                        seg_sig=seg_sig,
                        segment_outputs=segment_output_payloads
                        or [dict(outputs)],
                        segment_providers=providers_of,
                        providers_by_port=providers_by_port,
                    )
                )
                failed += final_failed
                batch_verdicts.append(
                    getattr(self, "_last_batch_verdict", None)
                )
                if final_results:
                    tile_results.extend(final_results)
                if final_metrics:
                    run_metrics = final_metrics
                if final_out:
                    consolidated_outputs = final_out
                    outputs.update(final_out)
                if cb.on_consolidated_outputs:
                    cb.on_consolidated_outputs(dict(consolidated_outputs))
            # No consolidators in the graph: nothing to merge, so the gallery
            # must NOT show a fake "Consolidated" block built from the last
            # segment batch (on_pipeline_finished already publishes the flat
            # products for non-segmented runs).

            if segments is not None:
                # Aggregate parent session_meta.json summarizing the batches.
                parent_meta = {
                    "pipeline_status": (
                        "cancelled"
                        if cancelled
                        else ("failed" if failed else "completed")
                    ),
                    "segments": [
                        {
                            "dir": f"lotes/segment_{k:04d}",
                            "tiles": len(batch_tiles),
                            "points": int(
                                sum(t.get("point_count", 0) for t in batch_tiles)
                            ),
                        }
                        for k, batch_tiles in enumerate(segments)
                    ],
                    "source_file": tiling.source_file if tiling else "",
                    "crs": session_crs,
                    "cell_size": RASTER_CELL_SIZE,
                    "tiles": len(tiles),
                    "nodes": list(self._modules.values()),
                    "instances": [
                        {"iid": iid, "module": module}
                        for iid, module in self._modules.items()
                    ],
                    "acceleration": self._acceleration.as_dict(),
                }
                (session_dir / "session_meta.json").write_text(
                    json.dumps(parent_meta, indent=2, default=str),
                    encoding="utf-8",
                )
                self._write_aggregate_quality_report(
                    session_dir, batch_verdicts, parent_meta,
                    tiles, run_metrics, cancelled, failed,
                )

            if not cancelled:
                if cb.on_finished:
                    cb.on_finished(
                        self._publishable_outputs(outputs),
                        tile_results, failed, tiles, run_metrics,
                    )
                if cb.on_message:
                    cb.on_message(
                        translate("Quality report written: {name}").format(
                            name="quality_report.json"
                        ),
                        "info",
                    )
                if cb.on_message:
                    if tiles:
                        seg_text = (
                            f" in {len(batches)} segments" if segments else ""
                        )
                        cb.on_message(
                            f"Pipeline finished — {len(tiles)} tiles processed "
                            f"({failed} with errors){seg_text}",
                            "success" if failed == 0 else "error",
                        )
                    else:
                        cb.on_message(
                            "Pipeline finished",
                            "success" if failed == 0 else "error",
                        )

        except Exception as exc:
            logger.error("Pipeline failed: %s", exc)
            if cb.on_message:
                cb.on_message(f"Pipeline error: {exc}", "error")
            self._fail_all(cb)
        finally:
            self._sweep_intermediates()
            self._cleanup(cb)

    def _publishable_outputs(self, outputs: dict) -> dict:
        """Gallery payloads without per-tile lists (deleted intermediates).

        In finals-only mode the tile LAZ files no longer exist after the
        sweep, so publishing a "Point Cloud (N tiles)" entry would offer a
        preview that can never load. Strip the tile lists; counts still
        travel through tile_results/metrics, and exported clouds preview
        from their final file. With intermediates kept, payloads pass
        through untouched.
        """
        if getattr(self, "_run_keep_intermediates", True):
            return outputs
        cleaned = {}
        for iid, payload in outputs.items():
            if isinstance(payload, dict) and payload.get("tiles"):
                payload = {
                    key: value
                    for key, value in payload.items()
                    if key != "tiles"
                }
            cleaned[iid] = payload
        return cleaned

    def _run_batch(
        self,
        iids: list[str],
        edges: list,
        configs: dict | None,
        cb: PipelineCallbacks,
        prior_root: Path,
        exclude_session: Path,
        session_dir: Path,
        tiles: list[dict],
        batch_index: int,
        batch_count: int,
        seg_sig: str = "",
        segment_outputs: list[dict] | None = None,
        segment_providers: dict[str, list[str]] | None = None,
        providers_by_port: dict[str, dict[str, list[str]]] | None = None,
    ) -> tuple:
        """Run the DAG over one tile batch as a self-contained mini-session.

        Each batch writes its own artifacts, session files, results and
        provenance under ``session_dir``; fingerprints are salted with the
        batch scope so engine reuse never links batches with different tiles.
        When this is the FINAL consolidation session (``segment_outputs`` is
        given), per-segment provider payloads and the split provider mapping
        are injected into the task context for the consolidators. Returns
        ``(outputs, tile_results, failed, tiles, run_metrics)``.
        """
        artifacts = self._artifacts_root(session_dir)
        artifacts.mkdir(parents=True, exist_ok=True)
        self._session_dir = session_dir

        scope = self._batch_scope(tiles, batch_index)
        fps = self._node_fingerprints(
            iids,
            edges,
            configs,
            self._run_source_fps,
            scope=scope,
            seg_sig=seg_sig,
        )
        order = compile_pipeline(iids, edges)
        reused: dict[str, dict] = {}
        tile_reuse: set[str] = set()
        barrier_reuse: set[str] = set()
        if cb.on_node_status:
            # Resolve candidates once per batch. Product inputs are excluded
            # because they are cheap to copy and must retain their IID.
            # Without intermediates every run is standalone: no reuse scan.
            if getattr(self, "_run_keep_intermediates", True):
                candidates = [
                    (prev_dir, entry)
                    for prev_dir, prov in self._prior_provenances(
                        prior_root, exclude_session
                    )
                    for entry in prov.get("nodes", {}).values()
                ]
            else:
                candidates = []
            consumed: set[int] = set()
            matched: dict[str, tuple] = {}
            for iid in order:
                module_id = self._modules.get(iid, iid)
                caps = self._node_caps(iid, configs)
                if not (
                    caps.get("tile_task")
                    or caps.get("barrier_task")
                    or caps.get("consolidate_task")
                ):
                    continue
                if caps.get("input_file_key"):
                    # Deliberate (S11b): product inputs always re-copy so two
                    # instances sharing a file never claim one fingerprint.
                    # They are still recorded below, hence scanned (and never
                    # claimed) on every batch.
                    continue
                prev_dir, entry = self._find_reusable(
                    candidates, consumed, module_id, fps.get(iid, "")
                )
                if prev_dir and entry:
                    matched[iid] = (prev_dir, entry)
            # Reverse fixpoint: a role reuses when its files exist upstream
            # or (tile roles only) every reader reuses. Barriers always need
            # their finals (the gallery publishes them).
            tile_reuse, barrier_reuse = self._effective_reuse(
                order, edges, matched, configs
            )
            self._reused_prev = dict(matched)
            reused: dict[str, dict] = {}
            for iid in order:
                if iid not in tile_reuse and iid not in barrier_reuse:
                    continue
                prev_dir, entry = matched[iid]
                self._materialize_bundle(prev_dir, artifacts, entry)
                if entry.get("output") is not None:
                    reused[iid] = self._rebase_payload(
                        entry.get("output"), prev_dir, artifacts
                    )
                if cb.on_node_status:
                    cb.on_node_status(iid, "cached")
        self._reused_iids = set(tile_reuse) | set(barrier_reuse)
        self._reused_tiles = set(tile_reuse)
        self._reused_barriers = set(barrier_reuse)
        self._reused_outputs = dict(reused)

        self._session_state = {
            "tiles": [dict(t) for t in tiles],
            "completed_nodes": [],
            "failed_nodes": [],
            "node_outputs": dict(reused),
            "node_modules": dict(self._modules),
            "pipeline_status": "running",
            "reused_nodes": list(self._reused_iids),
        }
        self._save_session_state()

        batch_points = int(sum(t.get("point_count", 0) for t in tiles))
        run_tiling = self._run_tiling
        run_source_fps = self._run_source_fps
        run_source_paths = self._run_source_paths
        run_tiling_triggers = self._run_tiling_triggers
        run_session_crs = self._run_session_crs
        (session_dir / "session_meta.json").write_text(
            json.dumps(
                {
                    "pipeline_status": "running",
                    "source_file": run_tiling.source_file if run_tiling else "",
                    "sources": [
                        {
                            "iid": iid,
                            "file": run_source_paths.get(iid, ""),
                            "source_fp": run_source_fps.get(iid, ""),
                            "tag": (
                                hashlib.sha1(
                                    run_source_fps[iid].encode()
                                ).hexdigest()[:8]
                                if iid in run_source_fps
                                else ""
                            ),
                            "crs": self._tilings[iid].crs
                            if iid in self._tilings
                            else "",
                            "tile_count": sum(
                                1
                                for t in tiles
                                if t["tile_id"].startswith(
                                    hashlib.sha1(
                                        run_source_fps[iid].encode()
                                    ).hexdigest()[:8]
                                )
                            ),
                        }
                        for iid in run_tiling_triggers
                    ],
                    "crs": run_session_crs,
                    "cell_size": RASTER_CELL_SIZE,
                    "tile_size_m": run_tiling.tile_size_m if run_tiling else 0,
                    "buffer_m": run_tiling.buffer_m if run_tiling else 0,
                    "total_points": batch_points,
                    "tiles": len(tiles),
                    "nodes": list(self._modules.values()),
                    "instances": [
                        {"iid": iid, "module": module}
                        for iid, module in self._modules.items()
                    ],
                    "source_meta": run_tiling.source_meta if run_tiling else {},
                    "reused_nodes": list(self._reused_iids),
                    "acceleration": self._acceleration.as_dict(),
                    "batch": (
                        {
                            "index": batch_index,
                            "count": batch_count,
                            "scope": scope,
                        }
                        if batch_count > 1
                        else {}
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        ctx = build_ctx(
            str(artifacts),
            run_session_crs,
            source_meta=run_tiling.source_meta if run_tiling else {},
            acceleration=self._acceleration.as_dict(),
        )
        configured_memory = (
            int(self._run_memory_budget_bytes)
            if self._run_memory_budget_bytes is not None
            and int(self._run_memory_budget_bytes) > 0
            else default_memory_budget_bytes()
        )
        worker_count = max(
            1, int(self._run_max_workers or os.cpu_count() or 4)
        )
        ctx["_raster_grid_memory_budget_bytes"] = max(
            1, configured_memory // worker_count
        )
        ctx["provenance"] = getattr(self, "_prov_blueprint", None) or {}
        ctx["node_fps"] = fps
        ctx["intermediate_ext"] = getattr(self, "_run_intermediate_ext", "laz")
        if segment_outputs is not None:
            # Consolidators read their provider's product from every segment
            # through these injection keys (their provider edges are trimmed
            # from the deferred subgraph).
            ctx["segment_outputs"] = segment_outputs
            ctx["segment_provider_iids"] = segment_providers or {}
            ctx["segment_providers_by_port"] = providers_by_port or {}
        outputs, tile_results, failed, _, run_metrics = self._execute_dag(
            iids, edges, tiles, ctx, cb, configs
        )
        failed += self._reject_ghost_products(
            outputs, iids, configs, cb, exclude=self._reused_barriers
        )
        self._write_quality_report(
            session_dir, iids, outputs, tile_results, failed, tiles,
            run_metrics, configs,
        )
        # Persist which node versions and products this batch used.
        if self._session_state.get("pipeline_status") == "completed":
            self._save_provenance(
                session_dir,
                run_source_fps,
                self._run_tiles_dir or Path(""),
                fps,
                order,
                tiles,
                getattr(self, "_last_outputs", None),
                configs,
            )
        return outputs, tile_results, failed, tiles, run_metrics

    def _reject_ghost_products(
        self,
        outputs: dict,
        iids: list,
        configs: dict | None,
        cb: PipelineCallbacks,
        exclude: set | None = None,
    ) -> int:
        """Drop barrier payloads advertising files that were never written.

        A barrier/consolidate task that returns ``{"file": path}`` without
        writing it (e.g. it silently read missing inputs) would otherwise
        publish a ghost product: the gallery offers it, QML styling warns,
        and previews fail. Ghosts become loud errors instead. Reused
        (republished) payloads are excluded: their files were verified at
        materialization time. Returns how many payloads were rejected.
        """
        excluded = exclude or set()
        rejected = 0
        for iid in list(outputs):
            if iid in excluded:
                continue
            try:
                caps = self._node_caps(iid, configs)
            except Exception:
                continue
            if not (caps.get("barrier_task") or caps.get("consolidate_task")):
                continue
            payload = outputs.get(iid)
            if not isinstance(payload, dict):
                continue
            advertised = payload.get("file")
            if (
                isinstance(advertised, str)
                and advertised
                and not Path(advertised).is_file()
            ):
                label = self._node_display_name(iid)
                if cb.on_node_status:
                    cb.on_node_status(iid, "error")
                if cb.on_message:
                    cb.on_message(
                        f"{label} reported '{Path(advertised).name}' but the "
                        "file was not written — check the node inputs",
                        "error",
                    )
                del outputs[iid]
                rejected += 1
        return rejected

    def _inputs_traced(self) -> bool | None:
        """Every external input fingerprinted (None when there are none).

        Loaders contribute size+mtime fingerprints; product inputs carry
        ``__src_fp__`` seeds resolved at run start. An empty fingerprint
        with a set path means the file vanished: untraceable.
        """
        found_any = False
        paths = getattr(self, "_run_source_paths", None) or {}
        fps = getattr(self, "_run_source_fps", None) or {}
        for iid, path in paths.items():
            if not path:
                continue
            found_any = True
            if not fps.get(iid):
                return False
        configs = getattr(self, "_run_configs", None) or {}
        for cfg in configs.values():
            if not isinstance(cfg, dict):
                continue
            for key, value in cfg.items():
                if not key.startswith("__src_fp__"):
                    continue
                fkey = key.split(":", 1)[1]
                if not cfg.get(fkey):
                    continue
                found_any = True
                if not value:
                    return False
        return True if found_any else None

    @staticmethod
    def _crs_origin_warnings(source_name: str, source_path: str) -> list[str]:
        """Warn when the CRS resolved without strict OGC WKT (pure, tested).

        USGS rejects GeoTIFF/ESRI CRS dialects for PDRF 6-10; a missing
        WKT is already covered by the CRS ingest warning. Header reads
        only (no point data).
        """
        try:
            import laspy

            from lynceus.processing.tiler import (
                _crs_provenance,
                _laz_backend,
            )

            with laspy.open(str(source_path), laz_backend=_laz_backend()) as reader:
                _crs, origin = _crs_provenance(reader.header)
        except Exception:
            return []
        if origin == "geokeys":
            return [
                f"{source_name}: CRS resolved from legacy GeoTIFF keys, "
                "not strict OGC WKT; reproject or normalize the CRS for "
                "USGS deliveries using PDRF 6-10."
            ]
        return []

    @staticmethod
    def _ingest_warnings(
        source_name: str,
        crs: str | None,
        point_format_id: int | None,
        version: str | None = None,
    ) -> list[str]:
        """Human-readable ingestion notices for one source (pure, tested).

        Missing WKT CRS and legacy point formats do not block the run
        (the engine handles rank angles and 5-bit classes), but the
        operator should know the fidelity limits.
        """
        messages: list[str] = []
        if crs is None or str(crs).strip() in ("", "Unknown"):
            messages.append(
                f"{source_name}: no CRS found (missing WKT); products "
                "carry no projection — check alignment in GIS."
            )
        try:
            pdrf = int(point_format_id) if point_format_id is not None else None
        except (TypeError, ValueError):
            pdrf = None
        if pdrf is not None and pdrf < 6:
            messages.append(
                f"{source_name}: legacy point format (PDRF {pdrf}"
                + (f", LAS {version}" if version else "")
                + "): 5-bit classes and rank angles only; processing "
                "continues with reduced fidelity."
            )
        return messages

    @staticmethod
    def _flag_warnings(
        source_name: str,
        total_points: int,
        flag_counts: dict | None,
        threshold: float = 0.01,
    ) -> list[str]:
        """Human-readable vendor-QA-flag notices for one source.

        Pure helper (tested): withheld/overlap fractions at or above the
        threshold warn. Withheld points are auto-excluded downstream;
        overlap doubles density in overlap zones (filter those strips or
        process them separately for area metrics). Synthetic/key-point
        fractions are reported silently in the quality report only.
        """
        messages: list[str] = []
        counts = dict(flag_counts or {})
        if total_points <= 0:
            return messages
        withheld = int(counts.get("withheld", 0) or 0)
        if withheld / total_points >= threshold:
            messages.append(
                f"{source_name}: {withheld:,} points "
                f"({100 * withheld / total_points:.1f}%) carry the vendor "
                "withheld flag and are excluded from surfaces and metrics."
            )
        overlap = int(counts.get("overlap", 0) or 0)
        if overlap / total_points >= threshold:
            messages.append(
                f"{source_name}: {overlap:,} points "
                f"({100 * overlap / total_points:.1f}%) carry the overlap "
                "flag: density is doubled in overlap zones; filter those "
                "strips or process them separately for area metrics."
            )
        return messages

    def _write_quality_report(
        self,
        session_dir,
        iids: list,
        outputs: dict,
        tile_results: list,
        failed: int,
        tiles: list,
        run_metrics: dict | None,
        configs: dict | None,
    ) -> str:
        """Write ``quality_report.json`` for one mini-session.

        Pure evidence sidecar: versioned spec, precomputed checks, PASS/FAIL
        verdict. Never raises (a skipped report only logs). Returns the
        verdict; the worker aggregates segment verdicts for the parent.
        """
        from lynceus import __version__ as app_version
        from lynceus.processing import quality as qual

        cancelled = self._cancel.is_set()
        status = "cancelled" if cancelled else ("failed" if failed else "completed")

        expected, written, empty = [], [], []
        for iid in iids:
            try:
                caps = self._node_caps(iid, configs)
            except Exception:
                continue
            if not (caps.get("barrier_task") or caps.get("consolidate_task")):
                continue
            expected.append(iid)
            payload = outputs.get(iid)
            path = payload.get("file") if isinstance(payload, dict) else None
            if isinstance(path, str) and path and Path(path).is_file():
                written.append(iid)
                try:
                    if Path(path).stat().st_size == 0:
                        empty.append(Path(path).name)
                except OSError:
                    pass
        crs_set = getattr(self, "_run_crs_set", None)
        n_warnings = 0
        for payload in outputs.values():
            if isinstance(payload, dict):
                warnings = payload.get("warnings")
                if isinstance(warnings, list):
                    n_warnings += len(warnings)
        checks = {
            "failed_tasks": failed,
            "expected_products": expected,
            "written_products": written,
            "crs_list": sorted(crs_set) if crs_set else None,
            "empty_products": empty,
            "traced_inputs": self._inputs_traced(),
            "warnings": n_warnings,
        }
        evaluation, verdict = qual.evaluate(qual.DEFAULT_SPEC, checks)
        lineage = {
            "app_version": app_version,
            "operator": getattr(self, "_run_operator", None),
            "sources": [
                {
                    "iid": iid,
                    "file": (getattr(self, "_run_source_paths", None) or {}).get(iid, ""),
                    "source_fp": (getattr(self, "_run_source_fps", None) or {}).get(iid, ""),
                }
                for iid in sorted(getattr(self, "_run_source_paths", None) or {})
            ],
            "nodes": [
                {"iid": iid, "module": self._modules.get(iid, iid)}
                for iid in iids
            ],
        }
        counts = {
            "tiles": len(tiles),
            "points": int(sum(t.get("point_count", 0) for t in tiles)),
            "products_expected": len(expected),
            "products_written": len(written),
        }
        by_source = getattr(self, "_run_flag_counts", None) or {}
        if by_source:
            flagged = {"total_points": 0}
            for entry in by_source.values():
                if not isinstance(entry, dict):
                    continue
                flagged["total_points"] += int(entry.get("total_points", 0) or 0)
                for name, num in (entry.get("flags", {}) or {}).items():
                    flagged[name] = int(flagged.get(name, 0)) + int(num or 0)
            counts["point_flags"] = flagged
        by_density = getattr(self, "_run_density_counts", None) or {}
        if by_density:
            import math

            dens = {
                "total_points": 0, "first_returns": 0, "area_m2": 0.0,
                "class_histogram": {}, "return_histogram": {},
            }
            for iid, entry in by_density.items():
                if not isinstance(entry, dict):
                    continue
                tilings = getattr(self, "_tilings", None) or {}
                tiling = tilings.get(iid)
                meta = getattr(tiling, "source_meta", None) or {}
                mins = meta.get("mins") or []
                maxs = meta.get("maxs") or []
                area = 0.0
                if len(mins) >= 2 and len(maxs) >= 2:
                    try:
                        area = max(
                            0.0,
                            (float(maxs[0]) - float(mins[0]))
                            * (float(maxs[1]) - float(mins[1])),
                        )
                    except (TypeError, ValueError):
                        area = 0.0
                dens["total_points"] += int(
                    getattr(tiling, "total_points", 0) or 0
                )
                dens["first_returns"] += int(entry.get("first_returns", 0) or 0)
                dens["area_m2"] += area
                for hist_key in ("class_histogram", "return_histogram"):
                    for key, num in (entry.get(hist_key, {}) or {}).items():
                        slot = dens[hist_key]
                        slot[str(key)] = int(slot.get(str(key), 0)) + int(num or 0)
            dens["area_m2"] = round(dens["area_m2"], 3)
            if dens["area_m2"] > 0:
                dens["npd_all_pts_m2"] = round(
                    dens["total_points"] / dens["area_m2"], 4
                )
                if dens["first_returns"] > 0:
                    dens["npd_first_pts_m2"] = round(
                        dens["first_returns"] / dens["area_m2"], 4
                    )
                    dens["nps_first_m"] = round(
                        1.0 / math.sqrt(
                            dens["first_returns"] / dens["area_m2"]
                        ),
                        4,
                    )
            counts["density"] = dens
        performance = dict(run_metrics or {})
        try:
            start = datetime.fromisoformat(str(performance.get("start_iso")))
            end = datetime.fromisoformat(str(performance.get("end_iso")))
            performance["duration_s"] = round(
                (end - start).total_seconds(), 1
            )
        except Exception:
            pass
        report = qual.build_quality_report(
            status=status,
            session_name=Path(session_dir).name,
            lineage=lineage,
            counts=counts,
            performance=performance,
            evaluation=evaluation,
            verdict=verdict,
            warnings=n_warnings,
        )
        try:
            Path(session_dir).mkdir(parents=True, exist_ok=True)
            (Path(session_dir) / "quality_report.json").write_text(
                json.dumps(report, indent=2, default=str), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning("Quality report skipped %s: %s", session_dir, exc)
        self._last_batch_verdict = (str(session_dir), verdict, status)
        return verdict

    def _write_aggregate_quality_report(
        self,
        session_dir,
        batch_verdicts: list,
        parent_meta: dict,
        tiles: list,
        run_metrics: dict | None,
        cancelled: bool,
        failed: int,
    ) -> None:
        """Write the parent-session aggregate quality report.

        Segmented runs keep one report per segment batch; the parent
        aggregates their verdicts (FAIL propagates) with run-level counts.
        """
        from lynceus import __version__ as app_version
        from lynceus.processing import quality as qual

        segments = []
        for entry in batch_verdicts:
            if not entry:
                continue
            seg_dir, verdict, _status = entry
            segments.append(
                {
                    "dir": Path(seg_dir).name,
                    "verdict": verdict,
                }
            )
        status = (
            "cancelled"
            if cancelled
            else ("failed" if failed else "completed")
        )
        fails = [s for s in segments if s["verdict"] == qual.VERDICT_FAIL]
        verdict = (
            qual.VERDICT_FAIL
            if (fails or status != "completed")
            else qual.VERDICT_PASS
        )
        evaluation = {
            "segments_conform": {
                "result": "pass" if not fails else "fail",
                "detail": (
                    f"{len(segments) - len(fails)}/{len(segments)} "
                    "segments conform"
                ),
            }
        }
        report = qual.build_quality_report(
            status=status,
            session_name=Path(session_dir).name,
            lineage={
                "app_version": app_version,
                "operator": getattr(self, "_run_operator", None),
                "nodes": [
                    {"iid": iid, "module": module}
                    for iid, module in self._modules.items()
                ],
            },
            counts={
                "tiles": len(tiles),
                "points": int(sum(t.get("point_count", 0) for t in tiles)),
                "segments": len(segments),
            },
            performance=dict(run_metrics or {}),
            evaluation=evaluation,
            verdict=verdict,
            segments=segments,
        )
        try:
            Path(session_dir).mkdir(parents=True, exist_ok=True)
            (Path(session_dir) / "quality_report.json").write_text(
                json.dumps(report, indent=2, default=str), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning(
                "Aggregate quality report skipped %s: %s", session_dir, exc
            )

    def _detect_tiling_plan(
        self, iids: list[str], edges, configs: dict | None
    ) -> dict[str, dict]:
        """Return ``{loader_iid: tiling config}`` for every source node.

        Tiling and segmentation parameters live in the loader's own config
        (Inspector overrides); unset keys fall back to the engine defaults at
        the tiler call site. There is no separate ``Tile``/``Segment`` node.
        """
        loaders = [
            iid
            for iid in iids
            if discover_node_capabilities(self._modules.get(iid, iid)).get(
                "triggers_tiling"
            )
        ]
        plan: dict[str, dict] = {}
        for iid in loaders:
            plan[iid] = dict(configs.get(iid, {})) if configs else {}
        return plan

    def _resolve_segment_cap(self, configs: dict | None, loader_iid: str) -> int:
        """Resolve the loader's ``max_points_per_part`` cap.

        The cap lives in the loader config (0 = single pass). A freshly
        placed loader ships no per-instance config; the schema default (0)
        is the engine fallback so running never flips into segmentation
        unexpectedly.
        """
        cfg = configs.get(loader_iid, {}) if configs else {}
        cap = cfg.get("max_points_per_part")
        if cap:
            return int(cap)
        schema = discover_node_capabilities(
            self._modules.get(loader_iid, loader_iid)
        ).get("config_schema", {})
        return int(schema.get("max_points_per_part", {}).get("default", 0))

    def _build_batches(self, tiles: list[dict], max_points_per_part=0) -> list[list[dict]]:
        """Group tiles into sequential batches not exceeding ``max_points_per_part``.

        Tiles are taken in row/column/subdivision order so each batch covers a
        spatially contiguous block. A single tile larger than the cap still
        forms its own batch. ``max_points_per_part <= 0`` returns the whole
        tile set as one batch.
        """
        cap = int(max_points_per_part or 0)
        if cap <= 0:
            return [list(tiles)]
        ordered = sorted(
            tiles,
            key=lambda t: (
                t.get("row", 0),
                t.get("col", 0),
                t.get("sub_row", 0),
                t.get("sub_col", 0),
            ),
        )
        batches: list[list[dict]] = []
        current: list[dict] = []
        count = 0
        for tile in ordered:
            n = int(tile.get("point_count") or 0)
            if current and count + n > cap:
                batches.append(current)
                current, count = [], 0
            current.append(tile)
            count += n
        if current:
            batches.append(current)
        return batches

    def _deferred_nodes(
        self, iids: list[str], edges: list, configs: dict | None = None
    ) -> set[str]:
        """Nodes deferred to the final consolidation session.

        Consolidators merge the per-segment products of their provider, so
        they cannot run inside any segment; neither can anything reachable
        from them (their downstream consumers). This set is the outer reachable
        closure seeded by every ``consolidator`` instance. The segment
        sessions then exclude it and a single barrier-only session runs the
        deferred subgraph against every segment output.
        """
        seed = {
            iid
            for iid in iids
            if effective_node_caps(
                self._modules.get(iid, iid), (configs or {}).get(iid)
            ).get("consolidator")
        }
        if not seed:
            return set()
        deferred = set(seed)
        pending = list(seed)
        while pending:
            src = pending.pop()
            for s, dst, _out, _in in edges:
                if s == src and dst not in deferred:
                    deferred.add(dst)
                    pending.append(dst)
        return deferred

    def _batch_scope(self, batch_tiles: list[dict], batch_index: int) -> str:
        """Stable per-batch fingerprint scope (index + exact tile content)."""
        tile_ids = "\n".join(sorted(t["tile_id"] for t in batch_tiles))
        return f"seg{batch_index:04d}:{hashlib.sha1(tile_ids.encode()).hexdigest()[:12]}"

    def _resolve_input_crs(self, configs: dict, iids: list[str]) -> str:
        """Resolve CRS for a barrier-only run from the first raster input.

        Barrier-only graphs have no tiling CRS. If no input CRS is available,
        consumers read CRS and transforms from their own input files.
        """
        for iid in iids:
            caps = self._node_caps(iid, configs)
            fkey = caps.get("input_file_key")
            if not fkey:
                continue
            path = (configs.get(iid) or {}).get(fkey) or ""
            if not path or not os.path.isfile(path):
                continue
            try:
                import rasterio

                with rasterio.open(path) as ds:
                    if ds.crs is not None:
                        return ds.crs.to_string()
            except Exception:
                continue
        return ""

    def _check_consent(self, cb: PipelineCallbacks) -> bool:
        """Rejects the run if any extension node has an EULA pending.

        Runs on the worker thread: only reads state and reports via
        callbacks (a Qt dialog would crash here).
        """
        if self.eula_checker is None:
            return True
        for module_id in sorted(set(self._modules.values())):
            try:
                reason = self.eula_checker(module_id)
            except Exception as exc:  # never break a run twice
                logger.warning("eula_checker failed for %s: %s", module_id, exc)
                reason = f"Consent check failed for '{module_id}': {exc}"
            if reason:
                if cb.on_message:
                    cb.on_message(reason, "error")
                if cb.on_finished:
                    cb.on_finished({}, [], 0, [], {"blocked_by_eula": True})
                self._state = "idle"
                return False
        return True

    def _check_available(self, cb: PipelineCallbacks) -> bool:
        """Rejects the run if any node's module is no longer registered.

        A node of a disabled or removed extension must not run (it silently
        does nothing). Defense in depth: the UI gates earlier, but a missing
        module must never reach the pool.
        """
        for module_id in sorted(set(self._modules.values())):
            try:
                present = manager.contains(module_id)
            except Exception as exc:
                logger.warning(
                    "availability check failed for %s: %s", module_id, exc
                )
                present = False
            if not present:
                if cb.on_message:
                    cb.on_message(
                        f"Node '{module_id}' uses a missing or disabled "
                        "extension — enable it or remove the node",
                        "error",
                    )
                if cb.on_finished:
                    cb.on_finished(
                        {}, [], 0, [], {"blocked_by_available": True}
                    )
                self._state = "idle"
                return False
        return True

    def _check_required_ports(
        self,
        iids: list[str],
        edges: list,
        cb: PipelineCallbacks,
        configs: dict | None = None,
    ) -> bool:
        """Block runs with missing required connections.

        Optional ports are ignored. This lightweight check runs before pool
        creation or tiling and reports missing typed edges through callbacks.
        Strategy nodes resolve their ports per instance (the selected variant
        defines what is required); non-variant nodes fall back to their static
        module ports.
        """
        configs = configs or {}
        connected: dict[str, set[str]] = {}
        for _src, dst, _out, in_type in edges:
            key = in_type.value if hasattr(in_type, "value") else str(in_type)
            connected.setdefault(dst, set()).add(key)
        missing: list[str] = []
        for iid in iids:
            module_id = self._modules.get(iid, iid)
            try:
                inputs, _outputs = resolved_ports(module_id, configs.get(iid))
            except Exception as exc:  # not importable -> skip the port gate
                logger.debug("resolved_ports(%s) skipped: %s", module_id, exc)
                continue
            if not inputs:  # non-variant static fallback
                # NOTE: an inert strategy base (INPUTS=()) also lands here
                # with no block. Strategy selection is enforced upstream by
                # the canvas wired-strategy gate, not by this port gate.
                try:
                    inputs, _outputs = manager.load_ports(module_id)
                except Exception as exc:
                    logger.debug("load_ports(%s) skipped: %s", module_id, exc)
                    continue
            for item in inputs:
                meta = port_metadata(item)
                if not meta.required:
                    continue
                pt = meta.port_type
                key = pt.value if hasattr(pt, "value") else str(pt)
                if key not in connected.get(iid, ()):
                    missing.append(
                        f"{self._node_display_name(iid)} ← "
                        f"{get_display_name(meta.port_type)}"
                    )
        if not missing:
            return True
        if cb.on_node_status:
            for iid in iids:
                cb.on_node_status(iid, "error")
        if cb.on_message:
            cb.on_message(
                "Missing required inputs:\n  " + "\n  ".join(sorted(missing)),
                "error",
            )
        if cb.on_finished:
            cb.on_finished({}, [], 0, [], {"blocked_by_required": True})
        self._state = "idle"
        return False

    def _tile_seen_cb(self, tile_id: str) -> None:
        self._tiles_seen += 1

    # ------------------------------------------------------------------
    # DAG
    # ------------------------------------------------------------------

    @staticmethod
    def _drop_skipped_tile_tasks(tasks: list, skip_iids: set) -> list:
        """Drop tile tasks of effectively reused nodes from the DAG.

        Every role reading their outputs reuses too (proven by the reverse
        fixpoint), so the files are needed by nobody in this batch.
        Remaining dependencies on skipped ids are stripped, which is
        equivalent to instant completion. Pure helper: duck-typed tasks.
        """
        if not skip_iids:
            return tasks
        skipped = {
            task.task_id
            for task in tasks
            if task.kind == "tile" and task.iid in skip_iids
        }
        if not skipped:
            return tasks
        kept = [task for task in tasks if task.task_id not in skipped]
        for task in kept:
            task.deps -= skipped
        return kept

    def _execute_dag(
        self,
        iids: list[str],
        edges: list,
        tiles: list[dict],
        ctx: dict,
        cb: PipelineCallbacks,
        configs: dict | None = None,
    ) -> tuple:
        """Run the DAG once; returns ``(outputs, tile_results, failed, tiles, metrics)``.

        The caller (single run or batch orchestrator) owns the final
        ``on_finished`` callback: mini-sessions suppress it per batch so the
        UI receives one aggregated completion at the end.
        """
        order = compile_pipeline(iids, edges)
        tasks = build_dag(
            order, edges, tiles, ctx, modules=self._modules, configs=configs
        )
        reused_tiles = getattr(self, "_reused_tiles", set()) or set()
        reused_barriers = getattr(self, "_reused_barriers", set()) or set()
        # Replace reused barriers with a no-op that republishes the cached
        # payload; files were already materialized in the current session.
        # Skip tile tasks of effectively reused nodes outright: every role
        # reading their outputs reuses too, so nothing needs them. Skipped
        # ids are stripped from the remaining dependencies (equivalent to
        # instant completion); tile_results simply omits skipped tasks.
        tasks = self._drop_skipped_tile_tasks(tasks, reused_tiles)
        refined = []
        for task in tasks:
            if task.iid in reused_barriers and task.kind == "barrier":
                payload = self._reused_outputs.get(task.iid, {})
                task.fn = _noop_barrier
                task.args = (payload,)
            refined.append(task)
        tasks = refined
        self._prepare_progress(tasks)

        if cb.on_node_status:
            for iid in order:
                caps = self._node_caps(iid, configs)
                tile_work = caps.get("tile_task") and iid not in reused_tiles
                barrier_work = (
                    caps.get("barrier_task") or caps.get("consolidate_task")
                ) and iid not in reused_barriers
                if tile_work or barrier_work:
                    cb.on_node_status(iid, "processing")
        if cb.on_message:
            if tiles:
                cb.on_message(
                    f"Processing {len(tiles)} tiles in parallel "
                    f"({ctx.get('cell_size', 1.0):.0f} m cells)...",
                    "info",
                )
            else:
                cb.on_message("Processing inputs...", "info")

        outputs, tile_results, failed, exec_stats = run_dag(
            self._pool,
            tasks,
            progress_callback=self._make_dag_progress(cb),
            cancel_flag=lambda: self._cancel.is_set(),
            pause_flag=lambda: not self._pause.is_set(),
        )

        self._emit_warnings(cb, outputs, tile_results)

        run_end_iso = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        run_metrics = {
            **exec_stats,
            "start_iso": self._run_start_iso,
            "end_iso": run_end_iso,
        }
        self.last_metrics = run_metrics
        self._save_run_metrics(run_metrics)
        self._last_outputs = dict(outputs)

        result = (outputs, tile_results, failed, tiles, run_metrics)
        cancelled = self._cancel.is_set()
        if cancelled:
            # Cancellation is not a task failure. Restore pending nodes to idle;
            # _cleanup emits the final cancellation message.
            self._session_state["pipeline_status"] = "cancelled"
            self._session_state["failed_nodes"] = sorted(self._node_failed)
            self._save_session_state()
            self._update_session_meta("cancelled")
            if cb.on_node_status:
                for iid in list(self._node_task_total):
                    if iid not in self._node_failed:
                        cb.on_node_status(iid, "idle")
            return result

        # Persist barrier outputs in session state, keyed by instance ID.
        for iid, out in outputs.items():
            self._session_state.setdefault("node_outputs", {})[iid] = out
            self._session_state.setdefault("completed_nodes", []).append(iid)
        self._session_state["failed_nodes"] = sorted(self._node_failed)
        self._session_state["pipeline_status"] = (
            "completed" if failed == 0 else "failed"
        )
        self._save_session_state()
        self._update_session_meta(
            "completed" if failed == 0 else "failed"
        )
        self._write_results(tile_results)

        # Persist barrier outputs under artifacts for cached republishing.
        ctx_root = Path(ctx.get("session_dir", ""))
        out_json = ctx_root / "node_outputs.json"
        try:
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_json.write_text(
                json.dumps(
                    self._session_state.get("node_outputs", {}),
                    default=str,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Could not persist node_outputs: %s", exc)

        # Apply node-declared QGIS styles in the worker; reused products are
        # included because their files are already materialized.
        if failed == 0:
            self._apply_qml(outputs, cb, configs)

        return result

    def _prepare_progress(self, tasks: list) -> None:
        self._last_tasks = tasks
        self._node_task_total = {}
        self._stage_total = {}
        for task in tasks:
            self._node_task_total[task.iid] = (
                self._node_task_total.get(task.iid, 0) + 1
            )
            if task.kind == "tile":
                self._stage_total[task.iid] = (
                    self._stage_total.get(task.iid, 0) + 1
                )
        self._node_task_done = {}
        self._stage_done = {}
        self._node_failed = set()

    def _make_dag_progress(self, cb: PipelineCallbacks):
        def progress(done: int, total: int, task, ok: bool, exc=None) -> None:
            iid = task.iid
            if ok:
                if iid in getattr(self, "_reused_barriers", set()):
                    # Reused barrier: do not overwrite "cached" with "done".
                    pass
                else:
                    self._node_task_done[iid] = (
                        self._node_task_done.get(iid, 0) + 1
                    )
                    if self._node_task_done[iid] >= self._node_task_total.get(
                        iid, 0
                    ):
                        if cb.on_node_status:
                            cb.on_node_status(iid, "done")
            elif iid not in self._node_failed:
                self._node_failed.add(iid)
                if cb.on_node_status:
                    cb.on_node_status(iid, "error")
                if exc is not None and cb.on_message:
                    # The failure reason used to die in the worker log: the
                    # UI showed red with no cause. Real failures (not
                    # dependency cascades, which carry no exception) report
                    # their reason as a persistent error notice.
                    reason = str(exc).strip().splitlines()
                    reason = reason[0] if reason else "unknown error"
                    if len(reason) > 400:
                        reason = reason[:400] + "…"
                    cb.on_message(
                        f"{self._node_display_name(iid)} failed: {reason}",
                        "error",
                    )
            if task.kind == "tile":
                self._stage_done[iid] = self._stage_done.get(iid, 0) + 1
                if cb.on_message:
                    parts = []
                    for node_iid, total_n in self._stage_total.items():
                        label = self._module_label(node_iid)
                        parts.append(
                            f"{label} {self._stage_done.get(node_iid, 0)}/{total_n}"
                        )
                    cb.on_message("  |  ".join(parts), "info")
            elif cb.on_message:
                cb.on_message(f"Merging {self._module_label(iid)}...", "info")

            if cb.on_progress:
                cb.on_progress(done, total)

        return progress

    def _apply_qml(
        self, outputs: dict, cb: PipelineCallbacks, configs: dict | None = None
    ) -> None:
        """Generate QGIS sidecar styles for run products.

        Nodes declare basename-to-ramp/field mappings. Missing styles use
        filename conventions and generic vector styling. Failures are logged
        because styling must never fail the pipeline.
        """
        try:
            from lynceus.processing.qml_style import style_outputs

            qml_specs: dict[str, dict] = {}
            for iid in outputs:
                caps = self._node_caps(iid, configs)
                qml_specs[iid] = caps.get("qml") or {}
            n = style_outputs(outputs, qml_specs)
            if n and cb.on_message:
                cb.on_message(f"QGIS styles generated ({n} .qml)", "info")
        except Exception as exc:
            logger.warning("QML generation skipped: %s", exc)

    @staticmethod
    def _emit_warnings(
        cb: PipelineCallbacks,
        outputs: dict,
        tile_results: list,
    ) -> None:
        """Forward warning lists from barriers and tile results.

        Warnings are deduplicated in stable order and emitted to the UI as
        ``on_message(kind="warning")``.
        """
        if not cb.on_message:
            return
        seen: set[str] = set()
        for source in (*outputs.values(), *tile_results):
            for warning in source.get("warnings", []) or []:
                if not isinstance(warning, str) or warning in seen:
                    continue
                seen.add(warning)
                cb.on_message(warning, "warning")
