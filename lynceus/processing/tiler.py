# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Tile LAS/LAZ point clouds into regular buffered tiles.

The implementation uses chunked streaming, vectorized assignment, parallel
LAZ writes, manifest caching, bounded I/O, a global memory budget, mtime-based
cache invalidation, and LAZ backend fallback.

Point-capped tiling (``max_points_per_tile``):
  When ``max_points_per_tile > 0`` the tiler performs a two-pass adaptive
  subdivision. Pass A streams the source building a per-base-tile sub-cell
  density histogram (up to ``MAX_SUB_DIV`` levels). Each base tile is assigned
  the smallest subdivision level whose sub-tiles all fit under the cap. Pass B
  streams the source again and routes every point into its final
  (possibly subdivided) tile, so no tile ever exceeds the point cap. The two
  passes share the same buffer-membership math, so the cap is a hard
  guarantee; the second decompression pass is paid once and cached by the
  manifest.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from concurrent.futures import (
    FIRST_COMPLETED,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import laspy
import numpy as np
import psutil

from lynceus.processing.acceleration import AccelerationInfo, gpu_array_module
from lynceus.processing import provenance

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default resource limits for a desktop workstation.
# ---------------------------------------------------------------------------
CHUNK_SIZE = 4_000_000         # Points per read chunk.
BUFFER_FLUSH_SIZE = 8_000_000  # Points per tile before writing to LAZ.
MAX_IO_THREADS = 4             # Concurrent writer threads.
IO_FUTURE_WINDOW = 4            # Maximum in-flight write futures.
# Parallel decode: the two full-file passes are decompression-bound (native LAZ
# decode releases the GIL), so spans are decoded by a bounded worker pool while
# the consumer routes strictly in file order (deterministic, worker-count
# independent). DECODE_WINDOW bounds in-flight decoded spans -> peak RAM.
DECODE_MAX_WORKERS = 8         # Peak parallel decode threads.
DECODE_WINDOW = 4              # In-flight decoded spans.
# Global buffered-tile budget in BYTES (not points). The historical 1e9-point
# budget buffered the whole dataset (~30 GB packed) before flushing, which
# swaps every machine with less RAM and makes tiling grind to a halt. The
# budget is RAM-adaptive: tiles are flushed to LAZ as soon as the working set
# exceeds it, so tiling stays O(N) in time and O(budget) in peak memory.
MEMORY_BUDGET_FLOOR_BYTES = 1_073_741_824  # absolute floor (1 GiB)
MEMORY_RAM_RATIO = 0.25  # budget = max(floor, ratio of system RAM)
CAP_WRITERS = 64               # Keep at most this many LAZ writers open.

# Adaptive subdivision of dense tiles (max_points_per_tile).
MAX_SUB_DIV = 4                # A base tile may be split into 4^4 = 256 cells.
SUB_HIST_LIMIT = 32_000_000    # Max histogram cells (int32 -> ~128 MB).
CAP_SLACK = 0.9                # Leave 10% headroom below the point cap.

# Bumped to 6 after the COPC seek fix: caches written by v5 may hold
# tiles poisoned with zero-filled records (parallel seek decode on COPC
# silently returned zeros, clipped into tile (0,0)). No migration: stale
# caches re-tile once.
MANIFEST_VERSION = 6


def system_memory_bytes() -> int:
    """Total system RAM in bytes; 0 if the API is unavailable."""
    try:
        return int(psutil.virtual_memory().total)
    except Exception:
        return 0


def default_memory_budget_bytes(total_ram: int | None = None) -> int:
    """RAM-adaptive buffered-tile budget (bytes) with an absolute floor."""
    ram = total_ram if total_ram is not None else system_memory_bytes()
    return max(MEMORY_BUDGET_FLOOR_BYTES, int(ram * MEMORY_RAM_RATIO))


def memory_budget_for_percent(
    percent: int | float, total_ram: int | None = None
) -> int:
    """Tile budget for a RAM percentage (Preferences > Performance).

    Pure helper so the UI math stays tested: clamps to [1, 100] and keeps
    the absolute floor (a tiny percentage never starves the tiler).
    """
    try:
        pct = float(percent)
    except (TypeError, ValueError):
        pct = MEMORY_RAM_RATIO * 100.0
    pct = min(100.0, max(1.0, pct))
    ram = total_ram if total_ram is not None else system_memory_bytes()
    return max(MEMORY_BUDGET_FLOOR_BYTES, int(ram * pct / 100.0))


@dataclass(frozen=True)
class TileIndex:
    col: int
    row: int
    tag: str = ""
    sub_level: int = 0
    sub_row: int = 0
    sub_col: int = 0

    def __str__(self) -> str:
        base = (
            f"{self.tag}_c{self.col:04d}_r{self.row:04d}"
            if self.tag
            else f"tile_c{self.col:04d}_r{self.row:04d}"
        )
        if self.sub_level:
            return f"{base}_s{self.sub_level}_{self.sub_row:02d}_{self.sub_col:02d}"
        return base


@dataclass
class TilingResult:
    source_file: str
    output_dir: str
    crs: str
    total_points: int
    tile_size_m: float
    buffer_m: float = 0.0
    max_points_per_tile: int = 0
    max_subdivision: int = 0
    tiles_generated: list[dict] = field(default_factory=list)
    tiles_skipped_empty: int = 0
    source_meta: dict = field(default_factory=dict)
    output_format: str = "laz"
    flag_counts: dict = field(default_factory=dict)
    """Vendor QA flag counts ({withheld, overlap, synthetic, key_point})
    over source points (each point once). Absent on cache hits from
    manifests written before this field existed; dimensions the reader
    does not surface are simply not counted."""
    density_counts: dict = field(default_factory=dict)
    """Density census for QA ({first_returns, class_histogram,
    return_histogram}, histograms keyed by str(int)). Same
    cache-absence semantics as flag_counts."""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _laz_backend() -> laspy.LazBackend:
    """Return the first available LAZ backend, preferring parallel encoding."""
    for backend in (
        laspy.LazBackend.LazrsParallel,
        laspy.LazBackend.Lazrs,
        laspy.LazBackend.Laszip,
    ):
        if backend.is_available():
            return backend
    raise RuntimeError("No compatible LAZ backend found (lazrs/laszip)")


def _crs_provenance(header: laspy.LasHeader) -> tuple[str | None, str]:
    """Resolve CRS metadata without requiring pyproj directly.

    Returns ``(crs, origin)`` with origin in
    ``laspy|wkt2112|geokeys|none``. Resolution order: laspy.parse_crs(),
    WKT VLR 2112, then GeoKeyDirectory EPSG keys 3076/2048. Raw VLR
    metadata remains available to callers.
    """
    from rasterio.crs import CRS

    try:
        crs = header.parse_crs()
        if crs is not None:
            return str(crs), "laspy"
    except Exception:
        pass

    for vlr in getattr(header, "vlrs", None) or []:
        record_id = getattr(vlr, "record_id", None)
        if record_id == 2112:
            try:
                wkt = vlr.bytes.decode("utf-8", errors="replace")
                return str(CRS.from_wkt(wkt)), "wkt2112"
            except Exception:
                continue
        if record_id in (34735, 34737):
            try:
                for key in vlr.geo_keys:
                    if key.id in (3076, 2048) and key.tiff_tag_location == 0:
                        return str(CRS.from_epsg(int(key.value_offset))), "geokeys"
            except Exception:
                continue
    return None, "none"


def _crs_from_rasterio(header: laspy.LasHeader) -> str | None:
    """Resolve CRS metadata without requiring pyproj directly.

    Resolution order: laspy.parse_crs(), WKT VLR 2112, then GeoKeyDirectory
    EPSG keys 3076/2048. Raw VLR metadata remains available to callers.
    """
    crs, _origin = _crs_provenance(header)
    return crs


def _require_no_waveform(header: laspy.LasHeader, source_path) -> None:
    """Refuse full-waveform clouds the pipeline cannot carry.

    Waveform samples live in EVLRs, which laspy silently drops on write:
    tiling PDRF 9/10 would destroy data without error. Fail loudly with
    the file named instead. Inspection-only paths (File Info) stay open.
    """
    try:
        pdrf = int(header.point_format.id)
    except Exception:
        return
    if pdrf in (9, 10):
        name = Path(source_path).name if source_path else "source"
        raise RuntimeError(
            f"{name}: full-waveform data (PDRF {pdrf}) cannot be processed: "
            "waveform packets live in EVLRs the pipeline cannot carry. "
            "Convert to discrete returns first."
        )


def _extract_crs(header: laspy.LasHeader) -> str:
    return _crs_from_rasterio(header) or "Unknown"


def _has_copc_vlr(vlrs) -> bool:
    """True when a VLR list carries COPC hierarchy metadata.

    Accepts live header VLR objects and ``describe_source`` dicts alike.
    """
    try:
        for v in vlrs or []:
            if isinstance(v, dict):
                uid = v.get("user_id", "")
            else:
                uid = getattr(v, "user_id", "")
            if str(uid or "").lower() == "copc":
                return True
        return False
    except Exception:
        return False


def describe_source(header: laspy.LasHeader) -> dict:
    """Return a JSON-serializable source descriptor.

    The descriptor is persisted in the manifest and source_meta.json, then
    passed through ``ctx["meta"]`` for nodes that use intensity, returns,
    scan angle, RGB, GPS time, and other dimensions.
    """
    dimensions = [
        {"name": d.name, "dtype": str(d.dtype)}
        for d in header.point_format.dimensions
    ]
    vlrs = [
        {
            "user_id": v.user_id,
            "record_id": int(v.record_id),
            "description": (v.description or "")[:120],
        }
        for v in header.vlrs
    ]
    return {
        "version": str(header.version),
        "point_format_id": int(header.point_format.id),
        "point_count": int(header.point_count),
        "scales": [float(s) for s in header.scales],
        "offsets": [float(o) for o in header.offsets],
        "mins": [float(v) for v in header.mins[:3]],
        "maxs": [float(v) for v in header.maxs[:3]],
        "dimensions": dimensions,
        "vlrs": vlrs,
        "crs_resolved": _crs_from_rasterio(header) or "Unknown",
    }


class LiDARTiler:
    def __init__(
        self,
        source_path: str | Path,
        output_dir: str | Path,
        tile_size: float = 100.0,
        buffer_m: float = 10.0,
        max_points_per_tile: int = 0,
        chunk_size: int = CHUNK_SIZE,
        buffer_flush_size: int = BUFFER_FLUSH_SIZE,
        memory_budget_bytes: int | None = None,
        output_format: str = "laz",
        max_workers: int | None = None,
        overwrite: bool = False,
        progress_callback: Callable | None = None,
        tile_callback: Callable[[str], None] | None = None,
        tag: str = "",
        acceleration: AccelerationInfo | None = None,
        provenance: dict | None = None,
    ):
        self.source_path = Path(source_path)
        self.output_dir = Path(output_dir)
        self.tile_size = tile_size
        self.buffer_m = buffer_m
        self.max_points = int(max_points_per_tile)
        self.chunk_size = chunk_size
        self.buffer_flush_size = buffer_flush_size
        self.memory_budget_bytes = int(
            default_memory_budget_bytes()
            if memory_budget_bytes is None
            else memory_budget_bytes
        )
        if output_format not in ("las", "laz"):
            raise ValueError(f"output_format must be 'las' or 'laz', got {output_format!r}")
        self.output_format = output_format
        self.max_workers = max_workers or (os.cpu_count() or 4)
        self.overwrite = overwrite
        self.progress_callback = progress_callback
        self.tile_callback = tile_callback
        self.tag = tag
        self.acceleration = acceleration
        self._provenance = provenance
        self._xp = gpu_array_module(acceleration) if acceleration else None
        self._backend = _laz_backend()
        # CuPy (GPU) arrays are not safe to share across decode threads; the
        # passes keep their legacy serial reader when acceleration is active.
        self._use_parallel_decode = self._xp is None
        # COPC sources always decode serially (see _read_grid): laspy
        # seek()+read_points() silently returns zero-filled records on the
        # COPC hierarchy layout, which once poisoned whole tiles with
        # (0,0,0) points clipped into tile (0,0).
        self._is_copc = False
        self._decode_workers = max(
            1, min((os.cpu_count() or 4) - 1, DECODE_MAX_WORKERS)
        )
        self._decode_window = max(
            1, min(self._decode_workers, DECODE_WINDOW)
        )

        self._io_executor = ThreadPoolExecutor(max_workers=MAX_IO_THREADS)
        self._io_semaphore = threading.Semaphore(IO_FUTURE_WINDOW)
        self._writers_lock = threading.Lock()
        self._tile_writers: dict[int, Any] = {}
        self._writers_order: list[int] = []
        self._writer_cap = CAP_WRITERS
        self._tile_paths: dict[int, Path] = {}
        self._tile_locks: dict[int, threading.Lock] = {}
        self._tile_stats: dict[int, dict[str, np.ndarray]] = {}
        self._tile_counts: dict[int, int] = {}
        self._tid_index: dict[int, TileIndex] = {}
        self._source_header: laspy.LasHeader | None = None
        self._flag_counts: dict[str, int] = {}
        self._flag_dims: tuple[str, ...] = ()
        self._density_first: int = 0
        self._density_class: dict[str, int] = {}
        self._density_returns: dict[str, int] = {}
        self._density_dims: tuple[str, ...] = ()
        self._grid_cols = 1
        self._buffer_flush = BUFFER_FLUSH_SIZE
        self._subdivided = False
        self._sub_levels: dict[int, int] = {}
        self._nonempty_bases: set[int] = set()
        self._aborted = False

    # ------------------------------------------------------------------
    # Public API.
    # ------------------------------------------------------------------

    def run(self, cancel_flag: Callable[[], bool] | None = None) -> TilingResult:
        manifest_path = self.output_dir / "tiling_manifest.json"
        if not self.overwrite and manifest_path.exists() and self._cache_is_fresh():
            logger.info("[Tiler] Valid cache; loading manifest...")
            data = json.loads(manifest_path.read_text())
            data.pop("source_size", None)
            data.pop("source_mtime", None)
            data.pop("manifest_version", None)
            return TilingResult(**data)

        try:
            return self._process(cancel_flag)
        finally:
            self._close_all_writers()
            self._io_executor.shutdown(wait=True)

    # ------------------------------------------------------------------
    # Cache.
    # ------------------------------------------------------------------

    def _cache_is_fresh(self) -> bool:
        manifest_path = self.output_dir / "tiling_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("manifest_version") != MANIFEST_VERSION:
                return False
            source_size = self.source_path.stat().st_size
            source_mtime = self.source_path.stat().st_mtime
            # Legacy manifests predate the format key and always hold LAZ
            # tiles; treat a missing key accordingly so existing caches reuse.
            manifest_format = str(manifest.get("output_format", "laz"))
            if manifest_format != str(self.output_format):
                return False
            key_params = {
                "tile_size_m": float(self.tile_size),
                "buffer_m": float(self.buffer_m),
                "max_points_per_tile": int(self.max_points),
            }
            for key, value in key_params.items():
                try:
                    if float(manifest.get(key, 0.0)) != float(value):
                        return False
                except (TypeError, ValueError):
                    return False
            # The tiles themselves must still be there, at their recorded
            # size: a valid manifest with deleted/truncated tiles must
            # re-tile instead of feeding downstream ghosts. Legacy manifests
            # predate file_size: existence alone still beats no check.
            for tile in manifest.get("tiles_generated", []) or []:
                if not isinstance(tile, dict):
                    continue
                path = tile.get("file")
                if not path:
                    return False
                try:
                    if not Path(path).is_file():
                        return False
                    expected = tile.get("file_size")
                    if expected is not None and Path(path).stat().st_size != int(expected):
                        return False
                except (OSError, TypeError, ValueError):
                    return False
            return (
                manifest.get("source_size") == source_size
                and manifest.get("source_mtime") == source_mtime
            )
        except Exception:
            return False

    def _manifest_meta(self) -> dict:
        stat = self.source_path.stat()
        return {
            "source_size": stat.st_size,
            "source_mtime": stat.st_mtime,
            "tile_size_m": float(self.tile_size),
            "buffer_m": float(self.buffer_m),
            "max_points_per_tile": int(self.max_points),
            "output_format": str(self.output_format),
        }

    # ------------------------------------------------------------------
    # Output writing.
    # ------------------------------------------------------------------

    def _tile_lock(self, tid: int) -> threading.Lock:
        with self._writers_lock:
            lock = self._tile_locks.get(tid)
            if lock is None:
                lock = threading.Lock()
                self._tile_locks[tid] = lock
            return lock

    def _open_tile_writer(self, tid: int):
        """Create a fresh writer, or reopen an evicted tile with append.

        ``mode="a"`` (laspy ``LasAppender``) resumes the LAZ stream, updating
        the header on close, so an evicted tile's later chunks keep appending
        to the same file without ever exceeding the open-handle cap.
        """
        index = self._tid_index.get(tid)
        if index is None:
            index = TileIndex(tid % self._grid_cols, tid // self._grid_cols, self.tag)
        out_path = self._tile_paths.get(tid)
        if out_path is None:
            out_path = self.output_dir / f"{index}.{self.output_format}"
            self._tile_paths[tid] = out_path
        if out_path.exists() and out_path.stat().st_size > 0:
            return laspy.open(
                str(out_path), mode="a", laz_backend=self._backend
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)

        header = self._source_header
        new_header = laspy.LasHeader(
            version=header.version, point_format=header.point_format
        )
        new_header.offsets = header.offsets.copy()
        new_header.scales = header.scales.copy()
        provenance.copy_product_vlrs(new_header, header)
        provenance.copy_header_identity(new_header, header)
        if self._provenance:
            provenance.fill_las_header(new_header, self._provenance)
        return laspy.open(
            str(out_path), mode="w", header=new_header, laz_backend=self._backend
        )

    def _get_tile_writer(self, tid: int):
        """Return the open writer for ``tid``, maintaining the LRU cap.

        Thread-safe wrt. the writer registry: the dict/order are guarded by
        ``_writers_lock`` and an evicted writer is closed under its own tile
        lock (never under ``_writers_lock``), so a close never races a
        concurrent ``write_points`` on the same object.
        """
        with self._writers_lock:
            writers = self._tile_writers
            order = self._writers_order
            existing = writers.get(tid)
            if existing is not None:
                try:
                    order.remove(tid)
                except ValueError:
                    pass
                order.append(tid)
                return existing
            lru = order.pop(0) if len(writers) >= self._writer_cap and order else None
            evicted = writers.pop(lru, None) if lru is not None else None

        if evicted is not None:
            self._close_writer(lru, evicted)

        writer = self._open_tile_writer(tid)
        with self._writers_lock:
            self._tile_writers[tid] = writer
            self._writers_order.append(tid)
        return writer

    def _close_writer(self, tid: int, writer: Any) -> None:
        with self._tile_lock(tid):
            try:
                writer.close()
            except Exception as e:
                logger.warning(f"[Tiler] error closing writer {tid}: {e}")

    def _write_points_to_tile(self, tid: int, data_chunks: list[np.ndarray]) -> None:
        with self._tile_lock(tid):
            writer = self._get_tile_writer(tid)
            header = writer.header
            write_fn = getattr(writer, "write_points", None)
            if write_fn is None:
                write_fn = writer.append_points
            for data in data_chunks:
                if data.size == 0:
                    continue
                record = laspy.ScaleAwarePointRecord(
                    data,
                    header.point_format,
                    scales=header.scales,
                    offsets=header.offsets,
                )
                write_fn(record)

    def _async_write_laz(self, tid: int, mem_buf: dict, io_futures: list) -> None:
        # Pass the buffered chunk list directly; the worker emits each chunk
        # with its own record, so we never materialize a ~2x transient copy.
        chunks = mem_buf[tid]
        mem_buf[tid] = []
        self._io_semaphore.acquire()
        future = self._io_executor.submit(
            self._write_points_to_tile, tid, chunks
        )
        io_futures.append(future)
        future.add_done_callback(lambda _f: self._io_semaphore.release())

    def _drain_futures(self, io_futures: list):
        """Wait for every in-flight write and surface the first failure."""
        first_error = None
        for future in as_completed(io_futures):
            try:
                future.result()
            except Exception as e:
                if first_error is None:
                    first_error = e
                else:
                    logger.warning(f"[Tiler] async write failed: {e}")
        io_futures.clear()
        return first_error

    def _close_all_writers(self) -> None:
        """Close every open writer. Idempotent; never raises.

        Snapshot under the registry lock, then close each writer under its
        tile lock, so a still-running write cannot be interrupted.
        """
        with self._writers_lock:
            snapshot = list(self._tile_writers.items())
            self._tile_writers.clear()
            self._writers_order.clear()
        for tid, writer in snapshot:
            self._close_writer(tid, writer)

    # ------------------------------------------------------------------
    # Processing.
    # ------------------------------------------------------------------

    def _read_grid(self) -> dict:
        """Open the source, resolve the base grid, and seed dimension ranges."""
        with laspy.open(
            str(self.source_path), laz_backend=self._backend
        ) as reader:
            header = reader.header
            self._source_header = header
            _require_no_waveform(header, self.source_path)
            self._is_copc = _has_copc_vlr(list(header.vlrs))
            if self._is_copc:
                # Span seekers are unusable on COPC (zero-filled reads);
                # fall back to the sequential chunk iterator below.
                self._use_parallel_decode = False
                logger.info(
                    "[Tiler] COPC source: serial decode (seek unsupported)"
                )
            x_min, y_min = float(header.mins[0]), float(header.mins[1])
            x_max, y_max = float(header.maxs[0]), float(header.maxs[1])
            cols = max(1, math.ceil((x_max - x_min) / self.tile_size))
            rows = max(1, math.ceil((y_max - y_min) / self.tile_size))
            self._grid_cols = cols
            self._dim_ranges: dict[str, list[float]] = {
                d.name: [np.inf, np.negative(np.inf)]
                for d in header.point_format.dimensions
            }
            self._buffer_flush = int(
                max(1, min(self.buffer_flush_size, self.max_points))
                if self.max_points > 0
                else self.buffer_flush_size
            )
            return {
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
                "cols": cols,
                "rows": rows,
                "tile_size": float(self.tile_size),
            }

    def _file_spans(self) -> list[tuple[int, int]]:
        """Contiguous point spans over the source (chunk_size each)."""
        total = int(self._source_header.point_count)
        return [
            (start, min(self.chunk_size, total - start))
            for start in range(0, total, self.chunk_size)
        ]

    def _decode_span(self, start: int, count: int):
        """Read + decode a contiguous span with an independent reader.

        Each span uses its own ``laspy.open``/laz decoder state, so spans can
        be decoded concurrently from worker threads without sharing readers.
        """
        with laspy.open(
            str(self.source_path), laz_backend=self._backend
        ) as reader:
            reader.seek(start)
            return reader.read_points(count)

    def _iter_chunks(self, cancel_flag=None):
        """Yield decoded point records in exact file order.

        Serial path (CuPy acceleration active, single worker, or ≤1 span)
        mirrors the legacy ``chunk_iterator`` 1:1. Parallel path decodes spans
        on a bounded worker pool (native LAZ decode releases the GIL, so many
        cores decode at once) while the consumer routes strictly in file order.
        Every downstream routing/flush decision and every tile byte is
        therefore identical to the serial run and independent of the number of
        workers — the output replicates deterministically across machines.
        """
        spans = self._file_spans()
        if (
            not self._use_parallel_decode
            or self._decode_workers <= 1
            or len(spans) <= 1
        ):
            with laspy.open(
                str(self.source_path), laz_backend=self._backend
            ) as reader:
                if self._is_copc:
                    # No seek on COPC: laspy seek()+read_points() returns
                    # zero-filled records on hierarchy layouts. The plain
                    # chunk iterator decodes correctly, one chunk at a time.
                    for chunk in reader.chunk_iterator(self.chunk_size):
                        if cancel_flag is not None and cancel_flag():
                            return
                        yield chunk
                    return
                for start, count in spans:
                    if cancel_flag is not None and cancel_flag():
                        return
                    reader.seek(start)
                    yield reader.read_points(count)
            return

        window = min(self._decode_window, len(spans))
        with ThreadPoolExecutor(max_workers=self._decode_workers) as ex:
            futures: dict = {}
            ready: dict = {}
            submitted = 0
            yielded = 0
            while yielded < len(spans):
                if cancel_flag is not None and cancel_flag():
                    break
                while len(futures) < window and submitted < len(spans):
                    start, count = spans[submitted]
                    futures[ex.submit(self._decode_span, start, count)] = (
                        submitted
                    )
                    submitted += 1
                if not futures:
                    break
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for fut in list(done):
                    ready[futures.pop(fut)] = fut.result()
                while yielded in ready:
                    yield ready.pop(yielded)
                    yielded += 1
            # Generator close (e.g. the caller cancelled mid-pass) lets the
            # executor finish only the already submitted (finite) spans.

    def _process(self, cancel_flag=None) -> TilingResult:
        grid = self._read_grid()
        self._aborted = False

        levels = None
        if self.max_points > 0:
            levels = self._density_pass(grid, cancel_flag)
            if levels is None:
                self._aborted = True
        if not self._aborted:
            self._assign_pass(grid, levels, cancel_flag)
        return self._finalize(grid, cancel_flag)

    # -- Pass A: density histogram and subdivision levels -----------------

    def _density_pass(
        self, grid: dict, cancel_flag
    ) -> np.ndarray | None:
        """Stream the source once, counting per-sub-cell densities.

        Returns a per-base-tile subdivision level array (``int8``) or None on
        cancellation. Levels are the smallest ``k <= K`` whose sub-tiles all
        fit under ``max_points_per_tile * CAP_SLACK``.
        """
        cols = grid["cols"]
        rows = grid["rows"]
        T = self.tile_size
        x_min, y_min = grid["x_min"], grid["y_min"]

        K = MAX_SUB_DIV
        while cols * rows * (1 << (2 * K)) > SUB_HIST_LIMIT and K > 0:
            K -= 1
        fine = 1 << K
        grid["K"] = K
        hist = np.zeros(cols * rows * fine * fine, dtype=np.int32)

        processed_pts = 0
        total_pts = int(self._source_header.point_count)
        for chunk in self._iter_chunks(cancel_flag):
            if cancel_flag is not None and cancel_flag():
                return None
            x = np.array(chunk.x)
            y = np.array(chunk.y)
            for name in self._dim_ranges:
                try:
                    arr = np.asarray(getattr(chunk, name.lower()))
                except AttributeError:
                    continue
                if arr.dtype.kind in "iuf":
                    rng = self._dim_ranges[name]
                    cmin, cmax = float(arr.min()), float(arr.max())
                    if cmin < rng[0]:
                        rng[0] = cmin
                    if cmax > rng[1]:
                        rng[1] = cmax

            c_lo = np.floor((x - x_min - self.buffer_m) / T).astype(np.int64).clip(0, cols - 1)
            c_hi = np.floor((x - x_min + self.buffer_m) / T).astype(np.int64).clip(0, cols - 1)
            r_lo = np.floor((y - y_min - self.buffer_m) / T).astype(np.int64).clip(0, rows - 1)
            r_hi = np.floor((y - y_min + self.buffer_m) / T).astype(np.int64).clip(0, rows - 1)

            for dc in range(2):
                for dr in range(2):
                    mask = (c_lo + dc <= c_hi) & (r_lo + dr <= r_hi)
                    if not mask.any():
                        continue
                    base = (r_lo[mask] + dr) * cols + (c_lo[mask] + dc)
                    if K:
                        local_x = x[mask] - (x_min + (c_lo[mask] + dc) * T)
                        local_y = y[mask] - (y_min + (r_lo[mask] + dr) * T)
                        cell = T / fine
                        sc = np.floor((local_x + self.buffer_m) / cell).astype(np.int64)
                        sr = np.floor((local_y + self.buffer_m) / cell).astype(np.int64)
                        np.clip(sc, 0, fine - 1, out=sc)
                        np.clip(sr, 0, fine - 1, out=sr)
                    else:
                        sc = np.zeros(base.shape, dtype=np.int64)
                        sr = np.zeros(base.shape, dtype=np.int64)
                    flat = (base * fine + sr) * fine + sc
                    # unique+scatter is ~4-6x faster than np.add.at and is
                    # bounded by the chunk size (unlike bincount, whose
                    # output can span the whole grid); counts are exact, so
                    # subdivision levels are unchanged.
                    vals, counts = np.unique(flat, return_counts=True)
                    hist[vals] += counts

            processed_pts += len(chunk)
            if self.progress_callback:
                self.progress_callback(processed_pts, total_pts)

        fine_h = hist.reshape(cols * rows, fine, fine)
        num_base = cols * rows
        levels_out = np.full(num_base, K, dtype=np.int8)
        thr = max(int(self.max_points * CAP_SLACK), 1)
        for k in range(K + 1):
            side = fine >> k  # level-k tile spans this many fine cells
            groups = fine // side
            tall = fine_h.reshape(num_base, groups, side, groups, side)
            tile_count = tall.sum(axis=(2, 4))  # (num_base, groups, groups)
            best = tile_count.max(axis=(1, 2))
            take = best <= thr
            if take.any():
                candidate = np.where(take, np.array(k, dtype=np.int8), levels_out)
                levels_out = np.minimum(levels_out, candidate)
        self._sub_levels = {int(i): int(v) for i, v in enumerate(levels_out)}
        return levels_out

    # -- Pass B: assignment ----------------------------------------------

    def _assign_pass(self, grid: dict, levels, cancel_flag=None) -> None:
        """Route every point to its final tile and buffer to LAZ.

        ``levels`` None keeps the legacy single-pass behavior (no point cap);
        an int8 array enables adaptive subdivision.
        """
        self._subdivided = levels is not None
        self._tid_index = {}
        self._tile_stats = {}
        self._tile_counts = {}
        self._nonempty_bases = set()

        cols = grid["cols"]
        rows = grid["rows"]
        T = grid["tile_size"]
        x_min, y_min = grid["x_min"], grid["y_min"]

        levels_arr = np.asarray(levels) if self._subdivided else None

        total_pts = int(self._source_header.point_count)
        xp = self._xp
        array_api = np if self._subdivided else (xp or np)
        try:
            _dims = set(self._source_header.point_format.dimension_names)
        except Exception:
            _dims = set()
        self._flag_dims = tuple(
            d for d in ("withheld", "overlap", "synthetic", "key_point")
            if d in _dims
        )
        self._density_dims = tuple(
            d for d in ("classification", "return_number")
            if d in _dims
        )

        tile_mem_buffer: dict[int, list[np.ndarray]] = {}
        tile_mem_counts: dict[int, int] = {}
        tile_mem_bytes: dict[int, int] = {}
        buffered_total = 0
        buffered_bytes = 0
        memory_budget_bytes = self.memory_budget_bytes
        processed_pts = 0
        io_futures: list = []
        cancelled = False

        for chunk in self._iter_chunks(cancel_flag):
            if cancel_flag is not None and cancel_flag():
                cancelled = True
                break
            if self._flag_dims:
                # Vendor QA flag census (each source point once; stats
                # must never fail tiling).
                try:
                    for _d in self._flag_dims:
                        self._flag_counts[_d] = self._flag_counts.get(_d, 0) + int(
                            np.asarray(getattr(chunk, _d)).sum()
                        )
                except Exception:
                    pass
            if self._density_dims:
                # Density census for QA (first returns + histograms; same
                # once-per-point, never-fail contract as flags).
                try:
                    if "classification" in self._density_dims:
                        for _k, _v in zip(*np.unique(
                            np.asarray(chunk.classification), return_counts=True
                        )):
                            _kk = str(int(_k))
                            self._density_class[_kk] = (
                                self._density_class.get(_kk, 0) + int(_v)
                            )
                    if "return_number" in self._density_dims:
                        _rn = np.asarray(chunk.return_number)
                        self._density_first += int((_rn == 1).sum())
                        for _k, _v in zip(*np.unique(
                            np.minimum(_rn, 15), return_counts=True
                        )):
                            _kk = str(int(_k))
                            self._density_returns[_kk] = (
                                self._density_returns.get(_kk, 0) + int(_v)
                            )
                except Exception:
                    pass
            raw_array = np.array(chunk.array)
            x = np.array(chunk.x)
            y = np.array(chunk.y)
            z = np.array(chunk.z)
            if self._subdivided:
                x = np.asarray(x)
                y = np.asarray(y)
                z = np.asarray(z)
            elif xp is not None:
                x = xp.asarray(x)
                y = xp.asarray(y)
                z = xp.asarray(z)

            if not self._subdivided:
                for name in self._dim_ranges:
                    try:
                        col = getattr(chunk, name.lower())
                    except AttributeError:
                        continue
                    arr = np.asarray(col)
                    if arr.dtype.kind in "iuf":
                        rng = self._dim_ranges[name]
                        cmin, cmax = float(arr.min()), float(arr.max())
                        if cmin < rng[0]:
                            rng[0] = cmin
                        if cmax > rng[1]:
                            rng[1] = cmax

            c_lo = array_api.floor((x - x_min - self.buffer_m) / T).astype(np.int32).clip(0, cols - 1)
            c_hi = array_api.floor((x - x_min + self.buffer_m) / T).astype(np.int32).clip(0, cols - 1)
            r_lo = array_api.floor((y - y_min - self.buffer_m) / T).astype(np.int32).clip(0, rows - 1)
            r_hi = array_api.floor((y - y_min + self.buffer_m) / T).astype(np.int32).clip(0, rows - 1)

            all_tids_list: list[np.ndarray] = []
            all_pids_list: list[np.ndarray] = []
            for dc in range(2):
                for dr in range(2):
                    mask = (c_lo + dc <= c_hi) & (r_lo + dr <= r_hi)
                    if not bool(array_api.any(mask)):
                        continue
                    pids = array_api.flatnonzero(mask)
                    if self._subdivided:
                        base = (r_lo[mask] + dr) * cols + (c_lo[mask] + dc)
                        lvl = levels_arr[base]
                        if (lvl > 0).any():
                            step = np.left_shift(
                                np.ones_like(lvl, dtype=np.int64), lvl.astype(np.int64)
                            )
                            local_x = x[mask] - (x_min + (c_lo[mask] + dc) * T)
                            local_y = y[mask] - (y_min + (r_lo[mask] + dr) * T)
                            cell = T / step
                            sc = np.floor((local_x + self.buffer_m) / cell).astype(np.int64)
                            sr = np.floor((local_y + self.buffer_m) / cell).astype(np.int64)
                            np.clip(sc, 0, step - 1, out=sc)
                            np.clip(sr, 0, step - 1, out=sr)
                            sub_idx = sr * step + sc
                            tids = base.astype(np.int64) * 256 + sub_idx
                        else:
                            tids = base.astype(np.int64) * 256
                    else:
                        tids = (r_lo[mask] + dr) * cols + (c_lo[mask] + dc)
                    all_tids_list.append(tids)
                    all_pids_list.append(pids)

            if not all_tids_list:
                processed_pts += len(chunk)
                if self.progress_callback:
                    self.progress_callback(processed_pts, total_pts)
                continue

            all_tids = array_api.concatenate(all_tids_list).astype(np.int32)
            all_pids = array_api.concatenate(all_pids_list)

            # Groups are already ordered by tile ID; no second sort needed.
            order = array_api.argsort(all_tids, kind="stable")
            sorted_tids = all_tids[order]
            sorted_pids = all_pids[order]
            split_at = array_api.flatnonzero(sorted_tids[1:] != sorted_tids[:-1]) + 1
            groups = array_api.split(sorted_pids, split_at)
            first_index = array_api.asarray([0], dtype=split_at.dtype)
            u_tids = sorted_tids[array_api.concatenate((first_index, split_at))]
            if xp is not None and not self._subdivided:
                groups = [xp.asnumpy(group) for group in groups]
                u_tids = xp.asnumpy(u_tids)

            for tid_val, group in zip(u_tids, groups):
                tid_i = int(tid_val)
                if tid_i not in tile_mem_buffer:
                    tile_mem_buffer[tid_i] = []
                    tile_mem_counts[tid_i] = 0
                    tile_mem_bytes[tid_i] = 0
                    self._tile_stats[tid_i] = {
                        "min": np.array([np.inf] * 3),
                        "max": np.array([-np.inf] * 3),
                    }
                    self._tile_counts[tid_i] = 0
                    if self._subdivided:
                        base_tid = tid_i // 256
                        sub_idx = tid_i % 256
                        level_i = self._sub_levels.get(base_tid, 0)
                        step = 1 << level_i
                        srow, scol = divmod(sub_idx, step)
                        self._tid_index[tid_i] = TileIndex(
                            base_tid % cols,
                            base_tid // cols,
                            self.tag,
                            sub_level=level_i,
                            sub_row=srow,
                            sub_col=scol,
                        )
                        self._nonempty_bases.add(base_tid)
                    if self.tile_callback:
                        index = self._tid_index.get(tid_i)
                        if index is None:
                            index = TileIndex(
                                tid_i % cols, tid_i // cols, self.tag
                            )
                        self.tile_callback(str(index))

                group_arr = raw_array[group]
                group_bytes = int(group_arr.nbytes)
                tile_mem_buffer[tid_i].append(group_arr)
                tile_mem_counts[tid_i] += group.size
                self._tile_counts[tid_i] += group.size
                tile_mem_bytes[tid_i] += group_bytes
                buffered_total += group.size
                buffered_bytes += group_bytes
                stats_x = xp.asnumpy(x[group]) if (xp is not None and not self._subdivided) else x[group]
                stats_y = xp.asnumpy(y[group]) if (xp is not None and not self._subdivided) else y[group]
                stats_z = xp.asnumpy(z[group]) if (xp is not None and not self._subdivided) else z[group]
                self._update_stats(
                    self._tile_stats[tid_i], stats_x, stats_y, stats_z
                )

            # Flush tiles that reached their per-tile threshold.
            for tid_i, count in list(tile_mem_counts.items()):
                if count >= self._buffer_flush:
                    self._async_write_laz(tid_i, tile_mem_buffer, io_futures)
                    buffered_total -= count
                    buffered_bytes -= tile_mem_bytes[tid_i]
                    tile_mem_counts[tid_i] = 0
                    tile_mem_bytes[tid_i] = 0

            # Flush while the global buffered-BYTE budget is exceeded.
            # Picking the heaviest bucket keeps peak RAM bounded without
            # burning the writer queue on many tiny partial tiles.
            while buffered_bytes > memory_budget_bytes:
                tid_big = max(tile_mem_bytes, key=tile_mem_bytes.get)
                count_big = tile_mem_counts[tid_big]
                if count_big == 0:
                    break
                self._async_write_laz(tid_big, tile_mem_buffer, io_futures)
                buffered_total -= count_big
                buffered_bytes -= tile_mem_bytes[tid_big]
                tile_mem_counts[tid_big] = 0
                tile_mem_bytes[tid_big] = 0

            processed_pts += len(chunk)
            if self.progress_callback:
                self.progress_callback(processed_pts, total_pts)
            logger.debug(f"[Tiler] {processed_pts / total_pts:.1%} — {processed_pts:,} pts")

        # Draining covers both success and cancellation: pending futures are
        # awaited so `run().finally` never closes writers over live I/O.
        if cancelled:
            self._drain_futures(io_futures)
            return

        # Flush remaining tile buffers. Skipped on cancellation: the partial
        # tiles must never become a reusable cache entry.
        for tid in list(tile_mem_buffer.keys()):
            if tile_mem_buffer[tid]:
                self._async_write_laz(tid, tile_mem_buffer, io_futures)

        # Surface async write failures: a disk/handle error must fail the run
        # instead of leaving silently truncated tiles.
        write_error = self._drain_futures(io_futures)
        self._close_all_writers()
        if write_error is not None:
            raise write_error

        # Possible final tiles for the emitted tiles_skipped_empty metric.
        if self._subdivided:
            possible = sum(4 ** self._sub_levels.get(tid, 0) for tid in self._nonempty_bases)
        else:
            possible = cols * rows
        grid["possible"] = possible

    # -- Finalization -----------------------------------------------------

    def _finalize(self, grid: dict, cancel_flag) -> TilingResult:
        cols = grid["cols"]
        rows = grid["rows"]
        T = grid["tile_size"]

        cancelled = self._aborted or bool(cancel_flag and cancel_flag())
        if cancelled:
            # An aborted run must not poison the shared tile cache: without a
            # manifest the next run re-tiles the source in full. Partial .laz
            # files are left behind only until the next run overwrites them.
            for stale in ("tiling_manifest.json", "source_meta.json"):
                try:
                    (self.output_dir / stale).unlink()
                except OSError:
                    pass
            return TilingResult(
                source_file=str(self.source_path.absolute()),
                output_dir=str(self.output_dir.absolute()),
                crs="Unknown",
                total_points=0,
                tile_size_m=float(self.tile_size),
                buffer_m=float(self.buffer_m),
                max_points_per_tile=int(self.max_points),
                max_subdivision=int(self._sub_levels.get(0, 0) if self._subdivided else 0),
                tiles_generated=[],
                tiles_skipped_empty=0,
                source_meta={},
            )

        final_info = []
        for tid_int, stats in self._tile_stats.items():
            count = self._tile_counts.get(tid_int, 0)
            if count == 0:
                continue
            index = self._tid_index.get(tid_int)
            if index is None:
                index = TileIndex(tid_int % cols, tid_int // cols, self.tag)
            tile_file = self.output_dir / f"{index}.{self.output_format}"
            try:
                tile_size = tile_file.stat().st_size
            except OSError:
                tile_size = 0
            x0 = grid["x_min"] + index.col * T
            y0 = grid["y_min"] + index.row * T
            w = T
            if index.sub_level:
                s = 1 << index.sub_level
                side = T / s
                x0 += index.sub_col * side
                y0 += index.sub_row * side
                w = side
            final_info.append(
                {
                    "tile_id": str(index),
                    "file": str(tile_file.absolute()),
                    "col": int(index.col),
                    "row": int(index.row),
                    "sub_level": int(index.sub_level),
                    "sub_row": int(index.sub_row),
                    "sub_col": int(index.sub_col),
                    "point_count": count,
                    "file_size": tile_size,
                    "x_min": float(stats["min"][0]),
                    "y_min": float(stats["min"][1]),
                    "x_max": float(stats["max"][0]),
                    "y_max": float(stats["max"][1]),
                    "buffer_m": float(self.buffer_m),
                    "core_x_min": x0,
                    "core_y_min": y0,
                    "core_x_max": x0 + w,
                    "core_y_max": y0 + w,
                }
            )

        final_info.sort(key=lambda item: (item["row"], item["col"], item["sub_row"], item["sub_col"]))
        ranges = {}
        for name, rng in self._dim_ranges.items():
            if np.isfinite(rng[0]) and np.isfinite(rng[1]):
                ranges[name] = [round(float(rng[0]), 6), round(float(rng[1]), 6)]
        with laspy.open(
            str(self.source_path), laz_backend=self._backend
        ) as reader:
            source_meta = describe_source(reader.header)
        source_meta["dimension_ranges"] = ranges

        possible = grid.get("possible", cols * rows)
        result = TilingResult(
            source_file=str(self.source_path.absolute()),
            output_dir=str(self.output_dir.absolute()),
            crs=_extract_crs(self._source_header) if self._source_header is not None else "Unknown",
            total_points=int(
                self._source_header.point_count if self._source_header else 0
            ),
            tile_size_m=float(self.tile_size),
            buffer_m=float(self.buffer_m),
            max_points_per_tile=int(self.max_points),
            max_subdivision=int(
                max((self._sub_levels.get(tid, 0) for tid in self._nonempty_bases), default=0)
                if self._subdivided
                else 0
            ),
            tiles_generated=final_info,
            tiles_skipped_empty=int(max(0, possible - len(final_info))),
            source_meta=source_meta,
            flag_counts=dict(self._flag_counts),
            density_counts={
                "first_returns": int(self._density_first),
                "class_histogram": dict(self._density_class),
                "return_histogram": dict(self._density_returns),
            },
        )
        result_json = result.to_json()
        manifest = json.loads(result_json)
        manifest["manifest_version"] = MANIFEST_VERSION
        manifest.update(self._manifest_meta())
        (self.output_dir / "tiling_manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )
        (self.output_dir / "source_meta.json").write_text(
            json.dumps(source_meta, indent=2)
        )
        return result

    @staticmethod
    def _update_stats(stat_dict, x, y, z) -> None:
        stat_dict["min"] = np.minimum(stat_dict["min"], [x.min(), y.min(), z.min()])
        stat_dict["max"] = np.maximum(stat_dict["max"], [x.max(), y.max(), z.max()])