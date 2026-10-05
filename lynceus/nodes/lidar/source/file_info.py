# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Node -- File Info (LiDAR source inspector).

Barrier-only: reads each connected LiDAR source once and writes a
structured, typed, hierarchical JSON report (header facts, CRS, channels,
classifications, flags, flight lines, VLRs, derived densities and QA
verdicts). One streaming pass with bounded memory; one report per loader.
"""

from __future__ import annotations

from lynceus.nodes.ports import PortType

NODE_ID = "lynceus.nodes.lidar.source.file_info"
NODE_NAME = "File Info"
NODE_CATEGORY = "LiDAR"

NODE_SUBCATEGORY = "Source"

NODE_DESCRIPTION = (
    "<b>File Info</b> -- LiDAR File Inspector<br><br>"
    "Reads a connected LiDAR source once and writes a structured JSON "
    "report: format and header facts, CRS, channels, classifications, "
    "flags, flight lines, VLRs, derived densities and QA verdicts.<br><br>"
    "<b>Process:</b> Single streaming pass over the source file "
    "(memory-bounded); one report per connected loader.<br><br>"
    "<b>Tips:</b> Wire it alone after a loader for a quick survey audit "
    "before heavy processing."
)

INPUTS = (PortType.POINT_CLOUD,)
OUTPUTS = ()

PROCESSING_SPECS = {
    "barrier_task": "barrier_file_info",
    "output_globs": ("*_file_info.json",),
    "config_schema": {},
}

# Normative thresholds (ASPRS/USGS; fixed spec, no operator tuning).
WITHHELD_WARN_PCT = 2.0
NPS_QL1_M = 0.35
NPD_QL1_PTS_M2 = 8.0
ASPECT_CORRIDOR_RATIO = 4.0
RAM_FACTOR = 4
RESOLUTION_PLACES = 6
# Inter-strip consistency (planar-patch RMSD): coarse cells, planarity
# gate, verdict threshold per ASPRS 2024 (RMSDz <= 8 cm).
STRIP_CELL_M = 5.0
STRIP_MIN_PTS = 5
STRIP_PLANAR_M = 0.5
STRIP_RMSD_M = 0.08
STRIP_CELL_CAP = 1_000_000


def get_config_defaults() -> dict:
    return {}


def _r4(value):
    """Round floats for byte-stable JSON (None-safe)."""
    if value is None:
        return None
    try:
        return round(float(value), RESOLUTION_PLACES)
    except (TypeError, ValueError):
        return None


def _version_tuple(version) -> tuple | None:
    try:
        return tuple(int(p) for p in str(version).split("."))
    except (TypeError, ValueError):
        return None


def _crs_details(header):
    """(wkt_or_none, epsg_or_none, wkt_valid)."""
    wkt = None
    for vlr in header.vlrs:
        try:
            if getattr(vlr, "record_id", None) != 2112:
                continue
            raw = bytes(vlr.bytes)
        except Exception:
            continue
        try:
            text = raw.decode("utf-8", errors="replace").strip()
        except Exception:
            continue
        if text:
            wkt = text
            break
    if not wkt:
        try:
            from lynceus.processing.tiler import _crs_from_rasterio

            wkt = _crs_from_rasterio(header)
        except Exception:
            wkt = None
    if not wkt:
        return None, None, False
    try:
        from rasterio.crs import CRS

        crs = CRS.from_string(wkt)
    except Exception:
        return wkt, None, False
    try:
        epsg = crs.to_epsg()
    except Exception:
        epsg = None
    return wkt, epsg, True


def _inspect_file(path: str) -> dict:
    """Build the 7-block report for one source file (single pass)."""
    import numpy as np

    import laspy
    from laspy import DecompressionSelection

    from lynceus.processing.tiler import _laz_backend

    decompression_selection = (
        DecompressionSelection.XY_RETURNS_CHANNEL
        | DecompressionSelection.Z
        | DecompressionSelection.CLASSIFICATION
        | DecompressionSelection.FLAGS
        | DecompressionSelection.INTENSITY
        | DecompressionSelection.SCAN_ANGLE
        | DecompressionSelection.POINT_SOURCE_ID
        | DecompressionSelection.GPS_TIME
    )
    with laspy.open(
        path,
        laz_backend=_laz_backend(),
        decompression_selection=decompression_selection,
    ) as reader:
        header = reader.header
        fmt = header.point_format
        dims = set(fmt.dimension_names)
        version = str(header.version)
        pdrf = int(fmt.id)
        total = int(header.point_count)
        try:
            record_len = int(fmt.size)
        except Exception:
            record_len = 0
        try:
            created = header.creation_date
            day_of_year = int(created.timetuple().tm_yday)
            year = int(created.year)
        except Exception:
            day_of_year, year = None, None
        generator = (
            str(getattr(header, "generating_software", "") or "").strip()
            or str(getattr(header, "system_identifier", "") or "").strip()
        )

        vlrs = list(header.vlrs)
        try:
            evlrs = list(header.evlrs)
        except Exception:
            evlrs = []
        vlr_count, evlr_count = len(vlrs), len(evlrs)
        from lynceus.processing.tiler import _has_copc_vlr

        copc = _has_copc_vlr(vlrs) or _has_copc_vlr(evlrs)
        extra = []
        try:
            for dim in fmt.extra_dimensions:
                extra.append({
                    "name": str(getattr(dim, "name", "")),
                    "data_type": str(getattr(dim, "dtype", "")),
                    "description": str(getattr(dim, "description", "") or ""),
                })
        except Exception:
            extra = []

        try:
            by_return = [int(v) for v in header.number_of_points_by_return]
        except Exception:
            by_return = []
        by_return = (by_return + [0] * 15)[:15]

        has_gps = "gps_time" in dims
        has_intensity = "intensity" in dims
        has_rgb = {"red", "green", "blue"} <= dims
        has_nir = "nir" in dims
        has_waveform = pdrf in (9, 10) or "waveform" in dims
        has_channel = "scanner_channel" in dims
        has_angle = "scan_angle" in dims
        has_rank = "scan_angle_rank" in dims
        has_returns = "return_number" in dims
        has_class = "classification" in dims

        gps_type = None
        if has_gps:
            try:
                gps_type = (
                    "adjusted_standard"
                    if int(header.global_encoding.gps_time_type) == 1
                    else "week_time"
                )
            except Exception:
                gps_type = None

        wdp = False
        if has_waveform:
            from pathlib import Path as _Path

            try:
                wdp = _Path(path).with_suffix(".wdp").is_file()
            except Exception:
                wdp = False

        # Streaming accumulators (bounded memory).
        gps_min, gps_max = np.inf, float("-inf")
        int_min, int_max = None, None
        int_depth = None
        ch_counts = np.zeros(4, dtype=np.int64)
        ang_sum, ang_n = 0.0, 0
        ang_min, ang_max = np.inf, float("-inf")
        ang_extreme = 0
        class_hist = np.zeros(256, dtype=np.int64)
        flag_counts = {"withheld": 0, "overlap": 0,
                       "synthetic": 0, "key_point": 0}
        first_returns = 0
        first_clean = 0
        ground_count = 0
        has_lines = "point_source_id" in dims
        line_counts = np.zeros(1 << 16, dtype=np.int64) if has_lines else None
        # Inter-strip consistency: (line, cell) -> [n, sum, min, max] on a
        # coarse grid; planar cells shared by 2+ lines feed the RMSD.
        try:
            _ox, _oy = float(header.mins[0]), float(header.mins[1])
        except Exception:
            _ox, _oy = 0.0, 0.0
        strip: dict[tuple, list] = {}
        strip_truncated = False
        counted = 0
        for chunk in reader.chunk_iterator(2_000_000):
            n = len(chunk.array)
            if n == 0:
                continue
            counted += n
            if has_gps:
                g = np.asarray(chunk.gps_time, dtype=np.float64)
                gps_min = min(gps_min, float(g.min()))
                gps_max = max(gps_max, float(g.max()))
            if has_intensity:
                iv = np.asarray(chunk.intensity)
                if int_depth is None:
                    int_depth = int(iv.dtype.itemsize * 8)
                lo, hi = int(iv.min()), int(iv.max())
                int_min = lo if int_min is None else min(int_min, lo)
                int_max = hi if int_max is None else max(int_max, hi)
            if has_channel:
                ch = np.asarray(chunk.scanner_channel)
                ch_counts += np.bincount(ch[ch <= 3], minlength=4)[:4]
            if has_angle or has_rank:
                if has_angle:
                    raw = np.asarray(chunk.scan_angle, dtype=np.float32)
                    sample = raw[:min(raw.size, 1 << 16)]
                    if sample.size and np.nanmax(np.abs(sample)) > 90.0:
                        raw = raw * np.float32(0.006)
                else:
                    raw = np.asarray(chunk.scan_angle_rank, dtype=np.float32)
                finite = raw[np.isfinite(raw)]
                if finite.size:
                    ang_sum += float(finite.sum())
                    ang_n += int(finite.size)
                    ang_min = min(ang_min, float(finite.min()))
                    ang_max = max(ang_max, float(finite.max()))
                    ang_extreme += int((np.abs(finite) > 30.0).sum())
            if has_class:
                cls = np.asarray(chunk.classification)
                class_hist += np.bincount(cls, minlength=256)[:256]
                ground_count += int((cls == 2).sum())
            for flag in flag_counts:
                if flag in dims:
                    flag_counts[flag] += int(
                        np.asarray(getattr(chunk, flag)).sum()
                    )
            if has_returns:
                rn = np.asarray(chunk.return_number)
                first_returns += int((rn == 1).sum())
                clean = np.ones(rn.shape, dtype=bool)
                if "overlap" in dims:
                    clean &= ~np.asarray(chunk.overlap, dtype=bool)
                if "withheld" in dims:
                    clean &= ~np.asarray(chunk.withheld, dtype=bool)
                first_clean += int(((rn == 1) & clean).sum())
            if line_counts is not None:
                line_ids = np.asarray(chunk.point_source_id, dtype=np.uint16)
                line_counts += np.bincount(line_ids, minlength=1 << 16)
            if has_lines and not strip_truncated:
                _sx = np.asarray(chunk.x, dtype=np.float64)
                _sy = np.asarray(chunk.y, dtype=np.float64)
                _sz = np.asarray(chunk.z, dtype=np.float64)
                _ps = np.asarray(chunk.point_source_id)
                _cc = np.floor((_sx - _ox) / STRIP_CELL_M).astype(np.int64)
                _cr = np.floor((_sy - _oy) / STRIP_CELL_M).astype(np.int64)
                _c0, _r0 = int(_cc.min()), int(_cr.min())
                _cspan = int(_cc.max()) - _c0 + 1
                _rspan = int(_cr.max()) - _r0 + 1
                _max_code = (
                    ((65535 * _cspan + _cspan - 1) * _rspan) + _rspan - 1
                )
                if _max_code <= np.iinfo(np.int64).max:
                    _packed = (
                        (_ps.astype(np.int64) * _cspan + (_cc - _c0)) * _rspan
                        + (_cr - _r0)
                    )
                    _uk, _inv, _ct = np.unique(
                        _packed, return_inverse=True, return_counts=True
                    )
                    _packed_keys = True
                else:
                    _keys = np.stack([_ps.astype(np.int64), _cc, _cr], axis=1)
                    _uk, _inv, _ct = np.unique(
                        _keys, axis=0, return_inverse=True, return_counts=True
                    )
                    _packed_keys = False
                _sum = np.bincount(_inv, weights=_sz)
                _mn = np.full(len(_uk), np.inf)
                np.minimum.at(_mn, _inv, _sz)
                _mx = np.full(len(_uk), float("-inf"))
                np.maximum.at(_mx, _inv, _sz)
                for _index, (_code, _nn, _ss, _lo, _hi) in enumerate(zip(
                    _uk, _ct, _sum, _mn, _mx
                )):
                    if _packed_keys:
                        _quotient, _row_offset = divmod(int(_code), _rspan)
                        _line, _col_offset = divmod(_quotient, _cspan)
                        _kk = (_line, _c0 + _col_offset, _r0 + _row_offset)
                    else:
                        _row = _code
                        _kk = (int(_row[0]), int(_row[1]), int(_row[2]))
                    if _kk in strip:
                        _e = strip[_kk]
                        _e[0] += int(_nn)
                        _e[1] += float(_ss)
                        _e[2] = min(_e[2], float(_lo))
                        _e[3] = max(_e[3], float(_hi))
                    elif len(strip) < STRIP_CELL_CAP:
                        strip[_kk] = [
                            int(_nn), float(_ss), float(_lo), float(_hi)
                        ]
                    else:
                        strip_truncated = True
                        break

    mins = [float(v) for v in header.mins[:3]]
    maxs = [float(v) for v in header.maxs[:3]]
    scales = [float(v) for v in header.scales[:3]]
    offsets = [float(v) for v in header.offsets[:3]]
    dx, dy, dz = maxs[0] - mins[0], maxs[1] - mins[1], maxs[2] - mins[2]
    area = dx * dy if dx > 0 and dy > 0 else 0.0
    if area > 0:
        aspect = max(dx, dy) / min(dx, dy)
        aspect_label = (
            "LINEAR_CORRIDOR" if aspect >= ASPECT_CORRIDOR_RATIO
            else "BLOCK_POLYGON"
        )
    else:
        aspect, aspect_label = None, None

    duration = None
    rate = None
    if has_gps and gps_max >= gps_min and counted:
        duration = _r4(gps_max - gps_min)
        if duration:
            rate = _r4(total / duration)

    def _density(count):
        return _r4(count / area) if area > 0 else None

    usgs_npd = None
    usgs_nps = None
    if has_returns and area > 0:
        _exact_npd = first_clean / area
        usgs_npd = _r4(_exact_npd)
        if usgs_npd:
            usgs_nps = _r4(_exact_npd ** -0.5)
    compliant = (
        usgs_nps is not None and usgs_npd is not None
        and usgs_nps <= NPS_QL1_M and usgs_npd >= NPD_QL1_PTS_M2
    )

    # Inter-strip consistency (planar-patch RMSD): cells shared by 2+
    # lines with enough points and a small altitude range contribute
    # their line means; the verdict needs evaluated pairs.
    strip_cells: dict[tuple, list] = {}
    if has_lines and not strip_truncated:
        for (line, c, r), (n, s, lo, hi) in strip.items():
            if n >= STRIP_MIN_PTS and hi - lo <= STRIP_PLANAR_M:
                strip_cells.setdefault((c, r), []).append(s / n)
    strip_sq, strip_pairs = 0.0, 0
    for means in strip_cells.values():
        if len(means) >= 2:
            for i in range(len(means)):
                for j in range(i + 1, len(means)):
                    strip_sq += (means[i] - means[j]) ** 2
                    strip_pairs += 1
    strip_rmsdz = _r4((strip_sq / strip_pairs) ** 0.5) if strip_pairs else None
    strip_ok = (
        strip_rmsdz is not None and strip_rmsdz <= STRIP_RMSD_M
        if strip_pairs and not strip_truncated else None
    )

    version_t = _version_tuple(version)
    modern = version_t is not None and version_t >= (1, 4)
    class_dict = {str(k): int(v) for k, v in enumerate(class_hist) if v}
    line_ids = np.flatnonzero(line_counts) if line_counts is not None else ()
    line_distribution = {
        str(int(line_id)): int(line_counts[line_id]) for line_id in line_ids
    }
    reserved_alert = bool(
        modern and (class_dict.get("8", 0) > 0 or class_dict.get("12", 0) > 0)
    )
    withheld_pct = (
        _r4(100.0 * flag_counts["withheld"] / total) if total else None
    )

    wkt, epsg, wkt_valid = _crs_details(header)
    gps_adjusted = gps_type == "adjusted_standard"

    return {
        "file_info": {
            "las_version": version,
            "pdrf_id": pdrf,
            "total_points": total,
            "record_length_bytes": record_len,
            "generator_software": generator,
            "creation_date": {"day_of_year": day_of_year, "year": year},
            "raw_size_bytes": total * record_len,
            "processing_ram_estimate_bytes": total * record_len * RAM_FACTOR,
        },
        "spatial_crs": {
            "crs_wkt": wkt,
            "epsg_code": epsg,
            "bounding_box": {
                "x_min": _r4(mins[0]), "x_max": _r4(maxs[0]),
                "y_min": _r4(mins[1]), "y_max": _r4(maxs[1]),
                "z_min": _r4(mins[2]), "z_max": _r4(maxs[2]),
            },
            "scale": {"scale_x": scales[0], "scale_y": scales[1],
                      "scale_z": scales[2]},
            "offset": {"offset_x": offsets[0], "offset_y": offsets[1],
                       "offset_z": offsets[2]},
            "theoretical_resolution_limit": _r4(min(scales)) if scales else None,
        },
        "channel_attributes": {
            "gps_time": {
                "present": bool(has_gps),
                "type": gps_type,
                "min": _r4(gps_min) if has_gps and counted else None,
                "max": _r4(gps_max) if has_gps and counted else None,
            },
            "intensity": {
                "present": bool(has_intensity),
                "bit_depth": int_depth,
                "min": int_min,
                "max": int_max,
                "linearly_normalized": None,
            },
            "color_rgb": bool(has_rgb),
            "infrared_nir": bool(has_nir),
            "waveform": {"present": bool(has_waveform), "external_wdp": bool(wdp)},
            "scanner_channel": {
                "present": bool(has_channel),
                "distribution": {str(k): int(v) for k, v in enumerate(ch_counts)},
            },
            "scan_angle": {
                "available": bool(has_angle or has_rank),
                "min": _r4(ang_min) if ang_n else None,
                "max": _r4(ang_max) if ang_n else None,
                "mean": _r4(ang_sum / ang_n) if ang_n else None,
                "extreme_pct": _r4(100.0 * ang_extreme / ang_n) if ang_n else None,
            },
        },
        "classification_flags": {
            "class_histogram": class_dict,
            "returns": {
                "counts_1_to_15": by_return,
                "max_allowed": 15 if modern else 5,
                "multi_return_pct": (
                    _r4(100.0 * (total - first_returns) / first_returns)
                    if has_returns and first_returns > 0 else None
                ),
            },
            "qa_flags": {
                "pct_withheld": _r4(100.0 * flag_counts["withheld"] / total) if total else None,
                "pct_overlap": _r4(100.0 * flag_counts["overlap"] / total) if total else None,
                "pct_keypoint": _r4(100.0 * flag_counts["key_point"] / total) if total else None,
                "pct_synthetic": _r4(100.0 * flag_counts["synthetic"] / total) if total else None,
            },
            "flight_lines": {
                "total_unique": len(line_distribution),
                "points_per_line": line_distribution,
            },
        },
        "vlr_metadata": {
            "vlr_count": vlr_count,
            "evlr_count": evlr_count,
            "extra_bytes": {"present": bool(extra), "attributes": extra},
            "copc_ready": bool(copc),
        },
        "derived_metrics": {
            "coverage_area_m2": _r4(area),
            "elevation_range_m": _r4(dz),
            "area_aspect": aspect_label,
            "area_aspect_ratio": _r4(aspect),
            "flight_duration_s": duration,
            "acquisition_rate_pts_s": rate,
            "mean_density_pts_m2": _density(total),
            "mean_spacing_m": (
                _r4((1.0 / (total / area)) ** 0.5) if area > 0 and total > 0 else None
            ),
            "usgs_npd_pts_m2": usgs_npd,
            "usgs_nps_m": usgs_nps,
            "ground_class2_density_pts_m2": _density(ground_count),
        },
        "qa_qc_checks": {
            "crs_wkt_valid": bool(wkt_valid),
            "gps_time_adjusted": bool(gps_adjusted),
            "reserved_classes_alert": bool(reserved_alert),
            "high_noise_withheld_alert": bool(
                withheld_pct is not None and withheld_pct > WITHHELD_WARN_PCT
            ),
            "usgs_nps_compliant": bool(compliant),
            "strip_rmsdz_m": strip_rmsdz,
            "strip_pairs_evaluated": strip_pairs,
            "strip_consistency_ok": strip_ok,
        },
    }


def barrier_file_info(ctx: dict) -> dict:
    """Inspect each connected LiDAR source into a JSON report."""
    import json

    from pathlib import Path

    sources = ctx.get("provenance", {}).get("sources", []) or []
    if not sources:
        raise RuntimeError("File Info requires a LiDAR source: wire a loader")
    session = Path(ctx["session_dir"])
    node_iid = str(ctx.get("node_iid") or "file_info")
    out_dir = session / node_iid
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"node": NODE_ID, "kind": "File Info"}
    files = []
    for index, source in enumerate(sorted(sources, key=lambda s: s.get("iid", ""))):
        raw = source.get("file", "") or ""
        if not raw or not Path(raw).is_file():
            raise RuntimeError(f"File Info: source file not found: {raw or '(none)'}")
        tag = source.get("tag", "") or source.get("iid", "") or f"src{index}"
        out_path = out_dir / f"{tag}_file_info.json"
        doc = _inspect_file(str(raw))
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(doc, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
        files.append(str(out_path))
    payload["file"] = files[0]
    for pos, path in enumerate(files[1:], start=2):
        payload[f"file_{pos}"] = path
    return payload
