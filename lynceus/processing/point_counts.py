# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Cheap LAS/LAZ header reads for the pre-flight advisor (domain, no Qt).

The advisor runs on the GUI thread before a run starts, so it must not pay
the cost of importing ``laspy`` or decoding point records: it only needs the
point count and the coordinate bounding box to size the workload. Both live
in the fixed-size LAS header at offsets mandated by the ASPRS LAS spec, so a
single ``read`` of the first 375 bytes plus ``struct.unpack_from`` is enough
(~microseconds per file).

Works for LAZ too: the compressing codecs keep the header (including the
point count) intact, and the point-data-format byte only carries a
compression flag that does not shift any of the offsets used here.
"""

from __future__ import annotations

import struct
from pathlib import Path

#: LAS public header block is 227 bytes for versions <= 1.3; 1.4 extends it to
#: 375. The extra fields (64-bit point count, EVLR offsets) sit at the tail.
_HEADER_READ_BYTES = 375
_MIN_HEADER_BYTES = 227
_LAS_SIGNATURE = b"LASF"

# Offsets into the fixed header (ASPRS LAS 1.4 R15).
_OFF_VERSION_MINOR = 25
_OFF_HEADER_SIZE = 94
_OFF_POINT_DATA_OFFSET = 96
_OFF_VLR_COUNT = 100
_OFF_POINT_FORMAT = 104
_OFF_LEGACY_COUNT = 107
_OFF_MAX_X = 179
_OFF_MIN_X = 187
_OFF_MAX_Y = 195
_OFF_MIN_Y = 203
_OFF_MAX_Z = 211
_OFF_MIN_Z = 219
_OFF_EXTENDED_COUNT = 247  # LAS 1.4 only (uint64)

_VLR_HEADER_LEN = 54
_WKT_RECORD_ID = 2112
#: Upper bound on the VLR region we are willing to read for CRS sniffing.
_MAX_VLR_BYTES = 65536


def _detect_geographic(
    path: str | Path,
    header_size: int,
    point_data_offset: int,
    vlr_count: int,
) -> bool | None:
    """True/False if the CRS VLR says geographic/projected, else ``None``.

    Reads only the VLR region (capped) and looks for the OGC WKT VLR
    (record id 2112). WKT1 uses ``GEOGCS``/``PROJCS``, WKT2 ``GEOGCRS``/
    ``PROJCRS``. Absent CRS metadata returns ``None`` (caller assumes a
    linear CRS, the overwhelmingly common case for production clouds).
    """
    if vlr_count <= 0 or header_size <= 0 or point_data_offset <= header_size:
        return None
    length = min(point_data_offset - header_size, _MAX_VLR_BYTES)
    try:
        with open(path, "rb") as fh:
            fh.seek(header_size)
            blob = fh.read(length)
    except OSError:
        return None
    pos = 0
    for _ in range(vlr_count):
        if pos + _VLR_HEADER_LEN > len(blob):
            break
        record_id = struct.unpack_from("<H", blob, pos + 18)[0]
        rec_len = struct.unpack_from("<H", blob, pos + 20)[0]
        payload = blob[pos + _VLR_HEADER_LEN : pos + _VLR_HEADER_LEN + rec_len]
        if record_id == _WKT_RECORD_ID:
            text = payload.rstrip(b"\x00").decode("utf-8", "ignore").upper()
            # Projected check first: a PROJCS/PROJCRS string embeds the
            # GEOGCS/GEOGCRS datum block, so ordering matters.
            if "PROJCS" in text or "PROJCRS" in text:
                return False
            if "GEOGCS" in text or "GEOGCRS" in text:
                return True
        pos += _VLR_HEADER_LEN + rec_len
    return None


def read_header(path: str | Path) -> dict | None:
    """Decode the fields the advisor needs, or ``None`` if unreadable.

    Returns ``{point_count, version_minor, point_format, geographic, bounds}``
    where ``bounds`` is ``(min_x, min_y, max_x, max_y)`` or ``None`` when the
    header carries no usable metric extent (no extent, or a geographic CRS
    whose degrees cannot be squared into meters). Never raises: a corrupt or
    missing file simply yields ``None`` so the advisor degrades to "no
    estimate".
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(_HEADER_READ_BYTES)
    except OSError:
        return None
    if len(head) < _MIN_HEADER_BYTES or head[:4] != _LAS_SIGNATURE:
        return None
    try:
        version_minor = head[_OFF_VERSION_MINOR]
        header_size = struct.unpack_from("<H", head, _OFF_HEADER_SIZE)[0]
        point_data_offset = struct.unpack_from("<I", head, _OFF_POINT_DATA_OFFSET)[0]
        vlr_count = struct.unpack_from("<I", head, _OFF_VLR_COUNT)[0]
        point_format = head[_OFF_POINT_FORMAT] & 0x3F
        legacy_count = struct.unpack_from("<I", head, _OFF_LEGACY_COUNT)[0]
        count = legacy_count
        if (
            version_minor >= 4
            and header_size >= _HEADER_READ_BYTES
            and len(head) >= _OFF_EXTENDED_COUNT + 8
        ):
            extended = struct.unpack_from("<Q", head, _OFF_EXTENDED_COUNT)[0]
            # 1.4 writers may leave the legacy field zero: the 64-bit one
            # is authoritative when present and non-zero.
            if extended > 0:
                count = extended
        geographic = _detect_geographic(
            path, header_size, point_data_offset, vlr_count
        )
        bounds = None
        if len(head) >= _OFF_MIN_Z + 8:
            max_x = struct.unpack_from("<d", head, _OFF_MAX_X)[0]
            min_x = struct.unpack_from("<d", head, _OFF_MIN_X)[0]
            max_y = struct.unpack_from("<d", head, _OFF_MAX_Y)[0]
            min_y = struct.unpack_from("<d", head, _OFF_MIN_Y)[0]
            # Geographic (degrees) extents cannot yield a metric area: drop
            # the bounds so callers fall back to a point-only estimate.
            if geographic is not True and max_x > min_x and max_y > min_y:
                bounds = (min_x, min_y, max_x, max_y)
    except struct.error:
        return None
    return {
        "point_count": int(count),
        "version_minor": int(version_minor),
        "point_format": int(point_format),
        "geographic": geographic,
        "bounds": bounds,
    }


def point_count(path: str | Path) -> int | None:
    """Header point count for a LAS/LAZ file, or ``None`` if unreadable."""
    header = read_header(path)
    if header is None:
        return None
    return header["point_count"]


def union_area_m2(paths: list[str | Path]) -> tuple[int, float | None]:
    """Total header point count and the union bbox area in square meters.

    ``area`` is ``None`` when no readable header carries an extent (or the
    files have no spatial overlap math possible), so callers can fall back to
    a point-only heuristic. Unreadable files contribute neither points nor
    area but never raise.
    """
    total = 0
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    any_bounds = False
    for path in paths:
        header = read_header(path)
        if header is None:
            continue
        total += header["point_count"]
        bounds = header["bounds"]
        if bounds is None:
            # A geographic CRS drops bounds on purpose (degrees are not
            # meters): keep adding its points, contribute no area.
            continue
        any_bounds = True
        min_x = min(min_x, bounds[0])
        min_y = min(min_y, bounds[1])
        max_x = max(max_x, bounds[2])
        max_y = max(max_y, bounds[3])
    if not any_bounds or max_x <= min_x or max_y <= min_y:
        return total, None
    width = max_x - min_x
    height = max_y - min_y
    return total, float(width * height)
