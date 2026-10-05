# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Tree viewer for structured JSON products (file info, quality reports).

Single-column tree (``key: value`` inline, no headers). A guide bar
explains curated metric paths (file-info glossary, translated like the
rest of the UI); anything else shows its full value. Population is
capped (depth, children per node, total items) so large session JSONs
stay interactive. Registered for ``json``.
"""

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QLabel, QTreeWidget, QTreeWidgetItem

from lynceus.plugins.locale import t
from lynceus.ui.translate import language_changed
from lynceus.ui.viewers.base import BaseViewer
from lynceus.ui.viewers.registry import register

MAX_DEPTH = 8
MAX_CHILDREN = 500
MAX_ITEMS = 20000
STR_LEN = 160

_PATH_ROLE = Qt.ItemDataRole.UserRole
_FULL_ROLE = Qt.ItemDataRole.UserRole + 1


def _guide_text(path: str | None) -> str | None:
    """Translated explanation for curated file-info metric paths.

    Literals (not a data table) so the i18n extractor enforces coverage
    in every language; evaluated at render time so language switches
    apply live.
    """
    table = {
        "file_info.processing_ram_estimate_bytes": t(
            "RAM recommended for in-memory processing (raw size by safety "
            "factor); tile the file if the machine is below it."
        ),
        "spatial_crs.epsg_code": t(
            "EPSG identifier resolved from the WKT. Null means the CRS "
            "could not be reduced to an EPSG code."
        ),
        "channel_attributes.gps_time.type": t(
            "Adjusted standard time is unambiguous across weeks; week "
            "time resets every 604800 s and can collide across flights."
        ),
        "channel_attributes.intensity.linearly_normalized": t(
            "Always null: linear normalization is a vendor claim and "
            "cannot be verified from the data alone."
        ),
        "channel_attributes.waveform.external_wdp": t(
            "True when waveform samples live in a sidecar .wdp next to "
            "the source instead of embedded packets."
        ),
        "channel_attributes.scan_angle.extreme_pct": t(
            "Share of points beyond 30 degrees off-nadir. High values "
            "widen edge artifacts; consider trimming by angle."
        ),
        "classification_flags.returns.multi_return_pct": t(
            "Share of pulses with more than one return (pulses are "
            "approximated by first returns). High values mean porous "
            "canopy or wires."
        ),
        "classification_flags.qa_flags.pct_withheld": t(
            "Share flagged by the vendor for exclusion. LynceusScan "
            "auto-excludes them from surfaces and metrics."
        ),
        "classification_flags.qa_flags.pct_overlap": t(
            "Share in flight-line overlap strips. Density is doubled "
            "there; filter those strips or process them separately for "
            "area metrics."
        ),
        "classification_flags.qa_flags.pct_keypoint": t(
            "Share of model key points. Decimation always keeps them."
        ),
        "classification_flags.qa_flags.pct_synthetic": t(
            "Share of post-processed (non-measured) points."
        ),
        "classification_flags.flight_lines.total_unique": t(
            "Distinct Point Source IDs: flight lines or tiles stitched "
            "into this file."
        ),
        "vlr_metadata.copc_ready": t(
            "True when COPC hierarchy metadata is present: the file "
            "supports spatial streaming."
        ),
        "vlr_metadata.extra_bytes.present": t(
            "True when the file carries custom per-point attributes "
            "(Extra Bytes)."
        ),
        "derived_metrics.area_aspect": t(
            "LINEAR_CORRIDOR (4:1 or more) vs BLOCK_POLYGON from the "
            "bounding-box sides; corridors need along-track care."
        ),
        "derived_metrics.flight_duration_s": t(
            "GPS time span of the survey in seconds."
        ),
        "derived_metrics.acquisition_rate_pts_s": t(
            "Mean acquisition rate; drops reveal pauses or multi-lift "
            "missions."
        ),
        "derived_metrics.mean_density_pts_m2": t(
            "All points over bounding-box area, voids included."
        ),
        "derived_metrics.mean_spacing_m": t(
            "1 over sqrt of mean density; nominal point spacing."
        ),
        "derived_metrics.usgs_npd_pts_m2": t(
            "USGS aggregate density: first returns only, overlap and "
            "withheld excluded."
        ),
        "derived_metrics.usgs_nps_m": t(
            "USGS nominal spacing from aggregate NPD; QL1 requires "
            "0.35 m or less."
        ),
        "derived_metrics.ground_class2_density_pts_m2": t(
            "Class-2 points per square meter; sparse ground weakens "
            "the DTM."
        ),
        "qa_qc_checks.crs_wkt_valid": t(
            "False means missing or unparseable WKT: products carry no "
            "projection."
        ),
        "qa_qc_checks.gps_time_adjusted": t(
            "False means week-time stamps: ambiguous across GPS weeks."
        ),
        "qa_qc_checks.reserved_classes_alert": t(
            "True means points labeled 8 or 12 in LAS 1.4+, where both "
            "are reserved: re-check the vendor classification."
        ),
        "qa_qc_checks.high_noise_withheld_alert": t(
            "True above 2% withheld: heavy vendor rejection, inspect "
            "the source."
        ),
        "qa_qc_checks.usgs_nps_compliant": t(
            "True when NPS is 0.35 m or less and NPD is 8 pts/m2 or "
            "more (USGS QL1)."
        ),
        "qa_qc_checks.strip_rmsdz_m": t(
            "Inter-strip vertical RMSD over shared planar patches; "
            "ASPRS 2024 requires 8 cm or less. Null when no planar "
            "overlap was found."
        ),
        "qa_qc_checks.strip_pairs_evaluated": t(
            "Flight-line pairs compared on shared planar cells."
        ),
        "qa_qc_checks.strip_consistency_ok": t(
            "True when strip RMSD is within tolerance. Null means "
            "unevaluable (single line or no planar overlap)."
        ),
    }
    return table.get(path or "")


def _short(text: str) -> str:
    text = str(text)
    return text if len(text) <= STR_LEN else text[:STR_LEN] + "…"


class JsonTreeViewer(BaseViewer):
    """Read-only hierarchical JSON inspector."""

    def __init__(self, kind: str, parent=None):
        super().__init__(kind, parent)
        self._tree = QTreeWidget(self)
        self._tree.setObjectName("jsonTree")
        self._tree.setHeaderHidden(True)
        self._tree.setAlternatingRowColors(True)
        self._tree.setRootIsDecorated(True)
        self._layout.addWidget(self._tree)
        self._tree.hide()
        self._guide = QLabel("", self)
        self._guide.setObjectName("jsonGuide")
        self._guide.setWordWrap(True)
        self._guide.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._layout.addWidget(self._guide)
        self._tree.currentItemChanged.connect(self._refresh_guide)
        language_changed.connect(self._refresh_guide)
        self._count = 0
        self._tree_generation = 0

    def _refresh_guide(self, *_args) -> None:
        """Show the curated explanation (or full value) for the selection."""
        item = self._tree.currentItem()
        if item is None:
            self._guide.setText("")
            return
        guide = _guide_text(item.data(0, _PATH_ROLE))
        if guide:
            self._guide.setText(guide)
            return
        full = item.data(0, _FULL_ROLE)
        self._guide.setText(str(full) if full else "")

    def set_payload(self, payload) -> None:
        from heapq import nsmallest
        from itertools import islice
        import json

        path = payload.get("file") if isinstance(payload, dict) else None
        self._source_path = str(path) if path else ""
        if not path:
            self.show_placeholder(t("No data available"))
            self._tree.hide()
            return

        def load(cancelled):
            if cancelled.is_set():
                return None
            with open(str(path), encoding="utf-8") as handle:
                doc = json.load(handle)
            if cancelled.is_set():
                return None
            if isinstance(doc, dict):
                root_limit = MAX_ITEMS - 1 if len(doc) > MAX_ITEMS else MAX_ITEMS
                entries = [
                    (str(key), doc[key], 0, str(key))
                    for key in nsmallest(root_limit, doc, key=str)
                ]
                if len(doc) > root_limit:
                    entries.append((f"… +{len(doc) - root_limit} more", None, -1, ""))
                return entries
            if isinstance(doc, list):
                root_limit = MAX_ITEMS - 1 if len(doc) > MAX_ITEMS else MAX_ITEMS
                entries = [
                    (f"[{pos}]", item, 0, "")
                    for pos, item in enumerate(islice(doc, root_limit))
                ]
                if len(doc) > root_limit:
                    entries.append((f"… +{len(doc) - root_limit} more", None, -1, ""))
                return entries
            return [("value", doc, 0, "")]

        def loaded(entries) -> None:
            if entries is None:
                return
            self._tree_generation += 1
            generation = self._tree_generation
            self._tree.clear()
            self._count = 0
            self._guide.setText("")
            self._tree.hide()
            self._placeholder.setText(t("Loading..."))
            self._placeholder.show()
            self._progress.show()
            index = 0

            def populate_batch() -> None:
                nonlocal index
                if generation != self._tree_generation:
                    return
                end = min(index + 8, len(entries))
                for key, value, depth, value_path in entries[index:end]:
                    if depth < 0:
                        self._tree.addTopLevelItem(QTreeWidgetItem([key]))
                        self._count += 1
                    else:
                        self._add(self._tree, key, value, depth, value_path)
                index = end
                if index < len(entries) and self._count < MAX_ITEMS:
                    QTimer.singleShot(0, populate_batch)
                    return
                self._tree.expandToDepth(1)
                self._tree.show()
                self._placeholder.hide()
                self._progress.hide()
                self.setWindowTitle(self._compose_title())

            populate_batch()

        def failed(_exc: str) -> None:
            self._tree.hide()
            self.show_placeholder(t("No data available"))

        self._tree.hide()
        self.start_async_load(load, loaded, failed)

    def _add(self, parent, key: str, value, depth: int, path: str) -> None:
        if self._count >= MAX_ITEMS:
            return
        self._count += 1
        if isinstance(value, dict) and depth < MAX_DEPTH:
            item = QTreeWidgetItem([key])
            self._attach(parent, item)
            item.setData(0, _PATH_ROLE, path)
            from itertools import islice

            items = list(islice(value.items(), MAX_CHILDREN))
            for sub_key, sub_value in items:
                self._add(item, str(sub_key), sub_value, depth + 1,
                          f"{path}.{sub_key}" if path else str(sub_key))
            if len(value) > len(items):
                self._attach(item, QTreeWidgetItem(
                    [f"… +{len(value) - len(items)} more"]))
                self._count += 1
        elif isinstance(value, list) and depth < MAX_DEPTH and value:
            item = QTreeWidgetItem([f"{key} [{len(value)}]"])
            self._attach(parent, item)
            item.setData(0, _PATH_ROLE, path)
            for pos, sub_value in enumerate(value[:MAX_CHILDREN]):
                self._add(item, f"[{pos}]", sub_value, depth + 1, path)
            if len(value) > MAX_CHILDREN:
                self._attach(item, QTreeWidgetItem(
                    [f"… +{len(value) - MAX_CHILDREN} more"]))
                self._count += 1
        else:
            if isinstance(value, bool):
                text = "true" if value else "false"
            elif value is None:
                text = "null"
            else:
                text = _short(value)
            label = f"{key}: {text}"
            item = QTreeWidgetItem([label])
            if isinstance(value, str) and len(value) > STR_LEN:
                item.setToolTip(0, value)
            item.setData(0, _PATH_ROLE, path)
            item.setData(0, _FULL_ROLE, label)
            self._attach(parent, item)

    @staticmethod
    def _attach(parent, item) -> None:
        if isinstance(parent, QTreeWidget):
            parent.addTopLevelItem(item)
        else:
            parent.addChild(item)


register("json", JsonTreeViewer)
