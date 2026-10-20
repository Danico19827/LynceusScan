"""Pre-flight advisor dialog smoke (offscreen Qt).

Purely structural: the advisor must never gate a run, so the dialog's
contract is exercised headlessly (cache override, dismiss flag, jump to
Performance) without a real pipeline."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QDialog, QPushButton

from lynceus.processing.preflight import Preflight, build_preflight
from lynceus.ui.preflight_dialog import (
    OPEN_PERFORMANCE,
    SHOW_LOADER,
    PreflightDialog,
)

_BIG = 120_000_000


def _advice(cache_enabled: bool = False) -> Preflight:
    return Preflight(
        total_points=_BIG,
        area_m2=2_560_000.0,
        tile_count_est=256,
        per_tile_points=469_000,
        heaviest_factor=3.0,
        tile_nodes=4,
        workers=8,
        est_peak_bytes=12 * 1024 ** 3,
        budget_bytes=16 * 1024 ** 3,
        cache_enabled=cache_enabled,
        ask_cache=not cache_enabled,
        ask_ram=True,
    )


class PreflightDialogTests(unittest.TestCase):
    def setUp(self) -> None:
        self._app = QApplication.instance() or QApplication([])

    def test_cache_checkbox_override_on_accept(self) -> None:
        dialog = PreflightDialog(None, _advice(cache_enabled=False))
        self.assertTrue(dialog._cache_box.isChecked())  # default on
        dialog._never.setChecked(True)
        dialog._on_accept()
        self.assertTrue(dialog.enable_cache())
        self.assertTrue(dialog.dont_show_again())
        self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)

    def test_open_performance_exit_code(self) -> None:
        dialog = PreflightDialog(None, _advice())
        dialog.done(OPEN_PERFORMANCE)
        self.assertEqual(dialog.result(), OPEN_PERFORMANCE)

    def test_no_advice_dialog_is_skipped_by_canvas_logic(self) -> None:
        # build_preflight must fall through to {} (continue) without advice.
        self.assertIsNone(build_preflight([], [1.0], workers=8))  # no files

    def test_cache_already_enabled_hides_cache_box(self) -> None:
        dialog = PreflightDialog(None, _advice(cache_enabled=True))
        self.assertIsNone(dialog._cache_box)

    def test_merge_suggestion_shows_loader_button(self) -> None:
        from dataclasses import replace

        advice = replace(_advice(), merge_floor_suggest=500000)
        dialog = PreflightDialog(None, advice)
        self.assertIsNotNone(dialog._loader_button)
        dialog._loader_button.click()
        self.assertEqual(dialog.result(), SHOW_LOADER)

    def test_no_merge_suggestion_hides_loader_button(self) -> None:
        dialog = PreflightDialog(None, _advice())
        self.assertIsNone(dialog._loader_button)

    def test_segments_suggestion_shares_loader_button(self) -> None:
        from dataclasses import replace

        advice = replace(
            _advice(), merge_floor_suggest=0,
            segments_cap_suggest=25_000_000, segments_count_est=47)
        dialog = PreflightDialog(None, advice)
        self.assertIsNotNone(dialog._segments_box)
        self.assertTrue(dialog._segments_box.isChecked())
        self.assertIsNotNone(dialog._loader_button)
        buttons = [
            b for b in dialog.findChildren(QPushButton)
            if b.text() == dialog._loader_button.text()
        ]
        self.assertEqual(len(buttons), 1)

    def test_segments_checkbox_applies_cap(self) -> None:
        from dataclasses import replace

        advice = replace(
            _advice(), merge_floor_suggest=0,
            segments_cap_suggest=25_000_000, segments_count_est=47)
        dialog = PreflightDialog(None, advice)
        dialog._on_accept()
        self.assertEqual(dialog.apply_segments(), 25_000_000)
        dialog._segments_box.setChecked(False)
        dialog._on_accept()
        self.assertEqual(dialog.apply_segments(), 0)

    def test_merge_checkbox_defaults_checked_and_applies(self) -> None:
        from dataclasses import replace

        advice = replace(_advice(), merge_floor_suggest=500000)
        dialog = PreflightDialog(None, advice)
        self.assertIsNotNone(dialog._merge_box)
        self.assertTrue(dialog._merge_box.isChecked())
        dialog._on_accept()
        self.assertTrue(dialog.apply_merge_floor())

    def test_merge_checkbox_unchecked_skips_apply(self) -> None:
        from dataclasses import replace

        advice = replace(_advice(), merge_floor_suggest=500000)
        dialog = PreflightDialog(None, advice)
        dialog._merge_box.setChecked(False)
        dialog._on_accept()
        self.assertFalse(dialog.apply_merge_floor())

    def test_no_merge_suggestion_disables_apply(self) -> None:
        dialog = PreflightDialog(None, _advice())
        self.assertIsNone(dialog._merge_box)
        self.assertFalse(dialog.apply_merge_floor())

    def test_accept_without_ram_block_never_crashes(self) -> None:
        # Regression: optional widgets once lived only inside their advice
        # branches, so accepting a merge/cache-only dialog raised
        # AttributeError and blocked the run.
        from dataclasses import replace

        advice = replace(
            _advice(), ask_ram=False, merge_floor_suggest=500000)
        dialog = PreflightDialog(None, advice)
        self.assertIsNone(dialog._workers_box)
        dialog._on_accept()
        self.assertTrue(dialog.apply_merge_floor())
        self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)

    def test_mosaic_block_and_resolution_apply(self) -> None:
        from dataclasses import replace

        advice = replace(
            _advice(), ask_mosaic=True, mosaic_bytes_est=69_000_000_000,
            budget_bytes=1_073_741_824, resolution_suggest=4.0)
        dialog = PreflightDialog(None, advice)
        self.assertIsNotNone(dialog._resolution_box)
        self.assertTrue(dialog._resolution_box.isChecked())
        self.assertIsNone(dialog._ram_box)
        dialog._on_accept()
        self.assertEqual(dialog.apply_resolution(), 4.0)
        self.assertEqual(dialog.apply_workers(), 0)
        self.assertEqual(dialog.apply_memory(), 0.0)

    def test_mosaic_raise_block_and_memory_apply(self) -> None:
        from dataclasses import replace

        advice = replace(
            _advice(), ask_mosaic=True,
            mosaic_bytes_est=4_800_000_000,
            budget_bytes=1_073_741_824,
            ram_budget_suggest=8 * 1024 ** 3)
        dialog = PreflightDialog(None, advice)
        self.assertIsNotNone(dialog._ram_box)
        self.assertTrue(dialog._ram_box.isChecked())
        self.assertIsNone(dialog._resolution_box)
        dialog._on_accept()
        self.assertEqual(dialog.apply_memory(), 8 * 1024 ** 3)
        dialog._ram_box.setChecked(False)
        dialog._on_accept()
        self.assertEqual(dialog.apply_memory(), 0.0)

    def test_workers_checkbox_applies_override(self) -> None:
        from dataclasses import replace

        advice = replace(_advice(), ask_ram=True, workers_suggest=3)
        dialog = PreflightDialog(None, advice)
        self.assertIsNotNone(dialog._workers_box)
        dialog._on_accept()
        self.assertEqual(dialog.apply_workers(), 3)
        dialog2 = PreflightDialog(None, _advice())
        self.assertIsNone(dialog2._workers_box)
        self.assertEqual(dialog2.apply_workers(), 0)
        self.assertEqual(dialog2.apply_resolution(), 0.0)


if __name__ == "__main__":
    unittest.main()