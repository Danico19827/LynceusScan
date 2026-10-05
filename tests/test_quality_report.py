# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Tests for the per-session quality report (Q1).

Pure builder/evaluation tests: versioned spec, PASS/FAIL verdicts with
reasons, and an explicit non-assessment for positional accuracy.
Stdlib + engine/domain modules only (no Qt, no real runs).
"""

from __future__ import annotations

import unittest

from lynceus.processing.quality import (
    DEFAULT_SPEC,
    SPEC_VERSION,
    VERDICT_FAIL,
    VERDICT_PASS,
    build_quality_report,
    evaluate,
)


def _checks(**overrides):
    base = {
        "failed_tasks": 0,
        "expected_products": ["a", "b"],
        "written_products": ["a", "b"],
        "crs_list": ["EPSG:5347"],
        "empty_products": [],
        "traced_inputs": True,
        "warnings": 0,
    }
    base.update(overrides)
    return base


class EvaluateTests(unittest.TestCase):
    def test_clean_run_passes(self) -> None:
        evaluation, verdict = evaluate(DEFAULT_SPEC, _checks())
        self.assertEqual(verdict, VERDICT_PASS)
        self.assertEqual(evaluation["no_task_failures"]["result"], "pass")
        self.assertEqual(evaluation["products_complete"]["result"], "pass")
        self.assertEqual(evaluation["crs_single"]["result"], "pass")
        self.assertEqual(
            evaluation["positional_accuracy"]["result"], "na"
        )

    def test_failed_tasks_fail(self) -> None:
        evaluation, verdict = evaluate(DEFAULT_SPEC, _checks(failed_tasks=2))
        self.assertEqual(verdict, VERDICT_FAIL)
        self.assertIn("2", evaluation["no_task_failures"]["detail"])

    def test_missing_product_fails_with_name(self) -> None:
        evaluation, verdict = evaluate(
            DEFAULT_SPEC, _checks(written_products=["a"])
        )
        self.assertEqual(verdict, VERDICT_FAIL)
        self.assertIn("b", evaluation["products_complete"]["detail"])

    def test_mixed_crs_fails(self) -> None:
        evaluation, verdict = evaluate(
            DEFAULT_SPEC, _checks(crs_list=["EPSG:5347", "EPSG:5348"])
        )
        self.assertEqual(verdict, VERDICT_FAIL)

    def test_no_crs_domain_is_na_not_fail(self) -> None:
        evaluation, verdict = evaluate(DEFAULT_SPEC, _checks(crs_list=None))
        self.assertEqual(evaluation["crs_single"]["result"], "na")
        self.assertEqual(verdict, VERDICT_PASS)

    def test_untraced_input_fails(self) -> None:
        evaluation, verdict = evaluate(
            DEFAULT_SPEC, _checks(traced_inputs=False)
        )
        self.assertEqual(verdict, VERDICT_FAIL)

    def test_no_external_inputs_is_na_not_fail(self) -> None:
        evaluation, verdict = evaluate(
            DEFAULT_SPEC, _checks(traced_inputs=None)
        )
        self.assertEqual(evaluation["inputs_traceable"]["result"], "na")
        self.assertEqual(verdict, VERDICT_PASS)

    def test_warnings_informational_by_default(self) -> None:
        evaluation, verdict = evaluate(DEFAULT_SPEC, _checks(warnings=3))
        self.assertEqual(evaluation["warnings_acceptable"]["result"], "pass")
        self.assertEqual(verdict, VERDICT_PASS)

    def test_warnings_fail_when_spec_says_so(self) -> None:
        spec = dict(DEFAULT_SPEC, warnings_fail=True)
        evaluation, verdict = evaluate(spec, _checks(warnings=1))
        self.assertEqual(evaluation["warnings_acceptable"]["result"], "fail")
        self.assertEqual(verdict, VERDICT_FAIL)


class BuildReportTests(unittest.TestCase):
    def test_report_shape(self) -> None:
        evaluation, verdict = evaluate(DEFAULT_SPEC, _checks())
        report = build_quality_report(
            status="completed",
            session_name="20260101_000000",
            lineage={"app_version": "0.1.0", "operator": None,
                      "sources": [], "nodes": []},
            counts={"tiles": 4, "points": 100},
            performance={"peak_memory_mb": 512.0},
            evaluation=evaluation,
            verdict=verdict,
        )
        self.assertEqual(report["schema"], "lynceus-quality-report")
        self.assertEqual(report["spec_version"], SPEC_VERSION)
        self.assertEqual(report["verdict"], VERDICT_PASS)
        self.assertEqual(report["status"], "completed")
        self.assertIn("professional", report["conformance"])
        self.assertIn("no_task_failures", report["criteria_guide"])
        self.assertIn("verdict", report["criteria_guide"])
        import json

        json.dumps(report)  # serializable


if __name__ == "__main__":
    unittest.main()
