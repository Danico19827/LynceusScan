# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Taritolay, Nicolás Daniel <lynceusscan@gmail.com>
"""Per-session quality report (domain, no Qt).

Every mini-session (main batch, segments, final consolidation) gets a
``quality_report.json`` next to its products: lineage, completeness,
consistency and performance evidence plus a conformance verdict against
an explicit, versioned product spec. The report documents the process;
positional accuracy still needs an external reference (declared as
``na`` instead of pretended), and the T&C requires verification by a
qualified professional before operational use.

The default spec lives in code (auditable, identical for everyone);
operator overrides arrive later. Content is English data (like warnings
and provenance): no i18n catalog impact.
"""

from __future__ import annotations

from datetime import datetime, timezone

SPEC_VERSION = 1

DEFAULT_SPEC = {
    "spec_version": SPEC_VERSION,
    "max_failed_tasks": 0,
    "require_all_products": True,
    "single_crs": True,
    "allow_empty_products": False,
    "warnings_fail": False,
}

VERDICT_PASS = "PASS"
VERDICT_FAIL = "FAIL"

_NA = "na"

CRITERIA_GUIDE = {
    "no_task_failures": "Failed tasks vs spec max_failed_tasks.",
    "products_complete": "Barrier products expected vs written to disk.",
    "crs_single": "One coordinate system across LiDAR sources.",
    "no_empty_products": "No zero-byte products (bytes, not content).",
    "inputs_traceable": "Every external input fingerprinted.",
    "positional_accuracy": "Always na: needs ground truth the software never has.",
    "warnings_acceptable": "Warning count; informational unless the spec fails on it.",
    "segments_conform": "Parent aggregate: conforming segments over total.",
    "result_pass": "Criterion evaluated and within threshold.",
    "result_fail": "Criterion breached; detail names the cause. One fail flips the verdict.",
    "result_na": "Not applicable to this run; never flips the verdict.",
    "verdict": "PASS = every applicable criterion green. FAIL = at least one red.",
}
"""Fixed reading guide embedded in every report (stable across versions)."""


def _criterion(result: str, detail: str = "") -> dict:
    return {"result": result, "detail": detail}


def evaluate(spec: dict, checks: dict) -> tuple[dict, str]:
    """Evaluate conformance checks against the spec.

    ``checks`` carries precomputed facts (the controller gathers them, no
    I/O here): ``failed_tasks`` (int), ``expected_products``/``written_products``
    (lists of iids), ``crs_list`` (list or None when there is no LiDAR
    domain), ``empty_products`` (list of names), ``traced_inputs`` (bool
    or None when the graph takes no external inputs), ``warnings``
    (int, informational unless the spec fails on them).

    Returns ``(evaluation, verdict)``; verdict is FAIL when any evaluated
    criterion fails, PASS otherwise.
    """
    spec = spec or {}
    evaluation = {}

    max_failed = spec.get("max_failed_tasks", 0)
    failed = checks.get("failed_tasks", 0)
    if failed <= max_failed:
        evaluation["no_task_failures"] = _criterion(
            "pass", f"{failed} failed task(s)"
        )
    else:
        evaluation["no_task_failures"] = _criterion(
            "fail", f"{failed} failed task(s), max {max_failed}"
        )

    if spec.get("require_all_products", True):
        expected = list(checks.get("expected_products", []))
        written = set(checks.get("written_products", []))
        missing = [iid for iid in expected if iid not in written]
        if not expected:
            evaluation["products_complete"] = _criterion(
                _NA, "no barrier products in this batch"
            )
        elif not missing:
            evaluation["products_complete"] = _criterion(
                "pass", f"{len(expected)} product(s) written"
            )
        else:
            evaluation["products_complete"] = _criterion(
                "fail", f"missing products: {', '.join(missing)}"
            )
    else:
        evaluation["products_complete"] = _criterion(_NA, "not required")

    if spec.get("single_crs", True):
        crs_list = checks.get("crs_list")
        if crs_list is None:
            evaluation["crs_single"] = _criterion(
                _NA, "no LiDAR domain in this batch"
            )
        elif len(set(crs_list)) <= 1:
            evaluation["crs_single"] = _criterion(
                "pass", f"single CRS ({crs_list[0]})" if crs_list else "pass"
            )
        else:
            evaluation["crs_single"] = _criterion(
                "fail", f"mixed CRS: {', '.join(sorted(set(crs_list)))}"
            )
    else:
        evaluation["crs_single"] = _criterion(_NA, "not required")

    if spec.get("allow_empty_products", False):
        evaluation["no_empty_products"] = _criterion(_NA, "allowed")
    else:
        empty = list(checks.get("empty_products", []))
        if not empty:
            evaluation["no_empty_products"] = _criterion(
                "pass", "no zero-byte products"
            )
        else:
            evaluation["no_empty_products"] = _criterion(
                "fail", f"empty products: {', '.join(empty)}"
            )

    traced = checks.get("traced_inputs")
    if traced is None:
        evaluation["inputs_traceable"] = _criterion(
            _NA, "no external inputs in this batch"
        )
    elif traced:
        evaluation["inputs_traceable"] = _criterion(
            "pass", "every external input fingerprinted"
        )
    else:
        evaluation["inputs_traceable"] = _criterion(
            "fail", "an external input has no fingerprint"
        )

    # Positional accuracy needs ground truth the software never has:
    # declared, never pretended.
    evaluation["positional_accuracy"] = _criterion(
        _NA, "requires an external reference (control points)"
    )

    # Warnings are observations by default; the spec may promote them.
    n_warnings = checks.get("warnings", 0)
    if n_warnings and spec.get("warnings_fail", False):
        evaluation["warnings_acceptable"] = _criterion(
            "fail", f"{n_warnings} warning(s)"
        )
    else:
        evaluation["warnings_acceptable"] = _criterion(
            "pass",
            f"{n_warnings} warning(s)"
            + (", informational" if n_warnings else ""),
        )

    verdict = VERDICT_FAIL if any(
        criterion["result"] == "fail" for criterion in evaluation.values()
    ) else VERDICT_PASS
    return evaluation, verdict


def build_quality_report(
    *,
    status: str,
    session_name: str,
    lineage: dict,
    counts: dict,
    performance: dict,
    evaluation: dict,
    verdict: str,
    warnings: int = 0,
    segments: list | None = None,
) -> dict:
    """Assemble the session quality report document (JSON-serializable)."""
    return {
        "schema": "lynceus-quality-report",
        "schema_version": 1,
        "spec_version": SPEC_VERSION,
        "status": status,
        "session": session_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "lineage": lineage,
        "counts": counts,
        "performance": performance,
        "evaluation": evaluation,
        "criteria_guide": CRITERIA_GUIDE,
        "warnings": warnings,
        "segments": segments or [],
        "conformance": (
            "Generated by LynceusScan; positional accuracy not assessed "
            "(external reference required). Verify with a qualified "
            "professional before operational use (see Terms & Conditions)."
        ),
    }
