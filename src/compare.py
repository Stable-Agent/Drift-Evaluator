# Copyright 2026 Stable-Agent Contributors
# SPDX-License-Identifier: MIT
"""Side-by-side comparison renderer for two :class:`EvalReport` objects.

Used to compare detector versions (v0 vs v2) or before/after fitting. The
headline entry points are :func:`render_comparison_markdown`,
:func:`render_comparison_json`, and :func:`write_comparison`. No metrics
computation, no LLM calls, no dataset I/O — this module only transforms
two pre-computed reports into human- or machine-readable comparisons.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from drift_evaluator.metrics import (
    EvalReport,
    SignalCorrelation,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


_DRIFT_TYPE_ORDER: Tuple[str, ...] = ("goal", "constraint", "consistency")
_IMPROVEMENT_THRESHOLD: float = 0.02
_NA: str = "n/a"
_EMPTY_MESSAGE: str = "Cannot compare: one or both reports are empty."


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_nanlike(x: Any) -> bool:
    """Return True for NaN, infinite, or non-numeric floats."""

    try:
        f = float(x)
    except (TypeError, ValueError):
        return True
    return math.isnan(f) or math.isinf(f)


def _fmt_float(x: Any, *, decimals: int = 3) -> str:
    """Format a float to ``decimals`` places, with NaN/inf/None → ``"n/a"``."""

    if x is None or _is_nanlike(x):
        return _NA
    return f"{float(x):.{decimals}f}"


def _fmt_int(x: Any) -> str:
    """Format an integer; NaN/None → ``"n/a"``."""

    if x is None:
        return _NA
    try:
        return str(int(x))
    except (TypeError, ValueError):
        return _NA


def _fmt_delta(a: Any, b: Any, *, decimals: int = 3) -> str:
    """Signed delta ``b - a`` formatted to ``decimals``. NaN inputs → ``"n/a"``."""

    if a is None or b is None or _is_nanlike(a) or _is_nanlike(b):
        return _NA
    delta = float(b) - float(a)
    sign = "+" if delta >= 0 else "-"
    return f"{sign}{abs(delta):.{decimals}f}"


def _safe_delta(a: Any, b: Any) -> Optional[float]:
    """Return ``b - a`` as a float, or ``None`` if either side is NaN/missing."""

    if a is None or b is None or _is_nanlike(a) or _is_nanlike(b):
        return None
    return float(b) - float(a)


def _safe_float(x: Any) -> Optional[float]:
    """Return a finite float, or ``None`` for NaN/inf/non-numeric."""

    if x is None or _is_nanlike(x):
        return None
    return float(x)


def _signal_lookup(
    correlations: List[SignalCorrelation],
) -> Dict[str, SignalCorrelation]:
    """Index a list of SignalCorrelation objects by ``signal_name``."""

    return {sc.signal_name: sc for sc in correlations}


def _signal_name_union(
    a: List[SignalCorrelation],
    b: List[SignalCorrelation],
) -> List[str]:
    """Union of signal names, preserving order: A first, then B-only names."""

    seen: Dict[str, None] = {}
    for sc in a:
        if sc.signal_name not in seen:
            seen[sc.signal_name] = None
    for sc in b:
        if sc.signal_name not in seen:
            seen[sc.signal_name] = None
    return list(seen.keys())


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _is_empty(report: EvalReport) -> bool:
    return int(report.n_turns) == 0


# ---------------------------------------------------------------------------
# Markdown table builders
# ---------------------------------------------------------------------------


def _md_row(cells: List[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _md_header(headers: List[str]) -> List[str]:
    return [
        _md_row(headers),
        _md_row(["---"] * len(headers)),
    ]


def _headline_table(
    report_a: EvalReport,
    report_b: EvalReport,
    name_a: str,
    name_b: str,
) -> List[str]:
    a = report_a.binary
    b = report_b.binary
    col_a = f"{name_a} (thr={a.threshold:.2f})"
    col_b = f"{name_b} (thr={b.threshold:.2f})"

    lines: List[str] = []
    lines.append("## Headline Metrics")
    lines.append("")
    lines.extend(_md_header(["Metric", col_a, col_b, "Δ (B - A)"]))
    rows: List[Tuple[str, float, float]] = [
        ("Precision", a.precision, b.precision),
        ("Recall", a.recall, b.recall),
        ("F1", a.f1, b.f1),
        ("Accuracy", a.accuracy, b.accuracy),
    ]
    for label, av, bv in rows:
        lines.append(
            _md_row(
                [
                    label,
                    _fmt_float(av),
                    _fmt_float(bv),
                    _fmt_delta(av, bv),
                ]
            )
        )
    lines.append("")
    return lines


def _calibration_table(
    report_a: EvalReport,
    report_b: EvalReport,
    name_a: str,
    name_b: str,
) -> List[str]:
    a = report_a.calibration
    b = report_b.calibration
    lines: List[str] = []
    lines.append("## Calibration")
    lines.append("")
    lines.extend(_md_header(["Metric", name_a, name_b, "Δ (B - A)"]))
    rows: List[Tuple[str, float, float]] = [
        ("Brier (lower better)", a.brier_score, b.brier_score),
        ("ROC-AUC", a.roc_auc, b.roc_auc),
        ("PR-AUC", a.pr_auc, b.pr_auc),
    ]
    for label, av, bv in rows:
        lines.append(
            _md_row(
                [
                    label,
                    _fmt_float(av),
                    _fmt_float(bv),
                    _fmt_delta(av, bv),
                ]
            )
        )
    lines.append("")
    return lines


def _per_drift_type_table(
    report_a: EvalReport,
    report_b: EvalReport,
    name_a: str,
    name_b: str,
) -> List[str]:
    lines: List[str] = []
    lines.append("## Per-Drift-Type Recall")
    lines.append("")
    lines.extend(_md_header(["Drift Type", name_a, name_b, "Δ (B - A)"]))
    for drift_type in _DRIFT_TYPE_ORDER:
        av = report_a.per_drift_type_recall.get(drift_type, float("nan"))
        bv = report_b.per_drift_type_recall.get(drift_type, float("nan"))
        lines.append(
            _md_row(
                [
                    drift_type,
                    _fmt_float(av),
                    _fmt_float(bv),
                    _fmt_delta(av, bv),
                ]
            )
        )
    lines.append("")
    return lines


def _signal_table(
    report_a: EvalReport,
    report_b: EvalReport,
    name_a: str,
    name_b: str,
) -> List[str]:
    a_lookup = _signal_lookup(report_a.signal_correlations)
    b_lookup = _signal_lookup(report_b.signal_correlations)
    names = _signal_name_union(
        report_a.signal_correlations, report_b.signal_correlations
    )

    lines: List[str] = []
    lines.append("## Per-Signal Predictive Power")
    lines.append("")
    lines.extend(
        _md_header(
            [
                "Signal",
                f"Pearson r ({name_a})",
                f"Pearson r ({name_b})",
                f"ROC-AUC ({name_a})",
                f"ROC-AUC ({name_b})",
            ]
        )
    )
    for name in names:
        sa = a_lookup.get(name)
        sb = b_lookup.get(name)
        pa = _fmt_float(sa.pearson_r) if sa is not None else _NA
        pb = _fmt_float(sb.pearson_r) if sb is not None else _NA
        ra = _fmt_float(sa.roc_auc) if sa is not None else _NA
        rb = _fmt_float(sb.roc_auc) if sb is not None else _NA
        lines.append(_md_row([name, pa, pb, ra, rb]))
    lines.append("")
    return lines


def _ttd_table(
    report_a: EvalReport,
    report_b: EvalReport,
    name_a: str,
    name_b: str,
) -> List[str]:
    a = report_a.time_to_detection
    b = report_b.time_to_detection
    lines: List[str] = []
    lines.append("## Time-to-Detection")
    lines.append("")
    lines.extend(_md_header(["Metric", name_a, name_b, "Δ (B - A)"]))

    int_rows: List[Tuple[str, int, int]] = [
        ("Detected", a.detected_count, b.detected_count),
        ("Missed", a.missed_count, b.missed_count),
        ("Early", a.early_count, b.early_count),
    ]
    for label, av, bv in int_rows:
        lines.append(
            _md_row(
                [
                    label,
                    _fmt_int(av),
                    _fmt_int(bv),
                    _fmt_delta(av, bv, decimals=0),
                ]
            )
        )

    float_rows: List[Tuple[str, float, float]] = [
        ("Median delay", a.median_delay, b.median_delay),
        ("Mean delay", a.mean_delay, b.mean_delay),
        ("P90 delay", a.p90_delay, b.p90_delay),
    ]
    for label, av, bv in float_rows:
        lines.append(
            _md_row(
                [
                    label,
                    _fmt_float(av),
                    _fmt_float(bv),
                    _fmt_delta(av, bv),
                ]
            )
        )
    lines.append("")
    return lines


def _severity_table(
    report_a: EvalReport,
    report_b: EvalReport,
    name_a: str,
    name_b: str,
) -> List[str]:
    av = report_a.severity_confusion.accuracy
    bv = report_b.severity_confusion.accuracy
    lines: List[str] = []
    lines.append("## Severity Confusion Accuracy")
    lines.append("")
    lines.extend(_md_header(["Metric", name_a, name_b, "Δ (B - A)"]))
    lines.append(
        _md_row(
            [
                "Accuracy",
                _fmt_float(av),
                _fmt_float(bv),
                _fmt_delta(av, bv),
            ]
        )
    )
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Improvement summary
# ---------------------------------------------------------------------------


def _summary_metrics(
    report_a: EvalReport,
    report_b: EvalReport,
) -> List[Tuple[str, float, float, bool]]:
    """Return list of (label, a_value, b_value, lower_is_better) tuples.

    Includes only metrics that are meaningful for direction-of-change checks.
    """

    out: List[Tuple[str, float, float, bool]] = [
        ("Precision", report_a.binary.precision, report_b.binary.precision, False),
        ("Recall", report_a.binary.recall, report_b.binary.recall, False),
        ("F1", report_a.binary.f1, report_b.binary.f1, False),
        ("Accuracy", report_a.binary.accuracy, report_b.binary.accuracy, False),
        (
            "Brier",
            report_a.calibration.brier_score,
            report_b.calibration.brier_score,
            True,
        ),
        (
            "ROC-AUC",
            report_a.calibration.roc_auc,
            report_b.calibration.roc_auc,
            False,
        ),
        (
            "PR-AUC",
            report_a.calibration.pr_auc,
            report_b.calibration.pr_auc,
            False,
        ),
        (
            "Severity accuracy",
            report_a.severity_confusion.accuracy,
            report_b.severity_confusion.accuracy,
            False,
        ),
    ]
    for drift_type in _DRIFT_TYPE_ORDER:
        out.append(
            (
                f"Recall ({drift_type})",
                float(
                    report_a.per_drift_type_recall.get(drift_type, float("nan"))
                ),
                float(
                    report_b.per_drift_type_recall.get(drift_type, float("nan"))
                ),
                False,
            )
        )
    return out


def _format_summary_line(
    label: str,
    a_value: float,
    b_value: float,
    lower_is_better: bool,
    *,
    improved: bool,
) -> str:
    delta = b_value - a_value
    sign = "+" if delta >= 0 else "-"
    delta_str = f"{sign}{abs(delta):.3f}"
    base = (
        f"{label}: {a_value:.3f} → {b_value:.3f} ({delta_str})"
    )
    marker = "✅ improvement" if improved else "❌ regression"
    suffix = ""
    if lower_is_better:
        suffix = " (note: lower is better)"
    return f"- {base} {marker}{suffix}"


def _improvement_summary(
    report_a: EvalReport,
    report_b: EvalReport,
) -> List[str]:
    improvements: List[str] = []
    regressions: List[str] = []

    for label, av, bv, lower_is_better in _summary_metrics(report_a, report_b):
        if _is_nanlike(av) or _is_nanlike(bv):
            continue
        delta = float(bv) - float(av)
        if abs(delta) < _IMPROVEMENT_THRESHOLD:
            continue
        if lower_is_better:
            is_improvement = delta < 0
        else:
            is_improvement = delta > 0
        line = _format_summary_line(
            label,
            float(av),
            float(bv),
            lower_is_better,
            improved=is_improvement,
        )
        if is_improvement:
            improvements.append(line)
        else:
            regressions.append(line)

    lines: List[str] = []
    lines.append("## Improvement Summary")
    lines.append("")
    if not improvements and not regressions:
        lines.append(
            "No metric changed by at least "
            f"{_IMPROVEMENT_THRESHOLD:.2f} in absolute terms."
        )
        lines.append("")
        return lines

    if improvements:
        lines.append("**Improvements:**")
        lines.append("")
        lines.extend(improvements)
        lines.append("")
    if regressions:
        lines.append("**Regressions:**")
        lines.append("")
        lines.extend(regressions)
        lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Markdown public API
# ---------------------------------------------------------------------------


def render_comparison_markdown(
    report_a: EvalReport,
    report_b: EvalReport,
    *,
    name_a: str,
    name_b: str,
    title: str = "Drift Detector Comparison",
    notes: Optional[str] = None,
    generated_at: Optional[str] = None,
) -> str:
    """Side-by-side markdown comparison.

    Renders a multi-section comparison: headline metrics, calibration,
    per-drift-type recall, per-signal predictive power, time-to-detection,
    severity confusion accuracy, and an improvement summary.
    """

    if _is_empty(report_a) or _is_empty(report_b):
        return _EMPTY_MESSAGE

    timestamp = generated_at if generated_at is not None else _now_iso()

    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"- **Generated at:** {timestamp}")
    lines.append(f"- **A:** {name_a}")
    lines.append(f"- **B:** {name_b}")
    lines.append(
        f"- **A coverage:** {report_a.n_conversations} conversations / "
        f"{report_a.n_turns} turns (drift rate "
        f"{_fmt_float(report_a.drift_rate)})"
    )
    lines.append(
        f"- **B coverage:** {report_b.n_conversations} conversations / "
        f"{report_b.n_turns} turns (drift rate "
        f"{_fmt_float(report_b.drift_rate)})"
    )
    if notes:
        lines.append(f"- **Notes:** {notes}")
    lines.append("")

    lines.extend(_headline_table(report_a, report_b, name_a, name_b))
    lines.extend(_calibration_table(report_a, report_b, name_a, name_b))
    lines.extend(_per_drift_type_table(report_a, report_b, name_a, name_b))
    lines.extend(_signal_table(report_a, report_b, name_a, name_b))
    lines.extend(_ttd_table(report_a, report_b, name_a, name_b))
    lines.extend(_severity_table(report_a, report_b, name_a, name_b))
    lines.extend(_improvement_summary(report_a, report_b))

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# JSON public API
# ---------------------------------------------------------------------------


def _json_pair(
    a_value: Any, b_value: Any, *, is_int: bool = False
) -> Dict[str, Any]:
    """Render an A/B/delta triple as a JSON-friendly dict.

    NaN/inf values become ``None``. Deltas are computed as floats; if either
    side is missing, the delta is ``None``.
    """

    if is_int:
        a_clean: Any = int(a_value) if a_value is not None else None
        b_clean: Any = int(b_value) if b_value is not None else None
    else:
        a_clean = _safe_float(a_value)
        b_clean = _safe_float(b_value)
    return {
        "a": a_clean,
        "b": b_clean,
        "delta": _safe_delta(a_value, b_value),
    }


def _json_headline(
    report_a: EvalReport, report_b: EvalReport
) -> Dict[str, Any]:
    a = report_a.binary
    b = report_b.binary
    return {
        "threshold_a": _safe_float(a.threshold),
        "threshold_b": _safe_float(b.threshold),
        "precision": _json_pair(a.precision, b.precision),
        "recall": _json_pair(a.recall, b.recall),
        "f1": _json_pair(a.f1, b.f1),
        "accuracy": _json_pair(a.accuracy, b.accuracy),
    }


def _json_calibration(
    report_a: EvalReport, report_b: EvalReport
) -> Dict[str, Any]:
    a = report_a.calibration
    b = report_b.calibration
    return {
        "brier_score": _json_pair(a.brier_score, b.brier_score),
        "roc_auc": _json_pair(a.roc_auc, b.roc_auc),
        "pr_auc": _json_pair(a.pr_auc, b.pr_auc),
    }


def _json_per_drift_type(
    report_a: EvalReport, report_b: EvalReport
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for drift_type in _DRIFT_TYPE_ORDER:
        out[drift_type] = _json_pair(
            report_a.per_drift_type_recall.get(drift_type, float("nan")),
            report_b.per_drift_type_recall.get(drift_type, float("nan")),
        )
    return out


def _json_signals(
    report_a: EvalReport, report_b: EvalReport
) -> List[Dict[str, Any]]:
    a_lookup = _signal_lookup(report_a.signal_correlations)
    b_lookup = _signal_lookup(report_b.signal_correlations)
    names = _signal_name_union(
        report_a.signal_correlations, report_b.signal_correlations
    )

    out: List[Dict[str, Any]] = []
    for name in names:
        sa = a_lookup.get(name)
        sb = b_lookup.get(name)
        out.append(
            {
                "signal_name": name,
                "pearson_r": {
                    "a": _safe_float(sa.pearson_r) if sa is not None else None,
                    "b": _safe_float(sb.pearson_r) if sb is not None else None,
                },
                "roc_auc": {
                    "a": _safe_float(sa.roc_auc) if sa is not None else None,
                    "b": _safe_float(sb.roc_auc) if sb is not None else None,
                },
            }
        )
    return out


def _json_ttd(report_a: EvalReport, report_b: EvalReport) -> Dict[str, Any]:
    a = report_a.time_to_detection
    b = report_b.time_to_detection
    return {
        "detected": _json_pair(a.detected_count, b.detected_count, is_int=True),
        "missed": _json_pair(a.missed_count, b.missed_count, is_int=True),
        "early": _json_pair(a.early_count, b.early_count, is_int=True),
        "median_delay": _json_pair(a.median_delay, b.median_delay),
        "mean_delay": _json_pair(a.mean_delay, b.mean_delay),
        "p90_delay": _json_pair(a.p90_delay, b.p90_delay),
    }


def _json_severity(
    report_a: EvalReport, report_b: EvalReport
) -> Dict[str, Any]:
    a = report_a.severity_confusion.accuracy
    b = report_b.severity_confusion.accuracy
    return {"accuracy": _json_pair(a, b)}


def _json_improvement_summary(
    report_a: EvalReport, report_b: EvalReport
) -> Dict[str, List[Dict[str, Any]]]:
    improvements: List[Dict[str, Any]] = []
    regressions: List[Dict[str, Any]] = []

    for label, av, bv, lower_is_better in _summary_metrics(report_a, report_b):
        if _is_nanlike(av) or _is_nanlike(bv):
            continue
        delta = float(bv) - float(av)
        if abs(delta) < _IMPROVEMENT_THRESHOLD:
            continue
        if lower_is_better:
            is_improvement = delta < 0
        else:
            is_improvement = delta > 0
        entry: Dict[str, Any] = {
            "metric": label,
            "a": float(av),
            "b": float(bv),
            "delta": delta,
            "lower_is_better": bool(lower_is_better),
        }
        if is_improvement:
            improvements.append(entry)
        else:
            regressions.append(entry)
    return {"improvements": improvements, "regressions": regressions}


def render_comparison_json(
    report_a: EvalReport,
    report_b: EvalReport,
    *,
    name_a: str,
    name_b: str,
    generated_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Same data as the markdown, but as a structured dict for CI tracking.

    NaN/inf values are converted to ``None`` so the result is JSON-safe.
    """

    timestamp = generated_at if generated_at is not None else _now_iso()

    if _is_empty(report_a) or _is_empty(report_b):
        return {
            "generated_at": timestamp,
            "name_a": name_a,
            "name_b": name_b,
            "empty": True,
            "message": _EMPTY_MESSAGE,
        }

    return {
        "generated_at": timestamp,
        "name_a": name_a,
        "name_b": name_b,
        "empty": False,
        "coverage": {
            "a": {
                "n_conversations": int(report_a.n_conversations),
                "n_turns": int(report_a.n_turns),
                "drift_rate": _safe_float(report_a.drift_rate),
            },
            "b": {
                "n_conversations": int(report_b.n_conversations),
                "n_turns": int(report_b.n_turns),
                "drift_rate": _safe_float(report_b.drift_rate),
            },
        },
        "headline": _json_headline(report_a, report_b),
        "calibration": _json_calibration(report_a, report_b),
        "per_drift_type_recall": _json_per_drift_type(report_a, report_b),
        "per_signal": _json_signals(report_a, report_b),
        "time_to_detection": _json_ttd(report_a, report_b),
        "severity_confusion": _json_severity(report_a, report_b),
        "improvement_summary": _json_improvement_summary(report_a, report_b),
    }


# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------


def write_comparison(
    report_a: EvalReport,
    report_b: EvalReport,
    output_dir: Path | str,
    *,
    name_a: str,
    name_b: str,
    name: str = "comparison",
    title: str = "Drift Detector Comparison",
    notes: Optional[str] = None,
) -> Tuple[Path, Path]:
    """Render markdown + JSON comparison files into ``output_dir``.

    Creates ``output_dir`` if missing. Writes ``{name}.md`` and ``{name}.json``
    using a single shared timestamp so the two artifacts agree. Returns
    ``(md_path, json_path)``.
    """

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = _now_iso()

    md_text = render_comparison_markdown(
        report_a,
        report_b,
        name_a=name_a,
        name_b=name_b,
        title=title,
        notes=notes,
        generated_at=timestamp,
    )
    json_data = render_comparison_json(
        report_a,
        report_b,
        name_a=name_a,
        name_b=name_b,
        generated_at=timestamp,
    )

    md_path = out_dir / f"{name}.md"
    json_path = out_dir / f"{name}.json"

    md_path.write_text(md_text, encoding="utf-8")
    json_path.write_text(
        json.dumps(json_data, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    return md_path, json_path
