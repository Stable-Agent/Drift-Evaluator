# Copyright 2026 Stable-Agent Contributors
# SPDX-License-Identifier: MIT
"""Drift-detection evaluation report rendering.

Pure formatting helpers that turn an :class:`EvalReport` (computed by
``metrics.py``) into either a markdown human-readable report or a JSON
machine-readable report. No metrics computation, no I/O of conversation
data, no external services.

The headline entry points are :func:`render_markdown`,
:func:`render_json`, and :func:`write_report`.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from drift_evaluator.metrics import (
    BinaryMetrics,
    CalibrationMetrics,
    EvalReport,
    SeverityConfusion,
    SignalCorrelation,
    ThresholdSweepRow,
    TimeToDetection,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_DRIFT_TYPE_ORDER: Tuple[str, ...] = ("goal", "constraint", "consistency")


def _is_nanlike(x: float) -> bool:
    """Return True for NaN or infinite floats."""

    try:
        f = float(x)
    except (TypeError, ValueError):
        return True
    return math.isnan(f) or math.isinf(f)


def _fmt(x: float) -> str:
    """Format a float for markdown: ``"n/a"`` for NaN/inf, else 3 decimals."""

    if _is_nanlike(x):
        return "n/a"
    return f"{float(x):.3f}"


def _pct(x: float) -> str:
    """Format a fraction in [0, 1] as a percentage to one decimal place."""

    if _is_nanlike(x):
        return "n/a"
    return f"{float(x) * 100.0:.1f}%"


def _json_num(x: float) -> Optional[float]:
    """Round-trip a float through JSON: NaN/inf become ``None``."""

    if _is_nanlike(x):
        return None
    return float(x)


def _now_iso() -> str:
    return datetime.now().isoformat()


def _delay_histogram(delays: List[int], *, bar_max: int = 40) -> List[str]:
    """Render a text histogram of integer delays.

    Returns a list of lines, one per bucket, each like ``"0: ████ 12"``.
    Buckets cover every integer in ``[min, max]`` so that empty buckets
    show up too. If ``delays`` is empty, returns an empty list.
    """

    if len(delays) == 0:
        return []

    int_delays = [int(d) for d in delays]
    lo = min(int_delays)
    hi = max(int_delays)
    counts: Dict[int, int] = {bucket: 0 for bucket in range(lo, hi + 1)}
    for d in int_delays:
        counts[d] += 1
    peak = max(counts.values()) if counts else 0
    width = len(str(hi))
    lines: List[str] = []
    for bucket in range(lo, hi + 1):
        c = counts[bucket]
        bar_len = 0 if peak == 0 else max(0, (c * bar_max) // peak)
        if c > 0 and bar_len == 0:
            bar_len = 1
        bar = "█" * bar_len
        lines.append(f"{str(bucket).rjust(width)}: {bar} {c}")
    return lines


def _argmax_f1_index(sweep: List[ThresholdSweepRow]) -> Optional[int]:
    """Return the index of the sweep row with highest F1, or None."""

    best_idx: Optional[int] = None
    best_f1 = -math.inf
    for i, row in enumerate(sweep):
        if _is_nanlike(row.f1):
            continue
        if row.f1 > best_f1:
            best_f1 = row.f1
            best_idx = i
    return best_idx


def _metadata_block(
    *,
    dataset_name: Optional[str],
    detector_name: Optional[str],
    generated_at: Optional[str],
) -> List[str]:
    ts = generated_at if generated_at is not None else _now_iso()
    lines = [
        f"- **Dataset:** {dataset_name if dataset_name else 'n/a'}",
        f"- **Detector:** {detector_name if detector_name else 'n/a'}",
        f"- **Generated at:** {ts}",
    ]
    return lines


# ---------------------------------------------------------------------------
# Section renderers (markdown)
# ---------------------------------------------------------------------------


def _md_overview(report: EvalReport) -> List[str]:
    return [
        "## Overview",
        "",
        f"- **Conversations:** {report.n_conversations}",
        f"- **Turns:** {report.n_turns}",
        f"- **Drift rate (positive turns):** {_pct(report.drift_rate)}",
    ]


def _md_headline(binary: BinaryMetrics) -> List[str]:
    out = [
        "## Headline Metrics",
        "",
        f"At threshold **{_fmt(binary.threshold)}**:",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Precision | {_fmt(binary.precision)} |",
        f"| Recall | {_fmt(binary.recall)} |",
        f"| F1 | {_fmt(binary.f1)} |",
        f"| Accuracy | {_fmt(binary.accuracy)} |",
        "",
        "Confusion-matrix counts:",
        "",
        "| | Predicted positive | Predicted negative |",
        "| --- | --- | --- |",
        f"| **Actual positive** | TP = {binary.true_positives} | FN = {binary.false_negatives} |",
        f"| **Actual negative** | FP = {binary.false_positives} | TN = {binary.true_negatives} |",
    ]
    return out


def _md_calibration(cal: CalibrationMetrics) -> List[str]:
    return [
        "## Calibration",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Brier score | {_fmt(cal.brier_score)} |",
        f"| ROC-AUC | {_fmt(cal.roc_auc)} |",
        f"| PR-AUC | {_fmt(cal.pr_auc)} |",
    ]


def _md_per_drift_type(per_type: Dict[str, float]) -> List[str]:
    out = [
        "## Per-Drift-Type Recall",
        "",
        "| Drift type | Recall |",
        "| --- | --- |",
    ]
    for drift_type in _DRIFT_TYPE_ORDER:
        recall = per_type.get(drift_type, float("nan"))
        out.append(f"| {drift_type} | {_fmt(recall)} |")
    return out


def _md_severity_confusion(sev: SeverityConfusion) -> List[str]:
    labels = list(sev.labels)
    header = "| true \\\\ pred | " + " | ".join(labels) + " |"
    sep = "| --- |" + " --- |" * len(labels)
    out = [
        "## Severity Confusion Matrix",
        "",
        f"Accuracy: {_fmt(sev.accuracy)}",
        "",
        header,
        sep,
    ]
    for i, true_label in enumerate(labels):
        cells: List[str] = []
        for j in range(len(labels)):
            if i < len(sev.matrix) and j < len(sev.matrix[i]):
                cells.append(str(int(sev.matrix[i][j])))
            else:
                cells.append("0")
        out.append(f"| **{true_label}** | " + " | ".join(cells) + " |")
    return out


def _md_time_to_detection(ttd: TimeToDetection) -> List[str]:
    out = [
        "## Time-to-Detection",
        "",
        f"- **Detected:** {ttd.detected_count}",
        f"- **Missed:** {ttd.missed_count}",
        f"- **Early (pre-onset positive):** {ttd.early_count}",
        "",
        "| Statistic | Value (turns) |",
        "| --- | --- |",
        f"| Median delay | {_fmt(ttd.median_delay)} |",
        f"| Mean delay | {_fmt(ttd.mean_delay)} |",
        f"| 90th-percentile delay | {_fmt(ttd.p90_delay)} |",
        "",
        "### Delay distribution",
        "",
    ]
    if len(ttd.delays) == 0:
        out.append("No detections (or no drift conversations).")
        return out
    out.append("```")
    out.extend(_delay_histogram(list(ttd.delays)))
    out.append("```")
    return out


def _md_signal_correlations(
    correlations: List[SignalCorrelation],
) -> List[str]:
    out = [
        "## Per-Signal Predictive Power",
        "",
        "| Signal | Pearson r | ROC-AUC |",
        "| --- | --- | --- |",
    ]
    for sc in correlations:
        out.append(
            f"| {sc.signal_name} | {_fmt(sc.pearson_r)} | {_fmt(sc.roc_auc)} |"
        )
    return out


def _md_threshold_sweep(sweep: List[ThresholdSweepRow]) -> List[str]:
    out = [
        "## Threshold Sweep",
        "",
        "| Threshold | Precision | Recall | F1 |",
        "| --- | --- | --- | --- |",
    ]
    best_idx = _argmax_f1_index(sweep)
    for i, row in enumerate(sweep):
        threshold_s = _fmt(row.threshold)
        precision_s = _fmt(row.precision)
        recall_s = _fmt(row.recall)
        f1_s = _fmt(row.f1)
        if i == best_idx:
            out.append(
                f"| **{threshold_s}** | **{precision_s}** | "
                f"**{recall_s}** | **{f1_s}** |"
            )
        else:
            out.append(
                f"| {threshold_s} | {precision_s} | {recall_s} | {f1_s} |"
            )
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_markdown(
    report: EvalReport,
    *,
    title: str = "Drift Detection Evaluation Report",
    dataset_name: Optional[str] = None,
    detector_name: Optional[str] = None,
    generated_at: Optional[str] = None,
    extra_notes: Optional[str] = None,
) -> str:
    """Render an :class:`EvalReport` as a markdown string.

    Sections (in order): title + metadata, overview, headline metrics,
    calibration, per-drift-type recall, severity confusion matrix,
    time-to-detection, per-signal predictive power, threshold sweep, and
    optional notes. Floats are rounded to 3 decimals; NaN/inf renders as
    ``"n/a"``. Empty evaluations short-circuit with a stub.
    """

    if report.n_turns == 0:
        empty_lines: List[str] = [f"# {title}", ""]
        empty_lines.extend(
            _metadata_block(
                dataset_name=dataset_name,
                detector_name=detector_name,
                generated_at=generated_at,
            )
        )
        empty_lines.extend(["", "Empty evaluation; no turns to report."])
        if extra_notes:
            empty_lines.extend(["", "## Notes", "", extra_notes])
        return "\n".join(empty_lines) + "\n"

    sections: List[List[str]] = [
        [f"# {title}", ""],
        _metadata_block(
            dataset_name=dataset_name,
            detector_name=detector_name,
            generated_at=generated_at,
        ),
        _md_overview(report),
        _md_headline(report.binary),
        _md_calibration(report.calibration),
        _md_per_drift_type(report.per_drift_type_recall),
        _md_severity_confusion(report.severity_confusion),
        _md_time_to_detection(report.time_to_detection),
        _md_signal_correlations(report.signal_correlations),
        _md_threshold_sweep(report.threshold_sweep),
    ]
    if extra_notes:
        sections.append(["## Notes", "", extra_notes])

    flat: List[str] = []
    for i, section in enumerate(sections):
        if i > 0:
            flat.append("")
        flat.extend(section)
    return "\n".join(flat) + "\n"


def render_json(
    report: EvalReport,
    *,
    dataset_name: Optional[str] = None,
    detector_name: Optional[str] = None,
    generated_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Render an :class:`EvalReport` as a JSON-ready dict.

    Mirrors :func:`render_markdown` content but as structured data.
    NaN/inf values are converted to ``None`` so the output round-trips
    through ``json.dump``. Numeric values are kept at full precision
    (no rounding).
    """

    ts = generated_at if generated_at is not None else _now_iso()

    if report.n_turns == 0:
        return {
            "title": "Drift Detection Evaluation Report",
            "metadata": {
                "dataset_name": dataset_name,
                "detector_name": detector_name,
                "generated_at": ts,
            },
            "empty": True,
            "message": "Empty evaluation; no turns to report.",
        }

    binary = report.binary
    cal = report.calibration
    sev = report.severity_confusion
    ttd = report.time_to_detection

    return {
        "metadata": {
            "dataset_name": dataset_name,
            "detector_name": detector_name,
            "generated_at": ts,
        },
        "overview": {
            "n_conversations": int(report.n_conversations),
            "n_turns": int(report.n_turns),
            "drift_rate": _json_num(report.drift_rate),
        },
        "binary_metrics": {
            "threshold": _json_num(binary.threshold),
            "precision": _json_num(binary.precision),
            "recall": _json_num(binary.recall),
            "f1": _json_num(binary.f1),
            "accuracy": _json_num(binary.accuracy),
            "true_positives": int(binary.true_positives),
            "false_positives": int(binary.false_positives),
            "true_negatives": int(binary.true_negatives),
            "false_negatives": int(binary.false_negatives),
        },
        "calibration": {
            "brier_score": _json_num(cal.brier_score),
            "roc_auc": _json_num(cal.roc_auc),
            "pr_auc": _json_num(cal.pr_auc),
        },
        "per_drift_type_recall": {
            drift_type: _json_num(
                report.per_drift_type_recall.get(drift_type, float("nan"))
            )
            for drift_type in _DRIFT_TYPE_ORDER
        },
        "severity_confusion": {
            "labels": list(sev.labels),
            "matrix": [[int(c) for c in row] for row in sev.matrix],
            "accuracy": _json_num(sev.accuracy),
        },
        "time_to_detection": {
            "detected_count": int(ttd.detected_count),
            "missed_count": int(ttd.missed_count),
            "early_count": int(ttd.early_count),
            "delays": [int(d) for d in ttd.delays],
            "median_delay": _json_num(ttd.median_delay),
            "mean_delay": _json_num(ttd.mean_delay),
            "p90_delay": _json_num(ttd.p90_delay),
        },
        "signal_correlations": [
            {
                "signal_name": sc.signal_name,
                "pearson_r": _json_num(sc.pearson_r),
                "roc_auc": _json_num(sc.roc_auc),
            }
            for sc in report.signal_correlations
        ],
        "threshold_sweep": [
            {
                "threshold": _json_num(row.threshold),
                "precision": _json_num(row.precision),
                "recall": _json_num(row.recall),
                "f1": _json_num(row.f1),
            }
            for row in report.threshold_sweep
        ],
    }


def write_report(
    report: EvalReport,
    output_dir: Path | str,
    *,
    name: str = "baseline",
    title: str = "Drift Detection Evaluation Report",
    dataset_name: Optional[str] = None,
    detector_name: Optional[str] = None,
    extra_notes: Optional[str] = None,
) -> Tuple[Path, Path]:
    """Render and write both markdown and JSON reports to ``output_dir``.

    Creates ``output_dir`` (and parents) if missing. Filenames are
    ``{name}.md`` and ``{name}.json``. Returns ``(md_path, json_path)``.
    """

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    generated_at = _now_iso()

    md_text = render_markdown(
        report,
        title=title,
        dataset_name=dataset_name,
        detector_name=detector_name,
        generated_at=generated_at,
        extra_notes=extra_notes,
    )
    json_obj = render_json(
        report,
        dataset_name=dataset_name,
        detector_name=detector_name,
        generated_at=generated_at,
    )

    md_path = out_path / f"{name}.md"
    json_path = out_path / f"{name}.json"

    md_path.write_text(md_text, encoding="utf-8")
    json_path.write_text(
        json.dumps(json_obj, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    return md_path, json_path
