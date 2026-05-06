# Copyright 2026 Stable-Agent Contributors
# SPDX-License-Identifier: MIT
"""Drift-detection evaluation metrics.

Pure functions that consume pre-collected evaluation rows and produce
aggregate metrics. No I/O, no LLM calls. Other modules (``harness.py``,
``report.py``) are responsible for collecting the rows and rendering
the resulting :class:`EvalReport`.

The headline data type is :class:`EvalRow` — one per-turn evaluation
result, pairing ground truth with the detector's prediction. The
headline output is :class:`EvalReport`, computed by
:func:`compute_full_report`.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EvalRow:
    """One per-turn evaluation result, paired ground-truth + prediction."""

    conversation_id: str
    turn: int
    drift_onset_turn: Optional[int]      # None if conversation has no drift, else 1-indexed turn
    # Ground truth
    true_drifted: bool
    true_severity: str                   # "none" | "low" | "medium" | "high"
    true_drift_types: List[str]
    true_violated_constraints: List[int]
    # Prediction
    pred_score: float                    # continuous total drift score, [0, 1]
    pred_drifted: bool                   # binary at detector's threshold
    pred_severity: str
    pred_violated_constraints: List[int]
    # Per-signal scores (so we can analyze each signal independently)
    pred_goal_drift: float
    pred_constraint_drift: float
    pred_consistency_drift: float


@dataclass
class BinaryMetrics:
    threshold: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int


@dataclass
class CalibrationMetrics:
    brier_score: float                   # mean squared error between pred_score and true_drifted (0/1)
    roc_auc: float                       # area under ROC; 0.5 = random, 1.0 = perfect
    pr_auc: float                        # area under precision-recall


@dataclass
class TimeToDetection:
    """For conversations with drift, how many turns after onset before we flagged it."""

    detected_count: int                   # convs where drift was eventually detected
    missed_count: int                     # convs where drift was never detected
    early_count: int                      # convs flagged BEFORE actual onset (false positives)
    delays: List[int]                    # turns between onset and first detection (>=0); excludes missed
    median_delay: float
    mean_delay: float
    p90_delay: float                     # 90th percentile delay


@dataclass
class SeverityConfusion:
    labels: List[str]                    # ["none", "low", "medium", "high"]
    matrix: List[List[int]]              # matrix[i][j] = count of (true=labels[i], pred=labels[j])
    accuracy: float


@dataclass
class SignalCorrelation:
    """How well each individual signal predicts true drift on its own."""

    signal_name: str                     # "goal", "constraint", "consistency", "total"
    pearson_r: float                     # correlation with true_drifted (as 0/1)
    roc_auc: float


@dataclass
class ThresholdSweepRow:
    threshold: float
    precision: float
    recall: float
    f1: float


@dataclass
class EvalReport:
    """All metrics. Consumed by report.py."""

    n_conversations: int
    n_turns: int
    drift_rate: float                    # fraction of turns with true_drifted=True
    binary: BinaryMetrics                # at the detector's configured threshold
    calibration: CalibrationMetrics
    time_to_detection: TimeToDetection
    severity_confusion: SeverityConfusion
    signal_correlations: List[SignalCorrelation]
    threshold_sweep: List[ThresholdSweepRow]
    # Per-drift-type breakdown (recall when only that drift type is present)
    per_drift_type_recall: Dict[str, float]   # keys: "goal", "constraint", "consistency"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


SEVERITY_LABELS: List[str] = ["none", "low", "medium", "high"]
DRIFT_TYPES: Tuple[str, ...] = ("goal", "constraint", "consistency")
_DEFAULT_SWEEP_THRESHOLDS: List[float] = [round(0.05 * i, 2) for i in range(1, 20)]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_nonempty(rows: Sequence[EvalRow]) -> None:
    if len(rows) == 0:
        raise ValueError("rows must be non-empty")


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _binary_counts(
    rows: Sequence[EvalRow], threshold: float
) -> Tuple[int, int, int, int]:
    """Return (tp, fp, tn, fn) using ``pred_score >= threshold``."""

    tp = fp = tn = fn = 0
    for row in rows:
        predicted = row.pred_score >= threshold
        actual = bool(row.true_drifted)
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif (not predicted) and (not actual):
            tn += 1
        else:
            fn += 1
    return tp, fp, tn, fn


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation manually; return 0.0 if either input is constant."""

    if x.size != y.size or x.size < 2:
        return 0.0
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std == 0.0 or y_std == 0.0:
        return 0.0
    x_centered = x - float(np.mean(x))
    y_centered = y - float(np.mean(y))
    numerator = float(np.sum(x_centered * y_centered))
    denominator = math.sqrt(
        float(np.sum(x_centered * x_centered))
        * float(np.sum(y_centered * y_centered))
    )
    if denominator == 0.0:
        return 0.0
    r = numerator / denominator
    # Clamp tiny FP overshoots into [-1, 1].
    if r > 1.0:
        return 1.0
    if r < -1.0:
        return -1.0
    return r


def _roc_auc_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC AUC that returns NaN when only one class is present."""

    unique = np.unique(y_true)
    if unique.size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _pr_auc_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Precision-recall AUC; if no positives, undefined → NaN."""

    if not np.any(y_true == 1):
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def _percentile(values: Sequence[float], q: float) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=float), q))


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def compute_binary_metrics(
    rows: List[EvalRow], threshold: float
) -> BinaryMetrics:
    """Recompute binary metrics from ``pred_score >= threshold``."""

    _ensure_nonempty(rows)
    tp, fp, tn, fn = _binary_counts(rows, threshold)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _f1(precision, recall)
    accuracy = _safe_div(tp + tn, tp + fp + tn + fn)
    return BinaryMetrics(
        threshold=float(threshold),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        accuracy=float(accuracy),
        true_positives=int(tp),
        false_positives=int(fp),
        true_negatives=int(tn),
        false_negatives=int(fn),
    )


def compute_calibration(rows: List[EvalRow]) -> CalibrationMetrics:
    """Calibration: Brier score + ROC AUC + PR AUC."""

    _ensure_nonempty(rows)
    y_true = np.asarray(
        [1 if row.true_drifted else 0 for row in rows], dtype=int
    )
    y_score = np.asarray([float(row.pred_score) for row in rows], dtype=float)

    # Brier is well-defined regardless of class balance.
    if np.unique(y_true).size < 2:
        # ``brier_score_loss`` requires at least one of each class for the
        # default ``pos_label`` inference; compute it manually instead.
        brier = float(np.mean((y_score - y_true.astype(float)) ** 2))
    else:
        brier = float(brier_score_loss(y_true, y_score))

    return CalibrationMetrics(
        brier_score=brier,
        roc_auc=_roc_auc_safe(y_true, y_score),
        pr_auc=_pr_auc_safe(y_true, y_score),
    )


def compute_time_to_detection(
    rows: List[EvalRow], threshold: float
) -> TimeToDetection:
    """How many turns after onset until the detector flagged drift."""

    _ensure_nonempty(rows)

    # Group rows by conversation_id, sorted by turn ascending.
    grouped: Dict[str, List[EvalRow]] = {}
    for row in rows:
        grouped.setdefault(row.conversation_id, []).append(row)
    for conv_id in grouped:
        grouped[conv_id].sort(key=lambda r: r.turn)

    detected_count = 0
    missed_count = 0
    early_count = 0
    delays: List[int] = []

    for conv_rows in grouped.values():
        onset = conv_rows[0].drift_onset_turn
        if onset is None:
            # No drift to detect for this conversation.
            continue

        # Any pre-onset positive prediction → "early" (false positive on time).
        early_flag = any(
            (r.pred_score >= threshold) and (r.turn < onset) for r in conv_rows
        )
        if early_flag:
            early_count += 1

        # First post-onset detection (turn >= onset).
        first_post_onset_detection: Optional[int] = None
        for r in conv_rows:
            if r.turn >= onset and r.pred_score >= threshold:
                first_post_onset_detection = r.turn
                break

        if first_post_onset_detection is not None:
            detected_count += 1
            delay = first_post_onset_detection - onset
            if delay < 0:
                delay = 0
            delays.append(int(delay))
        else:
            missed_count += 1

    if len(delays) == 0:
        median_delay = float("nan")
        mean_delay = float("nan")
        p90_delay = float("nan")
    else:
        median_delay = float(statistics.median(delays))
        mean_delay = float(statistics.fmean(delays))
        p90_delay = _percentile(delays, 90.0)

    return TimeToDetection(
        detected_count=int(detected_count),
        missed_count=int(missed_count),
        early_count=int(early_count),
        delays=delays,
        median_delay=median_delay,
        mean_delay=mean_delay,
        p90_delay=p90_delay,
    )


def compute_severity_confusion(rows: List[EvalRow]) -> SeverityConfusion:
    """4x4 severity confusion matrix over ``["none", "low", "medium", "high"]``."""

    _ensure_nonempty(rows)

    label_to_idx = {label: i for i, label in enumerate(SEVERITY_LABELS)}
    n = len(SEVERITY_LABELS)
    matrix: List[List[int]] = [[0 for _ in range(n)] for _ in range(n)]

    correct = 0
    counted = 0
    for row in rows:
        if row.true_severity not in label_to_idx:
            continue
        if row.pred_severity not in label_to_idx:
            continue
        i = label_to_idx[row.true_severity]
        j = label_to_idx[row.pred_severity]
        matrix[i][j] += 1
        counted += 1
        if i == j:
            correct += 1

    accuracy = _safe_div(correct, counted)
    return SeverityConfusion(
        labels=list(SEVERITY_LABELS),
        matrix=matrix,
        accuracy=float(accuracy),
    )


def compute_signal_correlations(
    rows: List[EvalRow],
) -> List[SignalCorrelation]:
    """Pearson correlation + ROC AUC for each individual signal vs. true drift."""

    _ensure_nonempty(rows)

    y_true = np.asarray(
        [1.0 if row.true_drifted else 0.0 for row in rows], dtype=float
    )
    signals: List[Tuple[str, np.ndarray]] = [
        (
            "goal",
            np.asarray([float(r.pred_goal_drift) for r in rows], dtype=float),
        ),
        (
            "constraint",
            np.asarray(
                [float(r.pred_constraint_drift) for r in rows], dtype=float
            ),
        ),
        (
            "consistency",
            np.asarray(
                [float(r.pred_consistency_drift) for r in rows], dtype=float
            ),
        ),
        (
            "total",
            np.asarray([float(r.pred_score) for r in rows], dtype=float),
        ),
    ]

    out: List[SignalCorrelation] = []
    for name, scores in signals:
        r = _pearson_r(scores, y_true)
        auc = _roc_auc_safe(y_true.astype(int), scores)
        out.append(
            SignalCorrelation(
                signal_name=name,
                pearson_r=float(r),
                roc_auc=float(auc),
            )
        )
    return out


def compute_threshold_sweep(
    rows: List[EvalRow], thresholds: Optional[List[float]] = None
) -> List[ThresholdSweepRow]:
    """Sweep precision/recall/F1 across candidate thresholds."""

    _ensure_nonempty(rows)

    if thresholds is None:
        thresholds = list(_DEFAULT_SWEEP_THRESHOLDS)

    sweep: List[ThresholdSweepRow] = []
    for t in thresholds:
        bm = compute_binary_metrics(rows, float(t))
        sweep.append(
            ThresholdSweepRow(
                threshold=float(t),
                precision=bm.precision,
                recall=bm.recall,
                f1=bm.f1,
            )
        )
    return sweep


def compute_per_drift_type_recall(
    rows: List[EvalRow], threshold: float
) -> Dict[str, float]:
    """Recall per drift type, restricted to rows where exactly one type is present."""

    _ensure_nonempty(rows)

    out: Dict[str, float] = {}
    for drift_type in DRIFT_TYPES:
        relevant = [
            r
            for r in rows
            if r.true_drifted
            and set(r.true_drift_types) == {drift_type}
        ]
        if len(relevant) == 0:
            out[drift_type] = float("nan")
            continue
        detected = sum(1 for r in relevant if r.pred_score >= threshold)
        out[drift_type] = float(detected) / float(len(relevant))
    return out


def compute_full_report(
    rows: List[EvalRow], detector_threshold: float
) -> EvalReport:
    """Run every metric and bundle them into a single :class:`EvalReport`."""

    _ensure_nonempty(rows)

    n_turns = len(rows)
    conversation_ids = {r.conversation_id for r in rows}
    n_conversations = len(conversation_ids)
    drift_rate = _safe_div(
        sum(1 for r in rows if r.true_drifted), n_turns
    )

    binary = compute_binary_metrics(rows, detector_threshold)
    calibration = compute_calibration(rows)
    ttd = compute_time_to_detection(rows, detector_threshold)
    severity = compute_severity_confusion(rows)
    signal_corrs = compute_signal_correlations(rows)
    sweep = compute_threshold_sweep(rows)
    per_type_recall = compute_per_drift_type_recall(rows, detector_threshold)

    return EvalReport(
        n_conversations=int(n_conversations),
        n_turns=int(n_turns),
        drift_rate=float(drift_rate),
        binary=binary,
        calibration=calibration,
        time_to_detection=ttd,
        severity_confusion=severity,
        signal_correlations=signal_corrs,
        threshold_sweep=sweep,
        per_drift_type_recall=per_type_recall,
    )
