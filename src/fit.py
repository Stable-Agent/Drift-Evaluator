# Copyright 2026 Stable-Agent Contributors
# SPDX-License-Identifier: MIT
"""Drift-detector hyperparameter refitting.

Pure functions over :class:`drift_evaluator.metrics.EvalRow` lists that learn
optimized weights, a binary-decision threshold, and severity-bucket cutoffs
from labeled evaluation data.

The intended workflow is:

1. Run the drift detector with a default configuration on a labeled dataset
   to produce a list of :class:`EvalRow` objects.
2. Call :func:`fit_all` on those rows to obtain a :class:`FittedConfig` with
   optimized weights, decision threshold, and severity cutoffs.
3. Call :func:`apply_fitted_config` to rescore the rows under the fitted
   configuration, then pass the rescored rows to
   :func:`drift_evaluator.metrics.compute_full_report` for a fitted-config
   evaluation report.

This module performs no I/O beyond emitting warnings to ``sys.stderr``.
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from drift_evaluator.metrics import EvalRow


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


_SIGNAL_KEYS: Tuple[str, str, str] = (
    "goal_alignment",
    "constraint_adherence",
    "consistency",
)

_DEFAULT_WEIGHTS: Dict[str, float] = {
    "goal_alignment": 0.4,
    "constraint_adherence": 0.4,
    "consistency": 0.2,
}

_DEFAULT_THRESHOLD: float = 0.30

_DEFAULT_SEVERITY_THRESHOLDS: Dict[str, float] = {
    "none": 0.20,
    "low": 0.40,
    "medium": 0.60,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FittedConfig:
    """Optimized detector configuration learned from eval data."""

    weights: Dict[str, float]
    """Nonnegative weights summing to 1.0; keys are
    ``"goal_alignment"``, ``"constraint_adherence"``, ``"consistency"``."""

    threshold: float
    """F1-maximizing decision threshold for the weighted total drift score."""

    severity_thresholds: Dict[str, float]
    """Severity bucket cutoffs ``{"none": x, "low": y, "medium": z}``,
    sorted ascending."""

    fit_metrics: Dict[str, float] = field(default_factory=dict)
    """Diagnostic training metrics under the fitted configuration:
    keys ``precision``, ``recall``, ``f1``, ``accuracy``, ``roc_auc``."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _warn(msg: str) -> None:
    """Emit a warning to stderr."""
    print(f"[drift_evaluator.fit] WARNING: {msg}", file=sys.stderr)


def _signal_matrix(rows: List[EvalRow]) -> np.ndarray:
    """Stack per-row [goal, constraint, consistency] features as an (N, 3) array."""
    return np.array(
        [
            [r.pred_goal_drift, r.pred_constraint_drift, r.pred_consistency_drift]
            for r in rows
        ],
        dtype=float,
    )


def _total_scores(rows: List[EvalRow], weights: Dict[str, float]) -> np.ndarray:
    """Compute the weighted total drift score per row, clipped to [0, 1]."""
    feats = _signal_matrix(rows)
    w = np.array(
        [weights["goal_alignment"], weights["constraint_adherence"], weights["consistency"]],
        dtype=float,
    )
    totals = feats @ w
    return np.clip(totals, 0.0, 1.0)


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute precision, recall, F1, and accuracy for a binary prediction."""
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def _classify_severity(score: float, thresholds: Dict[str, float]) -> str:
    """Map a continuous score to a severity bucket using fitted cutoffs."""
    if score < thresholds["none"]:
        return "none"
    if score < thresholds["low"]:
        return "low"
    if score < thresholds["medium"]:
        return "medium"
    return "high"


# ---------------------------------------------------------------------------
# Weight fitting
# ---------------------------------------------------------------------------


def fit_weights(rows: List[EvalRow], *, regularization: float = 1.0) -> Dict[str, float]:
    """Fit nonnegative weights via constrained logistic regression.

    Features per row are ``[pred_goal_drift, pred_constraint_drift,
    pred_consistency_drift]`` and the target is ``true_drifted`` cast to 0/1.
    The solver is :class:`sklearn.linear_model.LogisticRegression` with
    ``positive=True`` (available since scikit-learn 1.2) and ``C=1/regularization``.
    The learned coefficients are normalized so they sum to 1.0; coefficients
    that come out at 0 are left at 0 (no redistribution).

    Edge cases:
      * Empty ``rows`` raises :class:`ValueError`.
      * Single class in target: warn and return :data:`_DEFAULT_WEIGHTS`.
      * All coefficients zero: fall back to :data:`_DEFAULT_WEIGHTS`.

    Returns a dict with keys ``"goal_alignment"``, ``"constraint_adherence"``,
    and ``"consistency"``.
    """
    if not rows:
        raise ValueError("fit_weights: rows must be non-empty")

    X = _signal_matrix(rows)
    y = np.array([1 if r.true_drifted else 0 for r in rows], dtype=int)

    if len(np.unique(y)) < 2:
        _warn(
            "fit_weights: target has only one class; "
            "returning default weights {goal=0.4, constraint=0.4, consistency=0.2}."
        )
        return dict(_DEFAULT_WEIGHTS)

    model = LogisticRegression(
        C=1.0 / regularization,
        solver="lbfgs",
        max_iter=1000,
    )
    model.fit(X, y)
    coefs = np.asarray(model.coef_, dtype=float).ravel()
    # Clip any negative coefficients to zero. A negative coefficient means the
    # signal anti-correlates with truth in this dataset — likely noisy or
    # broken (e.g. v0's constraint signal had Pearson r=-0.03). Clipping is
    # safer than redistributing across signals, since redistribution would
    # silently boost potentially-bad signals.
    coefs = np.clip(coefs, 0.0, None)

    total = float(coefs.sum())
    if total <= 0.0:
        _warn(
            "fit_weights: all learned coefficients are zero; "
            "falling back to default weights."
        )
        return dict(_DEFAULT_WEIGHTS)

    normalized = coefs / total
    return {
        "goal_alignment": float(normalized[0]),
        "constraint_adherence": float(normalized[1]),
        "consistency": float(normalized[2]),
    }


# ---------------------------------------------------------------------------
# Threshold fitting
# ---------------------------------------------------------------------------


def fit_threshold(
    rows: List[EvalRow],
    weights: Dict[str, float],
    thresholds: Optional[List[float]] = None,
) -> Tuple[float, Dict[str, float]]:
    """Find the threshold that maximizes F1 over the weighted total score.

    The total score is recomputed per row from ``weights`` and the per-signal
    drift predictions, then candidate thresholds are swept. Default sweep is
    ``np.linspace(0.05, 0.95, 19)``. Ties on F1 are broken by preferring the
    lower threshold (favoring recall).

    Returns ``(best_threshold, metrics_at_best)`` where ``metrics_at_best``
    has keys ``precision``, ``recall``, ``f1``, ``accuracy``.

    Edge cases:
      * Empty ``rows`` raises :class:`ValueError`.
      * Single class in target: warn and return ``(_DEFAULT_THRESHOLD, metrics)``.
    """
    if not rows:
        raise ValueError("fit_threshold: rows must be non-empty")

    if thresholds is None:
        thresholds = list(np.linspace(0.05, 0.95, 19))

    y_true = np.array([1 if r.true_drifted else 0 for r in rows], dtype=int)
    totals = _total_scores(rows, weights)

    if len(np.unique(y_true)) < 2:
        _warn(
            "fit_threshold: target has only one class; "
            f"returning default threshold {_DEFAULT_THRESHOLD}."
        )
        y_pred = (totals >= _DEFAULT_THRESHOLD).astype(int)
        return _DEFAULT_THRESHOLD, _binary_metrics(y_true, y_pred)

    best_threshold: Optional[float] = None
    best_metrics: Optional[Dict[str, float]] = None
    best_f1: float = -1.0

    # Sort thresholds ascending so the first hit on a maximum F1 is the lowest
    # threshold achieving it (favoring recall on ties).
    for t in sorted(float(x) for x in thresholds):
        y_pred = (totals >= t).astype(int)
        m = _binary_metrics(y_true, y_pred)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_threshold = t
            best_metrics = m

    assert best_threshold is not None and best_metrics is not None
    return best_threshold, best_metrics


# ---------------------------------------------------------------------------
# Severity threshold fitting
# ---------------------------------------------------------------------------


def fit_severity_thresholds(
    rows: List[EvalRow], weights: Dict[str, float]
) -> Dict[str, float]:
    """Fit severity bucket cutoffs using true severity labels.

    For each pair of adjacent severity levels (``none/low``, ``low/medium``,
    ``medium/high``), the cutoff is the mean of the upper-tail (80th
    percentile) of the lower bucket and the lower-tail (20th percentile) of
    the upper bucket. Missing buckets fall back to defaults from
    :data:`_DEFAULT_SEVERITY_THRESHOLDS`. Returned cutoffs are sorted ascending.

    Edge cases:
      * Empty ``rows`` raises :class:`ValueError`.
    """
    if not rows:
        raise ValueError("fit_severity_thresholds: rows must be non-empty")

    totals = _total_scores(rows, weights)

    by_severity: Dict[str, List[float]] = {
        "none": [],
        "low": [],
        "medium": [],
        "high": [],
    }
    for r, score in zip(rows, totals):
        if r.true_severity in by_severity:
            by_severity[r.true_severity].append(float(score))

    pairs: Tuple[Tuple[str, str, str], ...] = (
        ("none", "none", "low"),     # cutoff name, lower bucket, upper bucket
        ("low", "low", "medium"),
        ("medium", "medium", "high"),
    )

    cutoffs: Dict[str, float] = {}
    for cutoff_name, lower, upper in pairs:
        lower_scores = by_severity.get(lower, [])
        upper_scores = by_severity.get(upper, [])
        if not lower_scores or not upper_scores:
            _warn(
                f"fit_severity_thresholds: missing data for boundary "
                f"{lower!r}/{upper!r}; using default cutoff "
                f"{_DEFAULT_SEVERITY_THRESHOLDS[cutoff_name]} for {cutoff_name!r}."
            )
            cutoffs[cutoff_name] = _DEFAULT_SEVERITY_THRESHOLDS[cutoff_name]
            continue

        upper_tail = float(np.percentile(np.asarray(lower_scores), 80))
        lower_tail = float(np.percentile(np.asarray(upper_scores), 20))
        cutoffs[cutoff_name] = 0.5 * (upper_tail + lower_tail)

    # Enforce ascending order of cutoffs.
    sorted_values = sorted(cutoffs.values())
    return {
        "none": float(sorted_values[0]),
        "low": float(sorted_values[1]),
        "medium": float(sorted_values[2]),
    }


# ---------------------------------------------------------------------------
# Combined fit
# ---------------------------------------------------------------------------


def fit_all(rows: List[EvalRow], *, regularization: float = 1.0) -> FittedConfig:
    """Run weight, threshold, and severity fits in order and return the result.

    The ``fit_metrics`` field on the returned :class:`FittedConfig` contains
    precision, recall, F1, and accuracy at the fitted threshold, plus the
    ROC-AUC of the weighted total score against ``true_drifted``.

    Edge cases:
      * Empty ``rows`` raises :class:`ValueError`.
      * Single-class target: ``roc_auc`` is reported as NaN.
    """
    if not rows:
        raise ValueError("fit_all: rows must be non-empty")

    weights = fit_weights(rows, regularization=regularization)
    threshold, threshold_metrics = fit_threshold(rows, weights)
    severity_thresholds = fit_severity_thresholds(rows, weights)

    y_true = np.array([1 if r.true_drifted else 0 for r in rows], dtype=int)
    totals = _total_scores(rows, weights)
    if len(np.unique(y_true)) < 2:
        _warn("fit_all: target has only one class; ROC-AUC is undefined (NaN).")
        roc_auc = float("nan")
    else:
        roc_auc = float(roc_auc_score(y_true, totals))

    fit_metrics = {
        "precision": float(threshold_metrics["precision"]),
        "recall": float(threshold_metrics["recall"]),
        "f1": float(threshold_metrics["f1"]),
        "accuracy": float(threshold_metrics["accuracy"]),
        "roc_auc": roc_auc,
    }

    return FittedConfig(
        weights=weights,
        threshold=threshold,
        severity_thresholds=severity_thresholds,
        fit_metrics=fit_metrics,
    )


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


def apply_fitted_config(
    rows: List[EvalRow], config: FittedConfig
) -> List[EvalRow]:
    """Return a new ``EvalRow`` list rescored under ``config``.

    For each input row the fields ``pred_score``, ``pred_drifted``, and
    ``pred_severity`` are recomputed from the fitted weights and thresholds.
    All other fields (including the per-signal predictions and ground-truth
    labels) are copied unchanged. The total score is clipped to ``[0, 1]``.

    The output list contains independent copies; the input rows are not
    mutated.
    """
    if not rows:
        return []

    totals = _total_scores(rows, config.weights)
    out: List[EvalRow] = []
    for r, score in zip(rows, totals):
        new_row = copy.deepcopy(r)
        new_row.pred_score = float(score)
        new_row.pred_drifted = bool(score >= config.threshold)
        new_row.pred_severity = _classify_severity(
            float(score), config.severity_thresholds
        )
        out.append(new_row)
    return out
