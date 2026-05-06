# Copyright 2026 Stable-Agent Contributors
# SPDX-License-Identifier: MIT
"""Unit tests for ``drift_evaluator.metrics``.

These tests are pure: they construct :class:`EvalRow` instances directly
and assert properties of the metrics functions. No detector, dataset, or
LLM is involved.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional

import pytest

from drift_evaluator.metrics import (
    EvalReport,
    EvalRow,
    compute_binary_metrics,
    compute_calibration,
    compute_full_report,
    compute_per_drift_type_recall,
    compute_severity_confusion,
    compute_signal_correlations,
    compute_threshold_sweep,
    compute_time_to_detection,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def make_row(**overrides: Any) -> EvalRow:
    """Construct an :class:`EvalRow` with sensible defaults.

    Tests override individual fields via keyword arguments.
    """
    defaults = {
        "conversation_id": "conv_0001",
        "turn": 1,
        "drift_onset_turn": None,
        "true_drifted": False,
        "true_severity": "none",
        "true_drift_types": [],
        "true_violated_constraints": [],
        "pred_score": 0.0,
        "pred_drifted": False,
        "pred_severity": "none",
        "pred_violated_constraints": [],
        "pred_goal_drift": 0.0,
        "pred_constraint_drift": 0.0,
        "pred_consistency_drift": 0.0,
    }
    defaults.update(overrides)
    return EvalRow(**defaults)


# ---------------------------------------------------------------------------
# Binary metrics
# ---------------------------------------------------------------------------


def test_binary_metrics_perfect_predictor() -> None:
    rows: List[EvalRow] = [
        make_row(turn=1, true_drifted=False, pred_score=0.0),
        make_row(turn=2, true_drifted=True, pred_score=1.0),
        make_row(turn=3, true_drifted=False, pred_score=0.0),
        make_row(turn=4, true_drifted=True, pred_score=1.0),
    ]
    bm = compute_binary_metrics(rows, threshold=0.5)
    assert bm.precision == pytest.approx(1.0)
    assert bm.recall == pytest.approx(1.0)
    assert bm.f1 == pytest.approx(1.0)
    assert bm.accuracy == pytest.approx(1.0)
    assert bm.false_positives == 0
    assert bm.false_negatives == 0
    assert bm.true_positives == 2
    assert bm.true_negatives == 2


def test_binary_metrics_all_negative_pred() -> None:
    rows: List[EvalRow] = [
        make_row(turn=1, true_drifted=True, pred_score=0.05),
        make_row(turn=2, true_drifted=True, pred_score=0.10),
        make_row(turn=3, true_drifted=False, pred_score=0.00),
    ]
    bm = compute_binary_metrics(rows, threshold=0.5)
    assert bm.recall == pytest.approx(0.0)
    assert bm.f1 == pytest.approx(0.0)
    # Precision is 0/0 → defined as 0.0 by the metrics module.
    assert bm.precision == pytest.approx(0.0)
    assert bm.true_positives == 0
    assert bm.false_positives == 0
    assert bm.false_negatives == 2


def test_binary_metrics_threshold_recompute() -> None:
    """Even if ``pred_drifted=False``, a high ``pred_score`` should count
    as predicted positive when the threshold is below it.
    """
    rows: List[EvalRow] = [
        make_row(
            turn=1,
            true_drifted=True,
            pred_score=0.8,
            pred_drifted=False,  # contradicts pred_score, intentionally.
        ),
    ]
    bm = compute_binary_metrics(rows, threshold=0.3)
    assert bm.true_positives == 1
    assert bm.false_negatives == 0
    assert bm.precision == pytest.approx(1.0)
    assert bm.recall == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def test_calibration_brier() -> None:
    # Perfect predictions: Brier = 0.0
    rows: List[EvalRow] = [
        make_row(turn=1, true_drifted=True, pred_score=1.0),
        make_row(turn=2, true_drifted=False, pred_score=0.0),
    ]
    cal = compute_calibration(rows)
    assert cal.brier_score == pytest.approx(0.0, abs=1e-9)

    # Perturbed: Brier = ((1 - 0.9)^2 + (0 - 0.1)^2) / 2 = 0.01
    rows_perturbed: List[EvalRow] = [
        make_row(turn=1, true_drifted=True, pred_score=0.9),
        make_row(turn=2, true_drifted=False, pred_score=0.1),
    ]
    cal_perturbed = compute_calibration(rows_perturbed)
    assert cal_perturbed.brier_score == pytest.approx(0.01, abs=1e-9)


def test_calibration_single_class_returns_nan_auc() -> None:
    rows: List[EvalRow] = [
        make_row(turn=1, true_drifted=True, pred_score=0.7),
        make_row(turn=2, true_drifted=True, pred_score=0.8),
        make_row(turn=3, true_drifted=True, pred_score=0.9),
    ]
    cal = compute_calibration(rows)
    # ROC-AUC requires both classes; undefined here.
    assert math.isnan(cal.roc_auc)
    # PR-AUC is defined when only positives exist (degenerate 1.0 — every recall
    # level has precision 1.0). Brier is also well-defined.
    assert not math.isnan(cal.pr_auc)
    assert not math.isnan(cal.brier_score)


# ---------------------------------------------------------------------------
# Time to detection
# ---------------------------------------------------------------------------


def _drifted_row(
    *,
    conv_id: str,
    turn: int,
    onset: Optional[int],
    score: float,
    true_drifted: bool,
) -> EvalRow:
    return make_row(
        conversation_id=conv_id,
        turn=turn,
        drift_onset_turn=onset,
        true_drifted=true_drifted,
        true_severity="medium" if true_drifted else "none",
        true_drift_types=["goal"] if true_drifted else [],
        pred_score=score,
    )


def test_time_to_detection_basic() -> None:
    # conv_A: onset=3, scores rise above threshold at turn 3 (delay=0).
    scores_a = [0.1, 0.2, 0.5, 0.6, 0.7]
    conv_a_rows = [
        _drifted_row(
            conv_id="conv_A",
            turn=i + 1,
            onset=3,
            score=score,
            true_drifted=(i + 1) >= 3,
        )
        for i, score in enumerate(scores_a)
    ]

    # conv_B: onset=2, all scores stay below threshold → never detected.
    scores_b = [0.1, 0.1, 0.1, 0.1]
    conv_b_rows = [
        _drifted_row(
            conv_id="conv_B",
            turn=i + 1,
            onset=2,
            score=score,
            true_drifted=(i + 1) >= 2,
        )
        for i, score in enumerate(scores_b)
    ]

    # Clean conversation should be skipped entirely.
    conv_c_rows = [
        _drifted_row(
            conv_id="conv_C",
            turn=i + 1,
            onset=None,
            score=0.0,
            true_drifted=False,
        )
        for i in range(3)
    ]

    rows: List[EvalRow] = conv_a_rows + conv_b_rows + conv_c_rows
    ttd = compute_time_to_detection(rows, threshold=0.3)

    assert ttd.detected_count == 1
    assert ttd.missed_count == 1
    assert ttd.delays == [0]
    assert ttd.median_delay == pytest.approx(0.0)
    assert ttd.mean_delay == pytest.approx(0.0)


def test_time_to_detection_early_detection() -> None:
    # onset=4. Pre-onset spike at turn 1 → early. Post-onset detection at
    # turn 4 → delay=0.
    scores = [0.5, 0.1, 0.1, 0.5, 0.6]
    rows = [
        _drifted_row(
            conv_id="conv_E",
            turn=i + 1,
            onset=4,
            score=score,
            true_drifted=(i + 1) >= 4,
        )
        for i, score in enumerate(scores)
    ]
    ttd = compute_time_to_detection(rows, threshold=0.3)
    assert ttd.early_count >= 1
    assert ttd.detected_count == 1
    assert ttd.missed_count == 0
    assert ttd.delays == [0]


def test_time_to_detection_empty_drift_convs() -> None:
    rows = [
        _drifted_row(
            conv_id="conv_clean",
            turn=i + 1,
            onset=None,
            score=0.05,
            true_drifted=False,
        )
        for i in range(4)
    ]
    ttd = compute_time_to_detection(rows, threshold=0.3)
    assert ttd.detected_count == 0
    assert ttd.missed_count == 0
    assert ttd.early_count == 0
    assert ttd.delays == []
    assert math.isnan(ttd.median_delay)
    assert math.isnan(ttd.mean_delay)
    assert math.isnan(ttd.p90_delay)


# ---------------------------------------------------------------------------
# Severity confusion
# ---------------------------------------------------------------------------


def test_severity_confusion_perfect() -> None:
    severities = ["none", "low", "medium", "high"]
    rows: List[EvalRow] = []
    for i, sev in enumerate(severities):
        rows.append(
            make_row(
                turn=i + 1,
                true_drifted=(sev != "none"),
                true_severity=sev,
                true_drift_types=[] if sev == "none" else ["goal"],
                pred_severity=sev,
            )
        )
    sc = compute_severity_confusion(rows)
    assert sc.labels == ["none", "low", "medium", "high"]
    for i in range(4):
        for j in range(4):
            expected = 1 if i == j else 0
            assert sc.matrix[i][j] == expected
    assert sc.accuracy == pytest.approx(1.0)


def test_severity_confusion_off_diagonal() -> None:
    rows = [
        make_row(
            turn=1,
            true_drifted=True,
            true_severity="high",
            true_drift_types=["goal"],
            pred_severity="low",
        ),
    ]
    sc = compute_severity_confusion(rows)
    # labels = ["none", "low", "medium", "high"] → true=high (idx 3), pred=low (idx 1).
    assert sc.matrix[3][1] == 1
    # All other cells must be zero.
    for i in range(4):
        for j in range(4):
            if (i, j) != (3, 1):
                assert sc.matrix[i][j] == 0
    assert sc.accuracy == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Signal correlations
# ---------------------------------------------------------------------------


def test_signal_correlations_perfect() -> None:
    # ``pred_goal_drift`` exactly matches ``true_drifted`` cast to float.
    rows = [
        make_row(
            turn=1,
            true_drifted=False,
            pred_goal_drift=0.0,
            pred_constraint_drift=0.7,
            pred_consistency_drift=0.2,
            pred_score=0.0,
        ),
        make_row(
            turn=2,
            true_drifted=True,
            true_severity="medium",
            true_drift_types=["goal"],
            pred_goal_drift=1.0,
            pred_constraint_drift=0.4,
            pred_consistency_drift=0.1,
            pred_score=1.0,
        ),
        make_row(
            turn=3,
            true_drifted=False,
            pred_goal_drift=0.0,
            pred_constraint_drift=0.6,
            pred_consistency_drift=0.5,
            pred_score=0.0,
        ),
        make_row(
            turn=4,
            true_drifted=True,
            true_severity="high",
            true_drift_types=["goal"],
            pred_goal_drift=1.0,
            pred_constraint_drift=0.3,
            pred_consistency_drift=0.4,
            pred_score=1.0,
        ),
    ]
    corrs = compute_signal_correlations(rows)
    by_name = {c.signal_name: c for c in corrs}
    assert set(by_name.keys()) == {"goal", "constraint", "consistency", "total"}
    assert by_name["goal"].pearson_r == pytest.approx(1.0, abs=1e-9)


def test_signal_correlations_constant_signal_returns_zero() -> None:
    rows = [
        make_row(turn=1, true_drifted=False, pred_goal_drift=0.5),
        make_row(
            turn=2,
            true_drifted=True,
            true_severity="medium",
            true_drift_types=["goal"],
            pred_goal_drift=0.5,
        ),
        make_row(turn=3, true_drifted=False, pred_goal_drift=0.5),
        make_row(
            turn=4,
            true_drifted=True,
            true_severity="medium",
            true_drift_types=["goal"],
            pred_goal_drift=0.5,
        ),
    ]
    corrs = compute_signal_correlations(rows)
    by_name = {c.signal_name: c for c in corrs}
    assert by_name["goal"].pearson_r == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------


def test_threshold_sweep_default() -> None:
    rows = [
        make_row(turn=1, true_drifted=False, pred_score=0.1),
        make_row(
            turn=2,
            true_drifted=True,
            true_severity="medium",
            true_drift_types=["goal"],
            pred_score=0.7,
        ),
    ]
    sweep = compute_threshold_sweep(rows)
    assert len(sweep) == 19
    expected_thresholds = [round(0.05 * i, 2) for i in range(1, 20)]
    actual_thresholds = [row.threshold for row in sweep]
    assert actual_thresholds == pytest.approx(expected_thresholds)


def test_threshold_sweep_custom() -> None:
    rows = [
        make_row(turn=1, true_drifted=False, pred_score=0.1),
        make_row(
            turn=2,
            true_drifted=True,
            true_severity="medium",
            true_drift_types=["goal"],
            pred_score=0.7,
        ),
    ]
    sweep = compute_threshold_sweep(rows, thresholds=[0.5])
    assert len(sweep) == 1
    assert sweep[0].threshold == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Per-drift-type recall
# ---------------------------------------------------------------------------


def test_per_drift_type_recall_isolation() -> None:
    rows = [
        # Pure-goal drift, detected.
        make_row(
            turn=1,
            true_drifted=True,
            true_severity="medium",
            true_drift_types=["goal"],
            pred_score=0.5,
        ),
        # Mixed goal+constraint drift — must NOT count toward any single-type recall.
        make_row(
            turn=2,
            true_drifted=True,
            true_severity="high",
            true_drift_types=["goal", "constraint"],
            true_violated_constraints=[0],
            pred_score=0.9,
        ),
    ]
    recall = compute_per_drift_type_recall(rows, threshold=0.3)
    assert recall["goal"] == pytest.approx(1.0)
    assert math.isnan(recall["constraint"])
    assert math.isnan(recall["consistency"])


# ---------------------------------------------------------------------------
# Full report smoke
# ---------------------------------------------------------------------------


def test_compute_full_report_smoke() -> None:
    rows: List[EvalRow] = []
    # Build 10 mixed rows across two conversations.
    for i in range(5):
        rows.append(
            make_row(
                conversation_id="conv_A",
                turn=i + 1,
                drift_onset_turn=3,
                true_drifted=(i + 1) >= 3,
                true_severity="medium" if (i + 1) >= 3 else "none",
                true_drift_types=["goal"] if (i + 1) >= 3 else [],
                pred_score=0.1 if (i + 1) < 3 else 0.7,
                pred_severity="medium" if (i + 1) >= 3 else "none",
            )
        )
    for i in range(5):
        rows.append(
            make_row(
                conversation_id="conv_B",
                turn=i + 1,
                drift_onset_turn=None,
                true_drifted=False,
                true_severity="none",
                pred_score=0.05,
                pred_severity="none",
            )
        )
    report = compute_full_report(rows, detector_threshold=0.3)
    assert isinstance(report, EvalReport)
    assert report.n_turns == 10
    assert report.n_conversations == 2
    # 3/10 turns are drifted.
    assert report.drift_rate == pytest.approx(0.3)
    assert report.binary is not None
    assert report.calibration is not None
    assert report.time_to_detection is not None
    assert report.severity_confusion is not None
    assert len(report.signal_correlations) == 4
    assert len(report.threshold_sweep) == 19
    assert set(report.per_drift_type_recall.keys()) == {
        "goal",
        "constraint",
        "consistency",
    }


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


def test_empty_rows_raises() -> None:
    empty: List[EvalRow] = []
    with pytest.raises(ValueError):
        compute_binary_metrics(empty, threshold=0.3)
    with pytest.raises(ValueError):
        compute_calibration(empty)
    with pytest.raises(ValueError):
        compute_time_to_detection(empty, threshold=0.3)
    with pytest.raises(ValueError):
        compute_severity_confusion(empty)
    with pytest.raises(ValueError):
        compute_signal_correlations(empty)
    with pytest.raises(ValueError):
        compute_threshold_sweep(empty)
    with pytest.raises(ValueError):
        compute_per_drift_type_recall(empty, threshold=0.3)
    with pytest.raises(ValueError):
        compute_full_report(empty, detector_threshold=0.3)
