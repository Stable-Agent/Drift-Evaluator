# Copyright 2026 Stable-Agent Contributors
# SPDX-License-Identifier: MIT
"""Unit tests for ``drift_evaluator.harness``.

These tests use a hand-rolled :class:`FakeDetector` rather than the real
``drift_detector.DriftDetector`` so the suite stays fast (no model load,
no LLM call). The harness only duck-types the detector via ``setup``,
``reset``, and ``check_response``, so the fake is sufficient.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import List

import pytest

from drift_evaluator.harness import evaluate_conversation, evaluate_dataset
from drift_evaluator.metrics import EvalReport, EvalRow
from drift_evaluator.schema import (
    Conversation,
    Turn,
    TurnLabel,
    save_conversation,
)


# ---------------------------------------------------------------------------
# FakeDetector
# ---------------------------------------------------------------------------


class FakeDetector:
    """Minimal duck-typed stand-in for ``drift_detector.DriftDetector``.

    Returns a deterministic score: 0.1 for turns < 3, 0.7 otherwise. The
    severity bucket and drift flag are derived from that score so the
    harness's downstream consumers see a coherent :class:`DriftResult`.
    """

    def __init__(self) -> None:
        self.history: List[str] = []
        self.goal: str | None = None
        self.constraints: List[str] | None = None

    def setup(self, goal: str, constraints: List[str]) -> None:
        self.goal = goal
        self.constraints = list(constraints)
        self.history = []

    def reset(self) -> None:
        self.history = []

    def check_response(self, response: str, turn: int) -> SimpleNamespace:
        score = 0.1 if turn < 3 else 0.7
        if score < 0.15:
            severity = "none"
        elif score < 0.3:
            severity = "low"
        elif score < 0.5:
            severity = "medium"
        else:
            severity = "high"
        self.history.append(response)
        return SimpleNamespace(
            turn=turn,
            response=response,
            goal_drift=score,
            constraint_drift=0.0,
            consistency_drift=0.0,
            total_drift=score,
            is_drift=score >= 0.3,
            severity=severity,
            violated_constraints=[],
        )


# ---------------------------------------------------------------------------
# Conversation builders
# ---------------------------------------------------------------------------


def _label(
    *, drifted: bool, severity: str, drift_types: List[str] | None = None,
    violated: List[int] | None = None,
) -> TurnLabel:
    return TurnLabel(
        drifted=drifted,
        drift_types=list(drift_types or []),
        severity=severity,
        violated_constraint_indices=list(violated or []),
    )


def _make_5turn_conv(conv_id: str = "conv_0001") -> Conversation:
    """Build a 5-turn conversation whose label transitions at turn 3."""
    turns: List[Turn] = []
    for i in range(1, 6):
        if i < 3:
            label = _label(drifted=False, severity="none")
        else:
            label = _label(
                drifted=True, severity="medium", drift_types=["goal"]
            )
        turns.append(
            Turn(
                turn=i,
                user=f"user message {i}",
                assistant=f"assistant message {i}",
                label=label,
            )
        )
    return Conversation(
        id=conv_id,
        goal="Help the user plan a vegetarian dinner.",
        constraints=[
            "Do not suggest meat or fish.",
            "Keep total prep time under 45 minutes.",
        ],
        turns=turns,
        metadata={"drift_type": "goal", "drift_onset_turn": 3},
    )


# ---------------------------------------------------------------------------
# evaluate_conversation
# ---------------------------------------------------------------------------


def test_evaluate_conversation_basic() -> None:
    conv = _make_5turn_conv("conv_basic")
    detector = FakeDetector()
    detector.setup(conv.goal, conv.constraints)
    rows = evaluate_conversation(conv, detector)

    assert len(rows) == 5
    for i, row in enumerate(rows, start=1):
        assert isinstance(row, EvalRow)
        assert row.conversation_id == "conv_basic"
        assert row.turn == i
        # FakeDetector: 0.1 below turn 3, else 0.7.
        expected_score = 0.1 if i < 3 else 0.7
        assert row.pred_score == pytest.approx(expected_score)
        assert row.pred_drifted == (expected_score >= 0.3)


def test_evaluate_conversation_label_passthrough() -> None:
    # Build a conversation whose labels carry distinctive markers so we can
    # confirm the harness propagates them verbatim.
    turns = [
        Turn(
            turn=1,
            user="user 1",
            assistant="assistant 1",
            label=_label(drifted=False, severity="none"),
        ),
        Turn(
            turn=2,
            user="user 2",
            assistant="assistant 2",
            label=_label(
                drifted=True,
                severity="high",
                drift_types=["constraint"],
                violated=[0],
            ),
        ),
    ]
    conv = Conversation(
        id="conv_passthrough",
        goal="Provide accurate factual answers.",
        constraints=[
            "Cite at least one source per claim.",
            "Do not speculate beyond available evidence.",
        ],
        turns=turns,
        metadata={"drift_type": "constraint", "drift_onset_turn": 2},
    )
    detector = FakeDetector()
    detector.setup(conv.goal, conv.constraints)
    rows = evaluate_conversation(conv, detector)

    assert len(rows) == 2

    assert rows[0].true_drifted is False
    assert rows[0].true_severity == "none"
    assert rows[0].true_drift_types == []
    assert rows[0].true_violated_constraints == []

    assert rows[1].true_drifted is True
    assert rows[1].true_severity == "high"
    assert rows[1].true_drift_types == ["constraint"]
    assert rows[1].true_violated_constraints == [0]


# ---------------------------------------------------------------------------
# evaluate_dataset
# ---------------------------------------------------------------------------


def _factory_fresh_each_call() -> FakeDetector:
    """A factory that returns a brand-new FakeDetector each time.

    Avoids loading the real SentenceTransformer-based detector.
    """
    return FakeDetector()


def test_evaluate_dataset_uses_factory_per_conversation(tmp_path: Path) -> None:
    conv_a = _make_5turn_conv("conv_0001")
    conv_b = _make_5turn_conv("conv_0002")
    save_conversation(conv_a, tmp_path / "conv_0001.json")
    save_conversation(conv_b, tmp_path / "conv_0002.json")

    rows, report = evaluate_dataset(
        tmp_path,
        _factory_fresh_each_call,
        detector_threshold=0.3,
        progress=False,
    )
    assert isinstance(rows, list) and len(rows) > 0
    assert isinstance(report, EvalReport)
    # 2 conversations × 5 turns each.
    assert report.n_conversations == 2
    assert report.n_turns == 10


def test_evaluate_dataset_handles_empty_turns(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # One real conversation + one with empty turns. The empty-turn one
    # should be skipped with a stderr warning.
    conv_real = _make_5turn_conv("conv_0001")
    save_conversation(conv_real, tmp_path / "conv_0001.json")

    conv_empty = Conversation(
        id="conv_0002",
        goal="placeholder goal",
        constraints=["c1"],
        turns=[],
        metadata={"drift_type": "clean"},
    )
    save_conversation(conv_empty, tmp_path / "conv_0002.json")

    rows, report = evaluate_dataset(
        tmp_path,
        _factory_fresh_each_call,
        detector_threshold=0.3,
        progress=False,
    )
    captured = capsys.readouterr()
    assert "conv_0002" in captured.err
    assert "no turns" in captured.err.lower() or "skip" in captured.err.lower()
    # Only conv_0001's 5 turns should be present.
    assert all(r.conversation_id == "conv_0001" for r in rows)
    assert report.n_conversations == 1
    assert report.n_turns == 5


def test_evaluate_dataset_raises_on_no_rows(tmp_path: Path) -> None:
    conv_empty = Conversation(
        id="conv_0001",
        goal="placeholder goal",
        constraints=["c1"],
        turns=[],
        metadata={"drift_type": "clean"},
    )
    save_conversation(conv_empty, tmp_path / "conv_0001.json")

    with pytest.raises(RuntimeError):
        evaluate_dataset(
            tmp_path,
            _factory_fresh_each_call,
            detector_threshold=0.3,
            progress=False,
        )
