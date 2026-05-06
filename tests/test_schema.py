# Copyright 2026 Stable-Agent Contributors
# SPDX-License-Identifier: MIT
"""Unit tests for ``drift_evaluator.schema``.

Covers dataclass validation, JSON round-trips, dataset listing, and
manifest writing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from drift_evaluator.schema import (
    Conversation,
    Turn,
    TurnLabel,
    conversation_from_dict,
    load_conversation,
    load_dataset,
    save_conversation,
    save_manifest,
    validate_conversation,
)


# ---------------------------------------------------------------------------
# Helpers
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


def _make_conversation(conv_id: str = "conv_0001") -> Conversation:
    turns = [
        Turn(
            turn=1,
            user="hi there",
            assistant="hello!",
            label=_label(drifted=False, severity="none"),
        ),
        Turn(
            turn=2,
            user="continue please",
            assistant="alright, here is more.",
            label=_label(
                drifted=True,
                severity="medium",
                drift_types=["goal"],
            ),
        ),
    ]
    return Conversation(
        id=conv_id,
        goal="Help with a math problem.",
        constraints=[
            "Show every step of the working.",
            "Use plain text only (no LaTeX).",
        ],
        turns=turns,
        metadata={"drift_type": "goal", "drift_onset_turn": 2},
    )


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_roundtrip(tmp_path: Path) -> None:
    conv = _make_conversation("conv_roundtrip")
    path = tmp_path / "conv_roundtrip.json"
    save_conversation(conv, path)
    assert path.exists()

    loaded = load_conversation(path)
    assert loaded.id == conv.id
    assert loaded.goal == conv.goal
    assert loaded.constraints == conv.constraints
    assert loaded.metadata == conv.metadata
    assert len(loaded.turns) == len(conv.turns)
    for orig, copy in zip(conv.turns, loaded.turns):
        assert copy.turn == orig.turn
        assert copy.user == orig.user
        assert copy.assistant == orig.assistant
        assert copy.label.drifted == orig.label.drifted
        assert copy.label.drift_types == orig.label.drift_types
        assert copy.label.severity == orig.label.severity
        assert (
            copy.label.violated_constraint_indices
            == orig.label.violated_constraint_indices
        )


# ---------------------------------------------------------------------------
# Validation: TurnLabel-level
# ---------------------------------------------------------------------------


def test_validate_severity() -> None:
    bad = {
        "id": "conv_bogus",
        "goal": "g",
        "constraints": [],
        "turns": [
            {
                "turn": 1,
                "user": "u",
                "assistant": "a",
                "label": {
                    "drifted": True,
                    "drift_types": ["goal"],
                    "severity": "bogus",
                    "violated_constraint_indices": [],
                },
            }
        ],
        "metadata": {},
    }
    with pytest.raises(ValueError):
        conversation_from_dict(bad)


def test_validate_drift_types() -> None:
    bad = {
        "id": "conv_bad_types",
        "goal": "g",
        "constraints": [],
        "turns": [
            {
                "turn": 1,
                "user": "u",
                "assistant": "a",
                "label": {
                    "drifted": True,
                    "drift_types": ["goal", "weird"],
                    "severity": "low",
                    "violated_constraint_indices": [],
                },
            }
        ],
        "metadata": {},
    }
    with pytest.raises(ValueError):
        conversation_from_dict(bad)


def test_validate_drifted_false_must_be_clean() -> None:
    # drifted=False but severity="low".
    with pytest.raises(ValueError):
        TurnLabel(
            drifted=False,
            drift_types=[],
            severity="low",
            violated_constraint_indices=[],
        )

    # drifted=False but drift_types non-empty.
    with pytest.raises(ValueError):
        TurnLabel(
            drifted=False,
            drift_types=["goal"],
            severity="none",
            violated_constraint_indices=[],
        )


# ---------------------------------------------------------------------------
# Validation: Conversation-level
# ---------------------------------------------------------------------------


def test_validate_constraint_indices_in_range() -> None:
    # Two constraints declared, but one turn references index 5.
    conv = Conversation(
        id="conv_oor",
        goal="g",
        constraints=["c0", "c1"],
        turns=[
            Turn(
                turn=1,
                user="u",
                assistant="a",
                label=_label(
                    drifted=True,
                    severity="high",
                    drift_types=["constraint"],
                    violated=[5],
                ),
            )
        ],
        metadata={},
    )
    with pytest.raises(ValueError):
        validate_conversation(conv)


def test_validate_turn_numbering() -> None:
    # Turns numbered [1, 3, 4] (skipping 2) → must be rejected.
    conv = Conversation(
        id="conv_skip",
        goal="g",
        constraints=[],
        turns=[
            Turn(
                turn=1,
                user="u1",
                assistant="a1",
                label=_label(drifted=False, severity="none"),
            ),
            Turn(
                turn=3,
                user="u3",
                assistant="a3",
                label=_label(drifted=False, severity="none"),
            ),
            Turn(
                turn=4,
                user="u4",
                assistant="a4",
                label=_label(drifted=False, severity="none"),
            ),
        ],
        metadata={},
    )
    with pytest.raises(ValueError):
        validate_conversation(conv)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def test_save_manifest(tmp_path: Path) -> None:
    convs = [
        _make_conversation("conv_0001"),
        _make_conversation("conv_0002"),
        _make_conversation("conv_0003"),
    ]
    for conv in convs:
        save_conversation(conv, tmp_path / f"{conv.id}.json")

    save_manifest(tmp_path, convs)
    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists()

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["count"] == 3
    assert isinstance(manifest["conversations"], list)
    assert len(manifest["conversations"]) == 3
    ids = {entry["id"] for entry in manifest["conversations"]}
    assert ids == {"conv_0001", "conv_0002", "conv_0003"}
    for entry in manifest["conversations"]:
        assert entry["n_turns"] == 2
        assert entry["drift_type"] == "goal"
        assert entry["drift_onset_turn"] == 2


# ---------------------------------------------------------------------------
# load_dataset ordering
# ---------------------------------------------------------------------------


def test_load_dataset_sorted(tmp_path: Path) -> None:
    # Write conv_0003 first, then conv_0001 — load_dataset should still
    # return them sorted by id.
    conv_3 = _make_conversation("conv_0003")
    conv_1 = _make_conversation("conv_0001")
    save_conversation(conv_3, tmp_path / "conv_0003.json")
    save_conversation(conv_1, tmp_path / "conv_0001.json")

    loaded = load_dataset(tmp_path)
    assert [c.id for c in loaded] == ["conv_0001", "conv_0003"]
