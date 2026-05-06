"""
Drift-Evaluator schema: labeled-conversation dataset format.

Copyright (c) 2026 Stable-Agent Contributors
Licensed under MIT License

This module defines the canonical data model for the drift-detection
evaluation framework. A dataset consists of multiple ``Conversation``
objects, each containing an explicit ``goal``, a list of ``constraints``,
and a sequence of ``Turn`` objects. Every turn carries a ``TurnLabel``
that indicates whether the assistant's response drifted from the original
goal/constraints, what kind of drift occurred, and the severity.

Drift taxonomy
--------------
Drift types (subset of):
    - ``"goal"``        : assistant pursued an objective other than the stated goal.
    - ``"constraint"``  : assistant violated one or more declared constraints.
    - ``"consistency"`` : assistant contradicted earlier turns or its own prior claims.

Severity levels:
    - ``"none"``   : no drift.
    - ``"low"``    : minor / partial drift.
    - ``"medium"`` : substantive drift but recoverable.
    - ``"high"``   : flagrant drift; goal/constraints clearly abandoned.

On-disk format
--------------
Each conversation is serialized as a single JSON file named ``conv_<id>.json``
within a dataset directory. A ``manifest.json`` summarizes the dataset
contents and is produced by ``save_manifest``.

This module is pure library code (stdlib only): ``dataclasses``, ``json``,
``datetime``, ``pathlib``, ``typing``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

__all__ = [
    "ALLOWED_DRIFT_TYPES",
    "ALLOWED_SEVERITIES",
    "MANIFEST_VERSION",
    "TurnLabel",
    "Turn",
    "Conversation",
    "conversation_to_dict",
    "conversation_from_dict",
    "save_conversation",
    "load_conversation",
    "load_dataset",
    "save_manifest",
    "validate_conversation",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_DRIFT_TYPES: frozenset[str] = frozenset({"goal", "constraint", "consistency"})
ALLOWED_SEVERITIES: frozenset[str] = frozenset({"none", "low", "medium", "high"})
MANIFEST_VERSION: str = "v1"

PathLike = Union[Path, str]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TurnLabel:
    """Per-turn drift annotation."""

    drifted: bool
    drift_types: List[str]
    severity: str
    violated_constraint_indices: List[int]

    def __post_init__(self) -> None:
        _validate_turn_label(self)


@dataclass
class Turn:
    """A single (user, assistant) exchange with its drift label."""

    turn: int
    user: str
    assistant: str
    label: TurnLabel

    def __post_init__(self) -> None:
        if not isinstance(self.turn, int) or isinstance(self.turn, bool):
            raise ValueError(
                f"Turn.turn must be an int, got {type(self.turn).__name__}"
            )
        if self.turn < 1:
            raise ValueError(
                f"Turn.turn must be 1-indexed (>= 1), got {self.turn}"
            )
        if not isinstance(self.user, str):
            raise ValueError("Turn.user must be a string")
        if not isinstance(self.assistant, str):
            raise ValueError("Turn.assistant must be a string")
        if not isinstance(self.label, TurnLabel):
            raise ValueError("Turn.label must be a TurnLabel instance")


@dataclass
class Conversation:
    """A labeled conversation within a drift-evaluation dataset."""

    id: str
    goal: str
    constraints: List[str]
    turns: List[Turn]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("Conversation.id must be a non-empty string")
        if not isinstance(self.goal, str):
            raise ValueError("Conversation.goal must be a string")
        if not isinstance(self.constraints, list) or not all(
            isinstance(c, str) for c in self.constraints
        ):
            raise ValueError("Conversation.constraints must be a list of strings")
        if not isinstance(self.turns, list) or not all(
            isinstance(t, Turn) for t in self.turns
        ):
            raise ValueError("Conversation.turns must be a list of Turn instances")
        if not isinstance(self.metadata, dict):
            raise ValueError("Conversation.metadata must be a dict")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_turn_label(label: TurnLabel) -> None:
    """Validate a TurnLabel in isolation (no cross-conversation checks)."""
    if not isinstance(label.drifted, bool):
        raise ValueError("TurnLabel.drifted must be a bool")

    if not isinstance(label.drift_types, list) or not all(
        isinstance(d, str) for d in label.drift_types
    ):
        raise ValueError("TurnLabel.drift_types must be a list of strings")

    bad_types = [d for d in label.drift_types if d not in ALLOWED_DRIFT_TYPES]
    if bad_types:
        raise ValueError(
            f"TurnLabel.drift_types contains invalid entries {bad_types!r}; "
            f"allowed: {sorted(ALLOWED_DRIFT_TYPES)}"
        )

    if len(set(label.drift_types)) != len(label.drift_types):
        raise ValueError(
            f"TurnLabel.drift_types must not contain duplicates: {label.drift_types!r}"
        )

    if not isinstance(label.severity, str):
        raise ValueError("TurnLabel.severity must be a string")
    if label.severity not in ALLOWED_SEVERITIES:
        raise ValueError(
            f"TurnLabel.severity {label.severity!r} not in "
            f"{sorted(ALLOWED_SEVERITIES)}"
        )

    if not isinstance(label.violated_constraint_indices, list) or not all(
        isinstance(i, int) and not isinstance(i, bool)
        for i in label.violated_constraint_indices
    ):
        raise ValueError(
            "TurnLabel.violated_constraint_indices must be a list of ints"
        )

    if not label.drifted:
        if label.drift_types:
            raise ValueError(
                "TurnLabel: when drifted is False, drift_types must be empty; "
                f"got {label.drift_types!r}"
            )
        if label.severity != "none":
            raise ValueError(
                "TurnLabel: when drifted is False, severity must be 'none'; "
                f"got {label.severity!r}"
            )
    else:
        if label.severity == "none":
            raise ValueError(
                "TurnLabel: when drifted is True, severity must not be 'none'"
            )


def validate_conversation(conv: Conversation) -> None:
    """Run all cross-field validation for a Conversation. Raises ValueError on issue."""
    if not isinstance(conv, Conversation):
        raise ValueError(
            f"validate_conversation expected Conversation, got {type(conv).__name__}"
        )

    # Re-validate atomic fields (in case the object was mutated post-construction).
    if not isinstance(conv.id, str) or not conv.id:
        raise ValueError("Conversation.id must be a non-empty string")
    if not isinstance(conv.goal, str):
        raise ValueError("Conversation.goal must be a string")
    if not isinstance(conv.constraints, list) or not all(
        isinstance(c, str) for c in conv.constraints
    ):
        raise ValueError("Conversation.constraints must be a list of strings")
    if not isinstance(conv.metadata, dict):
        raise ValueError("Conversation.metadata must be a dict")

    if not isinstance(conv.turns, list) or not all(
        isinstance(t, Turn) for t in conv.turns
    ):
        raise ValueError("Conversation.turns must be a list of Turn instances")

    n_constraints = len(conv.constraints)

    # Turn-by-turn validation: 1-indexed contiguous, label cross-checks.
    for expected_idx, turn in enumerate(conv.turns, start=1):
        if turn.turn != expected_idx:
            raise ValueError(
                f"Conversation {conv.id!r}: turns must be 1-indexed and "
                f"contiguous; expected turn {expected_idx}, got {turn.turn}"
            )

        _validate_turn_label(turn.label)

        for ci in turn.label.violated_constraint_indices:
            if ci < 0 or ci >= n_constraints:
                raise ValueError(
                    f"Conversation {conv.id!r} turn {turn.turn}: "
                    f"violated_constraint_indices contains out-of-range index "
                    f"{ci} (valid range: 0..{n_constraints - 1 if n_constraints else -1})"
                )

        if len(set(turn.label.violated_constraint_indices)) != len(
            turn.label.violated_constraint_indices
        ):
            raise ValueError(
                f"Conversation {conv.id!r} turn {turn.turn}: "
                f"violated_constraint_indices must not contain duplicates"
            )

        if (
            turn.label.violated_constraint_indices
            and "constraint" not in turn.label.drift_types
        ):
            raise ValueError(
                f"Conversation {conv.id!r} turn {turn.turn}: "
                f"violated_constraint_indices is non-empty but 'constraint' "
                f"is not in drift_types {turn.label.drift_types!r}"
            )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def conversation_to_dict(conv: Conversation) -> Dict[str, Any]:
    """Serialize a Conversation to a plain JSON-compatible dict."""
    validate_conversation(conv)
    return asdict(conv)


def conversation_from_dict(d: Dict[str, Any]) -> Conversation:
    """Construct a Conversation from a JSON-compatible dict, validating shape."""
    if not isinstance(d, dict):
        raise ValueError(
            f"conversation_from_dict expected dict, got {type(d).__name__}"
        )

    required_keys = {"id", "goal", "constraints", "turns"}
    missing = required_keys - d.keys()
    if missing:
        raise ValueError(
            f"conversation_from_dict missing required keys: {sorted(missing)}"
        )

    raw_turns = d["turns"]
    if not isinstance(raw_turns, list):
        raise ValueError("Conversation 'turns' must be a list")

    turns: List[Turn] = []
    for i, raw_turn in enumerate(raw_turns):
        if not isinstance(raw_turn, dict):
            raise ValueError(f"Turn at index {i} must be a dict")
        turn_required = {"turn", "user", "assistant", "label"}
        turn_missing = turn_required - raw_turn.keys()
        if turn_missing:
            raise ValueError(
                f"Turn at index {i} missing required keys: {sorted(turn_missing)}"
            )

        raw_label = raw_turn["label"]
        if not isinstance(raw_label, dict):
            raise ValueError(f"Turn at index {i}: 'label' must be a dict")
        label_required = {
            "drifted",
            "drift_types",
            "severity",
            "violated_constraint_indices",
        }
        label_missing = label_required - raw_label.keys()
        if label_missing:
            raise ValueError(
                f"Turn at index {i} label missing required keys: "
                f"{sorted(label_missing)}"
            )

        label = TurnLabel(
            drifted=raw_label["drifted"],
            drift_types=list(raw_label["drift_types"]),
            severity=raw_label["severity"],
            violated_constraint_indices=list(
                raw_label["violated_constraint_indices"]
            ),
        )

        turns.append(
            Turn(
                turn=raw_turn["turn"],
                user=raw_turn["user"],
                assistant=raw_turn["assistant"],
                label=label,
            )
        )

    metadata = d.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Conversation 'metadata' must be a dict if present")

    conv = Conversation(
        id=d["id"],
        goal=d["goal"],
        constraints=list(d["constraints"]),
        turns=turns,
        metadata=dict(metadata),
    )
    validate_conversation(conv)
    return conv


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def _coerce_path(path: PathLike) -> Path:
    if isinstance(path, Path):
        return path
    if isinstance(path, str):
        return Path(path)
    raise ValueError(
        f"Path argument must be Path or str, got {type(path).__name__}"
    )


def save_conversation(conv: Conversation, path: PathLike) -> None:
    """Validate and write a Conversation to ``path`` as JSON (utf-8, indent=2)."""
    validate_conversation(conv)
    p = _coerce_path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    payload = conversation_to_dict(conv)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=False)
        f.write("\n")


def load_conversation(path: PathLike) -> Conversation:
    """Read a Conversation from a JSON file at ``path``, validating its contents."""
    p = _coerce_path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return conversation_from_dict(data)


def load_dataset(directory: PathLike) -> List[Conversation]:
    """
    Load all ``conv_*.json`` files in ``directory`` and return them sorted by id.

    Raises ValueError if ``directory`` does not exist or is not a directory.
    """
    d = _coerce_path(directory)
    if not d.exists():
        raise ValueError(f"Dataset directory does not exist: {d}")
    if not d.is_dir():
        raise ValueError(f"Dataset path is not a directory: {d}")

    paths = sorted(d.glob("conv_*.json"))
    conversations: List[Conversation] = []
    for path in paths:
        conversations.append(load_conversation(path))

    conversations.sort(key=lambda c: c.id)
    return conversations


def save_manifest(
    directory: PathLike, conversations: List[Conversation]
) -> None:
    """
    Write a ``manifest.json`` summarizing ``conversations`` into ``directory``.

    Manifest schema::

        {
          "version": "v1",
          "count": <int>,
          "generated_at": <ISO8601 UTC>,
          "conversations": [
            {
              "id": <str>,
              "drift_type": <metadata['drift_type'] or null>,
              "drift_onset_turn": <metadata['drift_onset_turn'] or null>,
              "n_turns": <int>
            },
            ...
          ]
        }
    """
    d = _coerce_path(directory)
    if not d.exists():
        d.mkdir(parents=True, exist_ok=True)
    elif not d.is_dir():
        raise ValueError(f"Manifest target is not a directory: {d}")

    if not isinstance(conversations, list) or not all(
        isinstance(c, Conversation) for c in conversations
    ):
        raise ValueError("save_manifest: conversations must be a list of Conversation")

    for conv in conversations:
        validate_conversation(conv)

    entries: List[Dict[str, Any]] = []
    for conv in conversations:
        meta = conv.metadata or {}
        entries.append(
            {
                "id": conv.id,
                "drift_type": meta.get("drift_type"),
                "drift_onset_turn": meta.get("drift_onset_turn"),
                "n_turns": len(conv.turns),
            }
        )

    manifest: Dict[str, Any] = {
        "version": MANIFEST_VERSION,
        "count": len(conversations),
        "generated_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "conversations": entries,
    }

    manifest_path = d / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=False)
        f.write("\n")


# ---------------------------------------------------------------------------
# Internal: keep type-checkers happy about Iterable use without importing it
# at runtime in places where it is not needed.
# ---------------------------------------------------------------------------

_ = Iterable  # silence "imported but unused" if downstream tooling complains
