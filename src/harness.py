# Copyright 2026 Stable-Agent Contributors
# SPDX-License-Identifier: MIT
"""Drift-Evaluator harness.

Runs a :class:`drift_detector.DriftDetector` against a labeled conversation
dataset and emits one :class:`drift_evaluator.metrics.EvalRow` per turn,
pairing the recorded ground-truth label with the detector's prediction.

The collected rows are then handed to
:func:`drift_evaluator.metrics.compute_full_report` to produce an
:class:`drift_evaluator.metrics.EvalReport`.

The :class:`drift_detector.DriftDetector` constructor loads a
``SentenceTransformer`` model (~5-10 seconds), so this module never
instantiates a detector eagerly. Instead, callers obtain a
:data:`DetectorFactory` via :func:`make_default_detector_factory`, which
caches a single detector and returns the same (reset) instance on every
call. The ``drift_detector`` import itself is deferred until the factory
is built, so importing this module is cheap.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Tuple, Union

from drift_evaluator.metrics import EvalReport, EvalRow, compute_full_report
from drift_evaluator.schema import Conversation, Turn, load_dataset

if TYPE_CHECKING:  # pragma: no cover - typing only
    from drift_detector import DriftDetector, DriftResult


__all__ = [
    "DetectorFactory",
    "make_default_detector_factory",
    "make_v2_detector_factory",
    "make_v2_llm_judge_factory",
    "evaluate_conversation",
    "evaluate_dataset",
]


# ---------------------------------------------------------------------------
# Types & constants
# ---------------------------------------------------------------------------


DetectorFactory = Callable[[], "DriftDetector"]
PathLike = Union[Path, str]

_VALID_SEVERITIES: frozenset[str] = frozenset({"none", "low", "medium", "high"})


# ---------------------------------------------------------------------------
# Factory builder
# ---------------------------------------------------------------------------


def make_default_detector_factory(**detector_kwargs) -> DetectorFactory:
    """Build a :data:`DetectorFactory` that reuses a single detector instance.

    The returned factory caches one :class:`drift_detector.DriftDetector`
    (loading the ``SentenceTransformer`` exactly once) and, on each call,
    invokes ``reset()`` on it before returning. This clears
    ``response_history``, ``embedding_history``, and ``drift_history``
    while preserving the (expensive) embedder. The harness then calls
    ``setup(...)`` to refresh ``original_goal`` / ``original_constraints``
    / ``goal_embedding`` / ``constraint_embeddings`` for the next
    conversation.

    Conversations must be evaluated sequentially because the cached
    detector is shared.

    Parameters
    ----------
    **detector_kwargs
        Forwarded verbatim to :class:`drift_detector.DriftDetector`.

    Returns
    -------
    DetectorFactory
        Zero-argument callable returning a freshly-reset detector.
    """

    # Defer the import so that merely importing :mod:`drift_evaluator.harness`
    # does not trigger a heavyweight ``sentence_transformers`` load.
    from drift_detector import DriftDetector

    cached: List["DriftDetector"] = []

    def _factory() -> "DriftDetector":
        if not cached:
            cached.append(DriftDetector(**detector_kwargs))
        detector = cached[0]
        detector.reset()
        return detector

    return _factory


def make_v2_detector_factory(**detector_kwargs) -> DetectorFactory:
    """Build a factory that reuses a single :class:`DriftDetectorV2` instance.

    Same caching contract as :func:`make_default_detector_factory`: one
    detector is constructed (loading the SentenceTransformer embedder AND
    the NLI CrossEncoder once), then ``reset()`` is invoked between
    conversations to clear history while preserving the loaded models.
    """
    from drift_detector import DriftDetectorV2

    cached: List["DriftDetector"] = []

    def _factory() -> "DriftDetector":
        if not cached:
            cached.append(DriftDetectorV2(**detector_kwargs))
        detector = cached[0]
        detector.reset()
        return detector

    return _factory


def make_v2_llm_judge_factory(
    *,
    ollama_model: str = "llama3.1:8b",
    ollama_host: str = "http://localhost:11434",
    cache_path: "Path | str | None" = None,
    **detector_kwargs,
) -> DetectorFactory:
    """Build a factory for v2 with LLM-judge constraint mode.

    Constructs a single OllamaClient + DriftDetectorV2 with
    ``constraint_mode="llm_judge"``. The cache_path is forwarded to the
    LLMJudgeConstraintSignal so cached scores survive across runs.
    """
    from drift_detector import DriftDetectorV2
    from drift_evaluator.ollama_client import OllamaClient, OllamaConfig

    client = OllamaClient(OllamaConfig(model=ollama_model, host=ollama_host))
    if cache_path is not None and not isinstance(cache_path, Path):
        cache_path = Path(cache_path)

    detector_kwargs.setdefault("constraint_mode", "llm_judge")
    detector_kwargs.setdefault("llm_client", client)
    detector_kwargs.setdefault("llm_model", ollama_model)
    detector_kwargs.setdefault("llm_cache_path", cache_path)

    cached: List["DriftDetector"] = []

    def _factory() -> "DriftDetector":
        if not cached:
            cached.append(DriftDetectorV2(**detector_kwargs))
        detector = cached[0]
        detector.reset()
        return detector

    return _factory


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _coerce_path(path: PathLike) -> Path:
    if isinstance(path, Path):
        return path
    if isinstance(path, str):
        return Path(path)
    raise TypeError(
        f"dataset_dir must be Path or str, got {type(path).__name__}"
    )


def _normalize_severity(severity: object) -> str:
    """Map a detector-emitted severity onto the 4-bucket label space.

    The detector is expected to emit one of
    ``{"none", "low", "medium", "high"}``; anything else (unexpected
    string, ``None``, ...) is conservatively bucketed as ``"high"`` so
    spurious values do not silently appear as a benign category.
    """
    if isinstance(severity, str) and severity in _VALID_SEVERITIES:
        return severity
    return "high"


def _drift_onset_turn_for(conv: Conversation) -> "int | None":
    """Read the labeled drift-onset turn from ``conv.metadata``.

    Returns ``None`` for clean conversations (``drift_type == "clean"``)
    and when the metadata key is missing or non-integer.
    """
    meta = conv.metadata or {}
    if meta.get("drift_type") == "clean":
        return None
    onset = meta.get("drift_onset_turn")
    if isinstance(onset, bool) or not isinstance(onset, int):
        return None
    return onset


def _row_from_turn(
    conv: Conversation,
    turn: Turn,
    drift_onset_turn: "int | None",
    result: "DriftResult",
) -> EvalRow:
    """Pair a labeled :class:`Turn` with a :class:`DriftResult` prediction."""
    label = turn.label
    return EvalRow(
        conversation_id=conv.id,
        turn=turn.turn,
        drift_onset_turn=drift_onset_turn,
        # Ground truth from the dataset label.
        true_drifted=bool(label.drifted),
        true_severity=label.severity,
        true_drift_types=list(label.drift_types),
        true_violated_constraints=list(label.violated_constraint_indices),
        # Prediction from the detector.
        pred_score=float(result.total_drift),
        pred_drifted=bool(result.is_drift),
        pred_severity=_normalize_severity(result.severity),
        pred_violated_constraints=list(result.violated_constraints),
        # Per-signal scores.
        pred_goal_drift=float(result.goal_drift),
        pred_constraint_drift=float(result.constraint_drift),
        pred_consistency_drift=float(result.consistency_drift),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_conversation(
    conv: Conversation,
    detector: "DriftDetector",
) -> List[EvalRow]:
    """Run ``detector`` over every turn of ``conv`` and emit :class:`EvalRow` s.

    Assumes the caller has already invoked ``detector.setup(conv.goal,
    conv.constraints)`` and reset its history.

    If a turn raises during ``check_response``, the conversation is
    aborted: the error is logged to ``stderr``, partial rows for the
    aborted conversation are discarded, and an empty list is returned.
    No fake :class:`EvalRow` is fabricated.
    """
    drift_onset_turn = _drift_onset_turn_for(conv)
    rows: List[EvalRow] = []

    for turn in conv.turns:
        try:
            result = detector.check_response(turn.assistant, turn.turn)
        except Exception as exc:  # pragma: no cover - defensive
            sys.stderr.write(
                f"[harness] conversation {conv.id!r} turn {turn.turn}: "
                f"detector.check_response raised {type(exc).__name__}: {exc}; "
                f"aborting conversation\n"
            )
            return []

        rows.append(_row_from_turn(conv, turn, drift_onset_turn, result))

    return rows


def evaluate_dataset(
    dataset_dir: PathLike,
    detector_factory: DetectorFactory,
    *,
    detector_threshold: float = 0.30,
    progress: bool = True,
) -> Tuple[List[EvalRow], EvalReport]:
    """Run the detector over every conversation in ``dataset_dir``.

    For each :class:`Conversation`:

    1. Pull a fresh detector via ``detector_factory()`` (which already
       reset its per-conversation history).
    2. ``detector.setup(conv.goal, conv.constraints)`` to refresh the
       goal/constraint embeddings.
    3. Delegate to :func:`evaluate_conversation` to emit one
       :class:`EvalRow` per turn.

    Conversations whose turn list is empty are skipped with a stderr
    warning. Conversations whose detector raises mid-run are aborted
    (no rows emitted) and the error is logged to ``stderr``.

    After every conversation has been processed, the accumulated rows
    are passed to :func:`compute_full_report` together with
    ``detector_threshold``. Raises :class:`RuntimeError` if no turns
    were successfully evaluated.

    Parameters
    ----------
    dataset_dir
        Directory containing ``conv_*.json`` files.
    detector_factory
        Zero-argument callable returning a ready-to-use detector.
    detector_threshold
        Threshold used when computing binary metrics in the final report.
    progress
        If ``True``, write one progress line per conversation to
        ``stderr`` of the form ``"[i/N] {conv.id}: drift_rate={x:.2f}"``.

    Returns
    -------
    Tuple[List[EvalRow], EvalReport]
        The flat list of per-turn rows and the aggregate report.
    """
    directory = _coerce_path(dataset_dir)
    conversations = load_dataset(directory)
    total = len(conversations)

    rows: List[EvalRow] = []

    for idx, conv in enumerate(conversations, start=1):
        if not conv.turns:
            sys.stderr.write(
                f"[harness] conversation {conv.id!r} has no turns; skipping\n"
            )
            continue

        detector = detector_factory()
        detector.setup(conv.goal, conv.constraints)

        conv_rows = evaluate_conversation(conv, detector)
        rows.extend(conv_rows)

        if progress:
            if conv_rows:
                drift_rate = sum(
                    1 for r in conv_rows if r.true_drifted
                ) / float(len(conv_rows))
            else:
                drift_rate = 0.0
            sys.stderr.write(
                f"[{idx}/{total}] {conv.id}: drift_rate={drift_rate:.2f}\n"
            )

    if not rows:
        raise RuntimeError("No turns evaluated")

    report = compute_full_report(rows, detector_threshold)
    return rows, report
