# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stable-Agent Contributors
"""CLI entry point: generate a synthetic labeled drift-detection dataset.

Drives ``drift_evaluator.generator.generate_dataset`` against a local
Ollama server to produce a labeled corpus of conversations with known
drift onsets. Intended to be invoked from the command line; the dataset
output directory defaults to ``Drift-Evaluator/datasets/v1_synthetic``
relative to the project root, regardless of the caller's cwd.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from drift_evaluator.generator import GeneratorConfig, generate_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "datasets" / "v1_synthetic"

OLLAMA_HELP = (
    "Ollama not available at http://localhost:11434. "
    "Install: brew install ollama. "
    "Start: ollama serve. "
    "Pull model: ollama pull llama3.1:8b"
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_dataset.py",
        description=(
            "Generate a synthetic labeled drift-detection dataset using "
            "local Ollama."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="output directory (default: datasets/v1_synthetic)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=200,
        help="number of conversations (default: 200)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.1:8b",
        help="Ollama model name (default: llama3.1:8b)",
    )
    parser.add_argument(
        "--turns-min",
        type=int,
        default=6,
        dest="turns_min",
        metavar="N",
        help="min turns per conversation (default: 6)",
    )
    parser.add_argument(
        "--turns-max",
        type=int,
        default=12,
        dest="turns_max",
        metavar="N",
        help="max turns per conversation (default: 12)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="S",
        help="base random seed (default: 42)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="suppress per-conversation progress output",
    )
    return parser


def _warn_if_nonempty(out_dir: Path) -> None:
    if out_dir.exists() and out_dir.is_dir():
        try:
            has_entries = any(out_dir.iterdir())
        except OSError:
            has_entries = False
        if has_entries:
            print(
                "Output directory already contains files; new files will "
                "be added/overwritten.",
                file=sys.stderr,
            )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    out_dir: Path = args.out.resolve()

    _warn_if_nonempty(out_dir)

    config = GeneratorConfig(
        model=args.model,
        n_conversations=args.n,
        turns_per_conv_min=args.turns_min,
        turns_per_conv_max=args.turns_max,
        drift_onset_min=3,
        base_seed=args.seed,
    )

    try:
        result = generate_dataset(
            config,
            out_dir,
            progress=not args.no_progress,
        )
    except RuntimeError as exc:
        message = str(exc).lower()
        if "ollama" in message or "11434" in message or "not healthy" in message:
            print(OLLAMA_HELP, file=sys.stderr)
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    count = _result_count(result, out_dir)

    print(str(out_dir))
    print(f"conversations: {count}")
    return 0


def _result_count(result: object, out_dir: Path) -> int:
    """Best-effort extraction of the conversation count from the return value.

    The library may return an int, a list, or an object with a length;
    fall back to counting JSONL/JSON conversation files in ``out_dir``.
    """
    if isinstance(result, int):
        return result
    if hasattr(result, "__len__"):
        try:
            return len(result)  # type: ignore[arg-type]
        except TypeError:
            pass
    try:
        return sum(
            1
            for p in out_dir.iterdir()
            if p.is_file() and p.suffix in {".json", ".jsonl"}
        )
    except OSError:
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(1)
