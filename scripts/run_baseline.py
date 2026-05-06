# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stable-Agent Contributors
"""CLI entry point: run the current Drift-Detector against a labeled dataset.

Loads a labeled dataset, runs the default Drift-Detector configuration
through ``drift_evaluator.harness.evaluate_dataset``, and writes a
markdown + JSON report via ``drift_evaluator.report.write_report``.
Default paths are resolved relative to the project root, not the
caller's cwd, so the script behaves identically regardless of where it
is invoked from.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from drift_evaluator.harness import evaluate_dataset, make_default_detector_factory
from drift_evaluator.report import write_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "datasets" / "v1_synthetic"
DEFAULT_OUT = PROJECT_ROOT / "reports"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_baseline.py",
        description=(
            "Run the current Drift-Detector against a labeled dataset, "
            "write report."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="path to dataset dir (default: datasets/v1_synthetic)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="output directory for report (default: reports/)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.30,
        metavar="T",
        help="drift threshold for the detector (default: 0.30)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="baseline_v0",
        help="report filename stem (default: baseline_v0)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="suppress progress output",
    )
    return parser


def _format_metric(value: object) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"


def _get(report: object, *names: str) -> object:
    """Return the first attribute/key found on ``report`` from ``names``."""
    for name in names:
        if hasattr(report, name):
            val = getattr(report, name)
            if val is not None:
                return val
        if isinstance(report, dict) and name in report:
            val = report[name]
            if val is not None:
                return val
    return None


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    dataset_path: Path = args.dataset.resolve()
    out_path: Path = args.out.resolve()

    if not dataset_path.exists():
        print(
            f"Error: dataset directory not found: {dataset_path}",
            file=sys.stderr,
        )
        return 1

    factory = make_default_detector_factory(drift_threshold=args.threshold)

    try:
        rows, report = evaluate_dataset(
            args.dataset,
            factory,
            detector_threshold=args.threshold,
            progress=not args.no_progress,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    notes = (
        f"Threshold: {args.threshold}\n"
        f"Detector: drift-detector v0 (default weights)\n"
        f"Dataset: {args.dataset}"
    )

    written = write_report(
        report,
        args.out,
        name=args.name,
        title="Drift Detector Baseline Evaluation",
        dataset_name=Path(args.dataset).name,
        detector_name="drift-detector v0 (default config)",
        extra_notes=notes,
    )

    md_path, json_path = _resolve_report_paths(written, out_path, args.name)

    precision = _format_metric(report.binary.precision)
    recall = _format_metric(report.binary.recall)
    f1 = _format_metric(report.binary.f1)
    roc_auc = _format_metric(report.calibration.roc_auc)
    pr_auc = _format_metric(report.calibration.pr_auc)
    brier = _format_metric(report.calibration.brier_score)

    print(f"Wrote: {md_path}")
    print(f"Wrote: {json_path}")
    print(
        f"Headline (threshold={args.threshold:.2f}): "
        f"precision={precision} recall={recall} f1={f1}"
    )
    print(
        f"Calibration: ROC-AUC={roc_auc} PR-AUC={pr_auc} Brier={brier}"
    )
    return 0


def _resolve_report_paths(
    written: object,
    out_dir: Path,
    name: str,
) -> tuple[Path, Path]:
    """Extract md/json absolute paths from the ``write_report`` return value.

    Falls back to constructing the conventional paths from ``out_dir`` and
    ``name`` if the return value does not carry explicit paths.
    """
    md_path: Path | None = None
    json_path: Path | None = None

    if isinstance(written, dict):
        for key in ("md", "markdown", "md_path", "markdown_path"):
            if key in written and written[key]:
                md_path = Path(written[key]).resolve()
                break
        for key in ("json", "json_path"):
            if key in written and written[key]:
                json_path = Path(written[key]).resolve()
                break
    elif isinstance(written, (tuple, list)) and len(written) >= 2:
        md_path = Path(written[0]).resolve()
        json_path = Path(written[1]).resolve()
    elif written is not None:
        for md_attr, json_attr in (
            ("md_path", "json_path"),
            ("markdown_path", "json_path"),
            ("md", "json"),
        ):
            if hasattr(written, md_attr) and hasattr(written, json_attr):
                md_path = Path(getattr(written, md_attr)).resolve()
                json_path = Path(getattr(written, json_attr)).resolve()
                break

    if md_path is None:
        md_path = (out_dir / f"{name}.md").resolve()
    if json_path is None:
        json_path = (out_dir / f"{name}.json").resolve()

    return md_path, json_path


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(1)
