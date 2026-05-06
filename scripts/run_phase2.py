#!/usr/bin/env python
# Copyright 2026 Stable-Agent Contributors
# SPDX-License-Identifier: MIT
"""Phase 2 evaluation pipeline.

Runs:
  1. DriftDetectorV2 with default weights against the dataset.
  2. Fits weights, threshold, and severity buckets from the resulting EvalRows.
  3. Re-applies the fitted config (no re-running the detector — same per-signal scores).
  4. Loads the v0 baseline JSON report.
  5. Renders comparisons:
       - reports/v2_default.md
       - reports/v2_fitted.md
       - reports/comparison_v0_vs_v2_fitted.md
       - reports/fitted_config.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path,
                   default=PROJECT_ROOT / "datasets" / "v1_synthetic")
    p.add_argument("--reports-dir", type=Path,
                   default=PROJECT_ROOT / "reports")
    p.add_argument("--baseline", type=Path,
                   default=PROJECT_ROOT / "reports" / "baseline_v0.json",
                   help="path to v0 baseline JSON for comparison (optional; comparison skipped if missing)")
    p.add_argument("--threshold", type=float, default=0.50,
                   help="initial v2 threshold (default 0.50; will be refit)")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--rows-cache", type=Path,
                   default=PROJECT_ROOT / "reports" / "v2_default_rows.jsonl",
                   help="path to per-row cache (skip detector eval if exists)")
    p.add_argument("--force-eval", action="store_true",
                   help="ignore --rows-cache and re-run detector")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    dataset = args.dataset.resolve()
    reports_dir = args.reports_dir.resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not dataset.exists():
        print(f"Error: dataset not found: {dataset}", file=sys.stderr)
        return 1

    # Imports here so --help is fast and module-load is cheap
    from drift_evaluator.harness import evaluate_dataset, make_v2_detector_factory
    from drift_evaluator.metrics import compute_full_report
    from drift_evaluator.fit import fit_all, apply_fitted_config
    from drift_evaluator.report import write_report

    rows_cache: Path = args.rows_cache.resolve()
    if rows_cache.exists() and not args.force_eval:
        print(f"[1/4] Loading cached rows from {rows_cache} (use --force-eval to re-run)...",
              file=sys.stderr)
        rows_default = _load_rows(rows_cache)
        report_default = compute_full_report(rows_default, args.threshold)
    else:
        print("[1/4] Running v2 with default weights (loads NLI model on first call)...",
              file=sys.stderr)
        factory = make_v2_detector_factory(drift_threshold=args.threshold)
        rows_default, report_default = evaluate_dataset(
            dataset, factory,
            detector_threshold=args.threshold,
            progress=not args.no_progress,
        )
        rows_cache.parent.mkdir(parents=True, exist_ok=True)
        _save_rows(rows_default, rows_cache)
        print(f"[1/4] Cached rows to {rows_cache}", file=sys.stderr)

    print(f"[1/4] Done. v2_default: P={report_default.binary.precision:.3f} "
          f"R={report_default.binary.recall:.3f} F1={report_default.binary.f1:.3f} "
          f"ROC-AUC={report_default.calibration.roc_auc:.3f}", file=sys.stderr)

    md, _ = write_report(
        report_default, reports_dir, name="v2_default",
        title="Drift Detector v2 — Default Weights",
        dataset_name=dataset.name,
        detector_name="DriftDetectorV2 (default weights)",
        extra_notes=f"Threshold: {args.threshold}\nDataset: {dataset}",
    )
    print(f"[1/4] Wrote {md}", file=sys.stderr)

    print("[2/4] Fitting weights, threshold, severity buckets from v2 rows...",
          file=sys.stderr)
    fitted = fit_all(rows_default)
    print(f"[2/4] Fitted weights: {fitted.weights}", file=sys.stderr)
    print(f"[2/4] Fitted threshold: {fitted.threshold:.3f}", file=sys.stderr)
    print(f"[2/4] Fitted severity_thresholds: {fitted.severity_thresholds}",
          file=sys.stderr)

    fitted_path = reports_dir / "fitted_config.json"
    fitted_path.write_text(json.dumps({
        "weights": fitted.weights,
        "threshold": fitted.threshold,
        "severity_thresholds": fitted.severity_thresholds,
        "fit_metrics": fitted.fit_metrics,
    }, indent=2))
    print(f"[2/4] Wrote {fitted_path}", file=sys.stderr)

    print("[3/4] Re-scoring rows under fitted config and re-rendering report...",
          file=sys.stderr)
    rows_fitted = apply_fitted_config(rows_default, fitted)
    report_fitted = compute_full_report(rows_fitted, fitted.threshold)
    md, _ = write_report(
        report_fitted, reports_dir, name="v2_fitted",
        title="Drift Detector v2 — Fitted Weights & Threshold",
        dataset_name=dataset.name,
        detector_name="DriftDetectorV2 (fitted)",
        extra_notes=(
            f"Fitted threshold: {fitted.threshold:.3f}\n"
            f"Fitted weights: {fitted.weights}\n"
            f"Fitted severity_thresholds: {fitted.severity_thresholds}"
        ),
    )
    print(f"[3/4] v2_fitted: P={report_fitted.binary.precision:.3f} "
          f"R={report_fitted.binary.recall:.3f} F1={report_fitted.binary.f1:.3f} "
          f"ROC-AUC={report_fitted.calibration.roc_auc:.3f}", file=sys.stderr)
    print(f"[3/4] Wrote {md}", file=sys.stderr)

    if args.baseline.exists():
        print(f"[4/4] Comparing against {args.baseline}...", file=sys.stderr)
        from drift_evaluator.compare import write_comparison
        report_v0 = _report_from_json(args.baseline)
        md_path, _ = write_comparison(
            report_v0, report_fitted, reports_dir,
            name_a="v0 baseline (default config)",
            name_b="v2 fitted (NLI signals)",
            name="comparison_v0_vs_v2_fitted",
            title="Drift Detector: v0 baseline vs v2 fitted",
            notes=("v0 = original detector with embedding-cosine signals.\n"
                   "v2 = NLI-based constraint & contradiction signals + calibrated goal signal,\n"
                   "with weights/threshold/severity refit on the same dataset."),
        )
        print(f"[4/4] Wrote {md_path}", file=sys.stderr)
    else:
        print(f"[4/4] No baseline at {args.baseline}; skipping comparison.",
              file=sys.stderr)

    return 0


def _save_rows(rows, path: Path) -> None:
    """Save EvalRows to a JSONL file (one JSON object per line)."""
    from dataclasses import asdict
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(asdict(r)) + "\n")


def _load_rows(path: Path):
    from drift_evaluator.metrics import EvalRow
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rows.append(EvalRow(**d))
    return rows


def _report_from_json(path: Path):
    """Reconstruct an EvalReport from its JSON serialization (best-effort).

    The compare module only needs a subset of fields. We rebuild the dataclasses
    directly from the JSON written by report.write_report.
    """
    from drift_evaluator.metrics import (
        EvalReport, BinaryMetrics, CalibrationMetrics, TimeToDetection,
        SeverityConfusion, SignalCorrelation, ThresholdSweepRow,
    )
    import math
    data = json.loads(path.read_text())

    def _f(x):
        return float("nan") if x is None else float(x)

    overview = data.get("overview", {})
    headline = data.get("binary_metrics") or data.get("headline_metrics", {})
    cal = data.get("calibration", {})
    sev = data.get("severity_confusion", {})
    ttd = data.get("time_to_detection", {})
    sigs = data.get("signal_correlations") or data.get("per_signal", [])
    sweep = data.get("threshold_sweep", [])
    pdt = data.get("per_drift_type_recall", {})

    binary = BinaryMetrics(
        threshold=_f(headline.get("threshold")),
        precision=_f(headline.get("precision")),
        recall=_f(headline.get("recall")),
        f1=_f(headline.get("f1")),
        accuracy=_f(headline.get("accuracy")),
        true_positives=int(headline.get("true_positives", 0)),
        false_positives=int(headline.get("false_positives", 0)),
        true_negatives=int(headline.get("true_negatives", 0)),
        false_negatives=int(headline.get("false_negatives", 0)),
    )
    calibration = CalibrationMetrics(
        brier_score=_f(cal.get("brier_score")),
        roc_auc=_f(cal.get("roc_auc")),
        pr_auc=_f(cal.get("pr_auc")),
    )
    delays = ttd.get("delays") or []
    time_to_detection = TimeToDetection(
        detected_count=int(ttd.get("detected_count", 0)),
        missed_count=int(ttd.get("missed_count", 0)),
        early_count=int(ttd.get("early_count", 0)),
        delays=[int(d) for d in delays],
        median_delay=_f(ttd.get("median_delay")),
        mean_delay=_f(ttd.get("mean_delay")),
        p90_delay=_f(ttd.get("p90_delay")),
    )
    severity_confusion = SeverityConfusion(
        labels=list(sev.get("labels", ["none", "low", "medium", "high"])),
        matrix=[list(map(int, row)) for row in sev.get("matrix", [])],
        accuracy=_f(sev.get("accuracy")),
    )
    signal_correlations = [
        SignalCorrelation(
            signal_name=s.get("signal_name") or s.get("name", ""),
            pearson_r=_f(s.get("pearson_r")),
            roc_auc=_f(s.get("roc_auc")),
        ) for s in sigs
    ]
    threshold_sweep = [
        ThresholdSweepRow(
            threshold=_f(r.get("threshold")),
            precision=_f(r.get("precision")),
            recall=_f(r.get("recall")),
            f1=_f(r.get("f1")),
        ) for r in sweep
    ]
    return EvalReport(
        n_conversations=int(overview.get("n_conversations", 0)),
        n_turns=int(overview.get("n_turns", 0)),
        drift_rate=_f(overview.get("drift_rate")),
        binary=binary,
        calibration=calibration,
        time_to_detection=time_to_detection,
        severity_confusion=severity_confusion,
        signal_correlations=signal_correlations,
        threshold_sweep=threshold_sweep,
        per_drift_type_recall={k: _f(v) for k, v in pdt.items()},
    )


if __name__ == "__main__":
    sys.exit(main())
