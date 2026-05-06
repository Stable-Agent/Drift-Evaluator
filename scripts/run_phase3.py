#!/usr/bin/env python
# Copyright 2026 Stable-Agent Contributors
# SPDX-License-Identifier: MIT
"""Phase 3 evaluation: DriftDetectorV2 with LLM-judge constraint signal.

Runs v2 with constraint_mode="llm_judge" against the dataset, fits weights/
threshold, and renders comparison reports against:
  - v0 baseline (cosine signals)
  - v2 fitted (NLI signals)

Usage:
  python run_phase3.py --subset 25     # quick validation on first 25 conversations
  python run_phase3.py                  # full 200-conversation run (slow on first call;
                                          ~1-2hr while populating LLM-judge cache)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path,
                   default=PROJECT_ROOT / "datasets" / "v1_synthetic")
    p.add_argument("--reports-dir", type=Path,
                   default=PROJECT_ROOT / "reports")
    p.add_argument("--cache-dir", type=Path,
                   default=PROJECT_ROOT / ".cache",
                   help="LLM-judge response cache lives here (per-model)")
    p.add_argument("--baseline-v0", type=Path,
                   default=PROJECT_ROOT / "reports" / "baseline_v0.json")
    p.add_argument("--baseline-v2", type=Path,
                   default=PROJECT_ROOT / "reports" / "v2_fitted.json")
    p.add_argument("--threshold", type=float, default=0.50)
    p.add_argument("--ollama-model", type=str, default="llama3.1:8b")
    p.add_argument("--subset", type=int, default=0,
                   help="if >0, only evaluate the first N conversations (validation mode)")
    p.add_argument("--rows-cache", type=Path,
                   default=PROJECT_ROOT / "reports" / "v2_llm_judge_rows.jsonl",
                   help="path to per-row cache (skip detector eval if exists)")
    p.add_argument("--force-eval", action="store_true",
                   help="ignore --rows-cache and re-run detector")
    p.add_argument("--no-progress", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    dataset_dir = args.dataset.resolve()
    reports_dir = args.reports_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_dir.exists():
        print(f"Error: dataset not found: {dataset_dir}", file=sys.stderr)
        return 1

    # Imports here so --help is fast
    from drift_evaluator.harness import evaluate_dataset, make_v2_llm_judge_factory
    from drift_evaluator.metrics import compute_full_report
    from drift_evaluator.fit import fit_all, apply_fitted_config
    from drift_evaluator.report import write_report

    # Optional: subset the dataset by copying first N conv_*.json into a temp dir
    eval_dataset = dataset_dir
    if args.subset > 0:
        subset_dir = reports_dir / f"_subset_{args.subset}"
        subset_dir.mkdir(parents=True, exist_ok=True)
        # Wipe any prior subset content (but keep manifest/conv files we copy in)
        for old in subset_dir.glob("*.json"):
            old.unlink()
        files = sorted(p for p in dataset_dir.glob("conv_*.json"))[: args.subset]
        for src in files:
            shutil.copy(src, subset_dir / src.name)
        # Copy manifest if present (not strictly required by load_dataset)
        manifest = dataset_dir / "manifest.json"
        if manifest.exists():
            shutil.copy(manifest, subset_dir / "manifest.json")
        eval_dataset = subset_dir
        print(f"[subset] Evaluating first {len(files)} conversations from {subset_dir}",
              file=sys.stderr)

    # Per-model LLM-judge cache (so different models don't pollute each other)
    safe_model = args.ollama_model.replace(":", "_").replace("/", "_")
    llm_cache = cache_dir / f"llm_judge_{safe_model}.jsonl"

    rows_cache: Path = args.rows_cache.resolve()
    if rows_cache.exists() and not args.force_eval and args.subset == 0:
        print(f"[1/3] Loading cached rows from {rows_cache}", file=sys.stderr)
        rows_default = _load_rows(rows_cache)
        report_default = compute_full_report(rows_default, args.threshold)
    else:
        print(f"[1/3] Running v2 LLM-judge (cache: {llm_cache})...", file=sys.stderr)
        factory = make_v2_llm_judge_factory(
            ollama_model=args.ollama_model,
            cache_path=llm_cache,
            drift_threshold=args.threshold,
        )
        rows_default, report_default = evaluate_dataset(
            eval_dataset, factory,
            detector_threshold=args.threshold,
            progress=not args.no_progress,
        )
        if args.subset == 0:
            rows_cache.parent.mkdir(parents=True, exist_ok=True)
            _save_rows(rows_default, rows_cache)
            print(f"[1/3] Cached rows to {rows_cache}", file=sys.stderr)

    print(f"[1/3] v2-llm-judge default: P={report_default.binary.precision:.3f} "
          f"R={report_default.binary.recall:.3f} F1={report_default.binary.f1:.3f} "
          f"ROC-AUC={report_default.calibration.roc_auc:.3f}", file=sys.stderr)

    # Per-signal correlations (the headline question for phase 3)
    by_name = {s.signal_name: s for s in report_default.signal_correlations}

    def _auc(name: str) -> str:
        if name in by_name:
            return f"{by_name[name].roc_auc:.3f}"
        return "n/a"

    print(f"[1/3] Per-signal ROC-AUC: goal={_auc('goal')} "
          f"constraint={_auc('constraint')} consistency={_auc('consistency')}",
          file=sys.stderr)

    name_suffix = f"_subset{args.subset}" if args.subset > 0 else ""

    write_report(
        report_default, reports_dir,
        name=f"v2_llm_judge_default{name_suffix}",
        title="Drift Detector v2 — LLM-judge constraints (default weights)",
        dataset_name=eval_dataset.name,
        detector_name=f"DriftDetectorV2 (constraint_mode=llm_judge, model={args.ollama_model})",
        extra_notes=f"Threshold: {args.threshold}\nDataset: {eval_dataset}",
    )

    print("[2/3] Fitting weights/threshold from LLM-judge rows...", file=sys.stderr)
    fitted = fit_all(rows_default)
    print(f"[2/3] Fitted weights: {fitted.weights}", file=sys.stderr)
    print(f"[2/3] Fitted threshold: {fitted.threshold:.3f}", file=sys.stderr)

    rows_fitted = apply_fitted_config(rows_default, fitted)
    report_fitted = compute_full_report(rows_fitted, fitted.threshold)
    write_report(
        report_fitted, reports_dir,
        name=f"v2_llm_judge_fitted{name_suffix}",
        title="Drift Detector v2 — LLM-judge constraints (fitted)",
        dataset_name=eval_dataset.name,
        detector_name="DriftDetectorV2 (LLM-judge, fitted)",
        extra_notes=(
            f"Fitted threshold: {fitted.threshold:.3f}\n"
            f"Fitted weights: {fitted.weights}\n"
            f"Fitted severity_thresholds: {fitted.severity_thresholds}"
        ),
    )
    print(f"[2/3] v2-llm-judge fitted: P={report_fitted.binary.precision:.3f} "
          f"R={report_fitted.binary.recall:.3f} F1={report_fitted.binary.f1:.3f} "
          f"ROC-AUC={report_fitted.calibration.roc_auc:.3f}", file=sys.stderr)

    fitted_path = reports_dir / f"v2_llm_judge_fitted_config{name_suffix}.json"
    fitted_path.write_text(json.dumps({
        "weights": fitted.weights,
        "threshold": fitted.threshold,
        "severity_thresholds": fitted.severity_thresholds,
        "fit_metrics": fitted.fit_metrics,
    }, indent=2))

    if args.subset == 0:
        from drift_evaluator.compare import write_comparison
        if args.baseline_v0.exists():
            print(f"[3/3] Comparing vs v0 baseline...", file=sys.stderr)
            report_v0 = _report_from_json(args.baseline_v0)
            write_comparison(
                report_v0, report_fitted, reports_dir,
                name_a="v0 baseline (cosine)",
                name_b="v2 LLM-judge fitted",
                name="comparison_v0_vs_v2_llm_judge",
                title="Drift Detector: v0 baseline vs v2 LLM-judge",
                notes="v2 LLM-judge replaces NLI constraint with local Llama as judge.",
            )
        if args.baseline_v2.exists():
            print(f"[3/3] Comparing vs v2 NLI fitted...", file=sys.stderr)
            report_v2 = _report_from_json(args.baseline_v2)
            write_comparison(
                report_v2, report_fitted, reports_dir,
                name_a="v2 fitted (NLI constraints)",
                name_b="v2 LLM-judge fitted",
                name="comparison_v2_nli_vs_v2_llm_judge",
                title="Drift Detector: v2 NLI vs v2 LLM-judge",
                notes="Apples-to-apples: same goal/consistency signals; only constraint signal differs.",
            )

    print(f"\nReports written to: {reports_dir}", file=sys.stderr)
    return 0


def _save_rows(rows, path: Path) -> None:
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
            rows.append(EvalRow(**json.loads(line)))
    return rows


def _report_from_json(path: Path):
    """Reconstruct an EvalReport from the JSON written by report.write_report."""
    from drift_evaluator.metrics import (
        EvalReport, BinaryMetrics, CalibrationMetrics, TimeToDetection,
        SeverityConfusion, SignalCorrelation, ThresholdSweepRow,
    )
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

    return EvalReport(
        n_conversations=int(overview.get("n_conversations", 0)),
        n_turns=int(overview.get("n_turns", 0)),
        drift_rate=_f(overview.get("drift_rate")),
        binary=BinaryMetrics(
            threshold=_f(headline.get("threshold")),
            precision=_f(headline.get("precision")),
            recall=_f(headline.get("recall")),
            f1=_f(headline.get("f1")),
            accuracy=_f(headline.get("accuracy")),
            true_positives=int(headline.get("true_positives", 0)),
            false_positives=int(headline.get("false_positives", 0)),
            true_negatives=int(headline.get("true_negatives", 0)),
            false_negatives=int(headline.get("false_negatives", 0)),
        ),
        calibration=CalibrationMetrics(
            brier_score=_f(cal.get("brier_score")),
            roc_auc=_f(cal.get("roc_auc")),
            pr_auc=_f(cal.get("pr_auc")),
        ),
        time_to_detection=TimeToDetection(
            detected_count=int(ttd.get("detected_count", 0)),
            missed_count=int(ttd.get("missed_count", 0)),
            early_count=int(ttd.get("early_count", 0)),
            delays=[int(d) for d in (ttd.get("delays") or [])],
            median_delay=_f(ttd.get("median_delay")),
            mean_delay=_f(ttd.get("mean_delay")),
            p90_delay=_f(ttd.get("p90_delay")),
        ),
        severity_confusion=SeverityConfusion(
            labels=list(sev.get("labels", ["none", "low", "medium", "high"])),
            matrix=[list(map(int, row)) for row in sev.get("matrix", [])],
            accuracy=_f(sev.get("accuracy")),
        ),
        signal_correlations=[
            SignalCorrelation(
                signal_name=s.get("signal_name") or s.get("name", ""),
                pearson_r=_f(s.get("pearson_r")),
                roc_auc=_f(s.get("roc_auc")),
            ) for s in sigs
        ],
        threshold_sweep=[
            ThresholdSweepRow(
                threshold=_f(r.get("threshold")),
                precision=_f(r.get("precision")),
                recall=_f(r.get("recall")),
                f1=_f(r.get("f1")),
            ) for r in sweep
        ],
        per_drift_type_recall={k: _f(v) for k, v in pdt.items()},
    )


if __name__ == "__main__":
    sys.exit(main())
