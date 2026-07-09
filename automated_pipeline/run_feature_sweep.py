import argparse
import json
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd

from cache import MatrixCache
from config import PipelineConfig
from engine import DataBuilder, ModelEngine


TOLERANCES = [0.005, 0.01, 0.02]
FEATURE_MODES = ["full", "median_only"]


def tolerance_label(tolerance):
    return str(tolerance).replace(".", "p")


def configure_experiment(mode, tolerance, report_dir):
    cfg = PipelineConfig()
    cfg.FEATURE_AGGREGATION_MODE = mode
    cfg.INCLUDE_FIRST_FEATURES = mode == "full"
    cfg.USE_RECURSIVE_FEATURE_ELIMINATION = True
    cfg.RFE_PERFORMANCE_TOLERANCE = tolerance
    cfg.RFE_PROTECTED_FEATURES = ["race"]
    cfg.RFE_MIN_FEATURES = 40 if mode == "full" else 10
    cfg.RUN_SHADOW_FEATURE_FILTER_REPORT = False
    cfg.RUN_SHADOW_REJECT_COMPARISON = False
    cfg.RUN_INR_ABLATION_COMPARISON = False
    cfg.MODEL_REPORT_DIR = report_dir
    return cfg


def build_or_restore_matrix(cfg):
    matrix_cache = MatrixCache(cfg)
    matrix_path = matrix_cache.restore_raw_matrix()
    if matrix_path is not None:
        return matrix_path

    builder = DataBuilder(cfg)
    temp_spine_path = builder.build_spine_and_demographics()
    temp_vitals_path = builder.add_vitals(temp_spine_path)
    temp_matrix_path = builder.add_labs(temp_vitals_path)
    temp_matrix_path = builder.add_ecg_features(temp_matrix_path)
    temp_matrix_path = builder.add_medication_features(temp_matrix_path)
    matrix_cache.store_raw_matrix(temp_matrix_path)
    return temp_matrix_path


def read_summary(report_dir, run_id):
    summary_path = report_dir / f"{run_id}_main_summary_metrics.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Expected summary report not found: {summary_path}")
    return pd.read_csv(summary_path)


def run_experiment(mode, tolerance, sweep_id, report_dir):
    cfg = configure_experiment(mode, tolerance, report_dir)
    run_label = f"{sweep_id}_{mode}_tol{tolerance_label(tolerance)}"

    print("\n" + "=" * 80)
    print(f"Starting experiment: mode={mode}, tolerance={tolerance}")
    print("=" * 80)

    matrix_path = build_or_restore_matrix(cfg)
    engine = ModelEngine(cfg)
    engine.run_id = run_label
    X_train, y_train, X_test, y_test = engine.preprocess(matrix_path)
    engine.train_and_eval(X_train, y_train, X_test, y_test)

    summary = read_summary(report_dir, run_label)
    summary.insert(0, "sweep_id", sweep_id)
    summary.insert(1, "experiment_status", "complete")
    summary.insert(2, "feature_mode", mode)
    summary.insert(3, "configured_rfe_tolerance", tolerance)
    return summary


def append_results(results_path, rows):
    result_df = pd.concat(rows, ignore_index=True)
    result_df.to_csv(results_path, index=False)

    complete = result_df[result_df["experiment_status"] == "complete"].copy()
    if not complete.empty:
        ranked_path = results_path.with_name(f"{results_path.stem}_ranked.csv")
        ranked = complete.sort_values(
            by=[
                "holdout_average_precision",
                "holdout_pr_auc_trapezoidal",
                "selected_f1",
                "feature_count",
            ],
            ascending=[False, False, False, True],
        )
        ranked.to_csv(ranked_path, index=False)


def failure_row(sweep_id, mode, tolerance, error):
    return pd.DataFrame([{
        "sweep_id": sweep_id,
        "experiment_status": "failed",
        "feature_mode": mode,
        "configured_rfe_tolerance": tolerance,
        "error": str(error),
    }])


def main():
    parser = argparse.ArgumentParser(
        description="Run recursive feature-elimination sweeps across feature aggregation modes and tolerances."
    )
    parser.add_argument("--sweep-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--modes", nargs="+", default=FEATURE_MODES, choices=FEATURE_MODES)
    parser.add_argument("--tolerances", nargs="+", type=float, default=TOLERANCES)
    args = parser.parse_args()

    report_dir = Path("../data/processed/model_reports/feature_sweeps") / args.sweep_id
    report_dir.mkdir(parents=True, exist_ok=True)
    results_path = report_dir / f"{args.sweep_id}_sweep_summary.csv"
    manifest_path = report_dir / f"{args.sweep_id}_manifest.json"
    manifest_path.write_text(json.dumps({
        "sweep_id": args.sweep_id,
        "feature_modes": args.modes,
        "tolerances": args.tolerances,
        "race_protected": True,
        "selection_metric": "validation_average_precision",
        "reports_dir": str(report_dir),
    }, indent=2, sort_keys=True))

    rows = []
    for mode in args.modes:
        for tolerance in args.tolerances:
            try:
                rows.append(run_experiment(mode, tolerance, args.sweep_id, report_dir))
            except Exception as exc:
                print("\nEXPERIMENT FAILED")
                print(f"mode={mode}, tolerance={tolerance}")
                traceback.print_exc()
                rows.append(failure_row(args.sweep_id, mode, tolerance, exc))
            append_results(results_path, rows)
            print(f"\nUpdated sweep summary: {results_path}")

    print("\nSweep complete.")
    print(f"Summary: {results_path}")
    ranked_path = results_path.with_name(f"{results_path.stem}_ranked.csv")
    if ranked_path.exists():
        print(f"Ranked summary: {ranked_path}")


if __name__ == "__main__":
    main()
