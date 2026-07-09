import argparse
import json
import time
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd

from cache import MatrixCache
from config import PipelineConfig
from engine import DataBuilder, ModelEngine


TOLERANCES = [0.005, 0.01, 0.02]

ADDITIONAL_LABS = {
    50862: "albumin",
    50863: "alkaline_phosphatase",
    50867: "amylase",
    50883: "bilirubin_direct",
    50885: "bilirubin_total",
    50908: "ckmb_index",
    50910: "creatine_kinase",
    50911: "ck_mb",
    50956: "lipase",
    50963: "ntprobnp",
    51133: "absolute_lymphocyte_count",
    51146: "basophils",
    51199: "eosinophil_count",
    51200: "eosinophils",
    51244: "lymphocytes",
    51245: "lymphocyte_percent",
    51248: "mch",
    51249: "mchc",
    51250: "mcv",
    51253: "monocyte_count",
    51254: "monocytes",
    51256: "neutrophils",
    51274: "pt",
    51275: "ptt",
    51277: "rdw",
    51288: "esr",
    52069: "absolute_basophil_count",
    52073: "absolute_eosinophil_count",
    52074: "absolute_monocyte_count",
    52075: "absolute_neutrophil_count",
    52170: "rbc",
}

VARIANTS = {
    "median_base_labs": {
        "extra_labs": False,
        "medications": False,
    },
    "median_expanded_labs": {
        "extra_labs": True,
        "medications": False,
    },
    "median_base_labs_meds": {
        "extra_labs": False,
        "medications": True,
    },
    "median_expanded_labs_meds": {
        "extra_labs": True,
        "medications": True,
    },
}


def tolerance_label(tolerance):
    return str(tolerance).replace(".", "p")


def prior_sweep_complete(path, expected_runs):
    if not path.exists():
        return False
    df = pd.read_csv(path)
    if len(df) < expected_runs:
        return False
    return df["experiment_status"].isin(["complete", "failed"]).sum() >= expected_runs


def wait_for_prior_sweep(path, expected_runs, poll_seconds):
    if path is None:
        return
    path = Path(path)
    print(f"Waiting for prior sweep summary: {path}")
    while not prior_sweep_complete(path, expected_runs):
        print(f"  prior sweep not complete; sleeping {poll_seconds} seconds")
        time.sleep(poll_seconds)
    print("Prior sweep appears complete; starting expanded feature sweep.")


def configure_variant(variant_name, tolerance, report_dir):
    variant = VARIANTS[variant_name]
    cfg = PipelineConfig()
    cfg.FEATURE_AGGREGATION_MODE = "median_only"
    cfg.INCLUDE_FIRST_FEATURES = False
    cfg.USE_MEDICATION_FEATURES = variant["medications"]
    cfg.USE_RECURSIVE_FEATURE_ELIMINATION = True
    cfg.RFE_PERFORMANCE_TOLERANCE = tolerance
    cfg.RFE_PROTECTED_FEATURES = ["race"]
    cfg.RFE_MIN_FEATURES = 10
    cfg.RUN_SHADOW_FEATURE_FILTER_REPORT = False
    cfg.RUN_SHADOW_REJECT_COMPARISON = False
    cfg.RUN_INR_ABLATION_COMPARISON = False
    cfg.MODEL_REPORT_DIR = report_dir

    if variant["extra_labs"]:
        labs = dict(cfg.LABS_DICT)
        labs.update(ADDITIONAL_LABS)
        cfg.LABS_DICT = labs

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


def run_experiment(variant_name, tolerance, sweep_id, report_dir):
    cfg = configure_variant(variant_name, tolerance, report_dir)
    run_label = f"{sweep_id}_{variant_name}_tol{tolerance_label(tolerance)}"

    print("\n" + "=" * 80)
    print(f"Starting expanded experiment: variant={variant_name}, tolerance={tolerance}")
    print("=" * 80)

    matrix_path = build_or_restore_matrix(cfg)
    engine = ModelEngine(cfg)
    engine.run_id = run_label
    X_train, y_train, X_test, y_test = engine.preprocess(matrix_path)
    engine.train_and_eval(X_train, y_train, X_test, y_test)

    summary = read_summary(report_dir, run_label)
    summary.insert(0, "sweep_id", sweep_id)
    summary.insert(1, "experiment_status", "complete")
    summary.insert(2, "expanded_variant", variant_name)
    summary.insert(3, "configured_rfe_tolerance", tolerance)
    summary.insert(4, "extra_labs_enabled", VARIANTS[variant_name]["extra_labs"])
    summary.insert(5, "medications_enabled", VARIANTS[variant_name]["medications"])
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


def failure_row(sweep_id, variant_name, tolerance, error):
    return pd.DataFrame([{
        "sweep_id": sweep_id,
        "experiment_status": "failed",
        "expanded_variant": variant_name,
        "configured_rfe_tolerance": tolerance,
        "extra_labs_enabled": VARIANTS[variant_name]["extra_labs"],
        "medications_enabled": VARIANTS[variant_name]["medications"],
        "error": str(error),
    }])


def main():
    parser = argparse.ArgumentParser(
        description="Run median-only expanded-feature sweeps with additional labs and medication features."
    )
    parser.add_argument("--sweep-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--tolerances", nargs="+", type=float, default=TOLERANCES)
    parser.add_argument("--wait-for-summary", default=None)
    parser.add_argument("--expected-prior-runs", type=int, default=6)
    parser.add_argument("--poll-seconds", type=int, default=300)
    args = parser.parse_args()

    wait_for_prior_sweep(args.wait_for_summary, args.expected_prior_runs, args.poll_seconds)

    report_dir = Path("../data/processed/model_reports/expanded_feature_sweeps") / args.sweep_id
    report_dir.mkdir(parents=True, exist_ok=True)
    results_path = report_dir / f"{args.sweep_id}_expanded_sweep_summary.csv"
    manifest_path = report_dir / f"{args.sweep_id}_manifest.json"
    manifest_path.write_text(json.dumps({
        "sweep_id": args.sweep_id,
        "variants": args.variants,
        "tolerances": args.tolerances,
        "race_protected": True,
        "aggregation_mode": "median_only",
        "include_first_features": False,
        "additional_labs": ADDITIONAL_LABS,
        "medication_groups": list(PipelineConfig.MEDICATION_GROUP_PATTERNS),
        "reports_dir": str(report_dir),
    }, indent=2, sort_keys=True))

    rows = []
    for variant_name in args.variants:
        for tolerance in args.tolerances:
            try:
                rows.append(run_experiment(variant_name, tolerance, args.sweep_id, report_dir))
            except Exception as exc:
                print("\nEXPANDED EXPERIMENT FAILED")
                print(f"variant={variant_name}, tolerance={tolerance}")
                traceback.print_exc()
                rows.append(failure_row(args.sweep_id, variant_name, tolerance, exc))
            append_results(results_path, rows)
            print(f"\nUpdated expanded sweep summary: {results_path}")

    print("\nExpanded feature sweep complete.")
    print(f"Summary: {results_path}")
    ranked_path = results_path.with_name(f"{results_path.stem}_ranked.csv")
    if ranked_path.exists():
        print(f"Ranked summary: {ranked_path}")


if __name__ == "__main__":
    main()
