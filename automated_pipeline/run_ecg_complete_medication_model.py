import argparse
import json
from datetime import datetime
from pathlib import Path

from cache import MatrixCache
from config import PipelineConfig
from engine import DataBuilder, ModelEngine


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


def active_feature_list(cfg, feature_preset, drop_features=None):
    if feature_preset == "preprune_median_labs":
        return []
    drop_features = set(drop_features or [])
    features = list(cfg.FEATURE_SET_PRESETS[feature_preset])
    return [feature for feature in features if feature not in drop_features]


def configure(
    report_dir,
    feature_preset,
    cv_folds,
    drop_features=None,
    use_clinically_similar_controls=True,
    preprune_median_labs=False,
):
    cfg = PipelineConfig()
    cfg.USE_CLINICALLY_SIMILAR_CONTROLS = use_clinically_similar_controls
    cfg.FEATURE_AGGREGATION_MODE = "median_only"
    cfg.INCLUDE_FIRST_FEATURES = False
    cfg.USE_MEDICATION_FEATURES = (
        False
        if preprune_median_labs
        else feature_preset != "labs_only_pre_tmux_tol0p01"
    )
    features = active_feature_list(cfg, feature_preset, drop_features)
    if preprune_median_labs:
        cfg.ACTIVE_FEATURE_SET_PRESET = None
        cfg.FEATURES_TO_KEEP = []
    elif drop_features:
        cfg.ACTIVE_FEATURE_SET_PRESET = None
        cfg.FEATURES_TO_KEEP = features
    else:
        cfg.ACTIVE_FEATURE_SET_PRESET = feature_preset
    cfg.USE_RECURSIVE_FEATURE_ELIMINATION = False
    cfg.REQUIRE_ECG_MEASUREMENTS = True
    cfg.RUN_CROSS_VALIDATION = True
    cfg.CV_FOLDS = cv_folds
    cfg.MODEL_REPORT_DIR = report_dir
    cfg.LOW_IMPORTANCE_REPORT_N = len(features)
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed medication feature set on patients with in-window ECG "
            "machine measurements and write 5-fold CV plus curve reports."
        )
    )
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--feature-preset", default="medication_tol0p01")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--drop-features", nargs="*", default=[])
    parser.add_argument(
        "--preprune-median-labs",
        action="store_true",
        help=(
            "Use median-only base labs/vitals/ECG features without medication "
            "features, fixed feature presets, or recursive feature elimination."
        ),
    )
    parser.add_argument(
        "--no-control-icd-filter",
        action="store_true",
        help=(
            "Do not restrict controls to the configured clinically similar ICD "
            "diagnosis groups before selecting the most recent control admission."
        ),
    )
    args = parser.parse_args()

    report_dir = Path("../data/processed/model_reports/ecg_complete_medication_runs") / args.run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    cfg = configure(
        report_dir,
        args.feature_preset,
        args.cv_folds,
        args.drop_features,
        use_clinically_similar_controls=not args.no_control_icd_filter,
        preprune_median_labs=args.preprune_median_labs,
    )
    active_features = active_feature_list(cfg, args.feature_preset, args.drop_features)

    manifest = {
        "run_id": args.run_id,
        "feature_preset": args.feature_preset,
        "preprune_median_labs": args.preprune_median_labs,
        "dropped_features": args.drop_features,
        "active_features": active_features,
        "revert_feature_preset": "labs_only_pre_tmux_tol0p01",
        "revert_features": cfg.FEATURE_SET_PRESETS["labs_only_pre_tmux_tol0p01"],
        "require_ecg_measurements": cfg.REQUIRE_ECG_MEASUREMENTS,
        "cv_folds": cfg.CV_FOLDS,
        "aggregation_mode": cfg.FEATURE_AGGREGATION_MODE,
        "include_first_features": cfg.INCLUDE_FIRST_FEATURES,
        "use_medication_features": cfg.USE_MEDICATION_FEATURES,
        "use_clinically_similar_controls": cfg.USE_CLINICALLY_SIMILAR_CONTROLS,
        "reports_dir": str(report_dir),
    }
    (report_dir / f"{args.run_id}_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    (report_dir / f"{args.run_id}_active_features.txt").write_text(
        "\n".join(active_features) + "\n"
    )
    (report_dir / f"{args.run_id}_revert_labs_only_features.txt").write_text(
        "\n".join(cfg.FEATURE_SET_PRESETS["labs_only_pre_tmux_tol0p01"]) + "\n"
    )

    matrix_path = build_or_restore_matrix(cfg)
    engine = ModelEngine(cfg)
    engine.run_id = f"{args.run_id}_ecg_complete_{args.feature_preset}"
    X_train, y_train, X_test, y_test = engine.preprocess(matrix_path)
    if args.preprune_median_labs:
        active_features = list(X_train.columns)
        manifest["active_features"] = active_features
        manifest["active_feature_count"] = len(active_features)
        (report_dir / f"{args.run_id}_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True)
        )
        (report_dir / f"{args.run_id}_active_features.txt").write_text(
            "\n".join(active_features) + "\n"
        )
    engine.train_and_eval(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
