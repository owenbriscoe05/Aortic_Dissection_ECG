import argparse
import json
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import pandas as pd

from cache import MatrixCache
from config import PipelineConfig
from engine import DataBuilder, ModelEngine
from run_lab_vitals_first_model import (
    Tee,
    first_match,
    format_metric,
    html_escape,
    image_data_uri,
    plot_pr_curve,
    plot_roc_curve,
    plot_shap_all_features,
    plot_shap_beeswarm_all_features,
    train_model_for_shap,
)


CHEST_PAIN_CONTROL_GROUPS = {
    "chest_pain": {
        9: ["7865"],
        10: ["R07"],
    },
}

DEMOGRAPHIC_FEATURES = [
    "index_age",
    "gender",
    "race",
    "insurance",
    "marital_status",
]


def configure(report_dir, all_controls=False):
    cfg = PipelineConfig()
    cfg.MODEL_REPORT_DIR = report_dir
    cfg.USE_CLINICALLY_SIMILAR_CONTROLS = not all_controls
    cfg.CLINICALLY_SIMILAR_CONTROL_GROUPS = (
        {} if all_controls else CHEST_PAIN_CONTROL_GROUPS
    )
    cfg.VITALS_DICT = {}
    cfg.LABS_DICT = {}
    cfg.USE_EVENT_CACHE = False
    cfg.USE_ECG_FEATURES = True
    cfg.FEATURE_AGGREGATION_MODE = "first_only"
    cfg.INCLUDE_FIRST_FEATURES = True
    cfg.USE_MEDICATION_FEATURES = False
    cfg.REQUIRE_ECG_MEASUREMENTS = True
    cfg.ACTIVE_FEATURE_SET_PRESET = None
    cfg.FEATURES_TO_KEEP = []
    cfg.FEATURES_TO_DROP = ["ecg_count", "encounter_urgency"]
    cfg.LEAKAGE_LABS_TO_DROP = []
    cfg.USE_RECURSIVE_FEATURE_ELIMINATION = False
    cfg.RUN_SHADOW_FEATURE_FILTER_REPORT = False
    cfg.RUN_SHADOW_REJECT_COMPARISON = False
    cfg.RUN_INR_ABLATION_COMPARISON = False
    cfg.RUN_CROSS_VALIDATION = True
    cfg.CV_FOLDS = 5
    cfg.COL_MISSINGNESS_THRESHOLD = 0.999
    return cfg


def control_cohort_description(cfg):
    if getattr(cfg, "USE_CLINICALLY_SIMILAR_CONTROLS", False):
        return {
            "method": "demographics_plus_ecg_first_only_chest_pain_controls",
            "subtitle": (
                "Demographic variables plus first in-window ECG machine measurements; "
                "chest-pain-only ICD controls."
            ),
            "bullet": (
                "Controls are restricted to ICD-coded chest pain candidate admissions, "
                "excluding exact aortic dissection codes."
            ),
        }
    return {
        "method": "demographics_plus_ecg_first_only_all_controls",
        "subtitle": (
            "Demographic variables plus first in-window ECG machine measurements; "
            "all non-dissection controls with in-window ECGs."
        ),
        "bullet": (
            "Controls are not restricted by chest-pain or back-pain ICD diagnosis groups; "
            "each non-dissection control patient's most recent admission is eligible before "
            "the ECG-complete filter."
        ),
    }


def build_or_restore_matrix(cfg):
    matrix_cache = MatrixCache(cfg)
    cached_matrix_path = matrix_cache.restore_raw_matrix()
    if cached_matrix_path is not None:
        return cached_matrix_path

    builder = DataBuilder(cfg)
    temp_matrix_path = builder.build_spine_and_demographics()
    temp_matrix_path = builder.add_ecg_features(temp_matrix_path)
    matrix_cache.store_raw_matrix(temp_matrix_path)
    return temp_matrix_path


def feature_family_counts(active_features):
    demographic = [feature for feature in active_features if feature in DEMOGRAPHIC_FEATURES]
    ecg = [feature for feature in active_features if feature.startswith("ecg_")]
    other = [
        feature
        for feature in active_features
        if feature not in demographic and feature not in ecg
    ]
    return {
        "demographic_features": demographic,
        "ecg_features": ecg,
        "other_features": other,
    }


def threshold_rows_html(thresholds):
    display_cols = [
        "rule",
        "threshold",
        "precision",
        "recall",
        "specificity",
        "f1",
        "tp",
        "fp",
        "fn",
        "tn",
        "flagged_pct",
    ]
    available_cols = [col for col in display_cols if col in thresholds.columns]
    rows = []
    for row in thresholds[available_cols].itertuples(index=False):
        cells = []
        for col, value in zip(available_cols, row):
            if col == "rule":
                rendered = html_escape(value)
                align_left = " class=\"left\""
            elif col == "flagged_pct":
                rendered = f"{float(value):.1%}"
                align_left = ""
            elif col in {"threshold", "precision", "recall", "specificity", "f1"}:
                rendered = format_metric(value)
                align_left = ""
            else:
                rendered = f"{int(value):,}"
                align_left = ""
            cells.append(f"<td{align_left}>{rendered}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    header = "".join(f"<th>{html_escape(col)}</th>" for col in available_cols)
    body = "\n".join(rows)
    return f"<thead><tr>{header}</tr></thead><tbody>{body}</tbody>"


def create_html_presentation(report_dir, cfg, X_train, y_train, X_test, active_features):
    summary = pd.read_csv(first_match(report_dir, "_main_summary_metrics.csv")).iloc[0]
    cv = pd.read_csv(first_match(report_dir, "_main_cv_metrics.csv"))
    thresholds = pd.read_csv(first_match(report_dir, "_main_threshold_metrics.csv"))
    pr_path = first_match(report_dir, "_main_pr_curve.csv")
    roc_path = first_match(report_dir, "_main_roc_curve.csv")
    importance = pd.read_csv(first_match(report_dir, "_main_feature_importance.csv"))

    pr_png = report_dir / "demographics_ecg_pr_curve.png"
    roc_png = report_dir / "demographics_ecg_roc_curve.png"
    plot_pr_curve(pr_path, summary, pr_png)
    plot_roc_curve(roc_path, summary, roc_png)

    shap_model = train_model_for_shap(cfg, X_train, y_train, summary)
    shap_png = report_dir / "demographics_ecg_shap_all_features.png"
    shap_csv = report_dir / "demographics_ecg_shap_all_features.csv"
    shap_df, shap_values = plot_shap_all_features(shap_model, X_test, shap_png, shap_csv)
    shap_beeswarm_png = report_dir / "demographics_ecg_shap_beeswarm_all_features.png"
    plot_shap_beeswarm_all_features(shap_df, shap_values, X_test, shap_beeswarm_png)

    validation_row = thresholds[thresholds["rule"].eq("validation_f1")].iloc[0]
    default_row = thresholds[thresholds["rule"].eq("default_0.5")].iloc[0]
    top_features = importance.head(12)["Feature"].tolist()
    top_shap_features = shap_df.head(12)["feature"].tolist()
    families = feature_family_counts(active_features)
    control_text = control_cohort_description(cfg)

    fold_rows = "\n".join(
        "<tr>"
        f"<td>{int(row.fold)}</td>"
        f"<td>{format_metric(row.average_precision)}</td>"
        f"<td>{format_metric(row.pr_auc_trapezoidal)}</td>"
        f"<td>{format_metric(row.roc_auc)}</td>"
        "</tr>"
        for row in cv.itertuples(index=False)
    )
    feature_rows = "\n".join(
        "<tr>"
        f"<td class=\"left\">{html_escape(row.Feature)}</td>"
        f"<td>{format_metric(row.Importance, digits=5)}</td>"
        "</tr>"
        for row in importance.head(20).itertuples(index=False)
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Aortic Dissection Demographics/ECG Model</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17202a;
      --muted: #5d6875;
      --line: #d9dee7;
      --blue: #264f78;
      --panel: #f7f9fc;
    }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--ink);
      background: #ffffff;
      line-height: 1.45;
    }}
    main {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 28px 26px 56px;
    }}
    section {{
      border-top: 1px solid var(--line);
      padding: 28px 0;
      page-break-after: always;
    }}
    section:first-child {{ border-top: 0; }}
    h1 {{
      font-size: 34px;
      margin: 0 0 8px;
      letter-spacing: 0;
    }}
    h2 {{
      font-size: 24px;
      margin: 0 0 14px;
      color: var(--blue);
      letter-spacing: 0;
    }}
    p, li {{ font-size: 16px; }}
    .subtitle {{
      color: var(--muted);
      font-size: 17px;
      margin: 0 0 20px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin: 18px 0;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 13px 14px;
    }}
    .metric .label {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .metric .value {{
      display: block;
      font-size: 25px;
      font-weight: 700;
      margin-top: 3px;
    }}
    .two-col {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
      align-items: start;
    }}
    img {{
      max-width: 100%;
      height: auto;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: white;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 14px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 10px;
      text-align: right;
      white-space: nowrap;
    }}
    th {{
      color: var(--muted);
      font-weight: 700;
      background: var(--panel);
    }}
    th:first-child, td.left {{
      text-align: left;
      white-space: normal;
    }}
    .note {{
      color: var(--muted);
      font-size: 14px;
    }}
    @media (max-width: 820px) {{
      .grid, .two-col {{ grid-template-columns: 1fr; }}
      table {{ font-size: 12px; }}
      th, td {{ padding: 6px; }}
    }}
  </style>
</head>
<body>
<main>
  <section>
    <h1>Aortic Dissection Demographics/ECG Model</h1>
    <p class="subtitle">{html_escape(control_text["subtitle"])}</p>
    <ul>
      <li>{html_escape(control_text["bullet"])}</li>
      <li>Back-pain controls, labs, vitals, medication features, encounter urgency, and ECG count are not model predictors.</li>
      <li>The cohort is restricted to patients with at least one ECG machine-measurement row in the 24-hour feature window from index time.</li>
      <li>ECG features use first values only; derived ECG features include QT interval and Bazett QTc.</li>
    </ul>
  </section>

  <section>
    <h2>Cohort and Features</h2>
    <div class="grid">
      <div class="metric"><span class="label">Train rows</span><span class="value">{int(summary['train_rows']):,}</span></div>
      <div class="metric"><span class="label">Train positives</span><span class="value">{int(summary['train_positive']):,}</span></div>
      <div class="metric"><span class="label">Holdout rows</span><span class="value">{int(summary['holdout_rows']):,}</span></div>
      <div class="metric"><span class="label">Holdout positives</span><span class="value">{int(summary['holdout_positive']):,}</span></div>
    </div>
    <p>Feature count after preprocessing: <strong>{int(summary['feature_count'])}</strong>. Demographic features: <strong>{len(families['demographic_features'])}</strong>; ECG features: <strong>{len(families['ecg_features'])}</strong>.</p>
    <p>Top model-importance features: {", ".join(f"<strong>{html_escape(feature)}</strong>" for feature in top_features)}.</p>
    <p class="note">Active feature list and run metadata are written beside this presentation.</p>
  </section>

  <section>
    <h2>Holdout Performance</h2>
    <div class="grid">
      <div class="metric"><span class="label">Average precision</span><span class="value">{format_metric(summary['holdout_average_precision'])}</span></div>
      <div class="metric"><span class="label">PR-AUC</span><span class="value">{format_metric(summary['holdout_pr_auc_trapezoidal'])}</span></div>
      <div class="metric"><span class="label">ROC-AUC</span><span class="value">{format_metric(summary['holdout_roc_auc'])}</span></div>
      <div class="metric"><span class="label">Tuned F1</span><span class="value">{format_metric(summary['selected_f1'])}</span></div>
    </div>
    <ul>
      <li>Tuned threshold: {format_metric(summary['selected_threshold'])}</li>
      <li>Tuned precision / recall / F1: {format_metric(validation_row['precision'])} / {format_metric(validation_row['recall'])} / {format_metric(validation_row['f1'])}</li>
      <li>Default threshold F1: {format_metric(default_row['f1'])}</li>
    </ul>
  </section>

  <section>
    <h2>Threshold Operating Points</h2>
    <table>
      {threshold_rows_html(thresholds)}
    </table>
  </section>

  <section>
    <h2>PR and ROC Curves</h2>
    <div class="two-col">
      <div><img src="{image_data_uri(pr_png)}" alt="Precision recall curve"></div>
      <div><img src="{image_data_uri(roc_png)}" alt="ROC curve"></div>
    </div>
  </section>

  <section>
    <h2>Directional SHAP Beeswarm</h2>
    <p class="note">Computed with XGBoost native SHAP contribution values on the ECG-complete holdout set. Points to the right increase predicted risk on the model margin; points to the left decrease it. Color is feature value, normalized within each feature.</p>
    <img src="{image_data_uri(shap_beeswarm_png)}" alt="Directional all-feature SHAP beeswarm">
    <p>Top SHAP features: {", ".join(f"<strong>{html_escape(feature)}</strong>" for feature in top_shap_features)}.</p>
  </section>

  <section>
    <h2>All-Feature SHAP Magnitude</h2>
    <p class="note">Mean absolute SHAP contribution values on the ECG-complete holdout set. Values are mean absolute contributions to the model margin.</p>
    <img src="{image_data_uri(shap_png)}" alt="All-feature SHAP summary">
  </section>

  <section>
    <h2>Feature Importance</h2>
    <table>
      <thead><tr><th>Feature</th><th>Importance</th></tr></thead>
      <tbody>{feature_rows}</tbody>
    </table>
  </section>

  <section>
    <h2>5-Fold Cross-Validation</h2>
    <div class="grid">
      <div class="metric"><span class="label">CV AP</span><span class="value">{format_metric(summary['cv_average_precision_mean'])} +/- {format_metric(summary['cv_average_precision_sd'])}</span></div>
      <div class="metric"><span class="label">CV PR-AUC</span><span class="value">{format_metric(summary['cv_pr_auc_trapezoidal_mean'])} +/- {format_metric(summary['cv_pr_auc_trapezoidal_sd'])}</span></div>
      <div class="metric"><span class="label">CV ROC-AUC</span><span class="value">{format_metric(summary['cv_roc_auc_mean'])} +/- {format_metric(summary['cv_roc_auc_sd'])}</span></div>
      <div class="metric"><span class="label">Folds</span><span class="value">{int(summary['cv_folds'])}</span></div>
    </div>
    <table>
      <thead><tr><th>Fold</th><th>AP</th><th>PR-AUC</th><th>ROC-AUC</th></tr></thead>
      <tbody>{fold_rows}</tbody>
    </table>
  </section>
</main>
</body>
</html>
"""
    html_path = report_dir / "demographics_ecg_model_update.html"
    html_path.write_text(html)
    print(f"HTML presentation: {html_path}")
    print(f"PR curve: {pr_png}")
    print(f"ROC curve: {roc_png}")
    print(f"SHAP plot: {shap_png}")
    print(f"SHAP beeswarm: {shap_beeswarm_png}")
    print(f"SHAP values: {shap_csv}")


def write_manifest(report_dir, run_id, cfg, active_features=None):
    manifest = {
        "run_id": run_id,
        "method": control_cohort_description(cfg)["method"],
        "use_clinically_similar_controls": bool(cfg.USE_CLINICALLY_SIMILAR_CONTROLS),
        "control_groups": cfg.CLINICALLY_SIMILAR_CONTROL_GROUPS,
        "feature_aggregation_mode": cfg.FEATURE_AGGREGATION_MODE,
        "include_first_features": bool(cfg.INCLUDE_FIRST_FEATURES),
        "use_ecg_features": bool(cfg.USE_ECG_FEATURES),
        "require_ecg_measurements": bool(cfg.REQUIRE_ECG_MEASUREMENTS),
        "ecg_required_count_col": cfg.ECG_REQUIRED_COUNT_COL,
        "use_labs": bool(cfg.LABS_DICT),
        "use_vitals": bool(cfg.VITALS_DICT),
        "use_medication_features": bool(cfg.USE_MEDICATION_FEATURES),
        "features_to_drop": list(getattr(cfg, "FEATURES_TO_DROP", [])),
        "col_missingness_threshold": float(cfg.COL_MISSINGNESS_THRESHOLD),
        "row_missingness_threshold": float(cfg.ROW_MISSINGNESS_THRESHOLD),
        "cv_folds": int(cfg.CV_FOLDS),
        "reports_dir": str(report_dir),
    }
    if active_features is not None:
        families = feature_family_counts(active_features)
        manifest["active_features_after_preprocessing"] = active_features
        manifest["active_feature_count_after_preprocessing"] = len(active_features)
        manifest.update(families)

    manifest_path = report_dir / f"{run_id}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest_path


def run(args, report_dir):
    cfg = configure(report_dir, all_controls=args.all_controls)
    matrix_path = build_or_restore_matrix(cfg)
    write_manifest(report_dir, args.run_id, cfg)

    engine = ModelEngine(cfg)
    engine.run_id = f"{args.run_id}_demographics_ecg"
    X_train, y_train, X_test, y_test = engine.preprocess(matrix_path)
    active_features = list(X_train.columns)
    unexpected_features = feature_family_counts(active_features)["other_features"]
    if unexpected_features:
        raise ValueError(
            "Unexpected non-demographic/non-ECG model features remained after preprocessing: "
            f"{unexpected_features}"
        )

    (report_dir / f"{args.run_id}_active_features.txt").write_text(
        "\n".join(active_features) + "\n"
    )
    write_manifest(report_dir, args.run_id, cfg, active_features=active_features)
    engine.train_and_eval(X_train, y_train, X_test, y_test)
    create_html_presentation(report_dir, cfg, X_train, y_train, X_test, active_features)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run a demographics plus first-value ECG machine-measurement model "
            "using chest-pain-only ICD controls and a 24-hour ECG requirement."
        )
    )
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument(
        "--all-controls",
        action="store_true",
        help=(
            "Do not restrict controls to chest-pain ICD candidate admissions before "
            "requiring in-window ECG machine measurements."
        ),
    )
    args = parser.parse_args()

    report_dir = Path("../data/processed/model_reports/demographics_ecg_runs") / args.run_id
    report_dir.mkdir(parents=True, exist_ok=False)
    log_path = report_dir / "run_console.log"
    with log_path.open("w") as log_file:
        stdout_tee = Tee(sys.stdout, log_file)
        stderr_tee = Tee(sys.stderr, log_file)
        with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
            print(f"Run directory: {report_dir}")
            run(args, report_dir)
            print(f"Console log: {log_path}")


if __name__ == "__main__":
    main()
