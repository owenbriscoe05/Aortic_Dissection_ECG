import argparse
import base64
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from engine import ModelEngine
from run_ecg_complete_medication_model import build_or_restore_matrix, configure


def latest_run_dir(base_dir):
    candidates = [path for path in base_dir.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {base_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def first_match(report_dir, suffix):
    matches = sorted(report_dir.glob(f"*{suffix}"))
    if not matches:
        raise FileNotFoundError(f"No artifact matching *{suffix} in {report_dir}")
    return matches[0]


def image_data_uri(path):
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def format_metric(value, digits=3):
    return f"{float(value):.{digits}f}"


def train_model_for_shap(
    report_dir,
    run_id,
    feature_preset,
    cv_folds,
    summary,
    active_features,
    use_clinically_similar_controls=True,
    preprune_median_labs=False,
):
    cfg = configure(
        report_dir,
        feature_preset,
        cv_folds,
        use_clinically_similar_controls=use_clinically_similar_controls,
        preprune_median_labs=preprune_median_labs,
    )
    if preprune_median_labs:
        cfg.ACTIVE_FEATURE_SET_PRESET = None
        cfg.FEATURES_TO_KEEP = list(active_features)
        cfg.LOW_IMPORTANCE_REPORT_N = len(active_features)
    else:
        preset_features = list(cfg.FEATURE_SET_PRESETS[feature_preset])
        if list(active_features) != preset_features:
            cfg.ACTIVE_FEATURE_SET_PRESET = None
            cfg.FEATURES_TO_KEEP = list(active_features)
            cfg.LOW_IMPORTANCE_REPORT_N = len(active_features)
    matrix_path = build_or_restore_matrix(cfg)
    engine = ModelEngine(cfg)
    engine.run_id = f"{run_id}_shap_rebuild"
    X_train, y_train, X_test, y_test = engine.preprocess(matrix_path)

    params = json.loads(summary["best_params_json"])
    scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = xgb.XGBClassifier(
        **params,
        scale_pos_weight=scale_weight,
        eval_metric=cfg.XGB_EVAL_METRIC,
        random_state=cfg.RANDOM_STATE,
        tree_method="hist",
        enable_categorical=getattr(cfg, "USE_XGB_NATIVE_CATEGORICAL", True),
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model, X_test, y_test


def plot_shap_all_features(model, X_test, output_path, csv_path):
    booster = model.get_booster()
    dmatrix = xgb.DMatrix(X_test, enable_categorical=True)
    contributions = booster.predict(dmatrix, pred_contribs=True)
    shap_values = contributions[:, :-1]
    mean_abs = np.abs(shap_values).mean(axis=0)
    shap_df = (
        pd.DataFrame({
            "feature": X_test.columns,
            "mean_abs_shap_margin": mean_abs,
        })
        .sort_values("mean_abs_shap_margin", ascending=False)
        .reset_index(drop=True)
    )
    shap_df.to_csv(csv_path, index=False)

    ordered = shap_df.sort_values("mean_abs_shap_margin", ascending=True)
    height = max(7.5, 0.24 * len(ordered) + 1.8)
    fig, ax = plt.subplots(figsize=(8.4, height), dpi=160)
    ax.barh(
        ordered["feature"],
        ordered["mean_abs_shap_margin"],
        color="#345995",
        edgecolor="#1f2f46",
        linewidth=0.35,
    )
    ax.set_xlabel("Mean absolute SHAP contribution, model margin")
    ax.set_title("All-Feature SHAP Summary")
    ax.grid(axis="x", alpha=0.22)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return shap_df, shap_values


def feature_color_values(series):
    if hasattr(series.dtype, "categories") or str(series.dtype) == "category":
        values = series.cat.codes.replace(-1, np.nan).astype(float)
    else:
        values = pd.to_numeric(series, errors="coerce")

    values = values.astype(float)
    if values.notna().sum() == 0:
        return pd.Series(np.nan, index=series.index)

    lower = values.quantile(0.05)
    upper = values.quantile(0.95)
    if pd.isna(lower) or pd.isna(upper) or np.isclose(lower, upper):
        lower = values.min()
        upper = values.max()
    if pd.isna(lower) or pd.isna(upper) or np.isclose(lower, upper):
        normalized = values.copy()
        normalized.loc[values.notna()] = 0.5
        return normalized
    return ((values.clip(lower, upper) - lower) / (upper - lower)).clip(0, 1)


def plot_shap_beeswarm_all_features(shap_df, shap_values, X_test, output_path):
    ordered_features = shap_df.sort_values("mean_abs_shap_margin", ascending=True)["feature"].tolist()
    feature_to_index = {feature: idx for idx, feature in enumerate(X_test.columns)}
    rng = np.random.default_rng(42)
    height = max(8.0, 0.28 * len(ordered_features) + 2.0)
    fig, ax = plt.subplots(figsize=(9.2, height), dpi=170)

    for y_pos, feature in enumerate(ordered_features):
        feature_idx = feature_to_index[feature]
        x = shap_values[:, feature_idx]
        y = y_pos + rng.normal(0, 0.085, size=len(x))
        color_values = feature_color_values(X_test[feature]).to_numpy()
        valid_color = ~np.isnan(color_values)
        if valid_color.any():
            ax.scatter(
                x[valid_color],
                y[valid_color],
                c=color_values[valid_color],
                cmap="coolwarm",
                vmin=0,
                vmax=1,
                s=7,
                alpha=0.45,
                linewidths=0,
                rasterized=True,
            )
        if (~valid_color).any():
            ax.scatter(
                x[~valid_color],
                y[~valid_color],
                color="#9aa0a6",
                s=7,
                alpha=0.30,
                linewidths=0,
                rasterized=True,
            )

    ax.axvline(0, color="#2d3436", linewidth=1.0)
    ax.set_yticks(range(len(ordered_features)))
    ax.set_yticklabels(ordered_features)
    ax.set_xlabel("SHAP contribution to model margin")
    ax.set_title("Directional SHAP Beeswarm, All Features")
    ax.grid(axis="x", alpha=0.22)
    ax.set_axisbelow(True)
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.012, aspect=30)
    cbar.set_label("Feature value")
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def html_escape(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def feature_badges(features):
    return "\n".join(f"<span class=\"badge\">{html_escape(feature)}</span>" for feature in features)


def log_line_containing(report_dir, needle):
    log_path = report_dir / "run_console.log"
    if not log_path.exists():
        return None
    for line in log_path.read_text(errors="ignore").splitlines():
        if needle in line:
            return line.strip()
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Create an HTML presentation with PR/ROC curves and an all-feature SHAP plot."
    )
    parser.add_argument(
        "--report-dir",
        default=None,
        help="Run report directory. Defaults to the latest ecg_complete_medication_runs directory.",
    )
    parser.add_argument("--feature-preset", default="medication_tol0p01")
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()

    base_dir = Path("../data/processed/model_reports/ecg_complete_medication_runs")
    report_dir = Path(args.report_dir) if args.report_dir else latest_run_dir(base_dir)
    run_id = report_dir.name

    summary_path = first_match(report_dir, "_main_summary_metrics.csv")
    cv_path = first_match(report_dir, "_main_cv_metrics.csv")
    threshold_path = first_match(report_dir, "_main_threshold_metrics.csv")
    importance_path = first_match(report_dir, "_main_feature_importance.csv")
    pr_png = report_dir / "ecg_complete_pr_curve.png"
    roc_png = report_dir / "ecg_complete_roc_curve.png"
    if not pr_png.exists() or not roc_png.exists():
        raise FileNotFoundError("Expected PR/ROC PNGs are missing. Run create_ecg_complete_presentation.py first.")

    summary = pd.read_csv(summary_path).iloc[0]
    cv = pd.read_csv(cv_path)
    thresholds = pd.read_csv(threshold_path)
    importance = pd.read_csv(importance_path)
    manifest_path = report_dir / f"{run_id}_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    dropped_features = manifest.get("dropped_features", [])
    use_clinically_similar_controls = manifest.get("use_clinically_similar_controls", True)
    preprune_median_labs = manifest.get("preprune_median_labs", False)
    model_description = (
        "ECG-complete unrestricted-control median-only pre-pruning labs model with 5-fold cross-validation"
        if preprune_median_labs
        else "ECG-complete fixed medication feature model with 5-fold cross-validation"
    )
    feature_setup_text = (
        "Used median-only base labs/vitals/ECG features before recursive pruning; medication features were disabled."
        if preprune_median_labs
        else "Retained the medication tolerance 0.01 feature set as the active fixed feature list."
    )
    control_cohort_text = (
        "Controls were restricted to configured clinically similar ICD diagnosis groups before selecting each patient's most recent eligible admission."
        if use_clinically_similar_controls
        else "Controls were not restricted to the configured clinically similar ICD diagnosis groups; each non-dissection control patient's most recent admission was eligible."
    )
    ecg_filter_line = log_line_containing(report_dir, "ECG-complete cohort filter:")
    active_features_path = report_dir / f"{run_id}_active_features.txt"
    active_features = [
        line.strip()
        for line in active_features_path.read_text().splitlines()
        if line.strip()
    ]

    model, X_test, _ = train_model_for_shap(
        report_dir=report_dir,
        run_id=run_id,
        feature_preset=args.feature_preset,
        cv_folds=args.cv_folds,
        summary=summary,
        active_features=active_features,
        use_clinically_similar_controls=use_clinically_similar_controls,
        preprune_median_labs=preprune_median_labs,
    )
    shap_png = report_dir / "ecg_complete_shap_all_features.png"
    shap_csv = report_dir / "ecg_complete_shap_all_features.csv"
    shap_df, shap_values = plot_shap_all_features(model, X_test, shap_png, shap_csv)
    shap_beeswarm_png = report_dir / "ecg_complete_shap_beeswarm_all_features.png"
    plot_shap_beeswarm_all_features(shap_df, shap_values, X_test, shap_beeswarm_png)

    validation_row = thresholds[thresholds["rule"].eq("validation_f1")].iloc[0]
    default_row = thresholds[thresholds["rule"].eq("default_0.5")].iloc[0]
    top_features = importance.head(8)["Feature"].tolist()
    top_shap_features = shap_df.head(8)["feature"].tolist()

    fold_rows = "\n".join(
        "<tr>"
        f"<td>{int(row.fold)}</td>"
        f"<td>{format_metric(row.average_precision)}</td>"
        f"<td>{format_metric(row.pr_auc_trapezoidal)}</td>"
        f"<td>{format_metric(row.roc_auc)}</td>"
        "</tr>"
        for row in cv.itertuples(index=False)
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Aortic Dissection EHR Model Update</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17202a;
      --muted: #5d6875;
      --line: #d9dee7;
      --blue: #264f78;
      --green: #28724f;
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
    section:first-child {{
      border-top: 0;
    }}
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
    p, li {{
      font-size: 16px;
    }}
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
    }}
    th:first-child, td:first-child {{
      text-align: left;
    }}
    th {{
      color: var(--muted);
      font-weight: 700;
      background: var(--panel);
    }}
    .badge-list {{
      display: flex;
      flex-wrap: wrap;
      gap: 7px;
      margin-top: 12px;
    }}
    .badge {{
      display: inline-block;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 8px;
      background: var(--panel);
      font-size: 13px;
      white-space: nowrap;
    }}
    .note {{
      color: var(--muted);
      font-size: 14px;
    }}
    @media (max-width: 820px) {{
      .grid, .two-col {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
<main>
  <section>
    <h1>Aortic Dissection EHR Model Update</h1>
    <p class="subtitle">{html_escape(model_description)}</p>
    <ul>
      <li>{html_escape(feature_setup_text)}</li>
      <li>Preserved the labs-only pre-medication feature list as a documented revert option.</li>
      <li>{html_escape(control_cohort_text)}</li>
      <li>Removed target and control patients without in-window ECG machine measurements.</li>
      <li>Generated holdout PR/ROC curves plus all-feature directional SHAP beeswarm and summary plots.</li>
    </ul>
  </section>

  <section>
    <h2>ECG-Complete Cohort and Model</h2>
    <div class="grid">
      <div class="metric"><span class="label">Train rows</span><span class="value">{int(summary['train_rows']):,}</span></div>
      <div class="metric"><span class="label">Train positives</span><span class="value">{int(summary['train_positive']):,}</span></div>
      <div class="metric"><span class="label">Holdout rows</span><span class="value">{int(summary['holdout_rows']):,}</span></div>
      <div class="metric"><span class="label">Holdout positives</span><span class="value">{int(summary['holdout_positive']):,}</span></div>
    </div>
    <p>Feature preset: <strong>{html_escape(args.feature_preset)}</strong>; feature count: <strong>{int(summary['feature_count'])}</strong>.</p>
    <p>Removed features: <strong>{", ".join(html_escape(feature) for feature in dropped_features) if dropped_features else "None"}</strong>.</p>
    <p class="note">{html_escape(ecg_filter_line) if ecg_filter_line else "The run log documents the ECG-complete cohort filter."}</p>
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
    <h2>PR-AUC and ROC-AUC Curves</h2>
    <div class="two-col">
      <div>
        <img src="{image_data_uri(pr_png)}" alt="Precision recall curve">
      </div>
      <div>
        <img src="{image_data_uri(roc_png)}" alt="ROC curve">
      </div>
    </div>
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
      <tbody>
        {fold_rows}
      </tbody>
    </table>
  </section>

  <section>
    <h2>Directional SHAP Beeswarm</h2>
    <p class="note">Computed with XGBoost native SHAP contribution values on the ECG-complete holdout set. Points to the right increase predicted risk on the model margin; points to the left decrease it. Color is feature value, normalized within each feature.</p>
    <img src="{image_data_uri(shap_beeswarm_png)}" alt="Directional all-feature SHAP beeswarm">
    <p>Top SHAP features: {", ".join(f"<strong>{html_escape(feature)}</strong>" for feature in top_shap_features)}</p>
  </section>

  <section>
    <h2>All-Feature SHAP Magnitude Summary</h2>
    <p class="note">Mean absolute SHAP contribution values on the ECG-complete holdout set. Values are mean absolute contributions to the model margin.</p>
    <img src="{image_data_uri(shap_png)}" alt="All-feature SHAP summary">
  </section>

  <section>
    <h2>Feature List and Revert Path</h2>
    <p>Top model-importance features: {", ".join(f"<strong>{html_escape(feature)}</strong>" for feature in top_features)}.</p>
    <p>The active feature list and the labs-only revert list are written beside this presentation.</p>
    <div class="badge-list">
      {feature_badges(active_features)}
    </div>
  </section>
</main>
</body>
</html>
"""

    html_path = report_dir / "ecg_complete_model_update.html"
    html_path.write_text(html)
    print(f"HTML presentation: {html_path}")
    print(f"SHAP plot: {shap_png}")
    print(f"SHAP beeswarm: {shap_beeswarm_png}")
    print(f"SHAP values: {shap_csv}")


if __name__ == "__main__":
    main()
