import argparse
import base64
import json
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from engine import ModelEngine
from run_lab_vitals_first_model import (
    Tee,
    build_or_restore_matrix,
    configure,
    feature_color_values,
    first_match,
    format_metric,
    html_escape,
    image_data_uri,
    plot_pr_curve,
    plot_roc_curve,
    plot_shap_beeswarm_all_features,
    write_manifest,
)


CATBOOST_PARAM_GRID = [
    {
        "iterations": 300,
        "depth": 3,
        "learning_rate": 0.05,
        "l2_leaf_reg": 5.0,
    },
    {
        "iterations": 500,
        "depth": 2,
        "learning_rate": 0.03,
        "l2_leaf_reg": 10.0,
    },
]


def cat_feature_names(X):
    return [
        col
        for col in X.columns
        if str(X[col].dtype) in {"category", "object", "string", "str"}
        or hasattr(X[col].dtype, "categories")
    ]


def prepare_catboost_frame(X):
    X_cb = X.copy()
    for col in cat_feature_names(X_cb):
        X_cb[col] = X_cb[col].astype("string").fillna("MISSING").astype(str)
    return X_cb


def make_pool(X, y=None):
    X_cb = prepare_catboost_frame(X)
    cat_names = cat_feature_names(X_cb)
    cat_indices = [X_cb.columns.get_loc(col) for col in cat_names]
    return Pool(X_cb, label=y, cat_features=cat_indices), X_cb, cat_names


def catboost_model(cfg, params, y_fit, fold_offset=0):
    scale_weight = float((y_fit == 0).sum() / (y_fit == 1).sum())
    return CatBoostClassifier(
        **params,
        loss_function="Logloss",
        eval_metric="PRAUC",
        scale_pos_weight=scale_weight,
        random_seed=cfg.RANDOM_STATE + fold_offset,
        allow_writing_files=False,
        verbose=False,
        thread_count=-1,
    )


def run_catboost_cv(cfg, engine, X_train, y_train, params, model_label):
    folds = int(getattr(cfg, "CV_FOLDS", 5))
    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=cfg.RANDOM_STATE,
    )
    rows = []
    print(f"\n=== {folds}-Fold CatBoost Cross-Validation on Training Cohort ===")
    for fold_idx, (fit_idx, val_idx) in enumerate(splitter.split(X_train, y_train), start=1):
        X_fit = X_train.iloc[fit_idx]
        X_val = X_train.iloc[val_idx]
        y_fit = y_train.iloc[fit_idx]
        y_val = y_train.iloc[val_idx]
        fit_pool, _, _ = make_pool(X_fit, y_fit)
        val_pool, _, _ = make_pool(X_val, y_val)
        model = catboost_model(cfg, params, y_fit, fold_offset=fold_idx)
        model.fit(fit_pool)
        val_prob = model.predict_proba(val_pool)[:, 1]
        precision, recall, _ = precision_recall_curve(y_val, val_prob)
        fold_ap = average_precision_score(y_val, val_prob)
        fold_pr_auc = auc(recall, precision)
        fold_roc_auc = roc_auc_score(y_val, val_prob)
        rows.append({
            "run_id": engine.run_id,
            "model_label": model_label,
            "fold": fold_idx,
            "train_rows": int(len(y_fit)),
            "validation_rows": int(len(y_val)),
            "validation_positive": int(y_val.sum()),
            "validation_prevalence": float(y_val.mean()),
            "average_precision": fold_ap,
            "pr_auc_trapezoidal": fold_pr_auc,
            "roc_auc": fold_roc_auc,
        })
        print(
            f"    fold {fold_idx}/{folds}: "
            f"AP={fold_ap:.4f}; PR-AUC={fold_pr_auc:.4f}; ROC-AUC={fold_roc_auc:.4f}"
        )

    cv_metrics = pd.DataFrame(rows)
    print(
        "    CV mean +/- SD: "
        f"AP={cv_metrics['average_precision'].mean():.4f} +/- {cv_metrics['average_precision'].std(ddof=1):.4f}; "
        f"PR-AUC={cv_metrics['pr_auc_trapezoidal'].mean():.4f} +/- {cv_metrics['pr_auc_trapezoidal'].std(ddof=1):.4f}; "
        f"ROC-AUC={cv_metrics['roc_auc'].mean():.4f} +/- {cv_metrics['roc_auc'].std(ddof=1):.4f}"
    )

    if getattr(cfg, "WRITE_MODEL_REPORTS", True):
        engine.report_dir.mkdir(parents=True, exist_ok=True)
        path = engine.report_dir / f"{engine.run_id}_{model_label}_cv_metrics.csv"
        cv_metrics.to_csv(path, index=False)
        print(f"    CV metrics: {path}")
    return cv_metrics


def plot_catboost_shap_all_features(model, pool, X_test, output_path, csv_path):
    contributions = model.get_feature_importance(pool, type="ShapValues")
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
    ax.set_title("CatBoost All-Feature SHAP Summary")
    ax.grid(axis="x", alpha=0.22)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return shap_df, shap_values


def create_catboost_html(report_dir, run_id, lab_prevalence, cfg, shap_df):
    summary = pd.read_csv(first_match(report_dir, "_main_summary_metrics.csv")).iloc[0]
    cv = pd.read_csv(first_match(report_dir, "_main_cv_metrics.csv"))
    thresholds = pd.read_csv(first_match(report_dir, "_main_threshold_metrics.csv"))
    importance = pd.read_csv(first_match(report_dir, "_main_feature_importance.csv"))
    pr_path = first_match(report_dir, "_main_pr_curve.csv")
    roc_path = first_match(report_dir, "_main_roc_curve.csv")

    pr_png = report_dir / "lab_vitals_ecg_first_catboost_pr_curve.png"
    roc_png = report_dir / "lab_vitals_ecg_first_catboost_roc_curve.png"
    plot_pr_curve(pr_path, summary, pr_png)
    plot_roc_curve(roc_path, summary, roc_png)

    validation_row = thresholds[thresholds["rule"].eq("validation_f1")].iloc[0]
    default_row = thresholds[thresholds["rule"].eq("default_0.5")].iloc[0]
    top_features = importance.head(12)["Feature"].tolist()
    top_shap_features = shap_df.head(12)["feature"].tolist()

    fold_rows = "\n".join(
        "<tr>"
        f"<td>{int(row.fold)}</td>"
        f"<td>{format_metric(row.average_precision)}</td>"
        f"<td>{format_metric(row.pr_auc_trapezoidal)}</td>"
        f"<td>{format_metric(row.roc_auc)}</td>"
        "</tr>"
        for row in cv.itertuples(index=False)
    )
    prevalence_rows = "\n".join(
        "<tr>"
        f"<td>{int(row.prevalence_rank)}</td>"
        f"<td>{html_escape(row.label)}</td>"
        f"<td>{int(row.itemid)}</td>"
        f"<td>{html_escape(row.model_column)}</td>"
        f"<td>{row.control_prevalence:.1%}</td>"
        f"<td>{row.target_prevalence:.1%}</td>"
        f"<td>{row.overall_prevalence:.1%}</td>"
        "</tr>"
        for row in lab_prevalence.head(20).itertuples(index=False)
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Aortic Dissection CatBoost Lab/Vitals/ECG Model</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17202a;
      --muted: #5d6875;
      --line: #d9dee7;
      --blue: #264f78;
      --panel: #f7f9fc;
    }}
    body {{ margin: 0; font-family: Arial, Helvetica, sans-serif; color: var(--ink); background: #fff; line-height: 1.45; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 28px 26px 56px; }}
    section {{ border-top: 1px solid var(--line); padding: 28px 0; page-break-after: always; }}
    section:first-child {{ border-top: 0; }}
    h1 {{ font-size: 34px; margin: 0 0 8px; letter-spacing: 0; }}
    h2 {{ font-size: 24px; margin: 0 0 14px; color: var(--blue); letter-spacing: 0; }}
    p, li {{ font-size: 16px; }}
    .subtitle {{ color: var(--muted); font-size: 17px; margin: 0 0 20px; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 18px 0; }}
    .metric {{ background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 13px 14px; }}
    .metric .label {{ display: block; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }}
    .metric .value {{ display: block; font-size: 25px; font-weight: 700; margin-top: 3px; }}
    .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; align-items: start; }}
    img {{ max-width: 100%; height: auto; border: 1px solid var(--line); border-radius: 6px; background: white; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px 10px; text-align: right; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2), th:nth-child(4), td:nth-child(4) {{ text-align: left; }}
    th {{ color: var(--muted); font-weight: 700; background: var(--panel); }}
    .note {{ color: var(--muted); font-size: 14px; }}
    @media (max-width: 820px) {{ .grid, .two-col {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
<main>
  <section>
    <h1>Aortic Dissection CatBoost Model</h1>
    <p class="subtitle">Same ECG-complete first-value lab/vitals/ECG feature matrix as the full XGBoost run; CatBoost classifier; top-50 prevalent numeric labs; no medications.</p>
    <ul>
      <li>Controls are restricted to chest pain and back pain ICD-coded candidate admissions.</li>
      <li>The cohort is restricted to patients with at least one ECG machine-measurement row in the 24-hour feature window.</li>
      <li>Features use first values in the 24-hour window; no features were manually dropped.</li>
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
    <p>Feature count after preprocessing: <strong>{int(summary['feature_count'])}</strong>. Aggregation mode: <strong>{html_escape(summary['feature_aggregation_mode'])}</strong>.</p>
    <p>Top CatBoost feature-importance features: {", ".join(f"<strong>{html_escape(feature)}</strong>" for feature in top_features)}.</p>
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
    <h2>PR and ROC Curves</h2>
    <div class="two-col">
      <div><img src="{image_data_uri(pr_png)}" alt="Precision recall curve"></div>
      <div><img src="{image_data_uri(roc_png)}" alt="ROC curve"></div>
    </div>
  </section>

  <section>
    <h2>Directional SHAP Beeswarm</h2>
    <p class="note">Computed with CatBoost native SHAP contribution values on the ECG-complete holdout set.</p>
    <img src="{image_data_uri(report_dir / 'lab_vitals_ecg_first_catboost_shap_beeswarm_all_features.png')}" alt="Directional all-feature SHAP beeswarm">
    <p>Top SHAP features: {", ".join(f"<strong>{html_escape(feature)}</strong>" for feature in top_shap_features)}.</p>
  </section>

  <section>
    <h2>All-Feature SHAP Magnitude</h2>
    <img src="{image_data_uri(report_dir / 'lab_vitals_ecg_first_catboost_shap_all_features.png')}" alt="All-feature SHAP summary">
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

  <section>
    <h2>Top Lab Prevalence</h2>
    <p class="note">Prevalence means at least one numeric lab result in the model feature window. Full top-50 table is written as CSV beside this HTML file.</p>
    <table>
      <thead><tr><th>Rank</th><th>Lab</th><th>ItemID</th><th>Model column</th><th>Control</th><th>Target</th><th>Overall</th></tr></thead>
      <tbody>{prevalence_rows}</tbody>
    </table>
  </section>
</main>
</body>
</html>
"""
    html_path = report_dir / "lab_vitals_ecg_first_catboost_model_update.html"
    html_path.write_text(html)
    print(f"HTML presentation: {html_path}")
    print(f"PR curve: {pr_png}")
    print(f"ROC curve: {roc_png}")


def update_manifest_for_catboost(report_dir, run_id, cfg, params):
    manifest_path = report_dir / f"{run_id}_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["model_type"] = "catboost"
    manifest["catboost_param_grid"] = CATBOOST_PARAM_GRID
    manifest["best_catboost_params"] = params
    manifest["features_to_drop"] = list(getattr(cfg, "FEATURES_TO_DROP", []))
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def train_and_eval_catboost(cfg, engine, X_train, y_train, X_test, y_test):
    model_label = "main"
    print("\n=== CatBoost Training & Evaluation ===")
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=cfg.VALIDATION_SIZE,
        random_state=cfg.RANDOM_STATE,
        stratify=y_train,
    )
    best_params = None
    best_val_prob = None
    best_ap = -1
    candidate_rows = []

    print(f"    Trying {len(CATBOOST_PARAM_GRID)} CatBoost parameter set(s) on the training validation split...")
    for idx, params in enumerate(CATBOOST_PARAM_GRID, start=1):
        fit_pool, _, _ = make_pool(X_fit, y_fit)
        val_pool, _, _ = make_pool(X_val, y_val)
        model = catboost_model(cfg, params, y_fit)
        model.fit(fit_pool)
        val_prob = model.predict_proba(val_pool)[:, 1]
        val_ap = average_precision_score(y_val, val_prob)
        print(f"      candidate {idx}: validation average precision={val_ap:.4f}; params={params}")
        candidate_rows.append({
            "model_label": model_label,
            "candidate": idx,
            "validation_average_precision": val_ap,
            "params_json": json.dumps(params, sort_keys=True),
        })
        if val_ap > best_ap:
            best_ap = val_ap
            best_params = params
            best_val_prob = val_prob

    threshold = engine._select_threshold(y_val, best_val_prob)
    print(f"    Selected threshold on validation split: {threshold:.4f}")

    train_pool, X_train_cb, cat_features = make_pool(X_train, y_train)
    test_pool, X_test_cb, _ = make_pool(X_test, y_test)
    model = catboost_model(cfg, best_params, y_train)
    model.fit(train_pool)
    y_prob = model.predict_proba(test_pool)[:, 1]

    print("\n1. Holdout Evaluation at Tuned Threshold:")
    engine._print_threshold_metrics(y_test, y_prob, threshold)
    print("\n2. Holdout Evaluation at Default 0.5000 Threshold:")
    engine._print_threshold_metrics(y_test, y_prob, 0.5)
    print("\n3. Holdout Threshold Operating Points:")
    operating_points = engine._print_threshold_operating_points(y_test, y_prob, threshold)

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    holdout_ap = average_precision_score(y_test, y_prob)
    holdout_pr_auc = auc(recall, precision)
    holdout_roc_auc = roc_auc_score(y_test, y_prob)
    print(f"\n4. Holdout Average Precision: {holdout_ap:.4f}")
    print(f"   Holdout PR-AUC (trapezoidal): {holdout_pr_auc:.4f}")
    print(f"   Holdout ROC-AUC: {holdout_roc_auc:.4f}")
    print(f"   Best validation average precision: {best_ap:.4f}")
    print(f"   Best CatBoost params: {best_params}")

    cv_metrics = run_catboost_cv(cfg, engine, X_train, y_train, best_params, model_label)
    importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": model.get_feature_importance(train_pool, type="FeatureImportance"),
    }).sort_values("Importance", ascending=False)
    print("\n5. Top 25 Features:")
    print(importance.head(25).to_string(index=False))
    engine._print_low_importance_features(importance)

    engine._write_model_reports(
        model_label=model_label,
        importance_df=importance,
        operating_points=operating_points,
        summary_metrics=engine._summary_metrics(
            model_label=model_label,
            y_train=y_train,
            y_test=y_test,
            threshold=threshold,
            best_ap=best_ap,
            best_params=best_params,
            holdout_ap=holdout_ap,
            holdout_pr_auc=holdout_pr_auc,
            holdout_roc_auc=holdout_roc_auc,
            operating_points=operating_points,
            feature_count=X_train.shape[1],
            rfe_enabled=False,
            cv_metrics=cv_metrics,
        ),
        candidate_metrics=pd.DataFrame(candidate_rows),
    )
    engine._write_curve_reports(model_label, y_test, y_prob)

    shap_png = engine.report_dir / "lab_vitals_ecg_first_catboost_shap_all_features.png"
    shap_csv = engine.report_dir / "lab_vitals_ecg_first_catboost_shap_all_features.csv"
    shap_df, shap_values = plot_catboost_shap_all_features(model, test_pool, X_test, shap_png, shap_csv)
    beeswarm_png = engine.report_dir / "lab_vitals_ecg_first_catboost_shap_beeswarm_all_features.png"
    plot_shap_beeswarm_all_features(shap_df, shap_values, X_test_cb, beeswarm_png)
    print(f"SHAP plot: {shap_png}")
    print(f"SHAP beeswarm: {beeswarm_png}")
    print(f"SHAP values: {shap_csv}")
    print(f"CatBoost categorical features: {cat_features}")
    return best_params, shap_df


def run(args, report_dir):
    cfg = configure(report_dir, drop_features=args.drop_features)
    matrix_path, lab_prevalence = build_or_restore_matrix(
        cfg=cfg,
        report_dir=report_dir,
        run_id=args.run_id,
        top_n=args.top_labs,
    )
    write_manifest(report_dir, args.run_id, cfg, args.top_labs, lab_prevalence)

    engine = ModelEngine(cfg)
    engine.run_id = f"{args.run_id}_lab_vitals_ecg_first_catboost"
    X_train, y_train, X_test, y_test = engine.preprocess(matrix_path)
    active_features = list(X_train.columns)
    (report_dir / f"{args.run_id}_active_features.txt").write_text(
        "\n".join(active_features) + "\n"
    )
    write_manifest(
        report_dir,
        args.run_id,
        cfg,
        args.top_labs,
        lab_prevalence,
        active_features=active_features,
    )
    best_params, shap_df = train_and_eval_catboost(cfg, engine, X_train, y_train, X_test, y_test)
    update_manifest_for_catboost(report_dir, args.run_id, cfg, best_params)
    create_catboost_html(report_dir, engine.run_id, lab_prevalence, cfg, shap_df)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run a CatBoost first-value labs plus vitals plus ECG model using the "
            "same ECG-complete cohort and feature matrix as run_lab_vitals_first_model.py."
        )
    )
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--top-labs", type=int, default=50)
    parser.add_argument(
        "--drop-features",
        nargs="*",
        default=[],
        help="Feature columns to drop after preprocessing and before model training.",
    )
    args = parser.parse_args()

    report_dir = Path("../data/processed/model_reports/lab_vitals_ecg_first_catboost_runs") / args.run_id
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
