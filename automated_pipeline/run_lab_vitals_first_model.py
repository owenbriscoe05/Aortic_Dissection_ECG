import argparse
import base64
import json
import re
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cache import MatrixCache
from config import PipelineConfig
from engine import DataBuilder, ModelEngine


CHEST_BACK_CONTROL_GROUPS = {
    "chest_pain": {
        9: ["7865"],
        10: ["R07"],
    },
    "back_pain": {
        9: ["7245"],
        10: ["M54"],
    },
}

ORDERING_SENSITIVE_LAB_ITEMIDS = {
    50915,  # D-Dimer
    51003,  # Troponin T
    51002,  # Troponin I
    50889,  # C-Reactive Protein
    51214,  # Fibrinogen
}


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def configure(report_dir, drop_features=None):
    cfg = PipelineConfig()
    cfg.MODEL_REPORT_DIR = report_dir
    cfg.USE_CLINICALLY_SIMILAR_CONTROLS = True
    cfg.CLINICALLY_SIMILAR_CONTROL_GROUPS = CHEST_BACK_CONTROL_GROUPS
    cfg.FEATURE_AGGREGATION_MODE = "first_only"
    cfg.INCLUDE_FIRST_FEATURES = True
    cfg.USE_ECG_FEATURES = True
    cfg.USE_MEDICATION_FEATURES = False
    cfg.REQUIRE_ECG_MEASUREMENTS = True
    cfg.ACTIVE_FEATURE_SET_PRESET = None
    cfg.FEATURES_TO_KEEP = []
    cfg.FEATURES_TO_DROP = list(drop_features or [])
    cfg.USE_RECURSIVE_FEATURE_ELIMINATION = False
    cfg.RUN_SHADOW_FEATURE_FILTER_REPORT = False
    cfg.RUN_SHADOW_REJECT_COMPARISON = False
    cfg.RUN_INR_ABLATION_COMPARISON = False
    cfg.RUN_CROSS_VALIDATION = True
    cfg.CV_FOLDS = 5
    cfg.COL_MISSINGNESS_THRESHOLD = 0.999
    return cfg


def sanitize_feature_name(label, itemid, used_names):
    base = str(label).lower()
    base = re.sub(r"[^a-z0-9]+", "_", base).strip("_")
    base = re.sub(r"_+", "_", base)
    if not base:
        base = f"lab_{itemid}"
    if base[0].isdigit():
        base = f"lab_{base}"

    feature_name = base
    if feature_name in used_names:
        feature_name = f"{base}_{itemid}"
    used_names.add(feature_name)
    return feature_name


def load_lab_labels(cfg):
    d_labitems_path = cfg.DATA_DIR / "hosp" / "d_labitems.csv"
    labels = pd.read_csv(
        d_labitems_path,
        usecols=["itemid", "label", "fluid", "category"],
    )
    labels["itemid"] = pd.to_numeric(labels["itemid"], errors="coerce").astype("Int64")
    labels.dropna(subset=["itemid"], inplace=True)
    labels["itemid"] = labels["itemid"].astype("int64")
    return labels


def select_top_labs(cfg, spine_path, top_n):
    print(f"\nSelecting top {top_n} most prevalent in-window numeric labs...")
    spine = pd.read_csv(
        spine_path,
        usecols=[
            "subject_id",
            "index_time",
            "feature_window_end",
            "is_aortic_dissection",
        ],
    )
    spine["index_time"] = pd.to_datetime(spine["index_time"])
    spine["feature_window_end"] = pd.to_datetime(
        spine["feature_window_end"],
        errors="coerce",
    )
    default_end = spine["index_time"] + pd.to_timedelta(cfg.DAY_0_WINDOW_HOURS, unit="h")
    spine["feature_window_end"] = spine["feature_window_end"].fillna(default_end)

    subject_ids = set(spine["subject_id"])
    spine_times = spine[["subject_id", "index_time", "feature_window_end"]].copy()
    label_by_subject = spine[["subject_id", "is_aortic_dissection"]].copy()
    total_patients = int(spine["subject_id"].nunique())
    total_targets = int((spine["is_aortic_dissection"] == 1).sum())
    total_controls = int((spine["is_aortic_dissection"] == 0).sum())

    observed_pairs = []
    usecols = ["subject_id", "charttime", "itemid", "valuenum"]
    chunksize = getattr(cfg, "EVENT_CACHE_CHUNKSIZE", 3_000_000)
    for chunk_number, chunk in enumerate(
        pd.read_csv(cfg.DATA_DIR / "hosp" / "labevents.csv", usecols=usecols, chunksize=chunksize),
        start=1,
    ):
        chunk = chunk[chunk["subject_id"].isin(subject_ids)].copy()
        if chunk.empty:
            if chunk_number % 10 == 0:
                print(f"      scanned {chunk_number:,} lab chunks; no cohort rows in latest chunk")
            continue

        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk["valuenum"] = pd.to_numeric(chunk["valuenum"], errors="coerce")
        chunk.dropna(subset=["subject_id", "charttime", "itemid", "valuenum"], inplace=True)
        if chunk.empty:
            continue

        chunk = chunk.merge(spine_times, on="subject_id", how="inner")
        chunk = chunk[
            (chunk["charttime"] >= chunk["index_time"])
            & (chunk["charttime"] <= chunk["feature_window_end"])
        ]
        if not chunk.empty:
            observed_pairs.append(chunk[["subject_id", "itemid"]].drop_duplicates())

        if chunk_number % 10 == 0:
            observed_count = sum(len(pairs) for pairs in observed_pairs)
            print(
                f"      scanned {chunk_number:,} lab chunks; "
                f"accumulated {observed_count:,} subject-lab observations"
            )

    if not observed_pairs:
        raise ValueError("No in-window numeric labs were found for the configured cohort.")

    observed = pd.concat(observed_pairs, ignore_index=True).drop_duplicates()
    observed["itemid"] = pd.to_numeric(observed["itemid"], errors="coerce").astype("Int64")
    observed.dropna(subset=["itemid"], inplace=True)
    observed["itemid"] = observed["itemid"].astype("int64")
    observed = observed.merge(label_by_subject, on="subject_id", how="inner")

    overall_counts = observed.groupby("itemid")["subject_id"].nunique()
    target_counts = (
        observed[observed["is_aortic_dissection"] == 1]
        .groupby("itemid")["subject_id"]
        .nunique()
    )
    control_counts = (
        observed[observed["is_aortic_dissection"] == 0]
        .groupby("itemid")["subject_id"]
        .nunique()
    )

    prevalence = (
        pd.DataFrame({"observed_patients": overall_counts})
        .join(target_counts.rename("target_observed_patients"), how="left")
        .join(control_counts.rename("control_observed_patients"), how="left")
        .fillna(0)
        .reset_index()
    )
    count_cols = [
        "observed_patients",
        "target_observed_patients",
        "control_observed_patients",
    ]
    prevalence[count_cols] = prevalence[count_cols].astype(int)
    prevalence["overall_prevalence"] = prevalence["observed_patients"] / total_patients
    prevalence["target_prevalence"] = prevalence["target_observed_patients"] / total_targets
    prevalence["control_prevalence"] = prevalence["control_observed_patients"] / total_controls
    prevalence["total_patients"] = total_patients
    prevalence["total_targets"] = total_targets
    prevalence["total_controls"] = total_controls

    lab_labels = load_lab_labels(cfg)
    prevalence = prevalence.merge(lab_labels, on="itemid", how="left")
    prevalence["label"] = prevalence["label"].fillna(prevalence["itemid"].map(lambda itemid: f"itemid_{itemid}"))
    prevalence["fluid"] = prevalence["fluid"].fillna("")
    prevalence["category"] = prevalence["category"].fillna("")
    prevalence.sort_values(
        ["overall_prevalence", "observed_patients", "itemid"],
        ascending=[False, False, True],
        inplace=True,
    )
    prevalence["prevalence_rank"] = range(1, len(prevalence) + 1)
    selected = prevalence.head(top_n).copy()

    used_names = set()
    selected["feature_name"] = [
        sanitize_feature_name(label, itemid, used_names)
        for label, itemid in zip(selected["label"], selected["itemid"])
    ]
    selected["model_column"] = selected["feature_name"] + "_first"
    selected["ordering_sensitive"] = selected["itemid"].isin(ORDERING_SENSITIVE_LAB_ITEMIDS)

    lab_dict = dict(zip(selected["itemid"], selected["feature_name"]))
    print(
        f"    Selected {len(lab_dict):,} labs from {len(prevalence):,} observed numeric lab itemids."
    )
    print(
        "    Most prevalent selected labs: "
        + ", ".join(selected.head(10)["feature_name"].tolist())
    )
    return lab_dict, selected


def build_or_restore_matrix(cfg, report_dir, run_id, top_n):
    builder = DataBuilder(cfg)
    temp_spine_path = builder.build_spine_and_demographics()
    lab_dict, lab_prevalence = select_top_labs(cfg, temp_spine_path, top_n)
    cfg.LABS_DICT = lab_dict

    first_leakage_cols = [
        f"{lab_dict[itemid]}_first"
        for itemid in ORDERING_SENSITIVE_LAB_ITEMIDS
        if itemid in lab_dict
    ]
    cfg.LEAKAGE_LABS_TO_DROP = sorted(set(cfg.LEAKAGE_LABS_TO_DROP + first_leakage_cols))

    prevalence_path = report_dir / f"{run_id}_top{top_n}_lab_prevalence_by_label.csv"
    lab_prevalence.to_csv(prevalence_path, index=False)
    active_labs_path = report_dir / f"{run_id}_selected_lab_features.txt"
    active_labs_path.write_text("\n".join(lab_prevalence["model_column"].tolist()) + "\n")
    print(f"    Lab prevalence report: {prevalence_path}")
    print(f"    Selected lab features: {active_labs_path}")

    matrix_cache = MatrixCache(cfg)
    cached_matrix_path = matrix_cache.restore_raw_matrix()
    if cached_matrix_path is not None:
        Path(temp_spine_path).unlink(missing_ok=True)
        return cached_matrix_path, lab_prevalence

    temp_vitals_path = builder.add_vitals(temp_spine_path)
    # Top-prevalence labs are common enough that caching all matching MIMIC rows
    # creates a very large SQLite table. Stream labs for this cohort instead.
    builder.event_cache = None
    temp_matrix_path = builder.add_labs(temp_vitals_path)
    temp_matrix_path = builder.add_ecg_features(temp_matrix_path)
    matrix_cache.store_raw_matrix(temp_matrix_path)
    return temp_matrix_path, lab_prevalence


def first_match(report_dir, suffix):
    matches = sorted(report_dir.glob(f"*{suffix}"))
    if not matches:
        raise FileNotFoundError(f"No artifact matching *{suffix} in {report_dir}")
    return matches[0]


def plot_pr_curve(pr_path, summary, output_path):
    df = pd.read_csv(pr_path)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=160)
    ax.plot(df["recall"], df["precision"], color="#1f77b4", linewidth=2.2)
    prevalence = summary["holdout_prevalence"]
    ax.axhline(
        prevalence,
        color="#777777",
        linestyle="--",
        linewidth=1,
        label=f"Prevalence {prevalence:.2%}",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(
        f"Precision-Recall Curve (AP {summary['holdout_average_precision']:.3f}; "
        f"PR-AUC {summary['holdout_pr_auc_trapezoidal']:.3f})"
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_roc_curve(roc_path, summary, output_path):
    df = pd.read_csv(roc_path)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=160)
    ax.plot(
        df["false_positive_rate"],
        df["true_positive_rate"],
        color="#2ca02c",
        linewidth=2.2,
    )
    ax.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve (AUC {summary['holdout_roc_auc']:.3f})")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


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


def train_model_for_shap(cfg, X_train, y_train, summary):
    import xgboost as xgb

    best_params = json.loads(summary["best_params_json"])
    scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = xgb.XGBClassifier(
        **best_params,
        scale_pos_weight=scale_weight,
        eval_metric=cfg.XGB_EVAL_METRIC,
        random_state=cfg.RANDOM_STATE,
        tree_method="hist",
        enable_categorical=getattr(cfg, "USE_XGB_NATIVE_CATEGORICAL", True),
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def plot_shap_all_features(model, X_test, output_path, csv_path):
    import xgboost as xgb

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


def image_data_uri(path):
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def html_escape(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def format_metric(value, digits=3):
    return f"{float(value):.{digits}f}"


def core_vitals_complete_mask(X):
    required_cols = [
        "heart_rate_first",
        "resp_rate_first",
        "nibp_sys_first",
        "nibp_dias_first",
        "abp_sys_first",
        "abp_dias_first",
    ]
    missing_cols = [col for col in required_cols if col not in X.columns]
    if missing_cols:
        raise ValueError(
            "Core vitals completeness filter cannot run because these features are missing: "
            f"{missing_cols}"
        )

    has_heart_rate = X["heart_rate_first"].notna()
    has_resp_rate = X["resp_rate_first"].notna()
    has_nibp = X["nibp_sys_first"].notna() & X["nibp_dias_first"].notna()
    has_abp = X["abp_sys_first"].notna() & X["abp_dias_first"].notna()
    return has_heart_rate & has_resp_rate & (has_nibp | has_abp)


def apply_core_vitals_completeness_filter(X_train, y_train, X_test, y_test):
    train_mask = core_vitals_complete_mask(X_train)
    test_mask = core_vitals_complete_mask(X_test)
    stats = {
        "definition": (
            "Require heart_rate_first, resp_rate_first, and either complete "
            "nibp_sys_first/nibp_dias_first or complete abp_sys_first/abp_dias_first."
        ),
        "train_rows_before": int(len(X_train)),
        "train_positive_before": int(y_train.sum()),
        "holdout_rows_before": int(len(X_test)),
        "holdout_positive_before": int(y_test.sum()),
        "train_rows_after": int(train_mask.sum()),
        "train_positive_after": int(y_train.loc[train_mask].sum()),
        "holdout_rows_after": int(test_mask.sum()),
        "holdout_positive_after": int(y_test.loc[test_mask].sum()),
    }
    stats["train_rows_removed"] = stats["train_rows_before"] - stats["train_rows_after"]
    stats["train_positive_removed"] = stats["train_positive_before"] - stats["train_positive_after"]
    stats["holdout_rows_removed"] = stats["holdout_rows_before"] - stats["holdout_rows_after"]
    stats["holdout_positive_removed"] = stats["holdout_positive_before"] - stats["holdout_positive_after"]
    print(
        "    Experimental core-vitals completeness filter: "
        f"train kept {stats['train_rows_after']:,}/{stats['train_rows_before']:,} rows "
        f"({stats['train_positive_after']:,}/{stats['train_positive_before']:,} targets); "
        f"holdout kept {stats['holdout_rows_after']:,}/{stats['holdout_rows_before']:,} rows "
        f"({stats['holdout_positive_after']:,}/{stats['holdout_positive_before']:,} targets)."
    )
    if stats["train_positive_after"] == 0 or stats["holdout_positive_after"] == 0:
        raise ValueError("Core vitals completeness filter removed all positives from train or holdout.")

    return (
        X_train.loc[train_mask].copy(),
        y_train.loc[train_mask].copy(),
        X_test.loc[test_mask].copy(),
        y_test.loc[test_mask].copy(),
        stats,
    )


def heart_rate_resp_complete_mask(X):
    required_cols = ["heart_rate_first", "resp_rate_first"]
    missing_cols = [col for col in required_cols if col not in X.columns]
    if missing_cols:
        raise ValueError(
            "Heart-rate/respiratory-rate completeness filter cannot run because these "
            f"features are missing: {missing_cols}"
        )
    return X["heart_rate_first"].notna() & X["resp_rate_first"].notna()


def apply_heart_rate_resp_completeness_filter(X_train, y_train, X_test, y_test):
    train_mask = heart_rate_resp_complete_mask(X_train)
    test_mask = heart_rate_resp_complete_mask(X_test)
    stats = {
        "definition": "Require heart_rate_first and resp_rate_first; BP features may be excluded from modeling.",
        "train_rows_before": int(len(X_train)),
        "train_positive_before": int(y_train.sum()),
        "holdout_rows_before": int(len(X_test)),
        "holdout_positive_before": int(y_test.sum()),
        "train_rows_after": int(train_mask.sum()),
        "train_positive_after": int(y_train.loc[train_mask].sum()),
        "holdout_rows_after": int(test_mask.sum()),
        "holdout_positive_after": int(y_test.loc[test_mask].sum()),
    }
    stats["train_rows_removed"] = stats["train_rows_before"] - stats["train_rows_after"]
    stats["train_positive_removed"] = stats["train_positive_before"] - stats["train_positive_after"]
    stats["holdout_rows_removed"] = stats["holdout_rows_before"] - stats["holdout_rows_after"]
    stats["holdout_positive_removed"] = stats["holdout_positive_before"] - stats["holdout_positive_after"]
    print(
        "    Experimental heart-rate/respiratory-rate completeness filter: "
        f"train kept {stats['train_rows_after']:,}/{stats['train_rows_before']:,} rows "
        f"({stats['train_positive_after']:,}/{stats['train_positive_before']:,} targets); "
        f"holdout kept {stats['holdout_rows_after']:,}/{stats['holdout_rows_before']:,} rows "
        f"({stats['holdout_positive_after']:,}/{stats['holdout_positive_before']:,} targets)."
    )
    if stats["train_positive_after"] == 0 or stats["holdout_positive_after"] == 0:
        raise ValueError("Heart-rate/respiratory-rate filter removed all positives from train or holdout.")

    return (
        X_train.loc[train_mask].copy(),
        y_train.loc[train_mask].copy(),
        X_test.loc[test_mask].copy(),
        y_test.loc[test_mask].copy(),
        stats,
    )


def create_html_presentation(
    report_dir,
    run_id,
    lab_prevalence,
    cfg,
    X_train,
    y_train,
    X_test,
    core_vitals_filter_stats=None,
):
    summary = pd.read_csv(first_match(report_dir, "_main_summary_metrics.csv")).iloc[0]
    cv_path = first_match(report_dir, "_main_cv_metrics.csv")
    threshold_path = first_match(report_dir, "_main_threshold_metrics.csv")
    pr_path = first_match(report_dir, "_main_pr_curve.csv")
    roc_path = first_match(report_dir, "_main_roc_curve.csv")
    importance_path = first_match(report_dir, "_main_feature_importance.csv")

    cv = pd.read_csv(cv_path)
    thresholds = pd.read_csv(threshold_path)
    importance = pd.read_csv(importance_path)
    pr_png = report_dir / "lab_vitals_ecg_first_pr_curve.png"
    roc_png = report_dir / "lab_vitals_ecg_first_roc_curve.png"
    plot_pr_curve(pr_path, summary, pr_png)
    plot_roc_curve(roc_path, summary, roc_png)
    shap_model = train_model_for_shap(cfg, X_train, y_train, summary)
    shap_png = report_dir / "lab_vitals_ecg_first_shap_all_features.png"
    shap_csv = report_dir / "lab_vitals_ecg_first_shap_all_features.csv"
    shap_df, shap_values = plot_shap_all_features(shap_model, X_test, shap_png, shap_csv)
    shap_beeswarm_png = report_dir / "lab_vitals_ecg_first_shap_beeswarm_all_features.png"
    plot_shap_beeswarm_all_features(shap_df, shap_values, X_test, shap_beeswarm_png)

    validation_row = thresholds[thresholds["rule"].eq("validation_f1")].iloc[0]
    default_row = thresholds[thresholds["rule"].eq("default_0.5")].iloc[0]
    top_features = importance.head(12)["Feature"].tolist()
    top_shap_features = shap_df.head(12)["feature"].tolist()
    dropped_feature_items = "\n".join(
        f"<li>Dropped feature for this run: <strong>{html_escape(feature)}</strong></li>"
        for feature in getattr(cfg, "FEATURES_TO_DROP", [])
    )
    dropped_feature_block = (
        f"<ul>{dropped_feature_items}</ul>"
        if dropped_feature_items
        else "<p>No configured features were dropped before training.</p>"
    )
    if core_vitals_filter_stats:
        core_vitals_filter_block = f"""
    <p><strong>Experimental core-vitals completeness filter:</strong> {html_escape(core_vitals_filter_stats['definition'])}</p>
    <ul>
      <li>Train retained {core_vitals_filter_stats['train_rows_after']:,}/{core_vitals_filter_stats['train_rows_before']:,} rows and {core_vitals_filter_stats['train_positive_after']:,}/{core_vitals_filter_stats['train_positive_before']:,} targets.</li>
      <li>Holdout retained {core_vitals_filter_stats['holdout_rows_after']:,}/{core_vitals_filter_stats['holdout_rows_before']:,} rows and {core_vitals_filter_stats['holdout_positive_after']:,}/{core_vitals_filter_stats['holdout_positive_before']:,} targets.</li>
    </ul>"""
    else:
        core_vitals_filter_block = ""

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
  <title>Aortic Dissection Lab/Vitals/ECG First-Value Model</title>
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
    }}
    th:first-child, td:first-child,
    th:nth-child(2), td:nth-child(2),
    th:nth-child(4), td:nth-child(4) {{
      text-align: left;
    }}
    th {{
      color: var(--muted);
      font-weight: 700;
      background: var(--panel);
    }}
    .note {{
      color: var(--muted);
      font-size: 14px;
    }}
    @media (max-width: 820px) {{
      .grid, .two-col {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
<main>
  <section>
    <h1>Aortic Dissection Lab/Vitals/ECG Model</h1>
    <p class="subtitle">First in-window lab, vital, and ECG machine-measurement values; top-50 prevalent numeric labs; chest/back-pain ICD controls.</p>
    <ul>
      <li>Controls are restricted to chest pain and back pain ICD-coded candidate admissions, excluding exact aortic dissection codes.</li>
      <li>Features use first values in the 24-hour window from index time; medication features are disabled.</li>
      <li>The cohort is restricted to patients with at least one ECG machine-measurement row in the 24-hour feature window.</li>
      <li>The selected top-50 lab prevalence report includes separate target and control prevalence.</li>
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
    <p>Feature count after preprocessing: <strong>{int(summary['feature_count'])}</strong>. Aggregation mode: <strong>first_only</strong>.</p>
    {dropped_feature_block}
    {core_vitals_filter_block}
    <p>Top model-importance features: {", ".join(f"<strong>{html_escape(feature)}</strong>" for feature in top_features)}.</p>
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
    html_path = report_dir / "lab_vitals_ecg_first_model_update.html"
    html_path.write_text(html)
    print(f"HTML presentation: {html_path}")
    print(f"PR curve: {pr_png}")
    print(f"ROC curve: {roc_png}")
    print(f"SHAP plot: {shap_png}")
    print(f"SHAP beeswarm: {shap_beeswarm_png}")
    print(f"SHAP values: {shap_csv}")


def write_manifest(
    report_dir,
    run_id,
    cfg,
    top_n,
    lab_prevalence,
    active_features=None,
    core_vitals_filter_stats=None,
):
    manifest = {
        "run_id": run_id,
        "method": "top_prevalent_labs_plus_vitals_plus_ecg_first_only",
        "top_lab_count_requested": top_n,
        "top_lab_count_selected": int(len(lab_prevalence)),
        "selected_lab_itemids": [int(itemid) for itemid in lab_prevalence["itemid"].tolist()],
        "selected_lab_model_columns": lab_prevalence["model_column"].tolist(),
        "control_groups": cfg.CLINICALLY_SIMILAR_CONTROL_GROUPS,
        "feature_aggregation_mode": cfg.FEATURE_AGGREGATION_MODE,
        "include_first_features": bool(cfg.INCLUDE_FIRST_FEATURES),
        "use_ecg_features": bool(cfg.USE_ECG_FEATURES),
        "require_ecg_measurements": bool(cfg.REQUIRE_ECG_MEASUREMENTS),
        "use_medication_features": bool(cfg.USE_MEDICATION_FEATURES),
        "features_to_drop": list(getattr(cfg, "FEATURES_TO_DROP", [])),
        "require_core_vitals_complete": bool(getattr(cfg, "REQUIRE_CORE_VITALS_COMPLETE", False)),
        "require_heart_rate_resp_complete": bool(getattr(cfg, "REQUIRE_HEART_RATE_RESP_COMPLETE", False)),
        "col_missingness_threshold": float(cfg.COL_MISSINGNESS_THRESHOLD),
        "row_missingness_threshold": float(cfg.ROW_MISSINGNESS_THRESHOLD),
        "leakage_labs_to_drop": cfg.LEAKAGE_LABS_TO_DROP,
        "cv_folds": int(cfg.CV_FOLDS),
        "reports_dir": str(report_dir),
    }
    if active_features is not None:
        manifest["active_features_after_preprocessing"] = active_features
        manifest["active_feature_count_after_preprocessing"] = len(active_features)
    if core_vitals_filter_stats is not None:
        manifest["core_vitals_complete_filter"] = core_vitals_filter_stats

    manifest_path = report_dir / f"{run_id}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest_path


def run(args, report_dir):
    cfg = configure(report_dir, drop_features=args.drop_features)
    cfg.REQUIRE_CORE_VITALS_COMPLETE = args.require_core_vitals_complete
    cfg.REQUIRE_HEART_RATE_RESP_COMPLETE = args.require_heart_rate_resp_complete
    if args.require_core_vitals_complete and args.require_heart_rate_resp_complete:
        raise ValueError(
            "Use only one experimental vitals completeness filter at a time: "
            "--require-core-vitals-complete or --require-heart-rate-resp-complete."
        )
    matrix_path, lab_prevalence = build_or_restore_matrix(
        cfg=cfg,
        report_dir=report_dir,
        run_id=args.run_id,
        top_n=args.top_labs,
    )
    write_manifest(report_dir, args.run_id, cfg, args.top_labs, lab_prevalence)

    engine = ModelEngine(cfg)
    engine.run_id = f"{args.run_id}_lab_vitals_ecg_first"
    X_train, y_train, X_test, y_test = engine.preprocess(matrix_path)
    active_features = list(X_train.columns)
    core_vitals_filter_stats = None
    if args.require_core_vitals_complete:
        (
            X_train,
            y_train,
            X_test,
            y_test,
            core_vitals_filter_stats,
        ) = apply_core_vitals_completeness_filter(X_train, y_train, X_test, y_test)
    if args.require_heart_rate_resp_complete:
        (
            X_train,
            y_train,
            X_test,
            y_test,
            core_vitals_filter_stats,
        ) = apply_heart_rate_resp_completeness_filter(X_train, y_train, X_test, y_test)
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
        core_vitals_filter_stats=core_vitals_filter_stats,
    )
    engine.train_and_eval(X_train, y_train, X_test, y_test)
    create_html_presentation(
        report_dir,
        engine.run_id,
        lab_prevalence,
        cfg,
        X_train,
        y_train,
        X_test,
        core_vitals_filter_stats=core_vitals_filter_stats,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run a first-value labs plus vitals plus ECG model using the top "
            "prevalent in-window labs and chest/back-pain controls."
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
    parser.add_argument(
        "--require-core-vitals-complete",
        action="store_true",
        help=(
            "Experimental cohort filter: require heart rate, respiratory rate, "
            "and either complete noninvasive BP or complete arterial BP first values."
        ),
    )
    parser.add_argument(
        "--require-heart-rate-resp-complete",
        action="store_true",
        help=(
            "Experimental cohort filter: require heart_rate_first and resp_rate_first "
            "without requiring BP completeness."
        ),
    )
    args = parser.parse_args()

    report_dir = Path("../data/processed/model_reports/lab_vitals_ecg_first_runs") / args.run_id
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
