import argparse
import gc
import json
import os
import re
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

from cache import EventCache
from config import PipelineConfig
from engine import ModelEngine
from run_lab_vitals_first_model import (
    Tee,
    configure,
    first_match,
    format_metric,
    html_escape,
    image_data_uri,
    load_lab_labels,
    plot_pr_curve,
    plot_roc_curve,
    plot_shap_all_features,
    plot_shap_beeswarm_all_features,
    sanitize_feature_name,
    train_model_for_shap,
)


DEFAULT_COHORT_DIR = (
    Path("../data/processed/openai_discharge_note_parses")
    / "azure_gpt56_luna_full_160095_20260713"
    / "derived_llm_target_control_cohorts_20260713"
)
DEFAULT_TARGETS_PATH = DEFAULT_COHORT_DIR / "llm_targets_confirmed_new_non_type_b_filtered.csv"
DEFAULT_CONTROLS_PATH = DEFAULT_COHORT_DIR / "llm_controls_most_recent_chest_back_pain_useful.csv"
DEMOGRAPHIC_FEATURES = [
    "index_age",
    "gender",
    "race",
    "insurance",
    "marital_status",
]


def normalize_ids(df):
    df = df.copy()
    df["subject_id"] = pd.to_numeric(df["subject_id"], errors="coerce").astype("Int64")
    df["hadm_id"] = pd.to_numeric(df["hadm_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["subject_id", "hadm_id"]).copy()
    df["subject_id"] = df["subject_id"].astype("int64")
    df["hadm_id"] = df["hadm_id"].astype("int64")
    return df


def assign_grouped_split(spine, cfg):
    subject_labels = (
        spine.groupby("subject_id")["is_aortic_dissection"]
        .max()
        .reset_index()
    )
    _, test_subjects = train_test_split(
        subject_labels["subject_id"],
        test_size=cfg.TEST_SIZE,
        random_state=cfg.RANDOM_STATE,
        stratify=subject_labels["is_aortic_dissection"],
    )
    test_subjects = set(test_subjects)
    spine["cohort_split"] = "train"
    spine.loc[spine["subject_id"].isin(test_subjects), "cohort_split"] = "test"
    return spine


def build_llm_spine(cfg, targets_path, controls_path, report_dir, run_id):
    print("\n[1/4] Building LLM-derived Cohort Spine & Demographics...")
    targets = normalize_ids(pd.read_csv(targets_path, usecols=["subject_id", "hadm_id"]))
    controls = normalize_ids(pd.read_csv(controls_path, usecols=["subject_id", "hadm_id"]))
    targets["is_aortic_dissection"] = 1
    controls["is_aortic_dissection"] = 0

    overlap_hadms = set(targets["hadm_id"]) & set(controls["hadm_id"])
    if overlap_hadms:
        raise ValueError(f"Target/control hadm_id overlap detected: {sorted(overlap_hadms)[:10]}")
    overlap_subjects = set(targets["subject_id"]) & set(controls["subject_id"])
    if overlap_subjects:
        raise ValueError(f"Target/control subject_id overlap detected: {sorted(overlap_subjects)[:10]}")

    cohort = pd.concat([targets, controls], ignore_index=True)
    cohort["cohort_row_id"] = np.arange(len(cohort), dtype=np.int64)

    adm_cols = [
        "subject_id",
        "hadm_id",
        "admittime",
        "edregtime",
        "admission_type",
        "admission_location",
        "race",
        "insurance",
        "marital_status",
    ]
    admissions = pd.read_csv(cfg.DATA_DIR / "hosp" / "admissions.csv", usecols=adm_cols)
    admissions["subject_id"] = pd.to_numeric(admissions["subject_id"], errors="coerce").astype("Int64")
    admissions["hadm_id"] = pd.to_numeric(admissions["hadm_id"], errors="coerce").astype("Int64")
    admissions = admissions.dropna(subset=["subject_id", "hadm_id"]).copy()
    admissions["subject_id"] = admissions["subject_id"].astype("int64")
    admissions["hadm_id"] = admissions["hadm_id"].astype("int64")
    admissions["admittime"] = pd.to_datetime(admissions["admittime"], errors="coerce")
    admissions["edregtime"] = pd.to_datetime(admissions["edregtime"], errors="coerce")
    admissions = assign_index_time(admissions, use_ed_time=cfg.USE_EDREGTIME_AS_INDEX_TIME)

    spine = cohort.merge(admissions, on=["subject_id", "hadm_id"], how="left", validate="many_to_one")
    if spine["admittime"].isna().any():
        missing = spine.loc[spine["admittime"].isna(), ["subject_id", "hadm_id"]]
        raise ValueError(f"{len(missing)} cohort admissions did not match admissions.csv metadata.")

    patients = pd.read_csv(
        cfg.DATA_DIR / "hosp" / "patients.csv",
        usecols=["subject_id", "anchor_age", "anchor_year", "gender"],
    )
    spine = spine.merge(patients, on="subject_id", how="inner")
    spine["index_age"] = spine["anchor_age"] + (spine["index_time"].dt.year - spine["anchor_year"])
    before_age = len(spine)
    spine = spine[spine["index_age"] >= cfg.MIN_AGE].copy()
    if len(spine) != before_age:
        print(f"    Age filter removed {before_age - len(spine):,} rows below age {cfg.MIN_AGE}.")

    spine.rename(columns={"hadm_id": "index_hadm_id", "admission_type": "encounter_urgency"}, inplace=True)
    spine = assign_grouped_split(spine, cfg)
    spine["feature_window_end"] = spine["index_time"] + pd.to_timedelta(cfg.DAY_0_WINDOW_HOURS, unit="h")
    spine["diagnosis_proxy_time"] = pd.NaT
    spine["diagnosis_time_source"] = ""

    keep_cols = [
        "cohort_row_id",
        "subject_id",
        "index_hadm_id",
        "index_time",
        "index_time_source",
        "cohort_split",
        "is_aortic_dissection",
        "feature_window_end",
        "diagnosis_proxy_time",
        "diagnosis_time_source",
        "index_age",
        "gender",
        "race",
        "insurance",
        "marital_status",
        "encounter_urgency",
        "admission_location",
    ]
    spine = spine[keep_cols].copy()
    print(
        "    LLM cohort spine: "
        f"{len(spine):,} encounter rows; "
        f"{int(spine['is_aortic_dissection'].sum()):,} targets; "
        f"{int((spine['is_aortic_dissection'] == 0).sum()):,} controls; "
        f"{spine['subject_id'].nunique():,} subjects."
    )
    print(
        "    Split counts:\n"
        + spine.groupby(["cohort_split", "is_aortic_dissection"]).size().to_string()
    )
    cohort_audit_path = report_dir / f"{run_id}_llm_model_cohort_spine.csv"
    spine.to_csv(cohort_audit_path, index=False)
    print(f"    Cohort spine audit: {cohort_audit_path}")

    temp_path = "temp_llm_spine.csv"
    spine.to_csv(temp_path, index=False)
    return temp_path


def assign_index_time(admissions, use_ed_time=True):
    admissions = admissions.copy()
    has_ed_time = (
        admissions["edregtime"].notna()
        & admissions["admittime"].notna()
        & (admissions["edregtime"] <= admissions["admittime"])
        if use_ed_time
        else pd.Series(False, index=admissions.index)
    )
    ed_after_admit = (
        admissions["edregtime"].notna()
        & admissions["admittime"].notna()
        & (admissions["edregtime"] > admissions["admittime"])
        if use_ed_time
        else pd.Series(False, index=admissions.index)
    )
    admissions["index_time"] = admissions["admittime"]
    admissions.loc[has_ed_time, "index_time"] = admissions.loc[has_ed_time, "edregtime"]
    admissions["index_time_source"] = "admittime"
    admissions.loc[has_ed_time, "index_time_source"] = "edregtime"
    admissions.loc[ed_after_admit, "index_time_source"] = "admittime_edregtime_after_admit"
    return admissions


def iter_event_chunks(cfg, source_path, cache_name, itemids, subject_ids, use_cache=True):
    itemids = list(itemids)
    subject_ids = set(subject_ids)
    if use_cache:
        cache = EventCache(cfg)
        table_name = cache.ensure_table(cache_name, source_path, itemids)
        yield from cache.iter_subject_events(table_name, subject_ids, itemids)
        return

    chunksize = getattr(cfg, "EVENT_CACHE_CHUNKSIZE", 3_000_000)
    usecols = ["subject_id", "charttime", "itemid", "valuenum"]
    for chunk in pd.read_csv(source_path, usecols=usecols, chunksize=chunksize):
        chunk = chunk[
            (chunk["subject_id"].isin(subject_ids))
            & (chunk["itemid"].isin(itemids))
        ].copy()
        if not chunk.empty:
            yield chunk


def spine_times_for_merge(spine):
    out = spine[["cohort_row_id", "subject_id", "index_time", "feature_window_end"]].copy()
    out["index_time"] = pd.to_datetime(out["index_time"], errors="coerce")
    out["feature_window_end"] = pd.to_datetime(out["feature_window_end"], errors="coerce")
    return out


def aggregate_first_events_by_row(event_chunks, spine, item_map):
    spine_times = spine_times_for_merge(spine)
    partial_first = []
    for chunk in event_chunks:
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk["valuenum"] = pd.to_numeric(chunk["valuenum"], errors="coerce")
        chunk.dropna(subset=["subject_id", "charttime", "itemid", "valuenum"], inplace=True)
        if chunk.empty:
            continue
        chunk = chunk.merge(spine_times, on="subject_id", how="inner")
        chunk = chunk[
            (chunk["charttime"] >= chunk["index_time"])
            & (chunk["charttime"] <= chunk["feature_window_end"])
        ].copy()
        if chunk.empty:
            continue
        chunk["feature_name"] = chunk["itemid"].map(item_map)
        first_idx = chunk.groupby(["cohort_row_id", "feature_name"])["charttime"].idxmin()
        partial_first.append(chunk.loc[first_idx, ["cohort_row_id", "feature_name", "charttime", "valuenum"]])
        del chunk
        gc.collect()

    if not partial_first:
        return pd.DataFrame({"cohort_row_id": spine["cohort_row_id"]})

    first_values = (
        pd.concat(partial_first, ignore_index=True)
        .sort_values("charttime")
        .groupby(["cohort_row_id", "feature_name"])["valuenum"]
        .first()
    )
    wide = first_values.unstack()
    wide.columns = [f"{feature}_first" for feature in wide.columns]
    wide.reset_index(inplace=True)
    return wide


def select_top_labs_by_row(cfg, spine_path, top_n):
    print(f"\n[2/4] Selecting top {top_n} most prevalent in-window numeric labs...")
    spine = pd.read_csv(
        spine_path,
        usecols=[
            "cohort_row_id",
            "subject_id",
            "index_time",
            "feature_window_end",
            "is_aortic_dissection",
        ],
    )
    spine["index_time"] = pd.to_datetime(spine["index_time"], errors="coerce")
    spine["feature_window_end"] = pd.to_datetime(spine["feature_window_end"], errors="coerce")
    default_end = spine["index_time"] + pd.to_timedelta(cfg.DAY_0_WINDOW_HOURS, unit="h")
    spine["feature_window_end"] = spine["feature_window_end"].fillna(default_end)
    spine_times = spine_times_for_merge(spine)
    labels = spine[["cohort_row_id", "is_aortic_dissection"]].copy()

    observed_pairs = []
    usecols = ["subject_id", "charttime", "itemid", "valuenum"]
    chunksize = getattr(cfg, "EVENT_CACHE_CHUNKSIZE", 3_000_000)
    subject_ids = set(spine["subject_id"])
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
        ].copy()
        if not chunk.empty:
            observed_pairs.append(chunk[["cohort_row_id", "itemid"]].drop_duplicates())
        if chunk_number % 10 == 0:
            observed_count = sum(len(pairs) for pairs in observed_pairs)
            print(
                f"      scanned {chunk_number:,} lab chunks; "
                f"accumulated {observed_count:,} encounter-lab observations"
            )
        del chunk
        gc.collect()

    if not observed_pairs:
        raise ValueError("No in-window numeric labs were found for the LLM cohort.")

    observed = pd.concat(observed_pairs, ignore_index=True).drop_duplicates()
    observed["itemid"] = pd.to_numeric(observed["itemid"], errors="coerce").astype("Int64")
    observed.dropna(subset=["itemid"], inplace=True)
    observed["itemid"] = observed["itemid"].astype("int64")
    observed = observed.merge(labels, on="cohort_row_id", how="inner")

    total_rows = int(spine["cohort_row_id"].nunique())
    total_targets = int((spine["is_aortic_dissection"] == 1).sum())
    total_controls = int((spine["is_aortic_dissection"] == 0).sum())
    overall_counts = observed.groupby("itemid")["cohort_row_id"].nunique()
    target_counts = (
        observed[observed["is_aortic_dissection"] == 1]
        .groupby("itemid")["cohort_row_id"]
        .nunique()
    )
    control_counts = (
        observed[observed["is_aortic_dissection"] == 0]
        .groupby("itemid")["cohort_row_id"]
        .nunique()
    )
    prevalence = (
        pd.DataFrame({"observed_encounters": overall_counts})
        .join(target_counts.rename("target_observed_encounters"), how="left")
        .join(control_counts.rename("control_observed_encounters"), how="left")
        .fillna(0)
        .reset_index()
    )
    count_cols = [
        "observed_encounters",
        "target_observed_encounters",
        "control_observed_encounters",
    ]
    prevalence[count_cols] = prevalence[count_cols].astype(int)
    prevalence["overall_prevalence"] = prevalence["observed_encounters"] / total_rows
    prevalence["target_prevalence"] = prevalence["target_observed_encounters"] / total_targets
    prevalence["control_prevalence"] = prevalence["control_observed_encounters"] / total_controls
    prevalence["total_encounters"] = total_rows
    prevalence["total_targets"] = total_targets
    prevalence["total_controls"] = total_controls

    lab_labels = load_lab_labels(cfg)
    prevalence = prevalence.merge(lab_labels, on="itemid", how="left")
    prevalence["label"] = prevalence["label"].fillna(
        prevalence["itemid"].map(lambda itemid: f"itemid_{itemid}")
    )
    prevalence["fluid"] = prevalence["fluid"].fillna("")
    prevalence["category"] = prevalence["category"].fillna("")
    prevalence.sort_values(
        ["overall_prevalence", "observed_encounters", "itemid"],
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
    lab_dict = dict(zip(selected["itemid"], selected["feature_name"]))
    print(
        f"    Selected {len(lab_dict):,} labs from {len(prevalence):,} observed numeric lab itemids."
    )
    print(
        "    Most prevalent selected labs: "
        + ", ".join(selected["feature_name"].tolist())
    )
    return lab_dict, selected


def add_labs(cfg, spine_path):
    print("[2.5/4] Extracting First In-Window Labs...")
    spine = pd.read_csv(spine_path)
    spine["index_time"] = pd.to_datetime(spine["index_time"], errors="coerce")
    event_chunks = iter_event_chunks(
        cfg,
        cfg.DATA_DIR / "hosp" / "labevents.csv",
        "labevents_labs",
        cfg.LABS_DICT.keys(),
        spine["subject_id"],
        use_cache=False,
    )
    agg_labs = aggregate_first_events_by_row(event_chunks, spine, cfg.LABS_DICT)
    spine = spine.merge(agg_labs, on="cohort_row_id", how="left")
    temp_path = "temp_llm_spine_labs.csv"
    spine.to_csv(temp_path, index=False)
    os.remove(spine_path)
    return temp_path


def add_ecg_features(cfg, matrix_path):
    print("[3.5/4] Extracting Windowed ECG Machine Measurements...")
    df = pd.read_csv(matrix_path, low_memory=False)
    df["index_time"] = pd.to_datetime(df["index_time"], errors="coerce")
    df["feature_window_end"] = pd.to_datetime(df["feature_window_end"], errors="coerce")
    spine_times = spine_times_for_merge(df)
    subject_ids = set(df["subject_id"])
    ecg_features = list(getattr(cfg, "ECG_MEASUREMENT_FEATURES", []))
    derived_features = list(getattr(cfg, "ECG_DERIVED_FEATURES", []))
    all_features = ecg_features + derived_features
    usecols = ["subject_id", "ecg_time"] + ecg_features
    partial_first = []
    partial_counts = []
    for chunk in pd.read_csv(
        cfg.ECG_MEASUREMENTS_PATH,
        usecols=usecols,
        chunksize=getattr(cfg, "ECG_CHUNKSIZE", 1_000_000),
    ):
        chunk = chunk[chunk["subject_id"].isin(subject_ids)].copy()
        if chunk.empty:
            continue
        chunk["ecg_time"] = pd.to_datetime(chunk["ecg_time"], errors="coerce")
        chunk.dropna(subset=["subject_id", "ecg_time"], inplace=True)
        if chunk.empty:
            continue
        chunk = chunk.merge(spine_times, on="subject_id", how="inner")
        chunk = chunk[
            (chunk["ecg_time"] >= chunk["index_time"])
            & (chunk["ecg_time"] <= chunk["feature_window_end"])
        ].copy()
        if chunk.empty:
            continue
        for col in ecg_features:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        if "qt_interval" in derived_features and {"t_end", "qrs_onset"}.issubset(chunk.columns):
            chunk["qt_interval"] = chunk["t_end"] - chunk["qrs_onset"]
        if "qtc_bazett" in derived_features and {"qt_interval", "rr_interval"}.issubset(chunk.columns):
            rr_seconds = chunk["rr_interval"] / 1000
            chunk["qtc_bazett"] = chunk["qt_interval"] / np.sqrt(rr_seconds.where(rr_seconds > 0))
        available_features = [col for col in all_features if col in chunk.columns]
        chunk[available_features] = chunk[available_features].apply(pd.to_numeric, errors="coerce")
        partial_counts.append(chunk.groupby("cohort_row_id").size())
        first_idx = chunk.groupby("cohort_row_id")["ecg_time"].idxmin()
        partial_first.append(chunk.loc[first_idx, ["cohort_row_id", "ecg_time"] + available_features])
        del chunk
        gc.collect()

    if partial_first:
        first_values = (
            pd.concat(partial_first, ignore_index=True)
            .sort_values("ecg_time")
            .groupby("cohort_row_id")[all_features]
            .first()
        )
        ecg_wide = pd.DataFrame(index=first_values.index)
        for feature in all_features:
            ecg_wide[f"ecg_{feature}_first"] = first_values[feature]
        ecg_counts = pd.concat(partial_counts).groupby(level=0).sum()
        ecg_wide["ecg_count"] = ecg_counts
        ecg_wide.reset_index(inplace=True)
        df = df.merge(ecg_wide, on="cohort_row_id", how="left")
        print(f"    ECG features: {int(ecg_wide['cohort_row_id'].nunique()):,} encounters had at least one ECG in-window.")
    else:
        print("    ECG features: no in-window ECG machine measurements found.")
    df.drop(columns=["cohort_row_id"], inplace=True)
    df.to_csv(matrix_path, index=False)
    return matrix_path


def build_demographics_ecg_matrix(cfg, report_dir, run_id, targets_path, controls_path, top_labs=0):
    spine_path = build_llm_spine(cfg, targets_path, controls_path, report_dir, run_id)
    if top_labs:
        lab_dict, lab_prevalence = select_top_labs_by_row(cfg, spine_path, top_labs)
        cfg.LABS_DICT = lab_dict
        prevalence_path = report_dir / f"{run_id}_top{top_labs}_lab_prevalence_by_label.csv"
        lab_prevalence.to_csv(prevalence_path, index=False)
        active_labs_path = report_dir / f"{run_id}_selected_lab_features.txt"
        active_labs_path.write_text("\n".join(lab_prevalence["model_column"].tolist()) + "\n")
        print(f"    Lab prevalence audit: {prevalence_path}")
        print(f"    Selected lab feature list: {active_labs_path}")
        spine_path = add_labs(cfg, spine_path)
    matrix_path = add_ecg_features(cfg, spine_path)
    raw_matrix_audit = report_dir / f"{run_id}_raw_feature_matrix.csv"
    pd.read_csv(matrix_path).to_csv(raw_matrix_audit, index=False)
    print(f"    Raw feature matrix audit: {raw_matrix_audit}")
    return matrix_path


def feature_family_counts(active_features):
    demographic = [feature for feature in active_features if feature in DEMOGRAPHIC_FEATURES]
    ecg = [feature for feature in active_features if feature.startswith("ecg_")]
    lab = [
        feature
        for feature in active_features
        if feature.endswith("_first")
        and not feature.startswith("ecg_")
        and feature not in demographic
    ]
    other = [
        feature
        for feature in active_features
        if feature not in demographic and feature not in ecg and feature not in lab
    ]
    return {
        "demographic_features": demographic,
        "lab_features": lab,
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
    return f"<thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody>"


def create_demographics_ecg_html_presentation(report_dir, cfg, X_train, y_train, X_test, active_features):
    summary = pd.read_csv(first_match(report_dir, "_main_summary_metrics.csv")).iloc[0]
    cv = pd.read_csv(first_match(report_dir, "_main_cv_metrics.csv"))
    thresholds = pd.read_csv(first_match(report_dir, "_main_threshold_metrics.csv"))
    pr_path = first_match(report_dir, "_main_pr_curve.csv")
    roc_path = first_match(report_dir, "_main_roc_curve.csv")
    importance = pd.read_csv(first_match(report_dir, "_main_feature_importance.csv"))

    families = feature_family_counts(active_features)
    includes_labs = bool(families["lab_features"])
    report_slug = "demographics_labs_ecg_first" if includes_labs else "demographics_ecg_first"
    report_title = (
        "Aortic Dissection Demographics/Labs/ECG Model"
        if includes_labs
        else "Aortic Dissection Demographics/ECG Model"
    )
    subtitle = (
        "Demographics, top prevalent first in-window labs, and first in-window ECG machine-measurement values on LLM-derived target/control admissions."
        if includes_labs
        else "Demographics and first in-window ECG machine-measurement values on LLM-derived target/control admissions."
    )
    feature_bullet = (
        "The lab predictors are the top prevalent in-window numeric labs, aggregated as first values only; vitals, medication features, encounter urgency, and ECG count are not model predictors."
        if includes_labs
        else "Labs, vitals, medication features, encounter urgency, and ECG count are not model predictors in this run."
    )

    pr_png = report_dir / f"{report_slug}_pr_curve.png"
    roc_png = report_dir / f"{report_slug}_roc_curve.png"
    plot_pr_curve(pr_path, summary, pr_png)
    plot_roc_curve(roc_path, summary, roc_png)

    shap_model = train_model_for_shap(cfg, X_train, y_train, summary)
    shap_png = report_dir / f"{report_slug}_shap_all_features.png"
    shap_csv = report_dir / f"{report_slug}_shap_all_features.csv"
    shap_df, shap_values = plot_shap_all_features(shap_model, X_test, shap_png, shap_csv)
    shap_beeswarm_png = report_dir / f"{report_slug}_shap_beeswarm_all_features.png"
    plot_shap_beeswarm_all_features(shap_df, shap_values, X_test, shap_beeswarm_png)

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
  <title>{report_title}</title>
  <style>
    :root {{ color-scheme: light; --ink: #17202a; --muted: #5d6875; --line: #d9dee7; --blue: #264f78; --panel: #f7f9fc; }}
    body {{ margin: 0; font-family: Arial, Helvetica, sans-serif; color: var(--ink); background: #ffffff; line-height: 1.45; }}
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
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px 10px; text-align: right; white-space: nowrap; }}
    th {{ color: var(--muted); font-weight: 700; background: var(--panel); }}
    th:first-child, td.left {{ text-align: left; white-space: normal; }}
    .note {{ color: var(--muted); font-size: 14px; }}
    @media (max-width: 820px) {{ .grid, .two-col {{ grid-template-columns: 1fr; }} table {{ font-size: 12px; }} th, td {{ padding: 6px; }} }}
  </style>
</head>
<body>
<main>
  <section>
    <h1>{report_title}</h1>
    <p class="subtitle">{subtitle}</p>
    <ul>
      <li>Targets and controls are read from the derived discharge-note LLM cohort CSVs.</li>
      <li>The cohort unit is hospital admission / hadm_id, with encounter-keyed feature aggregation to avoid mixing multiple admissions for the same subject.</li>
      <li>{feature_bullet}</li>
      <li>The cohort is restricted to admissions with at least one ECG machine-measurement row in the 24-hour feature window.</li>
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
    <p>Feature count after preprocessing: <strong>{int(summary['feature_count'])}</strong>. Demographic/categorical features: <strong>{len(families['demographic_features'])}</strong>; lab features: <strong>{len(families['lab_features'])}</strong>; ECG features: <strong>{len(families['ecg_features'])}</strong>; other features: <strong>{len(families['other_features'])}</strong>.</p>
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
    <h2>Threshold Operating Points</h2>
    <table>{threshold_rows_html(thresholds)}</table>
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
    <p class="note">Computed with XGBoost native SHAP contribution values on the ECG-complete holdout set.</p>
    <img src="{image_data_uri(shap_beeswarm_png)}" alt="Directional all-feature SHAP beeswarm">
    <p>Top SHAP features: {", ".join(f"<strong>{html_escape(feature)}</strong>" for feature in top_shap_features)}.</p>
  </section>

  <section>
    <h2>All-Feature SHAP Magnitude</h2>
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
    html_path = report_dir / f"{report_slug}_model_update.html"
    html_path.write_text(html)
    print(f"HTML presentation: {html_path}")
    print(f"PR curve: {pr_png}")
    print(f"ROC curve: {roc_png}")
    print(f"SHAP plot: {shap_png}")
    print(f"SHAP beeswarm: {shap_beeswarm_png}")
    print(f"SHAP values: {shap_csv}")


def write_demographics_ecg_manifest(report_dir, run_id, cfg, targets_path, controls_path, active_features=None):
    method = (
        "llm_targets_controls_demographics_top_labs_plus_ecg_first_only"
        if getattr(cfg, "LABS_DICT", {})
        else "llm_targets_controls_demographics_plus_ecg_first_only"
    )
    manifest = {
        "run_id": run_id,
        "method": method,
        "cohort_source": "discharge-note LLM parsed phenotype cohorts",
        "llm_targets_path": str(targets_path),
        "llm_controls_path": str(controls_path),
        "cohort_unit": "hospital admission / hadm_id",
        "cohort_notes": (
            "Targets and controls are read from the derived LLM cohort CSVs. "
            "Feature aggregation is keyed by encounter row to avoid mixing multiple admissions for the same subject."
        ),
        "feature_aggregation_mode": cfg.FEATURE_AGGREGATION_MODE,
        "include_first_features": bool(cfg.INCLUDE_FIRST_FEATURES),
        "use_labs": bool(cfg.LABS_DICT),
        "use_vitals": bool(cfg.VITALS_DICT),
        "use_ecg_features": bool(cfg.USE_ECG_FEATURES),
        "require_ecg_measurements": bool(cfg.REQUIRE_ECG_MEASUREMENTS),
        "ecg_required_count_col": cfg.ECG_REQUIRED_COUNT_COL,
        "use_medication_features": bool(cfg.USE_MEDICATION_FEATURES),
        "features_to_drop": list(getattr(cfg, "FEATURES_TO_DROP", [])),
        "selected_lab_itemids": [int(itemid) for itemid in getattr(cfg, "LABS_DICT", {}).keys()],
        "selected_lab_features": list(getattr(cfg, "LABS_DICT", {}).values()),
        "col_missingness_threshold": float(cfg.COL_MISSINGNESS_THRESHOLD),
        "row_missingness_threshold": float(cfg.ROW_MISSINGNESS_THRESHOLD),
        "cv_folds": int(cfg.CV_FOLDS),
        "reports_dir": str(report_dir),
    }
    if active_features is not None:
        manifest["active_features_after_preprocessing"] = active_features
        manifest["active_feature_count_after_preprocessing"] = len(active_features)
        manifest.update(feature_family_counts(active_features))
    manifest_path = report_dir / f"{run_id}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest_path


def run(args, report_dir):
    cfg = configure(report_dir, drop_features=args.drop_features)
    cfg.USE_CLINICALLY_SIMILAR_CONTROLS = False
    cfg.CLINICALLY_SIMILAR_CONTROL_GROUPS = {}
    cfg.VITALS_DICT = {}
    cfg.LABS_DICT = {}
    cfg.LEAKAGE_LABS_TO_DROP = []
    cfg.USE_MATRIX_CACHE = False
    cfg.USE_MEDICATION_FEATURES = False
    cfg.FEATURES_TO_DROP = list(
        dict.fromkeys(list(args.drop_features or []) + ["ecg_count", "encounter_urgency"])
    )
    cfg.REQUIRE_ECG_MEASUREMENTS = True
    targets_path = Path(args.targets_path)
    controls_path = Path(args.controls_path)
    matrix_path = build_demographics_ecg_matrix(
        cfg,
        report_dir,
        args.run_id,
        targets_path,
        controls_path,
        top_labs=args.top_labs,
    )
    write_demographics_ecg_manifest(report_dir, args.run_id, cfg, targets_path, controls_path)

    engine = ModelEngine(cfg)
    run_slug = "demographics_labs_ecg_first" if args.top_labs else "demographics_ecg_first"
    engine.run_id = f"{args.run_id}_{run_slug}"
    X_train, y_train, X_test, y_test = engine.preprocess(matrix_path)
    active_features = list(X_train.columns)
    (report_dir / f"{args.run_id}_active_features.txt").write_text("\n".join(active_features) + "\n")
    write_demographics_ecg_manifest(
        report_dir,
        args.run_id,
        cfg,
        targets_path,
        controls_path,
        active_features=active_features,
    )
    engine.train_and_eval(X_train, y_train, X_test, y_test)
    create_demographics_ecg_html_presentation(
        report_dir,
        cfg,
        X_train,
        y_train,
        X_test,
        active_features,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run the demographics/ECG XGBoost model on LLM-derived target/control cohorts."
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--targets-path", default=str(DEFAULT_TARGETS_PATH))
    parser.add_argument("--controls-path", default=str(DEFAULT_CONTROLS_PATH))
    parser.add_argument("--drop-features", nargs="*", default=[])
    parser.add_argument("--top-labs", type=int, default=0)
    args = parser.parse_args()
    if args.top_labs < 0:
        parser.error("--top-labs must be non-negative")
    if args.run_id is None:
        prefix = (
            "llm_cohort_demographics_labs_ecg_xgb"
            if args.top_labs
            else "llm_cohort_demographics_ecg_xgb"
        )
        args.run_id = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    report_root = (
        Path("../data/processed/model_reports/llm_demographics_labs_ecg_runs")
        if args.top_labs
        else Path("../data/processed/model_reports/llm_demographics_ecg_runs")
    )
    report_dir = report_root / args.run_id
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
