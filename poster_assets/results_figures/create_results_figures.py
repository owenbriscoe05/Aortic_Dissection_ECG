#!/usr/bin/env python3
"""Create poster results figures from existing processed project artifacts.

The figures are intentionally aggregate/poster-oriented. They do not run models,
call LLMs, or write patient-note text.
"""

from __future__ import annotations

import json
import math
import os
import re
import textwrap
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import FancyArrowPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
CSV_DIR = OUT_DIR / "tables"
CSV_DIR.mkdir(parents=True, exist_ok=True)

PARSE_DIR = (
    PROJECT_ROOT
    / "data/processed/openai_discharge_note_parses/azure_gpt56_luna_full_160095_20260713"
)
DERIVED_DIR = PARSE_DIR / "derived_llm_target_control_cohorts_20260713"
LLM_MODEL_DIR = (
    PROJECT_ROOT
    / "data/processed/model_reports/llm_demographics_ecg_runs/"
    / "llm_cohort_demographics_ecg_xgb_20260714_102716"
)
STD_MODEL_DIR = PROJECT_ROOT / "data/processed/model_reports/demographics_ecg_runs/20260708_151834"
LLM_LABS_MODEL_DIR = (
    PROJECT_ROOT
    / "data/processed/model_reports/llm_demographics_labs_ecg_runs/"
    / "llm_cohort_demographics_labs_ecg_xgb_20260714_111259"
)
LLM_VITALS_MODEL_DIR = (
    PROJECT_ROOT
    / "data/processed/model_reports/vitals_ecg_first_runs/llm_cohort_vitals_ecg_xgb_20260713"
)
LLM_LAB_VITALS_MODEL_DIR = (
    PROJECT_ROOT / "data/processed/model_reports/lab_vitals_ecg_first_runs/llm_cohort_xgb_20260713"
)

DIAGNOSES_PATH = PROJECT_ROOT / "data/mimic-iv/hosp/diagnoses_icd.csv"

COLORS = {
    "ink": "#1F2933",
    "muted": "#5F7182",
    "line": "#B7C3CF",
    "blue": "#3D5A80",
    "blue_light": "#EDF2F7",
    "teal": "#2C7A7B",
    "teal_light": "#EAF5F4",
    "green": "#3A7D44",
    "green_light": "#EAF6EA",
    "red": "#A43E3E",
    "red_light": "#FCEEEE",
    "orange": "#B36B00",
    "orange_light": "#FFF6E6",
    "purple": "#6B4E9B",
    "purple_light": "#F7F1FF",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": COLORS["line"],
            "axes.labelcolor": COLORS["ink"],
            "text.color": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "savefig.bbox": "tight",
            "savefig.dpi": 220,
        }
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    for suffix in [".png", ".svg"]:
        fig.savefig(OUT_DIR / f"{name}{suffix}")
    plt.close(fig)


def fmt_int(value: float | int) -> str:
    return f"{int(round(float(value))):,}"


def fmt_pct(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return "NA"
    return f"{100 * float(value):.{digits}f}%"


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def clean_label(value: object, max_words: int = 6) -> str:
    text = str(value) if pd.notna(value) else "missing"
    text = text.replace("_", " ").strip()
    text = re.sub(r"\s+", " ", text)
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words]) + "..."
    return text


def wrap_label(value: object, width: int = 24) -> str:
    return "\n".join(textwrap.wrap(str(value), width=width, break_long_words=False))


def label_from_target(row: pd.Series) -> str:
    if int(row["is_aortic_dissection"]) == 1:
        stanford = str(row.get("anatomic_detail__stanford_type", "") or "").strip()
        if stanford and stanford.lower() not in {"nan", "none", "unclear"}:
            return f"Final target: Stanford {stanford}"
        return "Final target: type unclear"
    cleanup = str(row.get("overall__target_cleanup_label", "") or "")
    if cleanup == "dissection_ruled_out":
        return "Control: dissection ruled out"
    if cleanup == "no_aortic_syndrome_evidence":
        return "Control: no aortic syndrome"
    return "Control"


def load_llm_holdout() -> pd.DataFrame:
    raw_path = LLM_MODEL_DIR / "llm_cohort_demographics_ecg_xgb_20260714_102716_raw_feature_matrix.csv"
    pred_path = (
        LLM_MODEL_DIR
        / "llm_cohort_demographics_ecg_xgb_20260714_102716_demographics_ecg_first_main_holdout_predictions.csv"
    )
    raw = pd.read_csv(raw_path, low_memory=False)
    test = raw[raw["cohort_split"].eq("test")].copy().reset_index(drop=True)
    pred = pd.read_csv(pred_path)
    if len(test) != len(pred):
        raise ValueError(f"LLM test rows ({len(test)}) do not match predictions ({len(pred)}).")
    test["y_true"] = pred["y_true"].astype(int)
    test["y_probability"] = pred["y_probability"].astype(float)
    parse_cols = [
        "hadm_id",
        "first_ecg_time_24h",
        "ecg_count_24h",
        "admission_type",
        "admission_location",
        "presentation_phenotype__chest_pain",
        "presentation_phenotype__back_pain",
        "presentation_phenotype__abdominal_pain",
        "aortic_syndrome_status__confirmed_acute_aortic_dissection",
        "aortic_syndrome_status__chronic_or_prior_dissection_only",
        "aortic_syndrome_status__aneurysm_without_dissection",
        "aortic_syndrome_status__dissection_explicitly_ruled_out",
        "anatomic_detail__stanford_type",
        "control_usefulness__primary_discharge_diagnosis_or_alternative_explanation",
        "control_usefulness__plausible_aortic_dissection_mimic_presentation",
        "overall__target_cleanup_label",
    ]
    parse = pd.read_csv(PARSE_DIR / "openai_discharge_visit_parse_results.csv", usecols=parse_cols)
    parse = parse.rename(columns={"hadm_id": "index_hadm_id"})
    merged = test.merge(parse, on="index_hadm_id", how="left", validate="one_to_one")
    merged["phenotype_group"] = merged.apply(label_from_target, axis=1)
    return merged


def load_llm_all_parse() -> pd.DataFrame:
    cols = [
        "subject_id",
        "hadm_id",
        "index_time",
        "first_ecg_time_24h",
        "usage_input_tokens",
        "usage_cached_tokens",
        "usage_output_tokens",
        "usage_reasoning_tokens",
        "usage_total_tokens",
        "presentation_phenotype__chest_pain",
        "presentation_phenotype__back_pain",
        "aortic_syndrome_status__confirmed_acute_aortic_dissection",
        "aortic_syndrome_status__chronic_or_prior_dissection_only",
        "aortic_syndrome_status__aneurysm_without_dissection",
        "aortic_syndrome_status__dissection_explicitly_ruled_out",
        "anatomic_detail__stanford_type",
        "control_usefulness__plausible_aortic_dissection_mimic_presentation",
        "overall__target_cleanup_label",
    ]
    return pd.read_csv(PARSE_DIR / "openai_discharge_visit_parse_results.csv", usecols=cols, low_memory=False)


def exact_icd_hadm_ids() -> set[int]:
    target_codes = {
        9: {"441", "44100", "44101", "44103"},
        10: {"I7100", "I7101", "I71010", "I71012", "I71019", "I7103"},
    }
    hadm_ids: set[int] = set()
    for chunk in pd.read_csv(
        DIAGNOSES_PATH,
        usecols=["hadm_id", "icd_code", "icd_version"],
        chunksize=1_000_000,
    ):
        chunk["icd_code_norm"] = chunk["icd_code"].astype(str).str.upper().str.replace(".", "", regex=False)
        chunk["icd_version"] = pd.to_numeric(chunk["icd_version"], errors="coerce").astype("Int64")
        mask = pd.Series(False, index=chunk.index)
        for version, codes in target_codes.items():
            mask |= (chunk["icd_version"] == version) & chunk["icd_code_norm"].isin(codes)
        matched = pd.to_numeric(chunk.loc[mask, "hadm_id"], errors="coerce").dropna().astype("int64")
        hadm_ids.update(matched.tolist())
    return hadm_ids


def box_note(ax: plt.Axes, xy: tuple[float, float], text: str, color: str) -> None:
    ax.text(
        xy[0],
        xy[1],
        text,
        ha="center",
        va="center",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": color, "edgecolor": COLORS["line"]},
    )


def figure01_cohort_attrition(parse_summary: dict, derived_summary: dict) -> None:
    rows = [
        ("Adult ICD model spine", 223_452, "standard model"),
        ("ECG-complete ICD model", 70_094, "standard model"),
        ("ECG-qualified visits", parse_summary["qualifying_visits_with_ecg_24h"], "LLM parser"),
        ("With discharge notes", parse_summary["qualifying_visits_with_discharge_notes_seen"], "LLM parser"),
        ("Successful parses", parse_summary["parsed"], "LLM parser"),
        ("Base LLM targets", derived_summary["target_counts"]["base_confirmed_new_dissection_rows"], "LLM target cleanup"),
        ("Final LLM targets", derived_summary["target_counts"]["final_target_rows"], "LLM target cleanup"),
        ("Final LLM controls", derived_summary["control_counts"]["final_control_rows_subjects"], "LLM control selection"),
    ]
    df = pd.DataFrame(rows, columns=["step", "count", "workflow"])
    df.to_csv(CSV_DIR / "fig01_cohort_attrition_counts.csv", index=False)
    palette = {
        "standard model": COLORS["blue"],
        "LLM parser": COLORS["purple"],
        "LLM target cleanup": COLORS["red"],
        "LLM control selection": COLORS["green"],
    }
    fig, ax = plt.subplots(figsize=(13, 7))
    y = np.arange(len(df))[::-1]
    ax.barh(y, df["count"], color=[palette[w] for w in df["workflow"]], alpha=0.88)
    for yi, row in zip(y, df.itertuples(index=False)):
        ax.text(row.count * 1.02, yi, fmt_int(row.count), va="center", fontsize=12, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([wrap_label(s, 24) for s in df["step"]])
    ax.set_xscale("log")
    ax.set_xlabel("Count, log scale")
    ax.set_title("Cohort attrition and derived cohort yield")
    ax.grid(axis="x", alpha=0.25)
    handles = [
        Rectangle((0, 0), 1, 1, color=color, alpha=0.88, label=label)
        for label, color in palette.items()
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False)
    ax.text(
        0,
        -0.18,
        "Note: standard model and LLM parser starts are related but not identical denominators.",
        transform=ax.transAxes,
        color=COLORS["muted"],
    )
    save_figure(fig, "fig01_cohort_attrition_waterfall")


def sankey_like(ax: plt.Axes, table: pd.DataFrame, left_col: str, right_col: str, value_col: str) -> None:
    left_totals = table.groupby(left_col)[value_col].sum().sort_values(ascending=False)
    right_totals = table.groupby(right_col)[value_col].sum().sort_values(ascending=False)
    total = table[value_col].sum()
    left_y = {}
    right_y = {}
    y0 = 0.0
    for label, count in left_totals.items():
        h = count / total
        left_y[label] = [y0, y0 + h]
        y0 += h + 0.025
    y0 = 0.0
    for label, count in right_totals.items():
        h = count / total
        right_y[label] = [y0, y0 + h]
        y0 += h + 0.018
    left_offsets = {k: v[0] for k, v in left_y.items()}
    right_offsets = {k: v[0] for k, v in right_y.items()}
    colors = [COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["purple"], COLORS["red"], COLORS["teal"]]

    for i, row in enumerate(table.sort_values(value_col, ascending=False).itertuples(index=False)):
        left = getattr(row, left_col)
        right = getattr(row, right_col)
        value = getattr(row, value_col)
        h = value / total
        yl0 = left_offsets[left]
        yl1 = yl0 + h
        yr0 = right_offsets[right]
        yr1 = yr0 + h
        left_offsets[left] = yl1
        right_offsets[right] = yr1
        verts = [
            (0.18, yl0),
            (0.42, yl0),
            (0.58, yr0),
            (0.82, yr0),
            (0.82, yr1),
            (0.58, yr1),
            (0.42, yl1),
            (0.18, yl1),
            (0.18, yl0),
        ]
        codes = [
            MplPath.MOVETO,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.LINETO,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CLOSEPOLY,
        ]
        ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=colors[i % len(colors)], alpha=0.35, edgecolor="none"))

    for label, (y0, y1) in left_y.items():
        ax.add_patch(Rectangle((0.05, y0), 0.1, y1 - y0, color=COLORS["blue"], alpha=0.8))
        ax.text(0.04, (y0 + y1) / 2, f"{label}\n{fmt_int(left_totals[label])}", ha="right", va="center")
    for label, (y0, y1) in right_y.items():
        ax.add_patch(Rectangle((0.85, y0), 0.1, y1 - y0, color=COLORS["green"], alpha=0.8))
        ax.text(0.96, (y0 + y1) / 2, f"{label}\n{fmt_int(right_totals[label])}", ha="left", va="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, max(max(v[1] for v in left_y.values()), max(v[1] for v in right_y.values())) + 0.05)
    ax.axis("off")


def figure02_icd_vs_llm(parse_all: pd.DataFrame, icd_hadm_ids: set[int]) -> None:
    df = parse_all[["hadm_id", "overall__target_cleanup_label"]].copy()
    df["exact_icd_target"] = np.where(df["hadm_id"].isin(icd_hadm_ids), "Exact ICD target", "No exact ICD target")
    df["llm_label"] = df["overall__target_cleanup_label"].fillna("missing").map(
        {
            "newly_identified_dissection": "New dissection",
            "confirmed_acute_dissection": "New dissection",
            "previously_known_dissection_only": "Prior/known dissection",
            "aortic_syndrome_not_dissection": "Other aortic syndrome",
            "aneurysm_without_dissection": "Aneurysm only",
            "dissection_ruled_out": "Ruled out",
            "no_aortic_syndrome_evidence": "No aortic evidence",
            "unclear": "Unclear",
        }
    ).fillna(df["overall__target_cleanup_label"].map(clean_label))
    counts = df.groupby(["exact_icd_target", "llm_label"]).size().reset_index(name="count")
    top_labels = counts.groupby("llm_label")["count"].sum().nlargest(7).index
    counts["llm_label"] = np.where(counts["llm_label"].isin(top_labels), counts["llm_label"], "Other")
    counts = counts.groupby(["exact_icd_target", "llm_label"], as_index=False)["count"].sum()
    counts.to_csv(CSV_DIR / "fig02_icd_vs_llm_counts.csv", index=False)
    fig, ax = plt.subplots(figsize=(12, 7))
    sankey_like(ax, counts, "exact_icd_target", "llm_label", "count")
    ax.set_title("Exact ICD target status versus LLM phenotype labels", loc="left", fontsize=18, fontweight="bold")
    save_figure(fig, "fig02_icd_vs_llm_alluvial")


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": y, "p": p})
    df["decile"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False) + 1
    rows = []
    for decile, group in df.groupby("decile"):
        n = len(group)
        observed = group["y"].mean()
        se = math.sqrt(max(observed * (1 - observed), 0) / n) if n else np.nan
        rows.append(
            {
                "decile": int(decile),
                "n": n,
                "mean_predicted": group["p"].mean(),
                "observed": observed,
                "ci_low": max(0, observed - 1.96 * se),
                "ci_high": min(1, observed + 1.96 * se),
            }
        )
    return pd.DataFrame(rows)


def figure03_calibration(holdout: pd.DataFrame) -> None:
    cal = calibration_table(holdout["y_true"].to_numpy(), holdout["y_probability"].to_numpy())
    cal.to_csv(CSV_DIR / "fig03_calibration_deciles.csv", index=False)
    fig = plt.figure(figsize=(13, 6.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.15, 1])
    ax = fig.add_subplot(gs[0, 0])
    ax.errorbar(
        cal["mean_predicted"],
        cal["observed"],
        yerr=[cal["observed"] - cal["ci_low"], cal["ci_high"] - cal["observed"]],
        fmt="o-",
        color=COLORS["blue"],
        ecolor=COLORS["line"],
        capsize=3,
    )
    lim = max(cal["mean_predicted"].max(), cal["ci_high"].max()) * 1.1
    lim = max(lim, 0.08)
    ax.plot([0, lim], [0, lim], "--", color=COLORS["muted"], label="Perfect calibration")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Mean predicted risk")
    ax.set_ylabel("Observed prevalence")
    ax.set_title("Calibration by risk decile")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    ax_tbl = fig.add_subplot(gs[0, 1])
    ax_tbl.axis("off")
    display = cal.copy()
    display["mean_predicted"] = display["mean_predicted"].map(lambda v: f"{v:.3f}")
    display["observed"] = display["observed"].map(lambda v: f"{v:.3f}")
    display = display[["decile", "n", "mean_predicted", "observed"]]
    table = ax_tbl.table(cellText=display.values, colLabels=display.columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.45)
    ax_tbl.set_title("Calibration table", loc="left")
    save_figure(fig, "fig03_calibration_deciles")


def figure04_decision_curve(holdout: pd.DataFrame) -> None:
    y = holdout["y_true"].to_numpy()
    p = holdout["y_probability"].to_numpy()
    n = len(y)
    prevalence = y.mean()
    thresholds = np.linspace(0.01, 0.50, 100)
    rows = []
    for t in thresholds:
        pred = p >= t
        tp = ((pred == 1) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        net_benefit = tp / n - fp / n * (t / (1 - t))
        treat_all = prevalence - (1 - prevalence) * (t / (1 - t))
        rows.append({"threshold": t, "model": net_benefit, "treat_all": treat_all, "treat_none": 0.0})
    df = pd.DataFrame(rows)
    df.to_csv(CSV_DIR / "fig04_decision_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["threshold"], df["model"], color=COLORS["blue"], lw=2.5, label="LLM cohort model")
    ax.plot(df["threshold"], df["treat_all"], color=COLORS["red"], lw=2, label="Treat all")
    ax.plot(df["threshold"], df["treat_none"], color=COLORS["muted"], lw=2, label="Treat none")
    ax.set_xlabel("Risk threshold")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision-curve analysis")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, "fig04_decision_curve")


def figure05_risk_distribution(holdout: pd.DataFrame, parse_all: pd.DataFrame) -> None:
    scored = holdout.copy()
    order = (
        scored.groupby("phenotype_group")["y_probability"].median().sort_values(ascending=False).index.tolist()
    )
    fig = plt.figure(figsize=(13, 7))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.35, 1])
    ax = fig.add_subplot(gs[0, 0])
    data = [scored.loc[scored["phenotype_group"].eq(label), "y_probability"].dropna() for label in order]
    ax.boxplot(data, orientation="horizontal", patch_artist=True, tick_labels=[wrap_label(x, 24) for x in order])
    for patch in ax.artists:
        patch.set_facecolor(COLORS["blue_light"])
    ax.set_xlabel("Predicted risk in holdout")
    ax.set_title("Scored holdout risk by phenotype group")
    ax.grid(axis="x", alpha=0.25)

    ax2 = fig.add_subplot(gs[0, 1])
    label_counts = parse_all["overall__target_cleanup_label"].fillna("missing").map(clean_label).value_counts().head(8)
    ax2.barh(range(len(label_counts))[::-1], label_counts.values, color=COLORS["purple"], alpha=0.85)
    ax2.set_yticks(range(len(label_counts))[::-1])
    ax2.set_yticklabels([wrap_label(x, 22) for x in label_counts.index])
    ax2.set_xscale("log")
    ax2.set_title("All parsed labels, not all scored")
    ax2.set_xlabel("Parsed visit count, log scale")
    ax2.grid(axis="x", alpha=0.25)
    label_counts.reset_index().rename(columns={"index": "label", "overall__target_cleanup_label": "count"}).to_csv(
        CSV_DIR / "fig05_all_parse_label_counts.csv", index=False
    )
    save_figure(fig, "fig05_risk_distribution_by_phenotype")


def figure06_topk_workload(holdout: pd.DataFrame) -> None:
    df = holdout[["y_true", "y_probability"]].sort_values("y_probability", ascending=False).reset_index(drop=True)
    total_pos = int(df["y_true"].sum())
    rows = []
    for pct in [0.01, 0.02, 0.05, 0.10]:
        k = max(1, int(math.ceil(len(df) * pct)))
        selected = df.iloc[:k]
        tp = int(selected["y_true"].sum())
        fp = k - tp
        rows.append({"top_percent": pct, "k": k, "tp": tp, "fp": fp, "sensitivity": tp / total_pos, "ppv": tp / k})
    out = pd.DataFrame(rows)
    out.to_csv(CSV_DIR / "fig06_topk_workload.csv", index=False)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(out))
    width = 0.36
    ax1.bar(x - width / 2, out["sensitivity"], width, label="Sensitivity", color=COLORS["blue"])
    ax1.bar(x + width / 2, out["ppv"], width, label="PPV", color=COLORS["green"])
    ax1.set_ylim(0, max(0.1, out[["sensitivity", "ppv"]].max().max() * 1.25))
    ax1.set_ylabel("Rate")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Top {int(p * 100)}%\n(n={k})" for p, k in zip(out["top_percent"], out["k"])])
    ax1.set_title("Top-k review workload")
    ax1.legend(frameon=False, loc="upper left")
    ax2 = ax1.twinx()
    ax2.plot(x, out["fp"], color=COLORS["red"], marker="o", lw=2, label="False positives")
    ax2.set_ylabel("False positives")
    ax2.legend(frameon=False, loc="upper right")
    save_figure(fig, "fig06_topk_workload")


def collapse_diagnosis(value: object) -> str:
    text = str(value).lower() if pd.notna(value) else ""
    rules = [
        ("Pneumonia/respiratory", ["pneumonia", "copd", "respiratory", "pleuritic", "asthma"]),
        ("Cardiac/arrhythmia", ["atrial", "tachy", "chest pain", "myocard", "heart", "coronary"]),
        ("GI/abdominal", ["abdominal", "pancreatitis", "chole", "colitis", "diarrhea", "bowel"]),
        ("Infection/sepsis", ["sepsis", "infection", "bacter", "fever", "abscess"]),
        ("Neurologic", ["stroke", "seizure", "syncope", "altered"]),
        ("Musculoskeletal/pain", ["back pain", "fracture", "musculoskeletal", "pain"]),
    ]
    for label, needles in rules:
        if any(n in text for n in needles):
            return label
    return "Other/unclear"


def figure07_error_review(holdout: pd.DataFrame) -> None:
    threshold = pd.read_csv(
        LLM_MODEL_DIR
        / "llm_cohort_demographics_ecg_xgb_20260714_102716_demographics_ecg_first_main_summary_metrics.csv"
    ).iloc[0]["selected_threshold"]
    df = holdout.copy()
    df["pred"] = df["y_probability"] >= threshold
    fp = df[(df["pred"]) & (df["y_true"] == 0)].copy()
    fn = df[(~df["pred"]) & (df["y_true"] == 1)].copy()
    fp["error_category"] = fp["control_usefulness__primary_discharge_diagnosis_or_alternative_explanation"].map(
        collapse_diagnosis
    )
    fn["error_category"] = "Stanford " + fn["anatomic_detail__stanford_type"].fillna("unclear").astype(str)
    counts = pd.concat(
        [
            fp["error_category"].value_counts().rename_axis("category").reset_index(name="count").assign(error="False positive"),
            fn["error_category"].value_counts().rename_axis("category").reset_index(name="count").assign(error="False negative"),
        ],
        ignore_index=True,
    )
    counts.to_csv(CSV_DIR / "fig07_error_review_counts.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=False)
    for ax, err, color in zip(axes, ["False positive", "False negative"], [COLORS["red"], COLORS["orange"]]):
        sub = counts[counts["error"].eq(err)].sort_values("count").tail(8)
        ax.barh(sub["category"], sub["count"], color=color, alpha=0.82)
        ax.set_title(err)
        ax.set_xlabel("Holdout count")
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle(f"Error phenotype review at validation-selected threshold ({threshold:.3f})", fontweight="bold")
    save_figure(fig, "fig07_error_phenotype_review")


def figure08_feature_availability() -> None:
    raw = pd.read_csv(LLM_MODEL_DIR / "llm_cohort_demographics_ecg_xgb_20260714_102716_raw_feature_matrix.csv")
    families = {
        "Demographics": ["index_age", "gender", "race", "insurance", "marital_status"],
        "ECG intervals": ["ecg_rr_interval_first", "ecg_qt_interval_first", "ecg_qtc_bazett_first"],
        "ECG wave timing": [
            "ecg_p_onset_first",
            "ecg_p_end_first",
            "ecg_qrs_onset_first",
            "ecg_qrs_end_first",
            "ecg_t_end_first",
        ],
        "ECG axes": ["ecg_p_axis_first", "ecg_qrs_axis_first", "ecg_t_axis_first"],
    }
    rows = []
    for label, group in raw.groupby("is_aortic_dissection"):
        label_name = "Target" if label == 1 else "Control"
        for family, cols in families.items():
            existing = [c for c in cols if c in raw.columns]
            rows.append(
                {
                    "label": label_name,
                    "feature_family": family,
                    "missing_fraction": group[existing].isna().mean().mean(),
                    "observed_fraction": 1 - group[existing].isna().mean().mean(),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(CSV_DIR / "fig08_feature_availability.csv", index=False)
    pivot = out.pivot(index="feature_family", columns="label", values="observed_fraction")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(pivot.values, vmin=0, vmax=1, cmap="Greens")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, fmt_pct(pivot.values[i, j], 1), ha="center", va="center", fontweight="bold")
    ax.set_title("Feature observedness by family and outcome")
    fig.colorbar(im, ax=ax, label="Observed fraction")
    save_figure(fig, "fig08_feature_availability_heatmap")


def figure09_ecg_distributions() -> None:
    raw = pd.read_csv(LLM_MODEL_DIR / "llm_cohort_demographics_ecg_xgb_20260714_102716_raw_feature_matrix.csv")
    features = [
        ("ecg_qt_interval_first", "QT interval"),
        ("ecg_qtc_bazett_first", "Bazett QTc"),
        ("ecg_qrs_axis_first", "QRS axis"),
        ("ecg_t_axis_first", "T axis"),
        ("ecg_rr_interval_first", "RR interval"),
    ]
    fig, axes = plt.subplots(1, len(features), figsize=(16, 5), sharey=False)
    rows = []
    for ax, (col, label) in zip(axes, features):
        data = [
            raw.loc[raw["is_aortic_dissection"].eq(0), col].dropna(),
            raw.loc[raw["is_aortic_dissection"].eq(1), col].dropna(),
        ]
        ax.boxplot(data, tick_labels=["Control", "Target"], patch_artist=True)
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=25)
        for group_name, values in zip(["Control", "Target"], data):
            rows.append({"feature": label, "group": group_name, "n": len(values), "median": values.median(), "iqr": values.quantile(0.75) - values.quantile(0.25)})
    pd.DataFrame(rows).to_csv(CSV_DIR / "fig09_ecg_feature_distribution_summary.csv", index=False)
    fig.suptitle("First in-window ECG feature distributions", fontweight="bold")
    save_figure(fig, "fig09_ecg_feature_distributions")


def bootstrap_ap_ci(y: np.ndarray, p: np.ndarray, seed: int = 42, reps: int = 400) -> tuple[float, float, float]:
    if len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan
    ap = average_precision_score(y, p)
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(average_precision_score(y[idx], p[idx]))
    if not vals:
        return ap, np.nan, np.nan
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return ap, lo, hi


def figure10_subgroup_forest(holdout: pd.DataFrame) -> None:
    df = holdout.copy()
    df["age_group"] = pd.cut(df["index_age"], bins=[0, 55, 65, 75, 200], labels=["<55", "55-64", "65-74", "75+"])
    subgroup_specs = [
        ("Age", "age_group"),
        ("Sex", "gender"),
        ("Race", "race"),
        ("Insurance", "insurance"),
        ("Index source", "index_time_source"),
    ]
    rows = []
    for domain, col in subgroup_specs:
        counts = df[col].fillna("missing").value_counts()
        keep_values = counts.head(5).index
        for value in keep_values:
            sub = df[df[col].fillna("missing").eq(value)]
            if len(sub) < 40 or sub["y_true"].sum() < 3:
                continue
            ap, lo, hi = bootstrap_ap_ci(sub["y_true"].to_numpy(), sub["y_probability"].to_numpy())
            rows.append(
                {
                    "domain": domain,
                    "subgroup": str(value),
                    "n": len(sub),
                    "positives": int(sub["y_true"].sum()),
                    "average_precision": ap,
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
    out = pd.DataFrame(rows).sort_values("average_precision")
    out.to_csv(CSV_DIR / "fig10_subgroup_average_precision.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, max(6, 0.38 * len(out))))
    y = np.arange(len(out))
    ax.errorbar(
        out["average_precision"],
        y,
        xerr=[out["average_precision"] - out["ci_low"], out["ci_high"] - out["average_precision"]],
        fmt="o",
        color=COLORS["blue"],
        ecolor=COLORS["line"],
        capsize=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([f"{d}: {clean_label(s, 4)} (n={n}, pos={p})" for d, s, n, p in zip(out["domain"], out["subgroup"], out["n"], out["positives"])])
    ax.set_xlabel("Average precision, bootstrap 95% CI")
    ax.set_title("Subgroup performance forest plot")
    ax.grid(axis="x", alpha=0.25)
    save_figure(fig, "fig10_subgroup_performance_forest")


def figure11_timing_sensitivity(holdout: pd.DataFrame) -> None:
    df = holdout.copy()
    df["index_time"] = pd.to_datetime(df["index_time"], errors="coerce")
    df["first_ecg_time_24h"] = pd.to_datetime(df["first_ecg_time_24h"], errors="coerce")
    df["hours_to_first_ecg"] = (df["first_ecg_time_24h"] - df["index_time"]).dt.total_seconds() / 3600
    bins = [-0.001, 2, 6, 12, 24]
    labels = ["0-2h", "2-6h", "6-12h", "12-24h"]
    df["timing_bin"] = pd.cut(df["hours_to_first_ecg"], bins=bins, labels=labels)
    rows = []
    for label, sub in df.dropna(subset=["timing_bin"]).groupby("timing_bin", observed=True):
        ap = average_precision_score(sub["y_true"], sub["y_probability"]) if sub["y_true"].nunique() > 1 else np.nan
        roc = roc_auc_score(sub["y_true"], sub["y_probability"]) if sub["y_true"].nunique() > 1 else np.nan
        rows.append({"timing_bin": label, "n": len(sub), "positives": int(sub["y_true"].sum()), "average_precision": ap, "roc_auc": roc})
    out = pd.DataFrame(rows)
    out.to_csv(CSV_DIR / "fig11_timing_sensitivity.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    axes[0].hist(
        [df.loc[df["y_true"].eq(0), "hours_to_first_ecg"].dropna(), df.loc[df["y_true"].eq(1), "hours_to_first_ecg"].dropna()],
        bins=np.linspace(0, 24, 25),
        stacked=True,
        color=[COLORS["teal"], COLORS["red"]],
        label=["Control", "Target"],
    )
    axes[0].set_xlabel("Hours from index to first ECG")
    axes[0].set_ylabel("Holdout encounters")
    axes[0].set_title("First ECG timing distribution")
    axes[0].legend(frameon=False)
    axes[1].bar(out["timing_bin"].astype(str), out["average_precision"], color=COLORS["blue"], alpha=0.85)
    for i, row in enumerate(out.itertuples(index=False)):
        axes[1].text(i, row.average_precision + 0.005, f"n={row.n}\npos={row.positives}", ha="center", fontsize=9)
    axes[1].set_ylabel("Average precision")
    axes[1].set_title("Performance by ECG timing bin")
    axes[1].set_ylim(0, max(0.1, out["average_precision"].max() * 1.35))
    save_figure(fig, "fig11_first_ecg_timing_sensitivity")


def figure12_label_cleanup_yield(derived_summary: dict) -> None:
    target_counts = derived_summary["target_counts"]
    rows = [
        ("Base confirmed new", target_counts["base_confirmed_new_dissection_rows"]),
        ("Excluded Type B", target_counts["excluded_type_b_rows"]),
        ("Excluded prior/context", target_counts["excluded_prior_unrepaired_or_unclear_rows"]),
        ("Final targets", target_counts["final_target_rows"]),
    ]
    df = pd.DataFrame(rows, columns=["category", "count"])
    df["percent_of_base"] = df["count"] / target_counts["base_confirmed_new_dissection_rows"]
    df.to_csv(CSV_DIR / "fig12_label_cleanup_yield.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].bar(df["category"], df["count"], color=[COLORS["purple"], COLORS["red"], COLORS["red"], COLORS["green"]])
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set_ylabel("Rows")
    axes[0].set_title("Target cleanup yield")
    for i, row in enumerate(df.itertuples(index=False)):
        axes[0].text(i, row.count, fmt_int(row.count), ha="center", va="bottom", fontweight="bold")
    stanford = pd.Series(derived_summary["final_target_by_stanford"]).sort_values(ascending=False)
    axes[1].pie(stanford.values, labels=stanford.index, autopct="%1.0f%%", startangle=90, colors=[COLORS["blue"], COLORS["orange"], COLORS["line"]])
    axes[1].set_title("Final targets by Stanford type")
    save_figure(fig, "fig12_label_cleanup_yield")


def figure13_parser_qc(parse_all: pd.DataFrame, parse_summary: dict) -> None:
    manifest = read_json(PARSE_DIR / "manifest.json")
    start = datetime.fromisoformat(manifest["created_at"])
    end = datetime.fromisoformat(parse_summary["completed_at"])
    elapsed_hours = (end - start).total_seconds() / 3600
    throughput = parse_summary["parsed"] / elapsed_hours if elapsed_hours > 0 else np.nan
    qc = pd.DataFrame(
        [
            {"metric": "qualifying_visits_with_ecg_24h", "value": parse_summary["qualifying_visits_with_ecg_24h"]},
            {"metric": "staged_discharge_note_rows", "value": parse_summary["staged_discharge_note_rows"]},
            {"metric": "parsed", "value": parse_summary["parsed"]},
            {"metric": "errors", "value": parse_summary["errors"]},
            {"metric": "elapsed_hours", "value": elapsed_hours},
            {"metric": "parsed_per_hour", "value": throughput},
            {"metric": "parallel_workers", "value": parse_summary["parallel_workers"]},
        ]
    )
    qc.to_csv(CSV_DIR / "fig13_parser_qc_metrics.csv", index=False)
    fig = plt.figure(figsize=(14, 7))
    gs = gridspec.GridSpec(2, 2)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.bar(["Success", "Errors"], [parse_summary["parsed"], parse_summary["errors"]], color=[COLORS["green"], COLORS["red"]])
    ax0.set_yscale("log")
    ax0.set_title("Parse outcomes, log scale")
    ax0.set_ylabel("Visits")
    ax1 = fig.add_subplot(gs[0, 1])
    token_cols = ["usage_input_tokens", "usage_output_tokens", "usage_total_tokens"]
    token_data = [pd.to_numeric(parse_all[col], errors="coerce").dropna() for col in token_cols]
    ax1.boxplot(token_data, tick_labels=["Input", "Output", "Total"], showfliers=False)
    ax1.set_title("Token usage per parsed visit")
    ax1.set_ylabel("Tokens")
    ax2 = fig.add_subplot(gs[1, 0])
    labels = ["Elapsed hours", "Parsed/hour", "Workers"]
    vals = [elapsed_hours, throughput, parse_summary["parallel_workers"]]
    ax2.bar(labels, vals, color=[COLORS["blue"], COLORS["purple"], COLORS["orange"]])
    ax2.set_title("Run throughput summary")
    for i, v in enumerate(vals):
        ax2.text(i, v, f"{v:,.1f}", ha="center", va="bottom", fontsize=10)
    ax3 = fig.add_subplot(gs[1, 1])
    cached = pd.to_numeric(parse_all["usage_cached_tokens"], errors="coerce").fillna(0)
    ax3.hist(cached, bins=30, color=COLORS["teal"], alpha=0.85)
    ax3.set_title("Cached input tokens")
    ax3.set_xlabel("Cached tokens")
    fig.suptitle("Parser operations and QC", fontweight="bold")
    save_figure(fig, "fig13_parser_operations_qc")


def ece_score(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    df = calibration_table(y, p, bins=bins)
    return float(((df["n"] / df["n"].sum()) * (df["observed"] - df["mean_predicted"]).abs()).sum())


def model_summary_rows() -> pd.DataFrame:
    models = [
        (
            "ICD demo+ECG",
            STD_MODEL_DIR / "20260708_151834_demographics_ecg_main_summary_metrics.csv",
            STD_MODEL_DIR / "20260708_151834_demographics_ecg_main_holdout_predictions.csv",
            "low",
        ),
        (
            "LLM demo+ECG",
            LLM_MODEL_DIR / "llm_cohort_demographics_ecg_xgb_20260714_102716_demographics_ecg_first_main_summary_metrics.csv",
            LLM_MODEL_DIR / "llm_cohort_demographics_ecg_xgb_20260714_102716_demographics_ecg_first_main_holdout_predictions.csv",
            "low",
        ),
        (
            "LLM vitals+ECG",
            LLM_VITALS_MODEL_DIR / "llm_cohort_vitals_ecg_xgb_20260713_vitals_ecg_first_main_summary_metrics.csv",
            LLM_VITALS_MODEL_DIR / "llm_cohort_vitals_ecg_xgb_20260713_vitals_ecg_first_main_holdout_predictions.csv",
            "moderate",
        ),
        (
            "LLM labs/vitals/ECG",
            LLM_LAB_VITALS_MODEL_DIR / "llm_cohort_xgb_20260713_lab_vitals_ecg_first_main_summary_metrics.csv",
            LLM_LAB_VITALS_MODEL_DIR / "llm_cohort_xgb_20260713_lab_vitals_ecg_first_main_holdout_predictions.csv",
            "higher",
        ),
        (
            "LLM demo+labs+ECG",
            LLM_LABS_MODEL_DIR / "llm_cohort_demographics_labs_ecg_xgb_20260714_111259_demographics_labs_ecg_first_main_summary_metrics.csv",
            LLM_LABS_MODEL_DIR / "llm_cohort_demographics_labs_ecg_xgb_20260714_111259_demographics_labs_ecg_first_main_holdout_predictions.csv",
            "higher",
        ),
    ]
    rows = []
    for name, summary_path, pred_path, leakage_risk in models:
        if not summary_path.exists():
            continue
        s = pd.read_csv(summary_path).iloc[0].to_dict()
        row = {
            "model": name,
            "feature_count": s.get("feature_count"),
            "leakage_risk": leakage_risk,
            "average_precision": s.get("holdout_average_precision"),
            "roc_auc": s.get("holdout_roc_auc"),
            "selected_precision": s.get("selected_precision"),
            "selected_recall": s.get("selected_recall"),
            "holdout_rows": s.get("holdout_rows"),
            "holdout_positive": s.get("holdout_positive"),
        }
        if pred_path.exists():
            pred = pd.read_csv(pred_path)
            row["ece"] = ece_score(pred["y_true"].to_numpy(), pred["y_probability"].to_numpy())
        rows.append(row)
    return pd.DataFrame(rows)


def figure14_model_comparison_matrix() -> None:
    df = model_summary_rows()
    df.to_csv(CSV_DIR / "fig14_model_comparison_matrix.csv", index=False)
    metrics = ["feature_count", "average_precision", "roc_auc", "selected_precision", "selected_recall", "ece"]
    display = df.set_index("model")[metrics].astype(float)
    scaled = display.copy()
    for col in scaled.columns:
        denom = scaled[col].max() - scaled[col].min()
        scaled[col] = 0.5 if denom == 0 else (scaled[col] - scaled[col].min()) / denom
    fig, ax = plt.subplots(figsize=(12, 5.8))
    im = ax.imshow(scaled.values, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics])
    ax.set_yticks(range(len(display.index)))
    ax.set_yticklabels(display.index)
    for i in range(display.shape[0]):
        for j, col in enumerate(metrics):
            val = display.iloc[i, j]
            text = f"{val:.3f}" if col != "feature_count" else fmt_int(val)
            ax.text(j, i, text, ha="center", va="center", fontsize=10)
    ax.set_title("Model comparison matrix")
    fig.colorbar(im, ax=ax, label="Column-normalized value")
    save_figure(fig, "fig14_model_comparison_matrix")


def figure15_synthetic_vignette() -> None:
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.set_xlim(-0.5, 26)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.plot([0, 24], [2.5, 2.5], color=COLORS["ink"], lw=3)
    events = [
        (0, "Index time\nED registration/admission", COLORS["blue_light"], COLORS["blue"]),
        (1.2, "First ECG\nmachine measurements", COLORS["teal_light"], COLORS["teal"]),
        (6, "Early-window\nfeature vector", COLORS["orange_light"], COLORS["orange"]),
        (18, "Prediction\nrisk score", COLORS["green_light"], COLORS["green"]),
        (24, "Window closes\n+24 hours", COLORS["blue_light"], COLORS["blue"]),
    ]
    for x, label, fill, edge in events:
        ax.plot([x, x], [2.2, 2.8], color=edge, lw=3)
        ax.text(
            x,
            3.4,
            label,
            ha="center",
            va="center",
            fontsize=12,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": fill, "edgecolor": edge, "linewidth": 1.8},
        )
    note_x = 21
    ax.text(
        note_x,
        1.25,
        "Discharge-note parse later supports\nphenotype cleanup and control characterization",
        ha="center",
        va="center",
        fontsize=12,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": COLORS["purple_light"], "edgecolor": COLORS["purple"]},
    )
    ax.add_patch(FancyArrowPatch((18, 2.35), (note_x, 1.75), arrowstyle="-|>", mutation_scale=16, color=COLORS["muted"], lw=2))
    ax.text(12, 0.45, "Synthetic vignette: no patient text or identifiers shown", ha="center", color=COLORS["muted"])
    ax.set_title("Representative de-identified workflow vignette", loc="left", fontsize=18, fontweight="bold")
    save_figure(fig, "fig15_synthetic_workflow_vignette")


def write_readme() -> None:
    text = """# Poster Results Figures

Generated from existing processed artifacts. No LLM calls or model reruns are performed.

Figures:

1. `fig01_cohort_attrition_waterfall`: cohort attrition and derived cohort yield.
2. `fig02_icd_vs_llm_alluvial`: exact ICD status versus LLM phenotype labels.
3. `fig03_calibration_deciles`: calibration plot and decile table.
4. `fig04_decision_curve`: decision-curve analysis.
5. `fig05_risk_distribution_by_phenotype`: holdout risk distributions plus all parsed label counts.
6. `fig06_topk_workload`: review workload at top 1%, 2%, 5%, and 10% predicted risk.
7. `fig07_error_phenotype_review`: false-positive and false-negative phenotype review.
8. `fig08_feature_availability_heatmap`: feature observedness by family and outcome.
9. `fig09_ecg_feature_distributions`: first ECG feature distributions by label.
10. `fig10_subgroup_performance_forest`: subgroup average precision with bootstrap CIs.
11. `fig11_first_ecg_timing_sensitivity`: first ECG timing and performance by timing bin.
12. `fig12_label_cleanup_yield`: LLM target-cleanup yield.
13. `fig13_parser_operations_qc`: parser success, token use, and throughput QC.
14. `fig14_model_comparison_matrix`: model comparison matrix.
15. `fig15_synthetic_workflow_vignette`: synthetic de-identified timeline.

Companion CSVs are in `tables/`.
"""
    (OUT_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    setup_style()
    parse_summary = read_json(PARSE_DIR / "summary.json")
    derived_summary = read_json(DERIVED_DIR / "summary.json")
    holdout = load_llm_holdout()
    parse_all = load_llm_all_parse()
    icd_hadms = exact_icd_hadm_ids()

    figure01_cohort_attrition(parse_summary, derived_summary)
    figure02_icd_vs_llm(parse_all, icd_hadms)
    figure03_calibration(holdout)
    figure04_decision_curve(holdout)
    figure05_risk_distribution(holdout, parse_all)
    figure06_topk_workload(holdout)
    figure07_error_review(holdout)
    figure08_feature_availability()
    figure09_ecg_distributions()
    figure10_subgroup_forest(holdout)
    figure11_timing_sensitivity(holdout)
    figure12_label_cleanup_yield(derived_summary)
    figure13_parser_qc(parse_all, parse_summary)
    figure14_model_comparison_matrix()
    figure15_synthetic_vignette()
    write_readme()

    print("Generated results figures:")
    for path in sorted(OUT_DIR.glob("fig*.png")):
        print(f"  {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
