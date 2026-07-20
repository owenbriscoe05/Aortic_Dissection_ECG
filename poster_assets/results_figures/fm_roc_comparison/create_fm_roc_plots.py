#!/usr/bin/env python3
"""Plot FM and XGBoost AUROC curves from existing ROC CSV artifacts."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent

FM_ROC_PATH = PROJECT_ROOT / "data/intermediate/jonfm_full_refit_heldout_roc_curve.csv"

XGB_DEMOS_ROC_PATH = (
    PROJECT_ROOT
    / "data/processed/model_reports/llm_demographics_ecg_runs/"
    / "llm_cohort_demographics_ecg_xgb_20260714_102716/"
    / "llm_cohort_demographics_ecg_xgb_20260714_102716_demographics_ecg_first_main_roc_curve.csv"
)
XGB_DEMOS_SUMMARY_PATH = (
    PROJECT_ROOT
    / "data/processed/model_reports/llm_demographics_ecg_runs/"
    / "llm_cohort_demographics_ecg_xgb_20260714_102716/"
    / "llm_cohort_demographics_ecg_xgb_20260714_102716_demographics_ecg_first_main_summary_metrics.csv"
)

XGB_DEMOS_LABS_ROC_PATH = (
    PROJECT_ROOT
    / "data/processed/model_reports/llm_demographics_labs_ecg_runs/"
    / "llm_cohort_demographics_labs_ecg_xgb_20260714_111259/"
    / "llm_cohort_demographics_labs_ecg_xgb_20260714_111259_demographics_labs_ecg_first_main_roc_curve.csv"
)
XGB_DEMOS_LABS_SUMMARY_PATH = (
    PROJECT_ROOT
    / "data/processed/model_reports/llm_demographics_labs_ecg_runs/"
    / "llm_cohort_demographics_labs_ecg_xgb_20260714_111259/"
    / "llm_cohort_demographics_labs_ecg_xgb_20260714_111259_demographics_labs_ecg_first_main_summary_metrics.csv"
)


COLORS = {
    "fm": "#6B4E9B",
    "xgb_demos": "#2C7A7B",
    "xgb_demos_labs": "#B36B00",
    "chance": "#7B8794",
    "ink": "#1F2933",
}


def load_fm_curve() -> tuple[pd.DataFrame, float, str]:
    df = pd.read_csv(FM_ROC_PATH)
    required = {"fpr", "tpr", "auroc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"FM ROC file missing columns: {sorted(missing)}")
    model_name = str(df["model"].dropna().iloc[0]) if "model" in df.columns else "JonFM"
    auroc = float(df["auroc"].dropna().iloc[0])
    curve = df[["fpr", "tpr"]].copy()
    return curve, auroc, model_name


def load_xgb_curve(roc_path: Path, summary_path: Path) -> tuple[pd.DataFrame, float]:
    roc = pd.read_csv(roc_path)
    required = {"false_positive_rate", "true_positive_rate"}
    missing = required - set(roc.columns)
    if missing:
        raise ValueError(f"XGBoost ROC file missing columns: {sorted(missing)}")
    summary = pd.read_csv(summary_path).iloc[0]
    curve = roc.rename(
        columns={
            "false_positive_rate": "fpr",
            "true_positive_rate": "tpr",
        }
    )[["fpr", "tpr"]].copy()
    return curve, float(summary["holdout_roc_auc"])


def style_ax(ax: plt.Axes, title: str) -> None:
    ax.plot([0, 1], [0, 1], linestyle="--", color=COLORS["chance"], linewidth=1.5, label="Chance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title, fontsize=16, fontweight="bold", color=COLORS["ink"])
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="lower right")


def save(fig: plt.Figure, stem: str) -> None:
    for suffix in [".png", ".svg"]:
        fig.savefig(OUT_DIR / f"{stem}{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fm_curve, fm_auc, fm_name = load_fm_curve()
    xgb_demos_curve, xgb_demos_auc = load_xgb_curve(XGB_DEMOS_ROC_PATH, XGB_DEMOS_SUMMARY_PATH)
    xgb_demos_labs_curve, xgb_demos_labs_auc = load_xgb_curve(
        XGB_DEMOS_LABS_ROC_PATH,
        XGB_DEMOS_LABS_SUMMARY_PATH,
    )

    summary = pd.DataFrame(
        [
            {
                "model": fm_name,
                "display_label": "FM",
                "auroc": fm_auc,
                "source": str(FM_ROC_PATH.relative_to(PROJECT_ROOT)),
            },
            {
                "model": "XGBoost demographics + ECG",
                "display_label": "XGBoost ECG+demos",
                "auroc": xgb_demos_auc,
                "source": str(XGB_DEMOS_ROC_PATH.relative_to(PROJECT_ROOT)),
            },
            {
                "model": "XGBoost demographics + labs + ECG",
                "display_label": "XGBoost ECG+demos+labs",
                "auroc": xgb_demos_labs_auc,
                "source": str(XGB_DEMOS_LABS_ROC_PATH.relative_to(PROJECT_ROOT)),
            },
        ]
    )
    summary.to_csv(OUT_DIR / "auroc_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.plot(
        fm_curve["fpr"],
        fm_curve["tpr"],
        color=COLORS["fm"],
        linewidth=2.6,
        label=f"FM AUROC={fm_auc:.3f}",
    )
    style_ax(ax, "FM Heldout ROC Curve")
    save(fig, "jonfm_full_refit_heldout_roc_curve")

    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    ax.plot(
        xgb_demos_curve["fpr"],
        xgb_demos_curve["tpr"],
        color=COLORS["xgb_demos"],
        linewidth=2.5,
        label=f"XGBoost ECG+demos AUROC={xgb_demos_auc:.3f}",
    )
    ax.plot(
        xgb_demos_labs_curve["fpr"],
        xgb_demos_labs_curve["tpr"],
        color=COLORS["xgb_demos_labs"],
        linewidth=2.5,
        label=f"XGBoost ECG+demos+labs AUROC={xgb_demos_labs_auc:.3f}",
    )
    ax.plot(
        fm_curve["fpr"],
        fm_curve["tpr"],
        color=COLORS["fm"],
        linewidth=2.5,
        label=f"FM AUROC={fm_auc:.3f}",
    )
    style_ax(ax, "Heldout ROC Curve Comparison")
    save(fig, "combined_fm_xgboost_roc_curves")

    print("Wrote:")
    for path in sorted(OUT_DIR.iterdir()):
        if path.suffix in {".csv", ".png", ".svg"}:
            print(f"  {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
