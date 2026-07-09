import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


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


def plot_pr_curve(pr_path, summary, output_path):
    df = pd.read_csv(pr_path)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=160)
    ax.plot(df["recall"], df["precision"], color="#1f77b4", linewidth=2.2)
    prevalence = summary["holdout_prevalence"]
    ax.axhline(prevalence, color="#777777", linestyle="--", linewidth=1, label=f"Prevalence {prevalence:.2%}")
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
    ax.plot(df["false_positive_rate"], df["true_positive_rate"], color="#2ca02c", linewidth=2.2)
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


def metric(summary, name, digits=3):
    return f"{summary[name]:.{digits}f}"


def log_line_containing(report_dir, needle):
    log_path = report_dir / "run_console.log"
    if not log_path.exists():
        return None
    for line in log_path.read_text(errors="ignore").splitlines():
        if needle in line:
            return line.strip()
    return None


def main():
    parser = argparse.ArgumentParser(description="Create a short Markdown presentation for the ECG-complete medication run.")
    parser.add_argument(
        "--report-dir",
        default=None,
        help="Run report directory. Defaults to the latest ecg_complete_medication_runs directory.",
    )
    args = parser.parse_args()

    base_dir = Path("../data/processed/model_reports/ecg_complete_medication_runs")
    report_dir = Path(args.report_dir) if args.report_dir else latest_run_dir(base_dir)
    summary_path = first_match(report_dir, "_main_summary_metrics.csv")
    cv_path = first_match(report_dir, "_main_cv_metrics.csv")
    threshold_path = first_match(report_dir, "_main_threshold_metrics.csv")
    pr_path = first_match(report_dir, "_main_pr_curve.csv")
    roc_path = first_match(report_dir, "_main_roc_curve.csv")
    importance_path = first_match(report_dir, "_main_feature_importance.csv")

    summary = pd.read_csv(summary_path).iloc[0]
    cv = pd.read_csv(cv_path)
    thresholds = pd.read_csv(threshold_path)
    importance = pd.read_csv(importance_path)
    manifest_path = report_dir / f"{report_dir.name}_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    use_clinically_similar_controls = manifest.get("use_clinically_similar_controls", True)
    preprune_median_labs = manifest.get("preprune_median_labs", False)
    control_cohort_text = (
        "Controls were restricted to configured clinically similar ICD diagnosis groups."
        if use_clinically_similar_controls
        else "Controls were not restricted to the configured clinically similar ICD diagnosis groups."
    )
    feature_setup_text = (
        "Used median-only base labs/vitals/ECG features before recursive pruning; medication features were disabled."
        if preprune_median_labs
        else "Completed feature sweeps and retained the medication tolerance 0.01 feature set as the active fixed feature list."
    )
    ecg_filter_line = log_line_containing(report_dir, "ECG-complete cohort filter:")

    pr_png = report_dir / "ecg_complete_pr_curve.png"
    roc_png = report_dir / "ecg_complete_roc_curve.png"
    plot_pr_curve(pr_path, summary, pr_png)
    plot_roc_curve(roc_path, summary, roc_png)

    validation_row = thresholds[thresholds["rule"].eq("validation_f1")].iloc[0]
    default_row = thresholds[thresholds["rule"].eq("default_0.5")].iloc[0]
    top_features = importance.head(8)["Feature"].tolist()

    slides = f"""# Aortic Dissection EHR Model Update

## Work completed

- Updated cohort logic to use exact aortic dissection ICD codes.
- {control_cohort_text}
- Added median-only feature aggregation, ECG machine measurements, and experimental medication groups.
- {feature_setup_text}
- Preserved the labs-only pre-medication feature list as a documented revert option.
- Ran the new ECG-complete experiment: targets and controls without in-window ECG machine measurements were removed.

---

## ECG-complete medication model

- Feature preset: `{manifest.get('feature_preset', 'medication_tol0p01')}`
- Feature count: {int(summary['feature_count'])}
- Training rows: {int(summary['train_rows']):,}; positives: {int(summary['train_positive']):,}; prevalence: {summary['train_prevalence']:.2%}
- Holdout rows: {int(summary['holdout_rows']):,}; positives: {int(summary['holdout_positive']):,}; prevalence: {summary['holdout_prevalence']:.2%}
- 5-fold CV AP: {summary['cv_average_precision_mean']:.3f} +/- {summary['cv_average_precision_sd']:.3f}
- 5-fold CV ROC-AUC: {summary['cv_roc_auc_mean']:.3f} +/- {summary['cv_roc_auc_sd']:.3f}
- {ecg_filter_line if ecg_filter_line else "ECG-complete filter details are available in the run log."}

---

## Holdout performance

- Average precision: {metric(summary, 'holdout_average_precision')}
- PR-AUC, trapezoidal: {metric(summary, 'holdout_pr_auc_trapezoidal')}
- ROC-AUC: {metric(summary, 'holdout_roc_auc')}
- Tuned threshold: {summary['selected_threshold']:.3f}
- Tuned precision/recall/F1: {validation_row['precision']:.3f} / {validation_row['recall']:.3f} / {validation_row['f1']:.3f}
- Default threshold F1: {default_row['f1']:.3f}

![Precision-Recall curve](ecg_complete_pr_curve.png)

---

## ROC curve

![ROC curve](ecg_complete_roc_curve.png)

---

## Feature set and interpretation notes

- Top features: {", ".join(f"`{feature}`" for feature in top_features)}
- Medication features remain experimental and should be reviewed for treatment/ordering leakage before default adoption.
- The revert feature list is written beside the run reports as `*_revert_labs_only_features.txt`.
- The active feature list is written as `*_active_features.txt`.
"""

    presentation_path = report_dir / "ecg_complete_model_update.md"
    presentation_path.write_text(slides)
    print(f"Presentation: {presentation_path}")
    print(f"PR curve: {pr_png}")
    print(f"ROC curve: {roc_png}")


if __name__ == "__main__":
    main()
