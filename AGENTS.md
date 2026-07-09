# Repository Guidelines

## Project Purpose

This project builds standard EHR-based machine learning models for predicting aortic dissection from MIMIC-IV data, currently centered on XGBoost. The active workflow lives in `automated_pipeline/` and is designed to make cohort, feature, preprocessing, feature-selection, and model settings controllable from `config.py` or sweep scripts.

Runtime matters. Prefer cached, chunked, and configuration-driven changes over ad hoc large-table reparsing.

## Current Clinical Modeling Rules

- Target labels and control exclusions must use exact aortic dissection ICD codes only: ICD-9 `441`, `441.00`, `441.01`, `441.03`; ICD-10 `I71.00`, `I71.01`, `I71.010`, `I71.012`, `I71.019`, `I71.03`. Normalize by removing decimals. Do not use broad `441` or `I71` prefix matching.
- Controls are currently restricted to ICD-coded chest pain and back pain candidate admissions only. This is a temporary approximation until discharge-note LLM processing is available to identify the control presentation more accurately.
- Index time is `edregtime` when present and not after `admittime`; otherwise use `admittime`. Preserve `index_time_source`.
- Default feature window is 24 hours from index time. Diagnosis-time censoring is available but should only use real clinical timestamps such as radiology `charttime`, never fallback system/storage timestamps.
- Keep row/column missingness thresholds clinically defensible. Avoid retaining fully empty rows or mostly empty columns unless explicitly justified.
- Use XGBoost native categorical handling by default. One-hot encoding should be an explicit comparison choice.
- Report average precision, PR-AUC, ROC-AUC, F1, threshold operating points, feature importance, and run metadata.
- Protect `race` from automatic feature pruning unless the user explicitly revisits that decision.

## Feature Engineering State

- Current feature aggregation supports three modes via `FEATURE_AGGREGATION_MODE`:
  - `full`: emits min/max/mean aggregates and first values where enabled.
  - `median_only`: emits medians only; current expanded sweeps use no first features.
  - `first_only`: emits only the first in-window value for each feature.
- The active post-presentation direction is an ECG-complete lab/vitals/ECG first-value model: select the top 50 most prevalent numeric in-window labs across the full configured model cohort before train/test splitting, add vitals and ECG machine measurements, require at least one ECG machine-measurement row in the 24-hour feature window, and use first values only for all three feature families.
- Each lab/vitals run must write a lab prevalence table with control-set and target-set prevalence for the selected labs.
- Each model run should write an HTML presentation with relevant plots, including PR/ROC curves and SHAP beeswarm, threshold operating points, model metrics, feature summaries, run metadata, and cohort/feature caveats.
- ECG machine measurements are part of the active default model and should use first-only aggregation. Derived ECG features include QT interval and Bazett QTc.
- Medication features are experimental and grouped from `prescriptions.csv` by clinically relevant drug-name patterns. They should be evaluated as report-producing comparisons before any default adoption.
- Additional lab panels are experimental in `run_expanded_feature_sweep.py`; do not treat them as default without reviewing model performance and leakage implications.

## Feature Selection

- Recursive feature elimination should choose features using the training validation split, not the holdout set.
- Greedy backward ablation may accept a removal only when validation average precision improves or stays within the explicitly configured tolerance.
- RFE should return the best validation-AP subset encountered along the tolerated-drop path, not merely the final subset before stopping.
- Shadow-feature filtering is report-first: classify features as keep/tentative/reject, then review before permanent drops.

## Active Sweep Scripts

- `run_feature_sweep.py` runs aggregation-mode and RFE-tolerance sweeps across `full` and `median_only`.
- `run_expanded_feature_sweep.py` waits for the first sweep if requested, then tests median-only expanded lab and medication variants.
- `run_lab_vitals_first_model.py` is the current active model runner despite the historical filename. It builds a unique folder under `data/processed/model_reports/lab_vitals_ecg_first_runs/`, selects top-50 prevalent labs, reports target/control lab prevalence, trains the first-only labs/vitals/ECG model, and generates the associated HTML presentation.
- `run_lab_vitals_first_model.py --require-core-vitals-complete` is an experimental sensitivity run, not the default; it keeps only rows with `heart_rate_first`, `resp_rate_first`, and either complete NIBP or complete ABP first values.
- `run_lab_vitals_first_model.py --require-heart-rate-resp-complete` is a separate experimental sensitivity run for requiring only `heart_rate_first` and `resp_rate_first`; it can be combined with `--drop-features` to remove BP features from modeling.
- `run_lab_vitals_first_catboost_model.py` runs the same ECG-complete first-only labs/vitals/ECG feature matrix with CatBoost, writing equivalent metrics, lab prevalence, PR/ROC plots, SHAP plots, and HTML presentation artifacts under `data/processed/model_reports/lab_vitals_ecg_first_catboost_runs/`.
- Long sweeps may run in tmux. Check logs under `data/processed/model_reports/feature_sweeps/` or `data/processed/model_reports/expanded_feature_sweeps/`.

## Archived Historical Run Handoff

This handoff is stale as of 2026-07-08 and should not be treated as the current starting point. As of 2026-07-06, these long-running sweep sessions were expected:

- `feature_sweep_20260706_171031`
  - Script: `automated_pipeline/run_feature_sweep.py`
  - Log: `data/processed/model_reports/feature_sweeps/20260706_171031/sweep.log`
  - Summary: `data/processed/model_reports/feature_sweeps/20260706_171031/20260706_171031_sweep_summary.csv`
  - Scope: `full` and `median_only` aggregation modes across RFE tolerances `0.005`, `0.01`, and `0.02`.
- `expanded_feature_sweep_20260706_172057`
  - Script: `automated_pipeline/run_expanded_feature_sweep.py`
  - Log: `data/processed/model_reports/expanded_feature_sweeps/20260706_172057/expanded_sweep.log`
  - Summary: `data/processed/model_reports/expanded_feature_sweeps/20260706_172057/20260706_172057_expanded_sweep_summary.csv`
  - Scope: waits for the first sweep to finish, then runs median-only expanded lab and medication variants across the same RFE tolerances.

To resume old sweeps after a disconnect, first run `tmux ls`, then inspect the logs and summary CSVs. When sweeps finish, compare ranked summary files by holdout average precision, PR-AUC, ROC-AUC, F1, and feature count. Do not promote expanded labs or medication features to defaults unless the user explicitly revisits that decision.

## Project Structure & Module Organization

This repository contains a Python-based clinical data and modeling workflow for aortic dissection work on MIMIC-IV data. Core pipeline code lives in `automated_pipeline/`:

- `main.py` orchestrates cohort construction, feature extraction, preprocessing, and model evaluation.
- `config.py` stores paths, cohort settings, feature dictionaries, leakage exclusions, and model parameters.
- `engine.py` implements `DataBuilder` and `ModelEngine`.

Exploratory and model-development notebooks live in `notebooks/`. Raw and derived datasets are under `data/`, with MIMIC-IV source tables in `data/mimic-iv/`, intermediate CSVs in `data/intermediate/`, and processed outputs in `data/processed/`. The `src/` directory currently has no tracked source files.

## Build, Test, and Development Commands

No package metadata or Makefile is currently present. Run commands from `automated_pipeline/` because imports and relative data paths are written for that location:

```bash
cd automated_pipeline
python run_lab_vitals_first_model.py
```

This builds temporary cohort/feature CSVs, preprocesses the matrix, runs the active model workflow, and writes reports plus HTML presentation artifacts under a unique folder in `data/processed/model_reports/lab_vitals_ecg_first_runs/`. For notebook work, start Jupyter from the repository root:

```bash
jupyter lab notebooks
```

Use the project conda environment named `ECG` for development and pipeline work:

```bash
conda activate ECG
```

This environment should include at least `pandas`, `scikit-learn`, `xgboost`, and `jupyter`.

## Coding Style & Naming Conventions

Use Python 3 and follow PEP 8 where practical: 4-space indentation, snake_case for functions and variables, PascalCase for classes, and uppercase names for configuration constants. Keep pipeline settings in `PipelineConfig` rather than scattering literals through the code. Prefer chunked reads for large MIMIC tables, matching the existing memory-conscious style in `engine.py`.

## Testing Guidelines

There is no formal test suite yet. When adding tests, place them in a top-level `tests/` directory and name files `test_*.py`. Use small sample data from `data/mimic-iv/sample_data/` or synthetic fixtures; do not require full MIMIC-IV tables for unit tests. Validate new feature builders by checking expected columns, row counts, missingness handling, and temporary file cleanup.

## Commit & Pull Request Guidelines

Git history is not available from this working directory, so use concise imperative commit messages such as `Add lab feature aggregation` or `Fix cohort filtering`. Pull requests should describe the clinical/data change, list commands run, note required data files, and call out any leakage, cohort definition, or reproducibility implications.

## Security & Configuration Tips

Treat MIMIC-IV data and derived cohorts as controlled clinical data. Do not commit raw data exports, patient-level extracts, credentials, or ad hoc output files. Keep path changes centralized in `automated_pipeline/config.py`.
