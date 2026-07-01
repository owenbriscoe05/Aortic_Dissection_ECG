# Repository Guidelines

## Project Purpose

This project uses MIMIC-IV electronic health record data to develop standard EHR-based machine learning models for predicting aortic dissection, with current modeling focused on XGBoost and CatBoost. The active workflow is an automated pipeline in `automated_pipeline/` that is intended to make cohort, feature, preprocessing, and model decision changes easy to control through `config.py`, then rerun so accuracy and related performance changes can be compared.

At this stage, pipeline runtime is a key constraint because each iteration parses large CSV files from the MIMIC-IV data. Changes should preserve the ability to iterate quickly on configuration decisions, and performance improvements that reduce repeated large-table parsing are especially relevant.

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
python main.py
```

This builds temporary cohort/feature CSVs, preprocesses the matrix, and runs the model workflow. For notebook work, start Jupyter from the repository root:

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
