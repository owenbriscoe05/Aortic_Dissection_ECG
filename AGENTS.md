# Repository Guidelines

## Project Purpose

This project builds EHR-based workflows for aortic dissection work on MIMIC-IV data. The active code lives in `automated_pipeline/` and currently covers:

- ECG-complete early-window tabular models for aortic dissection prediction.
- Discharge-note LLM parsing for cohort phenotyping, target cleanup, and control characterization.

Runtime matters. Prefer chunked, resumable, configuration-driven processing over ad hoc full-table reparsing. Treat all MIMIC-IV source data and derived patient-level outputs as controlled clinical data.

## Current Clinical Rules

- Target labels and control exclusions must use exact aortic dissection ICD codes only: ICD-9 `441`, `441.00`, `441.01`, `441.03`; ICD-10 `I71.00`, `I71.01`, `I71.010`, `I71.012`, `I71.019`, `I71.03`. Normalize by removing decimals. Do not use broad `441` or `I71` prefix matching.
- Index time is `edregtime` when present and not after `admittime`; otherwise use `admittime`. Preserve `index_time_source`.
- Default feature window is 24 hours from index time.
- Discharge-note parsing is intended for every qualifying admission with at least one ECG in the first 24 hours and at least one attached non-empty discharge note. Do not limit parser work to old chest/back-pain control rules unless the user explicitly asks.
- Diagnosis-time censoring, when used, should rely only on real clinical timestamps such as radiology `charttime`, never fallback system/storage timestamps.
- Keep row/column missingness thresholds clinically defensible. Avoid retaining fully empty rows or mostly empty columns unless explicitly justified.
- Use XGBoost native categorical handling by default. One-hot encoding should be an explicit comparison choice.
- Report average precision, PR-AUC, ROC-AUC, F1, threshold operating points, feature importance, and run metadata.
- Protect `race` from automatic feature pruning unless the user explicitly revisits that decision.

## Discharge-Note LLM Parser

`automated_pipeline/parse_discharge.py` is the active discharge-note parser. It uses the OpenAI Python SDK against Azure's OpenAI-compatible endpoint, not DSPy.

- Default model is currently `gpt-5.6-luna`.
- Credentials come from `.env`: `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_BASE_URL`.
- Default Azure OpenAI-compatible base URL is `${AZURE_OPENAI_BASE_URL}/openai/v1`.
- The parser uses Responses API structured outputs with a strict JSON Schema and local `jsonschema` validation.
- It sends only discharge-note text to the LLM. Do not include `subject_id`, `hadm_id`, note IDs, note timestamps, or visit metadata in the model input unless explicitly requested.
- Output records may retain metadata for auditability.
- Do not add application-level parser caching or stored responses. Requests use `store=False`; prompt cache keys and retention are not set.
- Full paid execution is guarded: `--run` is required, and unbounded full runs also require `--allow-full-run`.
- For bounded tests, use `--max-visits N`; for exact encounter tests, use `--hadm-ids`.
- Parallel execution is supported with `--parallel-workers` and `--max-in-flight`. Keep worker counts conservative until throughput, error rates, and Azure limits are known.
- Results stream incrementally to JSONL and CSV:
  - `openai_discharge_visit_parse_results.jsonl`
  - `openai_discharge_visit_parse_results.csv`
  - `openai_discharge_visit_parse_errors.jsonl`
  - `openai_discharge_visit_parse_errors.csv`

Example bounded test, only after the user explicitly asks for paid calls:

```bash
python automated_pipeline/parse_discharge.py --run --max-visits 10 --parallel-workers 2 --max-in-flight 4
```

For a future full run, confirm the exact model, visit scope, worker count, Azure limits, expected cost posture, output path, and guardrails with the user before running anything.

Do not run paid LLM parsing unless the user explicitly asks in the current exchange.

## Active Tabular Model State

- Current feature aggregation supports three modes via `FEATURE_AGGREGATION_MODE`:
  - `full`: emits min/max/mean aggregates and first values where enabled.
  - `median_only`: emits medians only.
  - `first_only`: emits only the first in-window value for each feature.
- The active post-presentation tabular model direction is ECG-complete first-value labs/vitals/ECG:
  - Select the top 50 most prevalent numeric in-window labs across the configured model cohort before train/test splitting.
  - Add vitals and ECG machine measurements.
  - Require at least one ECG machine-measurement row in the 24-hour feature window.
  - Use first values for labs, vitals, and ECG features.
- Each lab/vitals/ECG run should write a lab prevalence table with control-set and target-set prevalence for selected labs.
- Each model run should write an HTML presentation with PR/ROC curves, SHAP beeswarm, threshold operating points, model metrics, feature summaries, run metadata, and cohort/feature caveats.
- ECG machine measurements are part of the active default model and should use first-only aggregation. Derived ECG features include QT interval and Bazett QTc.
- Medication features are experimental and should be evaluated as report-producing comparisons before any default adoption.

## Feature Selection

- Recursive feature elimination should choose features using the training validation split, not the holdout set.
- Greedy backward ablation may accept a removal only when validation average precision improves or stays within the explicitly configured tolerance.
- RFE should return the best validation-AP subset encountered along the tolerated-drop path, not merely the final subset before stopping.
- Shadow-feature filtering is report-first: classify features as keep/tentative/reject, then review before permanent drops.

## Key Scripts

- `run_lab_vitals_first_model.py`: active XGBoost ECG-complete first-only labs/vitals/ECG runner. Writes reports under `data/processed/model_reports/lab_vitals_ecg_first_runs/`.
- `run_lab_vitals_first_catboost_model.py`: CatBoost comparison for the same ECG-complete first-only feature matrix.
- `parse_discharge.py`: OpenAI-on-Azure structured discharge-note parser for phenotype labels.
- `run_feature_sweep.py` and `run_expanded_feature_sweep.py`: historical/experimental sweep scripts. Do not treat old sweep outputs as the current starting point without checking freshness and user intent.

## Project Structure

Core pipeline code lives in `automated_pipeline/`:

- `main.py` orchestrates cohort construction, feature extraction, preprocessing, and model evaluation.
- `config.py` stores paths, cohort settings, feature dictionaries, leakage exclusions, and model parameters.
- `engine.py` implements `DataBuilder` and `ModelEngine`.
- `parse_discharge.py` implements discharge-note LLM parsing.

Exploratory and model-development notebooks live in `notebooks/`. Raw and derived datasets are under `data/`, with MIMIC-IV source tables in `data/mimic-iv/`, intermediate CSVs in `data/intermediate/`, and processed outputs in `data/processed/`.

## Development Commands

Use the project conda environment named `ECG`:

```bash
conda activate ECG
```

Run tabular pipeline commands from `automated_pipeline/` when they rely on local relative imports or paths:

```bash
cd automated_pipeline
python run_lab_vitals_first_model.py
```

For the discharge parser, commands are normally run from the repository root:

```bash
python automated_pipeline/parse_discharge.py --help
```

For notebook work:

```bash
jupyter lab notebooks
```

## Coding Style

Use Python 3 with PEP 8 where practical: 4-space indentation, snake_case for functions and variables, PascalCase for classes, and uppercase names for configuration constants. Keep pipeline settings centralized in `PipelineConfig` or script-level argparse defaults rather than scattering literals.

Prefer structured APIs and parsers over ad hoc string manipulation. Prefer chunked reads for large MIMIC tables. Keep edits scoped and avoid unrelated refactors.

## Testing Guidelines

There is no formal test suite yet. When adding tests, place them in top-level `tests/` and name files `test_*.py`. Use small sample data from `data/mimic-iv/sample_data/` or synthetic fixtures; do not require full MIMIC-IV tables for unit tests.

For parser changes, validate without paid calls first:

```bash
python -m py_compile automated_pipeline/parse_discharge.py
python automated_pipeline/parse_discharge.py --help
python automated_pipeline/parse_discharge.py
python automated_pipeline/parse_discharge.py --run
```

The final command should refuse unbounded paid execution unless `--max-visits` or `--allow-full-run` is supplied.

## Security And Data Handling

- Do not commit raw MIMIC-IV data, patient-level extracts, credentials, or ad hoc output files.
- Treat generated note-parse outputs, full LLM input/output review files, and model cohorts as controlled clinical data.
- Keep API keys in `.env`; never print secret values.
- Before running paid or large-scale LLM parsing, state the exact command and confirm the guardrails: model, max visits or full-run flag, worker count, caching/storage posture, and output paths.
