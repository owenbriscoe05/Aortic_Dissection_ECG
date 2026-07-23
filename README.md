# Predicting Acute Aortic Dissection With XGBoost

This project looks at aortic dissection (AD) patients from the MIMIC-IV EHR. Although the final goal of the project was to predict acute (within 24 hours of encounter) AD using a foundation model (FM), this directory only contains more rudimentary XGBoost model workflows.

## Overview

Objective: To develop a model capable of outperforming clinical misdiagnosis rates (which is roughly 1 in 3 people).   
Data source: MIMIC-IV  
Main workflow: Housed under automated_pipeline/  
Current structure:   
Control and case cohorts have been defined using LLM parsing of encounter discharge notes. Subjects are restricted to adults, Stanford Type A dissections, and only those with at least one ECG measurement within the first 24 hours of their initial encounter with the hospital system.

The current workflow will take a changeable configuration (including the ability to change feature set and cohort definition) and fully run the model. Output is ROC and PR AUC plots, SHAP Beeswarm feature importance set, and an html presentation with other metrics. 

## Repository structure
--
.env contains the API keys (Bedrock and Azure)
--
data/mimic-iv/ houses all raw MIMIC-IV data. Included are hosp, icu, ecg, and note modules.  
data/intermediate houses important tables for ICD-coded aortic dissection patients only. This directory is not relevant for the current workflow, which uses LLM-parsed discharge notes to define dissection patients, instead of ICD codes.  
data/processed/pipeline_cache holds a cache that will always be attempted to be used for efficiency purposes. Often, the cache needs to be rebuilt for new feature sets and model parameters.  
data/processed/model_reports holds a massive amount of previous model runs with different feature sets and model specifications. most model run subdirectories include an html presentation with useful figures and metrics.  
notebooks/ holds useful exploratory analyses of the dataset. You can probably find useful statistics about the target and control populations here.

automated_pipeline/ contains the active pipeline
- 'config.py': central configuration for paths, cohort rules, feature windows, ICD code definitions (if using), feature dictionaries, missingness thresholds, cache behavior, model settings, and report options. This is the file to change if making only run-level changes. It's designed to have massive breadth in optionality.
- 'engine.py': Core implementation for cohort construction, feature extraction, preprocessing, train/test splitting, model training, evaluation, threshold summaries, and report artifacts.
- 'cache.py': caching helpers
- 'main.py': entrance point, but not current preferred model entry point
- 'run_*.py': anything beginning with run_ is a specific model type, the current heavy-use script is run_llm_cohort_xgboost_model.py, as it uses the LLM-defined cohort

## Setup

Use the project conda environment:
'''bash  
conda activate ECG

Most model commands should be run from automated_pipeline/ because several scripts use relative paths:

cd automated_pipeline  
python run_llm_cohort_xgboost_model.py

Core dependencies include pandas, numpy, scikit-learn, xgboost, matplotlib, jsonschema, python-dotenv, and openai.

## Security Notes

Do not commit anything in data/, .env, or poster_assets/

## Common Workflows

To run the LLM-derived cohort model:

cd automated_pipeline  
python run_llm_cohort_xgboost_model.py

Use --targets-path and --controls-path when evaluating specific parsed-cohort files. The default expectation is demographics + ECG, but passing --top-labs N enables a lab-inclusive variant with the top N most prevalent labs in the full cohort.

If more parsing using LLMs is necessary, note that the parse_discharge.py script takes a variety of args, some designed to prevent accidental massive API token use. 

Examples:

python automated_pipeline/parse_discharge.py --run --max-visits 10 --parallel-workers 2 --max-in-flight 4   
will allow the LLM to actually run with a maximum of 10 discharge notes to parse. The LLM will run in parallel with 2 workers.

python automated_pipeline/parse_discharge.py --run --hadm-ids 12345678 23456789  
will allow the LLM to actually run on these 2 specific encounters. No parallel work will be done.

If --run is not included, the LLM will not be called and the API token will not be used.

Parser outputs are written under data/processed/openai_discharge_notes_parses/<run-id>/

## Configuration

Most pipeline settings are centralized in automated_pipeline/config.py through 'PipelineConfig'

Important configuration areas:

- Paths:
    - 'DATA_DIR'
    - 'DIAGNOSES_ICD_PATH'
    - 'ECG_MEASUREMENTS_PATH'
    - 'MODEL_REPORT_DIR'

- Cohort rules:
    - 'MIN_AGE'
    - 'USE_EDREGTIME_AS_INDEX_TIME': some patients have an ED registration time earlier than their hadm time, this is an option to use the earlier time instead. +24 hours is enforced regardless
    - 'TARGET_DIAGNOSIS_CODES_BY_VERSION': ICD codes

- Feature window:
    - 'DAY_0_WINDOW_HOURS = 24':

- Feature families:
    - 'USE_ECG_FEATURES'
    - 'VITALS_DICT': itemIDs mapped to column names
    - 'LABS_DICT': itemIDs mapped to column names
    - 'USE_MEDICATION_FEAUTRES = False': currently, no prescription or POC med data is included, but a dict of useful meds is included in config

- 'FEATURE_AGGREGATION_MODE': full, median_only, or first_only (full includes max, min, mean, count)

- Preprocessing:
    - 'ROW_MISSINGNESS_THRESHOLD'
    - 'COL_MISSINGNESS_THRESHOLD'
    - 'LEAKAGE_LABS_TO_DROP': currently contains labs that are typically only ordered for suspected AD patients, eg ddimer and troponin

- Modeling:
    - 'TEST_SIZE'
    - 'VALIDATION_SIZE'
    - 'RANDOM_STATE'
    - 'USE_XGB_NATIVE_CATEGORICAL': xgboost has a native handling of categorical variables, currently True
    - 'USE_RECURSIVE_FEATURE_ELIMINATION': currently True, performs recursive feature removal checks for potential precision improvement
    - 'RFE_PROTECTED_FEATURES': features explicitly kept during feature elimination
    - 'THRESHOLD_SELECTION_METRIC'
    - 'THRESHOLD_RECALL_TARGETS'
    - 'THRESHOLD_FP_BUDGETS'


Note that most current modeling scripts pass in their own modifications to the config file, so the values you see when looking at it are not necessarily "correct" in the current modeling direction.

## Testing / Validation

There is no formal test setup yet.

## Maintainer Notes

Suggested next steps:
- Expand target cohort
- Increase unbiased feature count
- Test other models, XGBoost