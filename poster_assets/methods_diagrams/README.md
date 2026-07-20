# Poster Methods Diagrams

Editable SVG files and high-resolution PNG previews for the aortic-dissection ECG project.

Files:

- `01_end_to_end_workflow.svg`: overall MIMIC-IV to model and LLM-cohort workflow.
- `02_cohort_and_feature_window.svg`: cohort definition, index-time rule, and 24-hour feature window.
- `03_active_predictor_set.svg`: active demographics plus ECG predictor set and default exclusions.
- `04_discharge_note_llm_parser.svg`: discharge-note structured-output phenotyping workflow.
- `05_model_training_and_evaluation.svg`: model preprocessing, validation, holdout evaluation, and reporting.
- `06_numeric_icd_ecg_cohort.svg`: number-oriented ICD-based ECG-complete cohort attrition.
- `07_numeric_llm_parse_yield.svg`: number-oriented discharge-note LLM parse yield.
- `08_numeric_llm_cohort_derivation.svg`: number-oriented LLM target/control derivation.
- `09_numeric_model_cohort_comparison.svg`: number-oriented side-by-side cohort and split comparison.

The SVGs are the preferred poster assets because they remain sharp at print scale. PNG copies are exported at 3200 px width for quick insertion into slides or drafts.

Regenerate assets with:

```bash
python poster_assets/methods_diagrams/draw_methods_diagrams.py
```
