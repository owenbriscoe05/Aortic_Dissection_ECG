# Suggested Results Figures Beyond Current Model Presentations

These are designed to complement existing PR/ROC curves, SHAP plots, threshold tables, and summary metrics.

1. Cohort attrition waterfall or Sankey: admissions to adult eligible visits, ECG-complete visits, discharge-note-qualified visits, LLM-confirmed targets, excluded target phenotypes, and final controls.
2. ICD versus LLM phenotype alluvial plot: exact ICD target status flowing into newly identified dissection, previously known dissection only, aneurysm/no dissection, ruled out, unclear, and control-useful groups.
3. Calibration plot plus calibration table: predicted risk deciles with observed prevalence, confidence intervals, and counts per decile.
4. Decision-curve analysis: net benefit across clinically plausible thresholds compared with treat-all and treat-none strategies.
5. Risk score distribution by phenotype: violin or ridgeline plots for confirmed targets, useful controls, known/prior dissections, aneurysm-only cases, and ruled-out cases.
6. Top-k workload figure: sensitivity, PPV, and false-positive burden when reviewing the top 1%, 2%, 5%, and 10% of predicted-risk admissions.
7. False-positive and false-negative phenotype review: compact bar chart showing dominant alternative diagnoses, symptom patterns, or LLM labels among model errors.
8. Feature availability heatmap: missingness by feature family and outcome label, especially ECG machine-measurement completeness and demographic fields.
9. ECG feature distribution panels: clinically interpretable plots for QT interval, Bazett QTc, QRS axis, T axis, and RR interval by target/control label, with effect sizes.
10. Subgroup performance forest plot: average precision or sensitivity at a fixed specificity by age group, sex, race, insurance, index-time source, and control cohort definition.
11. Timing sensitivity figure: distribution of time from index to first ECG, plus model performance stratified by first ECG within 0-2h, 2-6h, 6-12h, and 12-24h.
12. Label-cleanup yield chart: proportion of ICD candidates removed for prior known dissection, chronic Type B, aneurysm without dissection, unclear evidence, or other aortic syndrome.
13. Parser operations/QC panel: parse success rate, schema-validation failures, error categories, tokens per visit, and throughput by worker count for the completed LLM run.
14. Model comparison matrix: active demographics+ECG model versus comparison feature sets, with columns for feature burden, leakage risk, average precision, calibration, and operating-point workload.
15. Representative de-identified workflow vignette: not patient text, but a synthetic timeline showing ED/admission index, first ECG, discharge-note phenotype extraction, and model prediction.
