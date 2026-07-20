# Poster Results Figures

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
