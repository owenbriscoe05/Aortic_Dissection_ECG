# config.py
from pathlib import Path

class PipelineConfig:
    # ==========================================
    # 1. PATHS & DIRECTORIES
    # ==========================================
    DATA_DIR = Path("../data/mimic-iv")
    TARGET_MACRO_PATH = "../data/intermediate/aortic_dissection_macro_visits.csv"
    DIAGNOSES_ICD_PATH = DATA_DIR / "hosp" / "diagnoses_icd.csv"
    RADIOLOGY_NOTES_PATH = DATA_DIR / "mimic-iv-note" / "2.2" / "note" / "radiology.csv"
    RADIOLOGY_DETAIL_PATH = DATA_DIR / "mimic-iv-note" / "2.2" / "note" / "radiology_detail.csv"
    CACHE_DIR = Path("../data/processed/pipeline_cache")

    # ==========================================
    # 2. COHORT DECISIONS
    # ==========================================
    MIN_AGE = 18
    USE_EDREGTIME_AS_INDEX_TIME = True
    # Keep the evaluation split at natural prevalence. Training imbalance is
    # handled by XGBoost scale_pos_weight rather than random cohort downsampling.
    CONTROL_DOWNSAMPLE_RATIO = None
    USE_NATURAL_PREVALENCE_HOLDOUT = True
    TARGET_DIAGNOSIS_CODES_BY_VERSION = {
        9: ["441", "44100", "44101", "44103"],
        10: ["I7100", "I7101", "I71010", "I71012", "I71019", "I7103"],
    }
    USE_CLINICALLY_SIMILAR_CONTROLS = True
    CLINICALLY_SIMILAR_CONTROL_GROUPS = {
        "chest_pain": {
            9: ["7865"],
            10: ["R07"],
        },
        "back_pain": {
            9: ["7245"],
            10: ["M54"],
        },
    }
    CONTROL_EXCLUSION_DIAGNOSIS_CODES_BY_VERSION = TARGET_DIAGNOSIS_CODES_BY_VERSION
    USE_TRAINING_HARD_CONTROLS = False
    HARD_CONTROL_ADMISSION_TYPES = {
        "EW EMER.",
        "URGENT",
        "DIRECT EMER.",
        "EU OBSERVATION",
        "OBSERVATION ADMIT",
    }
    HARD_CONTROL_ADMISSION_LOCATIONS = {
        "EMERGENCY ROOM",
        "WALK-IN/SELF REFERRAL",
        "TRANSFER FROM HOSPITAL",
        "CLINIC REFERRAL",
    }

    # ==========================================
    # 3. FEATURE ENGINEERING DECISIONS
    # ==========================================
    DAY_0_WINDOW_HOURS = 24
    USE_DIAGNOSIS_TIME_CENSORING = False
    DIAGNOSIS_TIME_SOURCE = "radiology_charttime_text"
    DIAGNOSIS_TIME_REQUIRE_RESULT_SECTION = True
    DIAGNOSIS_TIME_RESULT_SECTION_HEADERS = ["FINDINGS", "IMPRESSION"]
    DIAGNOSIS_TIME_REQUIRE_DIAGNOSTIC_EXAM = True
    DIAGNOSIS_TIME_EXAM_INCLUDE_PATTERNS = [
        r"\bCTA\b",
        r"\bCT\b",
        r"\bMRA\b",
        r"\bMRI\b",
        r"\bCARDIAC STRUCTURE\b",
    ]
    DIAGNOSIS_TIME_EXAM_ANATOMY_PATTERNS = [
        r"\bCHEST\b",
        r"\bABD\b",
        r"\bABDOMEN\b",
        r"\bPELVIS\b",
        r"\bAORTA\b",
        r"\bTHORACIC\b",
        r"\bCARDIAC\b",
    ]
    DIAGNOSIS_TIME_EXAM_EXCLUDE_PATTERNS = [
        r"\bCHEST \(PORTABLE",
        r"\bCHEST PORT",
        r"\bCHEST \(PA",
        r"\bPRE-OP AP",
        r"\bHEAD\b",
        r"\bNECK\b",
        r"\bSPINE\b",
        r"\bRENAL U\.S",
        r"\bULTRASOUND\b",
        r"\bDUPLEX\b",
    ]
    DIAGNOSIS_TIME_TEXT_PATTERNS = [
        r"\baortic dissection\b",
        r"\bdissection of (?:the )?aorta\b",
        r"\btype\s+[ab]\s+(?:aortic\s+)?dissection\b",
        r"\b(?:ascending|descending|thoracic)\s+aortic\s+dissection\b",
        r"\baorta[^\n.]{0,80}dissection\b",
        r"\bdissection[^\n.]{0,80}aorta\b",
    ]
    DIAGNOSIS_TIME_NEGATION_PATTERNS = [
        r"\bno (?:evidence of )?(?:acute )?aortic dissection\b",
        r"\bwithout (?:evidence of )?(?:acute )?aortic dissection\b",
        r"\bnegative for (?:acute )?aortic dissection\b",
        r"\brule out (?:acute )?aortic dissection\b",
        r"\bevaluate for (?:acute )?aortic dissection\b",
        r"\bconcern for (?:acute )?aortic dissection\b",
        r"\bquestion of (?:acute )?aortic dissection\b",
    ]
    USE_EVENT_CACHE = True
    FORCE_REBUILD_EVENT_CACHE = False
    EVENT_CACHE_CHUNKSIZE = 3_000_000
    EVENT_CACHE_QUERY_CHUNKSIZE = 1_000_000
    USE_MATRIX_CACHE = True
    FORCE_REBUILD_MATRIX_CACHE = False
    MATRIX_CACHE_VERSION = 9
    # Vitals to extract (ItemIDs mapped to column names)
    VITALS_DICT = {
        220045: "heart_rate", 220050: "abp_sys", 220051: "abp_dias",
        220179: "nibp_sys", 220180: "nibp_dias", 220210: "resp_rate", 
        220277: "spo2", 223761: "temp_f"
    }
    # Labs to extract (ItemIDs mapped to column names)
    LABS_DICT = {
        50915: "ddimer", 51003: "troponin_t", 51002: "troponin_i", 
        50889: "crp", 51214: "fibrinogen", 51237: "inr", 50912: "creatinine", 
        50983: "sodium", 50971: "potassium", 50931: "glucose", 51006: "bun", 
        50902: "chloride", 50882: "bicarbonate", 50868: "anion_gap", 
        50960: "magnesium", 50893: "calcium_total", 50970: "phosphate",
        51222: "hemoglobin", 51221: "hematocrit", 51265: "platelets", 
        51301: "wbc", 50820: "ph", 50813: "lactate", 50861: "alt", 50878: "ast"
    }
    USE_ECG_FEATURES = True
    ECG_MEASUREMENTS_PATH = DATA_DIR / "mimic-iv-ecg" / "1.0" / "machine_measurements.csv"
    ECG_CHUNKSIZE = 1_000_000
    ECG_MEASUREMENT_FEATURES = [
        "rr_interval",
        "p_onset",
        "p_end",
        "qrs_onset",
        "qrs_end",
        "t_end",
        "p_axis",
        "qrs_axis",
        "t_axis",
    ]
    ECG_DERIVED_FEATURES = ["qt_interval", "qtc_bazett"]
    # "full" emits min/max/mean, "median_only" emits medians, "first_only" emits first in-window values.
    FEATURE_AGGREGATION_MODE = "first_only"
    INCLUDE_FIRST_FEATURES = True
    USE_MEDICATION_FEATURES = False
    PRESCRIPTIONS_PATH = DATA_DIR / "hosp" / "prescriptions.csv"
    PRESCRIPTIONS_CHUNKSIZE = 1_000_000
    MEDICATION_GROUP_PATTERNS = {
        "beta_blocker": [
            r"\besmolol\b",
            r"\blabetalol\b",
            r"\bmetoprolol\b",
            r"\batenolol\b",
            r"\bcarvedilol\b",
            r"\bpropranolol\b",
        ],
        "vasodilator": [
            r"\bnicardipine\b",
            r"\bclevidipine\b",
            r"\bnitroprusside\b",
            r"\bnitroglycerin\b",
            r"\bhydralazine\b",
        ],
        "vasopressor": [
            r"\bnorepinephrine\b",
            r"\bepinephrine\b",
            r"\bphenylephrine\b",
            r"\bvasopressin\b",
            r"\bdopamine\b",
            r"\bdobutamine\b",
        ],
        "opioid_analgesic": [
            r"\bmorphine\b",
            r"\bhydromorphone\b",
            r"\bfentanyl\b",
            r"\boxycodone\b",
        ],
        "nonopioid_analgesic": [
            r"\bacetaminophen\b",
            r"\bketorolac\b",
        ],
        "anticoagulant": [
            r"\bheparin\b",
            r"\benoxaparin\b",
            r"\bwarfarin\b",
            r"\bapixaban\b",
            r"\brivaroxaban\b",
            r"\bbivalirudin\b",
            r"\bargatroban\b",
        ],
        "antiplatelet": [
            r"\baspirin\b",
            r"\bclopidogrel\b",
            r"\bticagrelor\b",
            r"\bprasugrel\b",
            r"\beptifibatide\b",
            r"\btirofiban\b",
        ],
        "statin": [
            r"\batorvastatin\b",
            r"\brosuvastatin\b",
            r"\bsimvastatin\b",
            r"\bpravastatin\b",
        ],
    }

    # add any other features you want to extract here
    OTHER_DICT = {
        ...
    }
 
    # ==========================================
    # 4. PRE-PROCESSING & LEAKAGE DECISIONS
    # ==========================================
    ROW_MISSINGNESS_THRESHOLD = 0.95    # Drop rows with nearly no observed feature data
    COL_MISSINGNESS_THRESHOLD = 0.80    # Drop features missing in > 80% of training patients
    # Explicitly drop these features before training to prevent physician test-ordering leakage
    # Notice 'inr' is missing from this list, meaning it will be kept!
    LEAKAGE_LABS_TO_DROP = [
        "ddimer_min", "ddimer_max", "ddimer_mean",
        "troponin_t_min", "troponin_t_max", "troponin_t_mean",
        "troponin_i_min", "troponin_i_max", "troponin_i_mean",
        "crp_min", "crp_max", "crp_mean",
        "fibrinogen_min", "fibrinogen_max", "fibrinogen_mean",
        "ddimer_median",
        "troponin_t_median",
        "troponin_i_median",
        "crp_median",
        "fibrinogen_median",
    ]
 
    # ==========================================
    # 5. MACHINE LEARNING DECISIONS
    # ==========================================
    TEST_SIZE = 0.20
    VALIDATION_SIZE = 0.20
    RANDOM_STATE = 42
    MODEL_REPORT_DIR = Path("../data/processed/model_reports")
    WRITE_MODEL_REPORTS = True
    USE_XGB_NATIVE_CATEGORICAL = True
    FEATURES_TO_DROP = []
    FEATURE_SET_PRESETS = {
        "medication_tol0p01": [
            "med_vasodilator_any",
            "med_beta_blocker_count",
            "med_vasopressor_any",
            "encounter_urgency",
            "med_anticoagulant_count",
            "lactate_median",
            "platelets_median",
            "inr_median",
            "med_opioid_analgesic_count",
            "bun_median",
            "hemoglobin_median",
            "med_statin_any",
            "med_antiplatelet_any",
            "ecg_count",
            "hematocrit_median",
            "race",
            "glucose_median",
            "wbc_median",
            "potassium_median",
            "alt_median",
            "calcium_total_median",
            "sodium_median",
            "bicarbonate_median",
            "magnesium_median",
            "phosphate_median",
            "creatinine_median",
            "ecg_t_end_median",
            "ecg_qrs_onset_median",
            "ecg_qrs_end_median",
            "ecg_qtc_bazett_median",
            "ecg_t_axis_median",
            "med_nonopioid_analgesic_count",
            "ecg_p_axis_median",
            "gender",
            "ecg_p_onset_median",
        ],
        "labs_only_pre_tmux_tol0p01": [
            "encounter_urgency",
            "lactate_median",
            "platelets_median",
            "index_age",
            "wbc_median",
            "ecg_qt_interval_median",
            "bun_median",
            "race",
            "magnesium_median",
            "anion_gap_median",
            "creatinine_median",
            "marital_status",
            "sodium_median",
            "ecg_qtc_bazett_median",
            "ecg_qrs_axis_median",
            "phosphate_median",
            "bicarbonate_median",
            "ecg_p_axis_median",
            "potassium_median",
            "ecg_qrs_onset_median",
        ],
    }
    ACTIVE_FEATURE_SET_PRESET = None
    FEATURES_TO_KEEP = []
    REQUIRE_ECG_MEASUREMENTS = False
    ECG_REQUIRED_COUNT_COL = "ecg_count"
    LOW_IMPORTANCE_REPORT_N = 25
    USE_RECURSIVE_FEATURE_ELIMINATION = True
    RFE_STRATEGY = "greedy_backward_ablation"
    RFE_MIN_FEATURES = 40
    RFE_PERFORMANCE_METRIC = "average_precision"
    RFE_PERFORMANCE_TOLERANCE = 0.01
    RFE_MAX_ITERATIONS = None
    RFE_PROTECTED_FEATURES = ["race"]
    RUN_SHADOW_FEATURE_FILTER_REPORT = False
    SHADOW_FILTER_REPEATS = 5
    SHADOW_FILTER_KEEP_HIT_RATE = 0.80
    SHADOW_FILTER_REJECT_HIT_RATE = 0.20
    RUN_SHADOW_REJECT_COMPARISON = False
    RUN_CROSS_VALIDATION = False
    CV_FOLDS = 5
    XGB_EVAL_METRIC = "aucpr"
    THRESHOLD_SELECTION_METRIC = "f1"
    THRESHOLD_RECALL_TARGETS = [0.80, 0.60, 0.40, 0.20]
    THRESHOLD_FP_BUDGETS = [10, 25, 50, 100]
    RUN_INR_ABLATION_COMPARISON = False
    INR_ABLATION_FEATURES = ["inr_min", "inr_mean", "inr_max"]
    XGB_PARAM_GRID = [
        {
            "n_estimators": 300,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 5,
            "reg_lambda": 5.0,
            "reg_alpha": 0.0,
        },
        {
            "n_estimators": 500,
            "max_depth": 2,
            "learning_rate": 0.03,
            "subsample": 0.90,
            "colsample_bytree": 0.90,
            "min_child_weight": 10,
            "reg_lambda": 10.0,
            "reg_alpha": 0.5,
        },
    ]
 
