# config.py
from pathlib import Path

class PipelineConfig:
    # ==========================================
    # 1. PATHS & DIRECTORIES
    # ==========================================
    DATA_DIR = Path("../data/mimic-iv")
    TARGET_MACRO_PATH = "../data/intermediate/aortic_dissection_macro_visits.csv"
    DIAGNOSES_ICD_PATH = DATA_DIR / "hosp" / "diagnoses_icd.csv"
    CACHE_DIR = Path("../data/processed/pipeline_cache")

    # ==========================================
    # 2. COHORT DECISIONS
    # ==========================================
    MIN_AGE = 18
    # Keep the evaluation split at natural prevalence. Training imbalance is
    # handled by XGBoost scale_pos_weight rather than random cohort downsampling.
    CONTROL_DOWNSAMPLE_RATIO = None
    USE_NATURAL_PREVALENCE_HOLDOUT = True
    USE_CLINICALLY_SIMILAR_CONTROLS = True
    CLINICALLY_SIMILAR_CONTROL_GROUPS = {
        "chest_pain": {
            9: ["7865"],
            10: ["R07"],
        },
        "back_or_abdominal_pain": {
            9: ["7245", "7890"],
            10: ["M54", "R10"],
        },
        "syncope": {
            9: ["7802"],
            10: ["R55"],
        },
        "acs_or_mi": {
            9: ["410", "411"],
            10: ["I20", "I21", "I22", "I24"],
        },
        "pulmonary_embolism": {
            9: ["4151"],
            10: ["I26"],
        },
        "hypertensive_crisis": {
            9: ["4010", "4372"],
            10: ["I16"],
        },
    }
    CONTROL_EXCLUSION_DIAGNOSIS_PREFIXES = {
        9: ["4410"],
        10: ["I710"],
    }
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
    DAY_0_WINDOW_HOURS = 6
    USE_EVENT_CACHE = True
    FORCE_REBUILD_EVENT_CACHE = False
    EVENT_CACHE_CHUNKSIZE = 3_000_000
    EVENT_CACHE_QUERY_CHUNKSIZE = 1_000_000
    USE_MATRIX_CACHE = True
    FORCE_REBUILD_MATRIX_CACHE = False
    MATRIX_CACHE_VERSION = 1
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

    # add any other features you want to extract here
    OTHER_DICT = {
        ...
    }
 
    # ==========================================
    # 4. PRE-PROCESSING & LEAKAGE DECISIONS
    # ==========================================
    ROW_MISSINGNESS_THRESHOLD = 1.00    # Keep all patients; XGBoost handles missing feature values
    COL_MISSINGNESS_THRESHOLD = 0.90    # Drop features missing in > 90% of patients
    # Explicitly drop these features before training to prevent physician test-ordering leakage
    # Notice 'inr' is missing from this list, meaning it will be kept!
    LEAKAGE_LABS_TO_DROP = [
        "ddimer_min", "ddimer_max", "ddimer_mean",
        "troponin_t_min", "troponin_t_max", "troponin_t_mean",
        "troponin_i_min", "troponin_i_max", "troponin_i_mean",
        "crp_min", "crp_max", "crp_mean",
        "fibrinogen_min", "fibrinogen_max", "fibrinogen_mean"
    ]
 
    # ==========================================
    # 5. MACHINE LEARNING DECISIONS
    # ==========================================
    TEST_SIZE = 0.20
    VALIDATION_SIZE = 0.20
    RANDOM_STATE = 42
    XGB_EVAL_METRIC = "aucpr"
    THRESHOLD_SELECTION_METRIC = "f1"
    THRESHOLD_RECALL_TARGETS = [0.80, 0.60, 0.40, 0.20]
    THRESHOLD_FP_BUDGETS = [10, 25, 50, 100]
    RUN_INR_ABLATION_COMPARISON = True
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
 
