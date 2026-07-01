# config.py
from pathlib import Path

class PipelineConfig:
    # ==========================================
    # 1. PATHS & DIRECTORIES
    # ==========================================
    DATA_DIR = Path("../data/mimic-iv")
    TARGET_MACRO_PATH = "../data/intermediate/aortic_dissection_macro_visits.csv"
    CACHE_DIR = Path("../data/processed/pipeline_cache")

    # ==========================================
    # 2. COHORT DECISIONS
    # ==========================================
    MIN_AGE = 18
    # Set to an integer (e.g., 5) to downsample controls 1:5. Set to None to use all controls.
    CONTROL_DOWNSAMPLE_RATIO = None 

    # ==========================================
    # 3. FEATURE ENGINEERING DECISIONS
    # ==========================================
    DAY_0_WINDOW_HOURS = 24
    USE_EVENT_CACHE = True
    FORCE_REBUILD_EVENT_CACHE = False
    EVENT_CACHE_CHUNKSIZE = 3_000_000
    EVENT_CACHE_QUERY_CHUNKSIZE = 1_000_000
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
    ROW_MISSINGNESS_THRESHOLD = 0.50    # Drop patients missing > 50% of features
    COL_MISSINGNESS_THRESHOLD = 0.50    # Drop features missing in > 50% of patients
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
    RANDOM_STATE = 42
    XGB_EVAL_METRIC = "logloss"
 
