import os
import gc
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
 
class DataBuilder:
    def __init__(self, config):
        self.cfg = config
 
    def build_spine_and_demographics(self):
        print("\n[1/4] Building Cohort Spine & Demographics...")
        pts = pd.read_csv(self.cfg.DATA_DIR / "hosp" / "patients.csv", usecols=["subject_id", "anchor_age", "anchor_year", "gender"])
        t_macro = pd.read_csv(self.cfg.TARGET_MACRO_PATH)
        t_macro["encounter_time"] = pd.to_datetime(t_macro["encounter_time"])
        t_macro["is_dissection_visit"] = t_macro["diagnosis"].fillna("").str.contains(r"441|I71", regex=True)
        targets = t_macro[t_macro["is_dissection_visit"]].sort_values(by=["subject_id", "encounter_time"]).groupby("subject_id").first().reset_index()
        targets["is_aortic_dissection"] = 1
        targets.rename(columns={"encounter_time": "index_time", "hadm_id": "index_hadm_id"}, inplace=True)
        target_ids = set(targets["subject_id"].unique())

        adm_cols = ["subject_id", "hadm_id", "admittime", "admission_type", "admission_location", "race", "insurance", "marital_status"]
        c_adms = []
        for chunk in pd.read_csv(self.cfg.DATA_DIR / "hosp" / "admissions.csv", usecols=adm_cols, chunksize=1_000_000):
            hit = chunk[~chunk["subject_id"].isin(target_ids)].copy()
            if not hit.empty: c_adms.append(hit)
        controls = pd.concat(c_adms, ignore_index=True)
        del c_adms # Free memory
        gc.collect()

        controls["index_time"] = pd.to_datetime(controls["admittime"])
        controls = controls.sort_values(by=["subject_id", "index_time"], ascending=[True, False]).groupby("subject_id").first().reset_index()
        controls.rename(columns={"hadm_id": "index_hadm_id", "admission_type": "encounter_urgency"}, inplace=True)
        controls["is_aortic_dissection"] = 0
        spine = pd.concat([targets, controls], ignore_index=True)
        del targets, controls # Free memory
        gc.collect()

        spine = spine.merge(pts, on="subject_id", how="inner")
        spine["index_age"] = spine["anchor_age"] + (spine["index_time"].dt.year - spine["anchor_year"])
        spine = spine[spine["index_age"] >= self.cfg.MIN_AGE].copy()
        if self.cfg.CONTROL_DOWNSAMPLE_RATIO:
            target_count = spine["is_aortic_dissection"].sum()
            sample_size = target_count * self.cfg.CONTROL_DOWNSAMPLE_RATIO
            t_df = spine[spine["is_aortic_dissection"] == 1]
            c_df = spine[spine["is_aortic_dissection"] == 0].sample(n=sample_size, random_state=self.cfg.RANDOM_STATE)
            spine = pd.concat([t_df, c_df], ignore_index=True)

        keep_cols = ["subject_id", "index_hadm_id", "index_time", "is_aortic_dissection", "index_age", "gender", "race", "insurance", "marital_status", "encounter_urgency"]
        # Save to temp CSV and clear RAM
        temp_path = "temp_spine.csv"
        spine[keep_cols].to_csv(temp_path, index=False)
        del spine
        gc.collect()
        return temp_path
 
    def add_vitals(self, spine_path):
        print("[2/4] Extracting Day-0 Vitals (Chunking...)")
        spine = pd.read_csv(spine_path)
        spine["index_time"] = pd.to_datetime(spine["index_time"])
        v_list = []
        for chunk in pd.read_csv(self.cfg.DATA_DIR / "icu" / "chartevents.csv", usecols=["subject_id", "charttime", "itemid", "valuenum"], chunksize=3_000_000):
            chunk = chunk[(chunk["subject_id"].isin(spine["subject_id"])) & (chunk["itemid"].isin(self.cfg.VITALS_DICT.keys()))].copy()
            if not chunk.empty:
                chunk["charttime"] = pd.to_datetime(chunk["charttime"])
                chunk = chunk.merge(spine[["subject_id", "index_time"]], on="subject_id", how="inner")
                delta = (chunk["charttime"] - chunk["index_time"]).dt.total_seconds() / 3600
                chunk = chunk[(delta >= 0) & (delta <= self.cfg.DAY_0_WINDOW_HOURS)].copy()
                chunk["vital_name"] = chunk["itemid"].map(self.cfg.VITALS_DICT)
                v_list.append(chunk[["subject_id", "vital_name", "valuenum"]])
        if v_list:
            v_df = pd.concat(v_list, ignore_index=True)
            del v_list
            gc.collect()
            agg_v = v_df.groupby(["subject_id", "vital_name"])["valuenum"].agg(["min", "max", "mean", "first"]).unstack()
            agg_v.columns = [f"{n}_{s}" for s, n in agg_v.columns]
            spine = spine.merge(agg_v.reset_index(), on="subject_id", how="left")
            del v_df, agg_v
            gc.collect()

        # Overwrite temp file protocol
        temp_path = "temp_spine_vitals.csv"
        spine.to_csv(temp_path, index=False)
        del spine
        gc.collect()
        os.remove(spine_path) # Delete the old file
        return temp_path
 
    def add_labs(self, spine_path):
        print("[3/4] Extracting Day-0 Labs (Chunking...)")
        spine = pd.read_csv(spine_path)
        spine["index_time"] = pd.to_datetime(spine["index_time"])
        l_list = []
        for chunk in pd.read_csv(self.cfg.DATA_DIR / "hosp" / "labevents.csv", usecols=["subject_id", "charttime", "itemid", "valuenum"], chunksize=3_000_000):
            chunk = chunk[(chunk["subject_id"].isin(spine["subject_id"])) & (chunk["itemid"].isin(self.cfg.LABS_DICT.keys()))].copy()
            if not chunk.empty:
                chunk["charttime"] = pd.to_datetime(chunk["charttime"])
                chunk = chunk.merge(spine[["subject_id", "index_time"]], on="subject_id", how="inner")
                delta = (chunk["charttime"] - chunk["index_time"]).dt.total_seconds() / 3600
                chunk = chunk[(delta >= 0) & (delta <= self.cfg.DAY_0_WINDOW_HOURS)].copy()
                chunk["valuenum"] = pd.to_numeric(chunk["valuenum"], errors="coerce")
                chunk.dropna(subset=["valuenum"], inplace=True)
                chunk["lab_name"] = chunk["itemid"].map(self.cfg.LABS_DICT)
                l_list.append(chunk[["subject_id", "lab_name", "valuenum"]])
        if l_list:
            l_df = pd.concat(l_list, ignore_index=True)
            del l_list
            gc.collect()
            agg_l = l_df.groupby(["subject_id", "lab_name"])["valuenum"].agg(["min", "max", "mean"]).unstack()
            agg_l.columns = [f"{n}_{s}" for s, n in agg_l.columns]
            spine = spine.merge(agg_l.reset_index(), on="subject_id", how="left")
            del l_df, agg_l
            gc.collect()
 
        # Final temp file
        temp_path = "temp_raw_matrix.csv"
        spine.to_csv(temp_path, index=False)
        del spine
        gc.collect()
        os.remove(spine_path) # Delete the old file
        return temp_path

    # BUILD NEW FUNCTION FOLLOWING SAME STYLE AS ABOVE IF NECESSARY
    def add_medications(self, spine_path):
        ... # Placeholder for future implementation
    
    # USE IF WANTING TO ENGINEER ADDITIONAL FEATURES
    def engineer_features(self, matrix_path):
        print("[3.5/4] Engineering Custom Clinical Features...")
        df = pd.read_csv(matrix_path)
        # 1. SHOCK INDEX (Heart Rate / Systolic Blood Pressure)
        # # We use the 'first' vital signs to capture their state at triage
        if'heart_rate_first'in df.columns and'abp_sys_first'in df.columns:
            df['shock_index_first'] = df['heart_rate_first'] / df['abp_sys_first']
        # 2. PULSE PRESSURE (Systolic - Diastolic)
        if'abp_sys_mean'in df.columns and'abp_dias_mean'in df.columns:
            df['pulse_pressure_mean'] = df['abp_sys_mean'] - df['abp_dias_mean']
 
        # 3. BUN-TO-CREATININE RATIO (Renal function proxy)
        if'bun_mean'in df.columns and'creatinine_mean'in df.columns:
            df['bun_creat_ratio'] = df['bun_mean'] / df['creatinine_mean']

        # Overwrite the temporary file and flush RAM
        df.to_csv(matrix_path, index=False)
        del df
        gc.collect()
        return matrix_path
 
 
 
class ModelEngine:
    def __init__(self, config):
        self.cfg = config

    def preprocess(self, matrix_path):
        print("\n[4/4] Pre-processing & Cleaning Matrix...")
        df = pd.read_csv(matrix_path)
        meta_cols = ["subject_id", "index_hadm_id", "index_time", "is_aortic_dissection", "gender", "race", "insurance", "marital_status", "encounter_urgency"]
        features = [c for c in df.columns if c not in meta_cols]
 
        # Row Drop
        row_miss = df[features].isnull().mean(axis=1)
        df = df[row_miss <= self.cfg.ROW_MISSINGNESS_THRESHOLD].copy()

        # Column Drop
        col_miss = df[features].isnull().mean(axis=0)
        keep_feats = col_miss[col_miss <= self.cfg.COL_MISSINGNESS_THRESHOLD].index.tolist()
        df = df[meta_cols + keep_feats].copy()
 
        # Leakage Drop
        for leak_col in self.cfg.LEAKAGE_LABS_TO_DROP:
            if leak_col in df.columns:
                df.drop(columns=[leak_col], inplace=True)
 
        # Encode Categoricals
        cat_cols = ["gender", "race", "insurance", "marital_status", "encounter_urgency"]
        cat_cols = [c for c in cat_cols if c in df.columns]
        df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        y = df_encoded["is_aortic_dissection"].astype(int)
        X = df_encoded.drop(columns=["subject_id", "index_hadm_id", "index_time", "is_aortic_dissection"])
        X = X.astype(float)
        # Cleanup final temp file
        os.remove(matrix_path)
        return X, y
 
    def train_and_eval(self, X, y):
        print("\n=== XGBoost Training & Evaluation ===")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.cfg.TEST_SIZE, random_state=self.cfg.RANDOM_STATE, stratify=y
        )
        scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
        model = xgb.XGBClassifier(scale_pos_weight=scale_weight, use_label_encoder=False, eval_metric=self.cfg.XGB_EVAL_METRIC, random_state=self.cfg.RANDOM_STATE)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        print("\n1. Classification Report:")
        print(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        print("2. Confusion Matrix (TN, FP | FN, TP):")
        print(cm)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        print(f"\n3. PR-AUC: {auc(recall, precision):.4f}")
        imp = pd.DataFrame({'Feature': X_train.columns, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False)
        print("\n4. Top 10 Features:")
        print(imp.head(10).to_string(index=False))
 