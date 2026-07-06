import os
import gc
import pandas as pd
from cache import EventCache
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc, average_precision_score
 
class DataBuilder:
    def __init__(self, config):
        self.cfg = config
        self.event_cache = EventCache(config) if self.cfg.USE_EVENT_CACHE else None
 
    def build_spine_and_demographics(self):
        print("\n[1/4] Building Cohort Spine & Demographics...")
        pts = pd.read_csv(self.cfg.DATA_DIR / "hosp" / "patients.csv", usecols=["subject_id", "anchor_age", "anchor_year", "gender"])
        adm_cols = ["subject_id", "hadm_id", "admittime", "admission_type", "admission_location", "race", "insurance", "marital_status"]
        admissions = pd.read_csv(self.cfg.DATA_DIR / "hosp" / "admissions.csv", usecols=adm_cols)
        admissions["admittime"] = pd.to_datetime(admissions["admittime"])

        t_macro = pd.read_csv(self.cfg.TARGET_MACRO_PATH)
        t_macro["encounter_time"] = pd.to_datetime(t_macro["encounter_time"])
        t_macro["hadm_id"] = pd.to_numeric(t_macro["hadm_id"], errors="coerce").astype("Int64")
        t_macro["is_dissection_visit"] = t_macro["diagnosis"].fillna("").str.contains(r"441|I71", regex=True)
        target_visits = t_macro[t_macro["is_dissection_visit"]].sort_values(by=["subject_id", "encounter_time"]).groupby("subject_id").first().reset_index()
        target_visits.rename(columns={"encounter_time": "index_time", "hadm_id": "index_hadm_id"}, inplace=True)
        target_visits = target_visits[["subject_id", "index_hadm_id", "index_time"]]

        targets = target_visits.merge(
            admissions,
            left_on=["subject_id", "index_hadm_id"],
            right_on=["subject_id", "hadm_id"],
            how="left",
            validate="one_to_one",
        )
        missing_target_admissions = targets["hadm_id"].isna().sum()
        if missing_target_admissions:
            print(f"WARNING: {missing_target_admissions} target admissions did not match admissions.csv metadata.")
        targets.rename(columns={"admission_type": "encounter_urgency"}, inplace=True)
        targets["is_aortic_dissection"] = 1
        target_ids = set(targets["subject_id"].unique())

        controls = admissions[~admissions["subject_id"].isin(target_ids)].copy()
        controls = self._apply_clinically_similar_controls(controls)

        controls["index_time"] = controls["admittime"]
        controls = controls.sort_values(by=["subject_id", "index_time"], ascending=[True, False]).groupby("subject_id").first().reset_index()
        controls.rename(columns={"hadm_id": "index_hadm_id", "admission_type": "encounter_urgency"}, inplace=True)
        controls["is_aortic_dissection"] = 0
        spine = pd.concat([targets, controls], ignore_index=True)
        del admissions, target_visits, targets, controls # Free memory
        gc.collect()

        spine = spine.merge(pts, on="subject_id", how="inner")
        spine["index_age"] = spine["anchor_age"] + (spine["index_time"].dt.year - spine["anchor_year"])
        spine = spine[spine["index_age"] >= self.cfg.MIN_AGE].copy()
        spine = self._assign_cohort_split(spine)
        spine = self._apply_training_hard_controls(spine)
        spine = self._apply_training_control_downsample(spine)

        self._warn_label_specific_missingness(spine)

        keep_cols = [
            "subject_id",
            "index_hadm_id",
            "index_time",
            "cohort_split",
            "is_aortic_dissection",
            "index_age",
            "gender",
            "race",
            "insurance",
            "marital_status",
            "encounter_urgency",
            "admission_location",
        ]
        # Save to temp CSV and clear RAM
        temp_path = "temp_spine.csv"
        spine[keep_cols].to_csv(temp_path, index=False)
        del spine
        gc.collect()
        return temp_path
 
    def _apply_clinically_similar_controls(self, controls):
        if not getattr(self.cfg, "USE_CLINICALLY_SIMILAR_CONTROLS", False):
            return controls

        if controls.empty:
            raise ValueError("No non-target control admissions are available before clinical filtering.")

        print("    Selecting clinically similar controls from diagnoses_icd.csv...")
        control_hadm_ids = set(pd.to_numeric(controls["hadm_id"], errors="coerce").dropna().astype("int64"))
        diagnoses = pd.read_csv(
            self.cfg.DIAGNOSES_ICD_PATH,
            usecols=["hadm_id", "icd_code", "icd_version"],
        )
        diagnoses["hadm_id"] = pd.to_numeric(diagnoses["hadm_id"], errors="coerce")
        diagnoses.dropna(subset=["hadm_id", "icd_code", "icd_version"], inplace=True)
        diagnoses["hadm_id"] = diagnoses["hadm_id"].astype("int64")
        diagnoses = diagnoses[diagnoses["hadm_id"].isin(control_hadm_ids)].copy()
        diagnoses["icd_code"] = diagnoses["icd_code"].astype(str).str.upper().str.replace(".", "", regex=False)
        diagnoses["icd_version"] = pd.to_numeric(diagnoses["icd_version"], errors="coerce").astype("Int64")

        mimic_mask = self._diagnosis_group_mask(
            diagnoses,
            getattr(self.cfg, "CLINICALLY_SIMILAR_CONTROL_GROUPS", {}),
        )
        exclusion_mask = self._diagnosis_prefix_mask(
            diagnoses,
            getattr(self.cfg, "CONTROL_EXCLUSION_DIAGNOSIS_PREFIXES", {}),
        )

        mimic_hadm_ids = set(diagnoses.loc[mimic_mask, "hadm_id"].unique())
        excluded_hadm_ids = set(diagnoses.loc[exclusion_mask, "hadm_id"].unique())
        eligible_hadm_ids = mimic_hadm_ids - excluded_hadm_ids
        if not eligible_hadm_ids:
            raise ValueError("Clinically similar control filtering removed every control admission.")

        filtered = controls[controls["hadm_id"].isin(eligible_hadm_ids)].copy()
        print(
            "    Clinically similar controls: "
            f"{len(filtered):,} admissions across {filtered['subject_id'].nunique():,} subjects "
            f"({len(mimic_hadm_ids):,} admissions matched mimic diagnoses; "
            f"{len(excluded_hadm_ids):,} excluded for dissection-like diagnosis prefixes)."
        )
        return filtered

    def _diagnosis_group_mask(self, diagnoses, diagnosis_groups):
        mask = pd.Series(False, index=diagnoses.index)
        for group_name, prefixes_by_version in diagnosis_groups.items():
            group_mask = self._diagnosis_prefix_mask(diagnoses, prefixes_by_version)
            matched = int(group_mask.sum())
            admissions = diagnoses.loc[group_mask, "hadm_id"].nunique()
            print(
                f"      {group_name}: {admissions:,} admissions "
                f"({matched:,} diagnosis rows)"
            )
            mask = mask | group_mask
        return mask

    def _diagnosis_prefix_mask(self, diagnoses, prefixes_by_version):
        mask = pd.Series(False, index=diagnoses.index)
        for icd_version, prefixes in prefixes_by_version.items():
            version_mask = diagnoses["icd_version"] == int(icd_version)
            if not prefixes:
                continue
            normalized_prefixes = tuple(str(prefix).upper().replace(".", "") for prefix in prefixes)
            mask = mask | (version_mask & diagnoses["icd_code"].str.startswith(normalized_prefixes))
        return mask

    def _assign_cohort_split(self, spine):
        if not getattr(self.cfg, "USE_NATURAL_PREVALENCE_HOLDOUT", True):
            spine = spine.copy()
            spine["cohort_split"] = "train"
            return spine

        _, test_idx = train_test_split(
            spine.index,
            test_size=self.cfg.TEST_SIZE,
            random_state=self.cfg.RANDOM_STATE,
            stratify=spine["is_aortic_dissection"],
        )
        spine = spine.copy()
        spine["cohort_split"] = "train"
        spine.loc[test_idx, "cohort_split"] = "test"
        if getattr(self.cfg, "USE_CLINICALLY_SIMILAR_CONTROLS", False):
            split_label = "Clinically similar-control cohort split"
        else:
            split_label = "Full natural-prevalence split"
        self._print_split_counts(spine, split_label)
        return spine

    def _apply_training_hard_controls(self, spine):
        if not getattr(self.cfg, "USE_TRAINING_HARD_CONTROLS", False):
            return spine

        admission_type_match = spine["encounter_urgency"].isin(
            getattr(self.cfg, "HARD_CONTROL_ADMISSION_TYPES", set())
        )
        admission_location_match = spine["admission_location"].isin(
            getattr(self.cfg, "HARD_CONTROL_ADMISSION_LOCATIONS", set())
        )
        hard_control = admission_type_match | admission_location_match
        train_control = (
            (spine["cohort_split"] == "train")
            & (spine["is_aortic_dissection"] == 0)
        )
        drop_mask = train_control & ~hard_control
        dropped = int(drop_mask.sum())
        kept_train_controls = int((train_control & hard_control).sum())
        if kept_train_controls == 0:
            raise ValueError("Hard-control filtering removed every training control.")

        spine = spine[~drop_mask].copy()
        print(
            "    Training hard-control filter: "
            f"kept {kept_train_controls:,} controls, removed {dropped:,}; "
            "test controls are unchanged."
        )
        self._print_split_counts(spine, "After training hard-control filter")
        return spine

    def _apply_training_control_downsample(self, spine):
        ratio = getattr(self.cfg, "CONTROL_DOWNSAMPLE_RATIO", None)
        if not ratio:
            return spine

        train_targets = (
            (spine["cohort_split"] == "train")
            & (spine["is_aortic_dissection"] == 1)
        )
        train_controls = (
            (spine["cohort_split"] == "train")
            & (spine["is_aortic_dissection"] == 0)
        )
        sample_size = int(train_targets.sum() * ratio)
        sample_size = min(sample_size, int(train_controls.sum()))
        sampled_controls = spine[train_controls].sample(
            n=sample_size,
            random_state=self.cfg.RANDOM_STATE,
        ).index
        drop_mask = train_controls & ~spine.index.isin(sampled_controls)
        spine = spine[~drop_mask].copy()
        print(
            "    Training-only control downsample: "
            f"kept {sample_size:,} controls at 1:{ratio}; test controls are unchanged."
        )
        self._print_split_counts(spine, "After training-only downsample")
        return spine

    def _print_split_counts(self, spine, label):
        counts = (
            spine.groupby(["cohort_split", "is_aortic_dissection"])
            .size()
            .unstack(fill_value=0)
        )
        print(f"    {label}:")
        for split_name, row in counts.iterrows():
            controls = int(row.get(0, 0))
            targets = int(row.get(1, 0))
            total = controls + targets
            prevalence = targets / total if total else 0
            print(
                f"      {split_name}: {total:,} rows "
                f"({targets:,} targets, {controls:,} controls; "
                f"prevalence={prevalence:.4%})"
            )

    def _warn_label_specific_missingness(self, spine):
        check_cols = ["race", "insurance", "marital_status", "encounter_urgency"]
        check_cols = [col for col in check_cols if col in spine.columns]
        if spine["is_aortic_dissection"].nunique() < 2:
            return

        missing_by_label = spine.groupby("is_aortic_dissection")[check_cols].apply(lambda x: x.isna().mean())
        for col in check_cols:
            missing_gap = abs(missing_by_label.loc[1, col] - missing_by_label.loc[0, col])
            if missing_gap >= 0.50:
                print(
                    "WARNING: Large label-specific missingness gap in "
                    f"{col}: controls={missing_by_label.loc[0, col]:.3f}, "
                    f"targets={missing_by_label.loc[1, col]:.3f}"
                )

    def _iter_event_chunks(self, source_path, cache_name, itemids, subject_ids):
        if self.event_cache:
            table_name = self.event_cache.ensure_table(cache_name, source_path, itemids)
            yield from self.event_cache.iter_subject_events(table_name, subject_ids, itemids)
            return

        chunksize = getattr(self.cfg, "EVENT_CACHE_CHUNKSIZE", 3_000_000)
        usecols = ["subject_id", "charttime", "itemid", "valuenum"]
        subject_ids = set(subject_ids)
        for chunk in pd.read_csv(source_path, usecols=usecols, chunksize=chunksize):
            chunk = chunk[
                (chunk["subject_id"].isin(subject_ids))
                & (chunk["itemid"].isin(itemids))
            ].copy()
            if not chunk.empty:
                yield chunk

    def _aggregate_day0_events(self, event_chunks, spine, item_map, include_first=False):
        partial_stats = []
        partial_first = []
        spine_times = spine[["subject_id", "index_time"]]

        for chunk in event_chunks:
            chunk["charttime"] = pd.to_datetime(chunk["charttime"])
            chunk["valuenum"] = pd.to_numeric(chunk["valuenum"], errors="coerce")
            chunk.dropna(subset=["charttime", "valuenum"], inplace=True)
            if chunk.empty:
                continue

            chunk = chunk.merge(spine_times, on="subject_id", how="inner")
            delta = (chunk["charttime"] - chunk["index_time"]).dt.total_seconds() / 3600
            chunk = chunk[(delta >= 0) & (delta <= self.cfg.DAY_0_WINDOW_HOURS)].copy()
            if chunk.empty:
                continue

            chunk["feature_name"] = chunk["itemid"].map(item_map)
            keys = ["subject_id", "feature_name"]
            partial_stats.append(
                chunk.groupby(keys)["valuenum"].agg(["min", "max", "sum", "count"])
            )

            if include_first:
                first_idx = chunk.groupby(keys)["charttime"].idxmin()
                partial_first.append(
                    chunk.loc[first_idx, keys + ["charttime", "valuenum"]]
                )

            del chunk
            gc.collect()

        if not partial_stats:
            return pd.DataFrame({"subject_id": spine["subject_id"]})

        stats = pd.concat(partial_stats).groupby(level=[0, 1]).agg(
            {"min": "min", "max": "max", "sum": "sum", "count": "sum"}
        )
        stats["mean"] = stats["sum"] / stats["count"]

        output_stats = ["min", "max", "mean"]
        if include_first and partial_first:
            first_values = (
                pd.concat(partial_first)
                .sort_values("charttime")
                .groupby(["subject_id", "feature_name"])["valuenum"]
                .first()
            )
            stats["first"] = first_values
            output_stats.append("first")

        wide = stats[output_stats].unstack()
        wide.columns = [f"{feature}_{stat}" for stat, feature in wide.columns]
        wide.reset_index(inplace=True)

        del partial_stats, partial_first, stats
        gc.collect()
        return wide

    def add_vitals(self, spine_path):
        print("[2/4] Extracting Day-0 Vitals...")
        spine = pd.read_csv(spine_path)
        spine["index_time"] = pd.to_datetime(spine["index_time"])
        itemids = list(self.cfg.VITALS_DICT.keys())
        event_chunks = self._iter_event_chunks(
            self.cfg.DATA_DIR / "icu" / "chartevents.csv",
            "chartevents_vitals",
            itemids,
            spine["subject_id"],
        )
        agg_v = self._aggregate_day0_events(
            event_chunks,
            spine,
            self.cfg.VITALS_DICT,
            include_first=True,
        )
        spine = spine.merge(agg_v, on="subject_id", how="left")
        del agg_v
        gc.collect()

        # Overwrite temp file protocol
        temp_path = "temp_spine_vitals.csv"
        spine.to_csv(temp_path, index=False)
        del spine
        gc.collect()
        os.remove(spine_path) # Delete the old file
        return temp_path
 
    def add_labs(self, spine_path):
        print("[3/4] Extracting Day-0 Labs...")
        spine = pd.read_csv(spine_path)
        spine["index_time"] = pd.to_datetime(spine["index_time"])
        itemids = list(self.cfg.LABS_DICT.keys())
        event_chunks = self._iter_event_chunks(
            self.cfg.DATA_DIR / "hosp" / "labevents.csv",
            "labevents_labs",
            itemids,
            spine["subject_id"],
        )
        agg_l = self._aggregate_day0_events(
            event_chunks,
            spine,
            self.cfg.LABS_DICT,
            include_first=False,
        )
        spine = spine.merge(agg_l, on="subject_id", how="left")
        del agg_l
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
        id_cols = [
            "subject_id",
            "index_hadm_id",
            "index_time",
            "cohort_split",
            "is_aortic_dissection",
            "admission_location",
        ]
        cat_cols = ["gender", "race", "insurance", "marital_status", "encounter_urgency"]
        cat_cols = [c for c in cat_cols if c in df.columns]
        non_numeric_cols = id_cols + cat_cols
        features = [c for c in df.columns if c not in non_numeric_cols]
 
        # Row Drop
        row_miss = df[features].isnull().mean(axis=1)
        df = df[row_miss <= self.cfg.ROW_MISSINGNESS_THRESHOLD].copy()

        if "cohort_split" not in df.columns:
            _, test_idx = train_test_split(
                df.index,
                test_size=self.cfg.TEST_SIZE,
                random_state=self.cfg.RANDOM_STATE,
                stratify=df["is_aortic_dissection"],
            )
            df["cohort_split"] = "train"
            df.loc[test_idx, "cohort_split"] = "test"

        train_df = df[df["cohort_split"] == "train"].copy()
        test_df = df[df["cohort_split"] == "test"].copy()
        if train_df.empty or test_df.empty:
            raise ValueError("Expected non-empty train and test splits after preprocessing.")

        # Leakage Drop
        leakage_cols = [c for c in self.cfg.LEAKAGE_LABS_TO_DROP if c in features]
        if leakage_cols:
            train_df.drop(columns=leakage_cols, inplace=True)
            test_df.drop(columns=leakage_cols, inplace=True)
        features = [c for c in train_df.columns if c not in non_numeric_cols]

        # Column Drop. Fit this on train only, then apply to the holdout.
        col_miss = train_df[features].isnull().mean(axis=0)
        keep_feats = col_miss[col_miss <= self.cfg.COL_MISSINGNESS_THRESHOLD].index.tolist()
        model_cols = [
            "subject_id",
            "index_hadm_id",
            "index_time",
            "cohort_split",
            "is_aortic_dissection",
        ] + cat_cols + keep_feats
        model_cols = [c for c in model_cols if c in train_df.columns]
        train_df = train_df[model_cols].copy()
        test_df = test_df[model_cols].copy()

        train_encoded = pd.get_dummies(train_df, columns=cat_cols, drop_first=True)
        test_encoded = pd.get_dummies(test_df, columns=cat_cols, drop_first=True)

        y_train = train_encoded["is_aortic_dissection"].astype(int)
        y_test = test_encoded["is_aortic_dissection"].astype(int)
        drop_cols = ["subject_id", "index_hadm_id", "index_time", "cohort_split", "is_aortic_dissection"]
        X_train = train_encoded.drop(columns=[c for c in drop_cols if c in train_encoded.columns])
        X_test = test_encoded.drop(columns=[c for c in drop_cols if c in test_encoded.columns])
        X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)

        print(
            f"    Train matrix: {X_train.shape[0]:,} rows, {X_train.shape[1]:,} features; "
            f"positive prevalence={y_train.mean():.4%}"
        )
        print(
            f"    Holdout matrix: {X_test.shape[0]:,} rows, {X_test.shape[1]:,} features; "
            f"positive prevalence={y_test.mean():.4%}"
        )

        # Cleanup final temp file
        os.remove(matrix_path)
        return X_train, y_train, X_test, y_test
 
    def train_and_eval(self, X_train, y_train, X_test, y_test):
        import xgboost as xgb

        print("\n=== XGBoost Training & Evaluation ===")
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train,
            y_train,
            test_size=self.cfg.VALIDATION_SIZE,
            random_state=self.cfg.RANDOM_STATE,
            stratify=y_train,
        )
        param_grid = getattr(self.cfg, "XGB_PARAM_GRID", [{}])
        best_params = None
        best_val_prob = None
        best_ap = -1

        print(f"    Trying {len(param_grid)} XGBoost parameter set(s) on the training validation split...")
        for idx, params in enumerate(param_grid, start=1):
            scale_weight = (y_fit == 0).sum() / (y_fit == 1).sum()
            model = xgb.XGBClassifier(
                **params,
                scale_pos_weight=scale_weight,
                eval_metric=self.cfg.XGB_EVAL_METRIC,
                random_state=self.cfg.RANDOM_STATE,
                tree_method="hist",
                n_jobs=-1,
            )
            model.fit(X_fit, y_fit)
            val_prob = model.predict_proba(X_val)[:, 1]
            val_ap = average_precision_score(y_val, val_prob)
            print(f"      candidate {idx}: validation average precision={val_ap:.4f}; params={params}")
            if val_ap > best_ap:
                best_ap = val_ap
                best_params = params
                best_val_prob = val_prob

        threshold = self._select_threshold(y_val, best_val_prob)
        print(f"    Selected threshold on validation split: {threshold:.4f}")

        scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
        model = xgb.XGBClassifier(
            **best_params,
            scale_pos_weight=scale_weight,
            eval_metric=self.cfg.XGB_EVAL_METRIC,
            random_state=self.cfg.RANDOM_STATE,
            tree_method="hist",
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        print("\n1. Holdout Evaluation at Tuned Threshold:")
        self._print_threshold_metrics(y_test, y_prob, threshold)
        print("\n2. Holdout Evaluation at Default 0.5000 Threshold:")
        self._print_threshold_metrics(y_test, y_prob, 0.5)
        print("\n3. Holdout Threshold Operating Points:")
        self._print_threshold_operating_points(y_test, y_prob, threshold)

        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        print(f"\n4. Holdout Average Precision: {average_precision_score(y_test, y_prob):.4f}")
        print(f"   Holdout PR-AUC (trapezoidal): {auc(recall, precision):.4f}")
        print(f"   Best validation average precision: {best_ap:.4f}")
        print(f"   Best XGBoost params: {best_params}")
        imp = pd.DataFrame({'Feature': X_train.columns, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False)
        print("\n5. Top 25 Features:")
        print(imp.head(25).to_string(index=False))

        if getattr(self.cfg, "RUN_INR_ABLATION_COMPARISON", False):
            self._run_inr_ablation_comparison(X_train, y_train, X_test, y_test)

    def _run_inr_ablation_comparison(self, X_train, y_train, X_test, y_test):
        import xgboost as xgb

        drop_cols = [col for col in getattr(self.cfg, "INR_ABLATION_FEATURES", []) if col in X_train.columns]
        if not drop_cols:
            print("\n=== INR Ablation Comparison Skipped ===")
            print("    None of the configured INR ablation features were present in the model matrix.")
            return

        print("\n=== INR Ablation Comparison ===")
        print(f"    Dropping for comparison only: {drop_cols}")
        X_train_ablate = X_train.drop(columns=drop_cols)
        X_test_ablate = X_test.drop(columns=drop_cols)

        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train_ablate,
            y_train,
            test_size=self.cfg.VALIDATION_SIZE,
            random_state=self.cfg.RANDOM_STATE,
            stratify=y_train,
        )
        param_grid = getattr(self.cfg, "XGB_PARAM_GRID", [{}])
        best_params = None
        best_val_prob = None
        best_ap = -1

        print(f"    Trying {len(param_grid)} XGBoost parameter set(s) without INR features...")
        for idx, params in enumerate(param_grid, start=1):
            scale_weight = (y_fit == 0).sum() / (y_fit == 1).sum()
            model = xgb.XGBClassifier(
                **params,
                scale_pos_weight=scale_weight,
                eval_metric=self.cfg.XGB_EVAL_METRIC,
                random_state=self.cfg.RANDOM_STATE,
                tree_method="hist",
                n_jobs=-1,
            )
            model.fit(X_fit, y_fit)
            val_prob = model.predict_proba(X_val)[:, 1]
            val_ap = average_precision_score(y_val, val_prob)
            print(f"      candidate {idx}: validation average precision={val_ap:.4f}; params={params}")
            if val_ap > best_ap:
                best_ap = val_ap
                best_params = params
                best_val_prob = val_prob

        threshold = self._select_threshold(y_val, best_val_prob)
        print(f"    Selected threshold on validation split without INR: {threshold:.4f}")

        scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
        model = xgb.XGBClassifier(
            **best_params,
            scale_pos_weight=scale_weight,
            eval_metric=self.cfg.XGB_EVAL_METRIC,
            random_state=self.cfg.RANDOM_STATE,
            tree_method="hist",
            n_jobs=-1,
        )
        model.fit(X_train_ablate, y_train)
        y_prob = model.predict_proba(X_test_ablate)[:, 1]

        print("\nA. Holdout Evaluation Without INR at Tuned Threshold:")
        self._print_threshold_metrics(y_test, y_prob, threshold)
        print("\nB. Holdout Threshold Operating Points Without INR:")
        self._print_threshold_operating_points(y_test, y_prob, threshold)

        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        print(f"\nC. Holdout Average Precision Without INR: {average_precision_score(y_test, y_prob):.4f}")
        print(f"   Holdout PR-AUC Without INR (trapezoidal): {auc(recall, precision):.4f}")
        print(f"   Best validation average precision without INR: {best_ap:.4f}")
        print(f"   Best XGBoost params without INR: {best_params}")
        imp = pd.DataFrame({
            "Feature": X_train_ablate.columns,
            "Importance": model.feature_importances_,
        }).sort_values(by="Importance", ascending=False)
        print("\nD. Top 25 Features Without INR:")
        print(imp.head(25).to_string(index=False))

    def _select_threshold(self, y_true, y_prob):
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        if len(thresholds) == 0:
            return 0.5

        metric = getattr(self.cfg, "THRESHOLD_SELECTION_METRIC", "f1")
        if metric == "target_recall":
            target_recall = getattr(self.cfg, "TARGET_RECALL", 0.80)
            viable = recall[:-1] >= target_recall
            if viable.any():
                viable_idx = precision[:-1][viable].argmax()
                return float(thresholds[viable][viable_idx])

        f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
        return float(thresholds[f1.argmax()])

    def _print_threshold_metrics(self, y_true, y_prob, threshold):
        y_pred = (y_prob >= threshold).astype(int)
        print(f"   Threshold: {threshold:.4f}")
        print(classification_report(y_true, y_pred, zero_division=0))
        cm = confusion_matrix(y_true, y_pred)
        print("   Confusion Matrix (TN, FP | FN, TP):")
        print(cm)

    def _print_threshold_operating_points(self, y_true, y_prob, selected_threshold):
        threshold_table = self._threshold_table(y_true, y_prob)
        rows = [
            self._metrics_at_threshold("validation_f1", y_true, y_prob, selected_threshold),
            self._metrics_at_threshold("default_0.5", y_true, y_prob, 0.5),
        ]

        for target_recall in getattr(self.cfg, "THRESHOLD_RECALL_TARGETS", []):
            threshold = self._threshold_for_target_recall(threshold_table, target_recall)
            if threshold is not None:
                rows.append(
                    self._metrics_at_threshold(
                        f"recall>={target_recall:.2f}",
                        y_true,
                        y_prob,
                        threshold,
                    )
                )

        for fp_budget in getattr(self.cfg, "THRESHOLD_FP_BUDGETS", []):
            threshold = self._threshold_for_fp_budget(threshold_table, fp_budget)
            if threshold is not None:
                rows.append(
                    self._metrics_at_threshold(
                        f"fp<={fp_budget}",
                        y_true,
                        y_prob,
                        threshold,
                    )
                )

        summary = pd.DataFrame(rows)
        formatters = {
            "threshold": "{:.4f}".format,
            "precision": "{:.3f}".format,
            "recall": "{:.3f}".format,
            "specificity": "{:.3f}".format,
            "f1": "{:.3f}".format,
            "flagged_pct": "{:.3%}".format,
        }
        print(summary.to_string(index=False, formatters=formatters))

    def _threshold_for_target_recall(self, threshold_table, target_recall):
        viable = threshold_table[threshold_table["recall"] >= target_recall]
        if viable.empty:
            return None

        best_idx = viable.sort_values(
            by=["precision", "threshold"],
            ascending=[False, False],
        ).index[0]
        return float(viable.loc[best_idx, "threshold"])

    def _threshold_for_fp_budget(self, threshold_table, fp_budget):
        viable = threshold_table[threshold_table["fp"] <= fp_budget]
        if viable.empty:
            return None

        best_idx = viable.sort_values(
            by=["recall", "precision", "threshold"],
            ascending=[False, False, False],
        ).index[0]
        return float(viable.loc[best_idx, "threshold"])

    def _threshold_table(self, y_true, y_prob):
        df = pd.DataFrame({
            "y_true": pd.Series(y_true).astype(int).to_numpy(),
            "y_prob": pd.Series(y_prob).astype(float).to_numpy(),
        }).dropna()
        grouped = (
            df.groupby("y_prob")["y_true"]
            .agg(["sum", "count"])
            .sort_index(ascending=False)
            .rename(columns={"sum": "positives", "count": "flagged"})
        )

        total_pos = int(df["y_true"].sum())
        total = len(df)
        total_neg = total - total_pos
        grouped["tp"] = grouped["positives"].cumsum()
        grouped["fp"] = grouped["flagged"].cumsum() - grouped["tp"]
        grouped["fn"] = total_pos - grouped["tp"]
        grouped["tn"] = total_neg - grouped["fp"]
        grouped["threshold"] = grouped.index.astype(float)
        grouped["precision"] = grouped["tp"] / (grouped["tp"] + grouped["fp"])
        grouped["recall"] = grouped["tp"] / total_pos if total_pos else 0.0
        grouped["specificity"] = grouped["tn"] / total_neg if total_neg else 0.0
        grouped["f1"] = (
            2 * grouped["precision"] * grouped["recall"]
            / (grouped["precision"] + grouped["recall"])
        ).fillna(0.0)
        grouped["flagged_pct"] = grouped["flagged"] / total if total else 0.0
        return grouped.reset_index(drop=True)

    def _metrics_at_threshold(self, label, y_true, y_prob, threshold):
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return {
            "rule": label,
            "threshold": float(threshold),
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "flagged_pct": float(y_pred.mean()),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        }
 
