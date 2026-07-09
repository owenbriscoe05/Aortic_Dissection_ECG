import os
import gc
import re
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from cache import EventCache
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc, average_precision_score, roc_auc_score, roc_curve
 
class DataBuilder:
    def __init__(self, config):
        self.cfg = config
        self.event_cache = EventCache(config) if self.cfg.USE_EVENT_CACHE else None
 
    def build_spine_and_demographics(self):
        print("\n[1/4] Building Cohort Spine & Demographics...")
        pts = pd.read_csv(self.cfg.DATA_DIR / "hosp" / "patients.csv", usecols=["subject_id", "anchor_age", "anchor_year", "gender"])
        adm_cols = [
            "subject_id",
            "hadm_id",
            "admittime",
            "edregtime",
            "admission_type",
            "admission_location",
            "race",
            "insurance",
            "marital_status",
        ]
        admissions = pd.read_csv(self.cfg.DATA_DIR / "hosp" / "admissions.csv", usecols=adm_cols)
        admissions["admittime"] = pd.to_datetime(admissions["admittime"])
        admissions["edregtime"] = pd.to_datetime(admissions["edregtime"])
        admissions = self._assign_index_time(admissions)
        target_hadm_ids = self._target_diagnosis_hadm_ids()

        t_macro = pd.read_csv(self.cfg.TARGET_MACRO_PATH)
        t_macro["hadm_id"] = pd.to_numeric(t_macro["hadm_id"], errors="coerce").astype("Int64")
        t_macro["is_dissection_visit"] = t_macro["hadm_id"].isin(target_hadm_ids)
        target_visits = t_macro[t_macro["is_dissection_visit"]][["subject_id", "hadm_id"]].drop_duplicates()

        targets = target_visits.merge(
            admissions,
            left_on=["subject_id", "hadm_id"],
            right_on=["subject_id", "hadm_id"],
            how="left",
            validate="one_to_one",
        )
        missing_target_admissions = targets["hadm_id"].isna().sum()
        if missing_target_admissions:
            print(f"WARNING: {missing_target_admissions} target admissions did not match admissions.csv metadata.")
        targets = (
            targets.sort_values(by=["subject_id", "index_time"])
            .groupby("subject_id")
            .first()
            .reset_index()
        )
        targets.rename(columns={"hadm_id": "index_hadm_id"}, inplace=True)
        targets.rename(columns={"admission_type": "encounter_urgency"}, inplace=True)
        targets["is_aortic_dissection"] = 1
        target_ids = set(targets["subject_id"].unique())

        controls = admissions[~admissions["subject_id"].isin(target_ids)].copy()
        controls = self._apply_clinically_similar_controls(controls)

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
        spine = self._add_feature_window_bounds(spine)

        self._warn_label_specific_missingness(spine)

        keep_cols = [
            "subject_id",
            "index_hadm_id",
            "index_time",
            "index_time_source",
            "cohort_split",
            "is_aortic_dissection",
            "feature_window_end",
            "diagnosis_proxy_time",
            "diagnosis_time_source",
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

    def _assign_index_time(self, admissions):
        admissions = admissions.copy()
        use_ed_time = getattr(self.cfg, "USE_EDREGTIME_AS_INDEX_TIME", False)
        has_ed_time = (
            admissions["edregtime"].notna()
            & (admissions["edregtime"] <= admissions["admittime"])
            if use_ed_time
            else pd.Series(False, index=admissions.index)
        )
        ed_after_admit = (
            admissions["edregtime"].notna()
            & (admissions["edregtime"] > admissions["admittime"])
            if use_ed_time
            else pd.Series(False, index=admissions.index)
        )
        admissions["index_time"] = admissions["admittime"]
        admissions.loc[has_ed_time, "index_time"] = admissions.loc[has_ed_time, "edregtime"]
        admissions["index_time_source"] = "admittime"
        admissions.loc[has_ed_time, "index_time_source"] = "edregtime"
        admissions.loc[ed_after_admit, "index_time_source"] = "admittime_edregtime_after_admit"
        return admissions

    def _target_diagnosis_hadm_ids(self):
        diagnoses = pd.read_csv(
            self.cfg.DIAGNOSES_ICD_PATH,
            usecols=["hadm_id", "icd_code", "icd_version"],
        )
        diagnoses = self._prepare_diagnosis_codes(diagnoses)
        target_mask = self._exact_diagnosis_code_mask(
            diagnoses,
            getattr(self.cfg, "TARGET_DIAGNOSIS_CODES_BY_VERSION", {}),
        )
        target_hadm_ids = set(diagnoses.loc[target_mask, "hadm_id"].unique())
        if not target_hadm_ids:
            raise ValueError("No target admissions matched the configured exact aortic dissection ICD codes.")

        print(
            "    Exact target diagnosis admissions: "
            f"{len(target_hadm_ids):,} admissions matched configured ICD codes."
        )
        return target_hadm_ids

    def _prepare_diagnosis_codes(self, diagnoses):
        diagnoses = diagnoses.copy()
        diagnoses["hadm_id"] = pd.to_numeric(diagnoses["hadm_id"], errors="coerce")
        diagnoses.dropna(subset=["hadm_id", "icd_code", "icd_version"], inplace=True)
        diagnoses["hadm_id"] = diagnoses["hadm_id"].astype("int64")
        diagnoses["icd_code_norm"] = (
            diagnoses["icd_code"]
            .astype(str)
            .str.upper()
            .str.replace(".", "", regex=False)
        )
        diagnoses["icd_version"] = pd.to_numeric(diagnoses["icd_version"], errors="coerce").astype("Int64")
        return diagnoses

    def _exact_diagnosis_code_mask(self, diagnoses, codes_by_version):
        mask = pd.Series(False, index=diagnoses.index)
        for icd_version, codes in codes_by_version.items():
            if not codes:
                continue
            normalized_codes = {
                str(code).upper().replace(".", "")
                for code in codes
            }
            mask = mask | (
                (diagnoses["icd_version"] == int(icd_version))
                & diagnoses["icd_code_norm"].isin(normalized_codes)
            )
        return mask
 
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
        diagnoses = self._prepare_diagnosis_codes(diagnoses)
        diagnoses = diagnoses[diagnoses["hadm_id"].isin(control_hadm_ids)].copy()

        mimic_mask = self._diagnosis_group_mask(
            diagnoses,
            getattr(self.cfg, "CLINICALLY_SIMILAR_CONTROL_GROUPS", {}),
        )
        exclusion_mask = self._exact_diagnosis_code_mask(
            diagnoses,
            getattr(self.cfg, "CONTROL_EXCLUSION_DIAGNOSIS_CODES_BY_VERSION", {}),
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
            f"{len(excluded_hadm_ids):,} excluded for exact dissection diagnosis codes)."
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
            mask = mask | (version_mask & diagnoses["icd_code_norm"].str.startswith(normalized_prefixes))
        return mask

    def _add_feature_window_bounds(self, spine):
        spine = spine.copy()
        default_window_end = spine["index_time"] + pd.to_timedelta(self.cfg.DAY_0_WINDOW_HOURS, unit="h")
        spine["feature_window_end"] = default_window_end
        spine["diagnosis_proxy_time"] = pd.NaT
        spine["diagnosis_time_source"] = pd.NA

        if not getattr(self.cfg, "USE_DIAGNOSIS_TIME_CENSORING", False):
            return spine

        proxy_times = self._radiology_diagnosis_proxy_times(spine)
        if proxy_times.empty:
            print("    Diagnosis-time censoring: no usable radiology charttime proxies found.")
            return spine

        spine = spine.merge(proxy_times, on=["subject_id", "index_hadm_id"], how="left")
        has_proxy = spine["radiology_diagnosis_proxy_time"].notna()
        positive_proxy = has_proxy & (spine["is_aortic_dissection"] == 1)
        if positive_proxy.any():
            proxy_time = spine.loc[positive_proxy, "radiology_diagnosis_proxy_time"]
            index_time = spine.loc[positive_proxy, "index_time"]
            bounded_proxy_time = proxy_time.where(proxy_time >= index_time, index_time)
            spine.loc[positive_proxy, "diagnosis_proxy_time"] = bounded_proxy_time
            spine.loc[positive_proxy, "diagnosis_time_source"] = self.cfg.DIAGNOSIS_TIME_SOURCE
            default_end_for_proxy = (
                spine.loc[positive_proxy, "index_time"]
                + pd.to_timedelta(self.cfg.DAY_0_WINDOW_HOURS, unit="h")
            )
            spine.loc[positive_proxy, "feature_window_end"] = pd.concat(
                [
                    default_end_for_proxy,
                    bounded_proxy_time,
                ],
                axis=1,
            ).min(axis=1)

        spine.drop(columns=["radiology_diagnosis_proxy_time"], inplace=True)
        censored_count = int(positive_proxy.sum())
        target_count = int((spine["is_aortic_dissection"] == 1).sum())
        before_index_count = int(
            (
                positive_proxy
                & (spine["diagnosis_proxy_time"] == spine["index_time"])
            ).sum()
        )
        print(
            "    Diagnosis-time censoring: "
            f"{censored_count:,}/{target_count:,} target rows received a radiology charttime proxy "
            f"({before_index_count:,} at or before index time; storetime was not used)."
        )
        return spine

    def _radiology_diagnosis_proxy_times(self, spine):
        notes_path = getattr(self.cfg, "RADIOLOGY_NOTES_PATH", None)
        if not notes_path or not os.path.exists(notes_path):
            print(f"WARNING: Radiology notes file not found for diagnosis-time censoring: {notes_path}")
            return pd.DataFrame(columns=["subject_id", "index_hadm_id", "radiology_diagnosis_proxy_time"])

        target_hadm_ids = set(
            pd.to_numeric(
                spine.loc[spine["is_aortic_dissection"] == 1, "index_hadm_id"],
                errors="coerce",
            )
            .dropna()
            .astype("int64")
        )
        if not target_hadm_ids:
            return pd.DataFrame(columns=["subject_id", "index_hadm_id", "radiology_diagnosis_proxy_time"])

        positive_pattern = re.compile(
            "|".join(getattr(self.cfg, "DIAGNOSIS_TIME_TEXT_PATTERNS", [])),
            re.IGNORECASE,
        )
        negation_patterns = getattr(self.cfg, "DIAGNOSIS_TIME_NEGATION_PATTERNS", [])
        negation_pattern = (
            re.compile("|".join(negation_patterns), re.IGNORECASE)
            if negation_patterns
            else None
        )

        proxy_chunks = []
        usecols = ["note_id", "subject_id", "hadm_id", "charttime", "text"]
        for chunk in pd.read_csv(notes_path, usecols=usecols, chunksize=100_000):
            chunk = chunk[chunk["hadm_id"].isin(target_hadm_ids)].copy()
            if chunk.empty:
                continue

            chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
            chunk.dropna(subset=["subject_id", "hadm_id", "charttime"], inplace=True)
            if chunk.empty:
                continue

            text = self._radiology_proxy_text(chunk["text"])
            positive = text.str.contains(positive_pattern, regex=True)
            if negation_pattern is not None:
                positive = positive & ~text.str.contains(negation_pattern, regex=True)

            if positive.any():
                proxy_chunks.append(
                    chunk.loc[positive, ["note_id", "subject_id", "hadm_id", "charttime"]]
                )

            del chunk
            gc.collect()

        if not proxy_chunks:
            return pd.DataFrame(columns=["subject_id", "index_hadm_id", "radiology_diagnosis_proxy_time"])

        proxies = pd.concat(proxy_chunks, ignore_index=True)
        if getattr(self.cfg, "DIAGNOSIS_TIME_REQUIRE_DIAGNOSTIC_EXAM", False):
            proxies = self._filter_diagnostic_radiology_exams(proxies)
            if proxies.empty:
                return pd.DataFrame(columns=["subject_id", "index_hadm_id", "radiology_diagnosis_proxy_time"])

        proxies["hadm_id"] = pd.to_numeric(proxies["hadm_id"], errors="coerce").astype("int64")
        proxies = (
            proxies.sort_values(["subject_id", "hadm_id", "charttime"])
            .groupby(["subject_id", "hadm_id"], as_index=False)["charttime"]
            .first()
            .rename(
                columns={
                    "hadm_id": "index_hadm_id",
                    "charttime": "radiology_diagnosis_proxy_time",
                }
            )
        )
        return proxies

    def _radiology_proxy_text(self, text_series):
        text = text_series.fillna("")
        if not getattr(self.cfg, "DIAGNOSIS_TIME_REQUIRE_RESULT_SECTION", False):
            return text

        header_pattern = "|".join(
            re.escape(header)
            for header in getattr(self.cfg, "DIAGNOSIS_TIME_RESULT_SECTION_HEADERS", [])
        )
        if not header_pattern:
            return text

        pattern = re.compile(rf"(?im)^\s*(?:{header_pattern})\s*:\s*")

        def extract_result_sections(note_text):
            matches = list(pattern.finditer(str(note_text)))
            if not matches:
                return ""
            return "\n".join(str(note_text)[match.start():] for match in matches)

        return text.map(extract_result_sections)

    def _filter_diagnostic_radiology_exams(self, proxies):
        detail_path = getattr(self.cfg, "RADIOLOGY_DETAIL_PATH", None)
        if not detail_path or not os.path.exists(detail_path):
            print(f"WARNING: Radiology detail file not found for exam filtering: {detail_path}")
            return proxies.iloc[0:0].copy()

        note_ids = set(proxies["note_id"].dropna())
        if not note_ids:
            return proxies.iloc[0:0].copy()

        exam_chunks = []
        usecols = ["note_id", "field_name", "field_value"]
        for chunk in pd.read_csv(detail_path, usecols=usecols, chunksize=500_000):
            chunk = chunk[
                chunk["note_id"].isin(note_ids)
                & chunk["field_name"].eq("exam_name")
            ]
            if not chunk.empty:
                exam_chunks.append(chunk[["note_id", "field_value"]])

        if not exam_chunks:
            return proxies.iloc[0:0].copy()

        exams = (
            pd.concat(exam_chunks, ignore_index=True)
            .groupby("note_id")["field_value"]
            .apply(lambda values: " | ".join(values.astype(str)))
            .reset_index(name="exam_name")
        )
        proxies = proxies.merge(exams, on="note_id", how="left")
        keep = proxies["exam_name"].fillna("").map(self._is_diagnostic_aortic_radiology_exam)
        return proxies[keep].copy()

    def _is_diagnostic_aortic_radiology_exam(self, exam_name):
        exam_name = str(exam_name).upper()
        include_patterns = getattr(self.cfg, "DIAGNOSIS_TIME_EXAM_INCLUDE_PATTERNS", [])
        anatomy_patterns = getattr(self.cfg, "DIAGNOSIS_TIME_EXAM_ANATOMY_PATTERNS", [])
        exclude_patterns = getattr(self.cfg, "DIAGNOSIS_TIME_EXAM_EXCLUDE_PATTERNS", [])

        has_modality = any(re.search(pattern, exam_name, re.IGNORECASE) for pattern in include_patterns)
        has_anatomy = any(re.search(pattern, exam_name, re.IGNORECASE) for pattern in anatomy_patterns)
        is_excluded = any(re.search(pattern, exam_name, re.IGNORECASE) for pattern in exclude_patterns)
        return has_modality and has_anatomy and not is_excluded

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
        aggregation_mode = getattr(self.cfg, "FEATURE_AGGREGATION_MODE", "full")
        if aggregation_mode == "first_only":
            include_first = True
        partial_stats = []
        partial_values = []
        partial_first = []
        spine_times = spine[["subject_id", "index_time"]].copy()
        if "feature_window_end" in spine.columns:
            spine_times["feature_window_end"] = pd.to_datetime(
                spine["feature_window_end"],
                errors="coerce",
            )
        else:
            spine_times["feature_window_end"] = (
                spine_times["index_time"]
                + pd.to_timedelta(self.cfg.DAY_0_WINDOW_HOURS, unit="h")
            )
        spine_times["feature_window_end"] = spine_times["feature_window_end"].fillna(
            spine_times["index_time"] + pd.to_timedelta(self.cfg.DAY_0_WINDOW_HOURS, unit="h")
        )

        for chunk in event_chunks:
            chunk["charttime"] = pd.to_datetime(chunk["charttime"])
            chunk["valuenum"] = pd.to_numeric(chunk["valuenum"], errors="coerce")
            chunk.dropna(subset=["charttime", "valuenum"], inplace=True)
            if chunk.empty:
                continue

            chunk = chunk.merge(spine_times, on="subject_id", how="inner")
            chunk = chunk[
                (chunk["charttime"] >= chunk["index_time"])
                & (chunk["charttime"] <= chunk["feature_window_end"])
            ].copy()
            if chunk.empty:
                continue

            chunk["feature_name"] = chunk["itemid"].map(item_map)
            keys = ["subject_id", "feature_name"]
            if aggregation_mode == "median_only":
                partial_values.append(chunk[keys + ["valuenum"]])
            elif aggregation_mode == "first_only":
                pass
            else:
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

        if aggregation_mode == "first_only":
            if not partial_first:
                return pd.DataFrame({"subject_id": spine["subject_id"]})

            first_values = (
                pd.concat(partial_first)
                .sort_values("charttime")
                .groupby(["subject_id", "feature_name"])["valuenum"]
                .first()
            )
            stats = pd.DataFrame({"first": first_values})
            output_stats = ["first"]
        elif aggregation_mode == "median_only":
            if not partial_values:
                return pd.DataFrame({"subject_id": spine["subject_id"]})

            stats = pd.DataFrame({
                "median": (
                    pd.concat(partial_values, ignore_index=True)
                    .groupby(["subject_id", "feature_name"])["valuenum"]
                    .median()
                )
            })
            output_stats = ["median"]
        else:
            if not partial_stats:
                return pd.DataFrame({"subject_id": spine["subject_id"]})

            stats = pd.concat(partial_stats).groupby(level=[0, 1]).agg(
                {"min": "min", "max": "max", "sum": "sum", "count": "sum"}
            )
            stats["mean"] = stats["sum"] / stats["count"]
            output_stats = ["min", "max", "mean"]

        if not partial_stats and not partial_values and not partial_first:
            return pd.DataFrame({"subject_id": spine["subject_id"]})
        if aggregation_mode != "first_only" and include_first and partial_first:
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

        del partial_stats, partial_values, partial_first, stats
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
            include_first=getattr(self.cfg, "INCLUDE_FIRST_FEATURES", True),
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
            include_first=getattr(self.cfg, "INCLUDE_FIRST_FEATURES", True),
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

    def add_ecg_features(self, matrix_path):
        if not getattr(self.cfg, "USE_ECG_FEATURES", False):
            return matrix_path

        print("[3.5/4] Extracting Windowed ECG Machine Measurements...")
        df = pd.read_csv(matrix_path, low_memory=False)
        df["index_time"] = pd.to_datetime(df["index_time"])
        df["feature_window_end"] = pd.to_datetime(df["feature_window_end"], errors="coerce")
        default_window_end = df["index_time"] + pd.to_timedelta(self.cfg.DAY_0_WINDOW_HOURS, unit="h")
        df["feature_window_end"] = df["feature_window_end"].fillna(default_window_end)

        ecg_features = list(getattr(self.cfg, "ECG_MEASUREMENT_FEATURES", []))
        derived_features = list(getattr(self.cfg, "ECG_DERIVED_FEATURES", []))
        all_features = ecg_features + derived_features
        usecols = ["subject_id", "ecg_time"] + ecg_features
        spine_times = df[["subject_id", "index_time", "feature_window_end"]]
        subject_ids = set(df["subject_id"])

        aggregation_mode = getattr(self.cfg, "FEATURE_AGGREGATION_MODE", "full")
        partial_stats = []
        partial_values = []
        partial_first = []
        partial_counts = []
        for chunk in pd.read_csv(
            self.cfg.ECG_MEASUREMENTS_PATH,
            usecols=usecols,
            chunksize=getattr(self.cfg, "ECG_CHUNKSIZE", 1_000_000),
        ):
            chunk = chunk[chunk["subject_id"].isin(subject_ids)].copy()
            if chunk.empty:
                continue

            chunk["ecg_time"] = pd.to_datetime(chunk["ecg_time"], errors="coerce")
            chunk.dropna(subset=["subject_id", "ecg_time"], inplace=True)
            if chunk.empty:
                continue

            chunk = chunk.merge(spine_times, on="subject_id", how="inner")
            chunk = chunk[
                (chunk["ecg_time"] >= chunk["index_time"])
                & (chunk["ecg_time"] <= chunk["feature_window_end"])
            ].copy()
            if chunk.empty:
                continue

            for col in ecg_features:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
            if "qt_interval" in derived_features and {"t_end", "qrs_onset"}.issubset(chunk.columns):
                chunk["qt_interval"] = chunk["t_end"] - chunk["qrs_onset"]
            if "qtc_bazett" in derived_features and {"qt_interval", "rr_interval"}.issubset(chunk.columns):
                rr_seconds = chunk["rr_interval"] / 1000
                chunk["qtc_bazett"] = chunk["qt_interval"] / np.sqrt(rr_seconds.where(rr_seconds > 0))

            available_features = [col for col in all_features if col in chunk.columns]
            chunk[available_features] = chunk[available_features].apply(pd.to_numeric, errors="coerce")
            partial_counts.append(chunk.groupby("subject_id").size())
            if aggregation_mode == "median_only":
                partial_values.append(chunk[["subject_id"] + available_features])
            elif aggregation_mode == "first_only":
                pass
            else:
                partial_stats.append(
                    chunk.groupby("subject_id")[available_features].agg(["min", "max", "sum", "count"])
                )

            if getattr(self.cfg, "INCLUDE_FIRST_FEATURES", True):
                first_idx = chunk.groupby("subject_id")["ecg_time"].idxmin()
                partial_first.append(chunk.loc[first_idx, ["subject_id", "ecg_time"] + available_features])

            del chunk
            gc.collect()

        if aggregation_mode == "first_only" and partial_first:
            first_values = (
                pd.concat(partial_first)
                .sort_values("ecg_time")
                .groupby("subject_id")[all_features]
                .first()
            )
            ecg_wide = pd.DataFrame(index=first_values.index)
            for feature in all_features:
                ecg_wide[f"ecg_{feature}_first"] = first_values[feature]
        elif aggregation_mode == "median_only" and partial_values:
            values = pd.concat(partial_values, ignore_index=True)
            medians = values.groupby("subject_id").median(numeric_only=True)
            ecg_wide = pd.DataFrame(index=medians.index)
            for feature in all_features:
                if feature in medians.columns:
                    ecg_wide[f"ecg_{feature}_median"] = medians[feature]
        elif partial_stats:
            stats = pd.concat(partial_stats)
            ecg_wide = pd.DataFrame(index=stats.index.unique())
            for feature in all_features:
                ecg_wide[f"ecg_{feature}_min"] = stats[(feature, "min")].groupby(level=0).min()
                ecg_wide[f"ecg_{feature}_max"] = stats[(feature, "max")].groupby(level=0).max()
                feature_sum = stats[(feature, "sum")].groupby(level=0).sum()
                feature_count = stats[(feature, "count")].groupby(level=0).sum()
                ecg_wide[f"ecg_{feature}_mean"] = feature_sum / feature_count

        if (partial_stats or partial_values or partial_first):
            if (
                aggregation_mode != "first_only"
                and getattr(self.cfg, "INCLUDE_FIRST_FEATURES", True)
                and partial_first
            ):
                first_values = (
                    pd.concat(partial_first)
                    .sort_values("ecg_time")
                    .groupby("subject_id")[all_features]
                    .first()
                )
                for feature in all_features:
                    ecg_wide[f"ecg_{feature}_first"] = first_values[feature]

            ecg_counts = pd.concat(partial_counts).groupby(level=0).sum()
            ecg_wide["ecg_count"] = ecg_counts
            ecg_wide.reset_index(names="subject_id", inplace=True)
            df = df.merge(ecg_wide, on="subject_id", how="left")
            print(
                "    ECG features: "
                f"{int(ecg_wide['subject_id'].nunique()):,} patients had at least one ECG in-window."
            )
        else:
            print("    ECG features: no in-window ECG machine measurements found.")

        df.to_csv(matrix_path, index=False)
        del df
        gc.collect()
        return matrix_path

    def add_medication_features(self, matrix_path):
        if not getattr(self.cfg, "USE_MEDICATION_FEATURES", False):
            return matrix_path

        prescriptions_path = getattr(self.cfg, "PRESCRIPTIONS_PATH", None)
        if not prescriptions_path or not os.path.exists(prescriptions_path):
            print(f"WARNING: prescriptions file not found for medication features: {prescriptions_path}")
            return matrix_path

        print("[3.6/4] Extracting Windowed Medication Features...")
        df = pd.read_csv(matrix_path, low_memory=False)
        df["index_time"] = pd.to_datetime(df["index_time"])
        df["feature_window_end"] = pd.to_datetime(df["feature_window_end"], errors="coerce")
        default_window_end = df["index_time"] + pd.to_timedelta(self.cfg.DAY_0_WINDOW_HOURS, unit="h")
        df["feature_window_end"] = df["feature_window_end"].fillna(default_window_end)

        spine_times = df[["subject_id", "index_hadm_id", "index_time", "feature_window_end"]].copy()
        spine_times.rename(columns={"index_hadm_id": "hadm_id"}, inplace=True)
        spine_times["hadm_id"] = pd.to_numeric(spine_times["hadm_id"], errors="coerce").astype("Int64")
        hadm_ids = set(spine_times["hadm_id"].dropna().astype("int64"))
        medication_patterns = getattr(self.cfg, "MEDICATION_GROUP_PATTERNS", {})
        compiled_patterns = {
            group: re.compile("|".join(patterns), re.IGNORECASE)
            for group, patterns in medication_patterns.items()
            if patterns
        }
        if not compiled_patterns:
            print("    Medication features: no medication group patterns configured.")
            return matrix_path

        group_counts = []
        scanned_chunks = 0
        matched_hadm_rows = 0
        windowed_order_rows = 0
        usecols = ["subject_id", "hadm_id", "starttime", "drug", "formulary_drug_cd", "route"]
        for chunk in pd.read_csv(
            prescriptions_path,
            usecols=usecols,
            chunksize=getattr(self.cfg, "PRESCRIPTIONS_CHUNKSIZE", 1_000_000),
        ):
            scanned_chunks += 1
            chunk["hadm_id"] = pd.to_numeric(chunk["hadm_id"], errors="coerce")
            chunk = chunk[chunk["hadm_id"].isin(hadm_ids)].copy()
            if chunk.empty:
                if scanned_chunks % 10 == 0:
                    print(
                        "      scanned "
                        f"{scanned_chunks:,} prescription chunks; "
                        f"matched {matched_hadm_rows:,} admission rows; "
                        f"{windowed_order_rows:,} in-window rows"
                    )
                continue
            matched_hadm_rows += len(chunk)

            chunk["hadm_id"] = chunk["hadm_id"].astype("int64")
            chunk["starttime"] = pd.to_datetime(chunk["starttime"], errors="coerce")
            chunk.dropna(subset=["subject_id", "hadm_id", "starttime"], inplace=True)
            if chunk.empty:
                continue

            chunk = chunk.merge(spine_times, on=["subject_id", "hadm_id"], how="inner")
            chunk = chunk[
                (chunk["starttime"] >= chunk["index_time"])
                & (chunk["starttime"] <= chunk["feature_window_end"])
            ].copy()
            if chunk.empty:
                if scanned_chunks % 10 == 0:
                    print(
                        "      scanned "
                        f"{scanned_chunks:,} prescription chunks; "
                        f"matched {matched_hadm_rows:,} admission rows; "
                        f"{windowed_order_rows:,} in-window rows"
                    )
                continue
            windowed_order_rows += len(chunk)

            drug_text = (
                chunk["drug"].fillna("").astype(str)
                + " "
                + chunk["formulary_drug_cd"].fillna("").astype(str)
                + " "
                + chunk["route"].fillna("").astype(str)
            )
            for group, pattern in compiled_patterns.items():
                matched = drug_text.str.contains(pattern, regex=True)
                if matched.any():
                    group_counts.append(
                        chunk.loc[matched].groupby("subject_id").size().rename(group)
                    )

            del chunk
            gc.collect()
            if scanned_chunks % 10 == 0:
                print(
                    "      scanned "
                    f"{scanned_chunks:,} prescription chunks; "
                    f"matched {matched_hadm_rows:,} admission rows; "
                    f"{windowed_order_rows:,} in-window rows"
                )

        print(
            "    Medication scan complete: "
            f"{scanned_chunks:,} prescription chunks; "
            f"{matched_hadm_rows:,} admission rows matched the cohort; "
            f"{windowed_order_rows:,} orders fell in-window."
        )

        if group_counts:
            med_wide = pd.concat(group_counts, axis=1)
            med_wide = med_wide.T.groupby(level=0).sum().T
            for group in compiled_patterns:
                if group not in med_wide.columns:
                    med_wide[group] = 0
            med_wide = med_wide[list(compiled_patterns)].fillna(0)
            med_wide.rename(columns={group: f"med_{group}_count" for group in med_wide.columns}, inplace=True)
            for group in compiled_patterns:
                count_col = f"med_{group}_count"
                med_wide[f"med_{group}_any"] = (med_wide[count_col] > 0).astype(int)
            med_wide.reset_index(names="subject_id", inplace=True)
            df = df.merge(med_wide, on="subject_id", how="left")
            med_cols = [col for col in med_wide.columns if col != "subject_id"]
            df[med_cols] = df[med_cols].fillna(0)
            print(
                "    Medication features: "
                f"{int((df[[col for col in med_cols if col.endswith('_any')]].sum(axis=1) > 0).sum()):,} "
                "patients had at least one configured in-window medication."
            )
        else:
            print("    Medication features: no configured in-window medication orders found.")

        df.to_csv(matrix_path, index=False)
        del df
        gc.collect()
        return matrix_path

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
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_dir = Path(getattr(self.cfg, "MODEL_REPORT_DIR", "../data/processed/model_reports"))

    def preprocess(self, matrix_path):
        print("\n[4/4] Pre-processing & Cleaning Matrix...")
        df = pd.read_csv(matrix_path, low_memory=False)
        id_cols = [
            "subject_id",
            "index_hadm_id",
            "index_time",
            "index_time_source",
            "feature_window_end",
            "diagnosis_proxy_time",
            "diagnosis_time_source",
            "cohort_split",
            "is_aortic_dissection",
            "admission_location",
        ]
        cat_cols = ["gender", "race", "insurance", "marital_status", "encounter_urgency"]
        cat_cols = [c for c in cat_cols if c in df.columns]
        non_numeric_cols = id_cols + cat_cols
        features = [c for c in df.columns if c not in non_numeric_cols]

        if getattr(self.cfg, "REQUIRE_ECG_MEASUREMENTS", False):
            ecg_count_col = getattr(self.cfg, "ECG_REQUIRED_COUNT_COL", "ecg_count")
            if ecg_count_col not in df.columns:
                raise ValueError(
                    f"REQUIRE_ECG_MEASUREMENTS=True but {ecg_count_col!r} is missing from the matrix."
                )
            before_rows = len(df)
            before_targets = int(df["is_aortic_dissection"].sum())
            before_controls = before_rows - before_targets
            df[ecg_count_col] = pd.to_numeric(df[ecg_count_col], errors="coerce")
            df = df[df[ecg_count_col].fillna(0) > 0].copy()
            after_rows = len(df)
            after_targets = int(df["is_aortic_dissection"].sum())
            after_controls = after_rows - after_targets
            print(
                "    ECG-complete cohort filter: "
                f"kept {after_rows:,}/{before_rows:,} rows "
                f"({after_targets:,}/{before_targets:,} targets; "
                f"{after_controls:,}/{before_controls:,} controls) with in-window ECG machine measurements."
            )
            if df.empty:
                raise ValueError("No rows remained after requiring ECG machine measurements.")
 
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

        if getattr(self.cfg, "USE_XGB_NATIVE_CATEGORICAL", True):
            train_encoded = train_df.copy()
            test_encoded = test_df.copy()
            for col in cat_cols:
                train_encoded[col] = train_encoded[col].fillna("MISSING").astype(str)
                test_encoded[col] = test_encoded[col].fillna("MISSING").astype(str)
                categories = sorted(
                    set(train_encoded[col].unique())
                    | set(test_encoded[col].unique())
                )
                dtype = pd.CategoricalDtype(categories=categories)
                train_encoded[col] = train_encoded[col].astype(dtype)
                test_encoded[col] = test_encoded[col].astype(dtype)
        else:
            train_encoded = pd.get_dummies(train_df, columns=cat_cols, drop_first=True)
            test_encoded = pd.get_dummies(test_df, columns=cat_cols, drop_first=True)

        y_train = train_encoded["is_aortic_dissection"].astype(int)
        y_test = test_encoded["is_aortic_dissection"].astype(int)
        drop_cols = ["subject_id", "index_hadm_id", "index_time", "cohort_split", "is_aortic_dissection"]
        X_train = train_encoded.drop(columns=[c for c in drop_cols if c in train_encoded.columns])
        X_test = test_encoded.drop(columns=[c for c in drop_cols if c in test_encoded.columns])
        X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)
        features_to_drop = [
            col
            for col in getattr(self.cfg, "FEATURES_TO_DROP", [])
            if col in X_train.columns
        ]
        if features_to_drop:
            X_train.drop(columns=features_to_drop, inplace=True)
            X_test.drop(columns=features_to_drop, inplace=True)
            print(f"    Pruned configured low-importance features: {features_to_drop}")
        features_to_keep = self._configured_features_to_keep()
        if features_to_keep:
            missing_features = [col for col in features_to_keep if col not in X_train.columns]
            if missing_features:
                raise ValueError(
                    "Configured feature set contains features missing after preprocessing: "
                    f"{missing_features}"
                )
            X_train = X_train[features_to_keep].copy()
            X_test = X_test[features_to_keep].copy()
            preset_name = getattr(self.cfg, "ACTIVE_FEATURE_SET_PRESET", None)
            label = f" preset {preset_name!r}" if preset_name else ""
            print(f"    Applied fixed feature set{label}: {len(features_to_keep):,} features.")
        numeric_cols = [col for col in X_train.columns if col not in cat_cols]
        X_train[numeric_cols] = X_train[numeric_cols].astype(float)
        X_test[numeric_cols] = X_test[numeric_cols].astype(float)

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

    def _configured_features_to_keep(self):
        direct_features = list(getattr(self.cfg, "FEATURES_TO_KEEP", []) or [])
        preset_name = getattr(self.cfg, "ACTIVE_FEATURE_SET_PRESET", None)
        if direct_features and preset_name:
            raise ValueError("Set either FEATURES_TO_KEEP or ACTIVE_FEATURE_SET_PRESET, not both.")
        if direct_features:
            return direct_features
        if not preset_name:
            return []
        presets = getattr(self.cfg, "FEATURE_SET_PRESETS", {})
        if preset_name not in presets:
            raise ValueError(f"Unknown ACTIVE_FEATURE_SET_PRESET: {preset_name}")
        return list(presets[preset_name])
 
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
        candidate_rows = []

        print(f"    Trying {len(param_grid)} XGBoost parameter set(s) on the training validation split...")
        for idx, params in enumerate(param_grid, start=1):
            scale_weight = (y_fit == 0).sum() / (y_fit == 1).sum()
            model = xgb.XGBClassifier(
                **params,
                scale_pos_weight=scale_weight,
                eval_metric=self.cfg.XGB_EVAL_METRIC,
                random_state=self.cfg.RANDOM_STATE,
                tree_method="hist",
                enable_categorical=getattr(self.cfg, "USE_XGB_NATIVE_CATEGORICAL", True),
                n_jobs=-1,
            )
            model.fit(X_fit, y_fit)
            val_prob = model.predict_proba(X_val)[:, 1]
            val_ap = average_precision_score(y_val, val_prob)
            print(f"      candidate {idx}: validation average precision={val_ap:.4f}; params={params}")
            candidate_rows.append({
                "model_label": "main",
                "candidate": idx,
                "validation_average_precision": val_ap,
                "params_json": json.dumps(params, sort_keys=True),
            })
            if val_ap > best_ap:
                best_ap = val_ap
                best_params = params
                best_val_prob = val_prob

        selected_features = list(X_train.columns)
        rfe_iterations = 0
        rfe_dropped_features = []
        if getattr(self.cfg, "USE_RECURSIVE_FEATURE_ELIMINATION", False):
            (
                selected_features,
                best_val_prob,
                best_ap,
                rfe_iterations,
                rfe_dropped_features,
            ) = self._run_recursive_feature_elimination(
                X_fit,
                y_fit,
                X_val,
                y_val,
                best_params,
            )
            X_train = X_train[selected_features].copy()
            X_test = X_test[selected_features].copy()

        shadow_report = None
        if getattr(self.cfg, "RUN_SHADOW_FEATURE_FILTER_REPORT", False):
            shadow_report = self._run_shadow_feature_filter_report(
                X_fit[selected_features].copy(),
                y_fit,
                X_val[selected_features].copy(),
                y_val,
                best_params,
                model_label="main",
            )

        threshold = self._select_threshold(y_val, best_val_prob)
        print(f"    Selected threshold on validation split: {threshold:.4f}")

        scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
        model = xgb.XGBClassifier(
            **best_params,
            scale_pos_weight=scale_weight,
            eval_metric=self.cfg.XGB_EVAL_METRIC,
            random_state=self.cfg.RANDOM_STATE,
            tree_method="hist",
            enable_categorical=getattr(self.cfg, "USE_XGB_NATIVE_CATEGORICAL", True),
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        print("\n1. Holdout Evaluation at Tuned Threshold:")
        self._print_threshold_metrics(y_test, y_prob, threshold)
        print("\n2. Holdout Evaluation at Default 0.5000 Threshold:")
        self._print_threshold_metrics(y_test, y_prob, 0.5)
        print("\n3. Holdout Threshold Operating Points:")
        operating_points = self._print_threshold_operating_points(y_test, y_prob, threshold)

        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        holdout_ap = average_precision_score(y_test, y_prob)
        holdout_pr_auc = auc(recall, precision)
        holdout_roc_auc = roc_auc_score(y_test, y_prob)
        print(f"\n4. Holdout Average Precision: {holdout_ap:.4f}")
        print(f"   Holdout PR-AUC (trapezoidal): {holdout_pr_auc:.4f}")
        print(f"   Holdout ROC-AUC: {holdout_roc_auc:.4f}")
        print(f"   Best validation average precision: {best_ap:.4f}")
        print(f"   Best XGBoost params: {best_params}")
        cv_metrics = None
        if getattr(self.cfg, "RUN_CROSS_VALIDATION", False):
            cv_metrics = self._run_cross_validation_report(
                X_train,
                y_train,
                best_params,
                model_label="main",
            )
        imp = pd.DataFrame({'Feature': X_train.columns, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False)
        print("\n5. Top 25 Features:")
        print(imp.head(25).to_string(index=False))
        self._print_low_importance_features(imp)
        self._write_model_reports(
            model_label="main",
            importance_df=imp,
            operating_points=operating_points,
            summary_metrics=self._summary_metrics(
                model_label="main",
                y_train=y_train,
                y_test=y_test,
                threshold=threshold,
                best_ap=best_ap,
                best_params=best_params,
                holdout_ap=holdout_ap,
                holdout_pr_auc=holdout_pr_auc,
                holdout_roc_auc=holdout_roc_auc,
                operating_points=operating_points,
                feature_count=X_train.shape[1],
                rfe_iterations=rfe_iterations,
                rfe_dropped_features=rfe_dropped_features,
                cv_metrics=cv_metrics,
            ),
            candidate_metrics=pd.DataFrame(candidate_rows),
        )
        self._write_curve_reports("main", y_test, y_prob)

        if (
            getattr(self.cfg, "RUN_SHADOW_REJECT_COMPARISON", False)
            and shadow_report is not None
        ):
            reject_features = shadow_report.loc[
                shadow_report["shadow_filter_status"] == "reject",
                "feature",
            ].tolist()
            self._run_feature_drop_comparison(
                X_train,
                y_train,
                X_test,
                y_test,
                drop_cols=reject_features,
                model_label="without_shadow_rejects",
                reason="shadow-filter reject features",
            )

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
        candidate_rows = []

        print(f"    Trying {len(param_grid)} XGBoost parameter set(s) without INR features...")
        for idx, params in enumerate(param_grid, start=1):
            scale_weight = (y_fit == 0).sum() / (y_fit == 1).sum()
            model = xgb.XGBClassifier(
                **params,
                scale_pos_weight=scale_weight,
                eval_metric=self.cfg.XGB_EVAL_METRIC,
                random_state=self.cfg.RANDOM_STATE,
                tree_method="hist",
                enable_categorical=getattr(self.cfg, "USE_XGB_NATIVE_CATEGORICAL", True),
                n_jobs=-1,
            )
            model.fit(X_fit, y_fit)
            val_prob = model.predict_proba(X_val)[:, 1]
            val_ap = average_precision_score(y_val, val_prob)
            print(f"      candidate {idx}: validation average precision={val_ap:.4f}; params={params}")
            candidate_rows.append({
                "model_label": "without_inr",
                "candidate": idx,
                "validation_average_precision": val_ap,
                "params_json": json.dumps(params, sort_keys=True),
            })
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
            enable_categorical=getattr(self.cfg, "USE_XGB_NATIVE_CATEGORICAL", True),
            n_jobs=-1,
        )
        model.fit(X_train_ablate, y_train)
        y_prob = model.predict_proba(X_test_ablate)[:, 1]

        print("\nA. Holdout Evaluation Without INR at Tuned Threshold:")
        self._print_threshold_metrics(y_test, y_prob, threshold)
        print("\nB. Holdout Threshold Operating Points Without INR:")
        operating_points = self._print_threshold_operating_points(y_test, y_prob, threshold)

        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        holdout_ap = average_precision_score(y_test, y_prob)
        holdout_pr_auc = auc(recall, precision)
        holdout_roc_auc = roc_auc_score(y_test, y_prob)
        print(f"\nC. Holdout Average Precision Without INR: {holdout_ap:.4f}")
        print(f"   Holdout PR-AUC Without INR (trapezoidal): {holdout_pr_auc:.4f}")
        print(f"   Holdout ROC-AUC Without INR: {holdout_roc_auc:.4f}")
        print(f"   Best validation average precision without INR: {best_ap:.4f}")
        print(f"   Best XGBoost params without INR: {best_params}")
        imp = pd.DataFrame({
            "Feature": X_train_ablate.columns,
            "Importance": model.feature_importances_,
        }).sort_values(by="Importance", ascending=False)
        print("\nD. Top 25 Features Without INR:")
        print(imp.head(25).to_string(index=False))
        self._print_low_importance_features(imp, label="Without INR")
        self._write_model_reports(
            model_label="without_inr",
            importance_df=imp,
            operating_points=operating_points,
            summary_metrics=self._summary_metrics(
                model_label="without_inr",
                y_train=y_train,
                y_test=y_test,
                threshold=threshold,
                best_ap=best_ap,
                best_params=best_params,
                holdout_ap=holdout_ap,
                holdout_pr_auc=holdout_pr_auc,
                holdout_roc_auc=holdout_roc_auc,
                operating_points=operating_points,
                feature_count=X_train_ablate.shape[1],
                rfe_enabled=False,
            ),
            candidate_metrics=pd.DataFrame(candidate_rows),
        )

    def _run_feature_drop_comparison(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        drop_cols,
        model_label,
        reason,
    ):
        import xgboost as xgb

        drop_cols = [col for col in drop_cols if col in X_train.columns]
        if not drop_cols:
            print(f"\n=== {model_label} Comparison Skipped ===")
            print(f"    No configured {reason} were present in the model matrix.")
            return

        print(f"\n=== {model_label} Comparison ===")
        print(
            f"    Dropping {len(drop_cols):,} {reason} for this comparison only; "
            f"{X_train.shape[1] - len(drop_cols):,} features remain."
        )
        print(f"    Dropped features: {drop_cols}")

        X_train_compare = X_train.drop(columns=drop_cols)
        X_test_compare = X_test.drop(columns=drop_cols)
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train_compare,
            y_train,
            test_size=self.cfg.VALIDATION_SIZE,
            random_state=self.cfg.RANDOM_STATE,
            stratify=y_train,
        )
        param_grid = getattr(self.cfg, "XGB_PARAM_GRID", [{}])
        best_params = None
        best_val_prob = None
        best_ap = -1
        candidate_rows = []

        print(f"    Trying {len(param_grid)} XGBoost parameter set(s) without {reason}...")
        for idx, params in enumerate(param_grid, start=1):
            scale_weight = (y_fit == 0).sum() / (y_fit == 1).sum()
            model = xgb.XGBClassifier(
                **params,
                scale_pos_weight=scale_weight,
                eval_metric=self.cfg.XGB_EVAL_METRIC,
                random_state=self.cfg.RANDOM_STATE,
                tree_method="hist",
                enable_categorical=getattr(self.cfg, "USE_XGB_NATIVE_CATEGORICAL", True),
                n_jobs=-1,
            )
            model.fit(X_fit, y_fit)
            val_prob = model.predict_proba(X_val)[:, 1]
            val_ap = average_precision_score(y_val, val_prob)
            print(f"      candidate {idx}: validation average precision={val_ap:.4f}; params={params}")
            candidate_rows.append({
                "model_label": model_label,
                "candidate": idx,
                "validation_average_precision": val_ap,
                "params_json": json.dumps(params, sort_keys=True),
            })
            if val_ap > best_ap:
                best_ap = val_ap
                best_params = params
                best_val_prob = val_prob

        threshold = self._select_threshold(y_val, best_val_prob)
        print(f"    Selected threshold on validation split for {model_label}: {threshold:.4f}")

        scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
        model = xgb.XGBClassifier(
            **best_params,
            scale_pos_weight=scale_weight,
            eval_metric=self.cfg.XGB_EVAL_METRIC,
            random_state=self.cfg.RANDOM_STATE,
            tree_method="hist",
            enable_categorical=getattr(self.cfg, "USE_XGB_NATIVE_CATEGORICAL", True),
            n_jobs=-1,
        )
        model.fit(X_train_compare, y_train)
        y_prob = model.predict_proba(X_test_compare)[:, 1]

        print(f"\n{model_label}: Holdout Evaluation at Tuned Threshold:")
        self._print_threshold_metrics(y_test, y_prob, threshold)
        print(f"\n{model_label}: Holdout Threshold Operating Points:")
        operating_points = self._print_threshold_operating_points(y_test, y_prob, threshold)

        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        holdout_ap = average_precision_score(y_test, y_prob)
        holdout_pr_auc = auc(recall, precision)
        holdout_roc_auc = roc_auc_score(y_test, y_prob)
        print(f"\n{model_label}: Holdout Average Precision: {holdout_ap:.4f}")
        print(f"   Holdout PR-AUC (trapezoidal): {holdout_pr_auc:.4f}")
        print(f"   Holdout ROC-AUC: {holdout_roc_auc:.4f}")
        print(f"   Best validation average precision: {best_ap:.4f}")
        print(f"   Best XGBoost params: {best_params}")

        imp = pd.DataFrame({
            "Feature": X_train_compare.columns,
            "Importance": model.feature_importances_,
        }).sort_values(by="Importance", ascending=False)
        print(f"\n{model_label}: Top 25 Features:")
        print(imp.head(25).to_string(index=False))
        self._print_low_importance_features(imp, label=model_label)
        self._write_model_reports(
            model_label=model_label,
            importance_df=imp,
            operating_points=operating_points,
            summary_metrics=self._summary_metrics(
                model_label=model_label,
                y_train=y_train,
                y_test=y_test,
                threshold=threshold,
                best_ap=best_ap,
                best_params=best_params,
                holdout_ap=holdout_ap,
                holdout_pr_auc=holdout_pr_auc,
                holdout_roc_auc=holdout_roc_auc,
                operating_points=operating_points,
                feature_count=X_train_compare.shape[1],
                rfe_enabled=False,
            ),
            candidate_metrics=pd.DataFrame(candidate_rows),
        )

    def _run_recursive_feature_elimination(self, X_fit, y_fit, X_val, y_val, params):
        import xgboost as xgb

        metric = getattr(self.cfg, "RFE_PERFORMANCE_METRIC", "average_precision")
        if metric != "average_precision":
            raise ValueError("RFE currently supports RFE_PERFORMANCE_METRIC='average_precision'.")

        strategy = getattr(self.cfg, "RFE_STRATEGY", "greedy_backward_ablation")
        if strategy != "greedy_backward_ablation":
            raise ValueError("RFE currently supports RFE_STRATEGY='greedy_backward_ablation'.")

        selected_features = list(X_fit.columns)
        dropped_features = []
        min_features = max(1, int(getattr(self.cfg, "RFE_MIN_FEATURES", 1)))
        tolerance = float(getattr(self.cfg, "RFE_PERFORMANCE_TOLERANCE", 0.0))
        max_iterations = getattr(self.cfg, "RFE_MAX_ITERATIONS", None)
        protected_features = [
            feature
            for feature in getattr(self.cfg, "RFE_PROTECTED_FEATURES", [])
            if feature in selected_features
        ]
        scale_weight = (y_fit == 0).sum() / (y_fit == 1).sum()

        def fit_validation_model(features):
            model = xgb.XGBClassifier(
                **params,
                scale_pos_weight=scale_weight,
                eval_metric=self.cfg.XGB_EVAL_METRIC,
                random_state=self.cfg.RANDOM_STATE,
                tree_method="hist",
                enable_categorical=getattr(self.cfg, "USE_XGB_NATIVE_CATEGORICAL", True),
                n_jobs=-1,
            )
            model.fit(X_fit[features], y_fit)
            val_prob = model.predict_proba(X_val[features])[:, 1]
            return average_precision_score(y_val, val_prob), val_prob

        print("\n=== Recursive Feature Elimination ===")
        print(
            "    Strategy: greedy single-feature backward ablation; "
            "selection metric: validation average precision; "
            f"min features={min_features}; allowed AP drop per step={tolerance:g}"
        )
        if protected_features:
            print(f"    Protected from RFE removal: {protected_features}")

        current_ap, current_val_prob = fit_validation_model(selected_features)
        best_ap = current_ap
        best_val_prob = current_val_prob
        best_features = selected_features.copy()
        print(
            f"    RFE baseline: {len(selected_features):,} features, "
            f"validation AP={current_ap:.4f}"
        )

        iteration = 0
        while len(selected_features) > min_features:
            removable_features = [
                feature for feature in selected_features
                if feature not in protected_features
            ]
            if not removable_features:
                print("    Stopping RFE because no removable features remain after protection rules.")
                break

            iteration += 1
            if max_iterations is not None and iteration > int(max_iterations):
                print("    Stopping RFE because the configured maximum iteration count was reached.")
                break

            print(
                f"    RFE pass {iteration}: testing {len(removable_features):,} "
                "single-feature removals..."
            )
            pass_best_feature = None
            pass_best_ap = -np.inf
            pass_best_prob = None
            pass_best_delta = -np.inf

            for idx, feature in enumerate(removable_features, start=1):
                candidate_features = [col for col in selected_features if col != feature]
                candidate_ap, candidate_prob = fit_validation_model(candidate_features)
                delta = candidate_ap - current_ap

                if idx == 1 or idx % 25 == 0 or idx == len(removable_features):
                    print(
                        f"      evaluated {idx:,}/{len(removable_features):,} removals; "
                        f"current best delta={pass_best_delta:+.4f}"
                    )

                if candidate_ap > pass_best_ap:
                    pass_best_feature = feature
                    pass_best_ap = candidate_ap
                    pass_best_prob = candidate_prob
                    pass_best_delta = delta

            if pass_best_feature is None or pass_best_ap < current_ap - tolerance:
                print(
                    "    Stopping RFE because the best single-feature removal "
                    f"changed validation AP by {pass_best_delta:+.4f}, "
                    f"which exceeds the allowed drop of {tolerance:.4f}."
                )
                break

            selected_features = [col for col in selected_features if col != pass_best_feature]
            dropped_features.append(pass_best_feature)
            current_ap = pass_best_ap
            current_val_prob = pass_best_prob
            if (
                current_ap > best_ap
                or (
                    np.isclose(current_ap, best_ap)
                    and len(selected_features) < len(best_features)
                )
            ):
                best_ap = current_ap
                best_val_prob = current_val_prob
                best_features = selected_features.copy()
            action = "improved" if pass_best_delta >= 0 else "changed"
            print(
                f"      dropped {pass_best_feature}; "
                f"validation AP {action} by {pass_best_delta:+.4f} to {current_ap:.4f}; "
                f"{len(selected_features):,} features remain."
            )

        final_dropped = [col for col in X_fit.columns if col not in best_features]
        print(
            "    RFE selected best validation subset: "
            f"{len(best_features):,}/{X_fit.shape[1]:,} features; "
            f"dropped {len(final_dropped):,}; best validation AP={best_ap:.4f}."
        )
        if final_dropped:
            print(f"    Final RFE-dropped features: {final_dropped}")

        return best_features, best_val_prob, best_ap, iteration, final_dropped

    def _run_shadow_feature_filter_report(self, X_fit, y_fit, X_val, y_val, params, model_label):
        import xgboost as xgb

        repeats = max(1, int(getattr(self.cfg, "SHADOW_FILTER_REPEATS", 5)))
        keep_hit_rate = float(getattr(self.cfg, "SHADOW_FILTER_KEEP_HIT_RATE", 0.80))
        reject_hit_rate = float(getattr(self.cfg, "SHADOW_FILTER_REJECT_HIT_RATE", 0.20))
        scale_weight = (y_fit == 0).sum() / (y_fit == 1).sum()
        real_features = list(X_fit.columns)
        real_rows = []
        repeat_rows = []

        print("\n=== Shadow Feature Filter Report ===")
        print(
            f"    Running {repeats} shadow-feature repeat(s) on "
            f"{len(real_features):,} selected features; report only, no automatic drops."
        )

        for repeat_idx in range(1, repeats + 1):
            seed = self.cfg.RANDOM_STATE + repeat_idx * 1009
            rng = np.random.default_rng(seed)
            X_shadow_fit = X_fit.copy()
            X_shadow_val = X_val.copy()
            shadow_features = []

            for feature in real_features:
                shadow_col = f"shadow__{feature}"
                shadow_features.append(shadow_col)
                fit_values = X_fit[feature].to_numpy(copy=True)
                val_values = X_val[feature].to_numpy(copy=True)
                rng.shuffle(fit_values)
                rng.shuffle(val_values)
                X_shadow_fit[shadow_col] = pd.Series(fit_values, index=X_fit.index)
                X_shadow_val[shadow_col] = pd.Series(val_values, index=X_val.index)
                if hasattr(X_fit[feature].dtype, "categories"):
                    X_shadow_fit[shadow_col] = X_shadow_fit[shadow_col].astype(X_fit[feature].dtype)
                    X_shadow_val[shadow_col] = X_shadow_val[shadow_col].astype(X_val[feature].dtype)

            model = xgb.XGBClassifier(
                **params,
                scale_pos_weight=scale_weight,
                eval_metric=self.cfg.XGB_EVAL_METRIC,
                random_state=seed,
                tree_method="hist",
                enable_categorical=getattr(self.cfg, "USE_XGB_NATIVE_CATEGORICAL", True),
                n_jobs=-1,
            )
            model.fit(X_shadow_fit, y_fit)
            val_prob = model.predict_proba(X_shadow_val)[:, 1]
            val_ap = average_precision_score(y_val, val_prob)

            importance = pd.Series(model.feature_importances_, index=X_shadow_fit.columns)
            shadow_max = float(importance[shadow_features].max())
            shadow_mean = float(importance[shadow_features].mean())
            shadow_p95 = float(importance[shadow_features].quantile(0.95))
            repeat_rows.append({
                "run_id": self.run_id,
                "model_label": model_label,
                "repeat": repeat_idx,
                "seed": seed,
                "feature_count": len(real_features),
                "shadow_max_importance": shadow_max,
                "shadow_mean_importance": shadow_mean,
                "shadow_p95_importance": shadow_p95,
                "validation_average_precision_with_shadows": val_ap,
            })

            for feature in real_features:
                real_importance = float(importance[feature])
                real_rows.append({
                    "run_id": self.run_id,
                    "model_label": model_label,
                    "repeat": repeat_idx,
                    "feature": feature,
                    "real_importance": real_importance,
                    "shadow_max_importance": shadow_max,
                    "shadow_mean_importance": shadow_mean,
                    "shadow_p95_importance": shadow_p95,
                    "beats_shadow_max": real_importance > shadow_max,
                    "importance_minus_shadow_max": real_importance - shadow_max,
                })

            print(
                f"    repeat {repeat_idx}/{repeats}: "
                f"shadow max={shadow_max:.6f}, validation AP with shadows={val_ap:.4f}"
            )

            del X_shadow_fit, X_shadow_val, model
            gc.collect()

        raw_df = pd.DataFrame(real_rows)
        report_df = (
            raw_df.groupby(["run_id", "model_label", "feature"], as_index=False)
            .agg(
                mean_real_importance=("real_importance", "mean"),
                median_real_importance=("real_importance", "median"),
                max_real_importance=("real_importance", "max"),
                mean_shadow_max_importance=("shadow_max_importance", "mean"),
                mean_shadow_p95_importance=("shadow_p95_importance", "mean"),
                hit_count=("beats_shadow_max", "sum"),
                hit_rate=("beats_shadow_max", "mean"),
                mean_importance_minus_shadow_max=("importance_minus_shadow_max", "mean"),
            )
        )

        report_df["shadow_filter_status"] = "tentative"
        report_df.loc[report_df["hit_rate"] >= keep_hit_rate, "shadow_filter_status"] = "keep"
        report_df.loc[report_df["hit_rate"] <= reject_hit_rate, "shadow_filter_status"] = "reject"
        report_df["shadow_filter_repeats"] = repeats
        report_df["keep_hit_rate_threshold"] = keep_hit_rate
        report_df["reject_hit_rate_threshold"] = reject_hit_rate
        report_df = report_df.sort_values(
            by=["shadow_filter_status", "hit_rate", "mean_importance_minus_shadow_max"],
            ascending=[True, False, False],
        )

        status_counts = report_df["shadow_filter_status"].value_counts().to_dict()
        print(
            "    Shadow filter status counts: "
            f"keep={status_counts.get('keep', 0)}, "
            f"tentative={status_counts.get('tentative', 0)}, "
            f"reject={status_counts.get('reject', 0)}"
        )
        rejected = report_df.loc[report_df["shadow_filter_status"] == "reject", "feature"].tolist()
        if rejected:
            print(f"    Shadow-filter reject candidates: {rejected}")

        self._write_shadow_feature_reports(
            model_label=model_label,
            feature_report=report_df,
            repeat_report=pd.DataFrame(repeat_rows),
        )
        return report_df

    def _print_low_importance_features(self, importance_df, label=None):
        report_n = getattr(self.cfg, "LOW_IMPORTANCE_REPORT_N", 0)
        if not report_n:
            return
        title = "Lowest-Importance Features"
        if label:
            title += f" {label}"
        print(f"\n{title}:")
        print(importance_df.tail(report_n).to_string(index=False))

    def _run_cross_validation_report(self, X_train, y_train, params, model_label):
        import xgboost as xgb

        folds = int(getattr(self.cfg, "CV_FOLDS", 5))
        splitter = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=self.cfg.RANDOM_STATE,
        )
        rows = []
        print(f"\n=== {folds}-Fold Cross-Validation on Training Cohort ===")
        for fold_idx, (fit_idx, val_idx) in enumerate(splitter.split(X_train, y_train), start=1):
            X_fit = X_train.iloc[fit_idx]
            X_val = X_train.iloc[val_idx]
            y_fit = y_train.iloc[fit_idx]
            y_val = y_train.iloc[val_idx]
            scale_weight = (y_fit == 0).sum() / (y_fit == 1).sum()
            model = xgb.XGBClassifier(
                **params,
                scale_pos_weight=scale_weight,
                eval_metric=self.cfg.XGB_EVAL_METRIC,
                random_state=self.cfg.RANDOM_STATE + fold_idx,
                tree_method="hist",
                enable_categorical=getattr(self.cfg, "USE_XGB_NATIVE_CATEGORICAL", True),
                n_jobs=-1,
            )
            model.fit(X_fit, y_fit)
            val_prob = model.predict_proba(X_val)[:, 1]
            precision, recall, _ = precision_recall_curve(y_val, val_prob)
            fold_ap = average_precision_score(y_val, val_prob)
            fold_pr_auc = auc(recall, precision)
            fold_roc_auc = roc_auc_score(y_val, val_prob)
            rows.append({
                "run_id": self.run_id,
                "model_label": model_label,
                "fold": fold_idx,
                "train_rows": int(len(y_fit)),
                "validation_rows": int(len(y_val)),
                "validation_positive": int(y_val.sum()),
                "validation_prevalence": float(y_val.mean()),
                "average_precision": fold_ap,
                "pr_auc_trapezoidal": fold_pr_auc,
                "roc_auc": fold_roc_auc,
            })
            print(
                f"    fold {fold_idx}/{folds}: "
                f"AP={fold_ap:.4f}; PR-AUC={fold_pr_auc:.4f}; ROC-AUC={fold_roc_auc:.4f}"
            )

        cv_metrics = pd.DataFrame(rows)
        print(
            "    CV mean +/- SD: "
            f"AP={cv_metrics['average_precision'].mean():.4f} +/- {cv_metrics['average_precision'].std(ddof=1):.4f}; "
            f"PR-AUC={cv_metrics['pr_auc_trapezoidal'].mean():.4f} +/- {cv_metrics['pr_auc_trapezoidal'].std(ddof=1):.4f}; "
            f"ROC-AUC={cv_metrics['roc_auc'].mean():.4f} +/- {cv_metrics['roc_auc'].std(ddof=1):.4f}"
        )

        if getattr(self.cfg, "WRITE_MODEL_REPORTS", True):
            self.report_dir.mkdir(parents=True, exist_ok=True)
            safe_label = re.sub(r"[^A-Za-z0-9_]+", "_", model_label).strip("_")
            path = self.report_dir / f"{self.run_id}_{safe_label}_cv_metrics.csv"
            cv_metrics.to_csv(path, index=False)
            print(f"    CV metrics: {path}")
        return cv_metrics

    def _write_curve_reports(self, model_label, y_test, y_prob):
        if not getattr(self.cfg, "WRITE_MODEL_REPORTS", True):
            return

        self.report_dir.mkdir(parents=True, exist_ok=True)
        safe_label = re.sub(r"[^A-Za-z0-9_]+", "_", model_label).strip("_")
        prefix = self.report_dir / f"{self.run_id}_{safe_label}"

        precision, recall, pr_thresholds = precision_recall_curve(y_test, y_prob)
        pr_curve = pd.DataFrame({
            "recall": recall,
            "precision": precision,
            "threshold": list(pr_thresholds) + [np.nan],
        })
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
        roc_curve_df = pd.DataFrame({
            "false_positive_rate": fpr,
            "true_positive_rate": tpr,
            "threshold": roc_thresholds,
        })
        predictions = pd.DataFrame({
            "y_true": y_test.to_numpy(),
            "y_probability": y_prob,
        })

        pr_path = prefix.with_name(f"{prefix.name}_pr_curve.csv")
        roc_path = prefix.with_name(f"{prefix.name}_roc_curve.csv")
        pred_path = prefix.with_name(f"{prefix.name}_holdout_predictions.csv")
        pr_curve.to_csv(pr_path, index=False)
        roc_curve_df.to_csv(roc_path, index=False)
        predictions.to_csv(pred_path, index=False)
        print("\nCurve report files written:")
        print(f"   PR curve:            {pr_path}")
        print(f"   ROC curve:           {roc_path}")
        print(f"   Holdout predictions: {pred_path}")

    def _summary_metrics(
        self,
        model_label,
        y_train,
        y_test,
        threshold,
        best_ap,
        best_params,
        holdout_ap,
        holdout_pr_auc,
        holdout_roc_auc,
        operating_points,
        feature_count=None,
        rfe_enabled=None,
        rfe_iterations=0,
        rfe_dropped_features=None,
        cv_metrics=None,
    ):
        selected = operating_points[operating_points["rule"] == "validation_f1"].iloc[0].to_dict()
        default = operating_points[operating_points["rule"] == "default_0.5"].iloc[0].to_dict()
        if rfe_dropped_features is None:
            rfe_dropped_features = []
        if rfe_enabled is None:
            rfe_enabled = bool(getattr(self.cfg, "USE_RECURSIVE_FEATURE_ELIMINATION", False))
        row = {
            "run_id": self.run_id,
            "model_label": model_label,
            "train_rows": int(len(y_train)),
            "train_positive": int(y_train.sum()),
            "train_prevalence": float(y_train.mean()),
            "holdout_rows": int(len(y_test)),
            "holdout_positive": int(y_test.sum()),
            "holdout_prevalence": float(y_test.mean()),
            "selected_threshold": float(threshold),
            "selected_precision": selected["precision"],
            "selected_recall": selected["recall"],
            "selected_specificity": selected["specificity"],
            "selected_f1": selected["f1"],
            "selected_tp": selected["tp"],
            "selected_fp": selected["fp"],
            "selected_fn": selected["fn"],
            "selected_tn": selected["tn"],
            "default_threshold_f1": default["f1"],
            "holdout_average_precision": holdout_ap,
            "holdout_pr_auc_trapezoidal": holdout_pr_auc,
            "holdout_roc_auc": holdout_roc_auc,
            "best_validation_average_precision": best_ap,
            "best_params_json": json.dumps(best_params, sort_keys=True),
            "feature_count": int(feature_count) if feature_count is not None else None,
            "rfe_enabled": bool(rfe_enabled),
            "rfe_iterations": int(rfe_iterations),
            "rfe_dropped_feature_count": int(len(rfe_dropped_features)),
            "rfe_dropped_features_json": json.dumps(rfe_dropped_features),
            "rfe_strategy": getattr(self.cfg, "RFE_STRATEGY", None),
            "rfe_performance_tolerance": float(getattr(self.cfg, "RFE_PERFORMANCE_TOLERANCE", 0.0)),
            "rfe_protected_features_json": json.dumps(getattr(self.cfg, "RFE_PROTECTED_FEATURES", [])),
            "feature_aggregation_mode": getattr(self.cfg, "FEATURE_AGGREGATION_MODE", "full"),
            "include_first_features": bool(getattr(self.cfg, "INCLUDE_FIRST_FEATURES", True)),
        }
        if cv_metrics is not None and not cv_metrics.empty:
            row.update({
                "cv_folds": int(len(cv_metrics)),
                "cv_average_precision_mean": float(cv_metrics["average_precision"].mean()),
                "cv_average_precision_sd": float(cv_metrics["average_precision"].std(ddof=1)),
                "cv_pr_auc_trapezoidal_mean": float(cv_metrics["pr_auc_trapezoidal"].mean()),
                "cv_pr_auc_trapezoidal_sd": float(cv_metrics["pr_auc_trapezoidal"].std(ddof=1)),
                "cv_roc_auc_mean": float(cv_metrics["roc_auc"].mean()),
                "cv_roc_auc_sd": float(cv_metrics["roc_auc"].std(ddof=1)),
            })
        return pd.DataFrame([row])

    def _write_model_reports(self, model_label, importance_df, operating_points, summary_metrics, candidate_metrics):
        if not getattr(self.cfg, "WRITE_MODEL_REPORTS", True):
            return

        self.report_dir.mkdir(parents=True, exist_ok=True)
        safe_label = re.sub(r"[^A-Za-z0-9_]+", "_", model_label).strip("_")
        prefix = self.report_dir / f"{self.run_id}_{safe_label}"

        importance_path = prefix.with_name(f"{prefix.name}_feature_importance.csv")
        operating_path = prefix.with_name(f"{prefix.name}_threshold_metrics.csv")
        summary_path = prefix.with_name(f"{prefix.name}_summary_metrics.csv")
        candidates_path = prefix.with_name(f"{prefix.name}_validation_candidates.csv")

        importance_df.assign(run_id=self.run_id, model_label=model_label).to_csv(importance_path, index=False)
        operating_points.assign(run_id=self.run_id, model_label=model_label).to_csv(operating_path, index=False)
        summary_metrics.to_csv(summary_path, index=False)
        candidate_metrics.assign(run_id=self.run_id).to_csv(candidates_path, index=False)

        print("\nModel report files written:")
        print(f"   Feature importances: {importance_path}")
        print(f"   Threshold metrics:   {operating_path}")
        print(f"   Summary metrics:     {summary_path}")
        print(f"   Validation grid:     {candidates_path}")

    def _write_shadow_feature_reports(self, model_label, feature_report, repeat_report):
        if not getattr(self.cfg, "WRITE_MODEL_REPORTS", True):
            return

        self.report_dir.mkdir(parents=True, exist_ok=True)
        safe_label = re.sub(r"[^A-Za-z0-9_]+", "_", model_label).strip("_")
        prefix = self.report_dir / f"{self.run_id}_{safe_label}"
        feature_path = prefix.with_name(f"{prefix.name}_shadow_feature_filter.csv")
        repeat_path = prefix.with_name(f"{prefix.name}_shadow_feature_filter_repeats.csv")

        feature_report.to_csv(feature_path, index=False)
        repeat_report.to_csv(repeat_path, index=False)

        print("\nShadow feature report files written:")
        print(f"   Feature status: {feature_path}")
        print(f"   Repeat audit:   {repeat_path}")

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
        return summary

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
 
