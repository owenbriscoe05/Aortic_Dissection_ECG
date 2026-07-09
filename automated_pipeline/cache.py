import gc
import json
import re
import shutil
import sqlite3
from pathlib import Path

import pandas as pd


class EventCache:
    """Single-file cache for filtered MIMIC event tables.

    The first run still scans the large source CSV once. Later runs reuse the
    indexed SQLite table as long as the source file and configured item IDs are
    unchanged.
    """

    SCHEMA_VERSION = 1
    EVENT_COLUMNS = ["subject_id", "charttime", "itemid", "valuenum"]

    def __init__(self, config):
        self.cfg = config
        self.cache_dir = Path(self.cfg.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "event_cache.sqlite"

    def ensure_table(self, cache_name, source_path, itemids):
        table_name = self._safe_identifier(cache_name)
        source_path = Path(source_path)
        itemids = sorted(int(itemid) for itemid in itemids)
        signature = self._signature(source_path, itemids)

        with self._connect() as conn:
            self._init_metadata(conn)
            if (
                not self.cfg.FORCE_REBUILD_EVENT_CACHE
                and self._metadata_matches(conn, table_name, signature)
                and self._table_exists(conn, table_name)
            ):
                print(f"    Reusing cached {cache_name}: {self.db_path}")
                return table_name

            print(f"    Building cached {cache_name} from {source_path}...")
            self._rebuild_table(conn, table_name, source_path, itemids)
            conn.execute(
                """
                INSERT OR REPLACE INTO cache_metadata(cache_name, signature_json)
                VALUES (?, ?)
                """,
                (table_name, json.dumps(signature, sort_keys=True)),
            )

        return table_name

    def iter_subject_events(self, table_name, subject_ids, itemids):
        table_name = self._safe_identifier(table_name)
        itemids = sorted(int(itemid) for itemid in itemids)
        query_chunksize = getattr(self.cfg, "EVENT_CACHE_QUERY_CHUNKSIZE", 1_000_000)

        with self._connect() as conn:
            conn.execute("DROP TABLE IF EXISTS temp.selected_subjects")
            conn.execute("CREATE TEMP TABLE selected_subjects(subject_id INTEGER PRIMARY KEY)")
            conn.executemany(
                "INSERT OR IGNORE INTO selected_subjects(subject_id) VALUES (?)",
                [(int(subject_id),) for subject_id in subject_ids],
            )

            placeholders = ",".join("?" for _ in itemids)
            query = f"""
                SELECT e.subject_id, e.charttime, e.itemid, e.valuenum
                FROM {table_name} e
                INNER JOIN selected_subjects s
                    ON s.subject_id = e.subject_id
                WHERE e.itemid IN ({placeholders})
            """
            for chunk in pd.read_sql_query(
                query,
                conn,
                params=itemids,
                chunksize=query_chunksize,
            ):
                yield chunk

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")
        return conn

    def _init_metadata(self, conn):
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_metadata (
                cache_name TEXT PRIMARY KEY,
                signature_json TEXT NOT NULL
            )
            """
        )

    def _metadata_matches(self, conn, cache_name, signature):
        row = conn.execute(
            "SELECT signature_json FROM cache_metadata WHERE cache_name = ?",
            (cache_name,),
        ).fetchone()
        if not row:
            return False
        return json.loads(row[0]) == signature

    def _table_exists(self, conn, table_name):
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        return row is not None

    def _rebuild_table(self, conn, table_name, source_path, itemids):
        build_table = self._safe_identifier(f"{table_name}__build")
        chunksize = getattr(self.cfg, "EVENT_CACHE_CHUNKSIZE", 3_000_000)

        conn.execute(f"DROP TABLE IF EXISTS {build_table}")
        conn.execute(
            f"""
            CREATE TABLE {build_table} (
                subject_id INTEGER NOT NULL,
                charttime TEXT NOT NULL,
                itemid INTEGER NOT NULL,
                valuenum REAL NOT NULL
            )
            """
        )

        rows_written = 0
        for chunk_number, chunk in enumerate(
            pd.read_csv(source_path, usecols=self.EVENT_COLUMNS, chunksize=chunksize),
            start=1,
        ):
            chunk = chunk[chunk["itemid"].isin(itemids)].copy()
            if chunk.empty:
                continue

            chunk["valuenum"] = pd.to_numeric(chunk["valuenum"], errors="coerce")
            chunk.dropna(subset=["subject_id", "charttime", "itemid", "valuenum"], inplace=True)
            if chunk.empty:
                continue

            chunk = chunk[self.EVENT_COLUMNS]
            chunk["subject_id"] = chunk["subject_id"].astype("int64")
            chunk["itemid"] = chunk["itemid"].astype("int64")
            chunk.to_sql(build_table, conn, if_exists="append", index=False)
            rows_written += len(chunk)

            if chunk_number % 10 == 0:
                print(f"      scanned {chunk_number:,} chunks; cached {rows_written:,} rows")
            del chunk
            gc.collect()

        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.execute(f"ALTER TABLE {build_table} RENAME TO {table_name}")
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_subject_item_time "
            f"ON {table_name}(subject_id, itemid, charttime)"
        )
        print(f"    Cached {rows_written:,} rows in table {table_name}")

    def _signature(self, source_path, itemids):
        stat = source_path.stat()
        return {
            "schema_version": self.SCHEMA_VERSION,
            "source_path": str(source_path.resolve()),
            "source_size": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
            "itemids": itemids,
        }

    def _safe_identifier(self, name):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"Unsafe SQLite identifier: {name}")
        return name


class MatrixCache:
    """Cache the raw patient-feature matrix before preprocessing/modeling.

    This cache is intentionally keyed to cohort and feature-extraction inputs,
    not model parameters. That lets threshold and XGBoost tuning runs skip the
    expensive large-table event aggregation while still rebuilding when the
    clinical feature window or cohort definition changes.
    """

    METADATA_FILE = "matrix_cache_metadata.json"
    MATRIX_FILE = "raw_matrix.csv"
    RESTORED_MATRIX_FILE = "temp_raw_matrix_cached.csv"

    def __init__(self, config):
        self.cfg = config
        self.cache_dir = Path(self.cfg.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.cache_dir / self.METADATA_FILE
        self.matrix_path = self.cache_dir / self.MATRIX_FILE

    def restore_raw_matrix(self):
        if not getattr(self.cfg, "USE_MATRIX_CACHE", False):
            return None
        if getattr(self.cfg, "FORCE_REBUILD_MATRIX_CACHE", False):
            print("    Matrix cache disabled for this run: FORCE_REBUILD_MATRIX_CACHE=True")
            return None
        if not self.matrix_path.exists() or not self.metadata_path.exists():
            return None

        cached_signature = json.loads(self.metadata_path.read_text())
        current_signature = self._signature()
        if cached_signature != current_signature:
            print("    Raw matrix cache is stale; rebuilding matrix.")
            return None

        restored_path = Path(self.RESTORED_MATRIX_FILE)
        shutil.copyfile(self.matrix_path, restored_path)
        print(f"    Reusing cached raw matrix: {self.matrix_path}")
        return str(restored_path)

    def store_raw_matrix(self, matrix_path):
        if not getattr(self.cfg, "USE_MATRIX_CACHE", False):
            return

        source_path = Path(matrix_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Cannot cache missing raw matrix: {source_path}")

        tmp_matrix_path = self.matrix_path.with_suffix(".csv.tmp")
        tmp_metadata_path = self.metadata_path.with_suffix(".json.tmp")
        shutil.copyfile(source_path, tmp_matrix_path)
        tmp_metadata_path.write_text(json.dumps(self._signature(), indent=2, sort_keys=True))
        tmp_matrix_path.replace(self.matrix_path)
        tmp_metadata_path.replace(self.metadata_path)
        print(f"    Cached raw matrix for faster reruns: {self.matrix_path}")

    def _signature(self):
        feature_config = {
            "matrix_cache_version": getattr(self.cfg, "MATRIX_CACHE_VERSION", 1),
            "min_age": self.cfg.MIN_AGE,
            "use_edregtime_as_index_time": self.cfg.USE_EDREGTIME_AS_INDEX_TIME,
            "control_downsample_ratio": self.cfg.CONTROL_DOWNSAMPLE_RATIO,
            "use_natural_prevalence_holdout": self.cfg.USE_NATURAL_PREVALENCE_HOLDOUT,
            "target_diagnosis_codes_by_version": self.cfg.TARGET_DIAGNOSIS_CODES_BY_VERSION,
            "use_clinically_similar_controls": self.cfg.USE_CLINICALLY_SIMILAR_CONTROLS,
            "clinically_similar_control_groups": self.cfg.CLINICALLY_SIMILAR_CONTROL_GROUPS,
            "control_exclusion_diagnosis_codes_by_version": self.cfg.CONTROL_EXCLUSION_DIAGNOSIS_CODES_BY_VERSION,
            "use_training_hard_controls": self.cfg.USE_TRAINING_HARD_CONTROLS,
            "hard_control_admission_types": self.cfg.HARD_CONTROL_ADMISSION_TYPES,
            "hard_control_admission_locations": self.cfg.HARD_CONTROL_ADMISSION_LOCATIONS,
            "test_size": self.cfg.TEST_SIZE,
            "random_state": self.cfg.RANDOM_STATE,
            "day_0_window_hours": self.cfg.DAY_0_WINDOW_HOURS,
            "use_diagnosis_time_censoring": self.cfg.USE_DIAGNOSIS_TIME_CENSORING,
            "diagnosis_time_source": self.cfg.DIAGNOSIS_TIME_SOURCE,
            "diagnosis_time_require_result_section": self.cfg.DIAGNOSIS_TIME_REQUIRE_RESULT_SECTION,
            "diagnosis_time_result_section_headers": self.cfg.DIAGNOSIS_TIME_RESULT_SECTION_HEADERS,
            "diagnosis_time_require_diagnostic_exam": self.cfg.DIAGNOSIS_TIME_REQUIRE_DIAGNOSTIC_EXAM,
            "diagnosis_time_exam_include_patterns": self.cfg.DIAGNOSIS_TIME_EXAM_INCLUDE_PATTERNS,
            "diagnosis_time_exam_anatomy_patterns": self.cfg.DIAGNOSIS_TIME_EXAM_ANATOMY_PATTERNS,
            "diagnosis_time_exam_exclude_patterns": self.cfg.DIAGNOSIS_TIME_EXAM_EXCLUDE_PATTERNS,
            "diagnosis_time_text_patterns": self.cfg.DIAGNOSIS_TIME_TEXT_PATTERNS,
            "diagnosis_time_negation_patterns": self.cfg.DIAGNOSIS_TIME_NEGATION_PATTERNS,
            "vitals_dict": self.cfg.VITALS_DICT,
            "labs_dict": self.cfg.LABS_DICT,
            "use_ecg_features": self.cfg.USE_ECG_FEATURES,
            "ecg_measurement_features": self.cfg.ECG_MEASUREMENT_FEATURES,
            "ecg_derived_features": self.cfg.ECG_DERIVED_FEATURES,
            "feature_aggregation_mode": getattr(self.cfg, "FEATURE_AGGREGATION_MODE", "full"),
            "include_first_features": getattr(self.cfg, "INCLUDE_FIRST_FEATURES", True),
            "use_medication_features": getattr(self.cfg, "USE_MEDICATION_FEATURES", False),
            "medication_group_patterns": getattr(self.cfg, "MEDICATION_GROUP_PATTERNS", {}),
        }
        source_paths = {
            "patients": self.cfg.DATA_DIR / "hosp" / "patients.csv",
            "admissions": self.cfg.DATA_DIR / "hosp" / "admissions.csv",
            "diagnoses_icd": self.cfg.DIAGNOSES_ICD_PATH,
            "target_macro": self.cfg.TARGET_MACRO_PATH,
            "chartevents": self.cfg.DATA_DIR / "icu" / "chartevents.csv",
            "labevents": self.cfg.DATA_DIR / "hosp" / "labevents.csv",
        }
        if getattr(self.cfg, "USE_DIAGNOSIS_TIME_CENSORING", False):
            source_paths["radiology_notes"] = self.cfg.RADIOLOGY_NOTES_PATH
            source_paths["radiology_detail"] = self.cfg.RADIOLOGY_DETAIL_PATH
        if getattr(self.cfg, "USE_ECG_FEATURES", False):
            source_paths["ecg_measurements"] = self.cfg.ECG_MEASUREMENTS_PATH
        if getattr(self.cfg, "USE_MEDICATION_FEATURES", False):
            source_paths["prescriptions"] = self.cfg.PRESCRIPTIONS_PATH
        return {
            "feature_config": self._normalize(feature_config),
            "sources": {
                name: self._file_signature(path)
                for name, path in source_paths.items()
            },
        }

    def _file_signature(self, path):
        path = Path(path)
        stat = path.stat()
        return {
            "path": str(path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    def _normalize(self, value):
        if isinstance(value, dict):
            return {
                str(key): self._normalize(value[key])
                for key in sorted(value, key=lambda item: str(item))
            }
        if isinstance(value, (set, list, tuple)):
            return [self._normalize(item) for item in sorted(value, key=lambda item: str(item))]
        if isinstance(value, Path):
            return str(value)
        return value
