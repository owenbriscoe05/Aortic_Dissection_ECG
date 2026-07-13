import argparse
import csv
import json
import os
import sqlite3
import sys
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from jsonschema import ValidationError, validate
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "mimic-iv"

ADMISSIONS_PATH = DATA_DIR / "hosp" / "admissions.csv"
ECG_MEASUREMENTS_PATH = DATA_DIR / "mimic-iv-ecg" / "1.0" / "machine_measurements.csv"
DISCHARGE_NOTES_PATH = DATA_DIR / "mimic-iv-note" / "2.2" / "note" / "discharge.csv"
DEFAULT_OUTPUT_BASE = PROJECT_ROOT / "data" / "processed" / "openai_discharge_note_parses"

DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_CHUNKSIZE = 250_000
DEFAULT_MAX_OUTPUT_TOKENS = 4500
DEFAULT_PARALLEL_WORKERS = 1

YES_NO_UNCLEAR = ["yes", "no", "unclear"]
CONFIDENCE = ["high", "medium", "low"]
THREAD_LOCAL = threading.local()


def yn_field(description):
    return {
        "type": "string",
        "enum": YES_NO_UNCLEAR,
        "description": description,
    }


# def confidence_field():
#     return {
#         "type": "string",
#         "enum": CONFIDENCE,
#         "description": "Confidence based only on explicit note evidence.",
#     }


def evidence_field():
    return {
        "type": "string",
        "description": "Brief quoted note evidence supporting this section, or empty string if none.",
    }


def obj(properties):
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


DISCHARGE_PHENOTYPE_SCHEMA = obj(
    {
        "chief_complaint_or_presenting_symptoms": obj(
            {
                "summary": {
                    "type": "string",
                    "description": "Concise presenting symptom summary/reason, or unclear.",
                },
                "evidence": evidence_field(),
                # "confidence": confidence_field(),
            }
        ),
        "presentation_phenotype": obj(
            {
                "chest_pain": yn_field("Any chest pain at presentation."),
                "back_pain": yn_field("Any back pain at presentation."),
                "abdominal_pain": yn_field("Any abdominal pain at presentation."),
                "pain_radiating_to_back": yn_field("Pain radiating to or from the back."),
                "migratory_pain": yn_field("Migratory pain."),
                "transfer_from_outside_hospital": yn_field("Transfer from an outside hospital or outside ED."),
                "evidence": evidence_field(),
                # "confidence": confidence_field(),
            }
        ),
        "aortic_syndrome_status": obj(
            {
                "confirmed_acute_aortic_dissection": yn_field("Aortic dissection first identified during this encounter."),
                "known_dissection_only": yn_field("Dissection WAS KNOWN PRIOR to this encounter."),
                "chronic_typeb_dissection": yn_field("Chronic Type B dissection."),
                "intramural_hematoma": yn_field("Intramural hematoma."),
                "penetrating_atherosclerotic_ulcer": yn_field("Penetrating atherosclerotic ulcer."),
                "aneurysm_without_dissection": yn_field("Aneurysm without dissection."),
                "rupture_or_contained_rupture": yn_field("Rupture or contained rupture."),
                "dissection_explicitly_ruled_out": yn_field("Aortic dissection explicitly ruled out."),
                "evidence": evidence_field(),
                # "confidence": confidence_field(),
            }
        ),
        "diagnosis_context": obj(
            {
                "diagnosed_before_arrival_at_outside_hospital": yn_field("Diagnosed before arrival or at an outside hospital."),
                "incidental_finding": yn_field("Incidental finding rather than symptom-driven diagnosis."),
                "suspected_because_of_symptoms": yn_field("Workup was driven by presenting symptoms."),
                "diagnostic_modality": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "CTA",
                            "CT chest/abdomen/pelvis",
                            "TEE",
                            "MRI/MRA",
                            "operative finding",
                            "outside-hospital imaging",
                            "other",
                            "unclear",
                        ],
                    },
                    "description": "All explicitly stated diagnostic modalities for the aorta condition.",
                },
                "how_diagnosis_was_made": {
                    "type": "string",
                    "description": "Brief description of diagnostic pathway, or unclear.",
                },
                "evidence": evidence_field(),
                # "confidence": confidence_field(),
            }
        ),
        "anatomic_detail": obj(
            {
                "stanford_type": {
                    "type": "string",
                    "enum": ["A", "B", "none", "unclear"],
                    "description": "Stanford dissection type, if written.",
                },
                "debakey_type": {
                    "type": "string",
                    "enum": ["I", "II", "III", "none", "unclear"],
                    "description": "DeBakey type, if written.",
                },
                "ascending_aorta": yn_field("Ascending aorta involvement."),
                "arch": yn_field("Aortic arch involvement."),
                "descending_thoracic": yn_field("Descending thoracic aorta involvement."),
                "abdominal": yn_field("Abdominal aorta involvement."),
                "iliac_extension": yn_field("Iliac extension."),
                "complicated_type_b_malperfusion": yn_field("Type B malperfusion."),
                "complicated_type_b_rupture": yn_field("Type B rupture."),
                "complicated_type_b_refractory_pain": yn_field("Type B refractory pain."),
                "complicated_type_b_uncontrolled_hypertension": yn_field("Type B uncontrolled hypertension."),
                "evidence": evidence_field(),
                # "confidence": confidence_field(),
            }
        ),
        "treatment": obj(
            {
                "open_surgical_repair": yn_field("Open surgical repair this admission."),
                "tevar_or_endovascular_repair": yn_field("TEVAR or other endovascular repair this admission."),
                "medical_management_only": yn_field("Medical management only."),
                "prior_repair_or_stent_graft_only_no_new_repair": yn_field("Prior repair/stent graft only, no new repair this admission."),
                "repaired_this_admission": yn_field("Any dissection/aortic repair this admission."),
                "evidence": evidence_field(),
                # "confidence": confidence_field(),
            }
        ),
        "control_usefulness": obj(
            {
                "primary_discharge_diagnosis_or_alternative_explanation": {
                    "type": "string",
                    "description": "Likely main diagnosis or alternative explanation for presentation.",
                },
                "plausible_aortic_dissection_mimic_presentation": yn_field("Are symptoms at presentation consistent with aortic dissection."),
                "evidence": evidence_field(),
                # "confidence": confidence_field(),
            }
        ),
        "overall": obj(
            {
                "target_cleanup_label": {
                    "type": "string",
                    "enum": [
                        "newly_identified_dissection",
                        "previously_known_dissection_only",
                        "aortic_syndrome_not_dissection",
                        "aneurysm_without_dissection",
                        "dissection_ruled_out",
                        "no_aortic_syndrome_evidence",
                        "unclear",
                    ],
                },
                "brief_summary": {
                    "type": "string",
                    "description": "One-sentence conservative summary of the aortic-dissection-relevant phenotype.",
                },
            }
        ),
    }
)


SYSTEM_INSTRUCTIONS = """
Extract clinical labels from MIMIC-IV discharge notes characterizing the encounter.

## Distinguishing between acute and prior dissections
A goal of this project is to identify patients having new dissections (dissections first identified
during a given encounter) versus patients with known dissections (dissections that were already known prior to the encounter).
Note that new dissections may be described as chronic or previously existing, but should be labeled as new if they were only identified
during this encounter and were not previously known.

Use only explicit note evidence. Prefer "unclear" when the note does not clearly support a label.
Do not infer aortic dissection from vague aortic disease, aneurysm, repair history, or rule-out language.
Return labels for the entire admission/visit. Multiple discharge-note rows may be combined in one input.
Evidence spans should be brief and copied from the note where possible.
"""


def env_value(*names):
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def normalize_base_url(url):
    url = url.strip().rstrip("/")
    if url.endswith("/responses"):
        return url[: -len("/responses")]
    return url


def load_openai_compatible_config(args):
    load_dotenv(PROJECT_ROOT / ".env")
    token = env_value("AZURE_OPENAI_API_KEY")
    if not token:
        raise RuntimeError("AZURE_OPENAI_API_KEY is required in .env or the environment.")

    raw_base_url = args.base_url or env_value("AZURE_OPENAI_BASE_URL")
    if not raw_base_url:
        raise RuntimeError("AZURE_OPENAI_BASE_URL is required in .env or the environment.")
    base_url = normalize_base_url(raw_base_url)
    if base_url.endswith("/openai"):
        base_url = f"{base_url}/v1"
    elif not base_url.endswith("/openai/v1"):
        base_url = f"{base_url}/openai/v1"
    return {
        "token": token,
        "provider": "azure",
        "base_url": base_url,
    }


def make_openai_compatible_client(config, args):
    return OpenAI(
        api_key=config["token"],
        base_url=config["base_url"],
        default_headers={"api-key": config["token"]},
        max_retries=args.num_retries,
        timeout=args.timeout_seconds,
    )


def get_thread_client(config, args):
    client_key = (
        config["base_url"],
        args.model,
        args.num_retries,
        args.timeout_seconds,
    )
    if getattr(THREAD_LOCAL, "client_key", None) != client_key:
        THREAD_LOCAL.client = make_openai_compatible_client(config, args)
        THREAD_LOCAL.client_key = client_key
    return THREAD_LOCAL.client


def load_openai_compatible_client(args):
    config = load_openai_compatible_config(args)
    client = make_openai_compatible_client(config, args)
    return client, config["provider"], config["base_url"]


def load_admissions(path):
    usecols = [
        "subject_id",
        "hadm_id",
        "admittime",
        "edregtime",
        "admission_type",
        "admission_location",
    ]
    df = pd.read_csv(path, usecols=usecols)
    df["subject_id"] = pd.to_numeric(df["subject_id"], errors="coerce").astype("Int64")
    df["hadm_id"] = pd.to_numeric(df["hadm_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["subject_id", "hadm_id"]).copy()
    df["subject_id"] = df["subject_id"].astype("int64")
    df["hadm_id"] = df["hadm_id"].astype("int64")
    df["admittime"] = pd.to_datetime(df["admittime"], errors="coerce")
    df["edregtime"] = pd.to_datetime(df["edregtime"], errors="coerce")

    valid_ed_time = df["edregtime"].notna() & df["admittime"].notna() & (df["edregtime"] <= df["admittime"])
    ed_after_admit = df["edregtime"].notna() & df["admittime"].notna() & (df["edregtime"] > df["admittime"])
    df["index_time"] = df["admittime"]
    df.loc[valid_ed_time, "index_time"] = df.loc[valid_ed_time, "edregtime"]
    df["index_time_source"] = "admittime"
    df.loc[valid_ed_time, "index_time_source"] = "edregtime"
    df.loc[ed_after_admit, "index_time_source"] = "admittime_edregtime_after_admit"
    df["feature_window_end"] = df["index_time"] + pd.to_timedelta(24, unit="h")
    return df.dropna(subset=["index_time"]).copy()


def find_visits_with_ecg_24h(admissions, ecg_path, chunksize):
    times = admissions[["subject_id", "hadm_id", "index_time", "feature_window_end"]].copy()
    subject_ids = set(times["subject_id"])
    counts = Counter()
    first_times = {}

    for chunk in pd.read_csv(ecg_path, usecols=["subject_id", "ecg_time"], chunksize=chunksize):
        chunk = chunk[chunk["subject_id"].isin(subject_ids)].copy()
        if chunk.empty:
            continue
        chunk["ecg_time"] = pd.to_datetime(chunk["ecg_time"], errors="coerce")
        chunk = chunk.dropna(subset=["subject_id", "ecg_time"])
        if chunk.empty:
            continue

        merged = chunk.merge(times, on="subject_id", how="inner")
        merged = merged[
            (merged["ecg_time"] >= merged["index_time"])
            & (merged["ecg_time"] <= merged["feature_window_end"])
        ]
        if merged.empty:
            continue

        counts.update({int(k): int(v) for k, v in merged.groupby("hadm_id").size().items()})
        for hadm_id, ecg_time in merged.groupby("hadm_id")["ecg_time"].min().items():
            hadm_id = int(hadm_id)
            if hadm_id not in first_times or ecg_time < first_times[hadm_id]:
                first_times[hadm_id] = ecg_time

    qualifying = admissions[admissions["hadm_id"].isin(counts)].copy()
    qualifying["ecg_count_24h"] = qualifying["hadm_id"].map(counts).astype("int64")
    qualifying["first_ecg_time_24h"] = qualifying["hadm_id"].map(first_times)
    return qualifying


def visit_lookup(visits):
    records = {}
    for row in visits.itertuples(index=False):
        records[int(row.hadm_id)] = {
            "subject_id": int(row.subject_id),
            "hadm_id": int(row.hadm_id),
            "index_time": str(row.index_time),
            "index_time_source": row.index_time_source,
            "feature_window_end": str(row.feature_window_end),
            "first_ecg_time_24h": str(row.first_ecg_time_24h),
            "ecg_count_24h": int(row.ecg_count_24h),
            "admission_type": row.admission_type,
            "admission_location": row.admission_location,
        }
    return records


def create_note_staging_db(discharge_path, qualifying_hadm_ids, chunksize):
    tmp = tempfile.NamedTemporaryFile(prefix="discharge_notes_", suffix=".sqlite", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()

    con = sqlite3.connect(tmp_path)
    con.execute(
        """
        CREATE TABLE notes (
            hadm_id INTEGER NOT NULL,
            subject_id INTEGER NOT NULL,
            note_id TEXT NOT NULL,
            note_type TEXT,
            note_seq TEXT,
            charttime TEXT,
            text TEXT NOT NULL
        )
        """
    )
    con.execute("CREATE INDEX idx_notes_hadm_id ON notes(hadm_id)")

    inserted = 0
    usecols = ["note_id", "subject_id", "hadm_id", "note_type", "note_seq", "charttime", "text"]
    for chunk in pd.read_csv(discharge_path, usecols=usecols, chunksize=chunksize):
        chunk = chunk.dropna(subset=["subject_id", "hadm_id", "text"]).copy()
        if chunk.empty:
            continue
        chunk["hadm_id"] = pd.to_numeric(chunk["hadm_id"], errors="coerce").astype("Int64")
        chunk["subject_id"] = pd.to_numeric(chunk["subject_id"], errors="coerce").astype("Int64")
        chunk = chunk.dropna(subset=["subject_id", "hadm_id"])
        chunk["hadm_id"] = chunk["hadm_id"].astype("int64")
        chunk["subject_id"] = chunk["subject_id"].astype("int64")
        chunk = chunk[chunk["hadm_id"].isin(qualifying_hadm_ids)].copy()
        if chunk.empty:
            continue
        chunk["text"] = chunk["text"].astype(str).str.strip()
        chunk = chunk[chunk["text"] != ""]
        if chunk.empty:
            continue

        rows = chunk[usecols].itertuples(index=False, name=None)
        con.executemany(
            """
            INSERT INTO notes
            (note_id, subject_id, hadm_id, note_type, note_seq, charttime, text)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        inserted += len(chunk)
        con.commit()

    return con, tmp_path, inserted


def fetch_visit_notes(con, hadm_id):
    rows = con.execute(
        """
        SELECT note_id, subject_id, hadm_id, note_type, note_seq, charttime, text
        FROM notes
        WHERE hadm_id = ?
        ORDER BY note_seq, charttime, note_id
        """,
        (int(hadm_id),),
    ).fetchall()
    notes = []
    for note_id, subject_id, hadm_id, note_type, note_seq, charttime, text in rows:
        notes.append(
            {
                "note_id": note_id,
                "subject_id": int(subject_id),
                "hadm_id": int(hadm_id),
                "note_type": note_type,
                "note_seq": note_seq,
                "charttime": charttime,
                "text": text,
            }
        )
    return notes


def combine_notes(notes):
    sections = []
    for i, note in enumerate(notes, start=1):
        if len(notes) > 1:
            sections.append(f"DISCHARGE NOTE {i}\n{note['text']}")
        else:
            sections.append(note["text"])
    return "\n\n".join(sections)


def build_user_input(visit, notes):
    return "Discharge note text:\n" + combine_notes(notes)


def extract_response_text(response):
    text = getattr(response, "output_text", None)
    if text:
        return text
    data = response.model_dump() if hasattr(response, "model_dump") else response
    parts = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if "text" in content:
                parts.append(content["text"])
    return "\n".join(parts)


def response_usage(response):
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    return {}


def flatten_value(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return "|".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return value


def flatten_json(value, prefix=""):
    flat = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}__{key}" if prefix else str(key)
            flat.update(flatten_json(child, child_prefix))
    else:
        flat[prefix] = flatten_value(value)
    return flat


def schema_leaf_columns(schema, prefix=""):
    if schema.get("type") == "object":
        columns = []
        for key, child in schema.get("properties", {}).items():
            child_prefix = f"{prefix}__{key}" if prefix else str(key)
            columns.extend(schema_leaf_columns(child, child_prefix))
        return columns
    return [prefix]


RESULT_BASE_COLUMNS = [
    "sequence_index",
    "subject_id",
    "hadm_id",
    "note_count",
    "note_ids",
    "note_types",
    "note_seqs",
    "note_charttimes",
    "index_time",
    "index_time_source",
    "feature_window_end",
    "first_ecg_time_24h",
    "ecg_count_24h",
    "admission_type",
    "admission_location",
    "usage_input_tokens",
    "usage_cached_tokens",
    "usage_output_tokens",
    "usage_reasoning_tokens",
    "usage_total_tokens",
]

ERROR_CSV_COLUMNS = [
    "sequence_index",
    "subject_id",
    "hadm_id",
    "note_count",
    "note_ids",
    "error",
]


def result_csv_columns():
    return RESULT_BASE_COLUMNS + schema_leaf_columns(DISCHARGE_PHENOTYPE_SCHEMA)


def note_meta(notes):
    return [
        {
            "note_id": note["note_id"],
            "note_type": note["note_type"],
            "note_seq": note["note_seq"],
            "charttime": note["charttime"],
        }
        for note in notes
    ]


def usage_fields(usage):
    input_details = usage.get("input_tokens_details") or {}
    output_details = usage.get("output_tokens_details") or {}
    return {
        "usage_input_tokens": usage.get("input_tokens", ""),
        "usage_cached_tokens": input_details.get("cached_tokens", ""),
        "usage_output_tokens": usage.get("output_tokens", ""),
        "usage_reasoning_tokens": output_details.get("reasoning_tokens", ""),
        "usage_total_tokens": usage.get("total_tokens", ""),
    }


def make_result_record(work_item, parsed_json, response):
    usage = response_usage(response)
    return {
        "sequence_index": work_item["sequence_index"],
        "subject_id": work_item["visit"]["subject_id"],
        "hadm_id": work_item["hadm_id"],
        "note_count": len(work_item["notes"]),
        "notes": work_item["note_meta"],
        "visit": work_item["visit"],
        "llm_parse": parsed_json,
        "usage": usage,
    }


def result_record_to_csv_row(record):
    notes = record["notes"]
    visit = record["visit"]
    row = {
        "sequence_index": record["sequence_index"],
        "subject_id": record["subject_id"],
        "hadm_id": record["hadm_id"],
        "note_count": record["note_count"],
        "note_ids": "|".join(str(note["note_id"]) for note in notes),
        "note_types": "|".join(str(note["note_type"]) for note in notes),
        "note_seqs": "|".join(str(note["note_seq"]) for note in notes),
        "note_charttimes": "|".join(str(note["charttime"]) for note in notes),
        "index_time": visit.get("index_time", ""),
        "index_time_source": visit.get("index_time_source", ""),
        "feature_window_end": visit.get("feature_window_end", ""),
        "first_ecg_time_24h": visit.get("first_ecg_time_24h", ""),
        "ecg_count_24h": visit.get("ecg_count_24h", ""),
        "admission_type": visit.get("admission_type", ""),
        "admission_location": visit.get("admission_location", ""),
    }
    row.update(usage_fields(record.get("usage") or {}))
    row.update(flatten_json(record["llm_parse"]))
    return {column: row.get(column, "") for column in result_csv_columns()}


def make_error_record(work_item, exc):
    return {
        "sequence_index": work_item["sequence_index"],
        "subject_id": work_item["visit"]["subject_id"],
        "hadm_id": work_item["hadm_id"],
        "note_count": len(work_item["notes"]),
        "notes": work_item["note_meta"],
        "error": str(exc),
    }


def error_record_to_csv_row(record):
    return {
        "sequence_index": record["sequence_index"],
        "subject_id": record["subject_id"],
        "hadm_id": record["hadm_id"],
        "note_count": record["note_count"],
        "note_ids": "|".join(str(note["note_id"]) for note in record["notes"]),
        "error": record["error"],
    }


def make_work_item(sequence_index, hadm_id, visit, notes):
    return {
        "sequence_index": sequence_index,
        "hadm_id": hadm_id,
        "visit": visit,
        "notes": notes,
        "note_meta": note_meta(notes),
    }


def parse_discharge_visit(client, args, visit, notes):
    response = client.responses.create(
        model=args.model,
        input=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": build_user_input(visit, notes)},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "discharge_aortic_phenotype",
                "strict": True,
                "schema": DISCHARGE_PHENOTYPE_SCHEMA,
            }
        },
        reasoning={"effort": args.reasoning_effort},
        max_output_tokens=args.max_output_tokens,
        store=False,
    )
    raw_text = extract_response_text(response)
    parsed = json.loads(raw_text)
    try:
        validate(instance=parsed, schema=DISCHARGE_PHENOTYPE_SCHEMA)
    except ValidationError as exc:
        path = ".".join(str(part) for part in exc.absolute_path) or "<root>"
        raise ValueError(f"LLM output failed JSON Schema validation at {path}: {exc.message}") from exc
    return parsed, response


def parse_work_item(config, args, work_item):
    client = get_thread_client(config, args)
    parsed_json, response = parse_discharge_visit(
        client,
        args,
        work_item["visit"],
        work_item["notes"],
    )
    return make_result_record(work_item, parsed_json, response)


def write_manifest(run_dir, args, provider, base_url):
    manifest = {
        "run_id": run_dir.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "implementation": "openai-python Responses API against Azure OpenAI-compatible endpoint",
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "provider": provider,
        "base_url": base_url,
        "store": False,
        "cache_policy": {
            "pipeline_cache": False,
            "prompt_cache_key": None,
            "prompt_cache_retention": None,
            "stored_responses": False,
        },
        "cost_guard": {
            "run_requires_flag": "--run",
            "full_run_requires_flag": "--allow-full-run",
            "max_visits": args.max_visits,
            "hadm_ids": args.hadm_ids,
        },
        "parallelism": {
            "parallel_workers": args.parallel_workers,
            "max_in_flight": args.max_in_flight,
        },
        "outputs": {
            "jsonl_results": "openai_discharge_visit_parse_results.jsonl",
            "csv_results": "openai_discharge_visit_parse_results.csv",
            "jsonl_errors": "openai_discharge_visit_parse_errors.jsonl",
            "csv_errors": "openai_discharge_visit_parse_errors.csv",
        },
        "qualification": (
            "Admissions with at least one ECG machine-measurement timestamp from index time through 24 hours "
            "and at least one attached discharge-note row. Index time is edregtime when present and not after "
            "admittime; otherwise admittime. All discharge-note rows for a qualifying admission are combined "
            "into one visit-level input."
        ),
        "schema": DISCHARGE_PHENOTYPE_SCHEMA,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run_parser(args):
    config = load_openai_compatible_config(args)
    provider = config["provider"]
    base_url = config["base_url"]
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_base) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    write_manifest(run_dir, args, provider, base_url)

    visits = find_visits_with_ecg_24h(
        load_admissions(Path(args.admissions_path)),
        Path(args.ecg_measurements_path),
        args.chunksize,
    )
    visits.to_csv(run_dir / "qualifying_ecg_24h_visits.csv", index=False)
    visits_by_hadm = visit_lookup(visits)
    qualifying_hadm_ids = set(visits_by_hadm)
    if args.hadm_ids:
        requested_hadm_ids = {int(hadm_id) for hadm_id in args.hadm_ids}
        missing_hadm_ids = sorted(requested_hadm_ids - qualifying_hadm_ids)
        if missing_hadm_ids:
            raise RuntimeError(
                "Requested hadm_ids are not in the ECG-24h qualifying visit set: "
                + ", ".join(str(hadm_id) for hadm_id in missing_hadm_ids)
            )
        qualifying_hadm_ids = requested_hadm_ids

    con = None
    staging_path = None
    try:
        con, staging_path, staged_note_rows = create_note_staging_db(
            Path(args.discharge_notes_path),
            qualifying_hadm_ids,
            args.chunksize,
        )

        output_path = run_dir / "openai_discharge_visit_parse_results.jsonl"
        error_path = run_dir / "openai_discharge_visit_parse_errors.jsonl"
        output_csv_path = run_dir / "openai_discharge_visit_parse_results.csv"
        error_csv_path = run_dir / "openai_discharge_visit_parse_errors.csv"
        parsed = 0
        errors = 0
        seen = 0

        def iter_work_items():
            sequence_index = 0
            for hadm_id in sorted(qualifying_hadm_ids):
                if args.max_visits is not None and sequence_index >= args.max_visits:
                    break
                notes = fetch_visit_notes(con, hadm_id)
                if not notes:
                    continue
                sequence_index += 1
                yield make_work_item(
                    sequence_index,
                    hadm_id,
                    visits_by_hadm[hadm_id],
                    notes,
                )

        def handle_future(future, out, err, out_csv, err_csv, result_csv_writer, error_csv_writer):
            nonlocal parsed, errors
            work_item = future_to_work_item.pop(future)
            try:
                record = future.result()
                out.write(json.dumps(record, default=str) + "\n")
                out.flush()
                result_csv_writer.writerow(result_record_to_csv_row(record))
                out_csv.flush()
                parsed += 1
            except Exception as exc:
                errors += 1
                error_record = make_error_record(work_item, exc)
                err.write(json.dumps(error_record, default=str) + "\n")
                err.flush()
                error_csv_writer.writerow(error_record_to_csv_row(error_record))
                err_csv.flush()
                if args.fail_fast:
                    raise

        max_workers = max(1, int(args.parallel_workers))
        max_in_flight = int(args.max_in_flight or max_workers * 2)
        max_in_flight = max(max_workers, max_in_flight)

        with (
            output_path.open("w", encoding="utf-8") as out,
            error_path.open("w", encoding="utf-8") as err,
            output_csv_path.open("w", encoding="utf-8", newline="") as out_csv,
            error_csv_path.open("w", encoding="utf-8", newline="") as err_csv,
            ThreadPoolExecutor(max_workers=max_workers) as executor,
        ):
            result_csv_writer = csv.DictWriter(out_csv, fieldnames=result_csv_columns())
            error_csv_writer = csv.DictWriter(err_csv, fieldnames=ERROR_CSV_COLUMNS)
            result_csv_writer.writeheader()
            error_csv_writer.writeheader()
            future_to_work_item = {}

            for work_item in iter_work_items():
                while len(future_to_work_item) >= max_in_flight:
                    done, _ = wait(future_to_work_item, return_when=FIRST_COMPLETED)
                    for future in done:
                        handle_future(future, out, err, out_csv, err_csv, result_csv_writer, error_csv_writer)

                future = executor.submit(parse_work_item, config, args, work_item)
                future_to_work_item[future] = work_item
                seen += 1

                if args.progress_every and seen % args.progress_every == 0:
                    print(
                        f"submitted_visits={seen:,} parsed={parsed:,} errors={errors:,} "
                        f"in_flight={len(future_to_work_item):,}",
                        flush=True,
                    )
                if args.sleep_seconds:
                    time.sleep(args.sleep_seconds)

            while future_to_work_item:
                done, _ = wait(future_to_work_item, return_when=FIRST_COMPLETED)
                for future in done:
                    handle_future(future, out, err, out_csv, err_csv, result_csv_writer, error_csv_writer)
                    completed = parsed + errors
                    if args.progress_every and completed % args.progress_every == 0:
                        print(
                            f"completed_visits={completed:,} parsed={parsed:,} errors={errors:,} "
                            f"in_flight={len(future_to_work_item):,}",
                            flush=True,
                        )

        summary = {
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "qualifying_visits_with_ecg_24h": int(len(visits)),
            "staged_discharge_note_rows": int(staged_note_rows),
            "qualifying_visits_with_discharge_notes_seen": int(seen),
            "parsed": int(parsed),
            "errors": int(errors),
            "output_path": str(output_path),
            "output_csv_path": str(output_csv_path),
            "errors_path": str(error_path),
            "errors_csv_path": str(error_csv_path),
            "parallel_workers": int(max_workers),
            "max_in_flight": int(max_in_flight),
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
    finally:
        if con is not None:
            con.close()
        if staging_path is not None and staging_path.exists() and not args.keep_staging_db:
            staging_path.unlink()


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Parse discharge notes for ECG-in-first-24-hours admissions using the OpenAI Python SDK "
            "against Azure's OpenAI-compatible endpoint. Paid calls require --run."
        )
    )
    parser.add_argument("--run", action="store_true", help="Actually call the LLM.")
    parser.add_argument("--allow-full-run", action="store_true", help="Allow an unbounded paid run.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--reasoning-effort", default="medium", choices=["none", "low", "medium", "high", "xhigh"])
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--num-retries", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--parallel-workers", type=int, default=DEFAULT_PARALLEL_WORKERS)
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=None,
        help="Maximum submitted-but-unfinished parse tasks. Defaults to 2 * parallel workers.",
    )
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE)
    parser.add_argument("--max-visits", type=int, default=None)
    parser.add_argument("--hadm-ids", nargs="+", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--keep-staging-db", action="store_true")
    parser.add_argument("--admissions-path", default=str(ADMISSIONS_PATH))
    parser.add_argument("--ecg-measurements-path", default=str(ECG_MEASUREMENTS_PATH))
    parser.add_argument("--discharge-notes-path", default=str(DISCHARGE_NOTES_PATH))
    parser.add_argument("--output-base", default=str(DEFAULT_OUTPUT_BASE))
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if not args.run:
        print("Refusing to call the LLM without --run. Use --max-visits N for bounded paid tests.")
        return 0
    if args.max_visits is None and not args.allow_full_run:
        print(
            "Refusing full paid run. Pass --max-visits N for a bounded run or --allow-full-run.",
            file=sys.stderr,
        )
        return 2
    run_parser(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
