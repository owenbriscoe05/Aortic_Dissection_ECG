#!/usr/bin/env python3
"""Generate poster-ready methods diagrams for the ECG/aortic dissection project."""

from __future__ import annotations

import html
import os
import shutil
import subprocess
import textwrap
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent

PALETTE = {
    "ink": "#1F2933",
    "muted": "#596B7A",
    "line": "#B7C3CF",
    "bg": "#FFFFFF",
    "source": "#EEF6F5",
    "source_stroke": "#2C7A7B",
    "process": "#F3F6FA",
    "process_stroke": "#3D5A80",
    "model": "#FFF6E6",
    "model_stroke": "#B36B00",
    "guard": "#FCEEEE",
    "guard_stroke": "#A43E3E",
    "output": "#EEF8EE",
    "output_stroke": "#3A7D44",
    "note": "#F8F4FF",
    "note_stroke": "#6B4E9B",
}


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def wrap_lines(text: str, width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph:
            lines.append("")
            continue
        lines.extend(textwrap.wrap(paragraph, width=width, break_long_words=False))
    return lines


def fmt_int(value: int) -> str:
    return f"{value:,}"


def fmt_pct(numerator: int, denominator: int, digits: int = 1) -> str:
    if denominator == 0:
        return "NA"
    return f"{100 * numerator / denominator:.{digits}f}%"


class Svg:
    def __init__(self, width: int = 1600, height: int = 900) -> None:
        self.width = width
        self.height = height
        self.arrow_color = PALETTE["muted"]
        self.parts: list[str] = []

    def add(self, raw: str) -> None:
        self.parts.append(raw)

    def text(
        self,
        x: float,
        y: float,
        text: str,
        size: int = 24,
        weight: int = 400,
        fill: str | None = None,
        anchor: str = "start",
    ) -> None:
        fill = fill or PALETTE["ink"]
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{esc(text)}</text>'
        )

    def multiline_text(
        self,
        x: float,
        y: float,
        lines: list[str],
        size: int = 20,
        fill: str | None = None,
        anchor: str = "start",
        line_height: int | None = None,
    ) -> None:
        fill = fill or PALETTE["ink"]
        line_height = line_height or int(size * 1.25)
        for i, line in enumerate(lines):
            self.add(
                f'<text x="{x:.1f}" y="{y + i * line_height:.1f}" font-size="{size}" '
                f'fill="{fill}" text-anchor="{anchor}">{esc(line)}</text>'
            )

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str,
        stroke: str,
        radius: int = 8,
        stroke_width: float = 2,
    ) -> None:
        self.add(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="{radius}" ry="{radius}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{stroke_width}"/>'
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        stroke: str | None = None,
        width: float = 2.5,
        dash: str | None = None,
        arrow: bool = False,
    ) -> None:
        stroke = stroke or PALETTE["muted"]
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        marker = ' marker-end="url(#arrow)"' if arrow else ""
        self.add(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{width}" stroke-linecap="round"{dash_attr}{marker}/>'
        )

    def path(
        self,
        d: str,
        stroke: str | None = None,
        width: float = 2.5,
        fill: str = "none",
        dash: str | None = None,
        arrow: bool = False,
    ) -> None:
        stroke = stroke or PALETTE["muted"]
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        marker = ' marker-end="url(#arrow)"' if arrow else ""
        self.add(
            f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{width}" '
            f'stroke-linecap="round" stroke-linejoin="round"{dash_attr}{marker}/>'
        )

    def circle(self, cx: float, cy: float, r: float, fill: str, stroke: str) -> None:
        self.add(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="2"/>'
        )

    def box(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        body: str,
        fill: str,
        stroke: str,
        title_size: int = 23,
        body_size: int = 18,
        wrap: int = 28,
    ) -> None:
        self.rect(x, y, w, h, fill, stroke)
        self.text(x + 22, y + 36, title, size=title_size, weight=700)
        lines = wrap_lines(body, wrap)
        self.multiline_text(x + 22, y + 68, lines, size=body_size, fill=PALETTE["muted"])

    def count_box(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        number: str,
        label: str,
        fill: str,
        stroke: str,
        sublabel: str | None = None,
        number_size: int = 42,
        label_size: int = 21,
    ) -> None:
        self.rect(x, y, w, h, fill, stroke)
        self.text(x + w / 2, y + 48, number, size=number_size, weight=800, anchor="middle")
        self.multiline_text(
            x + w / 2,
            y + 82,
            wrap_lines(label, 24),
            size=label_size,
            fill=PALETTE["ink"],
            anchor="middle",
            line_height=int(label_size * 1.2),
        )
        if sublabel:
            self.multiline_text(
                x + w / 2,
                y + h - 30,
                wrap_lines(sublabel, 29),
                size=17,
                fill=PALETTE["muted"],
                anchor="middle",
                line_height=20,
            )

    def mini_count(
        self,
        x: float,
        y: float,
        number: str,
        label: str,
        fill: str,
        stroke: str,
        w: float = 215,
        h: float = 96,
    ) -> None:
        self.rect(x, y, w, h, fill, stroke)
        self.text(x + w / 2, y + 42, number, size=29, weight=800, anchor="middle")
        self.multiline_text(
            x + w / 2,
            y + 68,
            wrap_lines(label, 22),
            size=16,
            fill=PALETTE["muted"],
            anchor="middle",
            line_height=18,
        )

    def arrow_label(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        label: str | None = None,
        stroke: str | None = None,
    ) -> None:
        self.line(x1, y1, x2, y2, stroke=stroke or PALETTE["muted"], arrow=True)
        if label:
            self.text((x1 + x2) / 2, (y1 + y2) / 2 - 14, label, size=17, fill=PALETTE["muted"], anchor="middle")

    def title(self, title: str, subtitle: str | None = None) -> None:
        self.text(70, 70, title, size=36, weight=800)
        if subtitle:
            self.multiline_text(70, 107, wrap_lines(subtitle, 105), size=18, fill=PALETTE["muted"])

    def render(self) -> str:
        defs = f"""
  <defs>
    <marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6"
            orient="auto" markerUnits="strokeWidth">
      <path d="M2,2 L10,6 L2,10 Z" fill="{self.arrow_color}"/>
    </marker>
    <style>
      text {{
        font-family: Arial, Helvetica, sans-serif;
        dominant-baseline: alphabetic;
      }}
    </style>
  </defs>
"""
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}" role="img">\n'
            f'<rect width="100%" height="100%" fill="{PALETTE["bg"]}"/>\n'
            f"{defs}\n"
            + "\n".join(self.parts)
            + "\n</svg>\n"
        )

    def save(self, filename: str) -> Path:
        path = OUT_DIR / filename
        path.write_text(self.render(), encoding="utf-8")
        return path


def write_png(svg_path: Path) -> None:
    converter = shutil.which("rsvg-convert")
    if converter is None:
        return
    png_path = svg_path.with_suffix(".png")
    env = os.environ.copy()
    env.setdefault("XDG_CACHE_HOME", "/tmp")
    env.setdefault("FONTCONFIG_PATH", "/etc/fonts")
    subprocess.run(
        [converter, "-w", "3200", str(svg_path), "-o", str(png_path)],
        check=True,
        env=env,
    )


def diagram_overall_workflow() -> Path:
    s = Svg()
    s.title(
        "End-to-End Study Workflow",
        "MIMIC-IV admissions, ECG machine measurements, and discharge notes feed two linked workflows: early-window tabular prediction and LLM-assisted phenotype refinement.",
    )

    for x, y, title, body in [
        (70, 190, "Hospital tables", "admissions, patients, diagnoses_icd"),
        (70, 340, "ECG table", "machine_measurements with ECG timestamps"),
        (70, 490, "Note tables", "discharge notes; radiology optional for censoring"),
    ]:
        s.box(x, y, 315, 105, title, body, PALETTE["source"], PALETTE["source_stroke"], wrap=30)

    s.box(
        500,
        185,
        360,
        150,
        "Cohort spine",
        "Adult admissions; exact ICD targets; index time from valid ED registration, otherwise admission time.",
        PALETTE["process"],
        PALETTE["process_stroke"],
        wrap=36,
    )
    s.box(
        500,
        375,
        360,
        135,
        "24-hour feature window",
        "Window starts at index time. Active default keeps demographics and first in-window ECG machine measurements.",
        PALETTE["process"],
        PALETTE["process_stroke"],
        wrap=36,
    )
    s.box(
        500,
        565,
        360,
        150,
        "Discharge-note parser",
        "Admissions with in-window ECG and non-empty discharge notes are parsed with a strict phenotype schema.",
        PALETTE["note"],
        PALETTE["note_stroke"],
        wrap=36,
    )

    s.box(
        980,
        250,
        330,
        145,
        "ECG-complete model",
        "XGBoost with native categorical handling; class imbalance handled by scale_pos_weight.",
        PALETTE["model"],
        PALETTE["model_stroke"],
        wrap=33,
    )
    s.box(
        980,
        485,
        330,
        145,
        "LLM-derived cohorts",
        "Target cleanup and control characterization can define encounter-level cohorts for model variants.",
        PALETTE["output"],
        PALETTE["output_stroke"],
        wrap=33,
    )

    s.box(
        1350,
        300,
        180,
        260,
        "Outputs",
        "HTML report\nPR/ROC curves\nthreshold table\nSHAP summaries\nrun metadata\nJSONL/CSV parse files",
        "#F7F9FC",
        PALETTE["line"],
        wrap=17,
        body_size=17,
    )

    s.line(385, 242, 500, 242, arrow=True)
    s.line(385, 392, 500, 443, arrow=True)
    s.line(385, 542, 500, 633, arrow=True)
    s.line(680, 320, 680, 375, arrow=True)
    s.line(860, 443, 980, 322, arrow=True)
    s.line(860, 633, 980, 558, arrow=True)
    s.line(1310, 322, 1350, 395, arrow=True)
    s.line(1310, 558, 1350, 465, arrow=True)
    s.path("M860,443 C910,443 925,558 980,558", arrow=True)
    return s.save("01_end_to_end_workflow.svg")


def diagram_cohort_window() -> Path:
    s = Svg()
    s.title(
        "Cohort Definition and Early Feature Window",
        "The active model uses a clinically anchored index time and an ECG-complete 24-hour measurement window.",
    )

    s.box(
        70,
        180,
        330,
        150,
        "Target admissions",
        "Exact ICD-9/ICD-10 aortic dissection codes after decimal normalization. No broad prefix matching.",
        PALETTE["guard"],
        PALETTE["guard_stroke"],
        wrap=33,
    )
    s.box(
        70,
        390,
        330,
        150,
        "Control admissions",
        "Exclude exact dissection codes. Standard run can restrict controls to clinically similar chest-pain diagnoses.",
        PALETTE["source"],
        PALETTE["source_stroke"],
        wrap=33,
    )
    s.box(
        70,
        600,
        330,
        145,
        "Eligibility filters",
        "Age at least 18 years. Require at least one in-window ECG row for the active default model.",
        PALETTE["process"],
        PALETTE["process_stroke"],
        wrap=34,
    )

    s.box(
        505,
        235,
        340,
        155,
        "Index time rule",
        "edregtime if present and not after admittime; otherwise admittime. Store index_time_source.",
        PALETTE["process"],
        PALETTE["process_stroke"],
        wrap=34,
    )
    s.box(
        505,
        500,
        340,
        145,
        "Admission-level unit",
        "LLM-derived cohorts preserve hadm_id; one admission remains one model row.",
        PALETTE["note"],
        PALETTE["note_stroke"],
        wrap=34,
    )

    s.text(1000, 225, "Feature timeline", size=30, weight=800)
    s.line(970, 430, 1450, 430, stroke=PALETTE["ink"], width=3)
    s.circle(1010, 430, 12, PALETTE["process"], PALETTE["process_stroke"])
    s.circle(1405, 430, 12, PALETTE["output"], PALETTE["output_stroke"])
    s.text(1010, 390, "Index time", size=22, weight=700, anchor="middle")
    s.text(1405, 390, "+24 hours", size=22, weight=700, anchor="middle")
    s.text(1210, 470, "Default feature window", size=22, weight=700, anchor="middle")
    s.path("M1010,500 C1120,570 1290,570 1405,500", stroke=PALETTE["process_stroke"], width=3)
    s.text(1210, 605, "First in-window ECG machine measurements", size=20, anchor="middle")
    s.text(1210, 635, "Demographics measured from admission/patient tables", size=18, fill=PALETTE["muted"], anchor="middle")
    s.line(1265, 430, 1265, 350, stroke=PALETTE["guard_stroke"], dash="7 7")
    s.text(1265, 330, "Optional diagnosis-time censoring", size=18, fill=PALETTE["guard_stroke"], anchor="middle")
    s.text(1265, 355, "radiology charttime only", size=16, fill=PALETTE["guard_stroke"], anchor="middle")

    s.line(400, 255, 505, 303, arrow=True)
    s.line(400, 465, 505, 303, arrow=True)
    s.line(400, 660, 505, 570, arrow=True)
    s.line(845, 303, 970, 430, arrow=True)
    s.line(845, 570, 970, 500, arrow=True)
    return s.save("02_cohort_and_feature_window.svg")


def diagram_predictor_set() -> Path:
    s = Svg()
    s.title(
        "Active Predictor Set",
        "Default ECG-complete tabular modeling uses demographics plus the first ECG machine-measurement values from the early window.",
    )

    s.box(
        90,
        190,
        390,
        250,
        "Included demographics",
        "index_age\ngender\nrace\ninsurance\nmarital_status",
        PALETTE["source"],
        PALETTE["source_stroke"],
        wrap=34,
        body_size=20,
    )
    s.box(
        600,
        190,
        420,
        250,
        "Included ECG features",
        "First in-window machine measurements: RR interval, wave onsets/ends, axes. Derived: QT interval and Bazett QTc.",
        PALETTE["process"],
        PALETTE["process_stroke"],
        wrap=38,
    )
    s.box(
        1135,
        190,
        360,
        250,
        "Default exclusions",
        "Labs, vitals, medications, encounter urgency, and ECG count are comparison-only unless explicitly requested.",
        PALETTE["guard"],
        PALETTE["guard_stroke"],
        wrap=34,
    )

    s.box(
        365,
        565,
        450,
        145,
        "Feature matrix",
        "ECG-complete admission rows with categorical predictors kept as native categorical variables for XGBoost.",
        PALETTE["model"],
        PALETTE["model_stroke"],
        wrap=44,
    )
    s.box(
        970,
        565,
        360,
        145,
        "Reporting discipline",
        "Missingness thresholds and protected race feature are documented in run metadata and caveats.",
        PALETTE["output"],
        PALETTE["output_stroke"],
        wrap=34,
    )
    s.line(480, 315, 600, 315, arrow=True)
    s.path("M285,440 C285,525 450,530 505,565", arrow=True)
    s.path("M810,440 C810,520 675,520 635,565", arrow=True)
    s.path("M1315,440 C1315,520 1240,520 1175,565", arrow=True, dash="8 8")
    s.line(815, 638, 970, 638, arrow=True)
    return s.save("03_active_predictor_set.svg")


def diagram_llm_parser() -> Path:
    s = Svg()
    s.title(
        "Discharge-Note Phenotyping Parser",
        "The parser converts visit-level discharge notes into auditable phenotype labels without sending visit metadata to the model input.",
    )

    s.box(
        70,
        180,
        340,
        145,
        "Qualifying admissions",
        "At least one ECG machine-measurement row in first 24 hours and at least one attached non-empty discharge note.",
        PALETTE["source"],
        PALETTE["source_stroke"],
        wrap=34,
    )
    s.box(
        70,
        420,
        340,
        145,
        "Visit-level note text",
        "All discharge-note rows for a qualifying admission are combined. Model input contains discharge-note text only.",
        PALETTE["note"],
        PALETTE["note_stroke"],
        wrap=34,
    )
    s.box(
        525,
        300,
        360,
        160,
        "Azure OpenAI Responses API",
        "OpenAI Python SDK against Azure-compatible endpoint; model default gpt-5.6-luna; store=False.",
        PALETTE["model"],
        PALETTE["model_stroke"],
        wrap=36,
    )
    s.box(
        1000,
        170,
        360,
        150,
        "Strict JSON Schema",
        "Presenting symptoms, aortic syndrome status, diagnosis context, anatomy, treatment, control usefulness, overall label.",
        PALETTE["process"],
        PALETTE["process_stroke"],
        wrap=36,
    )
    s.box(
        1000,
        415,
        360,
        150,
        "Local validation",
        "Parse JSON, validate against schema, stream successful records and errors incrementally.",
        PALETTE["guard"],
        PALETTE["guard_stroke"],
        wrap=36,
    )
    s.box(
        525,
        610,
        360,
        145,
        "Audit outputs",
        "JSONL and CSV results, JSONL and CSV errors, manifest, qualifying-visit file, and summary.",
        PALETTE["output"],
        PALETTE["output_stroke"],
        wrap=36,
    )

    s.line(240, 325, 240, 420, arrow=True)
    s.line(410, 492, 525, 380, arrow=True)
    s.line(885, 380, 1000, 245, arrow=True)
    s.line(1180, 320, 1180, 415, arrow=True)
    s.path("M1000,490 C930,560 870,600 780,610", arrow=True)
    s.path("M525,665 C425,665 365,570 310,565", dash="8 8")
    s.text(300, 618, "Metadata retained only in output records", size=18, fill=PALETTE["muted"], anchor="middle")
    return s.save("04_discharge_note_llm_parser.svg")


def diagram_model_training() -> Path:
    s = Svg()
    s.title(
        "Model Development and Evaluation",
        "Training choices are fit on the training data and validation split; the holdout remains reserved for final evaluation.",
    )

    stages = [
        (
            70,
            200,
            "Raw feature matrix",
            "Adult cohort rows with demographics, window bounds, labels, and first ECG features.",
            PALETTE["source"],
            PALETTE["source_stroke"],
        ),
        (
            405,
            200,
            "Preprocessing",
            "ECG-complete filter, row/column missingness thresholds, leakage drops, native categorical dtypes.",
            PALETTE["process"],
            PALETTE["process_stroke"],
        ),
        (
            740,
            200,
            "Train/validation",
            "Stratified split within training data; tune XGBoost parameter candidates by validation average precision.",
            PALETTE["model"],
            PALETTE["model_stroke"],
        ),
        (
            1075,
            200,
            "Final model",
            "Refit selected XGBoost model using scale_pos_weight and selected features.",
            PALETTE["model"],
            PALETTE["model_stroke"],
        ),
    ]
    for x, y, title, body, fill, stroke in stages:
        s.box(x, y, 295, 180, title, body, fill, stroke, wrap=28)

    s.box(
        1075,
        505,
        295,
        185,
        "Holdout evaluation",
        "Average precision, PR-AUC, ROC-AUC, F1, threshold operating points, feature importance, SHAP, run metadata.",
        PALETTE["output"],
        PALETTE["output_stroke"],
        wrap=28,
    )
    s.box(
        405,
        505,
        500,
        185,
        "Feature selection safeguards",
        "Recursive elimination, when enabled, uses validation data rather than holdout. Race is protected from automatic pruning.",
        PALETTE["guard"],
        PALETTE["guard_stroke"],
        wrap=50,
    )

    s.line(365, 290, 405, 290, arrow=True)
    s.line(700, 290, 740, 290, arrow=True)
    s.line(1035, 290, 1075, 290, arrow=True)
    s.line(1222, 380, 1222, 505, arrow=True)
    s.path("M885,380 C930,455 1000,500 1075,560", arrow=True)
    s.text(190, 480, "Natural-prevalence holdout is assigned before preprocessing", size=20, fill=PALETTE["muted"])
    s.path("M205,385 C205,450 1050,450 1120,505", stroke=PALETTE["muted"], dash="8 8", arrow=True)
    return s.save("05_model_training_and_evaluation.svg")


def diagram_numeric_icd_cohort() -> Path:
    s = Svg()
    s.title(
        "ICD-Based ECG-Complete Cohort",
        "Standard demographics + first-ECG all-controls run; counts are admission-row counts after adult cohort construction unless noted.",
    )

    adult_rows = 223_452
    adult_targets = 622
    adult_controls = 222_830
    ecg_rows = 70_094
    ecg_targets = 355
    ecg_controls = 69_739
    train_rows = 56_116
    train_targets = 285
    holdout_rows = 13_978
    holdout_targets = 70

    s.count_box(
        80,
        215,
        300,
        160,
        fmt_int(879),
        "exact ICD target admissions",
        PALETTE["guard"],
        PALETTE["guard_stroke"],
        "before subject/adult filters",
        number_size=46,
    )
    s.count_box(
        515,
        205,
        360,
        180,
        fmt_int(adult_rows),
        "adult cohort rows",
        PALETTE["process"],
        PALETTE["process_stroke"],
        f"{fmt_int(adult_targets)} targets | {fmt_int(adult_controls)} controls",
        number_size=50,
    )
    s.count_box(
        1015,
        205,
        360,
        180,
        fmt_int(ecg_rows),
        "ECG-complete rows",
        PALETTE["output"],
        PALETTE["output_stroke"],
        f"{fmt_pct(ecg_rows, adult_rows)} retained | {fmt_int(ecg_targets)} targets",
        number_size=50,
    )
    s.arrow_label(380, 295, 515, 295, "target/control spine")
    s.arrow_label(875, 295, 1015, 295, "24h ECG filter")

    s.mini_count(
        505,
        520,
        fmt_int(train_rows),
        f"train rows\n{fmt_int(train_targets)} targets",
        PALETTE["model"],
        PALETTE["model_stroke"],
        w=250,
        h=120,
    )
    s.mini_count(
        800,
        520,
        fmt_int(holdout_rows),
        f"holdout rows\n{fmt_int(holdout_targets)} targets",
        PALETTE["model"],
        PALETTE["model_stroke"],
        w=250,
        h=120,
    )
    s.mini_count(
        1095,
        520,
        "16",
        "active features\n5 demo + 11 ECG",
        PALETTE["source"],
        PALETTE["source_stroke"],
        w=250,
        h=120,
    )
    s.path("M1195,385 C1195,455 630,455 630,520", arrow=True)
    s.path("M1195,385 C1195,455 925,455 925,520", arrow=True)
    s.path("M1195,385 C1195,455 1220,455 1220,520", arrow=True)

    s.text(1015, 430, f"ECG-complete target prevalence: {fmt_pct(ecg_targets, ecg_rows, 2)}", size=22, weight=700)
    s.text(80, 810, "Source: demographics_ecg_runs/20260708_151834 run log and manifest.", size=18, fill=PALETTE["muted"])
    return s.save("06_numeric_icd_ecg_cohort.svg")


def diagram_numeric_llm_parse_yield() -> Path:
    s = Svg()
    s.title(
        "Discharge-Note LLM Parse Yield",
        "Counts from the completed Azure OpenAI structured-output discharge-note run.",
    )

    ecg_qual = 194_917
    note_visits = 160_095
    parsed = 160_093
    errors = 2
    subjects = 85_030

    boxes = [
        (80, fmt_int(ecg_qual), "24h ECG-qualified visits", PALETTE["source"], PALETTE["source_stroke"], None),
        (410, fmt_int(note_visits), "with discharge notes", PALETTE["note"], PALETTE["note_stroke"], f"{fmt_pct(note_visits, ecg_qual)} of ECG-qualified"),
        (740, fmt_int(parsed), "successful structured parses", PALETTE["output"], PALETTE["output_stroke"], f"{fmt_pct(parsed, note_visits, 3)} parse success"),
        (1070, fmt_int(subjects), "unique parsed subjects", PALETTE["process"], PALETTE["process_stroke"], None),
    ]
    for x, number, label, fill, stroke, sublabel in boxes:
        s.count_box(x, 245, 270, 190, number, label, fill, stroke, sublabel, number_size=42)
    s.arrow_label(350, 340, 410, 340)
    s.arrow_label(680, 340, 740, 340)
    s.arrow_label(1010, 340, 1070, 340)

    s.count_box(
        665,
        555,
        290,
        145,
        fmt_int(errors),
        "parse errors",
        PALETTE["guard"],
        PALETTE["guard_stroke"],
        "streamed to error files",
        number_size=44,
    )
    s.path("M875,435 C875,505 810,515 810,555", arrow=True, dash="8 8")
    s.text(80, 805, "Source: azure_gpt56_luna_full_160095_20260713/summary.json.", size=18, fill=PALETTE["muted"])
    return s.save("07_numeric_llm_parse_yield.svg")


def diagram_numeric_llm_cohort_derivation() -> Path:
    s = Svg()
    black = "#000000"
    white = "#FFFFFF"
    s.arrow_color = black
    s.text(70, 70, "LLM-Derived Target and Control Cohorts", size=36, weight=800, fill=black)

    def flow_box(
        x: float,
        y: float,
        w: float,
        h: float,
        lines: list[str],
        number: str | None = None,
        sublabel: str | None = None,
        number_size: int = 36,
        label_size: int = 20,
    ) -> None:
        s.rect(x, y, w, h, white, black, radius=6, stroke_width=2.2)
        if number:
            s.text(x + w / 2, y + 38, number, size=number_size, weight=800, fill=black, anchor="middle")
            first_y = y + 67
        else:
            first_y = y + h / 2 - ((len(lines) - 1) * 24 / 2) + 8
        s.multiline_text(
            x + w / 2,
            first_y,
            lines,
            size=label_size,
            fill=black,
            anchor="middle",
            line_height=24,
        )
        if sublabel:
            s.text(x + w / 2, y + h - 16, sublabel, size=16, fill=black, anchor="middle")

    def arrow(x1: float, y1: float, x2: float, y2: float, label: str | None = None, tx: float | None = None, ty: float | None = None) -> None:
        s.line(x1, y1, x2, y2, stroke=black, arrow=True)
        if label:
            s.text(tx if tx is not None else (x1 + x2) / 2, ty if ty is not None else (y1 + y2) / 2 - 12, label, size=21, fill=black, anchor="middle")

    flow_box(
        590,
        90,
        420,
        105,
        ["entire cohort"],
        fmt_int(85_200),
        f"{fmt_int(84_849)} + {fmt_int(351)}",
        number_size=40,
        label_size=18,
    )
    flow_box(590, 240, 420, 100, ["Confirmed new", "dissection?"])
    arrow(800, 195, 800, 240)

    flow_box(590, 400, 420, 100, ["Chest/back pain", "and clinically usable control?"])
    arrow(800, 340, 800, 400, f"No: {fmt_int(84_849)}", 855, 374)
    flow_box(160, 405, 260, 90, ["excluded"], fmt_int(76_013), number_size=36, label_size=18)
    arrow(590, 450, 420, 450, f"No: {fmt_int(76_013)}", 505, 431)

    flow_box(590, 560, 420, 100, ["Aortic/unclear", "phenotype?"])
    arrow(800, 500, 800, 560, f"Yes: {fmt_int(8_836)}", 855, 534)
    flow_box(160, 565, 260, 90, ["excluded"], fmt_int(1_193), number_size=36, label_size=18)
    arrow(590, 610, 420, 610, f"Yes: {fmt_int(1_193)}", 505, 591)

    flow_box(
        600,
        705,
        400,
        85,
        ["final controls"],
        fmt_int(7_643),
        None,
        number_size=36,
        label_size=18,
    )
    arrow(800, 660, 800, 705, f"No: {fmt_int(7_643)}", 850, 688)

    flow_box(1115, 225, 275, 90, ["Non-Type-B", "anatomy?"])
    arrow(1010, 290, 1115, 270, f"Yes: {fmt_int(351)}", 1062, 253)
    flow_box(1500, 225, 80, 90, ["excluded"], fmt_int(144), number_size=34, label_size=17)
    arrow(1390, 270, 1500, 270, f"No: {fmt_int(144)}", 1445, 252)

    flow_box(1115, 365, 275, 110, ["No prior or known", "unrepaired dissection", "context?"])
    arrow(1253, 315, 1253, 365, f"Yes: {fmt_int(207)}", 1310, 342)
    flow_box(1500, 375, 80, 90, ["excluded"], fmt_int(20), number_size=34, label_size=17)
    arrow(1390, 420, 1500, 420, f"No: {fmt_int(20)}", 1445, 402)

    flow_box(
        1095,
        525,
        320,
        85,
        ["final targets"],
        fmt_int(187),
        None,
        number_size=36,
        label_size=18,
    )
    arrow(1253, 475, 1253, 525, f"Yes: {fmt_int(187)}", 1310, 500)

    flow_box(
        1095,
        705,
        320,
        115,
        ["model encounters"],
        fmt_int(7_830),
        "187 targets | 7,643 controls",
        number_size=36,
        label_size=18,
    )
    arrow(1000, 747, 1095, 747)
    s.path("M1255,610 C1255,650 1255,675 1255,705", stroke=black, arrow=True)
    return s.save("08_numeric_llm_cohort_derivation.svg")


def diagram_numeric_model_cohort_comparison() -> Path:
    s = Svg()
    s.title(
        "Model Cohort Sizes and Splits",
        "Both models use the same active predictor structure: 16 features, first-only ECG aggregation, XGBoost native categorical handling.",
    )

    # Headers
    s.text(185, 190, "ICD-based ECG-complete cohort", size=28, weight=800)
    s.text(955, 190, "LLM-derived ECG-complete cohort", size=28, weight=800)

    s.count_box(
        130,
        230,
        330,
        170,
        fmt_int(70_094),
        "model rows",
        PALETTE["output"],
        PALETTE["output_stroke"],
        f"{fmt_int(355)} targets | {fmt_pct(355, 70_094, 2)} prevalence",
    )
    s.count_box(
        900,
        230,
        330,
        170,
        fmt_int(7_830),
        "model rows",
        PALETTE["output"],
        PALETTE["output_stroke"],
        f"{fmt_int(187)} targets | {fmt_pct(187, 7_830, 2)} prevalence",
    )

    s.mini_count(130, 505, fmt_int(56_116), f"train\n{fmt_int(285)} targets", PALETTE["model"], PALETTE["model_stroke"], w=230, h=120)
    s.mini_count(395, 505, fmt_int(13_978), f"holdout\n{fmt_int(70)} targets", PALETTE["model"], PALETTE["model_stroke"], w=230, h=120)
    s.mini_count(900, 505, fmt_int(6_264), f"train\n{fmt_int(150)} targets", PALETTE["model"], PALETTE["model_stroke"], w=230, h=120)
    s.mini_count(1165, 505, fmt_int(1_566), f"holdout\n{fmt_int(37)} targets", PALETTE["model"], PALETTE["model_stroke"], w=230, h=120)

    s.path("M295,400 C295,455 245,465 245,505", arrow=True)
    s.path("M295,400 C295,455 510,465 510,505", arrow=True)
    s.path("M1065,400 C1065,455 1015,465 1015,505", arrow=True)
    s.path("M1065,400 C1065,455 1280,465 1280,505", arrow=True)

    s.count_box(
        595,
        665,
        410,
        150,
        "16",
        "active predictors",
        PALETTE["source"],
        PALETTE["source_stroke"],
        "5 demo + 11 first ECG",
        number_size=40,
        label_size=19,
    )
    s.text(80, 845, "Sources: demographics_ecg_runs/20260708_151834 and llm_demographics_ecg_runs/llm_cohort_demographics_ecg_xgb_20260714_102716.", size=18, fill=PALETTE["muted"])
    return s.save("09_numeric_model_cohort_comparison.svg")


def write_readme() -> Path:
    readme = OUT_DIR / "README.md"
    readme.write_text(
        """# Poster Methods Diagrams

Editable SVG files and high-resolution PNG previews for the aortic-dissection ECG project.

Files:

- `01_end_to_end_workflow.svg`: overall MIMIC-IV to model and LLM-cohort workflow.
- `02_cohort_and_feature_window.svg`: cohort definition, index-time rule, and 24-hour feature window.
- `03_active_predictor_set.svg`: active demographics plus ECG predictor set and default exclusions.
- `04_discharge_note_llm_parser.svg`: discharge-note structured-output phenotyping workflow.
- `05_model_training_and_evaluation.svg`: model preprocessing, validation, holdout evaluation, and reporting.
- `06_numeric_icd_ecg_cohort.svg`: number-oriented ICD-based ECG-complete cohort attrition.
- `07_numeric_llm_parse_yield.svg`: number-oriented discharge-note LLM parse yield.
- `08_numeric_llm_cohort_derivation.svg`: number-oriented LLM target/control derivation.
- `09_numeric_model_cohort_comparison.svg`: number-oriented side-by-side cohort and split comparison.

The SVGs are the preferred poster assets because they remain sharp at print scale. PNG copies are exported at 3200 px width for quick insertion into slides or drafts.

Regenerate assets with:

```bash
python poster_assets/methods_diagrams/draw_methods_diagrams.py
```
""",
        encoding="utf-8",
    )
    return readme


def write_results_suggestions() -> Path:
    path = OUT_DIR / "poster_results_figure_suggestions.md"
    path.write_text(
        """# Suggested Results Figures Beyond Current Model Presentations

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
""",
        encoding="utf-8",
    )
    return path


def main() -> None:
    paths = [
        diagram_overall_workflow(),
        diagram_cohort_window(),
        diagram_predictor_set(),
        diagram_llm_parser(),
        diagram_model_training(),
        diagram_numeric_icd_cohort(),
        diagram_numeric_llm_parse_yield(),
        diagram_numeric_llm_cohort_derivation(),
        diagram_numeric_model_cohort_comparison(),
        write_readme(),
        write_results_suggestions(),
    ]
    for path in paths:
        if path.suffix == ".svg":
            write_png(path)
    print("Generated poster assets:")
    for path in sorted(OUT_DIR.iterdir()):
        if path.suffix in {".svg", ".png", ".md"}:
            print(f"  {path.relative_to(OUT_DIR.parent.parent)}")


if __name__ == "__main__":
    main()
