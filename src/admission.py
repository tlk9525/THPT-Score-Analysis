"""Admission scoring and ranking engine.

This module is separate from EDA because admissions must follow the
specific formula of each university/major.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import pandas as pd

from utils import detect_column_mapping, normalize_text, save_dataframe


@dataclass
class AdmissionRule:
    """A single admission formula for one major/program."""

    university: str
    major: str
    subject_columns: list[str]
    weights: list[float]
    bonus_points: float = 0.0
    threshold: float | None = None
    priority_points: float = 0.0
    note: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AdmissionRule":
        """Build a rule from JSON/dict configuration."""

        subjects = [normalize_text(str(item)) for item in data["subject_columns"]]
        weights = [float(item) for item in data["weights"]]
        if len(subjects) != len(weights):
            raise ValueError(
                f"Rule {data.get('university', '')} - {data.get('major', '')} "
                "phải có số môn và số trọng số bằng nhau."
            )
        return cls(
            university=str(data.get("university", "")).strip(),
            major=str(data.get("major", "")).strip(),
            subject_columns=subjects,
            weights=weights,
            bonus_points=float(data.get("bonus_points", 0.0)),
            threshold=float(data["threshold"]) if data.get("threshold") is not None else None,
            priority_points=float(data.get("priority_points", 0.0)),
            note=str(data.get("note", "")).strip(),
        )


@dataclass
class UEHDemoScenario:
    """Hard-coded UEH admission scenario for quick reporting."""

    option_name: str
    major_label: str
    exam_component: float
    school_component: float
    bonus_component: float = 0.0

    @property
    def total_score(self) -> float:
        """Return the total admission score on a 100-point scale."""

        return self.exam_component + self.school_component + self.bonus_component


UEH_HARDCODED_SCENARIOS: list[UEHDemoScenario] = [
    UEHDemoScenario(
        option_name="Phương án 1 (V-ACT)",
        major_label="UEH",
        exam_component=43.650,
        school_component=36.660,
        bonus_component=5.000,
    ),
    UEHDemoScenario(
        option_name="Phương án 2 (Tổ hợp A01)",
        major_label="UEH",
        exam_component=45.375,
        school_component=36.660,
        bonus_component=5.000,
    ),
    UEHDemoScenario(
        option_name="Phương án 3 (Tổ hợp D01)",
        major_label="UEH",
        exam_component=46.500,
        school_component=36.660,
        bonus_component=5.000,
    ),
]


def load_admission_rules(json_path: Path) -> list[AdmissionRule]:
    """Load admission rules from a JSON file."""

    if not json_path.exists():
        return []
    content = json.loads(json_path.read_text(encoding="utf-8"))
    rules = content if isinstance(content, list) else content.get("rules", [])
    return [AdmissionRule.from_dict(item) for item in rules]


def _resolve_subject_columns(df: pd.DataFrame, subject_columns: list[str]) -> list[str]:
    """Match rule subject names to dataframe columns."""

    available = detect_column_mapping(df.columns)
    reverse_lookup = {normalize_text(canonical): raw for raw, canonical in available.items()}
    reverse_lookup.update({normalize_text(col): col for col in df.columns})

    resolved: list[str] = []
    for subject in subject_columns:
        key = normalize_text(subject)
        if key in reverse_lookup:
            resolved.append(reverse_lookup[key])
        else:
            raise KeyError(f"Không tìm thấy cột môn '{subject}' trong dữ liệu.")
    return resolved


def calculate_admission_score(row: pd.Series, subject_columns: list[str], weights: list[float]) -> float | None:
    """Calculate the admission score for one candidate row.

    Returns None when any required subject is missing.
    """

    total = 0.0
    for column, weight in zip(subject_columns, weights, strict=True):
        value = row.get(column)
        if pd.isna(value):
            return None
        total += float(value) * float(weight)
    return total


def apply_admission_rule(df: pd.DataFrame, rule: AdmissionRule) -> pd.DataFrame:
    """Apply one admission rule and return a ranked result table."""

    subject_columns = _resolve_subject_columns(df, rule.subject_columns)
    working = df.copy()

    working["admission_score"] = working.apply(
        lambda row: calculate_admission_score(row, subject_columns, rule.weights),
        axis=1,
    )
    working = working.dropna(subset=["admission_score"]).copy()
    working["admission_score"] = working["admission_score"] + rule.bonus_points + rule.priority_points
    if rule.threshold is not None:
        working = working.loc[working["admission_score"] >= rule.threshold].copy()

    sort_columns = ["admission_score"]
    ascending = [False]
    if "sbd" in working.columns:
        sort_columns.append("sbd")
        ascending.append(True)

    working = working.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)
    working.insert(0, "rank", range(1, len(working) + 1))
    working.insert(1, "university", rule.university)
    working.insert(2, "major", rule.major)
    working["formula"] = " + ".join(
        f"{weight:g}*{subject}" for subject, weight in zip(rule.subject_columns, rule.weights, strict=True)
    )
    if rule.note:
        working["note"] = rule.note
    return working


def run_admission_analysis(
    df: pd.DataFrame,
    rules_path: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """Run all admission rules and save the combined rankings."""

    rules = load_admission_rules(rules_path)
    if not rules:
        return pd.DataFrame()

    results = [apply_admission_rule(df, rule) for rule in rules]
    combined = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    save_dataframe(combined, output_dir / "admission_rankings.csv")
    return combined


def build_ueh_hardcoded_demo_table() -> pd.DataFrame:
    """Build a fixed UEH-style summary table from hard-coded values."""

    rows = []
    for scenario in UEH_HARDCODED_SCENARIOS:
        rows.append(
            {
                "Thành phần điểm": scenario.option_name,
                "Điểm thi (60%)": scenario.exam_component,
                "Học bạ (40%)": scenario.school_component,
                "Điểm IELTS 6.5": scenario.bonus_component,
                "Tổng điểm xét tuyển": scenario.total_score,
            }
        )

    table = pd.DataFrame(rows)
    table = table[[
        "Thành phần điểm",
        "Điểm thi (60%)",
        "Học bạ (40%)",
        "Điểm IELTS 6.5",
        "Tổng điểm xét tuyển",
    ]]
    return table


def save_ueh_hardcoded_demo_table(output_dir: Path) -> pd.DataFrame:
    """Save the fixed UEH demo table for quick use."""

    table = build_ueh_hardcoded_demo_table()
    save_dataframe(table, output_dir / "ueh_hardcoded_demo.csv")
    return table
