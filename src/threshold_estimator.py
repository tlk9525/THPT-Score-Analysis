"""Estimate admission cutoff thresholds from score distributions.

The estimator is intentionally explicit about its assumptions:
- It uses the current score distribution from the cleaned 2026 dataset.
- It can optionally use a rules JSON and a historical cutoff CSV.
- If historical applicant counts are missing, it falls back to a
  user-configurable competition rate or a conservative pool assumption.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils import detect_column_mapping, normalize_text, save_dataframe


@dataclass
class ThresholdRule:
    """A single admission cutoff estimation rule."""

    university: str
    major: str
    subject_columns: list[str]
    weights: list[float]
    quota: int
    application_rate: float | None = None
    estimated_applicant_count: int | None = None
    bonus_points: float = 0.0
    priority_points: float = 0.0
    minimum_total_score: float | None = None
    note: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ThresholdRule":
        """Build a rule from a JSON-compatible mapping."""

        subjects = [normalize_text(str(item)) for item in data["subject_columns"]]
        weights = [float(item) for item in data["weights"]]
        if len(subjects) != len(weights):
            raise ValueError(
                f"Rule {data.get('university', '')} - {data.get('major', '')} "
                "phải có số môn và số trọng số bằng nhau."
            )

        quota = int(data["quota"])
        if quota < 1:
            raise ValueError("quota phải >= 1")

        application_rate = data.get("application_rate")
        estimated_applicant_count = data.get("estimated_applicant_count")

        return cls(
            university=str(data.get("university", "")).strip(),
            major=str(data.get("major", "")).strip(),
            subject_columns=subjects,
            weights=weights,
            quota=quota,
            application_rate=float(application_rate) if application_rate is not None else None,
            estimated_applicant_count=(
                int(estimated_applicant_count) if estimated_applicant_count is not None else None
            ),
            bonus_points=float(data.get("bonus_points", 0.0)),
            priority_points=float(data.get("priority_points", 0.0)),
            minimum_total_score=(
                float(data["minimum_total_score"])
                if data.get("minimum_total_score") is not None
                else None
            ),
            note=str(data.get("note", "")).strip(),
        )


@dataclass
class HistoricalCutoffRecord:
    """Historical cutoff metadata for one year."""

    university: str
    major: str
    year: int
    quota: int | None
    applicants: int | None
    cutoff_score: float | None


def load_threshold_rules(json_path: Path) -> list[ThresholdRule]:
    """Load threshold rules from a JSON file.

    The file is optional; missing file returns an empty list.
    """

    if not json_path.exists():
        return []
    content = json.loads(json_path.read_text(encoding="utf-8"))
    rules = content if isinstance(content, list) else content.get("rules", [])
    return [ThresholdRule.from_dict(item) for item in rules]


def load_historical_cutoffs(csv_path: Path) -> pd.DataFrame:
    """Load historical cutoff data if present."""

    if not csv_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    expected = {"university", "major", "year", "quota", "applicants", "cutoff_score"}
    if not expected.intersection(set(df.columns)):
        raise ValueError(
            "Historical cutoff CSV phải có ít nhất các cột: "
            "university, major, year, quota, applicants, cutoff_score"
        )
    return df


def _resolve_subject_columns(df: pd.DataFrame, subject_columns: list[str]) -> list[str]:
    """Resolve canonical subject names to actual dataframe columns."""

    available = detect_column_mapping(df.columns)
    reverse_lookup = {normalize_text(canonical): raw for raw, canonical in available.items()}
    reverse_lookup.update({normalize_text(col): col for col in df.columns})

    resolved: list[str] = []
    for subject in subject_columns:
        key = normalize_text(subject)
        if key not in reverse_lookup:
            raise KeyError(f"Không tìm thấy cột môn '{subject}' trong dữ liệu.")
        resolved.append(reverse_lookup[key])
    return resolved


def _calculate_weighted_score(
    df: pd.DataFrame,
    subject_columns: list[str],
    weights: list[float],
    bonus_points: float,
    priority_points: float,
) -> pd.Series:
    """Compute weighted admission scores for all rows."""

    working = df[subject_columns].copy()
    numeric = working.apply(pd.to_numeric, errors="coerce")
    score = pd.Series(0.0, index=numeric.index)

    for column, weight in zip(subject_columns, weights, strict=True):
        score = score + numeric[column] * float(weight)

    score = score + float(bonus_points) + float(priority_points)
    return score


def _estimate_applicant_count(
    rule: ThresholdRule,
    eligible_count: int,
    historical: pd.DataFrame,
) -> int:
    """Estimate applicant count using history or rule hints."""

    if rule.estimated_applicant_count is not None:
        return max(rule.quota, int(rule.estimated_applicant_count))

    if rule.application_rate is not None:
        return max(rule.quota, int(round(eligible_count * rule.application_rate)))

    if historical.empty:
        return max(rule.quota, eligible_count)

    mask = (
        historical["university"].astype(str).str.strip().str.lower().eq(rule.university.lower())
        & historical["major"].astype(str).str.strip().str.lower().eq(rule.major.lower())
    )
    matched = historical.loc[mask].copy()
    if matched.empty:
        return max(rule.quota, eligible_count)

    if "applicants" in matched.columns and "quota" in matched.columns:
        matched["competition_rate"] = pd.to_numeric(matched["applicants"], errors="coerce") / pd.to_numeric(
            matched["quota"], errors="coerce"
        )
        competition_rate = matched["competition_rate"].dropna()
        if not competition_rate.empty:
            avg_rate = float(competition_rate.mean())
            return max(rule.quota, int(round(rule.quota * avg_rate)))

    applicants = pd.to_numeric(matched.get("applicants"), errors="coerce").dropna()
    if not applicants.empty:
        return max(rule.quota, int(round(float(applicants.mean()))))

    return max(rule.quota, eligible_count)


def estimate_cutoff_for_rule(
    df: pd.DataFrame,
    rule: ThresholdRule,
    historical_cutoffs: pd.DataFrame | None = None,
) -> pd.Series:
    """Estimate the cutoff score for one rule and return a one-row Series."""

    historical = historical_cutoffs if historical_cutoffs is not None else pd.DataFrame()
    resolved_subjects = _resolve_subject_columns(df, rule.subject_columns)
    score = _calculate_weighted_score(
        df,
        subject_columns=resolved_subjects,
        weights=rule.weights,
        bonus_points=rule.bonus_points,
        priority_points=rule.priority_points,
    )

    eligible_mask = pd.Series(True, index=df.index)
    for column in resolved_subjects:
        eligible_mask = eligible_mask & df[column].notna()

    eligible_scores = score.loc[eligible_mask].dropna()
    if rule.minimum_total_score is not None:
        eligible_scores = eligible_scores.loc[eligible_scores >= rule.minimum_total_score]

    eligible_count = int(eligible_scores.count())
    if eligible_count == 0:
        return pd.Series(
            {
                "university": rule.university,
                "major": rule.major,
                "quota": rule.quota,
                "eligible_pool": 0,
                "estimated_applicants": np.nan,
                "competition_rate": np.nan,
                "target_quantile": np.nan,
                "estimated_cutoff_score": np.nan,
                "historical_average_cutoff": np.nan,
                "historical_average_applicants": np.nan,
                "delta_vs_history": np.nan,
                "note": rule.note,
            }
        )

    estimated_applicants = _estimate_applicant_count(rule, eligible_count, historical)
    estimated_applicants = max(rule.quota, estimated_applicants)
    competition_rate = estimated_applicants / rule.quota
    target_quantile = max(0.0, min(1.0, 1.0 - (rule.quota / estimated_applicants)))

    cutoff = float(eligible_scores.quantile(target_quantile, interpolation="linear"))

    history_mask = (
        historical.get("university", pd.Series(dtype=str)).astype(str).str.strip().str.lower().eq(rule.university.lower())
        & historical.get("major", pd.Series(dtype=str)).astype(str).str.strip().str.lower().eq(rule.major.lower())
    ) if not historical.empty else pd.Series(dtype=bool)

    historical_cutoff = np.nan
    historical_applicants = np.nan
    if not historical.empty and history_mask.any():
        matched = historical.loc[history_mask].copy()
        if "cutoff_score" in matched.columns:
            cutoff_series = pd.to_numeric(matched["cutoff_score"], errors="coerce").dropna()
            if not cutoff_series.empty:
                historical_cutoff = float(cutoff_series.mean())
        if "applicants" in matched.columns:
            app_series = pd.to_numeric(matched["applicants"], errors="coerce").dropna()
            if not app_series.empty:
                historical_applicants = float(app_series.mean())

    return pd.Series(
        {
            "university": rule.university,
            "major": rule.major,
            "quota": rule.quota,
            "eligible_pool": eligible_count,
            "estimated_applicants": estimated_applicants,
            "competition_rate": competition_rate,
            "target_quantile": target_quantile,
            "estimated_cutoff_score": cutoff,
            "historical_average_cutoff": historical_cutoff,
            "historical_average_applicants": historical_applicants,
            "delta_vs_history": cutoff - historical_cutoff if pd.notna(historical_cutoff) else np.nan,
            "note": rule.note,
        }
    )


def estimate_cutoffs(
    df: pd.DataFrame,
    rules_path: Path,
    historical_path: Path | None,
    output_dir: Path,
) -> pd.DataFrame:
    """Estimate cutoff scores for all configured rules and save the table."""

    rules = load_threshold_rules(rules_path)
    if not rules:
        return pd.DataFrame()

    historical = load_historical_cutoffs(historical_path) if historical_path else pd.DataFrame()
    rows = [estimate_cutoff_for_rule(df, rule, historical) for rule in rules]
    result = pd.DataFrame(rows)
    save_dataframe(result, output_dir / "cutoff_estimates.csv")
    return result

