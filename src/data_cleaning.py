"""Data loading and cleaning pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from utils import (
    CANONICAL_SCORE_COLUMNS,
    detect_column_mapping,
    save_dataframe,
)


@dataclass
class CleaningStepResult:
    """Summary of a single cleaning step."""

    step: str
    initial_count: int
    removed_count: int
    remaining_count: int


@dataclass
class CleaningResult:
    """Container for cleaned data and step summaries."""

    dataframe: pd.DataFrame
    score_columns: list[str]
    step_results: list[CleaningStepResult]


def load_raw_data(csv_path: Path) -> pd.DataFrame:
    """Load the raw CSV as strings to preserve IDs and empty cells."""

    if not csv_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {csv_path}")
    return pd.read_csv(csv_path, dtype=str)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw columns to canonical names when possible."""

    rename_map = detect_column_mapping(df.columns)
    standardized = df.rename(columns=rename_map).copy()
    return standardized


def identify_score_columns(df: pd.DataFrame) -> list[str]:
    """Return the score columns present in the dataframe."""

    score_columns = [column for column in CANONICAL_SCORE_COLUMNS if column in df.columns]
    if not score_columns:
        raise ValueError("Không nhận diện được cột điểm nào trong dữ liệu.")
    return score_columns


def _normalize_sbd(series: pd.Series) -> pd.Series:
    """Normalize the candidate ID to 8 digits when possible."""

    cleaned = series.astype(str).str.strip()
    cleaned = cleaned.str.replace(r"\.0$", "", regex=True)
    cleaned = cleaned.replace({"nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    return cleaned.str.zfill(8)


def _record_step(step: str, before: int, after: int) -> CleaningStepResult:
    """Create a cleaning summary record."""

    return CleaningStepResult(
        step=step,
        initial_count=before,
        removed_count=before - after,
        remaining_count=after,
    )


def clean_scores_data(df: pd.DataFrame, score_columns: Sequence[str]) -> CleaningResult:
    """Clean the score dataframe following the requested business rules."""

    working = df.copy()
    step_results: list[CleaningStepResult] = []

    if "sbd" in working.columns:
        working["sbd"] = _normalize_sbd(working["sbd"])

    for column in score_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    before = len(working)
    working = working.drop_duplicates().copy()
    step_results.append(_record_step("Drop duplicate rows", before, len(working)))

    before = len(working)
    invalid_mask = pd.Series(False, index=working.index)
    for column in score_columns:
        invalid_mask = invalid_mask | (
            working[column].notna() & ((working[column] < 0) | (working[column] > 10))
        )
    working = working.loc[~invalid_mask].copy()
    step_results.append(_record_step("Remove scores outside 0-10", before, len(working)))

    required_columns = [column for column in ("toan", "van") if column in working.columns]
    if len(required_columns) < 2:
        raise ValueError("Thiếu một trong hai cột bắt buộc: toan, van.")

    before = len(working)
    required_mask = working[required_columns].isna().any(axis=1)
    working = working.loc[~required_mask].copy()
    step_results.append(_record_step("Require Toan and Van", before, len(working)))

    working = working.reset_index(drop=True)
    return CleaningResult(dataframe=working, score_columns=list(score_columns), step_results=step_results)


def save_cleaned_data(df: pd.DataFrame, output_path: Path) -> None:
    """Save cleaned data to disk."""

    save_dataframe(df, output_path)
