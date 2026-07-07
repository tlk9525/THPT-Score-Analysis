"""Shared utility helpers for the THPT score analysis project."""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import pandas as pd


CANONICAL_SCORE_COLUMNS = [
    "toan",
    "ly",
    "hoa",
    "van",
    "su",
    "dia",
    "ngoai_ngu",
    "sinh",
]


def ensure_dir(path: Path) -> Path:
    """Create a directory if it does not already exist."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_text(value: str) -> str:
    """Normalize text for robust column matching."""

    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    return re.sub(r"[^0-9a-zA-Z]+", "_", ascii_text.lower()).strip("_")


def setup_logging(log_file: Path) -> logging.Logger:
    """Configure console and file logging."""

    ensure_dir(log_file.parent)
    logger = logging.getLogger("thpt_score_analysis")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False
    return logger


def save_dataframe(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Save a dataframe to CSV with consistent encoding."""

    ensure_dir(path.parent)
    df.to_csv(path, index=index, encoding="utf-8-sig")


def save_text(text: str, path: Path) -> None:
    """Save text content to a file."""

    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def subject_aliases() -> dict[str, str]:
    """Return canonical subject aliases for auto-mapping columns."""

    aliases = {
        "sbd": "sbd",
        "so_bao_danh": "sbd",
        "bao_danh": "sbd",
        "candidate_id": "sbd",
        "toan": "toan",
        "math": "toan",
        "mathematics": "toan",
        "vat_ly": "ly",
        "physics": "ly",
        "ly": "ly",
        "hoa": "hoa",
        "hoa_hoc": "hoa",
        "chemistry": "hoa",
        "van": "van",
        "ngu_van": "van",
        "literature": "van",
        "su": "su",
        "lich_su": "su",
        "history": "su",
        "dia": "dia",
        "dia_ly": "dia",
        "geography": "dia",
        "ngoai_ngu": "ngoai_ngu",
        "tieng_anh": "ngoai_ngu",
        "english": "ngoai_ngu",
        "sinh": "sinh",
        "sinh_hoc": "sinh",
        "biology": "sinh",
    }
    return aliases


def detect_column_mapping(columns: Iterable[str]) -> dict[str, str]:
    """Map raw column names to canonical names where possible."""

    aliases = subject_aliases()
    mapping: dict[str, str] = {}
    for column in columns:
        normalized = normalize_text(column)
        if normalized in aliases:
            mapping[column] = aliases[normalized]
            continue

        compact = normalized.replace("_", "")
        for alias, canonical in aliases.items():
            alias_compact = alias.replace("_", "")
            if compact == alias_compact or compact.startswith(alias_compact):
                mapping[column] = canonical
                break
    return mapping


def prepare_output_directories(project_root: Path) -> dict[str, Path]:
    """Create and return all project output directories."""

    data_raw = ensure_dir(project_root / "data" / "raw")
    data_processed = ensure_dir(project_root / "data" / "processed")
    figures = ensure_dir(project_root / "output" / "figures")
    reports = ensure_dir(project_root / "output" / "reports")
    tables = ensure_dir(project_root / "output" / "tables")
    logs = ensure_dir(project_root / "output" / "logs")

    return {
        "data_raw": data_raw,
        "data_processed": data_processed,
        "figures": figures,
        "reports": reports,
        "tables": tables,
        "logs": logs,
    }

