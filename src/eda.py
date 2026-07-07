"""Exploratory data analysis and summary tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils import save_dataframe


def _single_row_table(metric: str, value: int | float) -> pd.DataFrame:
    return pd.DataFrame({"metric": [metric], "value": [value]})


def build_eda_tables(df: pd.DataFrame, score_columns: list[str]) -> dict[str, pd.DataFrame]:
    """Build a full EDA table set from the cleaned dataframe."""

    tables: dict[str, pd.DataFrame] = {}

    tables["shape"] = pd.DataFrame(
        {"metric": ["rows", "columns"], "value": [df.shape[0], df.shape[1]]}
    )
    tables["data_types"] = pd.DataFrame(
        {"column": df.columns, "dtype": [str(dtype) for dtype in df.dtypes]}
    )
    tables["missing_values"] = pd.DataFrame(
        {"column": df.columns, "missing_count": df.isna().sum().values}
    )
    tables["duplicate_values"] = pd.DataFrame(
        {"metric": ["duplicate_rows"], "value": [int(df.duplicated().sum())]}
    )

    numeric_df = df[score_columns].copy()
    describe_df = numeric_df.describe().T
    tables["describe"] = describe_df.reset_index().rename(columns={"index": "subject"})

    stats_rows = []
    for column in score_columns:
        series = numeric_df[column]
        stats_rows.append(
            {
                "subject": column,
                "mean": series.mean(),
                "median": series.median(),
                "mode": series.mode().iloc[0] if not series.mode().empty else pd.NA,
                "variance": series.var(ddof=1),
                "std_dev": series.std(ddof=1),
                "min": series.min(),
                "q1": series.quantile(0.25),
                "q3": series.quantile(0.75),
                "max": series.max(),
                "iqr": series.quantile(0.75) - series.quantile(0.25),
            }
        )

    tables["summary_stats"] = pd.DataFrame(stats_rows)
    tables["mean_scores"] = tables["summary_stats"][["subject", "mean"]].copy()
    tables["median_scores"] = tables["summary_stats"][["subject", "median"]].copy()
    tables["mode_scores"] = tables["summary_stats"][["subject", "mode"]].copy()
    tables["variance_std"] = tables["summary_stats"][["subject", "variance", "std_dev"]].copy()
    tables["min_max_iqr"] = tables["summary_stats"][["subject", "min", "q1", "q3", "max", "iqr"]].copy()
    tables["correlation_matrix"] = numeric_df.corr().reset_index().rename(columns={"index": "subject"})

    return tables


def save_eda_tables(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Persist all EDA tables to disk."""

    for table_name, table_df in tables.items():
        save_dataframe(table_df, output_dir / f"{table_name}.csv")
