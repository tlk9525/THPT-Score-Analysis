"""Entry point for the THPT score analysis project."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis import (
    analyze_scores,
    generate_report,
    save_analysis_tables,
    save_report,
)
from data_cleaning import (
    clean_scores_data,
    identify_score_columns,
    load_raw_data,
    save_cleaned_data,
    standardize_columns,
)
from eda import build_eda_tables, save_eda_tables
from threshold_estimator import estimate_cutoffs
from utils import prepare_output_directories, save_dataframe, setup_logging
from visualization import create_visualizations


def _resolve_raw_data_path(project_root: Path) -> Path:
    """Find the raw CSV file in the project or nearby fallback locations."""

    raw_dir = project_root / "data" / "raw"
    preferred_files = sorted(raw_dir.glob("scores*.csv"))
    if preferred_files:
        return preferred_files[0]

    fallback_candidates = [
        project_root.parent / "scores_2026_full.csv",
        project_root / "scores_2026_full.csv",
    ]
    for candidate in fallback_candidates:
        if candidate.exists():
            return candidate

    csv_files = sorted(raw_dir.glob("*.csv"))
    if len(csv_files) == 1:
        return csv_files[0]
    if len(csv_files) > 1:
        available = ", ".join(path.name for path in csv_files)
        raise FileNotFoundError(
            "Có nhiều file CSV trong data/raw nhưng không có file nào bắt đầu bằng 'scores'. "
            f"Hãy giữ đúng 1 file đầu vào hoặc đổi tên file dữ liệu. Hiện có: {available}"
        )
    raise FileNotFoundError(
        "Không tìm thấy file CSV đầu vào. Hãy đặt dữ liệu vào data/raw/ trước khi chạy."
    )


def main() -> None:
    """Run the full data science pipeline."""

    project_root = Path(__file__).resolve().parents[1]
    paths = prepare_output_directories(project_root)
    logger = setup_logging(paths["logs"] / "pipeline.log")

    raw_path = _resolve_raw_data_path(project_root)
    logger.info("Đọc dữ liệu từ: %s", raw_path)
    raw_df = load_raw_data(raw_path)
    raw_rows = len(raw_df)

    standardized_df = standardize_columns(raw_df)
    score_columns = identify_score_columns(standardized_df)
    logger.info("Các cột điểm được nhận diện: %s", ", ".join(score_columns))

    cleaning_result = clean_scores_data(standardized_df, score_columns)
    cleaned_df = cleaning_result.dataframe
    cleaned_rows = len(cleaned_df)

    cleaned_output_path = paths["data_processed"] / "cleaned_scores.csv"
    save_cleaned_data(cleaned_df, cleaned_output_path)
    logger.info("Đã lưu dữ liệu sạch vào: %s", cleaned_output_path)

    cleaning_steps = [
        {
            "step": step.step,
            "initial_count": step.initial_count,
            "removed_count": step.removed_count,
            "remaining_count": step.remaining_count,
        }
        for step in cleaning_result.step_results
    ]

    for step in cleaning_steps:
        logger.info(
            "%s | ban dau=%s | loai=%s | con lai=%s",
            step["step"],
            step["initial_count"],
            step["removed_count"],
            step["remaining_count"],
        )

    save_dataframe(
        pd.DataFrame(cleaning_steps),
        paths["tables"] / "cleaning_steps.csv",
    )

    eda_tables = build_eda_tables(cleaned_df, score_columns)
    save_eda_tables(eda_tables, paths["tables"])
    logger.info("Đã sinh các bảng EDA.")

    analysis_tables = analyze_scores(cleaned_df, score_columns)
    save_analysis_tables(analysis_tables, paths["tables"])
    logger.info("Đã sinh các bảng phân tích thống kê.")

    create_visualizations(cleaned_df, score_columns, paths["figures"])
    logger.info("Đã sinh toàn bộ biểu đồ.")

    threshold_rules_path = project_root / "data" / "raw" / "ueh_threshold_rules.json"
    historical_cutoff_path = project_root / "data" / "raw" / "ueh_cutoffs_history.csv"
    cutoff_estimates = estimate_cutoffs(
        cleaned_df,
        rules_path=threshold_rules_path,
        historical_path=historical_cutoff_path,
        output_dir=paths["tables"],
    )
    if not cutoff_estimates.empty:
        logger.info(
            "Đã sinh bảng ước lượng ngưỡng cạnh tranh: %s",
            paths["tables"] / "cutoff_estimates.csv",
        )
    else:
        logger.info("Chưa có file UEH threshold rules, bỏ qua bước ước lượng ngưỡng.")

    report_text = generate_report(
        raw_rows=raw_rows,
        cleaned_rows=cleaned_rows,
        cleaning_steps=cleaning_steps,
        analysis_tables=analysis_tables,
        score_columns=score_columns,
    )
    report_path = paths["reports"] / "report.txt"
    save_report(report_text, report_path)
    logger.info("Đã sinh báo cáo cuối cùng: %s", report_path)

    print("\nHoàn tất pipeline.\n")
    print(f"Dữ liệu sạch: {cleaned_output_path}")
    print(f"Biểu đồ: {paths['figures']}")
    print(f"Bảng: {paths['tables']}")
    print(f"Báo cáo: {report_path}")


if __name__ == "__main__":
    main()
