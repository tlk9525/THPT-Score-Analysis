"""Statistical analysis and report generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from utils import save_dataframe, save_text


def _total_score(df: pd.DataFrame, score_columns: list[str]) -> pd.Series:
    """Calculate total score per candidate."""

    return df[score_columns].sum(axis=1)


def analyze_scores(df: pd.DataFrame, score_columns: list[str]) -> dict[str, pd.DataFrame]:
    """Run the requested statistical analysis and return all result tables."""

    results: dict[str, pd.DataFrame] = {}
    numeric_df = df[score_columns].copy()
    total_score = _total_score(df, score_columns)

    per_subject_rows = []
    outlier_rows = []
    for column in score_columns:
        series = numeric_df[column].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_mask = (series < lower_bound) | (series > upper_bound)

        per_subject_rows.append(
            {
                "subject": column,
                "mean": series.mean(),
                "median": series.median(),
                "std_dev": series.std(ddof=1),
                "variance": series.var(ddof=1),
                "pct_ge_5": (series.ge(5).mean()) * 100,
                "pct_ge_8": (series.ge(8).mean()) * 100,
                "pct_ge_9": (series.ge(9).mean()) * 100,
                "pct_10": (series.eq(10).mean()) * 100,
                "skewness": stats.skew(series, bias=False),
                "kurtosis": stats.kurtosis(series, bias=False),
                "min": series.min(),
                "max": series.max(),
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "outlier_count": int(outlier_mask.sum()),
                "outlier_rate_pct": (outlier_mask.mean()) * 100,
            }
        )

        outlier_rows.append(
            {
                "subject": column,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outlier_count": int(outlier_mask.sum()),
                "outlier_rate_pct": (outlier_mask.mean()) * 100,
            }
        )

    results["subject_statistics"] = pd.DataFrame(per_subject_rows)
    results["outlier_analysis"] = pd.DataFrame(outlier_rows)

    total_stats = pd.DataFrame(
        [
            {
                "metric": "total_score",
                "mean": total_score.mean(),
                "median": total_score.median(),
                "std_dev": total_score.std(ddof=1),
                "variance": total_score.var(ddof=1),
                "min": total_score.min(),
                "max": total_score.max(),
                "q1": total_score.quantile(0.25),
                "q3": total_score.quantile(0.75),
                "iqr": total_score.quantile(0.75) - total_score.quantile(0.25),
                "skewness": stats.skew(total_score, bias=False),
                "kurtosis": stats.kurtosis(total_score, bias=False),
            }
        ]
    )
    results["total_score_statistics"] = total_stats

    total_score_df = df[["sbd"]].copy() if "sbd" in df.columns else pd.DataFrame(index=df.index)
    total_score_df["total_score"] = total_score
    total_score_df = total_score_df.sort_values("total_score", ascending=False).reset_index(drop=True)
    results["top_10_total_scores"] = total_score_df.head(10).copy()
    results["bottom_10_total_scores"] = total_score_df.tail(10).sort_values(
        "total_score", ascending=True
    ).reset_index(drop=True)

    cor_rows = []
    for base in ("toan", "van"):
        if base in numeric_df.columns:
            for other in score_columns:
                if other == base:
                    continue
                cor_rows.append(
                    {
                        "base_subject": base,
                        "other_subject": other,
                        "pearson_correlation": numeric_df[base].corr(numeric_df[other]),
                    }
                )
    results["subject_correlations"] = pd.DataFrame(cor_rows)

    difficulty_rank = (
        results["subject_statistics"][["subject", "mean", "pct_ge_5", "pct_ge_8", "pct_10"]]
        .sort_values(["mean", "pct_ge_8"], ascending=[True, True])
        .reset_index(drop=True)
    )
    results["difficulty_ranking"] = difficulty_rank

    discrimination_rank = (
        results["subject_statistics"][["subject", "std_dev", "variance", "iqr"]]
        .sort_values(["std_dev", "iqr"], ascending=[False, False])
        .reset_index(drop=True)
    )
    results["discrimination_ranking"] = discrimination_rank

    distribution_rows = []
    for column in score_columns:
        series = numeric_df[column]
        if len(series) >= 8 and series.nunique(dropna=True) > 1:
            try:
                normaltest_result = stats.normaltest(series)
                normal_stat = normaltest_result.statistic
                normal_pvalue = normaltest_result.pvalue
            except Exception:
                normal_stat = np.nan
                normal_pvalue = np.nan
        else:
            normal_stat = np.nan
            normal_pvalue = np.nan
        distribution_rows.append(
            {
                "subject": column,
                "count": int(series.count()),
                "skewness": stats.skew(series, bias=False),
                "kurtosis": stats.kurtosis(series, bias=False),
                "normal_test_statistic": normal_stat,
                "normal_test_pvalue": normal_pvalue,
            }
        )
    results["distribution_analysis"] = pd.DataFrame(distribution_rows)

    total_bins = pd.cut(
        total_score,
        bins=[0, 10, 15, 20, 25, 30, 35, 40],
        include_lowest=True,
        right=False,
    )
    results["total_score_bins"] = (
        total_bins.value_counts().sort_index().rename_axis("score_bin").reset_index(name="count")
    )

    return results


def save_analysis_tables(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Persist analysis tables to disk."""

    for table_name, table_df in tables.items():
        save_dataframe(table_df, output_dir / f"{table_name}.csv")


def generate_report(
    raw_rows: int,
    cleaned_rows: int,
    cleaning_steps: list[dict[str, int | str]],
    analysis_tables: dict[str, pd.DataFrame],
    score_columns: list[str],
) -> str:
    """Generate a human-readable report from pipeline outputs."""

    subject_stats = analysis_tables["subject_statistics"]
    difficulty = analysis_tables["difficulty_ranking"]
    discrimination = analysis_tables["discrimination_ranking"]
    top10 = analysis_tables["top_10_total_scores"]
    bottom10 = analysis_tables["bottom_10_total_scores"]
    correlations = analysis_tables["subject_correlations"]
    outliers = analysis_tables["outlier_analysis"]
    total_stats = analysis_tables["total_score_statistics"].iloc[0]

    lines: list[str] = []
    lines.append("BÁO CÁO PHÂN TÍCH ĐIỂM THI THPT QUỐC GIA 2026")
    lines.append("=" * 60)
    lines.append("")
    lines.append("1. TỔNG QUAN DỮ LIỆU")
    lines.append(f"- Số bản ghi ban đầu: {raw_rows}")
    lines.append(f"- Số bản ghi sau làm sạch: {cleaned_rows}")
    lines.append(f"- Tỷ lệ giữ lại: {cleaned_rows / raw_rows * 100:.2f}%")
    lines.append(f"- Các môn được phân tích: {', '.join(score_columns)}")
    lines.append("")
    lines.append("2. KẾT QUẢ LÀM SẠCH DỮ LIỆU")
    for step in cleaning_steps:
        lines.append(
            f"- {step['step']}: loại {step['removed_count']} bản ghi, còn {step['remaining_count']}."
        )
    lines.append("")
    lines.append("3. THỐNG KÊ MÔ TẢ")
    for _, row in subject_stats.iterrows():
        lines.append(
            f"- {row['subject']}: mean={row['mean']:.2f}, median={row['median']:.2f}, "
            f"std={row['std_dev']:.2f}, min={row['min']:.2f}, max={row['max']:.2f}"
        )
    lines.append("")
    lines.append("4. PHÂN TÍCH")
    lines.append(
        f"- Tổng điểm trung bình: {total_stats['mean']:.2f}, trung vị: {total_stats['median']:.2f}, "
        f"độ lệch chuẩn: {total_stats['std_dev']:.2f}."
    )
    lines.append(
        f"- Môn có điểm trung bình cao nhất: {subject_stats.sort_values('mean', ascending=False).iloc[0]['subject']}."
    )
    lines.append(
        f"- Môn khó nhất theo mean thấp nhất: {difficulty.iloc[0]['subject']}."
    )
    lines.append(
        f"- Môn phân hóa tốt nhất theo std cao nhất: {discrimination.iloc[0]['subject']}."
    )
    lines.append(
        f"- Số môn có outlier theo IQR được phát hiện ở từng môn: {', '.join(outliers['subject'].tolist())}."
    )
    lines.append("")
    lines.append("5. TƯƠNG QUAN")
    for _, row in correlations.iterrows():
        lines.append(
            f"- {row['base_subject']} và {row['other_subject']}: r={row['pearson_correlation']:.3f}"
        )
    lines.append("")
    lines.append("6. TOP 10 TỔNG ĐIỂM CAO NHẤT")
    lines.append(top10.to_string(index=False))
    lines.append("")
    lines.append("7. TOP 10 TỔNG ĐIỂM THẤP NHẤT")
    lines.append(bottom10.to_string(index=False))
    lines.append("")
    lines.append("8. NHẬN XÉT")
    lines.append(
        "- Dữ liệu sau làm sạch chỉ giữ các thí sinh có đúng 4 môn điểm hợp lệ và có đủ Toán, Ngữ văn."
    )
    lines.append(
        "- Điểm cao tập trung ở một số môn cụ thể, trong khi các môn có std lớn hơn cho thấy phân hoá mạnh hơn."
    )
    lines.append(
        "- Nên dùng các bảng và biểu đồ trong output/ để chèn trực tiếp vào slide hoặc báo cáo môn học."
    )
    lines.append("")
    lines.append("9. KẾT LUẬN")
    lines.append(
        "- Pipeline đã hoàn thành từ làm sạch, EDA, trực quan hoá đến phân tích thống kê và xuất báo cáo."
    )
    lines.append(
        "- Project sẵn sàng đưa lên GitHub hoặc dùng làm đồ án thực hành."
    )

    return "\n".join(lines)


def save_report(report_text: str, output_path: Path) -> None:
    """Save the final report as plain text."""

    save_text(report_text, output_path)
