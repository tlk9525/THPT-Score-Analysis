"""Statistical analysis and report generation."""

from __future__ import annotations

from pathlib import Path
from html import escape

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
        "- Dữ liệu sau làm sạch giữ các bản ghi hợp lệ, loại trùng lặp, loại điểm ngoài khoảng 0-10 và yêu cầu có đủ Toán, Ngữ văn."
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


def _table_to_html(title: str, table: pd.DataFrame) -> str:
    """Render a dataframe as a titled HTML section."""

    return (
        f"<section class='card'>"
        f"<h2>{escape(title)}</h2>"
        f"{table.to_html(index=False, border=0, classes='data-table')}"
        f"</section>"
    )


def generate_html_report(
    raw_rows: int,
    cleaned_rows: int,
    cleaning_steps: list[dict[str, int | str]],
    analysis_tables: dict[str, pd.DataFrame],
    score_columns: list[str],
    figures_dir: Path | None = None,
) -> str:
    """Generate a browser-friendly HTML report."""

    subject_stats = analysis_tables["subject_statistics"]
    difficulty = analysis_tables["difficulty_ranking"]
    discrimination = analysis_tables["discrimination_ranking"]
    top10 = analysis_tables["top_10_total_scores"]
    bottom10 = analysis_tables["bottom_10_total_scores"]
    correlations = analysis_tables["subject_correlations"]
    outliers = analysis_tables["outlier_analysis"]
    total_stats = analysis_tables["total_score_statistics"].iloc[0]

    figure_items = []
    if figures_dir and figures_dir.exists():
        for figure_path in sorted(figures_dir.glob("*.png")):
            figure_items.append(
                f"<li><a href='../figures/{escape(figure_path.name)}'>{escape(figure_path.name)}</a></li>"
            )

    cleaning_rows = "".join(
        f"<tr><td>{escape(str(step['step']))}</td><td>{step['removed_count']}</td><td>{step['remaining_count']}</td></tr>"
        for step in cleaning_steps
    )

    correlation_rows = "".join(
        f"<tr><td>{escape(str(row['base_subject']))}</td><td>{escape(str(row['other_subject']))}</td><td>{row['pearson_correlation']:.3f}</td></tr>"
        for _, row in correlations.iterrows()
    )

    html = f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>THPT Score Analysis</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0b1020;
      --panel: #121a2f;
      --panel-2: #18233f;
      --text: #e8edf9;
      --muted: #b7c1da;
      --accent: #7dd3fc;
      --line: #2a385d;
    }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
      background: linear-gradient(180deg, #08101f, #0b1020 40%, #101934);
      color: var(--text);
    }}
    .wrap {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 32px 20px 56px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(125, 211, 252, 0.14), rgba(99, 102, 241, 0.12));
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 28px;
      margin-bottom: 20px;
    }}
    .hero h1 {{
      margin: 0 0 8px;
      font-size: 34px;
    }}
    .hero p {{
      margin: 6px 0;
      color: var(--muted);
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin: 20px 0;
    }}
    .stat {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px;
    }}
    .stat .label {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 6px;
    }}
    .stat .value {{
      font-size: 24px;
      font-weight: 700;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 18px;
      margin-top: 18px;
      overflow: auto;
    }}
    h2 {{
      margin-top: 0;
      margin-bottom: 12px;
      font-size: 22px;
    }}
    table.data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    .data-table th, .data-table td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
    }}
    .data-table th {{
      background: var(--panel-2);
      color: #fff;
    }}
    ul {{
      margin: 0;
      padding-left: 20px;
      color: var(--muted);
    }}
    .muted {{
      color: var(--muted);
    }}
    .grid-2 {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <h1>THPT Score Analysis</h1>
      <p>HTML report cho phân tích điểm thi THPTQG.</p>
      <p class="muted">Môn phân tích: {escape(", ".join(score_columns))}</p>
    </div>

    <div class="stats">
      <div class="stat"><div class="label">Bản ghi gốc</div><div class="value">{raw_rows}</div></div>
      <div class="stat"><div class="label">Bản ghi sau làm sạch</div><div class="value">{cleaned_rows}</div></div>
      <div class="stat"><div class="label">Tỷ lệ giữ lại</div><div class="value">{cleaned_rows / raw_rows * 100:.2f}%</div></div>
      <div class="stat"><div class="label">Điểm TB toàn bộ</div><div class="value">{total_stats['mean']:.2f}</div></div>
    </div>

    <div class="grid-2">
      <section class="card">
        <h2>Kết quả làm sạch</h2>
        <table class="data-table">
          <thead><tr><th>Bước</th><th>Loại</th><th>Còn lại</th></tr></thead>
          <tbody>{cleaning_rows}</tbody>
        </table>
      </section>
      <section class="card">
        <h2>Biểu đồ</h2>
        <ul>{''.join(figure_items) if figure_items else '<li>Không tìm thấy file biểu đồ.</li>'}</ul>
      </section>
    </div>

    {_table_to_html("Thống kê theo môn", subject_stats)}
    {_table_to_html("Xếp hạng môn khó", difficulty)}
    {_table_to_html("Xếp hạng môn phân hóa", discrimination)}
    {_table_to_html("Top 10 tổng điểm cao nhất", top10)}
    {_table_to_html("Top 10 tổng điểm thấp nhất", bottom10)}

    <section class="card">
      <h2>Tương quan</h2>
      <table class="data-table">
        <thead><tr><th>Môn gốc</th><th>Môn so sánh</th><th>Pearson r</th></tr></thead>
        <tbody>{correlation_rows}</tbody>
      </table>
    </section>

    <section class="card">
      <h2>Outlier theo IQR</h2>
      {outliers.to_html(index=False, border=0, classes='data-table')}
    </section>
  </div>
</body>
</html>"""
    return html


def save_html_report(html_text: str, output_path: Path) -> None:
    """Save the final report as HTML."""

    save_text(html_text, output_path)
