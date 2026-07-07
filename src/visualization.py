"""Visualization routines for the score analysis project."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.plotting import scatter_matrix


def _save_current_figure(path: Path) -> None:
    """Save the active matplotlib figure and close it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def _apply_style() -> None:
    """Apply a consistent visualization theme."""

    sns.set_theme(style="whitegrid", palette="Set2")


def create_visualizations(df: pd.DataFrame, score_columns: list[str], output_dir: Path) -> None:
    """Generate all requested visualizations."""

    _apply_style()

    plot_df = df[score_columns].copy()
    total_score = plot_df.sum(axis=1)

    # Histogram each subject
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(22, 10))
    axes = axes.flatten()
    for axis, column in zip(axes, score_columns):
        sns.histplot(plot_df[column], bins=20, kde=True, ax=axis, color="#4C72B0")
        axis.set_title(f"Histogram {column}")
        axis.set_xlabel("Điểm")
        axis.set_ylabel("Số thí sinh")
    for axis in axes[len(score_columns) :]:
        axis.axis("off")
    _save_current_figure(output_dir / "histograms_subjects.png")

    # Histogram total score
    plt.figure(figsize=(12, 7))
    sns.histplot(total_score, bins=30, kde=True, color="#DD8452")
    plt.title("Histogram Tổng Điểm")
    plt.xlabel("Tổng điểm")
    plt.ylabel("Số thí sinh")
    _save_current_figure(output_dir / "histogram_total_score.png")

    # Boxplot
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=plot_df, orient="h")
    plt.title("Boxplot Điểm Các Môn")
    plt.xlabel("Điểm")
    plt.ylabel("Môn")
    _save_current_figure(output_dir / "boxplot_subjects.png")

    # Violin plot
    plt.figure(figsize=(12, 7))
    sns.violinplot(data=plot_df, orient="h", inner="quartile")
    plt.title("Violin Plot Điểm Các Môn")
    plt.xlabel("Điểm")
    plt.ylabel("Môn")
    _save_current_figure(output_dir / "violinplot_subjects.png")

    # KDE plot
    plt.figure(figsize=(12, 7))
    for column in score_columns:
        sns.kdeplot(plot_df[column], label=column, fill=False)
    plt.title("KDE Plot Điểm Các Môn")
    plt.xlabel("Điểm")
    plt.ylabel("Mật độ")
    plt.legend()
    _save_current_figure(output_dir / "kde_subjects.png")

    # Heatmap correlation
    plt.figure(figsize=(10, 8))
    sns.heatmap(plot_df.corr(), annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Heatmap Tương Quan")
    _save_current_figure(output_dir / "heatmap_correlation.png")

    # Scatter matrix
    sample_size = min(2000, len(plot_df))
    sample_df = plot_df.sample(n=sample_size, random_state=42) if sample_size else plot_df
    scatter_matrix(sample_df, figsize=(14, 14), diagonal="kde", alpha=0.35)
    plt.suptitle("Scatter Plot Matrix", y=1.02)
    _save_current_figure(output_dir / "scatter_matrix_scores.png")

    # Pair plot
    pairplot_df = sample_df.copy()
    pairplot_df["total_score"] = total_score.loc[pairplot_df.index]
    pair = sns.pairplot(pairplot_df, diag_kind="hist", corner=True, plot_kws={"alpha": 0.35})
    pair.fig.suptitle("Pair Plot Scores", y=1.02)
    pair.savefig(output_dir / "pairplot_scores.png", dpi=220, bbox_inches="tight")
    plt.close(pair.fig)

    # Bar chart of mean scores
    mean_scores = plot_df.mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=mean_scores.values,
        y=mean_scores.index,
        hue=mean_scores.index,
        dodge=False,
        legend=False,
        palette="viridis",
    )
    plt.title("Điểm Trung Bình Theo Môn")
    plt.xlabel("Điểm trung bình")
    plt.ylabel("Môn")
    _save_current_figure(output_dir / "bar_chart_mean_scores.png")

    # Bar chart by score bands for each subject
    bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    labels = ["0-1", "1-2", "2-3", "3-4", "4-5", "5-6", "6-7", "7-8", "8-9", "9-10"]
    band_counts = pd.DataFrame(index=labels)
    for column in score_columns:
        band_counts[column] = (
            pd.cut(plot_df[column], bins=bins, labels=labels, include_lowest=True, right=True)
            .value_counts()
            .sort_index()
        )
    band_counts = band_counts.fillna(0)
    band_counts.plot(kind="bar", figsize=(14, 8))
    plt.title("Số Lượng Thí Sinh Theo Từng Mức Điểm")
    plt.xlabel("Mức điểm")
    plt.ylabel("Số thí sinh")
    plt.legend(title="Môn")
    _save_current_figure(output_dir / "bar_chart_score_bands.png")

    # Total score distribution by bands
    total_bins = [0, 10, 15, 20, 25, 30, 35, 40]
    total_labels = ["0-10", "10-15", "15-20", "20-25", "25-30", "30-35", "35-40"]
    total_band_counts = (
        pd.cut(total_score, bins=total_bins, labels=total_labels, include_lowest=True, right=False)
        .value_counts()
        .sort_index()
    )
    plt.figure(figsize=(12, 6))
    sns.barplot(x=total_band_counts.index.astype(str), y=total_band_counts.values, color="#55A868")
    plt.title("Phân Bố Tổng Điểm Theo Mức")
    plt.xlabel("Khoảng tổng điểm")
    plt.ylabel("Số thí sinh")
    _save_current_figure(output_dir / "bar_chart_total_score_bands.png")

    # Total score distribution
    plt.figure(figsize=(12, 7))
    sns.histplot(total_score, bins=30, kde=True, color="#C44E52")
    plt.title("Phân Bố Tổng Điểm")
    plt.xlabel("Tổng điểm")
    plt.ylabel("Số thí sinh")
    _save_current_figure(output_dir / "distribution_total_score.png")
