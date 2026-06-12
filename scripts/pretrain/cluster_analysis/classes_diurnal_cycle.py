"""
Plot the diurnal cycle of class occurrence for the grl_2026 video dataset.

Purpose:
This script computes how often each class occurs at each hour of the day.
It uses the cth crop-statistics CSV only as a source of:
- the video identifier (`crop`)
- the class label (`label`)
- the timestamp of each frame (`time`)

Input files:
- training:
  /sat_data/output/grl_2026/csv/crops_stats_var-cth_stats-50-95-25-75_frames-8_timedim_grl_2026_all_240216_imergmin.csv
- testing:
  /sat_data/output/grl_2026/csv/crops_stats_var-cth_stats-50-95-25-75_frames-8_timedim_grl_2026_test_all_7045_imergmin.csv

How the calculation works:
1. Read the cth CSV.
2. Group rows by `crop`, where each crop represents one 8-frame video.
3. Keep only videos with exactly 8 frames.
4. Compute the mean timestamp of the 8 frames to assign one representative time to each video.
5. Extract the hour of day from that mean timestamp.
6. Count class occurrences for each hour.
7. Normalize occurrences within each hour so the class fractions at a given hour sum to 1.

Plot style:
- hour-binned histogram-style diurnal cycles
- thick solid step lines
- class colors taken from `utils.plotting.class_colors`

Output directory:
/sat_data/output/grl_2026/figs/

Generated outputs:
- overall class plot:
    class_occurrence_diurnal_cycle.png
- class-group plots with member classes:
    {group_name}_diurnal_cycle.png
- single-class plots:
    class_{label}_diurnal_cycle.png

Example call:
python cluster_analysis/classes_diurnal_cycle.py --mode training
python cluster_analysis/classes_diurnal_cycle.py --mode testing

Author: Claudia Acquistapace
Date: 10 sept 2025
Modified: 3 June 2026
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add the repository root so top-level packages such as `utils` resolve
# regardless of the directory from which this script is launched.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from scripts.pretrain.cluster_analysis.var_class_temporal_series import read_csv_to_dataframe
from utils.plotting.class_colors import colors_per_class1_names, class_groups

CSV_FILES = {
    "training": Path("/sat_data/output/grl_2026/csv/crops_stats_var-cth_stats-50-95-25-75_frames-8_timedim_grl_2026_all_240216_imergmin.csv"),
    "testing": Path("/sat_data/output/grl_2026/csv/crops_stats_var-cth_stats-50-95-25-75_frames-8_timedim_grl_2026_test_all_7045_imergmin.csv"),
}
MODE_ALIASES = {
    "train": "training",
    "training": "training",
    "test": "testing",
    "testing": "testing",
}
OUTPUT_DIR = Path("/sat_data/output/grl_2026/figs")
EXPECTED_FRAMES_PER_VIDEO = 8
DIURNAL_LINEWIDTH = 4.0
INVALID_LABELS = {-100}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot class diurnal cycles from train or test crop stats.")
    parser.add_argument(
        "--mode",
        default="training",
        choices=sorted(MODE_ALIASES),
        help="Dataset split to plot: training/train or testing/test.",
    )
    return parser.parse_args()


def normalize_mode(mode: str) -> str:
    try:
        return MODE_ALIASES[mode.lower()]
    except KeyError as exc:
        valid_modes = ", ".join(sorted(MODE_ALIASES))
        raise ValueError(f"Invalid mode '{mode}'. Expected one of: {valid_modes}") from exc


def resolve_csv_file(mode: str) -> Path:
    csv_file = CSV_FILES[mode]
    if not csv_file.exists():
        raise FileNotFoundError(f"Missing {mode} crop statistics CSV: {csv_file}")
    return csv_file


def output_filename(filename: str, mode: str) -> Path:
    if mode == "training":
        return OUTPUT_DIR / filename

    stem, ext = os.path.splitext(filename)
    return OUTPUT_DIR / f"{stem}_{mode}{ext}"


def style_axis(ax):
    ax.grid(color="lightgray", linestyle="--", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.tick_params(width=1.5, length=7)


def plot_hourly_histogram(ax, hours, values, color, label):
    line_color = color if color is not None else "C0"
    ax.step(
        hours,
        values,
        where="mid",
        color=line_color,
        linewidth=DIURNAL_LINEWIDTH,
        label=label,
    )


def main(mode: str = "training"):
    mode = normalize_mode(mode)
    csv_file = resolve_csv_file(mode)

    print(f"Plotting diurnal cycle for {mode} dataset")
    print(f"Reading CSV: {csv_file}")

    df = read_csv_to_dataframe(str(csv_file))
    print("Column titles:", df.columns.tolist())

    video_df = derive_time_class(df)
    video_df = video_df[~video_df["label"].isin(INVALID_LABELS)].copy()
    print("Number of videos with 8 frames:", len(video_df))
    print(video_df.head())

    df_grouped = build_hourly_occurrence(video_df)
    print(df_grouped.head())

    plot_all_classes(df_grouped, mode)
    plot_single_class_groups(df_grouped, mode)
    plot_single_classes(df_grouped, mode)


def build_hourly_occurrence(df_times: pd.DataFrame) -> pd.DataFrame:
    df_grouped = df_times.groupby(["hour", "label"]).size().unstack(fill_value=0)
    df_grouped = df_grouped.reindex(range(24), fill_value=0)
    return df_grouped.div(df_grouped.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)


def plot_all_classes(df_grouped: pd.DataFrame, mode: str):
    fig, ax = plt.subplots(figsize=(12, 7))
    hours = df_grouped.index.to_numpy()

    for label in df_grouped.columns:
        color = colors_per_class1_names.get(str(label), None)
        plot_hourly_histogram(ax, hours, df_grouped[label].to_numpy(), color, f"Class {label}")

    ax.set_xlabel("Hour of the day", fontsize=16)
    ax.set_ylabel("Normalized occurrence", fontsize=16)
    ax.set_title("Occurrence of each class across the day", fontsize=16)
    ax.set_xticks(range(0, 24, 2))
    ax.legend(frameon=False, fontsize=11, ncol=3)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(output_filename("class_occurrence_diurnal_cycle.png", mode), transparent=True)
    plt.close(fig)


def plot_single_classes(df_grouped: pd.DataFrame, mode: str):
    hours = df_grouped.index.to_numpy()

    for label in df_grouped.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_hourly_histogram(
            ax,
            hours,
            df_grouped[label].to_numpy(),
            colors_per_class1_names.get(str(label), None),
            f"Class {label}",
        )
        ax.set_xlabel("Hour of the day [hh]", fontsize=16)
        ax.set_ylabel("Normalized occurrence", fontsize=16)
        ax.set_title(f"Occurrence of Class {label} across the day", fontsize=16)
        ax.set_xticks(range(24))
        style_axis(ax)
        fig.tight_layout()
        fig.savefig(output_filename(f"class_{label}_diurnal_cycle.png", mode), transparent=True)
        plt.close(fig)


def plot_single_class_groups(df_grouped: pd.DataFrame, mode: str):
    hours = df_grouped.index.to_numpy()

    for group_name, group_labels in class_groups.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        has_any_label = False

        for label in group_labels:
            if label not in df_grouped.columns:
                continue
            has_any_label = True
            plot_hourly_histogram(
                ax,
                hours,
                df_grouped[label].to_numpy(),
                colors_per_class1_names.get(str(label), None),
                f"Class {label}",
            )

        if not has_any_label:
            plt.close(fig)
            continue

        ax.set_xlabel("Hour of the day [hh]", fontsize=16)
        ax.set_ylabel("Normalized occurrence", fontsize=16)
        ax.set_title(f"Occurrence of {group_name} classes across the day", fontsize=16)
        ax.set_xticks(range(24))
        ax.legend(frameon=False, fontsize=11)
        style_axis(ax)
        fig.tight_layout()
        fig.savefig(output_filename(f"{group_name}_diurnal_cycle.png", mode), transparent=True)
        plt.close(fig)


def derive_time_class(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per video with mean time, label, and hour."""
    df_local = df.copy()
    df_local["time"] = pd.to_datetime(df_local["time"])

    grouped = df_local.groupby("crop")
    video_df = grouped.agg(
        n_frames=("time", "size"),
        time=("time", "mean"),
        label=("label", "first"),
    )

    video_df = video_df[video_df["n_frames"] == EXPECTED_FRAMES_PER_VIDEO].copy()
    video_df["hour"] = video_df["time"].dt.hour
    return video_df.reset_index(drop=True)

if __name__ == "__main__":
    args = parse_args()
    main(mode=args.mode)
