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
python cluster_analysis/classes_diurnal_cycle.py --training-only
python cluster_analysis/classes_diurnal_cycle.py --mode testing
python cluster_analysis/classes_diurnal_cycle.py --mode both

Author: Claudia Acquistapace
Date: 10 sept 2025
Modified: 3 June 2026
"""

import argparse
import os
import sys
from typing import Dict, List
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sys.path.append("/home/claudia/codes/ML_postprocessing")

from utils.configs import load_config
from scripts.pretrain.cluster_analysis.var_class_temporal_series import read_csv_to_dataframe
from utils.plotting.class_colors import colors_per_class1_names, class_groups
from utils.plotting.plot_class_analysis import plot_hourly_histogram


# Add the repository root so top-level packages such as `utils` resolve
# regardless of the directory from which this script is launched.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)


# read filename of csv files from config file process_run_GRL.yaml
config_path = "/home/claudia/codes/ML_postprocessing/configs/process_run_GRL.yaml"
config = load_config(config_path)
CSV_FILES = {
    "training": config["output_files"]["training_video_summary"],
    "testing": config["output_files"]["testing_video_summary"],
}

MODE_ALIASES = {
    "both": "both",
    "train": "training",
    "training": "training",
    "test": "testing",
    "testing": "testing",
}

# create output directory if it doesn't exist
OUTPUT_DIR = Path(config["output_files"]["figures_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_FRAMES_PER_VIDEO = 8
DIURNAL_LINEWIDTH = 4.0
INVALID_LABELS = {-100}
HOUR_BIN_SIZE = 2
HOUR_BINS = list(range(0, 24, HOUR_BIN_SIZE))
HOUR_BIN_CENTERS = np.array(HOUR_BINS) + HOUR_BIN_SIZE / 2
HOUR_BIN_LABELS = [f"{hour:02d}-{hour + HOUR_BIN_SIZE:02d}" for hour in HOUR_BINS]
SINGLE_CLASS_YMAX = 0.3
EXPECTED_VIDEO_SUMMARY_COLUMNS = [
    "crop",
    "label",
    "time_start",
    "time_end",
    "lat_mid",
    "lon_mid",
    "cth_mean",
    "cth_std",
    "cth_gradient",
    "cma_mean",
    "cma_std",
    "cma_gradient",
    "cot_mean",
    "cot_std",
    "cot_gradient",
    "precipitation_mean",
    "precipitation_std",
    "precipitation_gradient",
    "euclid_msg_grid_mean",
    "euclid_msg_grid_std",
    "euclid_msg_grid_gradient",
    "cth10plus_mean",
    "cth10plus_std",
    "cth10plus_gradient",
    "cot30plus_mean",
    "cot30plus_std",
    "cot30plus_gradient",
]
REQUIRED_TIME_COLUMNS = {"label", "time_start", "time_end"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot class diurnal cycles from train/test video summaries.")
    parser.add_argument(
        "--mode",
        default="both",
        choices=sorted(MODE_ALIASES),
        help="Dataset split to plot: training/train, testing/test, or both.",
    )
    parser.add_argument(
        "--training-only",
        dest="mode",
        action="store_const",
        const="training",
        help="Plot only the training dataset diurnal cycles.",
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
    if not Path(csv_file).exists():
        raise FileNotFoundError(f"Missing {mode} crop statistics CSV: {csv_file}")
    return Path(csv_file)


def output_filename(filename: str, mode: str) -> Path:
    if mode in ("training", "both"):
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


def set_hour_bin_axis(ax):
    ax.set_xticks(HOUR_BIN_CENTERS)
    ax.set_xticklabels(HOUR_BIN_LABELS, rotation=45, ha="right")
    ax.set_xlim(0, 24)



def load_video_summary_dataframe(csv_file: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_file, low_memory=False)
    if REQUIRED_TIME_COLUMNS.issubset(df.columns):
        repeated_header_rows = df["time_start"].eq("time_start")
        if repeated_header_rows.any():
            print(
                f"Warning: dropped {repeated_header_rows.sum()} repeated CSV header row(s)."
            )
            df = df.loc[~repeated_header_rows].copy()
        return df

    headerless_df = pd.read_csv(csv_file, header=None)
    if headerless_df.shape[1] != len(EXPECTED_VIDEO_SUMMARY_COLUMNS):
        raise ValueError(
            f"CSV {csv_file} is missing required columns {sorted(REQUIRED_TIME_COLUMNS)} "
            f"and has {headerless_df.shape[1]} columns, expected "
            f"{len(EXPECTED_VIDEO_SUMMARY_COLUMNS)}."
        )

    headerless_df.columns = EXPECTED_VIDEO_SUMMARY_COLUMNS
    if REQUIRED_TIME_COLUMNS.issubset(headerless_df.columns):
        print(
            "Warning: CSV appears to be missing its header row. "
            "Recovered fixed columns by position."
        )
        return headerless_df

    raise ValueError(
        f"CSV {csv_file} is missing required columns {sorted(REQUIRED_TIME_COLUMNS)}. "
        f"Found columns: {df.columns.tolist()}"
    )


def main(mode: str = "both"):
    mode = normalize_mode(mode)

    if mode == "both":
        grouped_by_dataset = {
            dataset_name: load_hourly_occurrence(dataset_name)
            for dataset_name in ("training", "testing")
        }
        plot_all_classes_comparison(grouped_by_dataset)
        plot_single_classes_comparison(grouped_by_dataset)
        return

    df_grouped = load_hourly_occurrence(mode)

    plot_all_classes(df_grouped, mode)
    #plot_single_class_groups(df_grouped, mode)
    plot_single_classes(df_grouped, mode)


def load_hourly_occurrence(mode: str) -> pd.DataFrame:
    csv_file = resolve_csv_file(mode)

    print(f"Plotting diurnal cycle for {mode} dataset")
    print(f"Reading CSV: {csv_file}")

    # read the CSV into a DataFrame and print column titles
    video_df = load_video_summary_dataframe(csv_file)
    print("Column titles:", video_df.columns.tolist())

    # calculate mid hour timestamp between time_start and time_end timestamps columns of video_df and add as mid_time column timestamp
    video_df["time_start"] = pd.to_datetime(video_df["time_start"])
    video_df["time_end"] = pd.to_datetime(video_df["time_end"])

    video_df["time_mid"] = (
        video_df["time_start"]
        + (video_df["time_end"] - video_df["time_start"]) / 2
    )

    video_df["time_mid"] = video_df["time_mid"].dt.strftime("%Y-%m-%d %H:%M:%S")


    # each row is a video, group videos by hour and class label, then normalize occurrences
    df_grouped = build_hourly_occurrence(video_df)
    print(df_grouped.head())
    return df_grouped


def build_hourly_occurrence(df_times: pd.DataFrame) -> pd.DataFrame:
    df_local = df_times.copy()
    df_local["label"] = pd.to_numeric(df_local["label"], errors="coerce")
    df_local = df_local[~df_local["label"].isin(INVALID_LABELS)].copy()
    df_local = df_local.dropna(subset=["label", "time_mid"])
    df_local["label"] = df_local["label"].astype(int)

    time_mid = pd.to_datetime(df_local["time_mid"])
    hour_bin = (time_mid.dt.hour // HOUR_BIN_SIZE) * HOUR_BIN_SIZE
    df_grouped = df_local.groupby([hour_bin, "label"]).size().unstack(fill_value=0)
    df_grouped = df_grouped.reindex(HOUR_BINS, fill_value=0)
    return df_grouped.div(df_grouped.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)


def plot_all_classes(df_grouped: pd.DataFrame, mode: str):

    fig, ax = plt.subplots(figsize=(12, 7))
    hours = HOUR_BIN_CENTERS

    for label in df_grouped.columns:
        color = colors_per_class1_names.get(str(label), None)
        plot_hourly_histogram(ax, hours, df_grouped[label].to_numpy(), color, f"Class {label}", linewidth=DIURNAL_LINEWIDTH)

    ax.set_xlabel("Hour of the day", fontsize=16)
    ax.set_ylabel("Normalized occurrence", fontsize=16)
    ax.set_title("Occurrence of each class across the day", fontsize=16)
    set_hour_bin_axis(ax)
    ax.legend(frameon=False, fontsize=11, ncol=3)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(output_filename("class_occurrence_diurnal_cycle.png", mode), transparent=True)
    plt.close(fig)


def get_all_labels(grouped_by_dataset: Dict[str, pd.DataFrame]) -> List[int]:
    labels = set()
    for df_grouped in grouped_by_dataset.values():
        labels.update(df_grouped.columns)
    return sorted(labels)


def plot_all_classes_comparison(grouped_by_dataset: Dict[str, pd.DataFrame]):
    fig, ax = plt.subplots(figsize=(13, 7))

    for label in get_all_labels(grouped_by_dataset):
        color = colors_per_class1_names.get(str(label), None)
        for dataset_name, linestyle in (("testing", "-"), ("training", "--")):
            df_grouped = grouped_by_dataset[dataset_name]
            if label not in df_grouped.columns:
                continue
            line_label = f"Class {label} ({dataset_name})"
            ax.step(
                HOUR_BIN_CENTERS,
                df_grouped[label].to_numpy(),
                where="mid",
                color=color,
                linestyle=linestyle,
                linewidth=DIURNAL_LINEWIDTH if dataset_name == "testing" else DIURNAL_LINEWIDTH * 0.75,
                label=line_label,
            )

    ax.set_xlabel("Hour of the day", fontsize=16)
    ax.set_ylabel("Normalized occurrence", fontsize=16)
    ax.set_title("Class occurrence across the day: training vs testing", fontsize=16)
    set_hour_bin_axis(ax)
    ax.legend(frameon=False, fontsize=9, ncol=4)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "class_occurrence_diurnal_cycle_train_test.png", transparent=True)
    plt.close(fig)


def plot_single_classes(df_grouped: pd.DataFrame, mode: str):
    hours = HOUR_BIN_CENTERS

    for label in df_grouped.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_hourly_histogram(
            ax,
            hours,
            df_grouped[label].to_numpy(),
            colors_per_class1_names.get(str(label), None),
            f"Class {label}",
            linewidth=DIURNAL_LINEWIDTH
        )
        ax.set_xlabel("Hour of the day [hh]", fontsize=16)
        ax.set_ylabel("Normalized occurrence", fontsize=16)
        ax.set_title(f"Occurrence of Class {label} across the day", fontsize=16)
        set_hour_bin_axis(ax)
        ax.set_ylim(0, get_single_class_ymax(label))
        style_axis(ax)
        fig.tight_layout()
        fig.savefig(output_filename(f"class_{label}_diurnal_cycle.png", mode), transparent=True)
        plt.close(fig)


def get_single_class_ymax(label) -> float:
    return SINGLE_CLASS_YMAX


def plot_single_classes_comparison(grouped_by_dataset: Dict[str, pd.DataFrame]):
    for label in get_all_labels(grouped_by_dataset):
        fig, ax = plt.subplots(figsize=(8, 5))
        color = colors_per_class1_names.get(str(label), None)
        has_data = False

        for dataset_name, linestyle in (("testing", "-"), ("training", "--")):
            df_grouped = grouped_by_dataset[dataset_name]
            if label not in df_grouped.columns:
                continue
            has_data = True
            ax.step(
                HOUR_BIN_CENTERS,
                df_grouped[label].to_numpy(),
                where="mid",
                color=color,
                linestyle=linestyle,
                linewidth=DIURNAL_LINEWIDTH if dataset_name == "testing" else DIURNAL_LINEWIDTH * 0.75,
                label=dataset_name,
            )

        if not has_data:
            plt.close(fig)
            continue

        ax.set_xlabel("Hour of the day [hh]", fontsize=16)
        ax.set_ylabel("Normalized occurrence", fontsize=16)
        ax.set_title(f"Occurrence of Class {label} across the day", fontsize=16)
        set_hour_bin_axis(ax)
        ax.set_ylim(0, get_single_class_ymax(label))
        ax.legend(frameon=False, fontsize=11)
        style_axis(ax)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"class_{label}_diurnal_cycle_train_test.png", transparent=True)
        plt.close(fig)


def plot_single_class_groups(df_grouped: pd.DataFrame, mode: str):
    hours = HOUR_BIN_CENTERS

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
                linewidth=DIURNAL_LINEWIDTH
            )

        if not has_any_label:
            plt.close(fig)
            continue

        ax.set_xlabel("Hour of the day [hh]", fontsize=16)
        ax.set_ylabel("Normalized occurrence", fontsize=16)
        ax.set_title(f"Occurrence of {group_name} classes across the day", fontsize=16)
        set_hour_bin_axis(ax)
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
    config_path = "/home/claudia/codes/ML_postprocessing/configs/process_run_GRL.yaml"
    config = load_config(config_path) 

    args = parse_args()
    main(mode=args.mode)
