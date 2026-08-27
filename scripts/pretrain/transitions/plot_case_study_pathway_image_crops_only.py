import argparse
import glob
import os
from datetime import datetime

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import sys

sys.path.append("/home/Daniele/codes/VISSL_postprocessing/")
from utils.plotting.class_colors import CLOUD_CLASS_INFO


mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.usetex": False,
    "font.size": 18,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "axes.linewidth": 1.2,
})


RUN_NAME = "dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc"
BASE_DIR = f"/data1/fig/{RUN_NAME}/epoch_800/test_traj"
OUT_DIR = f"{BASE_DIR}/pathway_analysis"
DF_PATH = os.path.join(OUT_DIR, "df_pathways_merged_no_dominance.csv")
CROP_DIR = "/data1/crops/test_case_essl_14-15-16-18-19-20-22-23-24_100x100_ir108_cma_traj/images/IR_108/png_vmin-vmax_greyscale_CMA"

DEFAULT_PATHWAY_RANKS = [2, 1]
DEFAULT_CASE_RANK = 1
DEFAULT_N_CROPS = 5
DEFAULT_TIME_MIN = -6.0
DEFAULT_TIME_MAX = 6.0
DEFAULT_TIME_BINS = [-6.0, -3.0, 0.0, 3.0, 6.0]
TITLE_FONTSIZE = 18
TIME_BIN_FONTSIZE = 18

PATHWAY_DISPLAY_NAMES = {
    20: "DC persistence",
    1: "EC->DC",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot crop-only case studies for the most common pathways.")
    parser.add_argument(
        "--pathway-ranks",
        type=int,
        nargs="+",
        default=DEFAULT_PATHWAY_RANKS,
        help="1-based ranks in the pathway-frequency table to plot. Default: 1 2",
    )
    parser.add_argument(
        "--case-rank",
        type=int,
        default=DEFAULT_CASE_RANK,
        help="1-based rank of the storm case inside each selected pathway. Lower values pick stronger hail/rain events first.",
    )
    parser.add_argument(
        "--n-crops",
        type=int,
        default=DEFAULT_N_CROPS,
        help="Number of crops to show in each row. Default: 5",
    )
    parser.add_argument(
        "--time-min",
        type=float,
        default=DEFAULT_TIME_MIN,
        help="Lower time bound in hours relative to t_aligned. Default: -6",
    )
    parser.add_argument(
        "--time-max",
        type=float,
        default=DEFAULT_TIME_MAX,
        help="Upper time bound in hours relative to t_aligned. Default: 6",
    )
    parser.add_argument(
        "--time-bins",
        type=float,
        nargs="+",
        default=DEFAULT_TIME_BINS,
        help="Shared target aligned-time bins for both pathways. Default: -6 -3 0 3 6",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(OUT_DIR, "case_study_pathway_crops_only_no_dom.png"),
        help="Output image path.",
    )
    return parser.parse_args()


def normalize_storm_id(storm_id):
    storm_str = str(storm_id)
    if storm_str.startswith("storm"):
        storm_str = storm_str.replace("storm", "", 1)
    try:
        return int(storm_str)
    except ValueError:
        return storm_str


def parse_crop_datetime(file_path):
    basename = os.path.basename(file_path)
    parts = basename.split("_")
    if len(parts) < 2:
        return pd.NaT

    dt_str = parts[1]
    parsed = pd.to_datetime(dt_str, errors="coerce")
    if pd.notna(parsed):
        return parsed.to_pydatetime()

    for fmt in ("%Y-%m-%dT%H-%M", "%Y%m%dT%H%M%S", "%Y%m%dT%H%M"):
        try:
            return datetime.strptime(dt_str.replace(":", "-"), fmt)
        except ValueError:
            continue

    return pd.NaT


def compute_pathway_ranking(df):
    ranking = (
        df.groupby(["pathway_id", "pathway"])["storm_id"]
        .nunique()
        .reset_index(name="n_storms")
        .sort_values(["n_storms", "pathway_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    return ranking


def select_storm_case(df_pw, pathway_id, case_rank=1):
    storm_summary = (
        df_pw.groupby("storm_id")
        .agg(
            max_hail_intensity=("max_hail_intensity", "max"),
            precipitation99=("precipitation99", "max"),
            datetime=("datetime", "min"),
        )
        .reset_index()
    )
    if storm_summary.empty:
        return None, storm_summary

    if int(pathway_id) == 20:
        sort_cols = ["max_hail_intensity", "precipitation99"]
    else:
        storm_summary["selection_score"] = storm_summary[["max_hail_intensity", "precipitation99"]].fillna(0).max(axis=1)
        sort_cols = ["selection_score", "max_hail_intensity", "precipitation99"]

    storm_summary = storm_summary.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)

    idx = min(max(case_rank, 1) - 1, len(storm_summary) - 1)
    selected = storm_summary.iloc[idx]
    return selected, storm_summary


def extract_image_records(trajectory_df, crop_dir, time_min=-6.0, time_max=6.0):
    records = []
    trajectory_df = trajectory_df.copy()
    trajectory_df["datetime"] = pd.to_datetime(trajectory_df["datetime"], errors="coerce", utc=True).dt.tz_convert(None)
    trajectory_df = trajectory_df.dropna(subset=["datetime"]).sort_values("datetime")

    if trajectory_df.empty:
        return records

    storm_id = normalize_storm_id(trajectory_df["storm_id"].iloc[0])
    search_pattern = os.path.join(crop_dir, f"storm{storm_id}_*_*.png")

    for file_path in glob.glob(search_pattern):
        img_dt = parse_crop_datetime(file_path)
        if pd.isna(img_dt):
            continue

        img_dt = pd.Timestamp(img_dt).tz_localize(None)
        time_diffs = (trajectory_df["datetime"] - img_dt).abs()
        if time_diffs.empty:
            continue

        matching_idx = time_diffs.idxmin()
        row = trajectory_df.loc[matching_idx]
        t_aligned = float(row.get("t_aligned", np.nan))
        if not np.isfinite(t_aligned):
            continue

        if not (time_min < t_aligned < time_max):
            continue

        state_color = "black"
        try:
            state_val = int(row["state"])
            state_color = CLOUD_CLASS_INFO.get(state_val, {}).get("color", "black")
        except Exception:
            pass

        records.append(
            {
                "file_path": file_path,
                "img_dt": img_dt,
                "t_aligned": t_aligned,
                "state_color": state_color,
            }
        )

    records.sort(key=lambda item: item["t_aligned"])
    return records


def select_records_for_bins(records, target_bins):
    selected = []
    remaining = list(records)

    for target in target_bins:
        if not remaining:
            break

        best_idx = min(
            range(len(remaining)),
            key=lambda idx: abs(remaining[idx]["t_aligned"] - target),
        )
        selected.append(remaining.pop(best_idx))

    return selected


def plot_crop_row(ax, trajectory_df, crop_dir, target_bins, time_min=-6.0, time_max=6.0, gap_pixels=18):
    ax.clear()

    records = extract_image_records(trajectory_df, crop_dir, time_min=time_min, time_max=time_max)
    if not records:
        ax.text(0.5, 0.5, "No crops in window", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    records = select_records_for_bins(records, target_bins)

    images = []
    labels = []
    colors = []
    for record in records:
        try:
            image = np.array(Image.open(record["file_path"]))
        except Exception:
            continue
        images.append(image)
        labels.append(f"{record['t_aligned']:+.1f} h")
        colors.append(record["state_color"])

    if not images:
        ax.text(0.5, 0.5, "No crops loaded", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    height, width = images[0].shape[:2]
    total_width = width * len(images) + gap_pixels * (len(images) - 1)
    composite = np.full((height, total_width), 255, dtype=images[0].dtype)

    current_x = 0
    for image in images:
        composite[:, current_x:current_x + width] = image[:, :width]
        current_x += width + gap_pixels

    ax.imshow(composite, cmap="gray", aspect="auto")

    current_x = 0
    tick_positions = []
    for color in colors:
        ax.add_patch(
            mpatches.Rectangle(
                (current_x, 0),
                width,
                height,
                fill=False,
                linewidth=5,
                edgecolor=color,
            )
        )
        tick_positions.append(current_x + width / 2)
        current_x += width + gap_pixels

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(labels, fontsize=TIME_BIN_FONTSIZE)
    ax.set_yticks([])


def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(DF_PATH, low_memory=False)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True).dt.tz_convert(None)
    df = df.dropna(subset=["datetime"])

    if "t_aligned" not in df.columns:
        df = df.copy()
        df["t_aligned"] = np.nans

    pathway_ranking = compute_pathway_ranking(df)
    print("Pathway ranking by storm count:")
    print(pathway_ranking[["rank", "pathway_id", "pathway", "n_storms"]].to_string(index=False))

    selected_paths = []
    for rank in args.pathway_ranks:
        row = pathway_ranking[pathway_ranking["rank"] == rank]
        if row.empty:
            print(f"Skipping missing pathway rank {rank}")
            continue
        selected_paths.append(row.iloc[0])

    if not selected_paths:
        raise ValueError("No pathway ranks selected.")

    fig, axes = plt.subplots(
        len(selected_paths),
        1,
        figsize=(12, 2.5 * len(selected_paths)),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    for row_idx, (ax, path_row) in enumerate(zip(axes, selected_paths)):
        pathway_id = int(path_row["pathway_id"])
        pathway_name = str(path_row["pathway"])
        display_name = PATHWAY_DISPLAY_NAMES.get(pathway_id, pathway_name)

        df_pw = df[df["pathway_id"] == pathway_id].copy()
        selected_storm, storm_summary = select_storm_case(df_pw, pathway_id=pathway_id, case_rank=args.case_rank)
        if selected_storm is None:
            ax.text(0.5, 0.5, "No storm cases found", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        storm_id = selected_storm["storm_id"]
        storm_id_for_crop = normalize_storm_id(storm_id)
        df_case = df_pw[df_pw["storm_id"] == storm_id].copy()

        if "t_aligned" not in df_case.columns or df_case["t_aligned"].isna().all():
            df_case = df_case.sort_values("datetime").copy()
            mid_dt = df_case["datetime"].iloc[len(df_case) // 2]
            df_case["t_aligned"] = (df_case["datetime"] - mid_dt).dt.total_seconds() / 3600.0

        plot_crop_row(
            ax,
            df_case,
            crop_dir=CROP_DIR,
            target_bins=args.time_bins,
            time_min=args.time_min,
            time_max=args.time_max,
        )

        storm_dt = pd.to_datetime(selected_storm["datetime"]).strftime("%Y-%m-%d")
        ax.set_title(
            f"{chr(ord('a') + row_idx)}) {display_name} | {storm_dt} | "
            f"Max hail diameter: {selected_storm['max_hail_intensity']:.2f} cm | Max rain rate: {selected_storm['precipitation99']:.2f} mm/h",
            fontsize=TITLE_FONTSIZE,
            fontweight="bold",
        )
        for spine in ax.spines.values():
            spine.set_color("white")

        print(
            f"Selected {display_name}: pathway_id={pathway_id}, storm_id={storm_id_for_crop}, "
            f"rank={args.case_rank}"
        )
        print(storm_summary.head(args.case_rank + 2).to_string(index=False))

    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()