import argparse
import os
import re
import sys

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cmcrameri import cm as cmc
from matplotlib.ticker import FixedLocator
from matplotlib.ticker import FuncFormatter


# ==================================================
# CONFIGURATION
# ==================================================
RUN_NAME = "dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc"
OUTPUT_PATH = f"/data1/fig/{RUN_NAME}/epoch_800/test_traj"
TRAINING_CSV_DEFAULT = (
    "/data1/fig/"
    "dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc/"
    "epoch_800/all/merged_crops_stats_cvc_imergtime_closest_1000.csv"
)
TRAINING_CSV_FALLBACK = (
    "/data1/fig/"
    "dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc/"
    "epoch_800/all/merged_crops_stats_cvc_imergtime_all_with_bt_area_precip.csv"
)

OROG_PATH = "/data1/DEM_EXPATS_0.01x0.01.nc"
MAP_EXTENT = [5, 16, 42, 51.5]
MARGIN_DEG = 2.0
BIN_DEG = 0.2
UTC_TIME_RANGES = [(0, 24)]

ROW_LAYOUT = [
    ["CS", "SC", None],
    ["MC1", "MC2", None],
    ["EC", "DC", "OA"],
]

# Typography
SUPTITLE_FONTSIZE = 27
PANEL_TITLE_FONTSIZE = 20
AXIS_TICK_FONTSIZE = 16
GRID_LABEL_FONTSIZE = 18
CBAR_LABEL_FONTSIZE = 20
CBAR_TICK_FONTSIZE = 20
CBAR_VMIN = 0.0
CBAR_VMAX = 2e-3

# Make occurrence layer more transparent so orography remains visible.
OCCURRENCE_ALPHA = 0.55

REDUCED_EXTENT = [
    MAP_EXTENT[0] + MARGIN_DEG,
    MAP_EXTENT[1] - MARGIN_DEG,
    MAP_EXTENT[2] + MARGIN_DEG,
    MAP_EXTENT[3] - MARGIN_DEG,
]


# ==================================================
# IMPORT PROJECT UTILITIES
# ==================================================
sys.path.append("/home/Daniele/codes/VISSL_postprocessing/scripts/pretrain/")
from transitions.plot_utils import plot_orography_map  # noqa: E402

sys.path.append("/home/Daniele/codes/VISSL_postprocessing/")
from utils.plotting.class_colors import CLOUD_CLASS_INFO  # noqa: E402


FILENAME_UTC_PATTERNS = [
    re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})T(?P<hour>\d{2})[:-](?P<minute>\d{2})(?::(?P<second>\d{2}))?"),
    re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})_(?P<hour>\d{2})[:-](?P<minute>\d{2})(?::(?P<second>\d{2}))?"),
]

SHORT_TO_LABEL = {info["short"]: cid for cid, info in CLOUD_CLASS_INFO.items()}
CLASS_ORDER = ["CS", "SC", "MC1", "MC2", "EC", "DC", "OA"]


def resolve_training_csv(training_csv):
    """Use provided training CSV, with fallback if the path does not exist."""
    if os.path.exists(training_csv):
        return training_csv
    if os.path.exists(TRAINING_CSV_FALLBACK):
        print(
            f"Requested training CSV not found: {training_csv}. "
            f"Using fallback: {TRAINING_CSV_FALLBACK}"
        )
        return TRAINING_CSV_FALLBACK
    raise FileNotFoundError(
        "Could not find training CSV. Checked: "
        f"{training_csv} and {TRAINING_CSV_FALLBACK}"
    )


def extract_utc_hour_from_filename(crop_path):
    """Extract the UTC hour from the crop filename."""
    crop_name = os.path.basename(str(crop_path))

    for pattern in FILENAME_UTC_PATTERNS:
        match = pattern.search(crop_name)
        if match:
            return float(match.group("hour"))

    raise ValueError(f"Could not extract UTC hour from crop filename: {crop_name}")


def add_utc_hour_column(df):
    """Attach a utc_hour column extracted from each crop filename."""
    out = df.copy()
    out["utc_hour"] = out["crop"].apply(extract_utc_hour_from_filename)
    return out


def filter_by_utc_range(df, utc_range):
    """Keep only rows inside a half-open UTC hour interval [start, end)."""
    start_hour, end_hour = utc_range
    if "utc_hour" not in df.columns:
        raise ValueError("Dataframe must contain a utc_hour column before filtering")

    filtered = df[pd.notna(df["utc_hour"])].copy()
    return filtered[(filtered["utc_hour"] >= start_hour) & (filtered["utc_hour"] < end_hour)]


def format_utc_range_label(utc_range):
    """Return a compact label such as 06-12 UTC."""
    start_hour, end_hour = utc_range
    return f"{start_hour:02d}-{end_hour:02d} UTC"


def build_time_range_output_file(output_file, utc_range):
    """Insert the UTC range into the output filename before the extension."""
    base_name, extension = os.path.splitext(output_file)
    start_hour, end_hour = utc_range
    return f"{base_name}_utc_{start_hour:02d}_{end_hour:02d}{extension}"


def prepare_training_occurrence_df(training_csv):
    """Prepare training dataframe using lat_mid/lon_mid and label columns."""
    df = pd.read_csv(training_csv, low_memory=False)

    required_cols = ["crop", "label", "lat_mid", "lon_mid"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            "Training CSV missing required columns for this workflow: "
            f"{missing}. Expected: crop, label, lat_mid, lon_mid"
        )

    out = df[["crop", "label", "lat_mid", "lon_mid"]].copy()
    out = out.rename(columns={"lat_mid": "center_lat", "lon_mid": "center_lon"})
    out["label"] = pd.to_numeric(out["label"], errors="coerce")
    out["center_lat"] = pd.to_numeric(out["center_lat"], errors="coerce")
    out["center_lon"] = pd.to_numeric(out["center_lon"], errors="coerce")
    out = out.dropna(subset=["label", "center_lat", "center_lon"])
    out = out.drop_duplicates(subset=["crop"])
    out["label"] = out["label"].astype(int)
    out = add_utc_hour_column(out)

    return out[["crop", "label", "center_lat", "center_lon", "utc_hour"]]


def compute_occurrence_histogram(df, class_short, extent, bin_deg, denom_class):
    """Compute normalized 2D histogram (frequency) for a class short name."""
    if class_short not in SHORT_TO_LABEL:
        raise ValueError(f"Class short '{class_short}' not found in CLOUD_CLASS_INFO")

    label_id = SHORT_TO_LABEL[class_short]
    sel = df[df["label"] == label_id]
    sel = sel[
        (sel["center_lon"] >= extent[0])
        & (sel["center_lon"] <= extent[1])
        & (sel["center_lat"] >= extent[2])
        & (sel["center_lat"] <= extent[3])
    ]

    lon_edges = np.arange(extent[0], extent[1] + bin_deg, bin_deg)
    lat_edges = np.arange(extent[2], extent[3] + bin_deg, bin_deg)

    hist_counts, _, _ = np.histogram2d(
        sel["center_lat"].values,
        sel["center_lon"].values,
        bins=[lat_edges, lon_edges],
    )

    if denom_class <= 0:
        hist = np.zeros_like(hist_counts, dtype=float)
    else:
        hist = hist_counts.astype(float) / float(denom_class)

    return hist, lon_edges, lat_edges, len(sel)


def plot_training_class_occurrence_multiplot(train_df, output_file, utc_range):
    """Plot training occurrence maps with the requested 3-row class layout."""
    train_df = filter_by_utc_range(train_df, utc_range)

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(14.0, 10.0),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    fig.subplots_adjust(left=0.05, right=0.9, bottom=0.07, top=0.95, wspace=0.0, hspace=0.16)

    cmap = cmc.nuuk

    all_hists = {}
    for row_classes in ROW_LAYOUT:
        for class_short in row_classes:
            if class_short is None:
                continue
            denom_class = int((train_df["label"] == SHORT_TO_LABEL[class_short]).sum())
            hist, lon_edges, lat_edges, n_points = compute_occurrence_histogram(
                train_df,
                class_short,
                extent=REDUCED_EXTENT,
                bin_deg=BIN_DEG,
                denom_class=denom_class,
            )
            all_hists[class_short] = (hist, lon_edges, lat_edges, n_points)

    panel_counter = 0
    mesh = None
    for row_idx, row_classes in enumerate(ROW_LAYOUT):
        for col_idx, class_short in enumerate(row_classes):
            ax = axes[row_idx, col_idx]
            if class_short is None:
                ax.axis("off")
                continue

            hist, lon_edges, lat_edges, n_points = all_hists[class_short]

            plot_orography_map(
                OROG_PATH,
                ax=ax,
                var_name="DEM",
                extent=REDUCED_EXTENT,
                cmap="Greys",
                levels=30,
                alpha=0.6,
            )

            mesh = ax.pcolormesh(
                lon_edges,
                lat_edges,
                hist,
                cmap=cmap,
                vmin=CBAR_VMIN,
                vmax=CBAR_VMAX,
                transform=ccrs.PlateCarree(),
                shading="auto",
                alpha=OCCURRENCE_ALPHA,
                zorder=5,
            )

            ax.set_extent(REDUCED_EXTENT, crs=ccrs.PlateCarree())
            ax.tick_params(axis="both", which="major", labelsize=AXIS_TICK_FONTSIZE)
            gl = ax.gridlines(draw_labels=True, alpha=0.3)
            gl.xlocator = FixedLocator([8, 10, 12])
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {"size": GRID_LABEL_FONTSIZE}
            gl.ylabel_style = {"size": GRID_LABEL_FONTSIZE}
            if col_idx > 0:
                gl.left_labels = False
            if row_idx < 2:
                gl.bottom_labels = False

            panel_letter = chr(ord("a") + panel_counter)
            panel_counter += 1
            ax.set_title(
                f"{panel_letter}) {class_short} (n={n_points})",
                fontsize=PANEL_TITLE_FONTSIZE,
                fontweight="bold",
            )

    # Hide the unused slots in the first two rows.
    axes[0, 2].axis("off")
    axes[1, 2].axis("off")

    cax = fig.add_axes([0.92, 0.16, 0.015, 0.68])
    cbar = fig.colorbar(mesh, cax=cax, orientation="vertical")
    cbar.set_label(
        f"Relative frequency ($\\times 10^{{-3}}$)",
        fontsize=CBAR_LABEL_FONTSIZE,
    )
    cbar.ax.tick_params(labelsize=CBAR_TICK_FONTSIZE)
    cbar.formatter = FuncFormatter(lambda val, _: f"{val * 1e3:.2f}")
    cbar.update_ticks()

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot training class occurrence maps (2D lat/lon histograms)."
    )
    parser.add_argument("--training-csv", default=TRAINING_CSV_DEFAULT)
    parser.add_argument(
        "--output-file",
        default=os.path.join(OUTPUT_PATH, "occurrence_hist_train_all_classes.png"),
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    training_csv = resolve_training_csv(args.training_csv)
    train_df = prepare_training_occurrence_df(training_csv)

    print(f"Prepared train={len(train_df)} crops for occurrence maps")
    for utc_range in UTC_TIME_RANGES:
        output_file = build_time_range_output_file(args.output_file, utc_range)
        print(f"Plotting occurrence maps for {format_utc_range_label(utc_range)}")
        plot_training_class_occurrence_multiplot(train_df, output_file, utc_range)
        print(f"Saved occurrence map to {output_file}")


if __name__ == "__main__":
    main()