import os
import sys
import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cmcrameri import cm as cmc
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import FixedLocator


# ==================================================
# CONFIGURATION
# ==================================================
RUN_NAME = "dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc"
OUTFILE = f"/data1/fig/{RUN_NAME}/epoch_800/occurrence_maps_training.png"


TRAINING_CSV = (
    "/data1/fig/"
    "dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc/"
    "epoch_800/all/merged_crops_stats_cvc_imergtime_all_with_bt_area_precip.csv"
)

OROG_PATH = "/data1/DEM_EXPATS_0.01x0.01.nc"
MAP_EXTENT = [5, 16, 42, 51.5]
MARGIN_DEG = 2.0
BIN_DEG = 0.2

UTC_TIME_RANGES = [(0, 24)]#[(0, 6), (6, 12), (12, 18), (18, 24), (0, 24)]

# Typography
SUPTITLE_FONTSIZE = 27
PANEL_TITLE_FONTSIZE = 20
AXIS_TICK_FONTSIZE = 16
GRID_LABEL_FONTSIZE = 18
CBAR_LABEL_FONTSIZE = 20
CBAR_TICK_FONTSIZE = 20
CBAR_VMIN = 0.0
CBAR_VMAX = 2e-3
ALPHA = 0.9

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

# ==================================================
# CLASS ORDER (YOUR REQUEST)
# ==================================================
CLASS_ORDER = ["CS", "SC", "MC1", "MC2", "EC", "DC", "OA"]

# If your dataset uses numeric labels, map them here:
SHORT_TO_LABEL = {
    "CS": 3,
    "SC": 0,
    "MC1": 6,
    "MC2": 5,
    "EC": 1,
    "DC": 2,
    "OA": 4,
}

# ==================================================
# UTILS
# ==================================================
def extract_utc_hour(filename):
    m = re.search(r"T(\d{2})", filename)
    if m:
        return float(m.group(1))
    return np.nan


def load_training(csv_path):
    df = pd.read_csv(csv_path)

    df = df.rename(columns={
        "lat_mid": "lat",
        "lon_mid": "lon"
    })

    df = df.dropna(subset=["label", "lat", "lon"])
    df["label"] = df["label"].astype(int)

    df["utc_hour"] = df["crop"].apply(extract_utc_hour)

    return df


def compute_hist(df, label_id, extent):
    sel = df[df["label"] == label_id]

    sel = sel[
        (sel["lon"] >= extent[0]) &
        (sel["lon"] <= extent[1]) &
        (sel["lat"] >= extent[2]) &
        (sel["lat"] <= extent[3])
    ]

    lon_edges = np.arange(extent[0], extent[1] + BIN_DEG, BIN_DEG)
    lat_edges = np.arange(extent[2], extent[3] + BIN_DEG, BIN_DEG)

    hist, _, _ = np.histogram2d(
        sel["lat"], sel["lon"],
        bins=[lat_edges, lon_edges]
    )

    hist = hist / max(len(sel), 1)

    return hist, lon_edges, lat_edges, len(sel)


# ==================================================
# PLOT
# ==================================================
def plot_training_maps(df):
    n = len(CLASS_ORDER)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(15, 12),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )

    axes = axes.flatten()
    cmap = cmc.nuuk

    mesh = None

    for i, cls in enumerate(CLASS_ORDER):
        ax = axes[i]

        label = SHORT_TO_LABEL[cls]

        hist, lon_e, lat_e, npts = compute_hist(df, label, MAP_EXTENT)

        mesh = ax.pcolormesh(
            lon_e,
            lat_e,
            hist,
            cmap=cmap,
            vmin=0,
            vmax=np.percentile(hist, 99),
            transform=ccrs.PlateCarree(),
            shading="auto",
            alpha=ALPHA
        )

        ax.set_extent(MAP_EXTENT)

        gl = ax.gridlines(draw_labels=True, alpha=0.3)
        gl.top_labels = False
        gl.right_labels = False
        if i % ncols != 0:
            gl.left_labels = False
        if i < (nrows - 1) * ncols:
            gl.bottom_labels = False

        gl.xlocator = FixedLocator([6, 8, 10, 12, 14])

        ax.set_title(f"{cls} (n={npts})")

    # hide empty panels
    for j in range(len(CLASS_ORDER), len(axes)):
        axes[j].set_visible(False)

    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cb = fig.colorbar(mesh, cax=cax)
    cb.set_label("Relative frequency")

    plt.tight_layout()
    plt.savefig(OUTFILE, dpi=300)
    plt.close()


# ==================================================
# MAIN
# ==================================================
if __name__ == "__main__":
    df = load_training(TRAINING_CSV)
    plot_training_maps(df)
