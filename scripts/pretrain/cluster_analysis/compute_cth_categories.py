"""
Compute fractional area of Cloud Top Height (CTH) categories for each crop
and add them to the existing CSV file.

Also, generate a stacked bar plot showing the composition of CTH categories per cloud class.

"""

import os
import io
import sys
import pandas as pd
import numpy as np
import xarray as xr
import boto3
from botocore.exceptions import ClientError
import matplotlib.pyplot as plt
from scipy.ndimage import binary_closing

import matplotlib.pyplot as plt
import seaborn as sns


sys.path.append('/home/Daniele/codes/VISSL_postprocessing/utils/')
from plotting.class_colors import CLOUD_CLASS_INFO
from buckets.get_data_from_buckets import read_file, Initialize_s3_client

# =====================
# CONFIG
# =====================
RUN_NAME = "dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc"
epoch = "epoch_800"
crop_sel = "all"


MIN_PIXEL_COUNT = 100

CTH_BINS = {
    # Assumes CTH is in meters. Adjust if your data is in km.
    "cth_low": (0, 2000),
    "cth_medium": (2000, 7000),
    "cth_high": (7000, 10000),
    "cth_very_high": (10000, np.inf)

}

COMPUTE_CTH = True
PLOT_RESULTS = False

PERCENTILE_COLS = ["25", "50", "75", "99"]

MIN_VALID_FRACTION = 0.05  # Minimum fraction of valid pixels required to compute CTH fractions

S3_BUCKET_NAME = "expats-cmsaf-cloud"
bucket_filename_prefix = "MCP_"
bucket_filename_suffix = "_regrid.nc"

path_to_dir = f"/data1/fig/{RUN_NAME}/{epoch}/{crop_sel}/"
crop_base_dir = "/data1/crops/ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc/nc/1/"

csv_path = os.path.join(
    path_to_dir,
    "merged_crops_stats_all_cvc_cot_fractions.csv"
)

output_path = os.path.join(
    path_to_dir,
    "merged_crops_stats_all_cvc_cot_cth_fractions.csv"
)

cloud_items = sorted(CLOUD_CLASS_INFO.items(), key=lambda x: x[1]["order"])

# =====================
# S3 SETUP
# =====================
sys.path.append("/home/Daniele/codes/VISSL_postprocessing/utils/buckets/")
from credentials_buckets import (
    S3_ACCESS_KEY,
    S3_SECRET_ACCESS_KEY,
    S3_ENDPOINT_URL,
)

s3 = Initialize_s3_client(S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY)

# =====================
# HELPERS
# =====================
def read_s3_nc(day_key, s3, BUCKET_NAME, var):
    y, m, d = map(int, day_key.split("-"))
    bucket_filename = (
        f"{bucket_filename_prefix}{y:04d}-{m:02d}-{d:02d}"
        f"{bucket_filename_suffix}" )
    try:
        obj = read_file(s3, bucket_filename, BUCKET_NAME)
        ds = xr.open_dataset(io.BytesIO(obj), engine="h5netcdf")
        ds_cot = ds[var]
        ds_cma = ds["cma"]
        return ds_cot, ds_cma
    except ClientError as e:
        print(f"❌ Failed to read {day_key}: {e}")
        return None


def compute_cth_fractions(ds, ds_cma):
    """
    Compute fractional area of CTH categories for one crop.
    Returns dict or None if not enough valid pixels.
    """

    cth = ds.values
    valid = np.isfinite(cth) 

    if valid.mean() < MIN_VALID_FRACTION:
        return None
    
    #build the cloud mask
    cma = ds_cma.values
    #apply closing algorithm (structure 3x3) to fill small holes in cloud mask
    cma = binary_closing(cma==1, structure=np.ones((3,3))).astype(int)
    cloud_mask = cma == 1  #consider as cloud pixels with c

    cth_valid = cth[valid & cloud_mask]
    total = cth_valid.size

    fractions = {}
    for name, (lo, hi) in CTH_BINS.items():
        fractions[name] = np.logical_and(cth_valid >= lo, cth_valid < hi).sum() / total
        #round fractions to 4 decimal places
        fractions[name] = round(fractions[name], 4)

    return fractions




def plot_cth_violin_by_class(df, items, savepath=None):
    """
    Violin plots of CTH distributions (low / medium / high)
    across cloud classes.

    One row, three columns.
    """

    class_col = "label"
    value_col = "None"
    cth_vars = ["cth_low", "cth_medium", "cth_high", "cth_very_high"]

    cth_var_titles = {
        "cth_low": "CTH Low (0–2 km)",
        "cth_medium": "CTH Medium (2–7 km)",
        "cth_high": "CTH High (7–10 km)",
        "cth_very_high": "CTH Very High (10+ km)",
    }

    # -----------------------------
    # Order, labels, colors
    # -----------------------------
    labels_ordered = [lbl for lbl, _ in items]
    short_labels = [info["short"] for _, info in items]
    colors = {lbl: info["color"] for lbl, info in items}

    # -----------------------------
    # Filter dataframe
    # -----------------------------
    df_cth = df[df["var"].isin(cth_vars)].copy()
    df_cth = df_cth[~df_cth[value_col].isna()]

    # -----------------------------
    # Figure
    # -----------------------------
    fig, axes = plt.subplots(
        1, len(cth_vars),
        figsize=(3. * len(cth_vars), 2),
        sharey=True
    )

    if len(cth_vars) == 1:
        axes = [axes]

    # -----------------------------
    # Plot each COT bin
    # -----------------------------
    for ax, cth_var in zip(axes, cth_vars):

        df_var = df_cth[df_cth["var"] == cth_var]
        df_var[value_col] = pd.to_numeric(df_var[value_col], errors="coerce")


        # remove classes with < 50 samples
        class_counts = df_var[class_col].value_counts()
        valid_labels = class_counts[class_counts >= 50].index

        # collect data per class in order
        data = []
        medians = []
        for lbl in labels_ordered:
            vals = (
                df_var.loc[df_var[class_col] == lbl, value_col]
                .astype(float)
                .dropna()
                .values
            )

            # handle rare / empty classes safely
            if len(vals) < 50:
                data.append([np.nan])   # placeholder
                medians.append(np.nan)
            else:
                data.append(vals)
                medians.append(np.nanmedian(vals))

            print(f"{cth_var} - {lbl}: median CTH fraction = {medians[-1]}")

        # -----------------------------
        # Violin plot
        # -----------------------------
        parts = ax.violinplot(
            data,
            positions=np.arange(len(labels_ordered)),
            widths=0.8,
            showmeans=False,
            showmedians=True,
            showextrema=False
        )

        # color each violin
        for body, lbl in zip(parts["bodies"], labels_ordered):
            body.set_facecolor(colors[lbl])
            body.set_edgecolor("black")
            body.set_alpha(0.9)

        # -----------------------------
        # Overlay medians
        # -----------------------------
        medians = [
            np.nanmedian(vals) if len(vals) > 0 else np.nan
            for vals in data
        ]

        ax.scatter(
            np.arange(len(labels_ordered)),
            medians,
            color="black",
            s=20,
            zorder=3
        )

        # -----------------------------
        # Axis formatting
        # -----------------------------
        ax.set_title(cth_var_titles[cth_var], fontsize=12, fontweight="bold")
        ax.set_xticks(np.arange(len(labels_ordered)))
        ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=12)

        if ax is axes[0]:
            ax.set_ylabel("Fractional Area", fontsize=12)
            ax.set_ylim(0, 1)
            ax.set_yticks(np.linspace(0, 1, 5))
        else:
            ax.set_ylabel("")

        ax.grid(axis="y", alpha=0.3)

    # -----------------------------
    # Save
    # -----------------------------
    if savepath is not None:
        plt.savefig(
            savepath,
            dpi=300,
            transparent=True,
            bbox_inches="tight"
        )
        print(f"✅ Saved CTH violin plot → {savepath}")

    return fig, axes




def load_csv(csv_path, random_sample=None):
    if random_sample is not None:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded CSV: {df.shape}")

        crop_indices = df["crop_index"].unique()
        print(f"Found {len(crop_indices)} unique crops")
        crop_indices = np.random.choice(crop_indices, size=random_sample, replace=False)
        df = df[df["crop_index"].isin(crop_indices)]
        print(f"Randomly sampled {random_sample} crops: {df.shape}")
        return df, crop_indices
    else:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded CSV: {df.shape}")
        print(df.columns.tolist())
        crop_indices = df["crop"].unique()
        print(f"Found {len(crop_indices)} unique crops")

    return df, crop_indices


def process_crop_statistics(df, crop_indices, crop_base_dir=None):
    new_rows = []
    for crop_idx in crop_indices:

        row = df[df["crop"] == crop_idx].iloc[0]
        nc_key = row["crop"]
        if crop_base_dir is not None:
            nc_key = os.path.join(crop_base_dir, nc_key)
        time = row["time"]
        print(time)
        #extract date in pandas
        date = pd.to_datetime(time).strftime("%Y-%m-%d")
        ds, ds_cma = read_s3_nc(date, s3, S3_BUCKET_NAME, var="cth")
        if ds is None:
            print(f"  ❌ Failed to open dataset: {e}")
            continue

        print(f"\n▶ Crop {crop_idx}")
        print(f"  File: {nc_key}")
        #open ds in nc_key
        try:
            ds_bt = xr.open_dataset(nc_key, engine="h5netcdf")
        except Exception as e:
            print(f"  ❌ Failed to open dataset: {e}")
            continue
        #print(ds_bt)
        latmin, latmax = round(ds_bt["lat"].min().item(),2), round(ds_bt["lat"].max().item(),2)
        lonmin, lonmax = round(ds_bt["lon"].min().item(),2), round(ds_bt["lon"].max().item(),2)
        ds = ds.sel(lat=slice(latmin, latmax), lon=slice(lonmin, lonmax)) 
        ds_cma = ds_cma.sel(lat=slice(latmin, latmax), lon=slice(lonmin, lonmax))
        ds = ds.sel(time=pd.to_datetime(time), method="nearest") 
        ds_cma = ds_cma.sel(time=pd.to_datetime(time), method="nearest")
        #print(ds)
        # -----------------------
        # CTH FRACTIONS
        # -----------------------
        cth_fracs = compute_cth_fractions(ds, ds_cma)

        if cth_fracs is None:
            print("  ⚠️ Not enough valid CTH pixels")
            print('  Saving the row as nighttime samples')
            new_row = row.copy()
            new_row["var"] = "nighttime_scene"
            new_row["None"] = True
            for p in PERCENTILE_COLS:
                new_row[p] = None
            new_rows.append(new_row)
            continue

        for var_name, frac in cth_fracs.items():
            new_row = row.copy()
            new_row["var"] = var_name
            new_row["None"] = frac
            for p in PERCENTILE_COLS:
                new_row[p] = None
        
            new_rows.append(new_row)

        print("  ✅ CTH fractions computed")

    return new_rows
    


def save_results(new_rows, df):
    if len(new_rows) > 0:
        df_out = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        df_out.to_csv(output_path, index=False)
        print(f"\n✅ Saved updated CSV with cth fractions → {output_path}")
    else:
        print("\n⚠️ No new rows added")



if __name__ == "__main__":
    
    if COMPUTE_CTH:
        df, crop_indices = load_csv(csv_path)#, random_sample=10)
        new_rows = process_crop_statistics(df, crop_indices, crop_base_dir=crop_base_dir)
        print(new_rows)
        save_results(new_rows, df)
    if PLOT_RESULTS:
    # Load updated dataframe for plotting
        df_out = pd.read_csv(output_path)
        print(df_out['var'].unique())
        
        

#1753677


