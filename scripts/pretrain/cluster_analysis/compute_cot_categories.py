"""
Compute fractional area of Cloud Optical Thickness (COT) categories for each crop
and add them to the existing CSV file.

Also, generate a stacked bar plot showing the composition of COT categories per cloud class.

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

COT_BINS = {
    #"cot_clear": (0, 1),
    "cot_thin": (0, 5),
    "cot_medium": (5, 30),
    #"cot_thick": (15, 30),
    "cot_thick": (30, np.inf),
}

COMPUTE_COT = False
PLOT_RESULTS = True

PERCENTILE_COLS = ["25", "50", "75", "99"]

MIN_VALID_FRACTION = 0.5

S3_BUCKET_NAME = "expats-cmsaf-cloud"
bucket_filename_prefix = "MCP_"
bucket_filename_suffix = "_regrid.nc"

path_to_dir = f"/data1/fig/{RUN_NAME}/{epoch}/{crop_sel}/"

csv_path = os.path.join(
    path_to_dir,
    "merged_crops_stats_all_cvc.csv"
)

output_path = os.path.join(
    path_to_dir,
    "merged_crops_stats_all_cvc_cot_fractions.csv"
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


def compute_cot_fractions(ds, ds_cma):
    """
    Compute fractional area of COT categories for one crop.
    Returns dict or None if not enough valid pixels.
    """

    cot = ds.values
    valid = np.isfinite(cot) 

    if valid.mean() < MIN_VALID_FRACTION:
        return None
    
    #build the cloud mask
    cma = ds_cma.values
    #apply closing algorithm (structure 3x3) to fill small holes in cloud mask
    cma = binary_closing(cma==1, structure=np.ones((3,3))).astype(int)
    cloud_mask = cma == 1  #consider as cloud pixels with c

    cot_valid = cot[valid & cloud_mask]
    total = cot_valid.size

    fractions = {}
    for name, (lo, hi) in COT_BINS.items():
        fractions[name] = np.logical_and(cot_valid >= lo, cot_valid < hi).sum() / total
        #round fractions to 4 decimal places
        fractions[name] = round(fractions[name], 4)

    return fractions




def plot_cot_violin_by_class(df, items, savepath=None):
    """
    Violin plots of COT distributions (thin / medium / thick)
    across cloud classes.

    One row, three columns.
    """

    class_col = "label"
    value_col = "None"
    cot_vars = ["cot_thin", "cot_medium", "cot_thick"]

    cot_var_titles = {
        "cot_thin": "COT Thin (0–5)",
        "cot_medium": "COT Medium (5–30)",
        "cot_thick": "COT Thick (30+)",
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
    df_cot = df[df["var"].isin(cot_vars)].copy()
    df_cot = df_cot[~df_cot[value_col].isna()]

    # -----------------------------
    # Figure
    # -----------------------------
    fig, axes = plt.subplots(
        1, len(cot_vars),
        figsize=(3. * len(cot_vars), 2),
        sharey=True
    )

    if len(cot_vars) == 1:
        axes = [axes]

    # -----------------------------
    # Plot each COT bin
    # -----------------------------
    for ax, cot_var in zip(axes, cot_vars):

        df_var = df_cot[df_cot["var"] == cot_var]
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

            print(f"{cot_var} - {lbl}: median COT fraction = {medians[-1]}")

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
        ax.set_title(cot_var_titles[cot_var], fontsize=12, fontweight="bold")
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
        print(f"✅ Saved COT violin plot → {savepath}")

    return fig, axes

#function to print and plot numerosity of daytime samples per cloud class
def print_daytime_numerosity_per_class(df, items, path_to_dir):
    labels_ordered = [lbl for lbl, _ in items]
    short_labels = [info["short"] for _, info in items]
    colors = {lbl: info["color"] for lbl, info in items}

    class_col = "label"
    #daytime is where column nighttime_scene is not True
    df_daytime = df[(df["var"] == "nighttime_scene") & (df["None"] != True)]
    class_counts = df_daytime[class_col].value_counts()
    print("Numerosity of daytime samples per cloud class:")
    for lbl, count in class_counts.items():
        print(f"  {lbl}: {count} samples")
    #order it by labels_ordered
    class_counts = class_counts.reindex(labels_ordered).fillna(0)

    plt.figure(figsize=(5, 3))
    sns.barplot(
        x=class_counts.index,
        y=class_counts.values,
        palette=[colors[lbl] for lbl in class_counts.index]
    )

    plt.ylabel("Number of Samples", fontsize=12)
    plt.yticks(fontsize=12)
    plt.xticks(ticks=np.arange(len(short_labels)), labels=short_labels, rotation=45, ha="right", fontsize=12)
    plt.title("Daytime Samples per Cloud Class", fontsize=12, fontweight="bold")
    plt.grid(axis="y", alpha=0.3)
    plt.savefig(
        os.path.join(path_to_dir, "daytime_numerosity_per_class.png"),
        dpi=300,
        transparent=True,
        bbox_inches="tight"
    )
    print(f"✅ Saved daytime numerosity plot → {os.path.join(path_to_dir, 'daytime_numerosity_per_class.png')}")
    plt.close()


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
        crop_indices = df["crop_index"].unique()
        print(f"Found {len(crop_indices)} unique crops")

    return df, crop_indices


def process_crop_statistics(df, crop_indices):
    new_rows = []
    for crop_idx in crop_indices:

        row = df[df["crop_index"] == crop_idx].iloc[0]
        nc_key = row["path"]
        time = row["time"]
        print(time)
        #extract date in pandas
        date = pd.to_datetime(time).strftime("%Y-%m-%d")
        ds, ds_cma = read_s3_nc(date, s3, S3_BUCKET_NAME, var="cot")
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
        # COT FRACTIONS
        # -----------------------
        cot_fracs = compute_cot_fractions(ds, ds_cma)

        if cot_fracs is None:
            print("  ⚠️ Not enough valid COT pixels")
            print('  Saving the row as nighttime samples')
            new_row = row.copy()
            new_row["var"] = "nighttime_scene"
            new_row["None"] = True
            for p in PERCENTILE_COLS:
                new_row[p] = None
            new_rows.append(new_row)
            continue

        for var_name, frac in cot_fracs.items():
            new_row = row.copy()
            new_row["var"] = var_name
            new_row["None"] = frac
            for p in PERCENTILE_COLS:
                new_row[p] = None
        
            new_rows.append(new_row)

        print("  ✅ COT fractions computed")

    return new_rows
    


def save_results(new_rows, df):
    if len(new_rows) > 0:
        df_out = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        df_out.to_csv(output_path, index=False)
        print(f"\n✅ Saved updated CSV with cot fractions → {output_path}")
    else:
        print("\n⚠️ No new rows added")



if __name__ == "__main__":
    
    if COMPUTE_COT:
        df, crop_indices = load_csv(csv_path)#, random_sample=10)
        new_rows = process_crop_statistics(df, crop_indices)
        print(new_rows)
        save_results(new_rows, df)
    if PLOT_RESULTS:
    # Load updated dataframe for plotting
        df_out = pd.read_csv(output_path)
        print(df_out['var'].unique())
        print_daytime_numerosity_per_class(df_out, cloud_items, path_to_dir)
        
        
        #if nighttime_scene column is True, remove those rows
        #df_out = df_out[~((df_out['var'] == 'nighttime_scene') & (df_out['None'] == True))]
        #plot_cot_violin_by_class(df_out, cloud_items, path_to_dir + "cot_violin_by_class.png")
        

#221767


