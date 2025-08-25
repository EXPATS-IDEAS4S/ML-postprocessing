"""
Description:
------------
This script generates overlaid histograms for selected satellite/cloud variables 
(e.g., IR_108, WV_062, COT, CTH, precipitation) per cluster/class label. 
It extracts data from S3, applies spatial and temporal selection, and optionally 
normalizes histograms to probability distributions.

"""

import os
import io
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

from utils.buckets.aux_functions_from_buckets import extract_coordinates, extract_datetime
from utils.buckets.get_data_from_buckets import read_file, Initialize_s3_client
from utils.buckets.credentials_buckets import S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY, S3_ENDPOINT_URL

sys.path.append("/home/Daniele/codes/visualization/cluster_analysis")

# -----------------------------
# Configuration
# -----------------------------
RUN_NAME = 'dcv2_ir108_200x200_k9_expats_70k_200-300K_closed-CMA'
SAMPLING_TYPE = 'closest'
VARIABLES = ['IR_108', 'WV_062', 'cot', 'cth', 'precipitation']
UNITS = ['K', 'K', None, 'm', 'mm/h']
LOGS = [False, False, True, False, True]
XLIMS = [(200, 320), (205, 250), (0, 150), (0, 17500), (0, 80)]
YLIMS = [(0, 0.50), (0, 0.25), (0, 0.70), (0, 0.15), (0, 0.70)]
ALPHA = 0.01
NSUBSAMPLE = 1000

BUCKETS = {
    'cmsaf': 'expats-cmsaf-cloud',
    'imerg': 'expats-imerg-prec',
    'msg': 'expats-msg-training'
}

OUTPUT_PATH = f'/data1/fig/{RUN_NAME}/{SAMPLING_TYPE}/'
os.makedirs(OUTPUT_PATH, exist_ok=True)

# -----------------------------
# Initialize S3
# -----------------------------
s3_client = Initialize_s3_client(S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY)

# -----------------------------
# Load crop labels
# -----------------------------
crop_list_file = f'{OUTPUT_PATH}crop_list_{RUN_NAME}_{NSUBSAMPLE}_{SAMPLING_TYPE}.csv'
df_labels = pd.read_csv(crop_list_file)
df_labels = df_labels[df_labels['label'] != -100]  # remove invalid labels

# -----------------------------
# Functions
# -----------------------------
def extract_variable_values(row, var):
    """Extract a single variable from S3 for the spatial-temporal extent of a crop."""
    crop_filename = row['path'].split('/')[-1]
    coords = extract_coordinates(crop_filename)
    lat_min, lat_max, lon_min, lon_max = coords['lat_min'], coords['lat_max'], coords['lon_min'], coords['lon_max']

    dt_info = extract_datetime(crop_filename)
    datetime_obj = np.datetime64(f"{dt_info['year']:04d}-{dt_info['month']:02d}-{dt_info['day']:02d}T"
                                 f"{dt_info['hour']:02d}:{dt_info['minute']:02d}:00")
    print("Processing timestamp:", datetime_obj)

    # Skip certain minutes for precipitation
    if var == 'precipitation' and dt_info['minute'] in [15, 45]:
        return []

    # Determine S3 bucket and filename
    if var in ['IR_108', 'WV_062']:
        bucket_name = BUCKETS['msg']
        bucket_filename = (f'/data/sat/msg/ml_train_crops/IR_108-WV_062-CMA_FULL_EXPATS_DOMAIN/'
                           f"{dt_info['year']:04d}/{dt_info['month']:02d}/"
                           f"merged_MSG_CMSAF_{dt_info['year']:04d}-{dt_info['month']:02d}-{dt_info['day']:02d}.nc")
    elif var == 'precipitation':
        bucket_name = BUCKETS['imerg']
        bucket_filename = f'IMERG_daily_{dt_info["year"]:04d}-{dt_info["month"]:02d}-{dt_info["day"]:02d}.nc'
    else:
        bucket_name = BUCKETS['cmsaf']
        bucket_filename = f'MCP_{dt_info["year"]:04d}-{dt_info["month"]:02d}-{dt_info["day"]:02d}_regrid.nc'

    try:
        my_obj = read_file(s3_client, bucket_filename, bucket_name)
        ds_day = xr.open_dataset(io.BytesIO(my_obj))[var]

        if isinstance(ds_day.indexes["time"], xr.CFTimeIndex):
            ds_day["time"] = ds_day["time"].astype("datetime64[ns]")

        ds_day = ds_day.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
        ds_day = ds_day.sel(time=datetime_obj)
        values = ds_day.values.flatten()
    except Exception as e:
        print(f"Error processing {var} for {row['path']}: {e}")
        values = []

    return values

def plot_histogram(df_label_subset, variable, unit, log_scale, xlim, ylim, alpha=0.01, bins=50):
    """Overlay histograms for all crops in the subset for a given variable."""
    bin_edges = np.linspace(xlim[0], xlim[1], bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    sum_in_bins = np.zeros(bins)
    count_in_bins = np.zeros(bins)

    fig, ax = plt.subplots(figsize=(8, 6))

    for _, row in df_label_subset.iterrows():
        values = extract_variable_values(row, variable)
        if len(values) == 0:
            continue

        sns.histplot(
            values,
            bins=bin_edges,
            kde=False,
            alpha=alpha,
            element="bars",
            fill=True,
            edgecolor=None,
            color='blue',
            stat="probability"
        )

        hist, _ = np.histogram(values, bins=bin_edges)
        bin_value_sums, _ = np.histogram(values, bins=bin_edges, weights=values)
        sum_in_bins += np.nan_to_num(bin_value_sums)
        count_in_bins += hist

    probability_per_bin = count_in_bins / count_in_bins.sum()
    ax.plot(bin_centers, probability_per_bin, color='red', lw=2, label='Mean')

    xlabel = f'{variable} [{unit}]' if unit else variable
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    if log_scale:
        ax.set_yscale('log')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(f'Distribution of {variable} for Class {df_label_subset["label"].iloc[0]}')
    ax.grid(True, linestyle='--', alpha=0.5)
    return fig, ax

# -----------------------------
# Main Loop
# -----------------------------
for label in df_labels['label'].unique():
    df_subset = df_labels[df_labels['label'] == label]

    for var, unit, log, xlim, ylim in zip(VARIABLES, UNITS, LOGS, XLIMS, YLIMS):
        print(f"Processing label {label}, variable {var}")
        fig, ax = plot_histogram(df_subset, var, unit, log, xlim, ylim, alpha=ALPHA)
        output_dir = f'{OUTPUT_PATH}overlaid_histograms/{label}/'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f'{output_dir}histogram_{var}_label_{label}.png', dpi=300, bbox_inches="tight")
        plt.close(fig)

print("All histograms generated successfully.")
