"""
Script: visualize_nc_crops.py

Description:
------------
This script visualizes NetCDF crop data over reduced embeddings (t-SNE / PCA) 
in three modes: scatter plot, grid plot, and table plot, with proper handling of NaNs 
and normalization.

Dependencies:
-------------
- numpy, pandas, xarray, matplotlib, cmcrameri, scipy
- Custom modules: utils.buckets, aux_functions_from_buckets
"""

import os
import io
import numpy as np
import pandas as pd
import xarray as xr
from glob import glob
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.colors import ListedColormap
from scipy.spatial import cKDTree
import cmcrameri.cm as cmc

from utils.buckets.get_data_from_buckets import read_file, Initialize_s3_client
from utils.buckets.credentials_buckets import S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY, S3_ENDPOINT_URL
from utils.buckets.aux_functions_from_buckets import extract_coordinates, extract_datetime

# -----------------------------
# Configuration
# -----------------------------
RUN_NAME = 'dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA'
RANDOM_STATE = 3
SAMPLING_TYPE = 'all'
REDUCTION_METHOD = 'tsne'
VARS = ['IR_108','WV_062', 'cot', 'cth','precipitation','cma', 'cph']
SAMPLE_TO_PLOT = 200

BUCKETS = {
    'cmsaf': 'expats-cmsaf-cloud',
    'imerg': 'expats-imerg-prec',
    'msg': 'expats-msg-training'
}

COLORMAPS = {
    "IR_108": plt.get_cmap("Greys"),
    "WV_062": plt.get_cmap("Greys"),
    "cot": plt.get_cmap("magma"),
    "cth": plt.get_cmap("cividis"),
    "precipitation": cmc.batlowK,
    "cma": plt.get_cmap("binary_r"),
    "cph": ListedColormap(["black", "lightblue", "darkorange"])
}

NORMS = {
    "IR_108": None,
    "WV_062": None,
    "cot": plt.Normalize(vmin=0, vmax=20),
    "cth": None,
    "precipitation": plt.Normalize(vmin=0, vmax=10),
    "cma": None,
    "cph": None
}

OUTPUT_PATH = f'/data1/fig/{RUN_NAME}/{SAMPLING_TYPE}/'
FILENAME = f'{REDUCTION_METHOD}_pca_cosine_{RUN_NAME}_{RANDOM_STATE}.npy'
os.makedirs(OUTPUT_PATH + 'crop_embeddings/', exist_ok=True)

# -----------------------------
# Initialize S3
# -----------------------------
s3 = Initialize_s3_client(S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY)

# -----------------------------
# Load embedding CSV
# -----------------------------
df_labels = pd.read_csv(f'{OUTPUT_PATH}merged_tsne_variables_{RUN_NAME}_{SAMPLING_TYPE}_{RANDOM_STATE}.csv')
df_labels = df_labels.loc[:, ~df_labels.columns.str.contains('^color')]  # remove color columns
df_labels = df_labels[df_labels['label'] != -100]  # remove invalid labels

# Map labels to colors
COLORS_PER_CLASS = {
    0: 'darkgray', 1: 'darkslategrey', 2: 'peru', 3: 'orangered',
    4: 'lightcoral', 5: 'deepskyblue', 6: 'purple', 7: 'lightblue', 8: 'green'
}
df_labels['color'] = df_labels['label'].map(COLORS_PER_CLASS)

# Sample for plotting
df_sample = df_labels.sample(n=SAMPLE_TO_PLOT, random_state=RANDOM_STATE)

# -----------------------------
# Functions
# -----------------------------
def select_ds_from_dataframe(row, var):
    """Retrieve NetCDF crop from S3 and subset spatially and temporally."""
    crop_filename = os.path.basename(row['path'])
    coords = extract_coordinates(crop_filename)
    lat_min, lat_max, lon_min, lon_max = coords['lat_min'], coords['lat_max'], coords['lon_min'], coords['lon_max']
    
    dt_info = extract_datetime(crop_filename)
    dt_obj = np.datetime64(f"{dt_info['year']:04d}-{dt_info['month']:02d}-{dt_info['day']:02d}T"
                           f"{dt_info['hour']:02d}:{dt_info['minute']:02d}:00")
    
    if var == 'precipitation' and dt_info['minute'] in [15, 45]:
        return None

    # Select bucket and file
    if var in ['IR_108', 'WV_062']:
        bucket, file_path = BUCKETS['msg'], f'/data/sat/msg/ml_train_crops/IR_108-WV_062-CMA_FULL_EXPATS_DOMAIN/{dt_info["year"]:04d}/{dt_info["month"]:02d}/merged_MSG_CMSAF_{dt_info["year"]:04d}-{dt_info["month"]:02d}-{dt_info["day"]:02d}.nc'
    elif var == 'precipitation':
        bucket, file_path = BUCKETS['imerg'], f'IMERG_daily_{dt_info["year"]:04d}-{dt_info["month"]:02d}-{dt_info["day"]:02d}.nc'
    else:
        bucket, file_path = BUCKETS['cmsaf'], f'MCP_{dt_info["year"]:04d}-{dt_info["month"]:02d}-{dt_info["day"]:02d}_regrid.nc'

    try:
        obj = read_file(s3, file_path, bucket)
        ds = xr.open_dataset(io.BytesIO(obj))[var]
        if isinstance(ds.indexes["time"], xr.CFTimeIndex):
            ds["time"] = ds["time"].astype("datetime64[ns]")
        ds = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
        ds = ds.sel(time=dt_obj)
        return ds if not np.isnan(ds.values).all() else None
    except Exception as e:
        print(f"Error loading {var} for {row['path']}: {e}")
        return None

def plot_nc_crops_scatter(df, var, output_path, filename, cmap, norm):
    """Scatter plot with images over embeddings."""
    fig, ax = plt.subplots(figsize=(8, 8))
    for idx, row in df.iterrows():
        ds = select_ds_from_dataframe(row, var)
        if ds is None:  # skip if no valid data
            continue
        img = np.flipud(ds.values.squeeze())
        imagebox = OffsetImage(img, zoom=0.3, cmap=cmap)
        ab = AnnotationBbox(imagebox, (row['Component_1'], row['Component_2']), frameon=False)
        ax.add_artist(ab)
    ax.axis('off')
    fig.savefig(os.path.join(output_path, f"{filename.split('.')[0]}_nc_scatter_{var}.png"), bbox_inches='tight', dpi=300)
    plt.close(fig)

def plot_nc_crops_grid(df, var, output_path, filename, cmap, norm, grid_size=10):
    """Regular grid visualization using nearest neighbor crop per cell."""
    fig, ax = plt.subplots(figsize=(8, 8))
    x_norm = (df['Component_1'] - df['Component_1'].min()) / (df['Component_1'].max() - df['Component_1'].min())
    y_norm = (df['Component_2'] - df['Component_2'].min()) / (df['Component_2'].max() - df['Component_2'].min())
    tree = cKDTree(np.c_[x_norm, y_norm])
    grid_points = np.c_[np.linspace(0, 1, grid_size).repeat(grid_size), np.tile(np.linspace(0, 1, grid_size), grid_size)]
    used_idx = set()
    for pt in grid_points:
        dist, idx = tree.query(pt)
        if idx in used_idx:
            continue
        used_idx.add(idx)
        ds = select_ds_from_dataframe(df.iloc[idx], var)
        if ds is None:
            continue
        img = np.flipud(ds.values.squeeze())
        ab = AnnotationBbox(OffsetImage(img, zoom=0.3, cmap=cmap), pt, frameon=False)
        ax.add_artist(ab)
    ax.axis('off')
    fig.savefig(os.path.join(output_path, f"{filename.split('.')[0]}_nc_grid_{var}.png"), bbox_inches='tight', dpi=300)
    plt.close(fig)

def plot_nc_crops_table(df, var, output_path, filename, cmap, norm, n=5, selection="closest"):
    """Table visualization of crops per class."""
    labels = df['label'].unique()
    fig, axes = plt.subplots(len(labels), n, figsize=(n*2, len(labels)*2))
    if len(labels) == 1:
        axes = np.expand_dims(axes, axis=0)
    for i, label in enumerate(labels):
        subset = df[df['label'] == label]
        if selection == "closest":
            subset = subset.nsmallest(n, 'distance')
        elif selection == "farthest":
            subset = subset.nlargest(n, 'distance')
        elif selection == "random":
            subset = subset.sample(n=min(n, len(subset)), random_state=RANDOM_STATE)
        for j, (_, row) in enumerate(subset.iterrows()):
            ds = select_ds_from_dataframe(row, var)
            if ds is None:
                continue
            img = np.flipud(ds.values.squeeze())
            ax = axes[i, j] if len(labels) > 1 else axes[j]
            ax.imshow(img, cmap=cmap)
            ax.axis('off')
            if j == 0:
                ax.set_ylabel(f"Label {label}", rotation=0, labelpad=30, va='center', fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(output_path, f"{filename.split('.')[0]}_{n}_{selection}_nc_table_{var}.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

# -----------------------------
# Main Loop
# -----------------------------
for var in VARS:
    print(f"Plotting {var}")
    plot_nc_crops_scatter(df_sample, var, OUTPUT_PATH+'crop_embeddings/', FILENAME, COLORMAPS[var], NORMS[var])
    # plot_nc_crops_grid(df_sample, var, OUTPUT_PATH+'crop_embeddings/', FILENAME, COLORMAPS[var], NORMS[var], grid_size=10)
    # plot_nc_crops_table(df_labels, var, OUTPUT_PATH+'crop_embeddings/', FILENAME, COLORMAPS[var], NORMS[var], n=10, selection="closest")
