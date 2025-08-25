"""
Module: percentile_maps_analysis
--------------------------------
This module provides functionality to:
1. Aggregate min/max values of variables across percentile NetCDF files.
2. Plot percentile maps per variable, class label, and hour using Cartopy.
3. Store results in CSV and PNG files.

Workflow:
---------
1. Scan all percentile NetCDF files and compute global min/max per variable.
2. Plot each variable in the dataset using a consistent colormap scale.
"""

import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cmcrameri.cm as cmc
import sys

# Load custom functions
sys.path.append('/home/Daniele/codes/visualization/cluster_analysis/')
from utils.buckets.aux_functions_from_buckets import get_variable_info


# -----------------------------
# Configuration
# -----------------------------
RUN_NAME = 'dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA'
SAMPLING_TYPE = 'all'
N_DIV = 16
VARIABLES_PATH = f'/data1/fig/{RUN_NAME}/{SAMPLING_TYPE}/percentile_maps/'
DEM_PATH = '/data1/other_data/DEM_EXPATS_0.01x0.01.nc'
OUTPUT_CSV = os.path.join(VARIABLES_PATH, "variable_min_max.csv")
COLORMAP = cmc.nuuk
LABELS = np.arange(0, 9)
HOURS = range(0, 24)
VARS_TO_PLOT = None  # None = all variables


# -----------------------------
# Functions
# -----------------------------
def compute_variable_min_max(file_pattern: str, output_csv: str):
    """
    Computes global min/max values for each variable across all NetCDF files
    matching the given pattern and saves results to a CSV file.

    Args:
        file_pattern (str): Glob pattern to select NetCDF files.
        output_csv (str): Path to save CSV with variable min/max.

    Returns:
        pd.DataFrame: DataFrame containing min/max for each variable.
    """
    file_list = sorted(glob.glob(file_pattern))
    print(f"Found {len(file_list)} files for min/max computation.")

    variable_stats = {}
    for file_path in file_list:
        print(f"Processing: {file_path}")
        ds = xr.open_dataset(file_path, decode_times=False, engine="h5netcdf")
        for var in ds.data_vars:
            values = ds[var].values
            var_min = np.nanmin(values)
            var_max = np.nanmax(values)
            if var not in variable_stats:
                variable_stats[var] = {"Min": var_min, "Max": var_max}
            else:
                variable_stats[var]["Min"] = min(variable_stats[var]["Min"], var_min)
                variable_stats[var]["Max"] = max(variable_stats[var]["Max"], var_max)
        ds.close()

    df = pd.DataFrame.from_dict(variable_stats, orient="index").reset_index()
    df.columns = ["Variable", "Min", "Max"]
    df.to_csv(output_csv, index=False)
    print(f"Saved variable min/max CSV: {output_csv}")
    return df


def plot_dataset_maps(ds: xr.Dataset, ds_oro: xr.Dataset, output_path: str, label: int, cmap, vmin_vmax_df: pd.DataFrame, hour: int):
    """
    Plots each variable in a dataset with Cartopy, overlaying orography and
    applying consistent color limits.

    Args:
        ds (xr.Dataset): Dataset with lat/lon and variables.
        ds_oro (xr.Dataset): Dataset containing orography (DEM) data.
        output_path (str): Path to save plots.
        label (int): Class label.
        cmap: Colormap to use.
        vmin_vmax_df (pd.DataFrame): DataFrame with variable min/max.
        hour (int): Hour of day for title annotation.
    """
    lat_edges = np.array(eval(ds.attrs["lat_edges"]))
    lon_edges = np.array(eval(ds.attrs["lon_edges"]))
    lat_centers = ds.lat.values
    lon_centers = ds.lon.values

    oro_lat = ds_oro.lat.values
    oro_lon = ds_oro.lon.values
    orography = ds_oro["DEM"].values

    for var in ds.data_vars:
        if VARS_TO_PLOT is not None and var not in VARS_TO_PLOT:
            continue

        print(f"Plotting variable: {var}")
        vmin = vmin_vmax_df.loc[vmin_vmax_df['Variable'] == var, 'Min'].values[0]
        vmax = vmin_vmax_df.loc[vmin_vmax_df['Variable'] == var, 'Max'].values[0]

        var_info = get_variable_info(var.split('-')[0])

        fig, ax = plt.subplots(figsize=(6, 5), subplot_kw={'projection': ccrs.PlateCarree()})
        ax.set_extent([lon_edges.min(), lon_edges.max(), lat_edges.min(), lat_edges.max()], crs=ccrs.PlateCarree())
        pcm = ax.pcolormesh(lon_centers, lat_centers, ds[var], cmap=cmap, shading="auto", alpha=1, transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)
        ax.pcolormesh(oro_lon, oro_lat, orography, cmap="Greys", shading="auto", alpha=0.3, transform=ccrs.PlateCarree())
        ax.add_feature(cfeature.BORDERS, linewidth=1, color="black")
        ax.add_feature(cfeature.COASTLINE, linewidth=1, color="black")
        ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.5)

        cbar = plt.colorbar(pcm, ax=ax, orientation="vertical", shrink=0.7)
        if var_info['unit']:
            cbar.set_label(f"{var_info['long_name']} ({var_info['unit']})", fontsize=12)
        else:
            cbar.set_label(f"{var_info['long_name']}", fontsize=12)

        percentile = var.split('-')[1] if '-' in var else None
        title_text = f"{var_info['long_name']} {percentile+'th percentile ' if percentile else ''}map for class {label} hour {hour}"
        plt.title(title_text, fontsize=14, fontweight="bold")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

        gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.5, alpha=0.7)
        gl.xformatter = LongitudeFormatter(number_format='.1f')
        gl.yformatter = LatitudeFormatter(number_format='.1f')
        gl.xlocator = plt.FixedLocator(np.arange(5, 16, 2))
        gl.ylocator = plt.FixedLocator(np.arange(42, 52, 2))
        gl.right_labels = False
        gl.top_labels = False
        gl.bottom_labels = True
        gl.left_labels = True
        gl.xlabel_style = {'fontsize': 12}
        gl.ylabel_style = {'fontsize': 12}

        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, f"percentile_{var}_maps_res_label_{label}_hour_{hour}.png")
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved plot: {output_file}")


# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    # Step 1: Compute or load min/max CSV
    if os.path.exists(OUTPUT_CSV):
        print(f"Min/max CSV already exists. Loading from {OUTPUT_CSV}")
        vmin_vmax_df = pd.read_csv(OUTPUT_CSV)
    else:
        print(f"Min/max CSV not found. Computing from NetCDF files.")
        file_pattern = os.path.join(VARIABLES_PATH, f"percentile_maps_res_{N_DIV}x{N_DIV}_label_*.nc")
        vmin_vmax_df = compute_variable_min_max(file_pattern, OUTPUT_CSV)


    # Step 2: Load DEM
    ds_dem = xr.open_dataset(DEM_PATH, decode_times=False, engine="h5netcdf")

    # Step 3: Loop over labels and hours to plot
    for label in LABELS:
        for hour in HOURS:
            nc_file = os.path.join(VARIABLES_PATH, f"{label}/percentile_maps_res_{N_DIV}x{N_DIV}_label_{label}_hour_{hour}.nc")
            if not os.path.exists(nc_file):
                print(f"File not found: {nc_file}, skipping.")
                continue
            ds = xr.open_dataset(nc_file, decode_times=False, engine="h5netcdf")
            plot_dataset_maps(ds, ds_dem, os.path.join(VARIABLES_PATH, f"{label}/"), label, COLORMAP, vmin_vmax_df, hour)
            ds.close()
