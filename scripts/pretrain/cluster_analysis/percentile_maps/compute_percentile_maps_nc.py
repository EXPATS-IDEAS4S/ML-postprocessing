"""
Cloud and Precipitation Grid Processing
---------------------------------------
This script extracts cloud, precipitation, and categorical variables from S3,
divides the lat/lon domain into a grid, computes percentiles and categorical
values, and saves the results as NetCDF files.

Features:
- Supports parallel processing across grid cells
- Handles multiple labels and optional hourly filtering
- Includes configurable variables, stats, and grid resolution
"""

import os
import io
import sys
import time
import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

# Project imports
sys.path.append(os.path.abspath("/home/Daniele/codes/visualization/cluster_analysis"))
from utils.buckets.aux_functions_from_buckets import (
    extract_datetime,
    compute_categorical_values,
    find_crops_in_range
)
from utils.buckets.get_data_from_buckets import read_file, Initialize_s3_client
from utils.buckets.credentials_buckets import S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY, S3_ENDPOINT_URL

# -----------------------------
# Configuration
# -----------------------------
RUN_NAME = 'dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA'
SAMPLING_TYPE = 'all'  # 'closest' or 'all'
N_SUBSAMPLES = 67425
VARS = ['cot', 'cth', 'cma', 'cph', 'precipitation']
STATS = [50, 99]
CATEG_VARS = ['cma', 'cph']
N_DIV = 16  # Grid resolution (lat/lon division)
OUTPUT_PATH = f'/data1/fig/{RUN_NAME}/{SAMPLING_TYPE}/'
BUCKET_CMSAF_NAME = 'expats-cmsaf-cloud'
BUCKET_IMERG_NAME = 'expats-imerg-prec'
BUCKET_CROP_MSG = 'expats-msg-training'

# -----------------------------
# Parallelism control
# -----------------------------
PARALLEL = True      # True = use parallel processing, False = sequential
N_CORES = max(1, os.cpu_count() - 4)  # number of cores to use if PARALLEL=True


# -----------------------------
# Initialize S3 client
# -----------------------------
s3_client = Initialize_s3_client(S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY)

# -----------------------------
# Helper Functions
# -----------------------------
def load_lat_lon_grid(sample_file: str):
    """Load latitude and longitude arrays from a sample MSG file."""
    my_obj = read_file(s3_client, sample_file, BUCKET_CROP_MSG)
    ds = xr.open_dataset(io.BytesIO(my_obj))
    lat, lon = ds['lat'].values, ds['lon'].values
    ds.close()
    return lat, lon

def process_lat_lon_cell(i, j, lat_inf, lat_sup, lon_inf, lon_sup, df_filtered, label, vars, categ_vars):
    """
    Process a single lat/lon grid cell:
    - Extract variable values from S3
    - Compute percentiles or categorical statistics
    """
    results = {}
    s3 = Initialize_s3_client(S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY)
    crops_list = find_crops_in_range(df_filtered, lat_inf, lat_sup, lon_inf, lon_sup)

    if not crops_list:
        return None

    data_values = {var: [] for var in vars}

    for crop_filename in crops_list:
        dt_info = extract_datetime(crop_filename)
        dt_obj = np.datetime64(f"{dt_info['year']:04d}-{dt_info['month']:02d}-{dt_info['day']:02d}T"
                               f"{dt_info['hour']:02d}:{dt_info['minute']:02d}:00")
        for var in vars:
            if var == 'precipitation' and dt_info['minute'] in [15, 45]:
                continue

            bucket_file = (f"MCP_{dt_info['year']:04d}-{dt_info['month']:02d}-{dt_info['day']:02d}_regrid.nc"
                           if var != 'precipitation' else
                           f"IMERG_daily_{dt_info['year']:04d}-{dt_info['month']:02d}-{dt_info['day']:02d}.nc")
            bucket_name = BUCKET_CMSAF_NAME if var != 'precipitation' else BUCKET_IMERG_NAME

            try:
                obj = read_file(s3, bucket_file, bucket_name)
                ds_day = xr.open_dataset(io.BytesIO(obj))[var]
                if isinstance(ds_day.indexes["time"], xr.CFTimeIndex):
                    ds_day["time"] = ds_day["time"].astype("datetime64[ns]")
                ds_day = ds_day.sel(lat=slice(lat_inf, lat_sup),
                                    lon=slice(lon_inf, lon_sup),
                                    time=dt_obj)
                values = ds_day.values.flatten()
                if values.size > 0:
                    data_values[var].extend(values)
                ds_day.close()
            except Exception as e:
                print(f"Error reading {var} for {crop_filename}: {e}")
                continue

    # Compute results
    for var in vars:
        arr = np.array(data_values[var])
        if var in categ_vars:
            results[var] = compute_categorical_values(arr, var)
        else:
            results[f"{var}-50"] = np.nanpercentile(arr, 50) if arr.size > 0 else np.nan
            results[f"{var}-99"] = np.nanpercentile(arr, 99) if arr.size > 0 else np.nan

    return results

def save_grid_to_netcdf(results_dict, lat_mids, lon_mids, lat_edges, lon_edges, label, hour):
    """Save processed grid results into a NetCDF file."""
    ds = xr.Dataset({key: (["lat", "lon"], val.astype(np.float32))
                     for key, val in results_dict.items()},
                    coords={"lat": lat_mids.astype(np.float32),
                            "lon": lon_mids.astype(np.float32)})
    ds.attrs.update({
        "description": f"Processed data for label {label} and hour {hour}",
        "note": "Lat/Lon coordinates represent midpoints of grid cells",
        "lat_edges": str(lat_edges.tolist()),
        "lon_edges": str(lon_edges.tolist())
    })
    ds["lat"].attrs["units"] = "degrees_north"
    ds["lon"].attrs["units"] = "degrees_east"

    out_dir = os.path.join(OUTPUT_PATH, f"percentile_maps/{label}/")
    os.makedirs(out_dir, exist_ok=True)
    output_file = os.path.join(out_dir, f"percentile_maps_res_{N_DIV}x{N_DIV}_label_{label}_hour_{hour}.nc")
    ds.to_netcdf(output_file, format="NETCDF4", engine="h5netcdf")
    ds.close()
    print(f"Saved {output_file}")

# -----------------------------
# Main Execution
# -----------------------------
def main():
    start_time = time.time()

    # Load lat/lon grid
    sample_file = '/data/sat/msg/ml_train_crops/IR_108-WV_062-CMA_FULL_EXPATS_DOMAIN/2018/06/merged_MSG_CMSAF_2018-06-24.nc'
    lat, lon = load_lat_lon_grid(sample_file)
    lat_edges = np.linspace(lat.min(), lat.max(), N_DIV + 1)
    lon_edges = np.linspace(lon.min(), lon.max(), N_DIV + 1)
    lat_mids = (lat_edges[:-1] + lat_edges[1:]) / 2
    lon_mids = (lon_edges[:-1] + lon_edges[1:]) / 2

    # Load crop list
    df_labels = pd.read_csv(f"{OUTPUT_PATH}crop_list_{RUN_NAME}_{N_SUBSAMPLES}_{SAMPLING_TYPE}.csv")
    df_labels = df_labels[df_labels['label'] != -100]

    # Determine cores for parallel processing
    n_cores = max(1, os.cpu_count() - 4)

    # Loop through labels and hours
    for label in sorted(df_labels['label'].unique()):
        df_filtered = df_labels[df_labels['label'] == label].copy()
        df_filtered['datetime'] = df_filtered['path'].apply(lambda p: pd.to_datetime(extract_datetime(os.path.basename(p))['year']*1000000 +
                                                                                   extract_datetime(os.path.basename(p))['month']*10000 +
                                                                                   extract_datetime(os.path.basename(p))['day']*100 +
                                                                                   extract_datetime(os.path.basename(p))['hour'],
                                                                                   errors='coerce'))
        df_filtered.dropna(subset=['datetime'], inplace=True)

        for hour in range(24):
            df_hour = df_filtered[df_filtered['datetime'].dt.hour == hour]

            # Parallel or sequential processing
            if PARALLEL:
                results_list = Parallel(n_jobs=N_CORES)(
                    delayed(process_lat_lon_cell)(
                        i, j, lat_edges[i], lat_edges[i+1],
                        lon_edges[j], lon_edges[j+1],
                        df_hour, label, VARS, CATEG_VARS
                    ) for i in range(N_DIV) for j in range(N_DIV)
                )
            else:
                results_list = []
                for i in range(N_DIV):
                    for j in range(N_DIV):
                        res = process_lat_lon_cell(
                            i, j, lat_edges[i], lat_edges[i+1],
                            lon_edges[j], lon_edges[j+1],
                            df_hour, label, VARS, CATEG_VARS
                        )
                        results_list.append(res)


            # Initialize results dict
            first_valid = next((res for res in results_list if res), None)
            if first_valid is None:
                print(f"No valid results for label {label}, hour {hour}")
                continue
            results = {k: np.full((N_DIV, N_DIV), np.nan) for k in first_valid.keys()}

            # Fill results
            for idx, res in enumerate(results_list):
                if res:
                    i, j = divmod(idx, N_DIV)
                    for k in results.keys():
                        results[k][i, j] = res[k]

            # Save NetCDF
            save_grid_to_netcdf(results, lat_mids, lon_mids, lat_edges, lon_edges, label, hour)

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
