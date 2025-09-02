"""
Compute crop-level and cluster-level statistics for CMSAF and IMERG datasets.

This script extracts statistics from S3 buckets for geospatial crops
(either in GeoTIFF or NetCDF format). It supports both serial and
parallel execution, depending on the `use_parallel` flag.

Main steps:
-----------
1. Load the crop list and labels from a CSV file.
2. For each crop:
   - Parse geographic bounding box (lat/lon) and timestamp from filename.
   - Retrieve corresponding CMSAF or IMERG data files from S3.
   - Subset the data in space and time.
   - Apply masking (using CMA variable where applicable).
   - Compute either:
        * Percentiles (e.g., 50th, 99th) for continuous variables, OR
        * Category counts/proportions for categorical variables.
3. Save per-crop statistics to a CSV.
4. Aggregate per-cluster statistics (mean, std) and save to another CSV.

Inputs:
-------
- Crop list CSV (paths + labels), generated in a prior preprocessing step.
- Crop files (GeoTIFFs or NetCDFs).
- CMSAF and IMERG datasets stored in S3 buckets.

Outputs:
--------
- CSV with crop-level statistics:
  `<output_path>/crops_stats_<run_name>_<sampling_type>_<n_subsample>.csv`
- CSV with cluster-level statistics:
  `<output_path>/clusters_stats_<run_name>_<sampling_type>_<n_subsample>.csv`

Configuration:
--------------
- `use_parallel` : bool
      Whether to use parallel processing via joblib or run serially.
- `data_format` : {'nc','tif'}
      Format of crop files.
- `vars`, `stats`, `categ_vars` : list
      Variables to process and statistics to compute.
- Bucket names, credentials, run_name, paths, etc.

Notes:
------
- Designed for large-scale analysis of satellite datasets stored on S3.
- Parallel processing significantly speeds up computations but increases memory usage.
"""


import pandas as pd
import xarray as xr
import numpy as np
import os, sys, io
from glob import glob
from joblib import Parallel, delayed

from aux_functions_from_buckets import (
    extract_coordinates, extract_datetime,
    compute_categorical_values, filter_cma_values,
    extract_coord_from_nc, extract_datetime_from_nc
)
from get_data_from_buckets import read_file, Initialize_s3_client


sys.path.append(os.path.abspath("/home/Daniele/codes/visualization/cluster_analysis"))  
from aux_functions import compute_percentile
from utils.configs import load_config
from utils.buckets.credentials_buckets import S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY, S3_ENDPOINT_URL

# ---------------- CONFIG ---------------- #
BUCKET_CMSAF_NAME = 'expats-cmsaf-cloud'
BUCKET_IMERG_NAME = 'expats-imerg-prec'

run_name       = 'dcv2_ir108_ot_100x100_k9_35k_nc_vit'
crops_name     = 'dcv2_ir108_OT_100x100_35k_nc'
sampling_type  = 'all'
epoch          = 500
data_format    = 'nc'   # 'nc' or 'tif'
vars           = ['cot','cth','cma','cph','precipitation']
stats          = [50, 99]
categ_vars     = ['cma','cph']
use_parallel   = True   # <--- SWITCH between parallel or serial
# ---------------------------------------- #

# build table entries
table_entries = [f"{var}-{num}" for var in vars if var not in categ_vars for num in stats]
table_entries.extend(categ_vars)
table_entries = [item if '-' in item else f"{item}-None" for item in table_entries]

# input/output paths
image_crops_path = f'/data1/crops/{crops_name}/{data_format}/1/'
output_path = f'/data1/fig/{run_name}/epoch_{epoch}/{sampling_type}/'
list_image_crops = sorted(glob(image_crops_path+'*.tif'))
n_samples = len(list_image_crops)
print('n samples:', n_samples)

n_subsample = n_samples if sampling_type == 'all' else 1000

# load crop list
df_labels = pd.read_csv(f'{output_path}crop_list_{run_name}_{sampling_type}_{n_subsample}.csv')
print(f"Loaded {len(df_labels)} rows from crop list.")

# ------------------------------------------------ #
# Core processing function (used in both modes)
# ------------------------------------------------ #
def process_row(row):
    s3 = Initialize_s3_client(S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY)
    crop_filename = os.path.basename(row['path'])

    # get lat/lon and datetime from filename
    if data_format == 'nc':
        coords = extract_coord_from_nc(crop_filename, image_crops_path)
        datetime_info = extract_datetime_from_nc(crop_filename, image_crops_path)
    else:
        coords = extract_coordinates(crop_filename)
        datetime_info = extract_datetime(crop_filename)

    lat_min, lat_max = coords['lat_min'], coords['lat_max']
    lon_min, lon_max = coords['lon_min'], coords['lon_max']
    year, month, day, hour, minute = (
        datetime_info[k] for k in ['year','month','day','hour','minute']
    )
    base_time = np.datetime64(f'{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:00')

    row_data = {}
    for var in vars:
        for stat in stats if var not in categ_vars else ['None']:
            entry = f"{var}-{stat}"
            try:
                if var == 'precipitation':
                    bucket_name = BUCKET_IMERG_NAME
                    bucket_filename = f'IMERG_daily_{year:04d}-{month:02d}-{day:02d}.nc'
                    time_to_use = [base_time]
                else:
                    bucket_name = BUCKET_CMSAF_NAME
                    bucket_filename = f'MCP_{year:04d}-{month:02d}-{day:02d}_regrid.nc'
                    time_to_use = (
                        [base_time, base_time + np.timedelta64(15,'m')]
                        if 'precipitation' in vars else [base_time]
                    )

                my_obj = read_file(s3, bucket_filename, bucket_name)
                ds_day = xr.open_dataset(io.BytesIO(my_obj))[var]

                # also open CMA for masking
                my_obj_cma = read_file(s3, f'MCP_{year:04d}-{month:02d}-{day:02d}_regrid.nc', BUCKET_CMSAF_NAME)
                ds_day_cma = xr.open_dataset(io.BytesIO(my_obj_cma))['cma']

                if isinstance(ds_day.indexes["time"], xr.CFTimeIndex):
                    ds_day["time"] = ds_day["time"].astype("datetime64[ns]")
                    ds_day_cma["time"] = ds_day_cma["time"].astype("datetime64[ns]")

                # spatial + temporal subsetting
                ds_day = ds_day.sel(lat=slice(lat_min,lat_max), lon=slice(lon_min,lon_max))
                ds_subset = ds_day.sel(time=time_to_use)
                ds_day_cma = ds_day_cma.sel(lat=slice(lat_min,lat_max), lon=slice(lon_min,lon_max))
                ds_subset_cma = ds_day_cma.sel(time=time_to_use)

                values = ds_subset.values.flatten()
                values_cma = ds_subset_cma.values.flatten()
                values = filter_cma_values(values, values_cma, var)

                if stat == 'None':
                    result = compute_categorical_values(values, var)
                else:
                    result = compute_percentile(values, int(stat))

            except Exception as e:
                print(f"Error processing {entry} for {row['path']}: {e}")
                result = np.nan

            row_data[entry] = result

    # add metadata
    row_data.update({
        'month': int(month),
        'hour': int(hour),
        'lat_mid': (lat_min + lat_max)/2,
        'lon_mid': (lon_min + lon_max)/2
    })
    return row_data

# ------------------------------------------------ #
# Run serial or parallel depending on config
# ------------------------------------------------ #
if use_parallel:
    num_cores = max(1, os.cpu_count()-5)
    results = Parallel(n_jobs=num_cores)(delayed(process_row)(row) for _, row in df_labels.iterrows())
else:
    results = [process_row(row) for _, row in df_labels.iterrows()]

# merge results with labels
df_results = pd.DataFrame(results)
df_labels = pd.concat([df_labels, df_results], axis=1)

# save per-crop stats
df_labels.to_csv(f'{output_path}crops_stats_{run_name}_{sampling_type}_{n_subsample}.csv', index=False)
print("Crop stats saved.")

# compute overall stats
continuous_stats = df_labels.groupby('label').agg(['mean','std'])
continuous_stats.columns = ['_'.join(col).strip() for col in continuous_stats.columns.values]
continuous_stats.reset_index(inplace=True)
continuous_stats.to_csv(f'{output_path}clusters_stats_{run_name}_{sampling_type}_{n_subsample}.csv', index=False)
print("Cluster-level stats saved.")
