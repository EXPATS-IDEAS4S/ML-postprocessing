import pandas as pd
import xarray as xr
import numpy as np
import io
import os
from joblib import Parallel, delayed

from aux_functions_from_buckets import (
    extract_datetime,
    compute_categorical_values,
    find_crops_in_range
)
from get_data_from_buckets import read_file, Initialize_s3_client
from credentials_buckets import S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY, S3_ENDPOINT_URL

# Initialize S3 client
BUCKET_CMSAF_NAME = 'expats-cmsaf-cloud'
BUCKET_IMERG_NAME = 'expats-imerg-prec'
BUCKET_CROP_MSG = 'expats-msg-training'

# Configuration
run_name = 'dcv2_ir108_200x200_k9_expats_70k_200-300K_closed-CMA'
sampling_type = 'closest'
n_subsamples = 1000
vars = ['cot', 'cth', 'cma', 'cph', 'precipitation']
stats = [50, 99]
categ_vars = ['cma', 'cph']

# Load lat/lon grid
# Initialize S3
s3 = Initialize_s3_client(S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY)
bucket_filename = '/data/sat/msg/ml_train_crops/IR_108-WV_062-CMA_FULL_EXPATS_DOMAIN/2018/06/merged_MSG_CMSAF_2018-06-24.nc'
my_obj = read_file(s3, bucket_filename, BUCKET_CROP_MSG)
ds_msg_day = xr.open_dataset(io.BytesIO(my_obj))
lat = ds_msg_day['lat'].values
lon = ds_msg_day['lon'].values

# Define grid
n_div = 8
lat_edges = np.linspace(lat.min(), lat.max(), n_div + 1)
lon_edges = np.linspace(lon.min(), lon.max(), n_div + 1)
lat_mids = (lat_edges[:-1] + lat_edges[1:]) / 2
lon_mids = (lon_edges[:-1] + lon_edges[1:]) / 2

# Read crop data
output_path = f'/data1/fig/{run_name}/{sampling_type}/'
df_labels = pd.read_csv(f'{output_path}crop_list_{run_name}_{n_subsamples}_{sampling_type}.csv')

# Remove the rows with label invalid (-100)
df_labels = df_labels[df_labels['label'] != -100]

# Take a sample (for testing)
#df_labels = df_labels.sample(n=3)

# Process per unique label
unique_labels = df_labels['label'].unique()

# Function to process a single lat/lon cell
def process_lat_lon(i, j, lat_inf, lat_sup, lon_inf, lon_sup, df_filtered, label):
    # Initialize S3
    s3 = Initialize_s3_client(S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY)

    # Initialize dict 
    results = {}

    # Find the list of crops in the range
    crops_list = sorted(find_crops_in_range(df_filtered, lat_inf, lat_sup, lon_inf, lon_sup))
    if not crops_list:
        return {key: np.nan for key in results.keys()}

    data_values = {var: [] for var in vars}

    for crop_filename in crops_list:
        datetime_info = extract_datetime(crop_filename)
        datetime_obj = np.datetime64(
            f"{datetime_info['year']:04d}-{datetime_info['month']:02d}-{datetime_info['day']:02d}T"
            f"{datetime_info['hour']:02d}:{datetime_info['minute']:02d}:00"
        )
        print(f"Processing {crop_filename} for label {label}")

        for var in vars:
            if var == 'precipitation' and (datetime_info['minute'] in [15, 45]):
                continue

            bucket_filename = (
                f"MCP_{datetime_info['year']:04d}-{datetime_info['month']:02d}-{datetime_info['day']:02d}_regrid.nc"
                if var != 'precipitation' else
                f"IMERG_daily_{datetime_info['year']:04d}-{datetime_info['month']:02d}-{datetime_info['day']:02d}.nc"
            )
            bucket_name = BUCKET_CMSAF_NAME if var != 'precipitation' else BUCKET_IMERG_NAME

            try:
                my_obj = read_file(s3, bucket_filename, bucket_name)
                ds_day = xr.open_dataset(io.BytesIO(my_obj))[var]
                ds_day = ds_day.sel(lat=slice(lat_inf, lat_sup), lon=slice(lon_inf, lon_sup), time=datetime_obj)
                values = ds_day.values.flatten()

                if values.size > 0:
                    data_values[var].extend(values)

            except Exception as e:
                print(f"Error processing {var} for {crop_filename}: {e}")
                print(datetime_obj)
                print(ds_day.time.values)
                print(lat_inf, lat_sup, ds_day.lat.values)
                print(lon_inf, lon_sup, ds_day.lon.values)
                print(ds_day.values)
                exit()
                continue

    # Compute percentiles and categorical values
    for var in vars:
        values = np.array(data_values[var])
        if var in categ_vars:
            results[var] = compute_categorical_values(values, var)
        else:
            results[f"{var}-50"] = np.nanpercentile(values, 50) if values.size > 0 else np.nan
            results[f"{var}-99"] = np.nanpercentile(values, 99) if values.size > 0 else np.nan

    return results

# Process data separately for each label in parallel
for label in unique_labels:
    print(f"Processing label: {label}")

    df_filtered = df_labels[df_labels['label'] == label]

    num_cores = max(1, os.cpu_count() - 4)
    results_list = Parallel(n_jobs=num_cores)(
        delayed(process_lat_lon)(i, j, lat_edges[i], lat_edges[i+1], lon_edges[j], lon_edges[j+1], df_filtered, label)
        for i in range(n_div) for j in range(n_div)
    )

    # Convert results into a structured format
    results = {key: np.full((n_div, n_div), np.nan) for key in results_list[0].keys()}

    for idx, result in enumerate(results_list):
        i, j = divmod(idx, n_div)
        for key in results.keys():
            results[key][i, j] = result[key]

    # Create dataset
    ds = xr.Dataset(
        {key: (["lat", "lon"], value) for key, value in results.items()},
        coords={"lat": lat_mids, "lon": lon_mids}
    )

    # Add metadata
    ds.attrs["description"] = f"Processed data for label: {label}"
    ds.attrs["note"] = "Lat/Lon coordinates represent the midpoints of grid cells derived from given edges."
    ds.attrs["lat_edges"] = lat_edges.tolist()
    ds.attrs["lon_edges"] = lon_edges.tolist()
    ds["lat"].attrs["units"] = "degrees_north"
    ds["lon"].attrs["units"] = "degrees_east"

    # Save dataset
    output_filename = f"{output_path}percentile_maps_label_{label}.nc"
    ds.to_netcdf(output_filename)

    print(f"Saved {output_filename}")

