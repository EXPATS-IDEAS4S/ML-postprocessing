import pandas as pd
import xarray as xr
import numpy as np
import io
import os
from glob import glob
from joblib import Parallel, delayed
import sys
import boto3

from aux_functions_from_buckets import extract_datetime, get_num_crop, find_crops_with_coordinates, compute_categorical_values
from get_data_from_buckets import read_file, Initialize_s3_client, get_list_objects
from credentials_buckets import S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY, S3_ENDPOINT_URL
sys.path.append(os.path.abspath("/home/Daniele/codes/visualization/cluster_analysis"))
from aux_functions import compute_percentile

# Initialize S3 client
BUCKET_CMSAF_NAME = 'expats-cmsaf-cloud'
BUCKET_IMERG_NAME = 'expats-imerg-prec'
BUCKET_CROP_MSG = 'expats-msg-training'

run_name = 'dcv2_ir108_200x200_k9_expats_70k_200-300K_closed-CMA'
sampling_type = 'all'
vars = ['cot', 'cth', 'cma', 'cph', 'precipitation']
stats = [50, 99]
categ_vars = ['cma', 'cph']

# Open bucket to retrieve lat/lon grid
s3 = Initialize_s3_client(S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY)
bucket_filename = f'/data/sat/msg/ml_train_crops/IR_108-WV_062-CMA_FULL_EXPATS_DOMAIN/2018/06/merged_MSG_CMSAF_2018-06-24.nc' 
my_obj = read_file(s3, bucket_filename, BUCKET_CROP_MSG)
ds_msg_day = xr.open_dataset(io.BytesIO(my_obj))
#lat = ds_msg_day['lat'].values
#lon = ds_msg_day['lon'].values
lat = np.arange(44.0, 44.16, 0.04)
lon = np.arange(8., 8.16, 0.04)

# Create lat-lon grid
lon_grid, lat_grid = np.meshgrid(lon, lat, indexing='ij')

# Read crop data
n_samples = get_num_crop(run_name, extenion='tif')
output_path = f'/data1/fig/{run_name}/{sampling_type}/'
df_labels = pd.read_csv(f'{output_path}crop_list_{run_name}_{n_samples}_{sampling_type}.csv')

# Take a random sample of the rows in df_labels
df_labels = df_labels.sample(n=5)

# Prepare storage for results
results = {f"{var}-{stat}": np.full(lon_grid.shape, np.nan) for var in vars if var not in categ_vars for stat in stats}
for var in categ_vars:
    results[var] = np.full(lon_grid.shape, np.nan)

# Process each lat/lon location
for i in range(lat_grid.shape[0]):
    for j in range(lat_grid.shape[1]):
        lat_val = lat_grid[i, j]
        lon_val = lon_grid[i, j]

        print(f"Processing lat: {lat_val:.2f}, lon: {lon_val:.2f}")

        crops_list = find_crops_with_coordinates(df_labels, lat_val, lon_val)
        if not crops_list:
            continue
        len_crops = len(crops_list)
        print(f"Found {len_crops} crops")

        # Prepare storage for data values
        data_values = {var: [] for var in vars}

        for crop_filename in crops_list:
            datetime_info = extract_datetime(crop_filename)
            year, month, day, hour, minute = datetime_info['year'], datetime_info['month'], datetime_info['day'], datetime_info['hour'], datetime_info['minute']
            datetime_obj = np.datetime64(f'{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:00')

            for var in vars:
                if var == 'precipitation' and (minute == 15 or minute == 45):
                    data_values[var].extend([np.nan])
                    print(f"Data values for {var}: {data_values[var]}")
                    continue
                else:
                    bucket_filename = f'MCP_{year:04d}-{month:02d}-{day:02d}_regrid.nc' if var != 'precipitation' else f'IMERG_daily_{year:04d}-{month:02d}-{day:02d}.nc'
                    bucket_name = BUCKET_CMSAF_NAME if var != 'precipitation' else BUCKET_IMERG_NAME

                    try:
                        my_obj = read_file(s3, bucket_filename, bucket_name)
                        ds_day = xr.open_dataset(io.BytesIO(my_obj))[var]
                        ds_day = ds_day.sel(lat=lat_val, lon=lon_val, method='nearest')
                        ds_day = ds_day.sel(time=datetime_obj)
                        values = ds_day.values.flatten()
                        #print(f"Values for {var}: {values}")

                        if values.size > 0:
                            data_values[var].extend(values)
                            print(f"Data values for {var}: {data_values[var]}")

                    except Exception as e:
                        print(f"Error processing {var} for {crop_filename}: {e}")
                        continue

        # Compute percentiles and categorical values
        for var in vars:
            values = np.array(data_values[var])
            if var in categ_vars:
                results[var][i, j] = compute_categorical_values(values, var)
            else:
                if values.size > 0:
                    results[f"{var}-50"][i, j] = np.nanpercentile(values, 50)
                    results[f"{var}-99"][i, j] = np.nanpercentile(values, 99)

#Save results in csv
df_results = pd.DataFrame(results)
df_results.to_csv(f'{output_path}crop_statistics_maps_{run_name}.csv')
print(f'Processing complete. CSV saved at {output_path}crop_statistics_maps_{run_name}.csv')

# # Convert results to xarray dataset
# ds = xr.Dataset(
#     {key: (["lat", "lon"], results[key]) for key in results},
#     coords={"lat": lat, "lon": lon}
# )
# print(ds)
# # Save to NetCDF
# nc_filename = f'{output_path}crop_statistics_maps_{run_name}.nc'
# ds.to_netcdf(nc_filename)
# print(f'Processing complete. NetCDF saved at {nc_filename}')
