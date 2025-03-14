import pandas as pd
import xarray as xr
import numpy as np
import io
import os
from joblib import Parallel, delayed
import sys

from aux_functions_from_buckets import extract_datetime, get_num_crop, find_crops_with_coordinates, compute_categorical_values
from get_data_from_buckets import read_file, Initialize_s3_client, get_list_objects
from credentials_buckets import S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY, S3_ENDPOINT_URL
#sys.path.append(os.path.abspath("/home/Daniele/codes/visualization/cluster_analysis"))

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
lat = ds_msg_day['lat'].values
lon = ds_msg_day['lon'].values
print(len(lat), lat.min(), lat.max())
print(len(lon), lon.min(), lon.max())
exit()
#lat = np.arange(45,45.1,0.04)
#lon = np.arange(9,9.1,0.04)

# Create lat-lon grid
lon_grid, lat_grid = np.meshgrid(lon, lat, indexing='ij')

# Read crop data
n_samples = get_num_crop(run_name, extenion='tif')
output_path = f'/data1/fig/{run_name}/{sampling_type}/'
df_labels = pd.read_csv(f'{output_path}crop_list_{run_name}_{n_samples}_{sampling_type}.csv')

# Get unique class labels
unique_labels = df_labels['label'].unique()
print(f"Processing {len(unique_labels)} unique classes: {unique_labels}")

# Define function to process each (lat, lon) point for a specific class
def process_lat_lon(i, j, lat_val, lon_val, df_filtered, label):
    # Reinitialize S3 client inside the worker
    s3 = Initialize_s3_client(S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY)
    
    print(f"[Class {label}] Processing lat: {lat_val:.2f}, lon: {lon_val:.2f}")
    crops_list = find_crops_with_coordinates(df_filtered, lat_val, lon_val)
    
    if not crops_list:
        return None  # Skip if no crops

    data_values = {var: [] for var in vars}
    
    for crop_filename in crops_list:
        datetime_info = extract_datetime(crop_filename)
        year, month, day, hour, minute = datetime_info['year'], datetime_info['month'], datetime_info['day'], datetime_info['hour'], datetime_info['minute']
        datetime_obj = np.datetime64(f'{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:00')
        print(datetime_obj)

        for var in vars:
            if var == 'precipitation' and (minute == 15 or minute == 45):
                data_values[var].extend([np.nan])
                continue
            
            bucket_filename = f'MCP_{year:04d}-{month:02d}-{day:02d}_regrid.nc' if var != 'precipitation' else f'IMERG_daily_{year:04d}-{month:02d}-{day:02d}.nc'
            bucket_name = BUCKET_CMSAF_NAME if var != 'precipitation' else BUCKET_IMERG_NAME

            try:
                my_obj = read_file(s3, bucket_filename, bucket_name)
                ds_day = xr.open_dataset(io.BytesIO(my_obj))[var]
                ds_day = ds_day.sel(lat=lat_val, lon=lon_val, method='nearest')
                ds_day = ds_day.sel(time=datetime_obj)
                values = ds_day.values.flatten()
                #print(var, values)
                
                if values.size > 0:
                    data_values[var].extend(values)

            except Exception as e:
                print(f"[Class {label}] Error processing {var} for {crop_filename}: {e}")
                continue

    # Compute percentiles and categorical values
    result = {"lat": lat_val, "lon": lon_val, "label": label}
    for var in vars:
        values = np.array(data_values[var])
        if var in categ_vars:
            result[var] = compute_categorical_values(values, var)
        else:
            if values.size > 0:
                result[f"{var}-50"] = np.nanpercentile(values, 50)
                result[f"{var}-99"] = np.nanpercentile(values, 99)
            else:
                result[f"{var}-50"] = np.nan
                result[f"{var}-99"] = np.nan

    return result

# Process data separately for each unique label
for label in unique_labels:
    print(f"Starting processing for class: {label}")

    # Filter df_labels by label
    df_filtered = df_labels[df_labels['label'] == label]

    # Run parallel processing for this label
    num_cores = os.cpu_count() - 4  # Use all but 4 cores
    results = Parallel(n_jobs=num_cores)(
        delayed(process_lat_lon)(i, j, lat_grid[i, j], lon_grid[i, j], df_filtered, label)
        for i in range(lat_grid.shape[0])
        for j in range(lat_grid.shape[1])
    )

    # Remove None values (skipped locations)
    results = [res for res in results if res is not None]

    # Convert results to DataFrame and save
    df_results = pd.DataFrame(results)
    class_output_path = f'{output_path}crop_statistics_maps_{run_name}_label_{label}.csv'
    df_results.to_csv(class_output_path, index=False)
    print(f'Processing complete for class {label}. CSV saved at {class_output_path}')

