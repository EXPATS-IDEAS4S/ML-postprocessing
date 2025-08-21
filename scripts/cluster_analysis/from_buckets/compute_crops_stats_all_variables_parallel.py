import pandas as pd
import xarray as xr
import numpy as np
import io
from glob import glob
import os
from joblib import Parallel, delayed
import sys
import boto3


from aux_functions_from_buckets import extract_coordinates, extract_datetime, compute_categorical_values, extract_variable_values, filter_cma_values, extract_coord_from_nc, extract_datetime_from_nc
from get_data_from_buckets import read_file, Initialize_s3_client
from credentials_buckets import S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY, S3_ENDPOINT_URL
sys.path.append(os.path.abspath("/home/Daniele/codes/visualization/cluster_analysis"))  
from aux_functions import compute_percentile

# Initialize S3 client
#s3 = Initialize_s3_client(S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY)

BUCKET_CMSAF_NAME = 'expats-cmsaf-cloud'
BUCKET_IMERG_NAME = 'expats-imerg-prec'

run_name = 'dcv2_ir108_ot_100x100_k9_35k_nc_vit'
crops_name = 'dcv2_ir108_OT_100x100_35k_nc'  # Name of the crops
sampling_type = 'all'
epoch = 500  # Epoch number for the run
data_format = 'nc'  # Image file extension
vars = ['cot', 'cth', 'cma', 'cph', 'precipitation']
stats = [50, 99]
categ_vars = ['cma', 'cph']
filter_daytime = False        # Enable daytime filter (06–16 UTC)
filter_imerg_minutes = False  # Only keep timestamps with minutes 00 or 30

filter_tags = []
if filter_daytime:
    filter_tags.append("daytime")
if filter_imerg_minutes:
    filter_tags.append("imergmin")

filter_suffix = "_" + "_".join(filter_tags) if filter_tags else ""

# Read data
# List of the image crops
image_crops_path = f'/data1/crops/{crops_name}/{data_format}/1/'
list_image_crops = sorted(glob(image_crops_path+'*.tif'))
n_samples = len( list_image_crops)
print('n samples: ', n_samples)

# Read data
if sampling_type == 'all':
    n_subsample = 35059  # Number of samples per cluster
else:
    n_subsample = 1000

# Generate column names
table_entries = [f"{var}-{num}" for var in vars if var not in categ_vars for num in stats]
table_entries.extend(categ_vars)
table_entries = [item if '-' in item else f"{item}-None" for item in table_entries]

# Path to fig folder for outputs
output_path = f'/data1/fig/{run_name}/epoch_{epoch}/{sampling_type}/'

# Load CSV file with the crops path and labels into a pandas DataFrame

df_labels = pd.read_csv(f'{output_path}crop_list_{run_name}_{sampling_type}_{n_subsample}{filter_suffix}.csv')
print(len(df_labels['path'].to_list()))

# Function to process each row in parallel
def process_row(row):
    s3 = Initialize_s3_client(S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY)

    crop_filename = os.path.basename(row['path'])
    print(crop_filename)
    if data_format == 'nc':
        coords = extract_coord_from_nc(crop_filename, image_crops_path)
        datetime_info = extract_datetime_from_nc(crop_filename, image_crops_path)
    else:
        coords = extract_coordinates(crop_filename)
        datetime_info = extract_datetime(crop_filename)
    print(coords)
    print(datetime_info)
    exit()

    lat_min, lat_max = coords['lat_min'], coords['lat_max']
    lon_min, lon_max = coords['lon_min'], coords['lon_max']
    year, month, day, hour, minute = (
        datetime_info[k] for k in ['year', 'month', 'day', 'hour', 'minute']
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

                    # Merge timestamps ONLY if 'precipitation' is among vars
                    time_to_use = (
                        [base_time, base_time + np.timedelta64(15, 'm')]
                        if 'precipitation' in vars else [base_time]
                    )

                my_obj = read_file(s3, bucket_filename, bucket_name)
                ds_day = xr.open_dataset(io.BytesIO(my_obj))[var]

                # Open CMA values
                my_obj_cma = read_file(s3,  f'MCP_{year:04d}-{month:02d}-{day:02d}_regrid.nc', BUCKET_CMSAF_NAME)
                ds_day_cma = xr.open_dataset(io.BytesIO(my_obj_cma))['cma']

                if isinstance(ds_day.indexes["time"], xr.CFTimeIndex):
                    ds_day["time"] = ds_day["time"].astype("datetime64[ns]")
                    ds_day_cma["time"] = ds_day_cma["time"].astype("datetime64[ns]")

                ds_day = ds_day.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
                ds_subset = ds_day.sel(time=time_to_use)
                ds_day_cma = ds_day_cma.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
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

    row_data.update({
        'month': int(month),
        'hour': int(hour),
        'lat_mid': (lat_min + lat_max) / 2,
        'lon_mid': (lon_min + lon_max) / 2
    })

    return row_data


# Run parallel processing
num_cores = os.cpu_count() - 5  # Use all but one CPU
results = Parallel(n_jobs=num_cores)(delayed(process_row)(row) for _, row in df_labels.iterrows())

# Convert results to DataFrame
df_results = pd.DataFrame(results)

# Merge results with original DataFrame
df_labels = pd.concat([df_labels, df_results], axis=1)

# Save output
df_labels.to_csv(f'{output_path}crops_stats_{run_name}_{sampling_type}_{n_subsample}{filter_suffix}.csv', index=False)
print('Parallelized processing complete. CSV saved!')


# # Compute stats for continuous variables
# continuous_stats = df_labels.groupby('label').agg(['mean', 'std'])
# continuous_stats.columns = ['_'.join(col).strip() for col in continuous_stats.columns.values]
# continuous_stats.reset_index(inplace=True)

# #Save continuous stats to a CSV file
# continuous_stats.to_csv(f'{output_path}clusters_stats_{run_name}_{sampling_type}_{n_subsample}.csv', index=False)
# print('Overall Stats for each cluster are saved to CSV files.')

#