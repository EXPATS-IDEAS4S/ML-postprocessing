import pandas as pd
import xarray as xr
import numpy as np
import io
from glob import glob
import os
from joblib import Parallel, delayed
import sys
import boto3


from aux_functions_from_buckets import extract_coordinates, extract_datetime, compute_categorical_values
from get_data_from_buckets import read_file, Initialize_s3_client
from credentials_buckets import S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY, S3_ENDPOINT_URL
sys.path.append(os.path.abspath("/home/Daniele/codes/visualization/cluster_analysis"))  
from aux_functions import compute_percentile

# Initialize S3 client
#s3 = Initialize_s3_client(S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY)

BUCKET_CMSAF_NAME = 'expats-cmsaf-cloud'
BUCKET_IMERG_NAME = 'expats-imerg-prec'

run_name = 'dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA'
sampling_type = 'all'
vars = ['cot', 'cth', 'cma', 'cph', 'precipitation']
stats = [50, 99]
categ_vars = ['cma', 'cph']

# Read data
# List of the image crops
image_crops_path = f'/data1/crops/{run_name}/1/'
list_image_crops = sorted(glob(image_crops_path+'*.tif'))
n_samples = len( list_image_crops)
print('n samples: ', n_samples)

# Read data
if sampling_type == 'all':
    n_subsample = n_samples  # Number of samples per cluster
else:
    n_subsample = 1000

# Generate column names
table_entries = [f"{var}-{num}" for var in vars if var not in categ_vars for num in stats]
table_entries.extend(categ_vars)
table_entries = [item if '-' in item else f"{item}-None" for item in table_entries]

# Path to fig folder for outputs
output_path = f'/data1/fig/{run_name}/{sampling_type}/'

# Load CSV file with the crops path and labels into a pandas DataFrame
df_labels = pd.read_csv(f'{output_path}crop_list_{run_name}_{n_subsample}_{sampling_type}.csv')
print(df_labels)

# Function to process each row in parallel
def process_row(row):
    # Reinitialize S3 client inside the worker
    s3 = Initialize_s3_client(S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY)

    crop_filename = row['path'].split('/')[-1]
    coords = extract_coordinates(crop_filename)
    lat_min, lat_max, lon_min, lon_max = coords['lat_min'], coords['lat_max'], coords['lon_min'], coords['lon_max']
    
    datetime_info = extract_datetime(crop_filename)
    year, month, day, hour, minute = datetime_info['year'], datetime_info['month'], datetime_info['day'], datetime_info['hour'], datetime_info['minute']

    datetime_obj = np.datetime64(f'{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:00')
    print(f"Processing {crop_filename} at {datetime_obj}")

    row_data = {}

    for entry in table_entries:
        var, stat = entry.split('-')
        
        if var == 'precipitation' and (minute == 15 or minute == 45):
            row_data[entry] = np.nan
        else:
            bucket_filename = f'MCP_{year:04d}-{month:02d}-{day:02d}_regrid.nc' if var != 'precipitation' else f'IMERG_daily_{year:04d}-{month:02d}-{day:02d}.nc'
            bucket_name = BUCKET_CMSAF_NAME if var != 'precipitation' else BUCKET_IMERG_NAME

            try:
                my_obj = read_file(s3, bucket_filename, bucket_name)
                ds_day = xr.open_dataset(io.BytesIO(my_obj))[var]

                if isinstance(ds_day.indexes["time"], xr.CFTimeIndex):
                    ds_day["time"] = ds_day["time"].astype("datetime64[ns]")

                ds_day = ds_day.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

                # Select time (use nearest if exact time is missing)
                ds_day = ds_day.sel(time=datetime_obj)#, method='nearest', tolerance=np.timedelta64(1, 'h'))
                values = ds_day.values.flatten()

                if stat != 'None':
                    values = compute_percentile(values, int(stat))
                else:
                    values = compute_categorical_values(values, var)

            except Exception as e:
                print(f"Error processing {var} for {row['path']}: {e}")
                values = np.nan

            row_data[entry] = values

    row_data.update({
        'month': int(month),
        'hour': int(hour),
        'lat_mid': (lat_min + lat_max) / 2,
        'lon_mid': (lon_min + lon_max) / 2
    })

    return row_data

# Run parallel processing
num_cores = os.cpu_count() - 3  # Use all but one CPU
results = Parallel(n_jobs=num_cores)(delayed(process_row)(row) for _, row in df_labels.iterrows())

# Convert results to DataFrame
df_results = pd.DataFrame(results)

# Merge results with original DataFrame
df_labels = pd.concat([df_labels, df_results], axis=1)

# Save output
df_labels.to_csv(f'{output_path}crops_stats_{run_name}_{sampling_type}_{n_subsample}.csv', index=False)
print('Parallelized processing complete. CSV saved!')


# # Compute stats for continuous variables
# continuous_stats = df_labels.groupby('label').agg(['mean', 'std'])
# continuous_stats.columns = ['_'.join(col).strip() for col in continuous_stats.columns.values]
# continuous_stats.reset_index(inplace=True)

# #Save continuous stats to a CSV file
# continuous_stats.to_csv(f'{output_path}clusters_stats_{run_name}_{sampling_type}_{n_subsample}.csv', index=False)
# print('Overall Stats for each cluster are saved to CSV files.')