import pandas as pd
import xarray as xr
import numpy as np
import io
import os
from joblib import Parallel, delayed
import cftime
import time 

from aux_functions_from_buckets import (
    extract_datetime,
    compute_categorical_values,
    find_crops_in_range
)
from get_data_from_buckets import read_file, Initialize_s3_client
from credentials_buckets import S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY, S3_ENDPOINT_URL

# Start tracking time
start_time = time.time()

# Initialize S3 client
BUCKET_CMSAF_NAME = 'expats-cmsaf-cloud'
BUCKET_IMERG_NAME = 'expats-imerg-prec'
BUCKET_CROP_MSG = 'expats-msg-training'

# Configuration
run_name = 'dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA'
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
#df_labels = df_labels.sample(n=10)

# Process per unique label
unique_labels = sorted(df_labels['label'].unique())

# Function to process a single lat/lon cell
def process_lat_lon(i, j, lat_inf, lat_sup, lon_inf, lon_sup, df_filtered, label):
    # Initialize S3
    s3 = Initialize_s3_client(S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY)

    # Initialize dict 
    results = {}

    # Find the list of crops in the range
    crops_list = sorted(find_crops_in_range(df_filtered, lat_inf, lat_sup, lon_inf, lon_sup))
    if not crops_list:
        print(f'No crops found for label {label} at lat: {lat_inf} to {lat_sup}, lon: {lon_inf} to {lon_sup}')
        return {key: np.nan for key in results.keys()}

    data_values = {var: [] for var in vars}

    for crop_filename in crops_list:
        datetime_info = extract_datetime(crop_filename)
        #skip if year is different than 2015
        #if datetime_info['year'] != 2015:
        #    continue
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
                if isinstance(ds_day.indexes["time"], xr.CFTimeIndex):
                    ds_day["time"] = ds_day["time"].astype("datetime64[ns]")
                ds_day = ds_day.sel(lat=slice(lat_inf, lat_sup), lon=slice(lon_inf, lon_sup), time=datetime_obj)
                values = ds_day.values.flatten()

                if values.size > 0:
                    data_values[var].extend(values)

            except Exception as e:
                print(f"Error processing {var} for {crop_filename}: {e}")
                print(datetime_obj)
                print(ds_day.time.values)
                #print(lat_inf, lat_sup, ds_day.lat.values)
                #print(lon_inf, lon_sup, ds_day.lon.values)
                #print(ds_day.values)
                
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
print(unique_labels)
for label in unique_labels:
    print(label)
    print(f"Processing label: {label}")

    # Filter the dataframe by the current label
    df_filtered = df_labels[df_labels['label'] == label]

    num_cores = max(1, os.cpu_count() - 4)

    # Parallel processing to get results for each (i, j) grid
    results_list = Parallel(n_jobs=num_cores)(
        delayed(process_lat_lon)(i, j, lat_edges[i], lat_edges[i+1], lon_edges[j], lon_edges[j+1], df_filtered, label)
        for i in range(n_div) for j in range(n_div)
    )

    # Filter out empty dictionaries from the results_list
    filtered_results_list = [res for res in results_list if res]

    # Initialize the results dictionary based on the first valid (non-empty) dictionary
    if filtered_results_list:
        # Use the keys of the first non-empty dictionary to initialize results
        results = {key: np.full((n_div, n_div), np.nan) for key in filtered_results_list[0].keys()}
    else:
        results = {}  # If no valid dictionary, results will be empty

    # Now populate the results dictionary with the processed values
    for idx, result in enumerate(results_list):
        if result:  # Only process non-empty results
            i, j = divmod(idx, n_div)
            for key in results.keys():
                results[key][i, j] = result[key]

    # Convert all arrays in results to np.float32
    for key in results.keys():
        results[key] = results[key].astype(np.float32)

    # Create dataset
    ds = xr.Dataset(
        {key: (["lat", "lon"], value) for key, value in results.items()},
        coords={"lat": lat_mids, "lon": lon_mids}
    )

    #for var in ds.data_vars:
    #    ds[var] = ds[var].astype("float32")

    ds["lat"] = ds["lat"].astype("float32")
    ds["lon"] = ds["lon"].astype("float32")

    #Add metadata
    ds.attrs["description"] = f"Processed data for label: {label}"
    ds.attrs["note"] = "Lat/Lon coordinates represent the midpoints of grid cells derived from given edges."
    ds.attrs["lat_edges"] = str(lat_edges.tolist())
    ds.attrs["lon_edges"] = str(lon_edges.tolist())
    ds["lat"].attrs["units"] = "degrees_north"
    ds["lon"].attrs["units"] = "degrees_east"

    print(ds)
    ds.close()  # Close before saving

    # # Save dataset
    output_filename = f"{output_path}percentile_maps_res_{n_div}x{n_div}_label_{label}.nc"
    ds.to_netcdf(output_filename, format="NETCDF4", engine="h5netcdf")

    print(f"Saved {output_filename}")

# End of script time tracking
end_time = time.time()
print(f"Total script execution time: {end_time - start_time:.2f} seconds")

#nohup 650899