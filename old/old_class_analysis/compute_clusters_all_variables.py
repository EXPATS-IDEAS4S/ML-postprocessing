##########################################################
## Compute stats and plot distr for CMSAF variables ##
##########################################################

import pandas as pd
import xarray as xr
import numpy as np
from glob import glob
import sys
import os
import io

from aux_functions_from_buckets import extract_coordinates, extract_datetime, compute_categorical_values
from get_data_from_buckets import read_file, Initialize_s3_client, get_list_objects
from utils.buckets.credentials_buckets import S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY, S3_ENDPOINT_URL
sys.path.append(os.path.abspath("/home/Daniele/codes/visualization/cluster_analysis"))  
from aux_functions import compute_percentile, concatenate_values, extend_labels, plot_single_vars, pick_variable, find_latlon_boundaries_from_ds, get_time_from_ds, select_ds, plot_joyplot

BUCKET_CMSAF_NAME = 'expats-cmsaf-cloud'
BUCKET_IMERG_NAME = 'expats-imerg-prec'

run_name = 'dcv2_ir108_200x200_k9_expats_70k_200-300K_closed-CMA'

# Define sampling type
sampling_type = 'all'  # Options: 'random', 'closest', 'farthest', 'all'

# Pick the statistics to compute for each crop, the percentile values
vars = ['cot','cth','cma','cph', 'precipitation']
stats = [50, 99] 
categ_vars = ['cma','cph']

# List of the image crops
image_crops_path = f'/data1/crops/{run_name}/1/'
list_image_crops = sorted(glob(image_crops_path+'*.tif'))

# Read data
n_samples = len( list_image_crops)
print('n samples: ', n_samples)

# Read data
if sampling_type == 'all':
    n_subsample = n_samples  # Number of samples per cluster
else:
    n_subsample = 100

# Select the correct varable information based on the data type  
#vars, vars_long_name, vars_units, vars_logscale, vars_dir = pick_variable('cmdaf')	

# Generate combinations for elements NOT in categ_list
table_entries = [f"{var}-{num}" for var in vars if var not in categ_vars for num in stats]

# Add elements from categ_list as they are
table_entries.extend(categ_vars)
table_entries = [item if '-' in item else f"{item}-None" for item in table_entries]
print(table_entries)

# Path to fig folder for outputs
output_path = f'/data1/fig/{run_name}/{sampling_type}/'

# Load CSV file with the crops path and labels into a pandas DataFrame
df_labels = pd.read_csv(f'{output_path}crop_list_{run_name}_{n_subsample}_{sampling_type}.csv')
print(df_labels)

# Initialize lists to hold data for continuous and categorical variables
cmsaf_data = {var: [] for var in table_entries}

# Get list of objects in the bucket
s3 = Initialize_s3_client(S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY)
#get_list_objects(s3, BUCKET_IMERG_NAME)
#get_list_objects(s3, BUCKET_CMSAF_NAME)

# Read the .nc files and extract data
for index, row in df_labels.iterrows():
    crop_filename = row['path'].split('/')[-1]
    #print(crop_filename)

    # get lat lon and time info from Dataset
    coords = extract_coordinates(crop_filename)
    lat_min, lat_max, lon_min, lon_max = coords['lat_min'], coords['lat_max'], coords['lon_min'], coords['lon_max']
    #print(lat_min, lat_max, lon_min, lon_max)
    #print(coords)
    
    #print(lat_min, lat_max, lon_min, lon_max)
    datetime_info = extract_datetime(crop_filename)
    year, month, day, hour, minute = datetime_info['year'], datetime_info['month'], datetime_info['day'], datetime_info['hour'], datetime_info['minute']    
    #print(year, month, day, hour)
    
    # Create a datetime64 object with minute and second set to 0
    datetime_obj = np.datetime64(f'{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:00')
    print(datetime_obj)

    for entry in table_entries:
        var = entry.split('-')[0]
        stat = entry.split('-')[1]
        #print(var, stat)

        # Deal with the fact that IMERG data are available every 30 minutes
        if var == 'precipitation' and (minute == 15 or minute == 45):
            values = np.nan
        else:

            if var != 'precipitation':	
                bucket_filename = f'MCP_{year:04d}-{month:02d}-{day:02d}_regrid.nc'

                # Read the cmsaf file from the bucket
                my_obj = read_file(s3, bucket_filename, BUCKET_CMSAF_NAME)
            else:
                bucket_filename = f'IMERG_daily_{year:04d}-{month:02d}-{day:02d}.nc'

                # Read the imerg file from the bucket
                my_obj = read_file(s3, bucket_filename, BUCKET_IMERG_NAME)

            try:
                ds_day = xr.open_dataset(io.BytesIO(my_obj))

                if isinstance(ds_day.indexes["time"], xr.CFTimeIndex):
                    ds_day["time"] = ds_day["time"].astype("datetime64[ns]")

                # Select the variable of interest
                ds_day  = ds_day[var]

                # Select the region of interest
                ds_day = ds_day.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

                #Select the Time of interest
            
                ds_day = ds_day.sel(time=datetime_obj)
                #print(ds_day)
                
                values = ds_day.values.flatten()
            
                # Apply percentile if 'stat' is provided (stat can be a percentile or another function)
                if stat != 'None':
                    try:
                        values = compute_percentile(values, int(stat))
                    except IndexError:
                        print(f"Not enough values to calculate {stat} for {var} in {row['path']}")
                        values = np.nan
                        #continue  # Skip if there are not enough values for the given percentile
                else:
                    values = compute_categorical_values(values, var)
            except KeyError:
                print(f"Data not available for {var} in {row['path']}")
                values = np.nan

        # Ensure values is a list or array before extending
        df_labels.at[index, entry] = values
        #continuous_data = concatenate_values(values, var, continuous_data)

    # Calculate the midpoints of latitude and longitude
    lat_mid = (lat_min + lat_max) / 2
    lon_mid = (lon_min + lon_max) / 2

    df_labels.at[index, 'month'] = int(month)
    df_labels.at[index, 'hour'] = int(hour)
    df_labels.at[index, 'lat_mid'] = lat_mid
    df_labels.at[index, 'lon_mid'] = lon_mid

    # print the current row
    #print(df_labels.loc[index])


# Save in case of crop stats are calculated
df_labels.to_csv(f'{output_path}crops_stats_{run_name}_{sampling_type}_{n_subsample}.csv', index=False)
print('Stats for each crop are saved to CSV files.')

# Compute stats for continuous variables
continuous_stats = df_labels.groupby('label').agg(['mean', 'std'])
continuous_stats.columns = ['_'.join(col).strip() for col in continuous_stats.columns.values]
continuous_stats.reset_index(inplace=True)

#Save continuous stats to a CSV file
continuous_stats.to_csv(f'{output_path}clusters_stats_{run_name}_{sampling_type}_{n_subsample}.csv', index=False)
print('Overall Stats for each cluster are saved to CSV files.')

#nohup 689381
