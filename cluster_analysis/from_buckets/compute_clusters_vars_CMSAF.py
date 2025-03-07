##########################################################
## Compute stats and plot distr for CMSAF variables ##
##########################################################

import pandas as pd
import xarray as xr
import numpy as np
from glob import glob
import sys
import os

sys.path.append(os.path.abspath("/home/Daniele/codes/visualization/cluster_analysis"))  
from aux_functions import compute_percentile, concatenate_values, extend_labels, plot_single_vars, pick_variable, find_latlon_boundaries_from_ds, get_time_from_ds, select_ds, plot_joyplot

run_name = 'dcv2_ir108_200x200_k9_expats_70k_200-300K_closed-CMA'

# Define sampling type
sampling_type = 'all'  # Options: 'random', 'closest', 'farthest', 'all'

data_type = 'cmsaf'

# Pick the statistics to compute for each crop, the percentile values
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
vars, vars_long_name, vars_units, vars_logscale, vars_dir = pick_variable(data_type)

# Generate combinations for elements NOT in categ_list
table_entries = [f"{var}-{num}" for var in vars if var not in categ_vars for num in stats]

# Add elements from categ_list as they are
table_entries.extend(categ_vars)

print(table_entries)

# Path to fig folder for outputs
output_path = f'/home/Daniele/fig/{run_name}/{sampling_type}/'

# Load CSV file with the crops path and labels into a pandas DataFrame
df_labels = pd.read_csv(f'{output_path}crop_list_{run_name}_{n_subsample}_{sampling_type}.csv')
print(df_labels)

# Initialize lists to hold data for continuous and categorical variables
cmsaf_data = {var: [] for var in table_entries}

# Read the .nc files and extract data
for i, row in df_labels.iterrows():
    ds_crops = xr.open_dataset(row['path'])

    # get lat lon and time info from Dataset
    lat_min, lat_max, lon_min, lon_max = find_latlon_boundaries_from_ds(ds_crops)
    #print(lat_min, lat_max, lon_min, lon_max)
    year, month, day, hour = get_time_from_ds(ds_crops)
    print(year, month, day, hour)
    exit()
    
    # Create a datetime64 object with minute and second set to 0
    datetime_obj = np.datetime64(f'{year:04d}-{month:02d}-{day:02d}T{hour:02d}:00:00')

    for var in vars:
        #print(var)

        # give the variable, select the correct dataset
        ds = select_ds(var, [ds_crops])#, ds_era5_time, ds_dem, ds_lsm])
        #print(ds)

        values = ds[var].sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)).values.flatten()
    
        # Apply percentile if 'stat' is provided (stat can be a percentile or another function)
        if stat:
            try:
                values = compute_percentile(values, stat)
            except IndexError:
                print(f"Not enough values to calculate {stat} for {var} in {row['path']}")
                continue  # Skip if there are not enough values for the given percentile

        # Ensure values is a list or array before extending
        continuous_data = concatenate_values(values, var, continuous_data)

    # Extend labels based on the number of valid entries for this dataset
    labels = extend_labels(values, labels, row, 'label')
    crop_names = extend_labels(values, crop_names, row, 'path')
    crop_distances = extend_labels(values, crop_distances, row, 'distance') 

# Create DataFrames for continuous and categorical variables
df_continuous = pd.DataFrame(continuous_data)
df_continuous['label'] = labels
df_continuous['path'] = crop_names
df_continuous['distance'] = crop_distances  

print(df_continuous)

# Save in case of crop stats are calculated
df_continuous.to_csv(f'{output_path}{data_type}_crops_stats_{run_name}_{sampling_type}_{n_subsample}_{stat}.csv', index=False)
print('Continous Stats for each crop are saved to CSV files.')

# Compute stats for continuous variables
continuous_stats = df_continuous.groupby('label').agg(['mean', 'std'])
continuous_stats.columns = ['_'.join(col).strip() for col in continuous_stats.columns.values]
continuous_stats.reset_index(inplace=True)

#Save continuous stats to a CSV file
continuous_stats.to_csv(f'{output_path}{data_type}_clusters_stats_{run_name}_{sampling_type}_{n_subsample}_{stat}.csv', index=False)
print('Overall Continous Stats for each cluster are saved to CSV files.')


