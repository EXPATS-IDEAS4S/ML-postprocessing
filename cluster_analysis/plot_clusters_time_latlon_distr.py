##########################################################
## Compute stats and plot distr for time and lat lon    ##
##########################################################

import pandas as pd
import xarray as xr

from aux_functions import concatenate_values, extend_labels, plot_single_vars, pick_variable, find_latlon_boundaries_from_ds, get_time_from_ds

run_names = ['10th-90th_CMA', '10th-90th']

# Define sampling type
sampling_type = 'closest'  # Options: 'random', 'closest', 'farthest', 'all'

stat = None

# Read data
n_subsample = 1000  # Number of samples per cluster

data_type = 'space-time'  #'continuous' 'topography' 'era5-land'

# Select the correct varable information based on the data type  
vars, vars_long_name, vars_units, vars_logscale, vars_dir = pick_variable(data_type)

for run_name in run_names:

    # Path to fig folder for outputs
    output_path = f'/home/daniele/Documenti/Data/Fig/{run_name}/{sampling_type}/'

    # Load CSV file with the crops path and labels into a pandas DataFrame
    df_labels = pd.read_csv(f'{output_path}crop_list_{run_name}_{n_subsample}_{sampling_type}.csv')

    # Initialize lists to hold data for continuous and categorical variables
    continuous_data = {var: [] for var in vars}
    labels = []

    # Read the .nc files and extract data
    for i, row in df_labels.iterrows():
        ds_crops = xr.open_dataset(row['path'])

        # get lat lon and time info from Dataset
        lat_min, lat_max, lon_min, lon_max = find_latlon_boundaries_from_ds(ds_crops)
        year, month, day, hour = get_time_from_ds(ds_crops)

        # Calculate the midpoints of latitude and longitude
        lat_mid = (lat_min + lat_max) / 2
        lon_mid = (lon_min + lon_max) / 2

        # Ensure values is a list or array before extending
        continuous_data = concatenate_values(month, 'month', continuous_data)
        continuous_data = concatenate_values(hour, 'hour', continuous_data)
        continuous_data = concatenate_values(lat_mid, 'lat_mid', continuous_data)
        continuous_data = concatenate_values(lon_mid, 'lon_mid', continuous_data)

        # Extend labels based on the number of valid entries for this dataset
        labels = extend_labels(month, labels, row)


    # Create DataFrames for continuous and categorical variables
    df_continuous = pd.DataFrame(continuous_data)
    df_continuous['label'] = labels

    print(df_continuous)

    # Save in case of crop stats are calculated 
    df_continuous.to_csv(f'{output_path}continuous_{data_type}_crops_stats_{run_name}_{sampling_type}_{n_subsample}_{stat}.csv', index=False)
    print('Continous Stats for each crop are saved to CSV files.')

    # Compute stats for continuous variables
    continuous_stats = df_continuous.groupby('label').agg(['mean', 'std'])
    continuous_stats.columns = ['_'.join(col).strip() for col in continuous_stats.columns.values]
    continuous_stats.reset_index(inplace=True)

    # Save continuous stats to a CSV file
    continuous_stats.to_csv(f'{output_path}continuous_{data_type}_clusters_stats_{run_name}_{sampling_type}_{n_subsample}_{stat}.csv', index=False)
    print('Overall Continous Stats for each cluster are saved to CSV files.')

    # Plotting continuous variables box plots
    for var, long_name, unit, direction, scale in zip(vars, vars_long_name, vars_units, vars_dir, vars_logscale):
        plot_single_vars(df_continuous, n_subsample, var, long_name, unit, direction, scale, output_path, run_name, sampling_type, stat, legend=var, hue=None, boxplot=False)

      