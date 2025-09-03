import pandas as pd
import xarray as xr
import numpy as np

from utils.processing.aux_functions import compute_percentile, concatenate_values, extend_labels, plot_single_vars, find_latlon_boundaries_from_ds

run_name = '10th-90th_CMA'

# Define sampling type
sampling_type = 'closest'  # Options: 'random', 'closest', 'farthest', 'all'

# Pick the statistics to compute for each crop, the percentile values
stats = [1,99,50,'25-75']  #'1%', '5%', '25%', '50%', '75%', '95%', '99%' ,None if all points are needed

# Path to topography data (DEM and land-sea mask)
DEM_path = f'/home/daniele/Documenti/Data/topography/DEM_EXPATS_0.01x0.01.nc'
landseamask_path = f'/home/daniele/Documenti/Data/topography/IMERG_landseamask_EXPATS_0.1x0.1.nc'
#snowcover_path = f'/home/daniele/Documenti/Data/ERA5-Land_t2m_snowc_u10_v10_sp_tp/'

#open files
ds_dem = xr.open_dataset(DEM_path)
ds_lsm = xr.open_dataset(landseamask_path)
#print(ds_dem.DEM.values)
#print(ds_lsm.landseamask.values) #100% all water, 0% all land

# Path to fig folder for outputs
output_path = f'/home/daniele/Documenti/Data/Fig/{run_name}/{sampling_type}/'

# Read data
n_subsample = 1000  # Number of samples per cluster

# Load CSV file with the crops path and labels into a pandas DataFrame
df_labels = pd.read_csv(f'{output_path}crop_list_{run_name}_{n_subsample}_{sampling_type}.csv')


##########################################################
## Compute stats and plot distr for Continous variables ##
##########################################################

continuous_vars = ['DEM', 'landseamask']
cont_vars_long_name = ['digital elevation model', 'land-sea mask']
cont_vars_units = ['m', None ]
cont_vars_logscale = [False, False]
cont_vars_dir = ['incr','incr']

for stat in stats:

    # Initialize lists to hold data for continuous and categorical variables
    continuous_data = {var: [] for var in continuous_vars}
    labels = []

    # Read the .nc files and extract data
    for i, row in df_labels.iterrows():
        ds_crops = xr.open_dataset(row['path'])

        lat_min, lat_max, lon_min, lon_max = find_latlon_boundaries_from_ds(ds_crops)
        
        # Loop over continuous variables
        for var in continuous_vars:
            
            # For DEM, use the ds_dem dataset and subset it based on the latitude and longitude range
            if var == 'DEM':
                # Select DEM values within the lat/lon range of ds_crops
                values = ds_dem[var].sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)).values.flatten()
                # Now `dem_values` contains DEM data corresponding to the lat/lon of ds_crops
                
            # For land-sea mask, use the ds_lsm dataset and subset it similarly
            elif var == 'landseamask':
                # Select land-sea mask values within the lat/lon range of ds_crops
                values = ds_lsm[var].sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)).values.flatten()
                # Now `lsm_values` contains land-sea mask data corresponding to the lat/lon of ds_crops
                
            # For other variables, use ds_crops directly
            else:
                # Flatten the array and remove NaN values (optional)
                print('wrong variable names!')
                exit()
            
            # Apply percentile if 'stat' is provided (stat can be a percentile or another function)
            if stat:
                try:
                    values = compute_percentile(values, stat)
                except IndexError:
                    print(f"Not enough values to calculate {stat} for {var} in {row['path']}")
                    continue  # Skip if there are not enough values for the given percentile
            else:
                values = np.random.choice(values, 500, replace=False)
            
            # Ensure values is a list or array before extending
            continuous_data =  concatenate_values(values, var, continuous_data)

        # Extend labels based on the number of valid entries for this dataset
        labels = extend_labels(values, labels, row)
    
    # Create DataFrames for continuous and categorical variables
    df_continuous = pd.DataFrame(continuous_data)
    df_continuous['label'] = labels

    print(df_continuous)

    # Save in case of crop stats are calculated
    if stat: 
        df_continuous.to_csv(f'{output_path}topography_crops_stats_{run_name}_{sampling_type}_{n_subsample}_{stat}.csv', index=False)
        print('Continous Stats for each crop are saved to CSV files.')

    # Compute stats for continuous variables
    continuous_stats = df_continuous.groupby('label').agg(['mean', 'std'])
    continuous_stats.columns = ['_'.join(col).strip() for col in continuous_stats.columns.values]
    continuous_stats.reset_index(inplace=True)

    # Save continuous stats to a CSV file
    continuous_stats.to_csv(f'{output_path}topography_clusters_stats_{run_name}_{sampling_type}_{n_subsample}_{stat}.csv', index=False)
    print('Overall Continous Stats for each cluster are saved to CSV files.')

    # Plotting continuous variables box plots
    for var, long_name, unit, direction, scale in zip(continuous_vars, cont_vars_long_name, cont_vars_units, cont_vars_dir, cont_vars_logscale):
        plot_single_vars(df_continuous, n_subsample, var, long_name, unit, direction, scale, output_path, run_name, sampling_type, stat)
        
