##########################################################
## Compute stats and plot distr for Continous variables ##
##########################################################

import pandas as pd
import xarray as xr
import numpy as np
from glob import glob

from aux_functions import compute_percentile, concatenate_values, extend_labels, plot_single_vars, pick_variable, find_latlon_boundaries_from_ds, get_time_from_ds, select_ds, plot_joyplot

run_names = ['10th-90th_CMA', '10th-90th']

# Define sampling type
sampling_type = 'closest'  # Options: 'random', 'closest', 'farthest', 'all'

# Pick the statistics to compute for each crop, the percentile values
stats = [None] #[1,50,99,'25-75']   #'1%', '5%', '25%', '50%', '75%', '95%', '99%' ,None if all points are needed
#use e.g '25-75' if interquartile range want to be calculated

# Paths to CMSAF cloud properties crops
cloud_properties_path = '/data1/crops/cmsaf_2013-2014_expats/nc_clouds/'
cloud_properties_crop_list = sorted(glob(cloud_properties_path + '*.nc'))
n_samples = len(cloud_properties_crop_list)

# Read data
if sampling_type == 'all':
    n_subsample = n_samples  # Number of samples per cluster
else:
    n_subsample = 100


# Path to topography data (DEM and land-sea mask)
era5_path = f'/data1/other_data/ERA5-Land_t2m_snowc_u10_v10_sp_tp/'

data_types = ['continuous'] #['era5-land','continuous','topography'] # 'era5-land'

#if data_type == 'topography':
# Path to topography data (DEM and land-sea mask)
DEM_path = f'/data1/other_data/DEM_EXPATS_0.01x0.01.nc'
landseamask_path = f'/data1/other_data/IMERG_landseamask_EXPATS_0.1x0.1.nc'
#snowcover_path = f'/home/daniele/Documenti/Data/ERA5-Land_t2m_snowc_u10_v10_sp_tp/'
#open files
ds_dem = xr.open_dataset(DEM_path)
ds_lsm = xr.open_dataset(landseamask_path)
#print(ds_dem.DEM.values)
#print(ds_lsm.landseamask.values) #100% all water, 0% all land


for data_type in data_types:

    # Select the correct varable information based on the data type  
    vars, vars_long_name, vars_units, vars_logscale, vars_dir = pick_variable(data_type)

    for run_name in run_names:

        # Path to fig folder for outputs
        output_path = f'/home/Daniele/fig/cma_analysis/{run_name}/{sampling_type}/'

        # Load CSV file with the crops path and labels into a pandas DataFrame
        df_labels = pd.read_csv(f'{output_path}crop_list_{run_name}_{n_subsample}_{sampling_type}.csv')
        print(df_labels)

        for stat in stats:

            # Initialize lists to hold data for continuous and categorical variables
            continuous_data = {var: [] for var in vars}
            labels = []
            crop_names = []
            crop_distances = []

            # Read the .nc files and extract data
            for i, row in df_labels.iterrows():
                ds_crops = xr.open_dataset(row['path'])

                # get lat lon and time info from Dataset
                lat_min, lat_max, lon_min, lon_max = find_latlon_boundaries_from_ds(ds_crops)
                #print(lat_min, lat_max, lon_min, lon_max)
                year, month, day, hour = get_time_from_ds(ds_crops)
                
                #if data_type == 'era5-land':
                # Create a datetime64 object with minute and second set to 0
                datetime_obj = np.datetime64(f'{year:04d}-{month:02d}-{day:02d}T{hour:02d}:00:00')
                # Construct the file path for the corresponding ERA5 data
                era5_file = f'{era5_path}{year}/{year}-{month:02d}_expats.nc'
                
                # Open the ERA5 dataset for that year and month
                ds_era5 = xr.open_dataset(era5_file)
                #print(ds_era5)

                # Select the ERA5 data for the exact hour
                ds_era5_time = ds_era5.sel(valid_time=datetime_obj)
                #print(ds_era5_time)

                for var in vars:
                    #print(var)

                    # give the variable, select the correct dataset
                    ds = select_ds(var, [ds_crops, ds_era5_time, ds_dem, ds_lsm])
                    #print(ds)

                    # Select values within the lat/lon range of ds
                    if data_type=='era5-land':
                        # Invert the latitude dimension to match crop coordinate
                        ds = ds.isel(latitude=slice(None, None, -1))
                        values = ds[var].sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max)).values.flatten()
                    else:
                        values = ds[var].sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)).values.flatten()
                  
                    # Apply percentile if 'stat' is provided (stat can be a percentile or another function)
                    if stat:
                        try:
                            values = compute_percentile(values, stat)
                        except IndexError:
                            print(f"Not enough values to calculate {stat} for {var} in {row['path']}")
                            continue  # Skip if there are not enough values for the given percentile
                    else:
                        if data_type=='topography':
                            #apply random samples whent the values contained in the variables are not the same size (like in topography)
                            random_samples = min(500, len(values))
                            values = np.random.choice(values, random_samples, replace=False)

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
            #if stat: 
            df_continuous.to_csv(f'{output_path}{data_type}_crops_stats_{run_name}_{sampling_type}_{n_subsample}_{stat}.csv', index=False)
            print('Continous Stats for each crop are saved to CSV files.')

            # # Compute stats for continuous variables
            # continuous_stats = df_continuous.groupby('label').agg(['mean', 'std'])
            # continuous_stats.columns = ['_'.join(col).strip() for col in continuous_stats.columns.values]
            # continuous_stats.reset_index(inplace=True)

            # Save continuous stats to a CSV file
            #continuous_stats.to_csv(f'{output_path}{data_type}_clusters_stats_{run_name}_{sampling_type}_{n_subsample}_{stat}.csv', index=False)
            #print('Overall Continous Stats for each cluster are saved to CSV files.')

            # #extract class label
            # label_names = np.unique(df_continuous['label'].values)
            # label_names = label_names[label_names != -100]
            
            # # Plotting continuous variables box plots
            # for var, long_name, unit, direction, scale in zip(vars, vars_long_name, vars_units, vars_dir, vars_logscale):
            #     #plot_single_vars(df_continuous, n_subsample, var, long_name, unit, direction, scale, output_path, run_name, sampling_type, stat)
            #     for class_label in label_names:
            #         plot_joyplot(df_continuous, class_label, var, long_name, unit, n_subsample, output_path, run_name, sampling_type)

                

