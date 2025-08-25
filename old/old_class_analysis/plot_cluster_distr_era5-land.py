import pandas as pd
import xarray as xr
import numpy as np

from aux_functions import compute_percentile, concatenate_values, extend_labels, plot_single_vars, find_latlon_boundaries_from_ds

run_names = ['10th-90th', '10th-90th_CMA']

# Define sampling type
sampling_type = 'closest'  # Options: 'random', 'closest', 'farthest', 'all'

# Read data
n_subsample = 100  # Number of samples per cluster

# Pick the statistics to compute for each crop, the percentile values
stats = [None] #[1,99,50,'25-75']  #'1%', '5%', '25%', '50%', '75%', '95%', '99%' ,None if all points are needed

for run_name in run_names:

    # Path to topography data (DEM and land-sea mask)
    era5_path = f'/home/daniele/Documenti/Data/ERA5-Land_t2m_snowc_u10_v10_sp_tp/'

    # Path to fig folder for outputs
    output_path = f'/home/daniele/Documenti/Data/Fig/{run_name}/{sampling_type}/'


    # Load CSV file with the crops path and labels into a pandas DataFrame
    df_labels = pd.read_csv(f'{output_path}crop_list_{run_name}_{n_subsample}_{sampling_type}.csv')


    ##########################################################
    ## Compute stats and plot distr for Continous variables ##
    ##########################################################

    vars = ['t2m', 'snowc','u10','v10','sp','tp']
    vars_long_name = ['2-m temperature', 'snow cover','10-m u wind speed', '10-m v wind speed', 'surface pressure','total precipitation']
    vars_units = ['K', '%','m/s','m/s','Pa','m' ]
    vars_logscale = [False, False,False,False,False,False]
    vars_dir = ['incr','incr','incr','incr','incr','incr']

    for stat in stats:

        # Initialize lists to hold data for continuous and categorical variables
        continuous_data = {var: [] for var in vars}
        labels = []

        # Read the .nc files and extract data
        for i, row in df_labels.iterrows():
            ds_crops = xr.open_dataset(row['path'])

            # Get the latitude and longitude values from the crop dataset
            lat_min, lat_max, lon_min, lon_max = find_latlon_boundaries_from_ds(ds_crops)

            #get time of the crop
            time_crop = ds_crops.time.values
            print(time_crop)

            # Extract year, month, and hour from the crop time
            time_crop_dt = pd.to_datetime(time_crop)
            year = time_crop_dt.year
            month = time_crop_dt.month
            day = time_crop_dt.day
            hour = time_crop_dt.hour

            # Create a datetime64 object with minute and second set to 0
            datetime_obj = np.datetime64(f'{year:04d}-{month:02d}-{day:02d}T{hour:02d}:00:00')
            #print(datetime_obj)

            # Construct the file path for the corresponding ERA5 data
            era5_file = f'{era5_path}{year}/{year}-{month:02d}_expats.nc'
            
            # Open the ERA5 dataset for that year and month
            ds_era5 = xr.open_dataset(era5_file)
            #print(ds_era5)

            # Select the ERA5 data for the exact hour
            ds_era5_time = ds_era5.sel(valid_time=datetime_obj)
            #print(ds_era5_time)
            
            # Loop over continuous variables
            for var in vars:
                # Select ERA5 data within the lat/lon range of the crops dataset
                values = ds_era5_time[var].sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max)).values.flatten()

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
            labels = extend_labels(values, labels, row)

        # Create DataFrames for continuous and categorical variables
        df_continuous = pd.DataFrame(continuous_data)
        df_continuous['label'] = labels

        print(df_continuous)

        # Save in case of crop stats are calculated
        if stat: 
            df_continuous.to_csv(f'{output_path}era5-land_crops_stats_{run_name}_{sampling_type}_{n_subsample}_{stat}.csv', index=False)
            print('Continous Stats for each crop are saved to CSV files.')

        # Compute stats for continuous variables
        continuous_stats = df_continuous.groupby('label').agg(['mean', 'std'])
        continuous_stats.columns = ['_'.join(col).strip() for col in continuous_stats.columns.values]
        continuous_stats.reset_index(inplace=True)

        # Save continuous stats to a CSV file
        continuous_stats.to_csv(f'{output_path}era5-land_clusters_stats_{run_name}_{sampling_type}_{n_subsample}_{stat}.csv', index=False)
        print('Overall Continous Stats for each cluster are saved to CSV files.')

        # Plotting continuous variables box plots
        for var, long_name, unit, direction, scale in zip(vars, vars_long_name, vars_units, vars_dir, vars_logscale):
            plot_single_vars(df_continuous, n_subsample, var, long_name, unit, direction, scale, output_path, run_name, sampling_type, stat)
            
