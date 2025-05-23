import pandas as pd
import xarray as xr
import numpy as np
import io
import os
import sys
import boto3
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

from aux_functions_from_buckets import extract_coordinates, extract_datetime, compute_categorical_values
from get_data_from_buckets import read_file, Initialize_s3_client
from credentials_buckets import S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY, S3_ENDPOINT_URL
sys.path.append(os.path.abspath("/home/Daniele/codes/visualization/cluster_analysis"))  

# Initialize S3 client
s3 = Initialize_s3_client(S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY)

# Constants
BUCKET_CMSAF_NAME = 'expats-cmsaf-cloud'
BUCKET_IMERG_NAME = 'expats-imerg-prec'
BUCKET_MSG_NAME = 'expats-msg-training'

run_name = 'dcv2_ir108_200x200_k9_expats_70k_200-300K_closed-CMA'
sampling_type = 'closest'
vars = ['IR_108', 'WV_062','cot', 'cth', 'precipitation']  # Variables to extract
#categ_vars = ['cma', 'cph']  # Categorical variables
units = ['K', 'K', None, 'm', 'mm/h']  # Units for each variable
logs = [False, False, True, False, True]  # Logarithmic scale for each variable
xlims = [(200, 320), (205, 250), (0, 150), (0, 17500), (0, 80)]  # X-axis limits for each variable
ylims = [(0, 0.50), (0, 0.25), (0, 0.70), (0, 0.15), (0, 0.70)]  # Y-axis limits for each variable
alpha = 0.01
# x_pixel_size = 128
# y_pixel_size = 128
# n_pixels = x_pixel_size * y_pixel_size

# Path to figures folder
output_path = f'/data1/fig/{run_name}/{sampling_type}/'
os.makedirs(output_path, exist_ok=True)

# Read data
# List of the image crops
image_crops_path = f'/data1/crops/{run_name}/1/'
list_image_crops = sorted(glob(image_crops_path+'*.tif'))
n_samples = len( list_image_crops)
print('n samples: ', n_samples)

if sampling_type == 'all':
    n_subsample = n_samples  # Number of samples per cluster
else:
    n_subsample = 1000

# Load crop labels
df_labels = pd.read_csv(f'{output_path}crop_list_{run_name}_{n_subsample}_{sampling_type}.csv')
print(df_labels)

#Reomve row with label -100
df_labels = df_labels[df_labels['label'] != -100]

#Randomly sample 1000 rows
#df_labels = df_labels.sample(n=100)

# Function to extract data from S3
def extract_variable_values(row, var):
    crop_filename = row['path'].split('/')[-1]
    coords = extract_coordinates(crop_filename)
    lat_min, lat_max, lon_min, lon_max = coords['lat_min'], coords['lat_max'], coords['lon_min'], coords['lon_max']
    
    datetime_info = extract_datetime(crop_filename)
    year, month, day, hour, minute = datetime_info['year'], datetime_info['month'], datetime_info['day'], datetime_info['hour'], datetime_info['minute']
    datetime_obj = np.datetime64(f'{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:00')
    print('processing timestamp:', datetime_obj)

    values = []  # Store label for grouping
        
    if var == 'precipitation' and (minute == 15 or minute == 45):
        return values

    if var in ['IR_108', 'WV_062']:
        bucket_name = BUCKET_MSG_NAME
        bucket_filename = f'/data/sat/msg/ml_train_crops/IR_108-WV_062-CMA_FULL_EXPATS_DOMAIN/{year:04d}/{month:02d}/merged_MSG_CMSAF_{year:04d}-{month:02d}-{day:02d}.nc'	
    elif var == 'precipitation':
        bucket_name = BUCKET_IMERG_NAME
        bucket_filename = f'IMERG_daily_{year:04d}-{month:02d}-{day:02d}.nc'
    else:
        bucket_name = BUCKET_CMSAF_NAME 
        bucket_filename = f'MCP_{year:04d}-{month:02d}-{day:02d}_regrid.nc'
    try:
        my_obj = read_file(s3, bucket_filename, bucket_name)
        ds_day = xr.open_dataset(io.BytesIO(my_obj))[var]

        if isinstance(ds_day.indexes["time"], xr.CFTimeIndex):
            ds_day["time"] = ds_day["time"].astype("datetime64[ns]")

        ds_day = ds_day.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
        ds_day = ds_day.sel(time=datetime_obj)

        values = ds_day.values.flatten()
  
    except Exception as e:
        print(f"Error processing {var} for {row['path']}: {e}")
        values = []

    return values


labels = df_labels['label'].unique()
#labels = [0] 
for label in labels:
    # Select the rows with the current label
    df_labels_selection = df_labels[df_labels['label'] == label]
    for entry, unit, log, xlim, ylim in zip(vars, units, logs, xlims, ylims):
        if entry == 'IR_108':
            print(f"Processing label {label} for variable {entry}")

            # Histogram bin setup
            num_bins = 50
            bin_edges = np.linspace(xlim[0], xlim[1], num_bins + 1)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            sum_in_bins = np.zeros(num_bins)
            count_in_bins = np.zeros(num_bins)
            #values_in_bins = [[] for _ in range(num_bins)]  # Optional: for percentiles

            # Create figure
            fig, ax = plt.subplots(figsize=(8, 6))
            for _, row in df_labels_selection.iterrows():
                values = extract_variable_values(row, entry) 
                if len(values) == 0:
                    continue

                #total_values = len(values)  # Get total count of values
                sns.histplot(
                    values,
                    bins=bin_edges,
                    kde=False,
                    alpha=alpha,
                    element="bars",
                    fill=True,
                    edgecolor=None,
                    color='blue',
                    stat="probability"  # Normalize histogram by total count
                )

                # Bin-wise aggregation for mean and percentiles
                hist, _ = np.histogram(values, bins=bin_edges)
                bin_value_sums, _ = np.histogram(values, bins=bin_edges, weights=values)

                sum_in_bins += np.nan_to_num(bin_value_sums)
                count_in_bins += hist
        
            # Final stats
            #mean_per_bin = sum_in_bins / np.maximum(count_in_bins, 1)
            probability_per_bin = count_in_bins / count_in_bins.sum()
            
            # Plot the summary curves
            ax.plot(bin_centers, probability_per_bin, color='red', lw=2, label='Mean')

            if unit:
                plt.xlabel(f'{entry} [{unit}]')
            else:
                plt.xlabel(f'{entry}')
            plt.ylabel('Frequency')
            if log:
                plt.yscale('log')
            #plt.xlim(xlim[0], xlim[1])
            plt.ylim(ylim[0], ylim[1])
            plt.title(f'Distribution of {entry} for Class {label}')
            plt.grid(True, linestyle='--', alpha=0.5)
            output_dir = f'{output_path}overlaid_histograms/{label}/'
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(f'{output_dir}histogram_{entry}_label_{label}.png', dpi=300, bbox_inches="tight")
            plt.close()

            print("Histograms saved successfully!")
        
#nohup 1816481