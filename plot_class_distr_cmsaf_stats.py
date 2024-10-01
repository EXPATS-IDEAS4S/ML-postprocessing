import pandas as pd
import torch
from glob import glob
import xarray as xr
import random
import numpy as np
import os

run_name = '10th-90th'

# Paths to CMSAF cloud properties crops
cloud_properties_path = '/data1/crops/cmsaf_2013-2014_expats/nc_clouds/'
cloud_properties_crop_list = sorted(glob(cloud_properties_path + '*.nc'))

# Path to cluster assignments of crops
labels_path = f'/data1/runs/dcv2_ir108_128x128_k9_germany_30kcrops_grey_{run_name}/checkpoints/assignments_800ep.pt'

# Path to cluster distances (from centroids)
distances_path = f'/data1/runs/dcv2_ir108_128x128_k9_germany_30kcrops_grey_{run_name}/checkpoints/distance_800ep.pt'

# Path to fig folder for outputs
output_path = f'/home/Daniele/fig/dcv_ir108_128x128_k9_30k_grey_{run_name}/'
os.makedirs(output_path, exist_ok=True)

# Define sampling type
sampling_type = 'closest'  # Options: 'random', 'closest', 'farthest', 'all'

# Read data
n_samples = len(cloud_properties_crop_list)
n_subsample = 1000  # Number of samples per cluster
n_subsample = min(n_subsample, n_samples)  # Ensure it doesn't exceed available samples

assignments = torch.load(labels_path, map_location='cpu')  # Cluster labels for each sample
distances = torch.load(distances_path, map_location='cpu')  # Distances to cluster centroids

# Convert to numpy arrays for easier manipulation
assignments = assignments[0].cpu().numpy()
distances = distances.cpu().numpy()

# Get unique cluster labels
unique_clusters = np.unique(assignments)

# Prepare a list for subsample indices
subsample_indices = []

# Loop over each cluster and sample data
for cluster in unique_clusters:
    cluster_indices = np.where(assignments == cluster)[0]  # Indices for all samples in this cluster
    cluster_distances = distances[cluster_indices]  # Distances for samples in this cluster

    # Determine subsample based on sampling_type
    if sampling_type == 'random':
        selected_indices = np.random.choice(cluster_indices, n_subsample, replace=False)
    elif sampling_type == 'closest':
        sorted_idx = np.argsort(cluster_distances)
        selected_indices = cluster_indices[sorted_idx[:n_subsample]]
    elif sampling_type == 'farthest':
        sorted_idx = np.argsort(cluster_distances)
        selected_indices = cluster_indices[sorted_idx[-n_subsample:]]
    elif sampling_type == 'all':
        selected_indices = cluster_indices[:n_subsample]
    else:
        raise ValueError("Invalid sampling type. Choose from 'random', 'closest', 'farthest', or 'all'.")
    
    subsample_indices.extend(selected_indices)

# Create DataFrame with selected subsamples
df_labels = pd.DataFrame({
    'path': [cloud_properties_crop_list[i] for i in subsample_indices],
    'label': [assignments[i] for i in subsample_indices]
})

# Define continuous variables for which we want to compute statistics
continuous_vars = ['cwp', 'cot', 'ctt', 'ctp', 'cth', 'cre']
stats_to_compute = ['mean', 'std', 'min', 'max', '25%', '50%', '75%']  # Percentiles can be adjusted as needed

# Initialize a dictionary to store statistics
stats_data = {var: [] for var in continuous_vars}
labels = []

# Read the .nc files and compute statistics
for i, row in df_labels.iterrows():
    ds = xr.open_dataset(row['path'])
    
    # Compute statistics for each continuous variable
    for var in continuous_vars:
        values = ds[var].values.flatten()
        stats = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            '25%': np.percentile(values, 25),
            '50%': np.median(values),
            '75%': np.percentile(values, 75)
        }
        stats_data[var].append([stats[stat] for stat in stats_to_compute])
    
    labels.append(row['label'])

# Convert the stats data to a DataFrame
stats_df = pd.DataFrame({
    var: [stats_data[var][i] for i in range(len(labels))]
    for var in continuous_vars
}, index=labels)

# Add multi-level column names for better representation
stats_df.columns = pd.MultiIndex.from_product([continuous_vars, stats_to_compute])

# Reset index to add labels as a column
stats_df.reset_index(inplace=True)
stats_df.rename(columns={'index': 'label'}, inplace=True)

# Save the computed statistics to a CSV file
stats_df.to_csv(f'{output_path}continuous_statistics_{n_subsample}.csv', index=False)
print(f'Continuous statistics saved to {output_path}continuous_statistics_{n_subsample}.csv')

