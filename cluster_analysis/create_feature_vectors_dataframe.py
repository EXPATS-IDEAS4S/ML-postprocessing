import pandas as pd
import torch
from glob import glob
import numpy as np
import os

run_names = ['10th-90th', '10th-90th_CMA']

# Define sampling type
sampling_type = 'all'  # Options: 'random', 'closest', 'farthest', 'all'

n_subsample = 100  # Number of samples per cluster

# Paths to CMSAF cloud properties crops
cloud_properties_path = '/data1/crops/cmsaf_2013-2014_expats/nc_clouds/'
cloud_properties_crop_list = sorted(glob(cloud_properties_path + '*.nc'))

for run_name in run_names:

    # Path to cluster assignments of crops
    labels_path = f'/data1/runs/dcv2_ir108_128x128_k9_germany_30kcrops_grey_{run_name}/checkpoints/assignments_800ep.pt'

    # Path to cluster distances (from centroids)
    distances_path = f'/data1/runs/dcv2_ir108_128x128_k9_germany_30kcrops_grey_{run_name}/checkpoints/distance_800ep.pt'

    # Path to fig folder for outputs
    output_path = f'/home/Daniele/fig/cma_analysis/{run_name}/{sampling_type}/'

    # Create the directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Read data
    n_samples = len(cloud_properties_crop_list)

    if sampling_type=='all':
        n_subsample = n_samples
    else:
        n_subsample = min(n_subsample, n_samples)  # Ensure it doesn't exceed available samples

    assignments = torch.load(labels_path, map_location='cpu')  # Cluster labels for each sample
    distances = torch.load(distances_path, map_location='cpu')  # Distances to cluster centroids

    # Convert to numpy arrays for easier manipulation
    assignments = assignments[0].cpu().numpy()
    distances = distances[0].cpu().numpy()

    # Get unique cluster labels
    unique_clusters = np.unique(assignments)
    print(unique_clusters)

    # Prepare a list for subsample indices
    subsample_indices = []
    subsample_distances = []

    # Loop over each cluster and sample data
    for cluster in unique_clusters:
        # Get indices for all samples in this cluster
        cluster_indices = np.where(assignments == cluster)[0]
        
        # Get distances for the samples in this cluster
        cluster_distances = distances[cluster_indices]

        # Determine subsample for this cluster based on sampling_type
        if sampling_type == 'random':
            # Randomly select indices from the current cluster
            if len(cluster_indices) <= n_subsample:
                selected_indices = cluster_indices
            else:
                selected_indices = np.random.choice(cluster_indices, n_subsample, replace=False)
        
        elif sampling_type == 'closest':
            # Sort by distance (ascending) and select the closest ones
            sorted_idx = np.argsort(cluster_distances)
            selected_indices = cluster_indices[sorted_idx[:n_subsample]]
        
        elif sampling_type == 'farthest':
            # Sort by distance (descending) and select the farthest ones
            sorted_idx = np.argsort(cluster_distances)
            selected_indices = cluster_indices[sorted_idx[-n_subsample:]]
        
        elif sampling_type == 'all':
            # Use all the available data from this cluster (up to n_subsample if specified)
            selected_indices = cluster_indices
        
        else:
            raise ValueError("Invalid sampling type. Choose from 'random', 'closest', 'farthest', or 'all'.")
        
        # Collect distances for the selected samples
        selected_distances = distances[selected_indices]

        # Add selected indices and distances to the lists
        subsample_indices.extend(selected_indices)
        subsample_distances.extend(selected_distances)

    # Now, create the DataFrame with the selected subsamples
    df_labels = pd.DataFrame({
        'path': [cloud_properties_crop_list[i] for i in subsample_indices],
        'label': [assignments[i] for i in subsample_indices],  # The labels of the subsamples
        'distance': subsample_distances  # Corresponding distances
    })

    # Filter out invalid labels (-100)
    df_labels = df_labels[df_labels['label'] != -100]

    # Optionally print and save the dataframe for inspection
    print(df_labels)

    df_labels.to_csv(f'{output_path}crop_list_{run_name}_{n_subsample}_{sampling_type}.csv', index=False)


