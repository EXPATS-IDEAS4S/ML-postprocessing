import pandas as pd
import torch
from glob import glob
import numpy as np
import os
import shutil  # For copying files
from PIL import Image

run_names = ['10th-90th', '10th-90th_CMA']

# Define sampling type
sampling_type = 'closest'  # Options: 'random', 'closest', 'farthest', 'all'

n_subsample = 10  # Number of samples per cluster

for run_name in run_names:

    # Paths to CMSAF cloud properties crops
    crops_path = f'/data1/crops/ir108_2013-2014_GS_{run_name}/1/'
    crops_list = sorted(glob(crops_path + '*.tif'))

    # Read data
    n_samples = len(crops_list)

    # Path to cluster assignments of crops
    labels_path = f'/data1/runs/dcv2_ir108_128x128_k9_germany_30kcrops_grey_{run_name}/checkpoints/assignments_800ep.pt'

    # Path to cluster distances (from centroids)
    distances_path = f'/data1/runs/dcv2_ir108_128x128_k9_germany_30kcrops_grey_{run_name}/checkpoints/distance_800ep.pt'

    # Path to fig folder for outputs
    output_path = f'/home/Daniele/fig/cma_analysis/{run_name}/{sampling_type}/{n_subsample}/'

    # Create the directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    if sampling_type == 'all':
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
        
        # Add selected indices to the subsample list
        subsample_indices.extend(selected_indices)

    # Now, create the DataFrame with the selected subsamples
    df_labels = pd.DataFrame({
        'path': [crops_list[i] for i in subsample_indices],
        'label': [assignments[i] for i in subsample_indices]  # The labels of the subsamples
    })

    # Filter out invalid labels (-100)
    df_labels = df_labels[df_labels['label'] != -100]

    # Optionally print and save the dataframe for inspection
    print(df_labels)

    # Save the DataFrame to CSV for later inspection
    df_labels.to_csv(f'{output_path}crop_list_{run_name}_{n_subsample}_{sampling_type}.csv', index=False)

    # Loop through each unique cluster
    for cluster in unique_clusters:
        # Filter the DataFrame for the current cluster
        cluster_df = df_labels[df_labels['label'] == cluster]
        
        # Loop through each file path and assign a new name based on index
        for idx, file_path in enumerate(cluster_df['path']):
            # Extract the original filename without the extension
            old_filename = os.path.basename(file_path).split('.')[0]
            
            # Define the new filename in png format
            new_filename = f"class-{cluster}_crop-{idx + 1}_{sampling_type}_{old_filename}.png"
            
            # Define the destination path (with the new filename)
            dest_path = os.path.join(output_path, new_filename)
            
            # Open the .tif image file
            with Image.open(file_path) as img:
                # Convert and save it as a .png file
                img.save(dest_path, 'PNG')

    print(f'Copied and renamed {len(df_labels)} files to {output_path}')
    #exit()