"""
Sample Cloud Properties Based on Cluster Proximity and Save Subsamples

This script processes cloud property crop images by sampling cluster-based subsamples
based on specified criteria. It supports filtering by daytime (06–16 UTC) and
IMERG-compatible timestamps (minutes == 00 or 30).
"""

import pandas as pd
import torch
from glob import glob
import numpy as np
import os
from datetime import datetime

# === CONFIGURATION ===
run_names = ['dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA']
sampling_type = 'all'  # Options: 'random', 'closest', 'farthest', 'all'
n_subsample = 67425  # Max samples per cluster
filter_daytime = False        # Enable daytime filter (06–16 UTC)
filter_imerg_minutes = True  # Only keep timestamps with minutes 00 or 30

# === HELPER FUNCTIONS ===
def parse_crop_datetime(filename: str) -> datetime:
    try:
        timestamp_str = os.path.basename(filename).split('_')[0]  # e.g., '20130401-00:00'
        return datetime.strptime(timestamp_str, "%Y%m%d-%H:%M")
    except Exception as e:
        print(f"Could not parse datetime from {filename}: {e}")
        return None

def is_daytime(filename: str) -> bool:
    dt = parse_crop_datetime(filename)
    return dt and (6 <= dt.hour <= 16)

def is_valid_imerg_minute(filename: str) -> bool:
    dt = parse_crop_datetime(filename)
    return dt and (dt.minute in [0, 30])

# === MAIN SCRIPT ===
for run_name in run_names:
    # Paths
    labels_path = f'/data1/runs/{run_name}/checkpoints/assignments_800ep.pt'
    distances_path = f'/data1/runs/{run_name}/checkpoints/distance_800ep.pt'
    image_crops_path = f'/data1/crops/{run_name}/1/'
    output_path = f'/data1/fig/{run_name}/{sampling_type}/'

    os.makedirs(output_path, exist_ok=True)

    # Load image paths
    list_image_crops = sorted(glob(image_crops_path + '*.tif'))
    n_samples = len(list_image_crops)
    print('Initial n samples:', n_samples)

    # Load cluster labels and distances
    assignments = torch.load(labels_path, map_location='cpu')[0].numpy()
    distances = torch.load(distances_path, map_location='cpu')[0].numpy()
    data_index = np.arange(n_samples)

    # Build DataFrame
    df_all = pd.DataFrame({
        'index': data_index,
        'path': list_image_crops,
        'assignment': assignments,
        'distance': distances
    })

    # Apply filters
    if filter_daytime:
        df_all = df_all[df_all['path'].apply(is_daytime)]
        print(f"After daytime filter: {len(df_all)} samples")

    if filter_imerg_minutes:
        df_all = df_all[df_all['path'].apply(is_valid_imerg_minute)]
        print(f"After IMERG minute filter: {len(df_all)} samples")

    df_all = df_all.reset_index(drop=True)

    # Adjust sample limit if sampling all
    if sampling_type == 'all':
        n_subsample = len(df_all)

    # Sample from clusters
    filtered_paths = df_all['path'].tolist()
    filtered_assignments = df_all['assignment'].to_numpy()
    filtered_distances = df_all['distance'].to_numpy()
    filtered_indices = df_all['index'].to_numpy()

    unique_clusters = np.unique(filtered_assignments)
    subsample_indices = []
    subsample_distances = []

    for cluster in unique_clusters:
        cluster_indices = np.where(filtered_assignments == cluster)[0]
        cluster_distances = filtered_distances[cluster_indices]

        if sampling_type == 'random':
            selected = np.random.choice(cluster_indices, min(n_subsample, len(cluster_indices)), replace=False)
        elif sampling_type == 'closest':
            sorted_idx = np.argsort(cluster_distances)
            selected = cluster_indices[sorted_idx[:n_subsample]]
        elif sampling_type == 'farthest':
            sorted_idx = np.argsort(cluster_distances)
            selected = cluster_indices[sorted_idx[-n_subsample:]]
        elif sampling_type == 'all':
            selected = cluster_indices
        else:
            raise ValueError(f"Invalid sampling type: {sampling_type}")

        subsample_indices.extend(selected)
        subsample_distances.extend(filtered_distances[selected])

    # Build result DataFrame
    df_labels = pd.DataFrame({
        'crop_index': [filtered_indices[i] for i in subsample_indices],
        'path': [filtered_paths[i] for i in subsample_indices],
        'label': [filtered_assignments[i] for i in subsample_indices],
        'distance': [subsample_distances[i] for i in range(len(subsample_indices))]
    })

    # Remove invalid labels
    df_labels = df_labels[df_labels['label'] != -100].reset_index(drop=True)
    print(f"Final number of samples: {len(df_labels)}")

    # === Construct filename with filters ===
    filter_tags = []
    if filter_daytime:
        filter_tags.append("daytime")
    if filter_imerg_minutes:
        filter_tags.append("imergmin")

    filter_suffix = "_" + "_".join(filter_tags) if filter_tags else ""

    csv_filename = f"crop_list_{run_name}_{n_subsample}_{sampling_type}{filter_suffix}.csv"
    output_file = os.path.join(output_path, csv_filename)

    # Save
    print(f"Saving {len(df_labels)} samples to {output_file} ...")
    df_labels.to_csv(output_file, index=False)
