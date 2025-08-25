"""
Cloud Crop Sampling by Cluster Proximity

This script processes cloud property crop images by sampling cluster-based
subsamples according to user-defined criteria. Supports filtering by:
  - Daytime hours (06–16 UTC)
  - IMERG-compatible timestamps (minutes == 00 or 30)

Sampling strategies:
  - 'random'   → random selection within each cluster
  - 'closest'  → nearest samples to cluster centroid
  - 'farthest' → farthest samples from cluster centroid
  - 'all'      → all available samples
"""

import os
import numpy as np
import pandas as pd
import torch
from glob import glob
from datetime import datetime
from typing import List

# === CONFIGURATION ===
RUN_NAMES = ['dcv2_ir108-cm_100x100_8frames_k9_70k_nc_r2dplus1']
CROPS_NAME = 'clips_ir108_100x100_8frames_2013-2020'  # Name of the crops
FILE_EXTENSION = 'nc'       # Image file extension
SAMPLING_TYPE = 'all'       # Options: 'random', 'closest', 'farthest', 'all'
N_SUBSAMPLE = 70072         # Max samples per cluster
EPOCH = 800                 # Epoch number for the run
FILTER_DAYTIME = False      # Enable daytime filter (06–16 UTC)
FILTER_IMERG_MINUTES = False  # Only keep timestamps with minutes 00 or 30


# === HELPER FUNCTIONS ===
def parse_crop_datetime(filename: str) -> datetime:
    """Extract datetime object from crop filename (format: YYYYMMDD-HH:MM_...)."""
    try:
        timestamp_str = os.path.basename(filename).split('_')[0]  # e.g., '20130401-00:00'
        return datetime.strptime(timestamp_str, "%Y%m%d-%H:%M")
    except Exception as e:
        print(f"Could not parse datetime from {filename}: {e}")
        return None


def is_daytime(filename: str) -> bool:
    """Return True if crop timestamp is between 06–16 UTC."""
    dt = parse_crop_datetime(filename)
    return dt and (6 <= dt.hour <= 16)


def is_valid_imerg_minute(filename: str) -> bool:
    """Return True if crop timestamp minute is 00 or 30 (IMERG-compatible)."""
    dt = parse_crop_datetime(filename)
    return dt and (dt.minute in [0, 30])


def load_dataframes(run_name: str, crops_name: str, file_extension: str, epoch: int) -> pd.DataFrame:
    """Load crop paths, cluster assignments, and distances into a DataFrame."""
    labels_path = f'/data1/runs/{run_name}/checkpoints/assignments.pt'
    distances_path = f'/data1/runs/{run_name}/checkpoints/distances.pt'
    image_crops_path = f'/data1/crops/{crops_name}/{file_extension}/1/'

    print(f"Loading image crops from {image_crops_path} ...")
    list_image_crops = sorted(glob(os.path.join(image_crops_path, f'*.{file_extension}')))
    n_samples = len(list_image_crops)
    print('Initial n samples:', n_samples)

    # Load cluster data
    assignments = torch.load(labels_path, map_location='cpu')[0].numpy()
    distances = torch.load(distances_path, map_location='cpu')[0].numpy()
    data_index = np.arange(n_samples)

    return pd.DataFrame({
        'index': data_index,
        'path': list_image_crops,
        'assignment': assignments,
        'distance': distances
    })


def apply_filters(df: pd.DataFrame, filter_daytime: bool, filter_imerg_minutes: bool) -> pd.DataFrame:
    """Apply daytime and IMERG filters to the dataset."""
    if filter_daytime:
        df = df[df['path'].apply(is_daytime)]
        print(f"After daytime filter: {len(df)} samples")

    if filter_imerg_minutes:
        df = df[df['path'].apply(is_valid_imerg_minute)]
        print(f"After IMERG minute filter: {len(df)} samples")

    return df.reset_index(drop=True)


def sample_clusters(df: pd.DataFrame, sampling_type: str, n_subsample: int) -> pd.DataFrame:
    """Perform cluster-based sampling according to the given strategy."""
    if sampling_type == 'all':
        n_subsample = len(df)  # override

    paths, assignments, distances, indices = (
        df['path'].to_numpy(),
        df['assignment'].to_numpy(),
        df['distance'].to_numpy(),
        df['index'].to_numpy()
    )

    unique_clusters = np.unique(assignments)
    subsample_indices, subsample_distances = [], []

    for cluster in unique_clusters:
        cluster_indices = np.where(assignments == cluster)[0]
        cluster_distances = distances[cluster_indices]

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
        subsample_distances.extend(distances[selected])

    df_labels = pd.DataFrame({
        'crop_index': indices[subsample_indices],
        'path': paths[subsample_indices],
        'label': assignments[subsample_indices],
        'distance': subsample_distances
    })

    return df_labels[df_labels['label'] != -100].reset_index(drop=True)


def save_results(df: pd.DataFrame, run_name: str, epoch: int, sampling_type: str,
                 n_subsample: int, filter_daytime: bool, filter_imerg_minutes: bool):
    """Save the sampled results to a CSV file."""
    output_path = f'/data1/fig/{run_name}/epoch_{epoch}/{sampling_type}/'
    os.makedirs(output_path, exist_ok=True)

    # Construct filter tags for filename
    filter_tags = []
    if filter_daytime:
        filter_tags.append("daytime")
    if filter_imerg_minutes:
        filter_tags.append("imergmin")
    filter_suffix = "_" + "_".join(filter_tags) if filter_tags else ""

    csv_filename = f"crop_list_{run_name}_{sampling_type}_{n_subsample}{filter_suffix}.csv"
    output_file = os.path.join(output_path, csv_filename)

    print(f"Saving {len(df)} samples to {output_file} ...")
    df.to_csv(output_file, index=False)


# === MAIN ===
def main():
    for run_name in RUN_NAMES:
        df_all = load_dataframes(run_name, CROPS_NAME, FILE_EXTENSION, EPOCH)
        df_all = apply_filters(df_all, FILTER_DAYTIME, FILTER_IMERG_MINUTES)
        df_labels = sample_clusters(df_all, SAMPLING_TYPE, N_SUBSAMPLE)
        print(f"Final number of samples: {len(df_labels)}")
        save_results(df_labels, run_name, EPOCH, SAMPLING_TYPE, N_SUBSAMPLE,
                     FILTER_DAYTIME, FILTER_IMERG_MINUTES)


if __name__ == "__main__":
    main()
