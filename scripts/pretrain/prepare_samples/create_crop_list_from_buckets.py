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
import sys

sys.path.append("/home/Daniele/codes/VISSL_postprocessing")
from utils.processing.features_utils import load_dataframes, is_daytime, is_valid_imerg_minute
from utils.configs import load_config


# === HELPER FUNCTIONS ===
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


def save_results(df: pd.DataFrame, path_out: str, run_name: str, epoch: int, sampling_type: str,
                 n_subsample: int, filter_daytime: bool, filter_imerg_minutes: bool):
    """Save the sampled results to a CSV file."""
    output_path = f'{path_out}/{run_name}/epoch_{epoch}/{sampling_type}/'
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
def main(config_path: str = "config.yaml"):
    config = load_config(config_path)
    run_names = config["experiment"]["run_names"]
    print(run_names)
    exit()
    epoch = config["experiment"]["epoch"]
    crops_name = config["data"]["crops_name"]
    file_extension = config["data"]["file_extension"]
    sampling_type = config["sampling"]["type"]
    n_subsample = config["sampling"]["n_subsample"]
    filter_daytime = config["filters"]["daytime"]
    filter_imerg_minutes = config["filters"]["imerg_minutes"]
    output_root = config["experiment"]["path_out"]
    base_path = config["experiment"]["base_path"]

    for run_name in run_names:
        df_all = load_dataframes(base_path, run_name, crops_name, file_extension, epoch)
        df_all = apply_filters(df_all, filter_daytime, filter_imerg_minutes)
        df_labels = sample_clusters(df_all, sampling_type, n_subsample)
        print(f"Final number of samples: {len(df_labels)}")
        save_results(df_labels, output_root, run_name, epoch, sampling_type, n_subsample,
                     filter_daytime, filter_imerg_minutes)


if __name__ == "__main__":
    config_path = "/home/Daniele/codes/VISSL_postprocessing/configs/process_run_config.yaml"
    main(config_path)
