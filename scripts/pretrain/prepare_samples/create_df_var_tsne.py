"""
Merge T-SNE Coordinates with Cluster Labels for Cloud Crop Visualization.

This script loads a precomputed T-SNE embedding (or another dimensionality reduction),
merges it with crop image labels and metadata, filters invalid labels, and assigns
colors for visualization. The result is saved as a merged CSV file for plotting.

Supports two modes:
    - FROM_CROP_STATS = True:  Uses crop statistics CSV for labels/paths
    - FROM_CROP_STATS = False: Uses crop list CSV for labels/paths

Author: Daniele
"""

import os
import sys
import numpy as np
import pandas as pd
from glob import glob
import sys


# === IMPORT HELPER FUNCTIONS ===
sys.path.append("/home/Daniele/codes/VISSL_postprocessing")
from utils.processing.features_utils import load_tsne_coordinates
from utils.plotting.class_colors import colors_per_class1_names
from utils.configs import load_config

# # === CONFIGURATION ===
# reduction_method = "tsne"     # Options: "tsne", "umap", etc.
# run_name = "dcv2_ir108-cm_100x100_8frames_k9_70k_nc_r2dplus1"
# crop_name = "clips_ir108_100x100_8frames_2013-2020"
# random_state = "3"            # Random seed used in T-SNE
# sampling_type = "all"         # Options: "random", "closest", "farthest", "all"
# file_extension = "nc"         # Crop image extension
# epoch = 800                   # Epoch of training
# FROM_CROP_STATS = False       # Use crop stats file or crop list file


# === HELPER FUNCTIONS ===
def load_labels(output_path: str, run_name: str, sampling_type: str, n_samples: int, from_crop_stats: bool) -> pd.DataFrame:
    """Load labels and crop paths from CSV file depending on the mode."""
    if from_crop_stats:
        fname = f"crops_stats_{run_name}_{sampling_type}_{n_samples}.csv"
    else:
        fname = f"crop_list_{run_name}_{sampling_type}_{n_samples}.csv"
    return pd.read_csv(os.path.join(output_path, fname))


def merge_and_filter(tsne_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """Merge T-SNE coordinates with labels, filter invalid entries, and assign colors."""
    # Align indices
    data_index = sorted(labels_df.crop_index.values)
    tsne_df = tsne_df[tsne_df.crop_index.isin(data_index)]
    labels_df = labels_df.set_index("crop_index").loc[data_index].reset_index()

    # Merge on crop_index
    merged = tsne_df.set_index("crop_index").join(labels_df.set_index("crop_index")).reset_index()

    # Remove invalid labels
    merged = merged[merged["label"] != -100]

    # Assign colors
    merged["color"] = merged["label"].map(lambda x: colors_per_class1_names[str(int(x))])
    return merged


def save_output(df: pd.DataFrame, output_path: str, run_name: str, sampling_type: str,
                random_state: str, epoch: int, from_crop_stats: bool) -> None:
    """Save merged dataframe to CSV file with appropriate filename."""
    if from_crop_stats:
        fname = f"merged_tsne_crop_stats_{run_name}_{sampling_type}_{random_state}_epoch_{epoch}.csv"
    else:
        fname = f"merged_tsne_crop_list_{run_name}_{sampling_type}_{random_state}_epoch_{epoch}.csv"
    df.to_csv(os.path.join(output_path, fname), index=False)
    print(f"Saved merged DataFrame to {fname}")


# === MAIN SCRIPT ===
def main(config_path: str):
    config = load_config(config_path)
    run_names = config["experiment"]["run_names"]
    base_path = config["experiment"]["base_path"]
    output_path = config["experiment"]["path_out"]
    crop_name = config["data"]["crops_name"]
    file_extension = config["data"]["file_extension"]
    sampling_type = config["sampling"]["type"]
    n_subsample = config["sampling"]["n_subsample"]
    epoch = config["experiment"]["epoch"]
    random_state = config["experiment"]["random_state"]
    from_crop_stats = config["experiment"]["from_crop_stats"]
    reduction_method = config["reduction"]["method"]
    perplexity = config["reduction"]["perplexity"]

    # Get number of samples
    if sampling_type == "all":
        image_path = f"{base_path}/crops/{crop_name}/{file_extension}/1/"
        crop_path_list = sorted(glob(image_path + "*." + file_extension))
        n_samples = len(crop_path_list)
    else:
        n_samples = n_subsample  # default per-cluster sample size

    
    for run_name in run_names:
        # Define paths
        output_path = f"{output_path}/{run_name}/epoch_{epoch}/{sampling_type}/"
        os.makedirs(output_path, exist_ok=True)
        filename = f"{reduction_method}_pca_cosine_perp-{perplexity}_{run_name}_{random_state}_epoch_{epoch}.npy"

        # Load data
        tsne_df = load_tsne_coordinates(output_path, filename)
        labels_df = load_labels(output_path, run_name, sampling_type, n_samples, from_crop_stats)

        # Merge + filter
        merged_df = merge_and_filter(tsne_df, labels_df)
        print(merged_df)

        # Save
        save_output(merged_df, output_path, run_name, sampling_type, random_state, epoch, from_crop_stats)


if __name__ == "__main__":
    config_path = "/home/Daniele/codes/VISSL_postprocessing/configs/process_run_config.yaml"
    main(config_path)
