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

# === CONFIGURATION ===
colors_per_class1_names = {
    '0': 'darkgray',
    '1': 'darkslategrey',
    '2': 'peru',
    '3': 'orangered',
    '4': 'lightcoral',
    '5': 'deepskyblue',
    '6': 'purple',
    '7': 'lightblue',
    '8': 'green',
    '9': 'goldenrod',
    '10': 'magenta',
    '11': 'dodgerblue',
    '12': 'darkorange',
    '13': 'olive',
    '14': 'crimson'
}

reduction_method = "tsne"     # Options: "tsne", "umap", etc.
run_name = "dcv2_ir108-cm_100x100_8frames_k9_70k_nc_r2dplus1"
crop_name = "clips_ir108_100x100_8frames_2013-2020"
random_state = "3"            # Random seed used in T-SNE
sampling_type = "all"         # Options: "random", "closest", "farthest", "all"
file_extension = "nc"         # Crop image extension
epoch = 800                   # Epoch of training
FROM_CROP_STATS = False       # Use crop stats file or crop list file

# === IMPORT HELPER FUNCTIONS ===
sys.path.append(os.path.abspath("/home/Daniele/codes/visualization/cluster_analysis"))
from feature_space_plot_functions import scale_to_01_range


# === HELPER FUNCTIONS ===
def load_tsne_coordinates(output_path: str, filename: str) -> pd.DataFrame:
    """Load T-SNE coordinates from .npy file and return as DataFrame."""
    tsne = np.load(os.path.join(output_path, filename))
    tx = scale_to_01_range(tsne[:, 0])
    ty = scale_to_01_range(tsne[:, 1])
    return pd.DataFrame({"Component_1": tx, "Component_2": ty, "crop_index": np.arange(len(tsne))})


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


# === MAIN SCRIPT ===
def main():
    # Get number of samples
    if sampling_type == "all":
        image_path = f"/data1/crops/{crop_name}/{file_extension}/1/"
        crop_path_list = sorted(glob(image_path + "*." + file_extension))
        n_samples = len(crop_path_list)
    else:
        n_samples = 1000  # default per-cluster sample size

    # Define paths
    output_path = f"/data1/fig/{run_name}/epoch_{epoch}/{sampling_type}/"
    filename = f"{reduction_method}_pca_cosine_perp-50_{run_name}_{random_state}_epoch_{epoch}.npy"

    # Load data
    tsne_df = load_tsne_coordinates(output_path, filename)
    labels_df = load_labels(output_path, run_name, sampling_type, n_samples, FROM_CROP_STATS)

    # Merge + filter
    merged_df = merge_and_filter(tsne_df, labels_df)
    print(merged_df)

    # Save
    save_output(merged_df, output_path, run_name, sampling_type, random_state, epoch, FROM_CROP_STATS)


if __name__ == "__main__":
    main()
