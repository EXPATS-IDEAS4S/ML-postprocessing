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
import pdb

# === IMPORT HELPER FUNCTIONS ===
sys.path.append("/home/claudia/codes/ML_postprocessing")
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
        # try different crop_stats filenames until one is found
        try:
            fname = f"crops_stats_var{run_name}_{sampling_type}_{n_samples}.csv"
            return pd.read_csv(os.path.join(output_path, fname))
        except FileNotFoundError:
            # read one of the crop_stats files among the list of csv contained in the output crops path
            csv_filenames = glob(os.path.join(output_path, "crops_stats_var*.csv"))
            if csv_filenames:
                fname = csv_filenames[0]
                print(f"Found crop_stats_var CSV file: {fname}")
                return pd.read_csv(fname)
            else:
                raise FileNotFoundError("No crop_stats_var CSV files found in the output path.")
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

def read_fields_from_features(csv_path: str, feature_filename: str, fields: list):
    """Read specific fields from the features CSV file."""
    df = pd.read_csv(os.path.join(csv_path, feature_filename))
    return df[fields]


# === MAIN SCRIPT ===
def main(config_path: str):

    # read config parameters
    config = load_config(config_path)
    run_names = config["experiment"]["run_names"]
    data_base_path = config["data"]["data_base_path"]
    output_path = config["experiment"]["path_out"]
    crop_name = config["data"]["crops_name"]
    file_extension = config["data"]["file_extension"]
    sampling_type = config["data"]["sampling_type"]
    n_subsample = config["data"]["n_subsample"]
    epoch = config["experiment"]["epoch"]
    random_state = config["experiment"]["random_state"]
    from_crop_stats = config["experiment"]["from_crop_stats"]
    reduction_method = config["reduction"]["method"]
    perplexity = config["reduction"]["perplexity"]
    csv_path = config["features_preparation"]["output_path"]
    crop_res = config["experiment"]["crop_resolution"] 
    input_vars = config["experiment"]["n_input_layers"]


    # Get number of samples by browsing the crop nc file directory
    if sampling_type == "all":
        # try two possible paths for image crops
        try:
            image_path = f"{data_base_path}/{crop_name}/{file_extension}/1/"
            crop_path_list = sorted(glob(image_path + "*." + file_extension))
            n_samples = len(crop_path_list)
            print(n_samples)
        except FileNotFoundError:
            image_path = f"{data_base_path}/{crop_name}/1/"
            print(image_path)
            crop_path_list = sorted(glob(image_path + "*." + file_extension))
            n_samples = len(crop_path_list)
            print(n_samples)
    else:
        n_samples = n_subsample  # default per-cluster sample size

    
    for run_name in run_names:
        # Define paths
        output_path = f"{output_path}/{run_name}/epoch_{epoch}/{sampling_type}/"
        os.makedirs(output_path, exist_ok=True)
        filename = f"{reduction_method}_opentsne_{run_name}_{random_state}_epoch_{epoch}.npy"

        # Load data
        tsne_df = load_tsne_coordinates(output_path, filename)
        print(f"Loaded T-SNE coordinates from {filename} with shape {tsne_df.shape} ...")

        # Process based on the selected mode
        if processing == "features":

            # load feature csv and store in a ds only path, distance, label
            feature_filename = f"{run_name}-features_backbone_r2dplus1_cropres_{crop_res}_inputvars_{input_vars}_epochs_{epoch}.csv"
            feature_ds = read_fields_from_features(csv_path, feature_filename, fields=["path", "distance", "label"])

            # merge tsne_df with feature_ds on path, and filter invalid labels
            merged_df = pd.concat(
                [tsne_df.reset_index(drop=True), feature_ds.reset_index(drop=True)],
                axis=1,
            )
            merged_df = merged_df[merged_df["label"] != -100]

        else:
            labels_df = load_labels(csv_path, run_name, sampling_type, n_samples, from_crop_stats)

            # Merge + filter
            merged_df = merge_and_filter(tsne_df, labels_df)

        # Save
        save_output(merged_df, csv_path, run_name, sampling_type, random_state, epoch, from_crop_stats)


if __name__ == "__main__":
    config_path = "/home/claudia/codes/ML_postprocessing/configs/process_run_GRL.yaml"

    # select if for processing you want to read from the crop_stats csv or from the crop_list csv or directly from the features
    processing = "features" # crops_stats # crop_list
    if processing == "features":
        print("Processing from features CSV...")
        main(config_path)
    elif processing == "crops_stats":
        print("Processing from crop_stats CSV...")
        main(config_path)
    elif processing == "crop_list":
        print("Processing from crop_list CSV...")
        main(config_path)
    else:        
        raise ValueError("Invalid processing option. Choose from 'features', 'crops_stats', or 'crop_list'.")  
