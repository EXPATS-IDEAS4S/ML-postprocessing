"""
This code :
1. Loads feature vectors for a given folder of crops provided in the config
3. Adds metadata columns for labels, distances, centroids, images paths, timestamp
5. Saves the final dataset as a CSV file.

Modules:
    - numpy, pandas: for data manipulation.
    - glob, os: for file path handling.
    - gc: for garbage collection.

Configuration:
    - training_run: Identifier for training dataset.
    - test_run: Identifier for test dataset.
    - feature dimension (n_dim), feature filenames.
    - Input paths for training/test crops.
    - Output path for saving final CSV.


Call for running the script:
python3 scripts/pretrain/prepare_samples/create_csv_features.py 
conda run -n vissl python scripts/pretrain/prepare_samples/create_csv_features.py
"""

import numpy as np
import pandas as pd
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from glob import glob
import gc
import torch
from sklearn.metrics.pairwise import cosine_similarity
import xarray as xr
import pdb

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from utils.configs import load_config

CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "process_run_GRL.yaml"


def load_feature_preparation_config(config_path: Path = CONFIG_PATH) -> dict:
    """Load the YAML sections needed for feature CSV creation."""
    config = load_config(str(config_path))

    return {
        "experiment": config["experiment"],
        "data": config["data"],
        "features_preparation": config["features_preparation"],
    }


CONFIG = load_feature_preparation_config()
EXPERIMENT_CONFIG = CONFIG["experiment"]
DATA_CONFIG = CONFIG["data"]
FEATURES_CONFIG = CONFIG["features_preparation"]

# ================= CONFIGURATION =================
run_name = EXPERIMENT_CONFIG["run_names"][0]
epoch = EXPERIMENT_CONFIG["epoch"]
base_path = EXPERIMENT_CONFIG["base_path"]
feature_path = os.path.join(base_path, run_name, "features", f"epoch_{epoch}")
n_dim = FEATURES_CONFIG["n_dim"]
output_path = os.path.join(EXPERIMENT_CONFIG["path_out"], run_name, FEATURES_CONFIG["output_path"])
os.makedirs(output_path, exist_ok=True)
video_keyword = FEATURES_CONFIG["video"]

# construct csv output filename
backbone = EXPERIMENT_CONFIG["backbone"]
crop_resolution = EXPERIMENT_CONFIG["crop_resolution"]
n_input_layers = EXPERIMENT_CONFIG["n_input_layers"]
csv_filename = f'{run_name}-features_backbone_{backbone}_cropres_{crop_resolution}_inputvars_{n_input_layers}_epochs_{epoch}.csv'
output_csv = os.path.join(output_path, csv_filename)    

# Paths to crops
image_path = FEATURES_CONFIG["images_path"]
crops_path = FEATURES_CONFIG["crops_path"]
# Feature files
feature_file_train_inds = FEATURES_CONFIG["feature_file_train_inds"]
feature_file_train_features = FEATURES_CONFIG["feature_file_train_features"]


# assignemnts (read from experiment and run_names from process_run_grl.yaml)
train_assignments_file = os.path.join(base_path, run_name, "checkpoints", "assignments.pt")
train_distances_file = os.path.join(base_path, run_name, "checkpoints", "distances.pt")
train_centroids_file = os.path.join(base_path, run_name, "checkpoints", "centroids0.pt")


# =================================================

def extract_timestamp(path: str, extension: str = '.nc') -> str:
    """Extract timestamp from crop file path."""
    # read time stamp from the file name in the format YYYYMMDD_HHMM_HHMM from file names like 20230930_2000_0300_IR_108_cma_1.nc
    time = path[27:45]
    return time


def build_video_image_index(image_path: str) -> Dict[str, List[str]]:
    """Group video frame images by crop stem and sort them by frame index."""
    frame_pattern = re.compile(r"^(?P<crop_stem>.+)_t(?P<frame_idx>\d+)_.*\.png$")
    image_index: Dict[str, List[Tuple[int, str]]] = {}

    for path in glob(os.path.join(image_path, "*.png")):
        match = frame_pattern.match(os.path.basename(path))
        if not match:
            continue

        crop_stem = match.group("crop_stem")
        frame_idx = int(match.group("frame_idx"))
        image_index.setdefault(crop_stem, []).append((frame_idx, path))

    return {
        crop_stem: [path for _, path in sorted(frame_paths, key=lambda item: item[0])]
        for crop_stem, frame_paths in image_index.items()
    }

def load_features_to_df(feature_path: str, indices_file: str, features_file: str, 
                        assignments_file: str, distances_file: str, centroids_file: str,
                        vector_type: str, image_path: str, crops_path: str, video: bool) -> pd.DataFrame:
    """Load features and indices, return as DataFrame with metadata.
    
    input:
        feature_path: path to the folder containing the feature files
        indices_file: name of the file containing the indices of the crops corresponding to the features
        features_file: name of the file containing the feature vectors
        assignments_file: name of the file containing the cluster assignments for each feature vector
        distances_file: name of the file containing the distances to the assigned cluster centroid for each feature vector
        centroids_file: name of the file containing the cluster centroids
        vector_type: type of feature vector (e.g., 'msg' for message vectors)
        image_path: path to the folder containing the original images corresponding to the crops
        crops_path: path to the folder containing the crop files (used to extract timestamps)
        video: whether the dataset is for video frames (affects how image paths are assigned)
    output:
        DataFrame with columns for feature dimensions, metadata (labels, distances, timestamps), and image paths.
    
    """



    indices = np.load(os.path.join(feature_path, indices_file))
    features = np.load(os.path.join(feature_path, features_file))

    crop_paths = sorted(glob(os.path.join(crops_path, '*.nc')))

    #extract only timestamp from path
    crop_timestamps = [extract_timestamp(p) for p in crop_paths]
    #print(crop_test_timestamps)

    df = pd.DataFrame(
        np.reshape(features, (len(indices), -1)),
        columns=[f'dim_{i+1}' for i in range(features.shape[1])],
        index=indices
    )

    # add timestamp from path
    df['path'] = [crop_paths[i] for i in indices]
    df['datetime'] = [crop_timestamps[i] for i in indices]
    
    #assign labels and distances 
    assignments = torch.load(os.path.join(feature_path, assignments_file), map_location="cpu")
    distances = torch.load(os.path.join(feature_path, distances_file), map_location="cpu")
    df['label'] = assignments[0].cpu().numpy()
    df['distance'] = distances[0].cpu().numpy()
    df['vector_type'] = vector_type

    if video == False:
        # assign the path of the single image corresponding to the crop
        df['image_path'] = df['datetime'].apply(lambda x: os.path.join(image_path, f"{x}_IR_108.png"))
    else:
        image_index = build_video_image_index(image_path)
        image_sequences = df['path'].apply(
            lambda crop_path: image_index.get(Path(crop_path).stem, [])
        ).tolist()

        max_frames = max((len(sequence) for sequence in image_sequences), default=0)
        for frame_idx in range(max_frames):
            df[f't{frame_idx}'] = [
                sequence[frame_idx] if frame_idx < len(sequence) else None
                for sequence in image_sequences
            ]
    return df.reset_index(drop=True)


def prepare_and_save_dataset():
    """Main function to load, merge, and save training and test datasets."""
    gc.collect()

    # # Load features
    df_features = load_features_to_df(
         feature_path=feature_path,
         indices_file=feature_file_train_inds,
         features_file=feature_file_train_features,
         assignments_file=train_assignments_file,
         distances_file=train_distances_file,
         centroids_file=train_centroids_file,
         vector_type='msg',
         image_path=image_path,
         crops_path=crops_path,
         video=video_keyword
    )
    print(df_features)

    # # Save to CSV
    df_features.to_csv(output_csv, index=False)
    print(f"Saved dataset with metadata to: {output_csv}")
    

if __name__ == "__main__":
    prepare_and_save_dataset()
