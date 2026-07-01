"""
Prepare test feature datasets with metadata for visualization or analysis.

This script:
1. Loads feature vectors from training and test crops.
2. Calculates cosine similarity between test features and training centroids to assign labels to test samples.
3. Adds metadata columns including 'crop_type', 'distance'
4. Stores test dataset with assigned labels and metadata in a csv file for further analysis.

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


how to run 
nohup python prepare_test_features_grl2026.py > prepare_test_features_grl2026.log 2>&1 &

pid: 4041551

"""

import numpy as np
import pandas as pd
import os
from glob import glob
import gc
import torch
from sklearn.metrics.pairwise import cosine_similarity
import xarray as xr

# ================= CONFIGURATION =================
run_name = 'grl_2026_k10'
#event_types = ["PRECIP", "HAIL"]
train_feat_dir = f'/sat_data/runs/{run_name}/features/epoch_800/'
n_dim = 128  # Feature vector dimension
output_path = f'/sat_data/fig/{run_name}/test/'
os.makedirs(output_path, exist_ok=True)

# Paths to crops
#image_train_path = '/sat_data/crops/ir108_100x100_2013-2020_3xrandomcrops_1xtimestamp_cma_nc/nc/1/'

# Feature files
feature_file_train_inds = 'rank0_chunk0_train_heads_inds.npy'
feature_file_train_features = 'rank0_chunk0_train_heads_features.npy'
feature_file_test_inds = 'rank0_chunk0_train_heads_inds.npy'
feature_file_test_features = 'rank0_chunk0_train_heads_features.npy'

#assignemnts
train_assignments_file = f'/sat_data/runs/{run_name}/checkpoints/assignments.pt'
train_distances_file = f'/sat_data/runs/{run_name}/checkpoints/distances.pt'
train_centroids_file = f'/sat_data/runs/{run_name}/checkpoints/centroids0.pt'


# =================================================

def extract_timestamp(path: str, extension: str = '.nc') -> str:
    """Extract timestamp from crop file path."""
    ds = xr.open_dataset(path, engine ='h5netcdf')
    time = pd.to_datetime(ds.time.values).strftime('%Y%m%d_%H%M')

    return time

def load_features_to_df(feature_path: str, indices_file: str, features_file: str, 
                        assignments_file: str, distances_file: str, centroids_file: str,
                        dataset: str, case_study: bool, crops_path: str) -> pd.DataFrame:
    """Load features and indices, return as DataFrame with metadata."""
    indices = np.load(os.path.join(feature_path, indices_file))
    features = np.load(os.path.join(feature_path, features_file))

    crop_test_paths = sorted(glob(os.path.join(crops_path, '*.nc')))
    #print(len(crop_test_paths))
    #extract only timestamp from path
    crop_test_timestamps = [extract_timestamp(p) for p in crop_test_paths]
    #print(crop_test_timestamps)

    df = pd.DataFrame(
        np.reshape(features, (len(indices), -1)),
        columns=[f'dim_{i+1}' for i in range(features.shape[1])],
        index=indices
    )

    # add timestamp from path
    df['path'] = [crop_test_paths[i] for i in indices]
    df['datetime'] = [crop_test_timestamps[i] for i in indices]
    
    #assign labels and distances 
    if dataset == 'train':
        assignments = torch.load(os.path.join(feature_path, assignments_file), map_location="cpu")
        distances = torch.load(os.path.join(feature_path, distances_file), map_location="cpu")
        #print(assignments.cpu().numpy().shape, distances.cpu().numpy().shape)
        df['label'] = assignments[0].cpu().numpy()
        df['distance'] = distances[0].cpu().numpy()
    else:
        df = add_labels_to_test(df, centroids_file)
 
    df['case_study'] = case_study
    
    return df.reset_index(drop=True)


def add_labels_to_test(df: pd.DataFrame, centroids_file: str) -> pd.DataFrame:
    """Assign labels to test DataFrame based on cosine similarity to centroids."""
    # Load centroids
    centroids = torch.load(centroids_file, map_location="cpu").cpu().numpy()
    print(f"Centroids shape: {centroids.shape}")
    # Extract features from dataframe
    features = df[[f'dim_{i+1}' for i in range(centroids.shape[1])]].values
    print(f"Features shape: {features.shape}")
    
    # Compute cosine similarity: shape (n_samples, n_centroids)
    sim = cosine_similarity(features, centroids)
    print(f"Cosine similarity shape: {sim.shape}")
    
    # Assign label of most similar centroid
    df["label"] = np.argmax(sim, axis=1)
    
    # Optionally, also keep the similarity score for confidence
    df["distance"] = np.max(sim, axis=1)
    
    return df


def prepare_and_save_dataset():
    """Main function to load, merge, and save training and test datasets."""
    gc.collect()

    # # Load training features
    # df_train = load_features_to_df(
    #     feature_path=train_feat_dir,
    #     indices_file=feature_file_train_inds,
    #     features_file=feature_file_train_features,
    #     assignments_file=train_assignments_file,
    #     distances_file=train_distances_file,
    #     centroids_file=train_centroids_file,
    #     vector_type='msg',
    #     dataset='train',  # For logging purposes
    #     case_study=False,
    #     crops_path=image_train_path
    # )
    # print(df_train)
    
    # # Save to CSV
    # output_train_csv = os.path.join(output_path, f'features_train_{run_name}.csv')
    # df_train.to_csv(output_train_csv, index=False)
    # print(f"Saved merged dataset with metadata to: {output_train_csv}")
    
    # Load test features for each event type and save separately
    df_test_list = []
    
    image_test_path = f'/sat_data/crops/test_grl_2026/1/' #png for visualization
    test_feat_dir = f'/sat_data/runs/{run_name}/features_test/'
    # /sat_data/runs/grl_2026/case_studies_features

    # Load test features
    df_test = load_features_to_df(
        feature_path=test_feat_dir,
        indices_file=feature_file_test_inds,
        features_file=feature_file_test_features,
        assignments_file=train_assignments_file,  # Using train assignments for test set
        distances_file=train_distances_file,      # Using train distances for test set
        centroids_file=train_centroids_file,
        dataset='test',  # For logging purposes
        case_study=True,
        crops_path=image_test_path
    )

    print(df_test)
    df_test_list.append(df_test)
    
    #save test case study crops
    output_test_csv = os.path.join(output_path, f'features_test_dataset_ESSL_{run_name}.csv')
    df_test.to_csv(output_test_csv, index=False)
    print(f"Saved test case-study dataset with metadata to: {output_test_csv}")

    # # Merge datasets
    # df_final = pd.concat([df_train, *df_test_list], ignore_index=True)
    # print(df_final)
    
    # # Save to CSV
    # output_csv = os.path.join(output_path, f'features_{run_name}.csv')
    # df_final.to_csv(output_csv, index=False)
    # print(f"Saved merged dataset with metadata to: {output_csv}")

if __name__ == "__main__":
    prepare_and_save_dataset()
