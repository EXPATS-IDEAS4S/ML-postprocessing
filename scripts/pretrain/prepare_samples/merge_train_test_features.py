"""
Prepare training and test feature datasets with metadata for visualization or analysis.

This script:
1. Loads feature vectors for training and test crops.
2. Identifies special case-study crops (MSG and ICON) in the test set.
3. Adds metadata columns including 'vector_type' and 'case_study'.
4. Merges training and filtered test datasets.
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
run_name = 'dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc'
event_types = ["PRECIP", "HAIL"]
train_feat_dir = f'/data1/runs/{run_name}/features/epoch_800/'
n_dim = 128  # Feature vector dimension
output_path = f'/data1/fig/{run_name}/test/'
os.makedirs(output_path, exist_ok=True)

# Paths to crops
image_train_path = '/data1/crops/ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc/nc/1/'
# Feature files
feature_file_train_inds = 'rank0_chunk0_train_heads_inds.npy'
feature_file_train_features = 'rank0_chunk0_train_heads_features.npy'
feature_file_test_inds = 'rank0_chunk0_train_heads_inds.npy'
feature_file_test_features = 'rank0_chunk0_train_heads_features.npy'

#assignemnts
train_assignments_file = f'/data1/runs/{run_name}/checkpoints/assignments.pt'
train_distances_file = f'/data1/runs/{run_name}/checkpoints/distances.pt'
train_centroids_file = f'/data1/runs/{run_name}/checkpoints/centroids0.pt'


# =================================================

def extract_timestamp(path: str, extension: str = '.nc') -> str:
    """Extract timestamp from crop file path."""
    ds = xr.open_dataset(path, engine ='h5netcdf')
    time = pd.to_datetime(ds.time.values).strftime('%Y%m%d_%H%M')

    return time

def load_features_to_df(feature_path: str, indices_file: str, features_file: str, 
                        assignments_file: str, distances_file: str, centroids_file: str,
                        vector_type: str, dataset: str, case_study: bool, crops_path: str) -> pd.DataFrame:
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
 
    df['vector_type'] = vector_type
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

    # Load training features
    df_train = load_features_to_df(
        feature_path=train_feat_dir,
        indices_file=feature_file_train_inds,
        features_file=feature_file_train_features,
        assignments_file=train_assignments_file,
        distances_file=train_distances_file,
        centroids_file=train_centroids_file,
        vector_type='msg',
        dataset='train',  # For logging purposes
        case_study=False,
        crops_path=image_train_path
    )
    print(df_train)
    
    # # Save to CSV
    # output_train_csv = os.path.join(output_path, f'features_train_{run_name}.csv')
    # df_train.to_csv(output_train_csv, index=False)
    # print(f"Saved merged dataset with metadata to: {output_train_csv}")
    
    df_test_list = []
    for event_type in event_types:
        image_test_path = f'/data1/crops/test_case_essl_13-14-15-17-18-19-20-22-23-24_100x100_ir108_cma/{event_type}/nc/1/' #png for visualization
        test_feat_dir = f'/data1/runs/{run_name}/test_features/epoch_800/{event_type}/'

        # Load test features
        df_test = load_features_to_df(
            feature_path=test_feat_dir,
            indices_file=feature_file_test_inds,
            features_file=feature_file_test_features,
            assignments_file=train_assignments_file,  # Using train assignments for test set
            distances_file=train_distances_file,      # Using train distances for test set
            centroids_file=train_centroids_file,
            vector_type=event_type,
            dataset='test',  # For logging purposes
            case_study=True,
            crops_path=image_test_path
        )
        print(df_test)
        df_test_list.append(df_test)
    
        #save test case study crops
        output_test_csv = os.path.join(output_path, f'features_test_case_study_{run_name}_{event_type}.csv')
        df_test.to_csv(output_test_csv, index=False)
        print(f"Saved test case-study dataset with metadata to: {output_test_csv}")
    
    # Merge datasets
    df_final = pd.concat([df_train, *df_test_list], ignore_index=True)
    print(df_final)
    
    # Save to CSV
    output_csv = os.path.join(output_path, f'features_{run_name}.csv')
    df_final.to_csv(output_csv, index=False)
    print(f"Saved merged dataset with metadata to: {output_csv}")

if __name__ == "__main__":
    prepare_and_save_dataset()
#649788