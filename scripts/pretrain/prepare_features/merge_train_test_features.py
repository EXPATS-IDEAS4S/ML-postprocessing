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

# ================= CONFIGURATION =================
training_run = 'dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA'
test_run = 'dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA_test_msg_icon'

n_dim = 128  # Feature vector dimension
output_path = f'/home/Daniele/fig/{test_run}/'
os.makedirs(output_path, exist_ok=True)

# Paths to crops
image_train_path = '/data1/crops/ir108_2013-2014-2015-2016_200K-300K_CMA/1/'
image_test_path = '/data1/crops/ir108_2013-2014-2015-2016_200K-300K_CMA_test_msg_icon/1/'

# Feature files
feature_file_train_inds = 'rank0_chunk0_train_heads_inds.npy'
feature_file_train_features = 'rank0_chunk0_train_heads_features.npy'

feature_file_test_inds = 'rank0_chunk0_train_heads_inds.npy'
feature_file_test_features = 'rank0_chunk0_train_heads_features.npy'

# Case-study patterns
case_study_msg_pattern = 'IR_108_128x128_20220915_*_200K-300K_greyscale_CMA.tif'
case_study_icon_pattern = 'cropped_icon_*_200K-300K_greyscale.tif'
# =================================================

def load_features_to_df(feature_path: str, indices_file: str, features_file: str, vector_type: str) -> pd.DataFrame:
    """Load features and indices, return as DataFrame with metadata."""
    indices = np.load(os.path.join(feature_path, indices_file))
    features = np.load(os.path.join(feature_path, features_file))
    
    df = pd.DataFrame(
        np.reshape(features, (len(indices), -1)),
        columns=[f'dim_{i+1}' for i in range(features.shape[1])],
        index=indices
    )
    df['vector_type'] = vector_type
    df['case_study'] = False
    return df.reset_index(drop=True)

def mark_case_study(df: pd.DataFrame, crop_path_list: list, msg_pattern: str, icon_pattern: str) -> pd.DataFrame:
    """Mark crops in the test DataFrame as case-study (MSG or ICON)."""
    df['location'] = crop_path_list
    df['case_study_msg'] = df['location'].str.contains(msg_pattern)
    df['case_study_icon'] = df['location'].str.contains(icon_pattern)
    df['vector_type'] = np.where(df['case_study_icon'], 'icon', 'msg')
    df['case_study'] = df['case_study_msg'] | df['case_study_icon']
    df = df[df['case_study']]  # Keep only case-study rows
    return df.drop(columns=['case_study_msg', 'case_study_icon', 'location']).reset_index(drop=True)

def prepare_and_save_dataset():
    """Main function to load, merge, and save training and test datasets."""
    gc.collect()
    
    # Load training features
    df_train = load_features_to_df(
        feature_path=f'/data1/runs/{training_run}/features/',
        indices_file=feature_file_train_inds,
        features_file=feature_file_train_features,
        vector_type='msg'
    )
    
    # Load test features
    df_test = load_features_to_df(
        feature_path=f'/data1/runs/{test_run}/features/',
        indices_file=feature_file_test_inds,
        features_file=feature_file_test_features,
        vector_type='msg'
    )
    
    # Get test crop file paths
    crop_test_paths = sorted(glob(os.path.join(image_test_path, '*.tif')))
    
    # Mark case-study crops
    df_test = mark_case_study(df_test, crop_test_paths, case_study_msg_pattern, case_study_icon_pattern)
    
    # Merge datasets
    df_final = pd.concat([df_train, df_test], ignore_index=True)
    
    # Save to CSV
    output_csv = os.path.join(output_path, f'features_{test_run}.csv')
    df_final.to_csv(output_csv, index=False)
    print(f"Saved merged dataset with metadata to: {output_csv}")

if __name__ == "__main__":
    prepare_and_save_dataset()
