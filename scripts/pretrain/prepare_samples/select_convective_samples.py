import numpy as np
import pandas as pd
import os
from glob import glob
import gc
import torch
import xarray as xr
import shutil
from sklearn.metrics.pairwise import cosine_similarity

# ================= CONFIGURATION =================
run_name = 'dcv2_resnet_k7_ir108_100x100_2013-2020_1xrandomcrops_1xtimestamp_cma_nc'
train_feat_dir = f'/data1/runs/{run_name}/features/epoch_800/'

convective_classes = [6, 1, 5]      # Convective class labels
distance_threshold = 0.5          # Cosine-distance threshold for "convectively related"
n_dim = 128                          # Feature dimension

# Output csf file
output_path = f'/data1/fig/{run_name}/convective_selected/'
os.makedirs(output_path, exist_ok=True)

# Crop path
image_train_path = '/data1/crops/ir108_100x100_2013-2020_3xrandomcrops_1xtimestamp_cma_nc/nc/1/'

# Output selected crops
output_nc_dir = '/data1/crops/ir108_100x100_2013-2020_3xrandomcrops_1xtimestamp_cma_nc/nc_convective/1/'
os.makedirs(output_nc_dir, exist_ok=True)


# Feature files
feature_file_train_inds = 'rank0_chunk0_train_heads_inds.npy'
feature_file_train_features = 'rank0_chunk0_train_heads_features.npy'

train_assignments_file = f'/data1/runs/{run_name}/checkpoints/assignments.pt'
train_distances_file = f'/data1/runs/{run_name}/checkpoints/distances.pt'
train_centroids_file = f'/data1/runs/{run_name}/checkpoints/centroids0.pt'


# =================================================

def extract_timestamp(path: str) -> str:
    """Extract timestamp from crop file path."""
    ds = xr.open_dataset(path, engine='h5netcdf')
    time = pd.to_datetime(ds.time.values).strftime('%Y%m%d_%H%M')
    ds.close()
    return time


def load_features_to_df(feature_path, indices_file, features_file,
                        assignments_file, distances_file,
                        centroids_file, crops_path):
    """Load features and metadata into a DataFrame"""
    
    # load np arrays
    indices = np.load(os.path.join(feature_path, indices_file))
    features = np.load(os.path.join(feature_path, features_file))

    # load crop paths
    crop_paths = sorted(glob(os.path.join(crops_path, "*.nc")))
    #crop_timestamps = [extract_timestamp(p) for p in crop_paths]

    df = pd.DataFrame(
        np.reshape(features, (len(indices), -1)),
        columns=[f"dim_{i+1}" for i in range(features.shape[1])],
        index=indices
    )

    df["path"] = [crop_paths[i] for i in indices]
    #df["datetime"] = [crop_timestamps[i] for i in indices]

    # load assignments + distances
    assignments = torch.load(os.path.join(feature_path, assignments_file), map_location="cpu")
    distances = torch.load(os.path.join(feature_path, distances_file), map_location="cpu")

    df["label"] = assignments[0].cpu().numpy()
    df["distance"] = distances[0].cpu().numpy()

    # load centroids
    centroids = torch.load(os.path.join(feature_path, centroids_file), map_location="cpu").numpy()

    return df.reset_index(drop=True), centroids



def select_convective_samples(df, centroids, convective_classes, dist_thresh):
    """Select convective samples + samples close to convective centroids."""

    # ---- get convective centroids ----
    conv_centroids = centroids[convective_classes]  # shape (#conv, 128)

    feature_cols = [c for c in df.columns if c.startswith("dim_")]
    feat_matrix = df[feature_cols].values

    # ---- compute cosine distance to each convective centroid ----
    dist_matrix = cosine_similarity(feat_matrix, conv_centroids)
    #print(dist_matrix.shape)
    #print(dist_matrix[:, :])
 
    min_dist = dist_matrix.max(axis=1)
    closest_centroid_idx = dist_matrix.argmax(axis=1)

    df["dist_to_convective"] = min_dist
    df["closest_conv_centroid"] = [convective_classes[i] for i in closest_centroid_idx]

    # ---- select ----
    df_convective = df[
        (df["label"].isin(convective_classes)) |
        (df["dist_to_convective"] >= dist_thresh)
    ].copy()

    #remove invalid labels
    df_convective = df_convective[df_convective["label"] != -100]

    print("Convective class distribution after selection:")
    print(df_convective["label"].value_counts())

    return df_convective



def copy_nc_files(df, dest_dir):
    """Copy selected .nc files to destination folder."""
    for p in df["path"]:
        if os.path.exists(p):
            shutil.copy(p, dest_dir)



def main():
    print("Loading features...")
    df, centroids = load_features_to_df(
        feature_path=train_feat_dir,
        indices_file=feature_file_train_inds,
        features_file=feature_file_train_features,
        assignments_file=train_assignments_file,
        distances_file=train_distances_file,
        centroids_file=train_centroids_file,
        crops_path=image_train_path
    )
    print(f"Loaded {len(df)} samples.")
    print(df.head())
    print(centroids.shape)

    #count how many samples belong to the convective classes
    print("Convective class distribution before selection:")
    print(df[df["label"].isin(convective_classes)]["label"].value_counts())
    

    print("Selecting convective samples...")
    df_sel = select_convective_samples(
        df=df,
        centroids=centroids,
        convective_classes=convective_classes,
        dist_thresh=distance_threshold
    )

    print(f"Selected {len(df_sel)} convective-related samples.")
    #print(df_sel.head())

    # ==== SAVE CSV ====
    output_csv = os.path.join(output_path, f"convective_selected_{run_name}.csv")
    df_sel.to_csv(output_csv, index=False)
    print(f"Saved selected samples: {output_csv}")

    # ==== COPY FILES ====
    print("Copying .nc files...")
    copy_nc_files(df_sel, output_nc_dir)
    print(f"Done. Copied files to {output_nc_dir}")


if __name__ == "__main__":
    main()
