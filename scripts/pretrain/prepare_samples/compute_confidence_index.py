"""
conf_margin → Best metric

High confidence = sample strongly belongs to one class

Low confidence = sample near a cluster boundary

Near-zero = ambiguous case, “transition state”

conf_std

Large → one centroid is clearly closest

Small → sample is equidistant from many → uncertainty

conf_slope

Large → distribution spreads (distinct best)

Small → distances nearly flat → uncertainty
"""

import numpy as np
import pandas as pd
import os
import torch
import xarray as xr
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
import os


def extract_timestamp(path: str) -> str:
    """Extract timestamp from crop file path."""
    ds = xr.open_dataset(path, engine='h5netcdf')
    time = pd.to_datetime(ds.time.values).strftime('%Y%m%d_%H%M')
    ds.close()
    return time


def load_features_to_df(feature_path, indices_file, features_file,
                        assignments_file, distances_file,
                        centroids_file):
    """Load features and metadata into a DataFrame"""
    
    # load np arrays
    indices = np.load(os.path.join(feature_path, indices_file))
    features = np.load(os.path.join(feature_path, features_file))

    df = pd.DataFrame(
        np.reshape(features, (len(indices), -1)),
        columns=[f"dim_{i+1}" for i in range(features.shape[1])],
        index=indices
    )

    # load assignments + distances
    assignments = torch.load(os.path.join(feature_path, assignments_file), map_location="cpu")
    distances = torch.load(os.path.join(feature_path, distances_file), map_location="cpu")

    df["label"] = assignments[0].cpu().numpy()
    df["distance"] = distances[0].cpu().numpy()

    #remove invalid class -100
    df = df[df["label"] != -100]

    # load centroids
    centroids = torch.load(os.path.join(feature_path, centroids_file), map_location="cpu").numpy()

    return df.reset_index(drop=True), centroids



def compute_confidence_indices(df, centroids):
    """
    Compute multiple confidence scores based on similarity to all centroids.
    Scores measure how 'certain' the class assignment is.
    """

    # ---- extract feature matrix ----
    feature_cols = [c for c in df.columns if c.startswith("dim_")]
    feat_matrix = df[feature_cols].values        # shape (N, 128)
    
    # ---- compute cosine similarity to all centroids ----
    sim_matrix = cosine_similarity(feat_matrix, centroids)  # shape (N, K)
    
    # ---- convert similarity to "distance" (bounded 0–2) ----
    # distance = 1 - cosine similarity
    dist_matrix = 1 - sim_matrix

    # ---- for each sample get sorted distances ----
    sorted_dists = np.sort(dist_matrix, axis=1)            # ascending: best centroid = lowest
    best = sorted_dists[:, 0]
    second = sorted_dists[:, 1]

    # 1) MARGIN SCORE: difference between 2 closest centroids
    margin_score = second - best      # larger = more confident

    # 2) STD SCORE: variability among all centroid distances
    std_score = dist_matrix.std(axis=1)

    # 3) SLOPE SCORE: slope between best and worst distances
    slope_score = sorted_dists[:, -1] - sorted_dists[:, 0]

    # ---- add to dataframe ----
    df["conf_margin"] = margin_score
    df["conf_std"] = std_score
    df["conf_slope"] = slope_score

    # ---- normalized confidence (0–1) for convenience ----
    df["conf_margin_norm"] = (margin_score - margin_score.min()) / (margin_score.max() - margin_score.min() + 1e-6)
    
    return df


def plot_confidence_distributions(df, output_dir, metrics=None):
    """
    Plot distribution of confidence metrics for each class.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: 'label' and metrics (e.g. confidence_margin, confidence_std, confidence_slope)
    output_dir : str
        Directory where plots will be saved
    metrics : list or None
        Which metrics to plot. Default: all three.
    """
    
    #check if metrics are in df columns
    if metrics in df.columns:
        print(f"Metrics found in dataframe: {metrics}")
    else:
        print(f"Some metrics not found in dataframe columns. Available columns: {df.columns.tolist()}")
        exit(1)
    

    os.makedirs(output_dir, exist_ok=True)

    labels_sorted = sorted(df["label"].unique())

    for metric in metrics:
        plt.figure(figsize=(10, 5))

        # violinplot
        sns.violinplot(
            data=df,
            x="label",
            y=metric,
            inner=None,
            scale="width",
            palette="Set2",
        )

        # add a thin boxplot on top
        sns.boxplot(
            data=df,
            x="label",
            y=metric,
            width=0.2,
            color="black",
            showcaps=False,
            boxprops={"facecolor": "none"},
            whiskerprops={"linewidth": 1.5},
            fliersize=0,
        )

        plt.title(f"Distribution of {metric} per class", fontsize=14, fontweight="bold")
        plt.xlabel("Class", fontsize=12)
        plt.ylabel(metric.replace("confidence_", "").capitalize(), fontsize=12)
        plt.xticks(labels_sorted)
        plt.grid(axis='y', alpha=0.3)

        out_path = os.path.join(output_dir, f"{metric}_distribution_per_class.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"✅ Saved: {out_path}")


# ================= CONFIGURATION =================
run_name = 'dcv2_resnet_k8_ir108_100x100_2013-2020_1xrandomcrops_1xtimestamp_cma_nc_convective'
epoch = 'epoch_800'
sampling_mode = 'all'   
train_feat_dir = f'/data1/runs/{run_name}/features/{epoch}/'

# Output csf file
output_path = f'/data1/fig/{run_name}/{epoch}/{sampling_mode}/'
os.makedirs(output_path, exist_ok=True)

# Feature files
feature_file_train_inds = 'rank0_chunk0_train_heads_inds.npy'
feature_file_train_features = 'rank0_chunk0_train_heads_features.npy'

train_assignments_file = f'/data1/runs/{run_name}/checkpoints/assignments.pt'
train_distances_file = f'/data1/runs/{run_name}/checkpoints/distances.pt'
train_centroids_file = f'/data1/runs/{run_name}/checkpoints/centroids0.pt'


def main():
    print("Loading features...")
    df, centroids = load_features_to_df(
        feature_path=train_feat_dir,
        indices_file=feature_file_train_inds,
        features_file=feature_file_train_features,
        assignments_file=train_assignments_file,
        distances_file=train_distances_file,
        centroids_file=train_centroids_file
    )
    print(f"Loaded {len(df)} samples.")
    print(df.head())
    print(centroids.shape)

    #count how many samples belong to the convective classes
    print("class distribution :")
    print(df["label"].value_counts())
    

    print("Computing confidence indices...")
    df = compute_confidence_indices(df, centroids)

    # ==== SAVE CSV ====
    output_csv = os.path.join(output_path, f"df_confidence_{run_name}.csv")
    df.to_csv(output_csv, index=False)

    print(f"Saved selected samples: {output_csv}")

    plot_confidence_distributions(
        df,
        output_dir=output_path+"/confidence_index/",
        metrics=["conf_margin", "conf_std", "conf_slope"]
    )

    
    cluster_conf = df.groupby("label")["conf_margin"].mean()
    print(cluster_conf)



if __name__ == "__main__":
    main()


