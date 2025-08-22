"""
Compute clustering metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin) for feature vectors
from multiple model scales and epochs. 

This script:
1. Loads feature vectors and cluster assignments for each scale and epoch.
2. Filters out invalid cluster labels (-100).
3. Computes clustering metrics directly on the features.
4. Saves the results to a CSV file.

All configuration parameters (scales, epochs, file paths, etc.) are defined at the top.
"""

import os
import numpy as np
import pandas as pd
import sys

sys.path.append("/home/Daniele/codes/VISSL_postprocessing/utils/analysis_utils")
from utils_clustering import load_cluster_labels, compute_clustering_metrics, filter_invalid_clusters



# =======================
# Configuration Parameters
# =======================
SCALES = [
    'dcv2_ir108_ot_100x100_k6_35k_nc_vit',
    'dcv2_ir108_ot_100x100_k7_35k_nc_vit',
    'dcv2_ir108_ot_100x100_k8_35k_nc_vit',
    'dcv2_ir108_ot_100x100_k9_35k_nc_vit',
    'dcv2_ir108_ot_100x100_k10_35k_nc_vit',
    'dcv2_ir108_ot_100x100_k11_35k_nc_vit',
    'dcv2_ir108_ot_100x100_k12_35k_nc_vit',
    'dcv2_ir108_ot_100x100_k13_35k_nc_vit',
    'dcv2_ir108_ot_100x100_k14_35k_nc_vit',
    'dcv2_ir108_ot_100x100_k15_35k_nc_vit'
]

EPOCHS = [500]
N_CROPS = 35092
FILENAME_FEATURES = 'rank0_chunk0_train_heads_features.npy'
SAMPLING_TYPE = 'all'

RUNS_DIR = '/data1/runs'
OUTPUT_DIR = "/data1/fig/k_optimization/dcv2_ir108_ot_100x100_35k_nc_vit/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "clustering_metrics_on_features.csv")


# =======================
# Functions
# =======================
def load_features(scale: str, epoch: int) -> np.ndarray:
    """Load and reshape feature vectors for a given scale and epoch."""
    feature_path = os.path.join(RUNS_DIR, scale, f'features/epoch_{epoch}', FILENAME_FEATURES)
    if not os.path.exists(feature_path):
        print(f"⚠️ Feature file not found: {feature_path}")
        return None
    data = np.load(feature_path)
    return np.reshape(data, (N_CROPS, -1))




def main():
    results = []

    for scale in SCALES:
        for epoch in EPOCHS:
            print(f"\n==> Processing scale: {scale}, epoch: {epoch}")

            data = load_features(scale, epoch)
            if data is None:
                continue

            labels = load_cluster_labels(scale)
            if labels is None:
                continue

            data, labels = filter_invalid_clusters(data, labels)
            print(f"✔ Filtered features: {data.shape}, valid clusters: {np.unique(labels)}")

            metrics = compute_clustering_metrics(data, labels, metric='cosine')
            if metrics is None:
                continue

            results.append({
                "Scale": scale,
                "Epoch": epoch,
                **metrics
            })

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Clustering metrics saved to: {OUTPUT_FILE}")


# =======================
# Run Script
# =======================
if __name__ == "__main__":
    main()

