"""
Compute clustering metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin) on t-SNE projections 
from multiple model RUN_NAMES and epochs.

This script:
1. Loads t-SNE projections for each scale and epoch.
2. Loads cluster assignments.
3. Computes clustering metrics for each t-SNE projection.
4. Aggregates mean and std across multiple t-SNE runs.
5. Saves the results to a CSV file.

All configuration parameters (RUN_NAMES, epochs, file paths, t-SNE settings) are defined at the top.
"""

import os
import glob
import numpy as np
import pandas as pd
import sys

sys.path.append("/home/Daniele/codes/VISSL_postprocessing/utils/analysis_utils")
from utils_clustering import load_cluster_labels, compute_clustering_metrics

# =======================
# Configuration Parameters
# =======================
RUN_NAMES = [
    'dcv2_ir108_ot_100x100_k6_35k_nc_vit',
    'dcv2_ir108_ot_100x100_k7_35k_nc_vit',
    'dcv2_ir108_ot_100x100_k8_35k_nc_vit',
    'dcv2_ir108_ot_100x100_k9_35k_nc_vit',
    'dcv2_ir108_ot_100x100_k10_35k_nc_vit'
]

EPOCHS = [500]
SAMPLING_TYPE = 'all'  # Options: 'all', 'subsampled'
PERPLEXITY = 50        # t-SNE perplexity parameter

RUNS_DIR = '/data1/runs'
TSNE_BASE_DIR = '/data1/fig'
OUTPUT_DIR = "/data1/fig/k_optimization/dcv2_ir108_ot_100x100_35k_nc_vit/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "clustering_metrics_summary.csv")

# =======================
# Functions
# =======================


def process_scale_epoch(scale: str, epoch: int, sampling_type: str, perplexity: int) -> dict:
    """Process all t-SNE files for a given scale and epoch, return aggregated metrics."""
    tsne_path = os.path.join(TSNE_BASE_DIR, scale, f'epoch_{epoch}/{sampling_type}/')
    tsne_filenames = glob.glob(os.path.join(tsne_path, f'tsne_pca_cosine_perp-{perplexity}_{scale}_*.npy'))

    if not tsne_filenames:
        print(f"⚠️ No t-SNE files found for epoch {epoch}. Skipping.")
        return None

    cluster_labels = load_cluster_labels(scale)
    if cluster_labels is None:
        return None

    silhouette_scores, calinski_scores, davies_scores = [], [], []

    for tsne_file in tsne_filenames:
        print(f"  → Processing t-SNE file: {tsne_file}")
        X = np.load(tsne_file)
        metrics = compute_clustering_metrics(X, cluster_labels)
        if metrics is None:
            continue
        silhouette_scores.append(metrics["Silhouette"])
        calinski_scores.append(metrics["Calinski-Harabasz"])
        davies_scores.append(metrics["Davies-Bouldin"])

    if not silhouette_scores:
        return None

    return {
        "Scale": scale,
        "Epoch": epoch,
        "Silhouette Mean": np.mean(silhouette_scores),
        "Silhouette Std": np.std(silhouette_scores),
        "Calinski-Harabasz Mean": np.mean(calinski_scores),
        "Calinski-Harabasz Std": np.std(calinski_scores),
        "Davies-Bouldin Mean": np.mean(davies_scores),
        "Davies-Bouldin Std": np.std(davies_scores)
    }


def main():
    results = []
    for scale in RUN_NAMES:
        for epoch in EPOCHS:
            print(f"\n==> Processing scale: {scale}, epoch: {epoch}")
            metrics = process_scale_epoch(scale, epoch, SAMPLING_TYPE, PERPLEXITY)
            if metrics is not None:
                results.append(metrics)

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Metrics saved to {OUTPUT_FILE}")


# =======================
# Run Script
# =======================
if __name__ == "__main__":
    main()

