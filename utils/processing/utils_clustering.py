"""
Utility functions for loading cluster assignments and computing clustering metrics.

Includes:
- load_cluster_labels: Load cluster assignments from checkpoint.
- compute_clustering_metrics: Compute Silhouette, Calinski-Harabasz, and Davies-Bouldin scores.
- filter_invalid_clusters: Remove samples with invalid cluster labels (-100).
"""



import os
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import torch



def load_cluster_labels(scale: str, RUNS_DIR: str) -> np.ndarray:
    """Load cluster assignments for a given scale."""
    checkpoint_path = os.path.join(RUNS_DIR, scale, 'checkpoints/assignments.pt')
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ Assignment file not found: {checkpoint_path}")
        return None
    assignments = torch.load(checkpoint_path, map_location='cpu')
    return assignments[0].cpu().numpy()



def compute_clustering_metrics(data: np.ndarray, labels: np.ndarray, metric='euclidean') -> dict:
    """Compute Silhouette, Calinski-Harabasz, and Davies-Bouldin scores."""
    try:
        silhouette = silhouette_score(data, labels, metric=metric)
        calinski = calinski_harabasz_score(data, labels)
        davies = davies_bouldin_score(data, labels)
        return {
            "Silhouette Mean": silhouette,
            "Calinski-Harabasz Mean": calinski,
            "Davies-Bouldin Mean": davies
        }
    except Exception as e:
        print(f"❌ Error computing metrics: {e}")
        return None



def filter_invalid_clusters(data: np.ndarray, labels: np.ndarray):
    """Filter out samples with invalid cluster label (-100)."""
    valid_mask = labels != -100
    return data[valid_mask], labels[valid_mask]