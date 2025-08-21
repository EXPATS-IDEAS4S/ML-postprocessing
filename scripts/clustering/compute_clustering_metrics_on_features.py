import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import torch
import pandas as pd
import os
import glob

# === Configuration ===
scales = ['dcv2_ir108_ot_100x100_k6_35k_nc_vit',
          'dcv2_ir108_ot_100x100_k7_35k_nc_vit',
          'dcv2_ir108_ot_100x100_k8_35k_nc_vit',
          'dcv2_ir108_ot_100x100_k9_35k_nc_vit',
          'dcv2_ir108_ot_100x100_k10_35k_nc_vit',
          'dcv2_ir108_ot_100x100_k11_35k_nc_vit',
          'dcv2_ir108_ot_100x100_k12_35k_nc_vit',
          'dcv2_ir108_ot_100x100_k13_35k_nc_vit',
          'dcv2_ir108_ot_100x100_k14_35k_nc_vit',
          'dcv2_ir108_ot_100x100_k15_35k_nc_vit']

epochs = [500]
n_crops = 35092
filename_features = 'rank0_chunk0_train_heads_features.npy'
sampling_type = 'all'

# === Output directory ===
output_dir = f"/data1/fig/k_optimization/dcv2_ir108_ot_100x100_35k_nc_vit/"
os.makedirs(output_dir, exist_ok=True)

results = []

for scale in scales:
    for epoch in epochs:
        print(f"\n==> Processing scale: {scale}, epoch: {epoch}")

        # === Load feature vectors ===
        feature_path = f'/data1/runs/{scale}/features/epoch_{epoch}/{filename_features}'
        if not os.path.exists(feature_path):
            print(f"⚠️ Feature file not found: {feature_path}")
            continue

        data = np.load(feature_path)
        data = np.reshape(data, (n_crops, -1))  # Should be (35092, 128)
        print(f"✔ Features loaded: {data.shape}")

        # === Load cluster assignments ===
        assignment_path = f'/data1/runs/{scale}/checkpoints/assignments.pt'
        if not os.path.exists(assignment_path):
            print(f"⚠️ Assignment file not found: {assignment_path}")
            continue

        assignments = torch.load(assignment_path, map_location='cpu')
        cluster_labels = assignments[0].cpu().numpy()
        print(f"✔ Cluster labels loaded: {np.unique(cluster_labels)}")

        # Filter out invalid cluster label (-100)
        valid_mask = cluster_labels != -100
        data = data[valid_mask]
        cluster_labels = cluster_labels[valid_mask]

        print(f"✔ Filtered features: {data.shape}, valid clusters: {np.unique(cluster_labels)}")

        # === Compute clustering metrics directly on features ===
        try:
            silhouette = silhouette_score(data, cluster_labels, metric='cosine')
            calinski = calinski_harabasz_score(data, cluster_labels)
            davies = davies_bouldin_score(data, cluster_labels)
        except Exception as e:
            print(f"❌ Error computing metrics: {e}")
            continue

        results.append({
            "Scale": scale,
            "Epoch": epoch,
            "Silhouette Mean": silhouette,
            "Calinski-Harabasz Mean": calinski,
            "Davies-Bouldin Mean": davies
        })

# === Save results ===
df = pd.DataFrame(results)
output_file = os.path.join(output_dir, f"clustering_metrics_on_features.csv")
df.to_csv(output_file, index=False)

print(f"\n✅ Clustering metrics saved to: {output_file}")
