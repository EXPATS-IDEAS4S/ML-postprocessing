import matplotlib.pyplot as plt
import pandas as pd
import os

# === Define runs ===
run_info = {
    6: 'dcv2_ir108_ot_100x100_k6_35k_nc_vit',
    7: 'dcv2_ir108_ot_100x100_k7_35k_nc_vit',
    8: 'dcv2_ir108_ot_100x100_k8_35k_nc_vit',
    9: 'dcv2_ir108_ot_100x100_k9_35k_nc_vit',
    10:'dcv2_ir108_ot_100x100_k10_35k_nc_vit',
    11: 'dcv2_ir108_ot_100x100_k11_35k_nc_vit',
    12: 'dcv2_ir108_ot_100x100_k12_35k_nc_vit',
    13: 'dcv2_ir108_ot_100x100_k13_35k_nc_vit',
    14: 'dcv2_ir108_ot_100x100_k14_35k_nc_vit',
    15:'dcv2_ir108_ot_100x100_k15_35k_nc_vit'}

# === Initialize storage ===
ks = []
silhouettes = []
davies_scores = []
calinski_scores = []

output_dir = "/data1/fig/k_optimization/dcv2_ir108_ot_100x100_35k_nc_vit/"
fname = "clustering_metrics_on_features"
metrics_path = f"/{output_dir}/{fname}.csv"

if not os.path.exists(metrics_path):
    print(f"Missing metrics: {metrics_path}")
    exit()

df = pd.read_csv(metrics_path)
print(df.head())

# === Load metrics for each run ===
for k, run_name in run_info.items():
    print(f"Processing run: {run_name} with k={k}")
    #select the rows corresponfing to the run_name
    df_run = df[df['Scale'] == run_name]
    #print(df_run)
    
    
    # If there's only one epoch, take first row
    silhouettes.append(df_run["Silhouette Mean"].iloc[0])
    davies_scores.append(df_run["Davies-Bouldin Mean"].iloc[0])
    calinski_scores.append(df_run["Calinski-Harabasz Mean"].iloc[0])
    ks.append(k)

# === Plot ===
fig, ax1 = plt.subplots(figsize=(10, 6))

# Silhouette Score
ax1.set_xlabel("Number of Clusters (k)", fontsize=14)
ax1.set_ylabel("Silhouette", color="tab:green", fontsize=14)
ax1.plot(ks, silhouettes, label="Silhouette", color="tab:green", marker='o')
ax1.tick_params(axis='y', labelcolor="tab:green")
ax1.set_ylim(0, max(silhouettes)*1.1)

# Davies-Bouldin Score (secondary axis)
ax2 = ax1.twinx()
ax2.spines["right"].set_position(("axes", 1.12))
ax2.set_frame_on(True)
ax2.patch.set_visible(False)
ax2.set_ylabel("Davies-Bouldin", color="tab:red", fontsize=14)
ax2.plot(ks, davies_scores, label="Davies-Bouldin", color="tab:red", marker='s')
ax2.tick_params(axis='y', labelcolor="tab:red")
ax2.set_ylim(min(davies_scores)*0.9, max(davies_scores)*1.1)

# Calinski-Harabasz Score (third axis)
ax3 = ax1.twinx()
ax3.spines["right"].set_position(("axes", 1.24))
ax3.set_frame_on(True)
ax3.patch.set_visible(False)
ax3.set_ylabel("Calinski-Harabasz", color="tab:purple", fontsize=14)
ax3.plot(ks, calinski_scores, label="Calinski-Harabasz", color="tab:purple", marker='^')
ax3.tick_params(axis='y', labelcolor="tab:purple")
ax3.set_ylim(min(calinski_scores)*0.9, max(calinski_scores)*1.1)

# Ticks size
ax1.tick_params(axis='both', labelsize=12)
ax2.tick_params(axis='both', labelsize=12)
ax3.tick_params(axis='both', labelsize=12)

# Collect legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines3, labels3 = ax3.get_legend_handles_labels()
all_lines = lines1 + lines2 + lines3
all_labels = labels1 + labels2 + labels3
ax1.legend(all_lines, all_labels, loc="upper left", fontsize=10)

# Title and layout
plt.title("Clustering Metrics vs. Number of Clusters (k)", fontsize=16, fontweight='bold')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}{fname}.png", dpi=300, bbox_inches='tight')
plt.show()
