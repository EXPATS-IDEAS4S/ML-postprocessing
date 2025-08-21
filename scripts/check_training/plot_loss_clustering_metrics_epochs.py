import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from check_training.plot_training import load_data_new

run_name = 'dcv2_ir108-cm_100x100_8frames_k9_70k_nc_r2dplus1'
output_dir = f"/data1/fig/{run_name}/clustering_metrics_output/"
k = 9

# Load loss data
file_path = f'/data1/runs/{run_name}/checkpoints/'
data = load_data_new(file_path + 'stdout.json')
parameter = 'loss'

# Average loss per epoch
loss_epochs = sorted(set(entry['ep'] for entry in data))
loss_values = [np.mean([e[parameter] for e in data if e['ep'] == ep]) for ep in loss_epochs]

# Load clustering metrics
metrics_path = f'{output_dir}clustering_metrics_summary.csv'
metrics_df = pd.read_csv(metrics_path)

# Extract clustering metrics
epochs = metrics_df["Epoch"]
silhouette = metrics_df["Silhouette Mean"]
davies = metrics_df["Davies-Bouldin Mean"]
calinski = metrics_df["Calinski-Harabasz Mean"]

# === Plot ===
fig, ax1 = plt.subplots(figsize=(11, 5))

# Loss on left y-axis
ax1.set_xlabel("Epoch", fontsize=14)
ax1.set_ylabel("Loss", color="tab:blue", fontsize=14)
ax1.plot(loss_epochs, loss_values, label="Loss", color="tab:blue", marker='.', alpha=0.7)
ax1.tick_params(axis='y', labelcolor="tab:blue")
ax1.set_ylim(0.1, max(loss_values) * 1.1)

# Silhouette on right y-axis (1st)
ax2 = ax1.twinx()
ax2.spines["right"].set_position(("axes", 1.0))
ax2.set_ylabel("Silhouette", color="tab:green", fontsize=14)
ax2.plot(epochs, silhouette, label="Silhouette", color="tab:green", marker='s')
ax2.tick_params(axis='y', labelcolor="tab:green")
ax2.set_ylim(0, 0.2)

# Davies-Bouldin on another right y-axis (2nd)
ax3 = ax1.twinx()
ax3.spines["right"].set_position(("axes", 1.12))  # Slightly offset
ax3.set_frame_on(True)
ax3.patch.set_visible(False)
ax3.set_ylabel("Davies-Bouldin", color="tab:red", fontsize=14)
ax3.plot(epochs, davies, label="Davies-Bouldin", color="tab:red", marker='^')
ax3.tick_params(axis='y', labelcolor="tab:red")
ax3.set_ylim(2, 2.8)

# Calinski-Harabasz on another right y-axis (3rd)
ax4 = ax1.twinx()
ax4.spines["right"].set_position(("axes", 1.21))  # Further offset
ax4.set_frame_on(True)
ax4.patch.set_visible(False)
ax4.set_ylabel("Calinski-Harabasz", color="tab:purple", fontsize=14)
ax4.plot(epochs, calinski, label="Calinski-Harabasz", color="tab:purple", marker='x')
ax4.tick_params(axis='y', labelcolor="tab:purple")
ax4.set_ylim(20000, 28000)

# Imcrease ticks size   
ax1.tick_params(axis='both', labelsize=12)
ax2.tick_params(axis='both', labelsize=12)
ax3.tick_params(axis='both', labelsize=12)
ax4.tick_params(axis='both', labelsize=12)

# Collect legends from all axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines3, labels3 = ax3.get_legend_handles_labels()
lines4, labels4 = ax4.get_legend_handles_labels()

all_lines = lines1 + lines2 + lines3 + lines4
all_labels = labels1 + labels2 + labels3 + labels4
ax1.legend(all_lines, all_labels, loc="upper left", fontsize=10)

# Title and layout
plt.title(f"Loss and Clustering Metrics Across Epochs - k={k}", fontsize=14, fontweight='bold')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}loss_clustering_metrics_plot_stacked.png", dpi=300)


