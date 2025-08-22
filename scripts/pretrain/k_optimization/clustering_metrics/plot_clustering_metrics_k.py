"""
Plot clustering metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz) 
as a function of the number of clusters (k) for multiple runs.

This script:
1. Loads a CSV file containing clustering metrics for different runs.
2. Extracts metrics for each run (based on k).
3. Plots the metrics on a single figure with multiple y-axes.
4. Saves the plot to the specified output directory.

Configuration parameters (runs, file paths, plot settings) are defined at the top.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# =======================
# Configuration
# =======================
RUN_INFO = {
    6: 'dcv2_ir108_ot_100x100_k6_35k_nc_vit',
    7: 'dcv2_ir108_ot_100x100_k7_35k_nc_vit',
    8: 'dcv2_ir108_ot_100x100_k8_35k_nc_vit',
    9: 'dcv2_ir108_ot_100x100_k9_35k_nc_vit',
    10:'dcv2_ir108_ot_100x100_k10_35k_nc_vit',
    11: 'dcv2_ir108_ot_100x100_k11_35k_nc_vit',
    12: 'dcv2_ir108_ot_100x100_k12_35k_nc_vit',
    13: 'dcv2_ir108_ot_100x100_k13_35k_nc_vit',
    14: 'dcv2_ir108_ot_100x100_k14_35k_nc_vit',
    15:'dcv2_ir108_ot_100x100_k15_35k_nc_vit'
}

OUTPUT_DIR = "/data1/fig/k_optimization/dcv2_ir108_ot_100x100_35k_nc_vit/"
CSV_FILE = os.path.join(OUTPUT_DIR, "clustering_metrics_on_features.csv")
PLOT_FILE = os.path.join(OUTPUT_DIR, "clustering_metrics_on_features.png")

# =======================
# Functions
# =======================
def load_metrics(csv_file: str, run_info: dict):
    """Load clustering metrics for each run from the CSV file."""
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Missing metrics file: {csv_file}")

    df = pd.read_csv(csv_file)

    ks, silhouettes, davies_scores, calinski_scores = [], [], [], []

    for k, run_name in run_info.items():
        df_run = df[df['Scale'] == run_name]
        if df_run.empty:
            print(f"⚠️ No data for run: {run_name}")
            continue

        ks.append(k)
        silhouettes.append(df_run["Silhouette Mean"].iloc[0])
        davies_scores.append(df_run["Davies-Bouldin Mean"].iloc[0])
        calinski_scores.append(df_run["Calinski-Harabasz Mean"].iloc[0])

    return ks, silhouettes, davies_scores, calinski_scores


def plot_metrics(ks, silhouettes, davies_scores, calinski_scores, plot_file: str):
    """Plot clustering metrics against number of clusters (k)."""
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

    # Increase tick label size
    ax1.tick_params(axis='both', labelsize=12)
    ax2.tick_params(axis='both', labelsize=12)
    ax3.tick_params(axis='both', labelsize=12)

    # Combine legends from all axes
    lines, labels = ax1.get_legend_handles_labels()
    for ax in [ax2, ax3]:
        l, lab = ax.get_legend_handles_labels()
        lines += l
        labels += lab
    ax1.legend(lines, labels, loc="upper left", fontsize=10)

    # Title and layout
    plt.title("Clustering Metrics vs. Number of Clusters (k)", fontsize=16, fontweight='bold')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()


# =======================
# Main Execution
# =======================
def main():
    ks, silhouettes, davies_scores, calinski_scores = load_metrics(CSV_FILE, RUN_INFO)
    plot_metrics(ks, silhouettes, davies_scores, calinski_scores, PLOT_FILE)


if __name__ == "__main__":
    main()
