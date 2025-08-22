"""
Plot multiple clustering metrics across epochs for a given run.

Uses shared utilities from `check_training_utils.py`.
"""

import os
import pandas as pd
from collections import OrderedDict
import sys

sys.path.append("/home/Daniele/codes/VISSL_postprocessing/utils/plot_utils")
from check_training_utils import plot_multiple_metrics

# === CONFIGURATION ===
RUN_NAME = 'dcv2_ir108-cm_100x100_8frames_k9_70k_nc_r2dplus1'
OUTPUT_DIR = f"/data1/fig/{RUN_NAME}/clustering_metrics_output/"
METRICS_FILE = f"{OUTPUT_DIR}clustering_metrics_summary.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD METRICS ===
metrics_df = pd.read_csv(METRICS_FILE)
epochs = metrics_df["Epoch"]

# === PREPARE METRICS DICT ===
metrics_dict = OrderedDict({
    'Silhouette': (metrics_df["Silhouette Mean"].values, 'green'),
    'Davies-Bouldin': (metrics_df["Davies-Bouldin Mean"].values, 'red'),
    'Calinski-Harabasz': (metrics_df["Calinski-Harabasz Mean"].values, 'purple')
})

# === PLOT ===
plot_multiple_metrics(
    metrics_dict,
    output_file=f"{OUTPUT_DIR}clustering_metrics_plot.png",
    xlabel='Epoch'
)
