"""
Plot multiple clustering metrics across epochs for a given run.

Uses shared utilities from `check_training_utils.py`.
"""

import os
import pandas as pd
from collections import OrderedDict
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from utils.plotting.check_training_utils import plot_multiple_metrics

# === CONFIGURATION ===
RUN_NAME = 'grl_2026_k10'
OUTPUT_DIR = f"/sat_data/fig/{RUN_NAME}/clustering_metrics_output/"
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
