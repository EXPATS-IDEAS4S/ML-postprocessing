"""
Plot average training loss over epochs for a given run.

Configuration is set at the top. Uses shared utilities from `utils.plotting`.
"""

import os
import pandas as pd
import sys

sys.path.append("/home/Daniele/codes/VISSL_postprocessing/utils/plot_utils")
from utils.plot_utils.check_training_utils import load_json_lines, compute_epoch_average, plot_metric_over_epochs

# === CONFIGURATION ===
RUN_NAME = 'dcv2_ir108-cm_100x100_8frames_k9_70k_nc_r2dplus1'
OUTPUT_DIR = f"/data1/fig/{RUN_NAME}/clustering_metrics_output/"
FILE_PATH = f'/data1/runs/{RUN_NAME}/checkpoints/stdout.json'
PARAMETER = 'loss'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD DATA ===
data = load_json_lines(FILE_PATH)

# === COMPUTE AVERAGE PER EPOCH ===
epochs, avg_loss = compute_epoch_average(data, PARAMETER)

# === PLOT ===
plot_metric_over_epochs(
    epochs,
    avg_loss,
    label='Loss',
    color='red',
    output_file=f"{OUTPUT_DIR}loss_avg_per_epoch.png"
)
