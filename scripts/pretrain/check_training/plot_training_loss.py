"""
Script to plot the average training loss over epochs for one or more runs.

The script:
1. Loads configuration from a YAML file.
2. Reads training logs (JSON lines).
3. Computes epoch-wise averages of a specified metric.
4. Plots the metric over epochs and saves the figure.

Requires:
    - utils.plotting.check_training_utils (load_json_lines, compute_epoch_average, plot_metric_over_epochs)
    - utils.configs.load_config
"""

import os
import sys
#import pandas as pd 

# Ensure custom utilities are in path
sys.path.append("/home/Daniele/codes/VISSL_postprocessing/")
from utils.plotting.check_training_utils import (
    load_json_lines,
    compute_epoch_average,
    plot_metric_over_epochs,
)
from utils.configs import load_config


def main(config_path: str = "config.yaml"):
    """
    Plot average loss over epochs for all runs defined in the config.

    Args:
        config_path (str): Path to YAML configuration file.
                           Default is "config.yaml".
    """
    # === LOAD CONFIG ===
    config = load_config(config_path)
    run_names = config["experiment"]["run_names"]
    base_path = config["experiment"]["base_path"]
    output_root = config["experiment"]["path_out"]
    parameter = config["experiment"].get("parameter", "loss")  # default to 'loss'

    # === PROCESS EACH RUN ===
    for run_name in run_names:
        print(f"Processing run: {run_name}")

        file_path = os.path.join(base_path, f'runs/{run_name}', "checkpoints", "stdout.json")
        output_dir = os.path.join(output_root, run_name)
        os.makedirs(output_dir, exist_ok=True)

        # === LOAD DATA ===
        data = load_json_lines(file_path)

        # === COMPUTE AVERAGE PER EPOCH ===
        epochs, avg_metric = compute_epoch_average(data, parameter)

        # === PLOT ===
        plot_metric_over_epochs(
            epochs,
            avg_metric,
            label=parameter.capitalize(),
            color="red",
            output_file=os.path.join(output_dir, f"{parameter}_avg_per_epoch.png"),
        )
        print(f"Saved plot to {output_dir}")


if __name__ == "__main__":
    main('/home/Daniele/codes/VISSL_postprocessing/configs/process_run_config.yaml')

