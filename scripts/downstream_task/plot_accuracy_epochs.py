"""
Script to analyze and visualize training and testing performance
for hail classification runs.

This script:
1. Loads accuracy metrics from `metrics.json` (train & test accuracies).
2. Optionally loads loss values from `stdout.json` and computes mean loss per epoch.
3. Produces a combined plot of train/test accuracy and mean loss per epoch.

Outputs:
- A PNG figure saved to the configured output folder.

Configuration parameters are defined at the beginning of the script.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

# ====================
# CONFIGURATION
# ====================
PLOT_LOSS = True
RUN_NAME = "supervised_ir108-cm_75x75_5frames_12k_nc_r2dplus1"

FOLDER_PATH = f"/data1/runs/{RUN_NAME}"
METRICS_FILE = f"{FOLDER_PATH}/checkpoints/metrics.json"
LOSS_FILE = f"{FOLDER_PATH}/checkpoints/stdout.json"  # line-delimited JSON for losses

OUTPUT_FOLDER = f"/data1/fig/{RUN_NAME}/"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ====================
# FUNCTIONS
# ====================
def load_metrics(metrics_file):
    """Load metrics.json file into a list of dictionaries."""
    metrics = []
    with open(metrics_file, "r") as f:
        for line in f:
            if line.strip():
                metrics.append(json.loads(line))
    return metrics


def extract_accuracies(metrics):
    """
    Extract train/test accuracy values and corresponding epochs.

    Returns:
        epochs_train (list[int])
        train_acc (list[float])
        epochs_test (list[int])
        test_acc (list[float])
    """
    epochs_train, train_acc = [], []
    epochs_test, test_acc = [], []

    for m in metrics:
        if "train_accuracy_list_meter" in m:
            train_acc.append(m["train_accuracy_list_meter"]["top_1"]["0"])
            epochs_train.append(m["train_phase_idx"])

        if "test_accuracy_list_meter" in m:
            test_acc.append(m["test_accuracy_list_meter"]["top_1"]["0"])
            epochs_test.append(m["train_phase_idx"])

    return epochs_train, train_acc, epochs_test, test_acc


def load_losses(loss_file):
    """
    Load line-delimited JSON loss logs and compute mean loss per epoch.

    Returns:
        epochs_loss (list[int])
        mean_loss (list[float])
    """
    loss_per_epoch = defaultdict(list)
    with open(loss_file, "r") as f:
        for line in f:
            if line.strip():
                l = json.loads(line)
                loss_per_epoch[l["ep"]].append(l["loss"])

    epochs_loss = sorted(loss_per_epoch.keys())
    mean_loss = [np.mean(loss_per_epoch[ep]) for ep in epochs_loss]

    return epochs_loss, mean_loss


def plot_metrics(epochs_train, train_acc, epochs_test, test_acc,
                 epochs_loss=None, mean_loss=None, plot_loss=True):
    """
    Plot train/test accuracies and optional mean loss per epoch.

    Saves the plot as PNG in OUTPUT_FOLDER.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Train/test accuracy
    ax1.plot(epochs_train, train_acc, marker='o', linestyle='-', color='blue',
             label='Train Top-1 Accuracy')
    ax1.plot(epochs_test, test_acc, marker='s', linestyle='-', color='red',
             label='Test Top-1 Accuracy')
    ax1.set_xlabel("Epoch", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Accuracy (%)", fontsize=14, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.tick_params(axis='both', labelsize=12)

    # Loss (secondary axis)
    if plot_loss and epochs_loss and mean_loss:
        ax2 = ax1.twinx()
        ax2.plot(epochs_loss, mean_loss, marker='^', linestyle='--',
                 color='green', label='Mean Loss')
        ax2.set_ylabel("Loss", fontsize=14, fontweight="bold")
        ax2.tick_params(axis='y', labelsize=12)

    # Legend
    lines, labels = ax1.get_legend_handles_labels()
    if plot_loss and epochs_loss and mean_loss:
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines += lines2
        labels += labels2
    ax1.legend(lines, labels, loc='center left', bbox_to_anchor=(1.15, 0.5), fontsize=12)

    plt.title("Hail Classification: Accuracy & Loss", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FOLDER}/accuracy_loss_plot_hail_classif.png", bbox_inches="tight")
    plt.show()


# ====================
# MAIN
# ====================
def main():
    metrics = load_metrics(METRICS_FILE)
    epochs_train, train_acc, epochs_test, test_acc = extract_accuracies(metrics)

    if PLOT_LOSS:
        epochs_loss, mean_loss = load_losses(LOSS_FILE)
    else:
        epochs_loss, mean_loss = None, None

    plot_metrics(epochs_train, train_acc, epochs_test, test_acc,
                 epochs_loss, mean_loss, plot_loss=PLOT_LOSS)


if __name__ == "__main__":
    main()
