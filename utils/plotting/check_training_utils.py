"""
Utility functions for loading data and plotting metrics.

Provides:
- JSON data loading for training logs.
- Epoch-wise averaging.
- Generic plotting functions for single or multiple metrics/datasets.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os

def load_json_lines(file_path):
    """
    Load JSON objects from a file, line by line, ignoring errors.

    :param file_path: Path to JSON lines file
    :return: List of dicts
    """
    data = []
    with open(file_path, "r") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data

def compute_epoch_average(data, parameter):
    """
    Compute average of a parameter per epoch.

    :param data: List of dicts with 'ep' and parameter values
    :param parameter: Parameter key to average
    :return: (epochs, averaged_values)
    """
    epoch_dict = {}
    for entry in data:
        epoch_dict.setdefault(entry['ep'], []).append(entry[parameter])
    epochs = sorted(epoch_dict.keys())
    avg_values = [np.mean(epoch_dict[e]) for e in epochs]
    return epochs, avg_values

def plot_metric_over_epochs(epochs, values, label, color, output_file, xlabel='Epoch', ylabel=None):
    """
    Generic single-parameter plot.

    :param epochs: List of epochs
    :param values: Values corresponding to epochs
    :param label: Legend label
    :param color: Line color
    :param output_file: Path to save the figure
    :param xlabel: X-axis label
    :param ylabel: Y-axis label
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, values, label=label, color=color, linewidth=2)
    ax.set_xlabel(xlabel, fontsize=14) 
    ax.set_ylabel(ylabel or label, fontsize=14)
    ax.set_title(f'{label} over {xlabel}s', fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True)
    #ax.legend()
    fig.savefig(output_file, bbox_inches='tight')
    plt.close()

def plot_multiple_metrics(metrics_dict, output_file, xlabel='Epoch'):
    """
    Plot multiple metrics on separate y-axes with offset.

    :param metrics_dict: Ordered dict of {metric_name: (values, color)}
    :param output_file: Path to save figure
    :param xlabel: Label for x-axis
    """
    fig, ax_primary = plt.subplots(figsize=(11, 5))
    axes = [ax_primary]

    for i, (metric_name, (values, color)) in enumerate(metrics_dict.items()):
        if i == 0:
            ax = ax_primary
            ax.set_ylabel(metric_name, color=color, fontsize=14)
            ax.plot(values, label=metric_name, color=color, marker='o')
            ax.tick_params(axis='y', labelcolor=color)
        else:
            ax = ax_primary.twinx()
            ax.spines["right"].set_position(("axes", 1 + 0.12 * i))
            ax.set_frame_on(True)
            ax.patch.set_visible(False)
            ax.set_ylabel(metric_name, color=color, fontsize=14)
            ax.plot(values, label=metric_name, color=color, marker='o')
            ax.tick_params(axis='y', labelcolor=color)
        axes.append(ax)

    # Collect legends
    all_lines, all_labels = [], []
    for ax in axes:
        lines, labels = ax.get_legend_handles_labels()
        all_lines += lines
        all_labels += labels
    ax_primary.legend(all_lines, all_labels, loc='upper left', fontsize=10)

    ax_primary.set_xlabel(xlabel, fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(output_file, dpi=300)
    plt.close()
