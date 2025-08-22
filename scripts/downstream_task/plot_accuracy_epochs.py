import json
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

# === CONFIG ===
PLOT_LOSS = True
run_name = "dcv2_ir108-cm_100x100_8frames_k9_70k_nc_r2dplus1"
folder_path = f"/data1/runs/{run_name}/hail_linear_classif"
metrics_file = f"{folder_path}/metrics.json"
loss_file = f"{folder_path}/stdout.json"  # line-delimited JSON for losses
output_folder = f"/data1/fig/{run_name}/epoch_800/downstream_task/"
os.makedirs(output_folder, exist_ok=True)

# === READ METRICS.JSON ===
metrics = []
with open(metrics_file, "r") as f:
    for line in f:
        if line.strip():
            metrics.append(json.loads(line))

# Extract accuracies
epochs_train = []
epochs_test = []
train_acc = []
test_acc = []

for m in metrics:
    # Train accuracy
    if "train_accuracy_list_meter" in m:
        train_acc.append(m["train_accuracy_list_meter"]["top_1"]["0"])
        epochs_train.append(m["train_phase_idx"])

    # Test accuracy
    if "test_accuracy_list_meter" in m:
        test_acc.append(m["test_accuracy_list_meter"]["top_1"]["0"])
        epochs_test.append(m["train_phase_idx"])

# === READ LOSS.JSON AND COMPUTE MEAN LOSS PER EPOCH ===
if PLOT_LOSS:
    loss_per_epoch = defaultdict(list)
    with open(loss_file, "r") as f:
        for line in f:
            if line.strip():
                l = json.loads(line)
                loss_per_epoch[l["ep"]].append(l["loss"])
    
    epochs_loss = sorted(loss_per_epoch.keys())
    mean_loss = [np.mean(loss_per_epoch[ep]) for ep in epochs_loss]

# === PLOT ===
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot train/test accuracy
ax1.plot(epochs_train, train_acc, marker='o', linestyle='-', color='blue', label='Train Top-1 Accuracy')
ax1.plot(epochs_test, test_acc, marker='s', linestyle='-', color='red', label='Test Top-1 Accuracy')
ax1.set_xlabel("Epoch", fontsize=14, fontweight="bold")
ax1.set_ylabel("Accuracy (%)", fontsize=14, fontweight="bold")
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.tick_params(axis='both', labelsize=12)

# Plot loss on secondary y-axis
if PLOT_LOSS:
    ax2 = ax1.twinx()
    ax2.plot(epochs_loss, mean_loss, marker='^', linestyle='--', color='green', label='Mean Loss')
    ax2.set_ylabel("Loss", fontsize=14, fontweight="bold")
    ax2.tick_params(axis='y', labelsize=12)

# Combine legends
lines, labels = ax1.get_legend_handles_labels()
if PLOT_LOSS:
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines += lines2
    labels += labels2

# Place legend outside to the right
ax1.legend(lines, labels, loc='center left', bbox_to_anchor=(1.15, 0.5), fontsize=12)

plt.title("Hail Classification: Accuracy & Loss", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{output_folder}/accuracy_loss_plot_hail_classif.png", bbox_inches="tight")
plt.show()
