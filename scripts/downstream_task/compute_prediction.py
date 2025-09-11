import numpy as np
import torch
import os

# Paths
features_base_path = "/data1/runs/supervised_ir108-cm_75x75_5frames_12k_nc_r2dplus1/features/epoch_50/"
test_features_path = os.path.join(features_base_path, "rank0_chunk0_test_heads_features.npy")
train_features_path = os.path.join(features_base_path, "rank0_chunk0_train_heads_features.npy")

# Load features (logits)
test_logits = torch.from_numpy(np.load(test_features_path))
train_logits = torch.from_numpy(np.load(train_features_path))

# Compute predictions
test_preds = torch.argmax(test_logits, dim=1)
train_preds = torch.argmax(train_logits, dim=1)
print("Test predictions shape:", test_preds.shape)
print("Train predictions shape:", train_preds.shape)
print("First 5 test predictions:", test_preds[:5])
print("First 5 train predictions:", train_preds[:5])

# Save predictions
test_save_path = os.path.join(os.path.dirname(test_features_path), "rank0_chunk0_test_heads_preds.npy")
train_save_path = os.path.join(os.path.dirname(train_features_path), "rank0_chunk0_train_heads_preds.npy")
np.save(test_save_path, test_preds.numpy())
np.save(train_save_path, train_preds.numpy())
