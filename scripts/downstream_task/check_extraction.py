import numpy as np

# Paths to your files
base_path = "/data1/runs/supervised_ir108-cm_75x75_5frames_12k_nc_r2dplus1/features/epoch_50/"
features_file = base_path + "rank0_chunk0_test_heads_features.npy"
inds_file     = base_path + "rank0_chunk0_test_heads_inds.npy"
targets_file  = base_path + "rank0_chunk0_test_heads_targets.npy"

# Load arrays
features = np.load(features_file)
inds     = np.load(inds_file)
targets  = np.load(targets_file)

print("Features shape:", features.shape)   # typically [N, D]
print("Indices shape:", inds.shape)        # typically [N]
print("Targets shape:", targets.shape)     # typically [N]

# Example: inspect first few entries
print("\nFirst 5 indices:", inds[:5])
print("First 5 targets:", targets[:5])
print("First feature vector:", features[0][:10])  # print first 10 dims

#count of each class in target
unique, counts = np.unique(targets, return_counts=True)
class_counts = dict(zip(unique, counts))
print("\nClass distribution in targets:", class_counts)
