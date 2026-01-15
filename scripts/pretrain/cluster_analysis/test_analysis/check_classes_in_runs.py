import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc

# ================================================
# LOAD DATA
# ================================================
run_name_conv = "dcv2_resnet_k8_ir108_100x100_2013-2020_1xrandomcrops_1xtimestamp_cma_nc_convective"
run_name_norm = "dcv2_resnet_k7_ir108_100x100_2013-2020_1xrandomcrops_1xtimestamp_cma_nc"

convective_path = f"/data1/fig/{run_name_conv}/test/"
output_test_csv = os.path.join(convective_path, f"features_train_test_{run_name_conv}.csv")

normal_path = f"/data1/fig/{run_name_norm}/epoch_800/all/"
normal_csv = os.path.join(normal_path, f"crop_list_{run_name_norm}_all_140371.csv")

df_conv = pd.read_csv(output_test_csv, low_memory=False)
df_norm = pd.read_csv(normal_csv, low_memory=False)

print("Convective columns:", df_conv.columns.tolist())
print("Normal columns:", df_norm.columns.tolist())

# Keep only TRAIN in the convective run
df_conv_train = df_conv[df_conv["vector_type"] == "TRAIN"].copy()

#remove invalid labels (-100) in both dataframes
df_conv_train = df_conv_train[df_conv_train["label"] != -100]
df_norm = df_norm[df_norm["label"] != -100]

# ================================================
# PREPROCESS FILE NAMES
# ================================================
def extract_filename(path):
    return os.path.basename(path).strip()

df_conv_train["file"] = df_conv_train["path"].apply(extract_filename)
df_norm["file"] = df_norm["path"].apply(extract_filename)

# Check keys
print("Unique TRAIN convective files:", df_conv_train["file"].nunique())
print("Unique normal-run files:", df_norm["file"].nunique())


### Filter df_nrom to only keep wors that are also matcing the convective run
common_files = set(df_conv_train["file"]).intersection(set(df_norm["file"]))
df_norm_filtered = df_norm[df_norm["file"].isin(common_files)].copy()
print("Filtered normal-run files:", df_norm_filtered["file"].nunique())


df_merge = df_conv_train.merge(
    df_norm_filtered[["file", "label"]].rename(columns={"label": "label_normal"}),
    on="file",
    how="inner",
)


# -----------------------------------------
# BUILD TRANSITION MATRIX (counts)
# -----------------------------------------
all_conv_labels = sorted(df_conv_train["label"].dropna().unique())
all_norm_labels = sorted(df_norm_filtered["label"].dropna().unique())

matrix = pd.DataFrame(
    0,
    index=all_conv_labels,
    columns=all_norm_labels
)

for _, row in df_merge.iterrows():
    conv_label = row["label"]
    norm_label = row["label_normal"]
    if pd.notna(conv_label) and pd.notna(norm_label):
        matrix.loc[conv_label, norm_label] += 1

print("\n=== LABEL TRANSITION MATRIX (COUNTS) ===")
print(matrix)

# -----------------------------------------
# NORMALIZE ROWS (each row sums to 1)
# -----------------------------------------
matrix_norm = matrix.div(matrix.sum(axis=1), axis=0).fillna(0)

print("\n=== LABEL TRANSITION MATRIX (ROW-NORMALIZED) ===")
print(matrix_norm)

# -----------------------------------------
# OVERALL RELATIVE FREQUENCY ACROSS ALL CONVECTIVE SAMPLES
# -----------------------------------------
overall_counts = matrix.sum(axis=0)                  # sum over convective labels → counts per normal label
overall_relative = overall_counts / overall_counts.sum()

plt.figure(figsize=(5, 3))
overall_relative.plot(
    kind="bar",
    color="steelblue",
    edgecolor="black"
)
plt.ylabel("Relative Frequency", fontsize=14)
plt.xlabel("Normal Run Label", fontsize=14)
#rotate x axis labels 
plt.xticks(rotation=45, ha="right", fontsize=14)
plt.yticks(fontsize=14)
plt.title("Overall Normal-run Label Distribution \n of Convective Samples",
          fontsize=14, fontweight='bold')
plt.tight_layout()

outfile_overall = os.path.join(convective_path, "overall_label_distribution.png")
plt.savefig(outfile_overall, dpi=300, bbox_inches="tight", transparent=True)
plt.close()

print("\nSaved overall distribution bar plot to:", outfile_overall)



# -----------------------------------------
# PLOT NORMALIZED MATRIX AS HEATMAP
# -----------------------------------------
plt.figure(figsize=(5, 4))
plt.imshow(matrix_norm, cmap=cmc.lapaz, vmin=0, vmax=1)

cb = plt.colorbar(label="Relative Frequency")
cb.ax.tick_params(labelsize=14)
cb.ax.title.set_fontsize(14)

plt.xticks(range(len(all_norm_labels)), all_norm_labels, rotation=45, ha="right", fontsize=14)
plt.yticks(range(len(all_conv_labels)), all_conv_labels, fontsize=14)

plt.xlabel("Normal Run Label", fontsize=14)
plt.ylabel("Convective Run Label", fontsize=14)
plt.title("Label Transition Matrix", 
          fontsize=14, fontweight='bold')

plt.tight_layout()
outfile = os.path.join(convective_path, "label_transition_matrix_normalized.png")
plt.savefig(outfile, dpi=300, bbox_inches="tight", transparent=True)
plt.close()

print("\nSaved heatmap to:", outfile)

