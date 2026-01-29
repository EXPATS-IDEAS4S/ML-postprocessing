import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
import matplotlib as mpl
import cmcrameri.cm as cmc
import numpy as np
from matplotlib.lines import Line2D
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

sys.path.append("/home/Daniele/codes/VISSL_postprocessing/")
from utils.plotting.class_colors import CLOUD_CLASS_INFO

# ================================================
# LOAD DATA
# ================================================
run_name = "dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc"
output_path = f"/data1/fig/{run_name}/epoch_800/test_traj"
output_test_csv = os.path.join(output_path, f"features_train_test_{run_name}_2nd_labels.csv")

plot_dir = os.path.join(output_path, "hypersphere_analysis")
os.makedirs(plot_dir, exist_ok=True)

df = pd.read_csv(output_test_csv, low_memory=False)
#print(df)#.columns.tolist())

LAT_DIVISION = 47
WITHOUT_EXTRAPOLATED = True

MIN_SAMPLES = 100  # Minimum samples to have in a neighborhood
SIM_THRESHOLD = 0.9 # Cosine similarity threshold for neighbors
#Skipped 31 test vectors due to insufficient neighbors

#dictionary with group names and colors
groups_dict = [
    {'name' :"ALL"},
    {'name': "PRECIP"},
    {'name': "HAIL" },
    {'name': "MIXED" }
]


cloud_items_ordered = sorted(
    CLOUD_CLASS_INFO.items(),
    key=lambda x: x[1]["order"]
)

labels_ordered = [lbl for lbl, _ in cloud_items_ordered]
short_labels = [info["short"] for _, info in cloud_items_ordered]
colors_ordered = [info["color"] for _, info in cloud_items_ordered]

labels_sorted = sorted(CLOUD_CLASS_INFO.keys())
#print(labels_sorted)

#extract filenmae in df (from path)
df['filename'] = df['path'].apply(lambda x: os.path.basename(x))


# feature columns
dim_cols = [c for c in df.columns if c.startswith("dim_")]

# split train / test
df_train = df[df["vector_type"] == "TRAIN"].copy()
df_test  = df[df["vector_type"] != "TRAIN"].copy()

#remove invalid rows (label -100) in train
df_train = df_train[df_train["label"] != -100].copy()

#add hour column
df_train['hour'] = pd.to_datetime(df_train['datetime']).dt.hour

# labels ordered (consistent with CLOUD_CLASS_INFO)
label_to_idx = {l: i for i, l in enumerate(labels_ordered)}

X_train = normalize(df_train[dim_cols].values)
X_test  = normalize(df_test[dim_cols].values)
#print(X_train, X_test)

y_train = df_train["label"].values

results = []
skipped_rows = []

for i in tqdm(range(len(df_test)), desc="Processing test vectors"):

    test_row = df_test.iloc[i]
    test_vec = X_test[i].reshape(1, -1)

    # cosine similarity with all training vectors
    sims = cosine_similarity(test_vec, X_train).ravel()

    # K = 200
    # idx = np.argsort(sims)[-K:]
    # neighbor_labels = y_train[idx]

    # select neighbors
    mask = sims >= SIM_THRESHOLD
    neighbor_labels = y_train[mask]

    if len(neighbor_labels) < MIN_SAMPLES:
        # not enough neighbors, skip
        skipped_rows.append(test_row["filename"])
        
    row_out = {
        "filename": test_row["filename"],
        'lat': test_row['lat'],
        'lon': test_row['lon'],
        'storm_type': test_row['storm_type'],
        'crop_type': test_row['crop_type'],
        "label": test_row["label"],
        "label_2nd": test_row.get("label_2nd", np.nan),
        "n_neighbors": int(mask.sum()),
        "mean_similarity": float(sims[mask].mean()) if mask.any() else np.nan,
        "std_similarity": float(sims[mask].std()) if mask.any() else np.nan,
    }

    if len(neighbor_labels) < MIN_SAMPLES:
        # not enough neighbors, fill with NaNs
        for lbl in labels_ordered:
            row_out[f"perc_label_{lbl}"] = np.nan
        row_out["dominant_label"] = np.nan
        row_out["dominance"] = np.nan
        results.append(row_out)
        continue
    
    # label composition
    for lbl in labels_ordered:
        row_out[f"perc_label_{lbl}"] = (
            np.mean(neighbor_labels == lbl) * 100.0
        )
    
    #check hour composition of the neighbors
    neighbor_hours = df_train.loc[mask, "hour"].values
    for hr in range(24):
        row_out[f"perc_hour_{hr}"] = (
            np.mean(neighbor_hours == hr) * 100.0
        )

    weights = sims[mask]
    for lbl in labels_ordered:
        row_out[f"wperc_label_{lbl}"] = (
            weights[neighbor_labels == lbl].sum() / weights.sum() * 100
        )

    row_out["dominant_label"] = max(
        labels_ordered,
        key=lambda l: row_out[f"perc_label_{l}"]
    )

    row_out["dominance"] = max(
        row_out[f"perc_label_{l}"] for l in labels_ordered
    )
    results.append(row_out)

#check how many rows were skipped
print(f"Skipped {len(skipped_rows)} test vectors due to insufficient neighbors.")
print(f"Skipped filenames: {skipped_rows}")

df_neighbors = pd.DataFrame(results)

out_csv = os.path.join(
    plot_dir,
    "test_vectors_local_label_composition.csv"
)
df_neighbors.to_csv(out_csv, index=False)

print(f"Saved: {out_csv}")

#809125