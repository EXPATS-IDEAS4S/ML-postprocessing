#!/usr/bin/env python3
import os
import re
from collections import defaultdict
from glob import glob
import pandas as pd

# ==== CONFIG ====
run_name = 'dcv2_ir108-cm_100x100_8frames_k9_70k_nc_r2dplus1'
crops_name = 'clips_ir108_100x100_8frames_2013-2020'  # Name of the crops
random_state = 3   # all visualization were made with random state 3
sampling_type = 'all'
reduction_method = 'tsne'  # Options: 'tsne', 'isomap'
epoch = 800  # Epoch number for the run
file_extension = 'png'  # Image file extension
#substitute_path = True
variable_type = 'IR_108_cm'  # 'WV_062-IR_108', etc.
VIDEO = True
N_FRAMES = 8
N_SAMPLES = None   # set e.g. 1000 to sample N videos, or None to keep all

output_path = f'/data1/fig/{run_name}/epoch_{epoch}/{sampling_type}/'

# ==== LOAD FILES ====
# List of the image crops
image_crops_path = f'/data1/crops/{crops_name}/img/{variable_type}/1/'
list_image_crops = sorted(glob(image_crops_path + '*.' + file_extension))

# Open CSV file with already labels and dim red features
csv_path = f'{output_path}merged_tsne_crop_list_{run_name}_{sampling_type}_{random_state}_epoch_{epoch}.csv'
df_labels = pd.read_csv(csv_path)

# drop any color columns if exist
df_labels = df_labels.loc[:, ~df_labels.columns.str.contains('^color')]
print("Columns:", df_labels.columns)
print("Original rows:", len(df_labels))

# ==== OPTIONAL SAMPLING ====
if N_SAMPLES is not None:
    df_labels = df_labels.sample(n=N_SAMPLES, random_state=random_state)
    print(f"Sampled rows: {len(df_labels)}")

# ==== INDEX FRAMES ====
frame_dict = defaultdict(dict)
pattern = re.compile(r"(MSG_timeseries_\d{4}-\d{2}-\d{2}_\d{4}_crop\d+)_t(\d+)_")

for p in list_image_crops:
    m = pattern.search(os.path.basename(p))
    if m:
        video_stem, frame_idx = m.group(1), int(m.group(2))
        frame_dict[video_stem][frame_idx] = p


# ==== EXPAND DF ====
#if substitute_path:
if VIDEO:
    expanded_rows = []
    for _, row in df_labels.iterrows():
        # Get stem from original .nc filename in path column
        video_stem = os.path.splitext(os.path.basename(row['path']))[0]

        if video_stem not in frame_dict:
            print(f"⚠️ No frames found for {video_stem}")
            continue

        frames = frame_dict[video_stem]

        # Ensure all frames exist
        if not all(idx in frames for idx in range(N_FRAMES)):
            print(f"⚠️ Missing frames for {video_stem}")
            continue

        # Add one row per frame
        for frame_idx in range(N_FRAMES):
            new_row = row.copy()
            new_row['path'] = frames[frame_idx]
            new_row['frame_idx'] = frame_idx
            expanded_rows.append(new_row)

    df_labels_expanded = pd.DataFrame(expanded_rows)

else:
    df_labels['path'] = df_labels['crop_index'].apply(
        lambda x: list_image_crops[int(x)]
    )
    df_labels_expanded = df_labels.copy()
#else:
#    df_labels_expanded = df_labels.copy()

print("Expanded rows:", len(df_labels_expanded))

# ==== SAVE RESULT ====
expanded_csv_path = os.path.join(
    os.path.dirname(csv_path),
    f"merged_tsne_crop_list_{run_name}_{sampling_type}_{random_state}_epoch_{epoch}_expanded.csv"
)
df_labels_expanded.to_csv(expanded_csv_path, index=False)
print(f"✅ Expanded dataframe saved to: {expanded_csv_path}")
