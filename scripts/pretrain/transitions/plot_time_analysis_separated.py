import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ==================================================
# IMPORT PROJECT UTILITIES
# ==================================================
sys.path.append("/home/Daniele/codes/VISSL_postprocessing/scripts/pretrain/")
from cluster_analysis.test_analysis.utils_func import (
    filter_rows_in_event_window,
    stratifiy_by_latitude,
    build_event_groups
)

# from transitions.plot_utils import (
#     update_df_with_bt_stats,
#     plot_event_density_map,
#     plot_bt_boxplot_by_event,
#     plot_orography_map
# )

# === CONFIG ===
RUN_NAME = 'dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc'
EVENT_TYPES = ["ALL", "RAIN_ALL", "RAIN_PERC", "HAIL_ALL", "HAIL_PERC"]
BASE_DIR = f"/data1/fig/{RUN_NAME}/test"
LAT_DIVISION = 47
PERCENTILE = 50  # Percentile for intensity thresholding

def filter_rows_in_event_window(df):
    """
    Keep only rows where:
        start_time <= datetime <= end_time
    Assumes all datetimes are in UTC.
    """

    # Work on a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Convert to datetime if not already
    for col in ["datetime", "start_time", "end_time"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Ensure all are tz-aware (UTC) to avoid comparison issues
    for col in ["datetime", "start_time", "end_time"]:
        if df[col].dt.tz is None:
            df[col] = df[col].dt.tz_localize("UTC")

    # Drop rows with missing timestamps
    df = df.dropna(subset=["datetime", "start_time", "end_time"])

    # Keep only rows where datetime is inside the event window
    mask_in_window = (df["datetime"] >= df["start_time"]) & (df["datetime"] <= df["end_time"])
    df_filtered = df[mask_in_window]

    return df_filtered.reset_index(drop=True)


def build_event_groups(df, percentile: int = 50, region: str = None, LAT_DIVISION: float = LAT_DIVISION):
    """
    Creates boolean masks for:
      - ALL
      - RAIN_ALL
      - RAIN_10th (top 10% rain intensity)
      - HAIL_ALL
      - HAIL_10th (top 10% hail intensity)

    vector_type is used to detect event type.
    """

    # Filer all rows
    if region is not None:
        if region == "NORTH":
            df = df[df["latitude"] >= LAT_DIVISION]
        elif region == "SOUTH":
            df = df[df["latitude"] < LAT_DIVISION]
        else:
            raise ValueError(f"Unknown region: {region}")

    # --------------------------------------
    # 1. Identify rain vs hail using vector_type
    # --------------------------------------
    rain_mask = df["vector_type"].str.contains("PRECIP", case=False, na=False)
    hail_mask = df["vector_type"].str.contains("HAIL", case=False, na=False)
    print(f"Rain mask sum: {rain_mask.sum()}, Hail mask sum: {hail_mask.sum()}")
    #print the overlap (should be zero)
    overlap = (rain_mask & hail_mask).sum()
    print(f"Overlap between rain and hail masks: {overlap}")

    # --------------------------------------
    # 2. Compute 90th percentiles separately
    # --------------------------------------
    # Rain
    rain_vals = df.loc[rain_mask, "max_intensity"].dropna()
    rain_p = np.percentile(rain_vals, percentile) if len(rain_vals) > 0 else np.nan

    # Hail
    hail_vals = df.loc[hail_mask, "max_intensity"].dropna()
    hail_p = np.percentile(hail_vals, percentile) if len(hail_vals) > 0 else np.nan
    print(f"Rain {percentile}th percentile (mm): {rain_p}")
    print(f"Hail {percentile}th percentile (cm): {hail_p}")

    # --------------------------------------
    # 3. Group masks returned as dictionary
    # --------------------------------------

    return {
        "ALL": rain_mask | hail_mask,
        "RAIN_ALL": rain_mask,
        "RAIN_PERC": rain_mask & (df["max_intensity"] >= rain_p),
        "HAIL_ALL": hail_mask,
        "HAIL_PERC": hail_mask & (df["max_intensity"] >= hail_p)
    }

def load_case(event):
    """Load and preprocess CSV for a given event."""
    path = f"{BASE_DIR}/features_train_test_{RUN_NAME}.csv"
    df = pd.read_csv(path, low_memory=False)
    df = df[build_event_groups(df, percentile=PERCENTILE, region=REGION, LAT_DIVISION=LAT_DIVISION)[event]]
    df['datetime'] = pd.to_datetime(df['datetime'], format="%Y-%m-%d %H:%M:%S", errors='coerce')
    df = df.dropna(subset=['datetime'])
    df = df.sort_values('datetime')
    df = filter_rows_in_event_window(df)
    
    return df


def compute_diurnal_frequency(df):
    """Compute relative frequency per hour per label."""
    df['hour'] = df['datetime'].dt.hour
    counts = (
        df.groupby(["hour", "label"])
        .size()
        .reset_index(name="count")
    )
    counts["relative_freq"] = counts.groupby("hour")["count"].transform(lambda x: x / x.sum())
    return counts.pivot(index='label', columns='hour', values='relative_freq').fillna(0)



def plot_diurnal_heatmap(freq_matrix, event, out_path):
    """Plot diurnal label occurrence heatmap."""
    plt.figure(figsize=(9, 4))
    ax = sns.heatmap(
        freq_matrix,
        cmap='viridis',
        cbar_kws={'label': 'Relative Frequency'},
        linewidths=0.5,
        vmax=0.5
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label("Relative Frequency", fontsize=16, fontweight="bold")

    plt.title(f"Daily Class Occurrence ({event})", fontsize=16, fontweight='bold')
    plt.xlabel("Hour (UTC)", fontsize=14, fontweight='bold')
    plt.ylabel("Label", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, fontsize=14)
    plt.yticks(rotation=0, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved diurnal plot: {out_path}")



def compute_transitions_and_persistence(df):
    """Compute transition probabilities and persistence per label."""
    labels = df['label'].tolist()

    # Compute blocks of consecutive identical labels
    blocks = []
    prev_label = None
    block_len = 0
    for lbl in labels:
        if lbl == prev_label:
            block_len += 1
        else:
            if prev_label is not None:
                blocks.append((prev_label, block_len))
            block_len = 1
            prev_label = lbl
    blocks.append((prev_label, block_len))

    blocks = pd.DataFrame(blocks, columns=['label', 'length'])
    #convert lenght in duration in minutes
    blocks['duration_min'] = blocks['length'] * 15  # 15-min timesteps → minutes
    #blocks['duration_h'] = blocks['length'] * 0.25  # 15-min timesteps → hours

    # Persistence = mean duration per label (round to integer) 
    persistence = blocks.groupby('label')['duration_min'].mean().round().astype(int).to_dict()

    # Compute transitions between blocks
    from_labels = blocks['label'][:-1].values
    to_labels = blocks['label'][1:].values
    trans_df = pd.DataFrame({'from': from_labels, 'to': to_labels})
    transition_counts = pd.crosstab(trans_df['from'], trans_df['to'])
    transition_probs = transition_counts.div(transition_counts.sum(axis=1), axis=0).fillna(0)

    # Insert persistence on diagonal
    all_labels = sorted(set(df['label'].unique()))
    transition_matrix = pd.DataFrame(0, index=all_labels, columns=all_labels, dtype=float)

    for i in all_labels:
        for j in all_labels:
            if i == j:
                transition_matrix.loc[i, j] = persistence.get(i, 0)
            else:
                transition_matrix.loc[i, j] = transition_probs.loc[i, j] if i in transition_probs.index and j in transition_probs.columns else 0

    return transition_matrix, persistence



def plot_transition_matrix(matrix, persistence, event, out_path):
    """Plot transition matrix with persistence values on diagonal."""
    off_diag = matrix.copy()
    np.fill_diagonal(off_diag.values, np.nan)

    plt.subplots(figsize=(7, 6))
    #2 decimal places only for off-diagonal
    ax =sns.heatmap(off_diag, annot=True, fmt=".2f", cmap="Blues",
                cbar_kws={'label': 'Transition Probability'}, vmax=0.4)
    #increase font size of colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label("Transition Probability", fontsize=16, fontweight="bold")
    # Overlay persistence (diagonal) with another color scale
    for i, label in enumerate(matrix.index):
        val = matrix.loc[label, label]
        #integer values in red bold font
        ax.text(i + 0.5, i + 0.5, f"{val.astype(int)}", ha='center', va='center',
                color='red', fontsize=14, fontweight='bold')

    plt.title(f"Class Transition Matrix ({event})", fontsize=16, fontweight='bold')
    plt.xlabel("To Class", fontsize=15)
    plt.ylabel("From Class", fontsize=15)
    plt.xticks(rotation=45, fontsize=15)
    plt.yticks(rotation=0, fontsize=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved transition matrix: {out_path}")

    # Save persistence values
    with open(out_path.replace(".png", "_persistence.txt"), "w") as f:
        f.write("Average class persistence (hours):\n")
        for k, v in persistence.items():
            f.write(f"Label {k}: {v:.2f} h\n")
    print(f"🕒 Saved persistence summary: {out_path.replace('.png', '_persistence.txt')}")


if __name__ == "__main__":
    # === MAIN LOOP ===
    all_data = []

    #divide by region   
    df_north, df_south = stratifiy_by_latitude(df, LAT_DIVISION)

    REGIONS = {
        "North": df_north,
        "South": df_south
    }

    for region, df_region in REGIONS.items():
        for event in EVENT_TYPES:
            print(f"\n🔍 Processing {event} for region {region}...")
            df_event = load_case(event)
            all_data.append(df_event)

        # 1️⃣ Diurnal frequency
        freq_matrix = compute_diurnal_frequency(df_event)
        plot_diurnal_heatmap(freq_matrix, event, f"{BASE_DIR}/case_study_labels_over_time_{event}_{region}.png")
        # 2️⃣ Transitions + persistence
        matrix, persistence = compute_transitions_and_persistence(df_event)
        plot_transition_matrix(matrix, persistence, event, f"{BASE_DIR}/label_transition_matrix_{event}_{region}.png")

    # # === COMBINED "ALL" CASE ===
    # df_all = pd.concat(all_data, ignore_index=True)
    # print("\n🔍 Processing combined case: ALL")

    # freq_matrix = compute_diurnal_frequency(df_all)
    # plot_diurnal_heatmap(freq_matrix, "ALL", f"{BASE_DIR}/case_study_labels_over_time_ALL.png")

    # matrix, persistence = compute_transitions_and_persistence(df_all)
    # plot_transition_matrix(matrix, persistence, "ALL", f"{BASE_DIR}/label_transition_matrix_ALL.png")

    # print("\n✅ All plots and persistence summaries generated successfully.")
