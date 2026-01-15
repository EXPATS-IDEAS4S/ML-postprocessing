import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from glob import glob
from sklearn.metrics.pairwise import cosine_distances

# ================================================
# LOAD DATA
# ================================================
run_name = "dcv2_resnet_k8_ir108_100x100_2013-2020_1xrandomcrops_1xtimestamp_cma_nc_convective"
output_path = f"/data1/fig/{run_name}/test/"
output_test_csv = os.path.join(output_path, f"features_train_test_{run_name}.csv")

df = pd.read_csv(output_test_csv, low_memory=False)
print(df.columns.tolist())

#remove rows with label -100 (invalid)
df = df[df["label"] != -100]

# feature columns
dim_cols = [c for c in df.columns if c.startswith("dim_")]

# ================================================
# EVENT TYPE DEFINITIONS
# ================================================
# dictionary describes intensity field + units for each event category
EVENT_TYPES = {
    "PRECIP": {
        "intensity": "max_intensity", #mean or max intensity field
        "unit": "mm",
        "long_name": "Precipitation Amount"
    },
    "HAIL": {
        "intensity": "max_intensity", #mean or max intensity field
        "unit": "cm",
        "long_name": "Maximum Hail Size"
    }
}

# ensure intensity column is numeric for all event types
df["max_intensity"] = pd.to_numeric(df["max_intensity"], errors="coerce")


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



# ================================================
# GROUP BUILDER
# ================================================
def build_event_groups(df):
    """
    Creates boolean masks for:
      - ALL
      - RAIN_ALL
      - RAIN_10th (top 10% rain intensity)
      - HAIL_ALL
      - HAIL_10th (top 10% hail intensity)

    vector_type is used to detect event type.
    """

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
    rain_p90 = np.percentile(rain_vals, 90) if len(rain_vals) > 0 else np.nan

    # Hail
    hail_vals = df.loc[hail_mask, "max_intensity"].dropna()
    hail_p90 = np.percentile(hail_vals, 90) if len(hail_vals) > 0 else np.nan

    print(f"Rain 90th percentile (mm): {rain_p90}")
    print(f"Hail 90th percentile (cm): {hail_p90}")

    # --------------------------------------
    # 3. Group masks returned as dictionary
    # --------------------------------------
    return {
        "ALL": rain_mask | hail_mask,
        "RAIN_ALL": rain_mask,
        "RAIN_10th": rain_mask & (df["max_intensity"] >= rain_p90),
        "HAIL_ALL": hail_mask,
        "HAIL_10th": hail_mask & (df["max_intensity"] >= hail_p90)
    }


# ================================================
# BUILD GROUP MASKS
# ================================================
groups = build_event_groups(df)

df_all     = df[groups["ALL"]]
df_rain_top10 = df[groups["RAIN_10th"]]
df_hail_top10 = df[groups["HAIL_10th"]]
df_rain_all  = df[groups["RAIN_ALL"]]
df_hail_all  = df[groups["HAIL_ALL"]]

# print sample counts considering single events (unique datetime and lat_centre and lon_centre)

print("Samples per group:")
df["date"] = pd.to_datetime(df["datetime"]).dt.date
for name, mask in groups.items():
    df_g = df[mask]

    # unique events per day+lat+lon
    n_events = df_g.groupby(["date", "lat_centre", "lon_centre", "vector_type"]).ngroup().nunique()

    print(f"{name:<12}: {n_events}")

# -----------------------------
# 2️⃣ PLOT GROUP COUNTS & LABEL COUNTS
# -----------------------------
plot_dir = os.path.join(output_path, "analysis_plots")
os.makedirs(plot_dir, exist_ok=True)

# ----------------------------------------------------------
# Combined bar plot: counts per label for all groups
# ----------------------------------------------------------

# list group names in a meaningful order
group_order = ["ALL", "RAIN_ALL", "RAIN_10th", "HAIL_ALL", "HAIL_10th"]

# assign a color for each group
colors = {
    "ALL": "#4E79A7",
    "RAIN_ALL": "#59A14F",
    "RAIN_10th": "#8CD17D",
    "HAIL_ALL": "#F28E2B",
    "HAIL_10th": "#E15759",
}

# compute counts per label for each group
label_values = sorted(df["label"].dropna().unique())
label_indices = np.arange(len(label_values))

bar_width = 0.15

plt.figure(figsize=(8, 4))

for i, gname in enumerate(group_order):
    mask = groups[gname]
    gdf = df[mask]
    #get only times inside event window
    gdf = filter_rows_in_event_window(gdf)

    counts = gdf["label"].value_counts().reindex(label_values, fill_value=0)

    # offset each group's bars
    plt.bar(
        label_indices + i * bar_width,
        counts.values, 
        width=bar_width,
        color=colors[gname],
        label=gname,
        alpha=0.9,
    )

plt.xticks(label_indices + bar_width * (len(group_order)-1) / 2, label_values)
plt.xlabel("Label", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.yscale("log")
plt.title("Label Distribution per Group", fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "combined_label_distribution.png"), dpi=300, bbox_inches="tight")
plt.close()


# -----------------------------
# 3️⃣ MEAN VECTOR + 4️⃣ DISTANCE DISTRIBUTION
# -----------------------------


def plot_distance_distributions_edges(groups, df, dim_cols, colors, plot_path, bins=50):
    """
    Plot cosine distance distributions to mean vector for all groups in the same figure
    using only edge colors (no filled histograms).
    """
    plt.figure(figsize=(8,4))

    for gname, mask in groups.items():
        gdf = df[mask]
        if len(gdf) == 0:
            continue

        # Filter rows inside event window
        gdf = filter_rows_in_event_window(gdf)
        if len(gdf) == 0:
            continue

        X = gdf[dim_cols].values
        mean_vec = X.mean(axis=0, keepdims=True)
        dists = cosine_distances(X, mean_vec).flatten()

        plt.hist(
            dists, 
            bins=bins, 
            histtype='step',        # only edges, no fill
            linewidth=2,
            color=colors.get(gname, "#333333"), 
            label=gname
        )

    plt.xlabel("Cosine Distance to Mean Vector", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title("Distance Distribution to Mean Vector by Group", fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

# Plot all distance distributions in one figure
plot_distance_distributions_edges(
    groups, 
    df, 
    dim_cols, 
    colors, 
    os.path.join(plot_dir, "all_groups_distance_distribution.png"),
    bins=50
)


# -----------------------------
# 5️⃣ Extract IR_108 from NC files
# -----------------------------

def extract_ir108_stats(nc_path):
    try:
        ds = xr.open_dataset(nc_path, engine='netcdf4')
        if "IR_108" not in ds.variables:
            return np.nan, np.nan
        arr = ds["IR_108"].values
        return np.nanmean(arr), np.nanmin(arr)
    except Exception:
        return np.nan, np.nan


# --- IR_108 distributions for all groups in a single plot ---
plt.figure(figsize=(10,6))
for gname, mask in groups.items():
    print(f"Processing IR_108 stats for group {gname}...")
    gdf = df[mask]
    if len(gdf) == 0:
        continue

    # Filter rows inside event window
    gdf = filter_rows_in_event_window(gdf)
    if len(gdf) == 0:
        continue

    means = []
    mins = []

    for p in gdf["path"].values:
        m, mn = extract_ir108_stats(p)
        means.append(m)
        mins.append(mn)

    # Store results back (optional)
    gdf.loc[:, "ir108_mean"] = means
    gdf.loc[:, "ir108_min"] = mins

    # Plot mean values
    plt.hist([v for v in means if not np.isnan(v)],
             bins=50,
             histtype='step',        # only edges
             linewidth=2,
             color=colors.get(gname, "#333333"),
             label=f"{gname} mean")

plt.xlabel("IR_108 Mean Value", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("IR_108 Mean Distribution by Group", fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "IR108_mean_all_groups.png"), dpi=300)
plt.close()


# --- IR_108 min values ---
plt.figure(figsize=(10,6))
for gname, mask in groups.items():
    gdf = df[mask]
    if len(gdf) == 0:
        continue
    gdf = filter_rows_in_event_window(gdf)
    if len(gdf) == 0:
        continue

    mins = [extract_ir108_stats(p)[1] for p in gdf["path"].values]
    gdf.loc[:, "ir108_min"] = mins

    plt.hist([v for v in mins if not np.isnan(v)],
             bins=50,
             histtype='step',
             linewidth=2,
             color=colors.get(gname, "#333333"),
             label=f"{gname} min")

plt.xlabel("IR_108 Min Value", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("IR_108 Min Distribution by Group", fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "IR108_min_all_groups.png"), dpi=300)
plt.close()

print("✓ All plots saved to:", plot_dir)
