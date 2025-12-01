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

# =============================================================
# 2D FEATURE SPACE PLOT WITH GROUP MEANS + DENSITY CONTOURS
# =============================================================
import seaborn as sns
from scipy.stats import gaussian_kde

feature_cols_2d = ["2d_dim_1", "2d_dim_2"]

# ----------------------------
# Background = TRAIN points
# ----------------------------
df_train = df[df["vector_type"] == "TRAIN"].dropna(subset=feature_cols_2d)

#filter only the time in the event window
print(f"Training samples: {len(df_train)}")

# === COLOR MAP ===
COLORS_PER_CLASS = {
    '0': 'darkgray', '1': 'darkslategrey', '2': 'peru', '3': 'orangered',
    '4': 'lightcoral', '5': 'deepskyblue', '6': 'purple', '7': 'lightblue',
    '8': 'green', '9': 'goldenrod', '10': 'magenta', '11': 'dodgerblue',
    '12': 'darkorange', '13': 'olive', '14': 'crimson'
}

plt.figure(figsize=(10, 8))

# scatter background
plt.scatter(
    df_train["2d_dim_1"],
    df_train["2d_dim_2"],
    c=df_train["label"].astype(str).map(COLORS_PER_CLASS),
    s=5,
    alpha=0.1,
    label="training"
)

# ----------------------------------------------------
# MARKERS for the group means
# ----------------------------------------------------
group_markers = {
    "ALL": "X",
    "RAIN_ALL": "P",
    "RAIN_10th": "D",
    "HAIL_ALL": "s",
    "HAIL_10th": "o"
}

# ----------------------------------------------------
# Function: density contour for a group
# ----------------------------------------------------
def plot_density_contours(df_g, color):
    if len(df_g) < 10:
        return
    x = df_g["2d_dim_1"].values
    y = df_g["2d_dim_2"].values
    kde = gaussian_kde(np.vstack([x, y]))
    
    # Grid for KDE
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    density = kde(positions).reshape(xx.shape)
    
    # two contour levels: 50% and 90%
    levels = np.percentile(density, [50, 10])
    
    plt.contour(
        xx, yy, density,
        levels=levels,
        colors=[color],
        linewidths=1.6,
        alpha=1.0
    )


# ----------------------------------------------------
# PLOT each group
# ----------------------------------------------------
for name, mask in groups.items():
    
    gdf = df[mask]
    gdf = filter_rows_in_event_window(gdf)

    # plot density contours
    #plot_density_contours(gdf, colors[gname])

    # compute mean vector position
    mean_x = gdf["2d_dim_1"].mean()
    mean_y = gdf["2d_dim_2"].mean()

    # plot mean marker
    plt.scatter(
        mean_x,
        mean_y,
        c=colors[name],
        s=200,
        marker=group_markers[name],
        edgecolor="black",
        linewidth=1.2,
        label=f"{name} mean",
        alpha=0.8
    )


plt.xlabel("2d_dim_1", fontsize=14)
plt.ylabel("2d_dim_2", fontsize=14)
plt.title("2D Feature Space with Group Means & Density Contours", fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.tight_layout()

plot_path_2d = os.path.join(plot_dir, "2d_feature_space_groups.png")
plt.savefig(plot_path_2d, dpi=300, bbox_inches="tight", transparent=True)
plt.close()

print("Saved:", plot_path_2d)
