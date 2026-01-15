import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

# === CONFIG ===
RUN_NAME = "dcv2_resnet_k8_ir108_100x100_2013-2020_1xrandomcrops_1xtimestamp_cma_nc_convective"
path_to_dir = f"/data1/fig/{RUN_NAME}/epoch_800/closest/"
merged_path = os.path.join(path_to_dir, "merged_crops_stats_cvc_imergtime_closest_1000.csv")

output_dir = os.path.join(path_to_dir, "diurnal_and_seasonal_cycles/")
os.makedirs(output_dir, exist_ok=True)

# === COLOR MAP ===
COLORS_PER_CLASS = {
    '0': 'darkgray', '1': 'darkslategrey', '2': 'peru', '3': 'orangered',
    '4': 'lightcoral', '5': 'deepskyblue', '6': 'purple', '7': 'lightblue',
    '8': 'green', '9': 'goldenrod', '10': 'magenta', '11': 'dodgerblue',
    '12': 'darkorange', '13': 'olive', '14': 'crimson'
}

# Variables to plot
VARIABLES = {
    "cma": {"label": "Cloud Cover (%)", "percentile": None, "scale": 100, "log": False, "vmin": 0, "vmax": 100},
    "cth": {"label": "Cloud Top \n Height (km)", "percentile": 50, "scale": 0.001, "log": False, "vmin": 1, "vmax": 11},
    #"cth": {"label": "Cloud Top \n Height (km)", "percentile": 99, "scale": 0.001, "log": False, "vmin": 2, "vmax": 13},
    #"cot": {"label": "Cloud Optical \n Thickness", "percentile": 50, "scale": 1, "log": False, "vmin": 1, "vmax": 30},
    "ccv": {"label": "Convective \n Cloud Cover (%)", "percentile": None, "scale": 1, "log": True, "vmin": 0.02, "vmax": 70},
    #"precipitation": {"label": "Rain Rate (mm/h)", "percentile": 50, "scale": 1, "log": True},
    "precipitation": {"label": "Rain Rate (mm/h)", "percentile": 99, "scale": 1, "log": True},
    #"precipitation": {"label": "Total Prec (mm)", "percentile": None, "scale": 0.5, "log": True, "vmin": 1, "vmax": 5000},
    #"euclid_msg_grid": {"label": "Lightning Count", "percentile": None, "scale": 1, "log": True},
}

INCLUDE_CLASS_OCC = True

# -----------------------------
# === Helper functions ===
# -----------------------------
def extract_hour_from_path(time):
    """Extract hour time yyyy-mm-dd hh:mm:ss from filename"""
    return pd.to_datetime(time).hour    


def extract_month_from_path(time):
    """Extract month (1–12) from filename"""
    return pd.to_datetime(time).month    
    

# -----------------------------
# === LOAD DATA ===
# -----------------------------
df = pd.read_csv(merged_path)
print(f"Loaded: {merged_path} ({df.shape})")

df["hour"] = df["time"].apply(extract_hour_from_path)
df["month"] = df["time"].apply(extract_month_from_path)
df = df.dropna(subset=["hour", "month"])

labels = sorted(df["label"].unique())


# ------------------------------------------------------
# === GLOBAL DIURNAL CYCLE (CLASS OCCURRENCE ONLY) ===
# ------------------------------------------------------
# plt.figure(figsize=(7, 4))

# for label in labels:
#     subset = df[df["label"] == label]
#     hourly_counts = subset["hour"].value_counts().sort_index()
#     hourly_percent = hourly_counts / hourly_counts.sum() * 100

#     plt.plot(
#         hourly_percent.index,
#         hourly_percent.values,
#         linewidth=2.2,
#         label=f"Class {label}",
#         color=COLORS_PER_CLASS[str(label)]
#     )

# plt.title("Diurnal Cycle – Class Occurrence", fontsize=14, fontweight="bold")
# plt.xlabel("Hour (UTC)")
# plt.ylabel("Occurrence (%)")
# plt.xticks(range(0, 24, 2))
# plt.ylim(0, 20)
# plt.grid(alpha=0.3)
# plt.legend(ncol=3)

# plt.savefig(os.path.join(output_dir, "diurnal_classes.png"),
#             dpi=300, transparent=True, bbox_inches="tight")
# plt.close()
# print("Saved diurnal class cycle.")


# ------------------------------------------------------
# === MULTI-VARIABLE DIURNAL CYCLE (SUBPLOTS) ===
# ------------------------------------------------------

n_vars = len(VARIABLES) + 1  # +1 for class occurrence
fig, axes = plt.subplots(n_vars, 1, figsize=(5, 2*n_vars), sharex=True)

# ---------- Row 1: Class occurrence ----------
ax = axes[0]

# ensure hour is integer 0..23
df['hour'] = df['hour'].astype(int)

# pivot: rows = hour, cols = label, values = counts
counts = df.groupby(['hour', 'label']).size().unstack(fill_value=0)

# ensure all hours 0..23 present
all_hours = np.arange(24)
counts = counts.reindex(all_hours, fill_value=0)

# row-wise normalization -> percentage of each label within that hour
totals_per_hour = counts.sum(axis=1).replace(0, np.nan)  # avoid div-by-zero

rel_pct = counts.div(totals_per_hour, axis=0) * 100
rel_pct = rel_pct.fillna(0)  # hours with no samples -> zeros

# plot
for label in sorted(rel_pct.columns):
    color = COLORS_PER_CLASS.get(str(label), 'black')
    ax.plot(rel_pct.index, rel_pct[label], label=f"Label {label}", color=color, linewidth=2)

ax.set_ylabel("Class Occ (%)", fontsize=12)
ax.set_title("Class Occurrence", fontsize=12, fontweight="bold")
ax.tick_params(axis="y", labelsize=12)
ax.grid(alpha=0.3)
ax.set_ylim(0, 45)

# ---------- Other rows: Variables ----------
for i, (var, info) in enumerate(VARIABLES.items(), start=1):
    print(f"Processing diurnal for variable: {var} – {info}")
    
    ax = axes[i]
    #identify row with the variable
    df_var = df[df['var']==var]

    for label in labels:
        subset = df_var[df_var["label"] == label].copy()

        if info["percentile"] is not None:
            colname = f"{info['percentile']}"
        else:
            colname = "None"

        if colname not in df.columns:
            continue
            
        #apply scale before avarage
        subset[colname] = subset[colname] * info["scale"]
        hourly_avg = subset.groupby("hour")[colname].mean()

        ax.plot(
            hourly_avg.index,
            hourly_avg.values,
            color=COLORS_PER_CLASS[str(label)],
            linewidth=2,
            label=f"Class {label}"
        )

    if info["percentile"] is not None:
        ax.set_ylabel(f"{info['label']} (p{info['percentile']})", fontsize=12)
    else:
        ax.set_ylabel(info["label"], fontsize=12)
    clean_label = info["label"].replace("\n", "")
    ax.set_title(clean_label, fontsize=12, fontweight="bold")
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(alpha=0.3)
    if info["log"]:
        ax.set_yscale("log")
    if "vmin" in info and "vmax" in info:
        ax.set_ylim(info["vmin"], info["vmax"])

plt.xlabel("Hour (UTC)", fontsize=12)
plt.xticks(range(0, 24, 2), fontsize=12)
plt.tight_layout()

plt.savefig(os.path.join(output_dir, "diurnal_multi_variable.png"),
            dpi=300, transparent=True, bbox_inches="tight")
plt.close()
print("Saved multi-variable diurnal plot.")




# ======================================================
# === SEASONAL CYCLES (MONTHLY) ===
# ======================================================
fig, axes = plt.subplots(n_vars, 1, figsize=(5, 2*n_vars), sharex=True)

# ======================================================
# === Row 1 — Class Occurrence (Normalized by Month) ===
# ======================================================
ax = axes[0]

df["month"] = df["month"].astype(int)

# pivot: rows = month, columns = label, values = counts
monthly_counts = df.groupby(["month", "label"]).size().unstack(fill_value=0)

# ensure months 4..9 exist
all_months = np.arange(4, 10)
monthly_counts = monthly_counts.reindex(all_months, fill_value=0)

# normalize per month (row-wise)
totals_per_month = monthly_counts.sum(axis=1).replace(0, np.nan)
rel_pct = monthly_counts.div(totals_per_month, axis=0) * 100
rel_pct = rel_pct.fillna(0)

# plot
for label in sorted(rel_pct.columns):
    color = COLORS_PER_CLASS.get(str(label), "black")
    ax.plot(rel_pct.index, rel_pct[label],
            linewidth=2, color=color, label=f"Label {label}")

ax.set_ylabel("Class Occ (%)")
ax.set_title("Seasonal Cycle – Class Occurrence", fontweight="bold")
ax.grid(alpha=0.3)
ax.set_ylim(0, 30)
ax.tick_params(axis="y", labelsize=12)

# Month labels
ax.set_xticks(all_months)
ax.set_xticklabels(["Apr","May","Jun",
                    "Jul","Aug","Sep"],
                   rotation=0, fontsize=12)


# ======================================================
# === Other Variables (Monthly Means per Class) ===
# ======================================================
for i, (var, info) in enumerate(VARIABLES.items(), start=1):
    ax = axes[i]
    df_var = df[df["var"] == var]

    if info["percentile"] is not None:
        colname = f"{info['percentile']}"
    else:
        colname = "None"

    if colname not in df.columns:
        continue

    for label in labels:
        subset = df_var[df_var["label"] == label]

        if subset.empty:
            continue

        # compute monthly averages
        monthly_avg = subset.groupby("month")[colname].mean() * info["scale"]

        # ensure months 1..12 exist
        monthly_avg = monthly_avg.reindex(all_months)

        ax.plot(
            monthly_avg.index,
            monthly_avg.values,
            linewidth=2,
            color=COLORS_PER_CLASS[str(label)],
            label=f"Class {label}"
        )

    if info["percentile"] is not None:
        ax.set_ylabel(f"{info['label']} (p{info['percentile']})", fontsize=12)
    else:
        ax.set_ylabel(info["label"], fontsize=12)
    clean_label = info["label"].replace("\n", "")
    ax.set_title(clean_label, fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.tick_params(axis="y", labelsize=12)

    if info.get("log", False):
        ax.set_yscale("log")

    if "vmin" in info and "vmax" in info:
        ax.set_ylim(info["vmin"], info["vmax"])

plt.xlabel("Month", fontsize=12)
plt.tight_layout()

plt.savefig(os.path.join(output_dir, "seasonal_multi_variable.png"),
            dpi=300, transparent=True, bbox_inches="tight")
plt.close()

print("Saved seasonal plots.")

