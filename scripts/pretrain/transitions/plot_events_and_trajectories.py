#!/usr/bin/env python3
"""
Simplified multiplot: ESWD events (colored by month) and storm trajectories (colored by relative time).
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ==================================================
# CONFIGURATION
# ==================================================
ESWD_CSV = "/data1/crops/test_case_essl_14-15-16-18-19-20-22-23-24_100x100_ir108_cma_traj/eswd-v2-2012-2025_expats.csv"
TRAJ_CSV = "/data1/crops/test_case_essl_14-15-16-18-19-20-22-23-24_100x100_ir108_cma_traj/storm_trajectories_after_merge.csv"

OUTPUT_DIR = "/data1/crops/test_case_essl_14-15-16-18-19-20-22-23-24_100x100_ir108_cma_traj"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DOMAIN_EXTENT = [5, 16, 42, 51.5]
OROG_PATH = "/data1/DEM_EXPATS_0.01x0.01.nc"

# Filter parameters
TARGET_YEARS = [2014, 2015, 2016, 2018, 2019, 2020, 2022, 2023, 2024]
TARGET_MONTHS = [4, 5, 6, 7, 8, 9]  # April to September

# Plotting
EVENT_ALPHA = 0.7
TRAJ_LINEWIDTH = 2.0

# Month colors
MONTH_COLORS = {
    4: '#1f77b4',   # April - blue
    5: '#ff7f0e',   # May - orange
    6: '#2ca02c',   # June - green
    7: '#d62728',   # July - red
    8: '#9467bd',   # August - purple
    9: '#8c564b',   # September - brown
}

# ==================================================
# IMPORTS FOR OROGRAPHY
# ==================================================
sys.path.append("/home/Daniele/codes/VISSL_postprocessing/scripts/pretrain/")
from transitions.plot_utils import plot_orography_map  # noqa: E402


def load_eswd_events(eswd_csv):
    """Load ESWD events and normalize types/timestamps."""
    if not os.path.exists(eswd_csv):
        raise FileNotFoundError(f"ESWD CSV not found: {eswd_csv}")

    events = pd.read_csv(eswd_csv, low_memory=False)
    required_cols = {"TIME_EVENT", "LATITUDE", "LONGITUDE", "TYPE_EVENT"}
    missing_cols = [c for c in required_cols if c not in events.columns]
    if missing_cols:
        raise ValueError(
            f"ESWD CSV missing required columns: {missing_cols}. "
            f"Found: {list(events.columns)}"
        )

    events = events.copy()
    events["TIME_EVENT"] = pd.to_datetime(events["TIME_EVENT"], errors="coerce", utc=True).dt.tz_convert(None)
    events["LATITUDE"] = pd.to_numeric(events["LATITUDE"], errors="coerce")
    events["LONGITUDE"] = pd.to_numeric(events["LONGITUDE"], errors="coerce")
    events["TYPE_EVENT"] = events["TYPE_EVENT"].astype(str).str.upper().str.strip()
    events = events.dropna(subset=["TIME_EVENT", "LATITUDE", "LONGITUDE"])

    # Extract year and month
    events["year"] = events["TIME_EVENT"].dt.year
    events["month"] = events["TIME_EVENT"].dt.month

    print(f"Loaded {len(events)} ESWD events from {eswd_csv}")
    return events


def filter_events_by_year_month(events_df, target_years, target_months):
    """Filter events to keep only specified years and months."""
    filtered = events_df[
        (events_df["year"].isin(target_years)) &
        (events_df["month"].isin(target_months))
    ]
    print(f"After filtering: {len(filtered)} events in years {target_years} and months {target_months}")
    return filtered


def load_trajectories(traj_csv):
    """Load trajectory data."""
    if not os.path.exists(traj_csv):
        raise FileNotFoundError(f"Trajectory CSV not found: {traj_csv}")

    traj = pd.read_csv(traj_csv, low_memory=False)
    traj["time"] = pd.to_datetime(traj["time"], errors="coerce", utc=True).dt.tz_convert(None)
    traj = traj.dropna(subset=["time", "lat", "lon", "storm_id"])

    print(f"Loaded {len(traj)} trajectory points for {traj['storm_id'].nunique()} storms")
    return traj


def plot_events_subplot(ax, events_df):
    """Plot ESWD events with markers by type, colored by month."""
    
    # Separate by event type
    hail_events = events_df[events_df["TYPE_EVENT"] == "HAIL"]
    precip_events = events_df[events_df["TYPE_EVENT"] == "PRECIP"]
    
    # Plot HAIL events (triangles)
    if not hail_events.empty:
        for month in sorted(hail_events["month"].unique()):
            month_data = hail_events[hail_events["month"] == month]
            ax.scatter(
                month_data["LONGITUDE"],
                month_data["LATITUDE"],
                marker="^",
                s=80,
                color=MONTH_COLORS.get(month, 'gray'),
                edgecolors="black",
                linewidths=0.5,
                alpha=EVENT_ALPHA,
                transform=ccrs.PlateCarree(),
                zorder=15,
                label=f"HAIL (Month {month})" if month == sorted(hail_events["month"].unique())[0] else ""
            )

    # Plot PRECIP events (circles)
    if not precip_events.empty:
        for month in sorted(precip_events["month"].unique()):
            month_data = precip_events[precip_events["month"] == month]
            ax.scatter(
                month_data["LONGITUDE"],
                month_data["LATITUDE"],
                marker="o",
                s=60,
                color=MONTH_COLORS.get(month, 'gray'),
                edgecolors="black",
                linewidths=0.5,
                alpha=EVENT_ALPHA,
                transform=ccrs.PlateCarree(),
                zorder=15,
                label=f"PRECIP (Month {month})" if month == sorted(precip_events["month"].unique())[0] else ""
            )

    ax.set_title("ESWD Events (April-September, 2014-2024)", fontsize=14, fontweight="bold")
    ax.coastlines("50m", linewidth=0.8, color="black")
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="black")
    ax.set_extent(DOMAIN_EXTENT, crs=ccrs.PlateCarree())
    
    gl = ax.gridlines(draw_labels=True, alpha=0.4, linewidth=0.7, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 10}
    gl.ylabel_style = {"size": 10}


def plot_trajectories_subplot(ax, traj_df):
    """Plot trajectories colored by relative time within each storm."""
    
    storms = traj_df["storm_id"].unique()
    print(f"Plotting {len(storms)} storms...")
    
    for storm_id in storms:
        storm_data = traj_df[traj_df["storm_id"] == storm_id].sort_values("time")
        
        if len(storm_data) < 2:
            continue
        
        # Calculate relative time (in hours from storm start)
        start_time = storm_data["time"].min()
        storm_data_copy = storm_data.copy()
        storm_data_copy["rel_time_hours"] = (storm_data_copy["time"] - start_time).dt.total_seconds() / 3600.0
        
        max_rel_time = storm_data_copy["rel_time_hours"].max()
        
        # Plot trajectory segments with color gradient
        for i in range(len(storm_data_copy) - 1):
            lon1 = storm_data_copy.iloc[i]["lon"]
            lat1 = storm_data_copy.iloc[i]["lat"]
            lon2 = storm_data_copy.iloc[i + 1]["lon"]
            lat2 = storm_data_copy.iloc[i + 1]["lat"]
            
            # Normalize color by relative time
            rel_time = storm_data_copy.iloc[i]["rel_time_hours"]
            color_val = rel_time / (max_rel_time + 1e-6)  # Normalize to [0, 1]
            color = plt.cm.viridis(color_val)
            
            ax.plot(
                [lon1, lon2],
                [lat1, lat2],
                color=color,
                linewidth=TRAJ_LINEWIDTH,
                transform=ccrs.PlateCarree(),
                zorder=10,
            )
    
    ax.set_title("Storm Trajectories (Colored by Relative Time)", fontsize=14, fontweight="bold")
    ax.coastlines("50m", linewidth=0.8, color="black")
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="black")
    ax.set_extent(DOMAIN_EXTENT, crs=ccrs.PlateCarree())
    
    gl = ax.gridlines(draw_labels=True, alpha=0.4, linewidth=0.7, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 10}
    gl.ylabel_style = {"size": 10}
    
    # Add colorbar for relative time
    sm = ScalarMappable(cmap=plt.cm.viridis, norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8)
    cbar.set_label("Relative Time (normalized)", fontsize=10)


def main():
    print("Loading events and trajectories...")
    
    # Load and filter ESWD events
    events_df = load_eswd_events(ESWD_CSV)
    events_filtered = filter_events_by_year_month(events_df, TARGET_YEARS, TARGET_MONTHS)
    
    # Load trajectories
    traj_df = load_trajectories(TRAJ_CSV)
    
    # Create figure with 2 subplots
    fig = plt.figure(figsize=(16, 7))
    
    # Left subplot: Events
    ax1 = fig.add_subplot(121, projection=ccrs.PlateCarree())
    plot_orography_map(
        OROG_PATH,
        ax=ax1,
        var_name="DEM",
        extent=DOMAIN_EXTENT,
        cmap="Greys",
        levels=30,
        alpha=0.5,
    )
    plot_events_subplot(ax1, events_filtered)
    
    # Right subplot: Trajectories
    ax2 = fig.add_subplot(122, projection=ccrs.PlateCarree())
    plot_orography_map(
        OROG_PATH,
        ax=ax2,
        var_name="DEM",
        extent=DOMAIN_EXTENT,
        cmap="Greys",
        levels=30,
        alpha=0.5,
    )
    plot_trajectories_subplot(ax2, traj_df)
    
    # Create legend for events (months and types)
    legend_elements = []
    for month in sorted(TARGET_MONTHS):
        legend_elements.append(
            mpatches.Patch(
                facecolor=MONTH_COLORS[month],
                edgecolor="black",
                label=f"Month {month}"
            )
        )
    legend_elements.append(mpatches.Patch(facecolor="none", edgecolor="none", label=""))
    legend_elements.append(mpatches.Line2D([0], [0], marker="^", color="w", 
                                          markerfacecolor="gray", markersize=8, 
                                          markeredgecolor="black", label="HAIL events"))
    legend_elements.append(mpatches.Line2D([0], [0], marker="o", color="w", 
                                          markerfacecolor="gray", markersize=7, 
                                          markeredgecolor="black", label="PRECIP events"))
    
    ax1.legend(handles=legend_elements, loc="upper left", fontsize=9, ncol=2)
    
    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, "events_and_trajectories.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {output_file}")
    plt.close()


if __name__ == "__main__":
    main()
