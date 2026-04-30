#!/usr/bin/env python3
"""
Compact multiplot with 2 sections:
- Left: ESWD events density map over orography
- Right: Storm trajectories colored by relative time
Uses storms from pathway CSV only.
"""
import io
import os
import re
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import gaussian_kde
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import boto3
import xarray as xr
from scipy.ndimage import binary_closing

# ==================================================
# CONFIGURATION
# ==================================================
PATHWAY_CSV = "/data1/fig/dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc/epoch_800/test_traj/pathway_analysis/df_pathways_merged_no_dominance.csv"
ESWD_CSV = "/data1/crops/test_case_essl_14-15-16-18-19-20-22-23-24_100x100_ir108_cma_traj/eswd-v2-2012-2025_expats.csv"
TRAJ_CSV = "/data1/crops/test_case_essl_14-15-16-18-19-20-22-23-24_100x100_ir108_cma_traj/storm_trajectories_after_merge.csv"

OUTPUT_DIR = "/data1/crops/test_case_essl_14-15-16-18-19-20-22-23-24_100x100_ir108_cma_traj"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DOMAIN_EXTENT = [5, 16, 41.9, 51.5]
OROG_PATH = "/data1/DEM_EXPATS_0.01x0.01.nc"
S3_DOMAIN_ROOT = "/data/sat/msg/ml_train_crops/IR_108-WV_062-CMA_FULL_EXPATS_DOMAIN"

# Filter parameters
TARGET_YEARS = [2014, 2015, 2016, 2018, 2019, 2020, 2022, 2023, 2024]
TARGET_MONTHS = [4, 5, 6, 7, 8, 9]  # April to September

# IR108 example day
EXAMPLE_DATE = "2023-08-16"

# Plotting
TRAJ_LINEWIDTH = 0.9
FRAME_SIZE_PX = 100
BT_MIN = 240
BT_MAX = 320
DOMAIN_ALPHA = 0.16
CROP_ALPHA = 0.92

# ==================================================
# IMPORTS FOR OROGRAPHY AND S3
# ==================================================
sys.path.append("/home/Daniele/codes/VISSL_postprocessing/scripts/pretrain/")
from transitions.plot_utils import plot_orography_map  # noqa: E402

sys.path.append("/home/Daniele/codes/ML_data_generator/")
from credentials_buckets import (  # noqa: E402
    S3_BUCKET_NAME,
    S3_ACCESS_KEY,
    S3_SECRET_ACCESS_KEY,
    S3_ENDPOINT_URL,
)


def make_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_ACCESS_KEY,
    )


def read_s3_nc(s3, key):
    obj = s3.get_object(Bucket=S3_BUCKET_NAME, Key=key)
    return xr.open_dataset(io.BytesIO(obj["Body"].read()), engine="h5netcdf")


def build_daily_key(ts):
    year = ts.strftime("%Y")
    month = ts.strftime("%m")
    day = ts.strftime("%d")
    return (
        f"{S3_DOMAIN_ROOT}/{year}/{month}/"
        f"merged_MSG_CMSAF_{year}-{month}-{day}.nc"
    )


def parse_crop_metadata(crop_name):
    """Extract storm_id from crop filename."""
    basename = os.path.basename(str(crop_name))
    pattern = r"^storm(?P<storm_id>\d+)_"
    match = re.match(pattern, basename)
    if match is None:
        return pd.NA
    return int(match.group("storm_id"))


def load_pathway_storms():
    """Load storms from pathway CSV."""
    if not os.path.exists(PATHWAY_CSV):
        raise FileNotFoundError(f"Pathway CSV not found: {PATHWAY_CSV}")
    
    pathway_df = pd.read_csv(PATHWAY_CSV, low_memory=False)
    pathway_df["storm_id"] = pathway_df["crop"].apply(parse_crop_metadata)
    pathway_df = pathway_df.dropna(subset=["storm_id"])
    pathway_df["storm_id"] = pathway_df["storm_id"].astype(int)
    
    valid_storms = set(pathway_df["storm_id"].unique())
    print(f"Loaded {len(valid_storms)} unique storms from pathway CSV")
    return valid_storms


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
    events["date"] = events["TIME_EVENT"].dt.date

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


def load_trajectories(traj_csv, valid_storms):
    """Load trajectory data for storms in valid_storms."""
    if not os.path.exists(traj_csv):
        raise FileNotFoundError(f"Trajectory CSV not found: {traj_csv}")

    traj = pd.read_csv(traj_csv, low_memory=False)
    traj["time"] = pd.to_datetime(traj["time"], errors="coerce", utc=True).dt.tz_convert(None)
    traj = traj.dropna(subset=["time", "lat", "lon", "storm_id"])
    
    # Filter to valid storms only
    traj["storm_id"] = traj["storm_id"].astype(int)
    traj = traj[traj["storm_id"].isin(valid_storms)]

    print(f"Loaded {len(traj)} trajectory points for {traj['storm_id'].nunique()} storms in pathway")
    return traj


def apply_cma_mask_to_ir108(ir108_field, cma_field):
    """Apply 3x3 binary closing to cma and set non-cloud IR108 pixels to 320."""
    cma_binary = np.asarray(cma_field) > 0
    closed_cma = binary_closing(cma_binary, structure=np.ones((3, 3), dtype=bool))

    ir108_masked = np.array(ir108_field, copy=True)
    ir108_masked[~closed_cma] = BT_MAX
    return ir108_masked


def get_crop_bounds_from_center(ds, center_lat, center_lon, frame_size_px=100):
    """Get 100x100 crop indices centered on given lat/lon, clipped to domain."""
    lon = ds.lon.values
    lat = ds.lat.values

    nx = len(lon)
    ny = len(lat)
    half = frame_size_px // 2

    ix_center = int((abs(lon - center_lon)).argmin())
    iy_center = int((abs(lat - center_lat)).argmin())

    ix0 = max(0, min(ix_center - half, nx - frame_size_px))
    iy0 = max(0, min(iy_center - half, ny - frame_size_px))
    ix1 = ix0 + frame_size_px
    iy1 = iy0 + frame_size_px

    return ix0, ix1, iy0, iy1


def filter_events_for_frame(events_df, frame_start, lon_min, lon_max, lat_min, lat_max):
    """Filter ESWD events inside [frame_start, frame_start+15min) and crop bbox."""
    if events_df.empty:
        return events_df

    t0 = pd.Timestamp(frame_start)
    t1 = t0 + pd.Timedelta(minutes=15)

    return events_df[
        (events_df["TIME_EVENT"] >= t0)
        & (events_df["TIME_EVENT"] < t1)
        & (events_df["LONGITUDE"] >= lon_min)
        & (events_df["LONGITUDE"] <= lon_max)
        & (events_df["LATITUDE"] >= lat_min)
        & (events_df["LATITUDE"] <= lat_max)
    ]


def plot_eswd_events(ax, events_df):
    """Overlay ESWD events with marker by TYPE_EVENT."""
    if events_df.empty:
        return

    hail_events = events_df[events_df["TYPE_EVENT"] == "HAIL"]
    precip_events = events_df[events_df["TYPE_EVENT"] == "PRECIP"]

    if not hail_events.empty:
        ax.scatter(
            hail_events["LONGITUDE"],
            hail_events["LATITUDE"],
            marker="^",
            s=55,
            color="yellow",
            edgecolors="black",
            linewidths=0.5,
            transform=ccrs.PlateCarree(),
            zorder=20,
        )

    if not precip_events.empty:
        ax.scatter(
            precip_events["LONGITUDE"],
            precip_events["LATITUDE"],
            marker="o",
            s=45,
            color="lightblue",
            edgecolors="black",
            linewidths=0.5,
            transform=ccrs.PlateCarree(),
            zorder=20,
        )


def get_events_for_date(events_df, target_date):
    """Get events for a specific date."""
    return events_df[events_df["date"] == target_date]


def style_geo_axes(ax, show_left_labels=True, show_bottom_labels=True, label_size=12):
    """Apply consistent map style and avoid duplicated geo labels."""
    ax.coastlines("50m", linewidth=0.8, color="black")
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="black")
    ax.set_extent(DOMAIN_EXTENT, crs=ccrs.PlateCarree())

    gl = ax.gridlines(draw_labels=True, alpha=0.4, linewidth=0.7, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = show_left_labels
    gl.bottom_labels = show_bottom_labels
    gl.xlocator = mticker.FixedLocator([6, 9, 12, 15])
    gl.ylocator = mticker.FixedLocator([42, 45, 48, 51])
    gl.xlabel_style = {"size": label_size}
    gl.ylabel_style = {"size": label_size}


def plot_events_density_subplot(ax, events_df):
    """Plot ESWD events as contour density map overlaid on orography."""
    
    if events_df.empty:
        ax.set_title("ESWD Events Density Map", fontsize=12, fontweight="bold")
        return
    
    lon_vals = events_df["LONGITUDE"].values
    lat_vals = events_df["LATITUDE"].values

    if len(lon_vals) >= 3:
        xy = np.vstack([lon_vals, lat_vals])
        kde = gaussian_kde(xy, bw_method=0.2)

        grid_lon = np.linspace(DOMAIN_EXTENT[0], DOMAIN_EXTENT[1], 220)
        grid_lat = np.linspace(DOMAIN_EXTENT[2], DOMAIN_EXTENT[3], 220)
        xx, yy = np.meshgrid(grid_lon, grid_lat)
        zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

        low_level = np.nanpercentile(zz, 45)
        high_level = np.nanmax(zz)
        levels = np.linspace(low_level, high_level, 18)
        levels = levels[levels > 0]
        if len(levels) > 1:
            ax.contourf(
                xx,
                yy,
                zz,
                levels=levels,
                cmap="YlOrRd",
                alpha=0.68,
                transform=ccrs.PlateCarree(),
                zorder=8,
            )
            ax.contour(
                xx,
                yy,
                zz,
                levels=levels,
                colors="maroon",
                linewidths=0.45,
                alpha=0.7,
                transform=ccrs.PlateCarree(),
                zorder=9,
            )
    
    ax.set_title("ESWD Events Density Map", fontsize=12, fontweight="bold")
    style_geo_axes(ax, show_left_labels=True, show_bottom_labels=True, label_size=13)


def plot_trajectories_subplot(ax, traj_df):
    """Plot trajectories colored by relative time within each storm."""
    
    storms = traj_df["storm_id"].unique()
    n_storms = len(storms)
    print(f"Plotting {n_storms} storms in trajectories...")
    
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
    
    ax.set_title(f"Storm Trajectories (n={n_storms})", fontsize=12, fontweight="bold")
    style_geo_axes(ax, show_left_labels=False, show_bottom_labels=True, label_size=13)


def main():
    print("=" * 60)
    print("Creating comprehensive multiplot with events, IR108, and trajectories")
    print("=" * 60)
    
    # Load pathway storms
    valid_storms = load_pathway_storms()
    
    # Load and filter ESWD events
    print("\nLoading ESWD events...")
    events_df = load_eswd_events(ESWD_CSV)
    events_filtered = filter_events_by_year_month(events_df, TARGET_YEARS, TARGET_MONTHS)
    
    # Load trajectories (filtered to pathway storms)
    print("\nLoading trajectories...")
    traj_df = load_trajectories(TRAJ_CSV, valid_storms)
    
    # Load IR108 for example date
    print(f"\nLoading IR108 data for {EXAMPLE_DATE}...")
    s3 = make_s3_client()
    example_date_obj = pd.to_datetime(EXAMPLE_DATE)
    day_key = build_daily_key(example_date_obj)
    
    try:
        ds_example = read_s3_nc(s3, day_key)
        print(f"Loaded: {day_key}")
    except Exception as e:
        print(f"Warning: Could not load {day_key}: {e}")
        ds_example = None
    
    # Create figure with simpler 2-panel layout
    fig = plt.figure(figsize=(12.8, 4.8))
    gs = fig.add_gridspec(1, 2, wspace=0.16)

    # Left plot: Events contour density map
    print("\nCreating left subplot (events density map)...")
    ax_events = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    plot_orography_map(
        OROG_PATH,
        ax=ax_events,
        var_name="DEM",
        extent=DOMAIN_EXTENT,
        cmap="Greys",
        levels=30,
        alpha=0.6,
    )
    plot_events_density_subplot(ax_events, events_filtered)

    # Right plot: Trajectories
    print("Creating right subplot (trajectories)...")
    ax_traj = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
    plot_orography_map(
        OROG_PATH,
        ax=ax_traj,
        var_name="DEM",
        extent=DOMAIN_EXTENT,
        cmap="Greys",
        levels=30,
        alpha=0.6,
    )
    plot_trajectories_subplot(ax_traj, traj_df)
    
    plt.tight_layout()
    
    output_file = os.path.join(OUTPUT_DIR, "events_trajectories_ir108_comprehensive.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\n✅ Saved figure to {output_file}")
    plt.close()


if __name__ == "__main__":
    main()
