#!/usr/bin/env python3
"""
Simplified ESWD events + orography plotting workflow.

This script creates multiple outputs in a dedicated folder:
1) KDE density contour maps:
   - all events
   - precip-only events
   - hail-only events
2) Orography maps with event points (hail and precip in different colors):
   - full dataset
3) For the 5 busiest ESWD days, plot one map per 15-min time slot with events overlaid.
4) For each of the 5 busiest days, identify the peak 15-min slot and plot surrounding
   10 slots (peak ±5) with DBSCAN clustering (radius 150 km):
   - points colored by cluster
   - marker shape by event type (hail triangle, precip circle)
   - centroid stars
   - 150 km circle around each centroid
5) Build global 15-min counts across all events and run DBSCAN clustering on the
    10 busiest timestamps, to compare with the per-day peak-window strategy.

All plots use only orography as map background (no IR108).
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.geodesic import Geodesic
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN

import sys
sys.path.append("/home/Daniele/codes/VISSL_postprocessing/scripts/pretrain/")
from transitions.plot_utils import plot_orography_map  # noqa: E402


# ==================================================
# CONFIGURATION
# ==================================================
ESWD_CSV = "/data1/crops/test_case_essl_14-15-16-18-19-20-22-23-24_100x100_ir108_cma_traj/eswd-v2-2012-2025_expats.csv"
OROG_PATH = "/data1/DEM_EXPATS_0.01x0.01.nc"

OUTPUT_ROOT = Path(
    "/data1/crops/test_case_essl_14-15-16-18-19-20-22-23-24_100x100_ir108_cma_traj"
)
OUTPUT_DIR = OUTPUT_ROOT / "events_orography_plots"

DOMAIN_EXTENT = [5, 16, 41.9, 51.5]  # lon_min, lon_max, lat_min, lat_max

TARGET_YEARS = [2014, 2015, 2016, 2018, 2019, 2020, 2022, 2023, 2024] #skip 2013, 2017, 2021, 2025 used in training data
TARGET_MONTHS = [4, 5, 6, 7, 8, 9]
VALID_TYPES = {"HAIL", "PRECIP"}

# Plot-group switches
PLOT_DENSITY_MAPS = False
PLOT_OROGRAPHY_DOTS = True
PLOT_DAILY_15MIN = False
PLOT_TOP10_DBSCAN = True
PLOT_GLOBAL_TOP10_DBSCAN = True

# Keep this block name for compatibility with the original request.
# Strategy is now busiest-day based (not random-day based).
BUSIEST_DAYS_15MIN = 5

DBSCAN_RADIUS_KM = 150.0
EARTH_RADIUS_KM = 6371.0


# ==================================================
# DATA LOADING AND PREP
# ==================================================
def load_eswd_events(path: str) -> pd.DataFrame:
    """Load and sanitize ESWD events used by all plots."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"ESWD CSV not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    required = {"TIME_EVENT", "LATITUDE", "LONGITUDE", "TYPE_EVENT"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["TIME_EVENT"] = pd.to_datetime(df["TIME_EVENT"], errors="coerce", utc=True).dt.tz_convert(None)
    df["LATITUDE"] = pd.to_numeric(df["LATITUDE"], errors="coerce")
    df["LONGITUDE"] = pd.to_numeric(df["LONGITUDE"], errors="coerce")
    df["TYPE_EVENT"] = df["TYPE_EVENT"].astype(str).str.upper().str.strip()
    df = df.dropna(subset=["TIME_EVENT", "LATITUDE", "LONGITUDE", "TYPE_EVENT"])

    # Domain/type/time filtering to keep relevant events.
    df = df[
        (df["LONGITUDE"] >= DOMAIN_EXTENT[0])
        & (df["LONGITUDE"] <= DOMAIN_EXTENT[1])
        & (df["LATITUDE"] >= DOMAIN_EXTENT[2])
        & (df["LATITUDE"] <= DOMAIN_EXTENT[3])
        & (df["TYPE_EVENT"].isin(VALID_TYPES))
    ].copy()

    df["year"] = df["TIME_EVENT"].dt.year
    df["month"] = df["TIME_EVENT"].dt.month
    df = df[df["year"].isin(TARGET_YEARS) & df["month"].isin(TARGET_MONTHS)].copy()

    df["date"] = df["TIME_EVENT"].dt.date
    df["time_15min"] = df["TIME_EVENT"].dt.floor("15min")

    return df


# ==================================================
# PLOTTING HELPERS
# ==================================================
def style_geo_axes(ax, label_size: int = 14) -> None:
    """Apply consistent map styling."""
    ax.coastlines("50m", linewidth=0.8, color="black")
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="black")
    ax.set_extent(DOMAIN_EXTENT, crs=ccrs.PlateCarree())

    gl = ax.gridlines(draw_labels=True, alpha=0.35, linewidth=0.6, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator([6, 9, 12, 15])
    gl.ylocator = mticker.FixedLocator([42, 45, 48, 51])
    gl.xlabel_style = {"size": label_size}
    gl.ylabel_style = {"size": label_size}


def add_orography(ax) -> None:
    """Draw orography layer as the map background."""
    plot_orography_map(
        OROG_PATH,
        ax=ax,
        var_name="DEM",
        extent=DOMAIN_EXTENT,
        cmap="Greys",
        levels=30,
        alpha=0.65,
    )


def plot_density_contours(ax, events_df: pd.DataFrame, title: str) -> None:
    """Plot KDE contour map for event locations."""
    add_orography(ax)

    if len(events_df) >= 3:
        lon = events_df["LONGITUDE"].to_numpy()
        lat = events_df["LATITUDE"].to_numpy()

        kde = gaussian_kde(np.vstack([lon, lat]), bw_method=0.2)
        grid_lon = np.linspace(DOMAIN_EXTENT[0], DOMAIN_EXTENT[1], 220)
        grid_lat = np.linspace(DOMAIN_EXTENT[2], DOMAIN_EXTENT[3], 220)
        xx, yy = np.meshgrid(grid_lon, grid_lat)
        zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

        low_level = np.nanpercentile(zz, 45)
        levels = np.linspace(low_level, np.nanmax(zz), 15)
        levels = levels[levels > 0]

        if len(levels) > 1:
            ax.contourf(
                xx,
                yy,
                zz,
                levels=levels,
                cmap="YlOrRd",
                alpha=0.65,
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

    style_geo_axes(ax)
    ax.set_title(title, fontsize=11, fontweight="bold")


def plot_event_points(ax, events_df: pd.DataFrame, title: str, point_size: float = 7.0) -> None:
    """Plot hail/precip points over orography with different colors."""
    add_orography(ax)

    hail = events_df[events_df["TYPE_EVENT"] == "HAIL"]
    precip = events_df[events_df["TYPE_EVENT"] == "PRECIP"]

    if not precip.empty:
        ax.scatter(
            precip["LONGITUDE"],
            precip["LATITUDE"],
            s=point_size,
            c="deepskyblue",
            marker="o",
            alpha=0.7,
            linewidths=0,
            transform=ccrs.PlateCarree(),
            zorder=12,
            label="Precip",
        )

    if not hail.empty:
        ax.scatter(
            hail["LONGITUDE"],
            hail["LATITUDE"],
            s=point_size,
            c="orange",
            marker="o",
            alpha=0.7,
            linewidths=0,
            transform=ccrs.PlateCarree(),
            zorder=13,
            label="Hail",
        )

    style_geo_axes(ax)
    ax.legend(loc="lower left", frameon=True, fontsize=12)
    #ax.set_title(title, fontsize=11, fontweight="bold")


def plot_timestep_events(ax, events_df: pd.DataFrame, ts: pd.Timestamp) -> None:
    """Plot one 15-min slot with larger markers and type-specific marker shapes."""
    add_orography(ax)

    hail = events_df[events_df["TYPE_EVENT"] == "HAIL"]
    precip = events_df[events_df["TYPE_EVENT"] == "PRECIP"]

    if not precip.empty:
        ax.scatter(
            precip["LONGITUDE"],
            precip["LATITUDE"],
            s=120,
            c="deepskyblue",
            marker="o",
            edgecolors="black",
            linewidths=0.5,
            transform=ccrs.PlateCarree(),
            zorder=12,
            label="Precip",
        )

    if not hail.empty:
        ax.scatter(
            hail["LONGITUDE"],
            hail["LATITUDE"],
            s=140,
            c="gold",
            marker="^",
            edgecolors="black",
            linewidths=0.5,
            transform=ccrs.PlateCarree(),
            zorder=13,
            label="Hail",
        )

    style_geo_axes(ax, label_size=12)
    ax.set_title(f"{ts:%Y-%m-%d %H:%M} UTC", fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", frameon=True, fontsize=12)


# ==================================================
# DBSCAN CLUSTER PLOT HELPERS
# ==================================================
def dbscan_haversine(lon: np.ndarray, lat: np.ndarray, radius_km: float) -> np.ndarray:
    """Run DBSCAN in geographic coordinates using haversine distance."""
    if len(lon) == 0:
        return np.array([], dtype=int)

    coords_rad = np.deg2rad(np.column_stack([lat, lon]))  # sklearn expects [lat, lon]
    eps = radius_km / EARTH_RADIUS_KM
    model = DBSCAN(eps=eps, min_samples=3, metric="haversine")
    labels = model.fit_predict(coords_rad)
    return labels


def draw_cluster_radius_circle(ax, lon_c: float, lat_c: float, radius_km: float, color: str) -> None:
    """Draw a geodesic circle around centroid with radius in km."""
    geod = Geodesic()
    circle = geod.circle(lon=lon_c, lat=lat_c, radius=radius_km * 1000.0, n_samples=180)
    ax.plot(
        circle[:, 0],
        circle[:, 1],
        color=color,
        linewidth=1.0,
        alpha=0.9,
        transform=ccrs.PlateCarree(),
        zorder=16,
    )


def plot_dbscan_clusters_for_slot(ax, slot_df: pd.DataFrame, slot_ts: pd.Timestamp) -> None:
    """Plot one 15-min slot with DBSCAN clusters, centroids and radius circles."""
    add_orography(ax)

    if slot_df.empty:
        style_geo_axes(ax, label_size=12)
        ax.set_title(f"{slot_ts:%Y-%m-%d %H:%M} UTC", fontsize=13, fontweight="bold")
        return

    lon = slot_df["LONGITUDE"].to_numpy()
    lat = slot_df["LATITUDE"].to_numpy()
    labels = dbscan_haversine(lon, lat, DBSCAN_RADIUS_KM)
    slot_df = slot_df.copy()
    slot_df["cluster"] = labels

    unique_clusters = sorted([c for c in np.unique(labels) if c >= 0])
    cmap = plt.cm.get_cmap("tab10", max(1, len(unique_clusters)))

    # Noise points
    noise = slot_df[slot_df["cluster"] == -1]
    if not noise.empty:
        for etype, marker in [("PRECIP", "o"), ("HAIL", "^")]:
            part = noise[noise["TYPE_EVENT"] == etype]
            if part.empty:
                continue
            ax.scatter(
                part["LONGITUDE"],
                part["LATITUDE"],
                s=100,
                marker=marker,
                c="lightgray",
                edgecolors="black",
                linewidths=0.4,
                transform=ccrs.PlateCarree(),
                zorder=14,
            )

    # Clustered points, centroids, and 150-km circles
    for i, cid in enumerate(unique_clusters):
        cluster_df = slot_df[slot_df["cluster"] == cid]
        color = cmap(i)

        for etype, marker in [("PRECIP", "o"), ("HAIL", "^")]:
            part = cluster_df[cluster_df["TYPE_EVENT"] == etype]
            if part.empty:
                continue
            ax.scatter(
                part["LONGITUDE"],
                part["LATITUDE"],
                s=115,
                marker=marker,
                c=[color],
                edgecolors="black",
                linewidths=0.4,
                transform=ccrs.PlateCarree(),
                zorder=15,
            )

        lon_c = float(cluster_df["LONGITUDE"].mean())
        lat_c = float(cluster_df["LATITUDE"].mean())

        ax.scatter(
            lon_c,
            lat_c,
            marker="*",
            s=250,
            c=[color],
            edgecolors="black",
            linewidths=0.6,
            transform=ccrs.PlateCarree(),
            zorder=17,
        )
        draw_cluster_radius_circle(ax, lon_c, lat_c, DBSCAN_RADIUS_KM, color)

    style_geo_axes(ax, label_size=12)
    ax.set_title(
        f"{slot_ts:%Y-%m-%d %H:%M} UTC",
        fontsize=13,
        fontweight="bold",
    )


# ==================================================
# OUTPUT TASKS
# ==================================================
def save_density_maps(events: pd.DataFrame, out_dir: Path) -> None:
    """Create and save all/precip/hail density contour maps."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[density] Output dir: {out_dir}")

    sets = [
        ("all_events_density.png", events, "All Events Density Contours"),
        ("precip_events_density.png", events[events["TYPE_EVENT"] == "PRECIP"], "Precip Events Density Contours"),
        ("hail_events_density.png", events[events["TYPE_EVENT"] == "HAIL"], "Hail Events Density Contours"),
    ]

    for fname, df_plot, title in sets:
        print(f"[density] Plotting {fname} with {len(df_plot)} events")
        fig = plt.figure(figsize=(8.2, 6.6))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        plot_density_contours(ax, df_plot, title)
        fig.savefig(out_dir / fname, dpi=250, bbox_inches="tight")
        plt.close(fig)
        print(f"[density] Saved: {out_dir / fname}")


def save_orography_point_maps(events: pd.DataFrame, out_dir: Path) -> None:
    """Save full-data point map on top of orography."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[points] Output dir: {out_dir}")
    print(f"[points] Plotting all-events map with {len(events)} events")

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    plot_event_points(ax, events, "All Events on Orography", point_size=8.0)
    fig.savefig(out_dir / "all_events_points.png", dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"[points] Saved: {out_dir / 'all_events_points.png'}")


def select_busy_slots(events: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.DataFrame, list[pd.Timestamp]]]:
    """
    Select busiest timestamps per busiest day.

    Returns tuples of:
    - peak timestamp for the day
    - day-filtered events dataframe
    - non-empty timestamps within the peak ±5 slot window
    """
    if events.empty:
        return []

    selections: list[tuple[pd.Timestamp, pd.DataFrame, list[pd.Timestamp]]] = []
    day_counts = events.groupby("date").size().sort_values(ascending=False)
    busiest_days = day_counts.head(BUSIEST_DAYS_15MIN).index.tolist()
    busiest_days = sorted(busiest_days)

    for day in busiest_days:
        day_events = events[events["date"] == day].copy()
        slot_counts = day_events.groupby("time_15min").size().sort_values(ascending=False)
        if slot_counts.empty:
            continue

        peak_ts = slot_counts.index[0]
        window_slots = [peak_ts + pd.Timedelta(minutes=15 * i) for i in range(-5, 5)]
        non_empty_counts = (
            day_events[day_events["time_15min"].isin(window_slots)]
            .groupby("time_15min")
            .size()
            .sort_values(ascending=False)
        )
        selected_slots = sorted(non_empty_counts.index.tolist())

        if selected_slots:
            selections.append((peak_ts, day_events, selected_slots))

    return selections


def save_busiest_days_15min_maps(events: pd.DataFrame, out_dir: Path) -> None:
    """For busiest days, plot only selected busy 15-min timestamps."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[busiest-days] Output dir: {out_dir}")

    selections = select_busy_slots(events)
    if not selections:
        print("[busiest-days] No busy slots available, skipping")
        return

    print(f"[busiest-days] Selected {len(selections)} busiest days")

    for peak_ts, day_events, selected_slots in selections:
        day = peak_ts.date()
        day_dir = out_dir / f"day_{day}"
        day_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[busiest-days] Day {day}: peak={peak_ts:%H:%M}, selected busy slots={len(selected_slots)}"
        )

        for ts in selected_slots:
            slot_df = day_events[day_events["time_15min"] == ts].copy()
            print(f"[busiest-days] {ts:%Y-%m-%d %H:%M}: n_events={len(slot_df)}")
            fig = plt.figure(figsize=(7.2, 5.8))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            plot_timestep_events(ax, slot_df, ts)

            fname = f"events_{ts:%Y%m%d_%H%M}.png"
            fig.savefig(day_dir / fname, dpi=230, bbox_inches="tight")
            plt.close(fig)
        print(f"[busiest-days] Completed day {day}")


def save_top10_timeslots_dbscan(events: pd.DataFrame, out_dir: Path) -> None:
    """
    Save events and DBSCAN cluster maps for per-day peak windows.
    
    For each of the 5 busiest days, identify the most-populated 15-min slot,
    then plot 10 surrounding slots (peak ±5 slots):
    - One plot with just events (events-only)
    - One plot with DBSCAN clustering
    
    This reduces computation compared to searching all global time slots.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[dbscan] Output dir: {out_dir}")

    selections = select_busy_slots(events)
    if not selections:
        print("[dbscan] No busy slots available, skipping")
        return

    print(f"[dbscan] Selected {len(selections)} busiest days")

    rank = 0
    for day_idx, (peak_ts, day_events, selected_slots) in enumerate(selections, start=1):
        day = peak_ts.date()
        print(f"[dbscan] Day {day_idx}: {day} with {len(day_events)} events")

        print(
            f"[dbscan] Day {day}: peak slot {peak_ts:%Y-%m-%d %H:%M} | selected busy slots={len(selected_slots)}"
        )

        # Generate two plots per slot in the window: events-only and clustered
        for slot_ts in selected_slots:
            rank += 1
            slot_df = day_events[day_events["time_15min"] == slot_ts].copy()
            n_events = len(slot_df)
            print(f"[dbscan] Rank {rank:02d} slot {slot_ts:%Y-%m-%d %H:%M}: n_events={n_events}")
            
            # Plot 1: Events only
            fig = plt.figure(figsize=(7.2, 5.8))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            plot_timestep_events(ax, slot_df, slot_ts)
            fname_events = f"rank_{rank:02d}_{slot_ts:%Y%m%d_%H%M}_events.png"
            fig.savefig(out_dir / fname_events, dpi=250, bbox_inches="tight")
            plt.close(fig)
            print(f"[dbscan] Saved events: {out_dir / fname_events}")
            
            # Plot 2: DBSCAN clustering
            fig = plt.figure(figsize=(7.2, 5.8))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            plot_dbscan_clusters_for_slot(ax, slot_df, slot_ts)
            fname_dbscan = f"rank_{rank:02d}_{slot_ts:%Y%m%d_%H%M}_dbscan150km.png"
            fig.savefig(out_dir / fname_dbscan, dpi=250, bbox_inches="tight")
            plt.close(fig)
            print(f"[dbscan] Saved clustered: {out_dir / fname_dbscan}")
    
    print(f"[dbscan] Completed: generated {rank * 2} plots (2 per slot) from {len(selections)} busiest days")


def save_global_top10_timeslots_dbscan(events: pd.DataFrame, out_dir: Path) -> None:
    """Save DBSCAN cluster maps for the 10 globally busiest 15-min timestamps."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[global-dbscan] Output dir: {out_dir}")

    if events.empty:
        print("[global-dbscan] No events available, skipping")
        return

    global_counts = events.groupby("time_15min").size().sort_values(ascending=False)
    top_slots = global_counts.head(10)
    if top_slots.empty:
        print("[global-dbscan] No 15-min slots available, skipping")
        return

    print(f"[global-dbscan] Selected {len(top_slots)} global busiest timestamps")

    for rank, (slot_ts, count) in enumerate(top_slots.items(), start=1):
        slot_df = events[events["time_15min"] == slot_ts].copy()
        print(
            f"[global-dbscan] Rank {rank:02d} {slot_ts:%Y-%m-%d %H:%M}: n_events={len(slot_df)} (count={count})"
        )

        fig = plt.figure(figsize=(7.2, 5.8))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        plot_dbscan_clusters_for_slot(ax, slot_df, slot_ts)

        fname = f"global_rank_{rank:02d}_{slot_ts:%Y%m%d_%H%M}_dbscan150km.png"
        fig.savefig(out_dir / fname, dpi=250, bbox_inches="tight")
        plt.close(fig)
        print(f"[global-dbscan] Saved clustered: {out_dir / fname}")

    print(f"[global-dbscan] Completed: generated {len(top_slots)} global top-slot clustering plots")


# ==================================================
# MAIN
# ==================================================
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("Starting events-orography plotting workflow")
    print(f"Output root: {OUTPUT_DIR}")
    print("=" * 70)

    events = load_eswd_events(ESWD_CSV)
    print(f"Loaded filtered events: {len(events)}")
    print(events["TYPE_EVENT"].value_counts(dropna=False))

    if PLOT_DENSITY_MAPS:
        print("[main] Running density maps")
        save_density_maps(events, OUTPUT_DIR / "01_density_maps")
    else:
        print("[main] Skipping density maps")

    if PLOT_OROGRAPHY_DOTS:
        print("[main] Running orography points map")
        save_orography_point_maps(events, OUTPUT_DIR / "02_orography_points")
    else:
        print("[main] Skipping orography points map")

    if PLOT_DAILY_15MIN:
        print("[main] Running busiest-days 15-min maps")
        save_busiest_days_15min_maps(events, OUTPUT_DIR / "03_busiest_days_15min")
    else:
        print("[main] Skipping busiest-days 15-min maps")

    if PLOT_TOP10_DBSCAN:
        print("[main] Running top-10 timeslot DBSCAN maps")
        save_top10_timeslots_dbscan(events, OUTPUT_DIR / "04_top10_timeslots_dbscan")
    else:
        print("[main] Skipping top-10 timeslot DBSCAN maps")

    if PLOT_GLOBAL_TOP10_DBSCAN:
        print("[main] Running global top-10 busiest-timestamp DBSCAN maps")
        save_global_top10_timeslots_dbscan(events, OUTPUT_DIR / "05_global_top10_dbscan")
    else:
        print("[main] Skipping global top-10 busiest-timestamp DBSCAN maps")

    print(f"Saved all outputs under: {OUTPUT_DIR}")
    print("Workflow completed")


if __name__ == "__main__":
    main()
