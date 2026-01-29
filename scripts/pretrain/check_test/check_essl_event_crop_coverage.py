import os
import pandas as pd
import numpy as np
import xarray as xr
from datetime import timedelta

# ======================================================
# CONFIG
# ======================================================
RUN_NAME = "dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc"

REPORTS_BASE = f"/data1/fig/{RUN_NAME}/epoch_800/test/essl_reports"
CROPS_BASE = (
    "/data1/crops/"
    "test_case_essl_14-15-16-18-19-20-22-23-24_100x100_ir108_cma"
)

EVENT_TYPES = ["HAIL", "PRECIP"]
TIME_TOL = pd.Timedelta("15min")

# ======================================================
# HELPERS
# ======================================================
def parse_nc_filename(fname):
    """
    Extract datetime, lat, lon from filename:
    YYYY-MM-DDTHH:MM_lat_lon.nc
    """
    base = os.path.basename(fname).replace(".nc", "")
    time_str, lat, lon = base.split("_")
    return (
        pd.to_datetime(time_str, utc=True),
        float(lat),
        float(lon),
    )


def load_nc_inventory(event):
    """
    Build a dataframe with all NC crops for an event
    """
    nc_dir = os.path.join(CROPS_BASE, event, "nc", "1")
    records = []

    for f in os.listdir(nc_dir):
        if not f.endswith(".nc"):
            continue
        try:
            ts, lat, lon = parse_nc_filename(f)
            records.append({
                "filename": f,
                "datetime": ts,
                "date": ts.date(),
                "lat_centre": lat,
                "lon_centre": lon,
            })
        except Exception:
            continue

    return nc_dir, pd.DataFrame(records)


def get_crop_bounds(nc_path):
    with xr.open_dataset(nc_path) as ds:
        return (
            ds["lat"].min().item(),
            ds["lat"].max().item(),
            ds["lon"].min().item(),
            ds["lon"].max().item(),
        )

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def distance_outside_crop(lat, lon, lat_min, lat_max, lon_min, lon_max):
    """
    Distance (km) outside crop.
    0 if inside.
    """
    if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
        return 0.0

    dlat = max(lat_min - lat, 0, lat - lat_max)
    dlon = max(lon_min - lon, 0, lon - lon_max)

    # convert degrees to km (approx)
    return np.sqrt((dlat * 111)**2 + (dlon * 111 * np.cos(np.radians(lat)))**2)


# ======================================================
# MAIN
# ======================================================
all_results = []

for event in EVENT_TYPES:
    print(f"\n=== Processing {event} ===")

    # -------------------------------
    # Load events
    # -------------------------------
    df_events = pd.read_csv(
        os.path.join(REPORTS_BASE, f"{event}_grouped.csv")
    )
    df_events["TIME_EVENT"] = pd.to_datetime(
        df_events["TIME_EVENT"], utc=True
    )
    df_events["date"] = df_events["TIME_EVENT"].dt.date

    # -------------------------------
    # Load NC inventory
    # -------------------------------
    nc_dir, df_nc = load_nc_inventory(event)

    print(f"Events: {len(df_events)}")
    print(f"NC files: {len(df_nc)}")

    # -------------------------------
    # Iterate over EVENTS (row-wise)
    # -------------------------------
    for _, ev in df_events.iterrows():
        ev_time = ev["TIME_EVENT"]
        ev_date = ev["date"]
        ev_lat = ev["LATITUDE"]
        ev_lon = ev["LONGITUDE"]

        # 1️⃣ Same-day crops
        nc_day = df_nc[df_nc["date"] == ev_date]

        if nc_day.empty:
            all_results.append({
                "event": event,
                "issue": "missing_day",
                "TIME_EVENT": ev_time,
                "LATITUDE": ev_lat,
                "LONGITUDE": ev_lon,
            })
            continue

        # 2️⃣ Time proximity
        dt_diff = (nc_day["datetime"] - ev_time).abs()
        close_nc = nc_day[dt_diff <= TIME_TOL]

        if close_nc.empty:
            all_results.append({
                "event": event,
                "issue": "missing_time",
                "TIME_EVENT": ev_time,
                "LATITUDE": ev_lat,
                "LONGITUDE": ev_lon,
                "n_nc_same_day": len(nc_day),
            })
            continue

        # 3️⃣ Spatial check (use first matching NC)
        nc_row = close_nc.iloc[0]
        nc_path = os.path.join(nc_dir, nc_row["filename"])

        lat_min, lat_max, lon_min, lon_max = get_crop_bounds(nc_path)

        inside = (
            lat_min <= ev_lat <= lat_max and
            lon_min <= ev_lon <= lon_max
        )

        if not inside:
            all_results.append({
                "event": event,
                "issue": "outside_crop",
                "TIME_EVENT": ev_time,
                "LATITUDE": ev_lat,
                "LONGITUDE": ev_lon,
                "nc_file": nc_row["filename"],
                "lat_min": lat_min,
                "lat_max": lat_max,
                "lon_min": lon_min,
                "lon_max": lon_max,
                "distance_outside_km": distance_outside_crop(
                    ev_lat, ev_lon, lat_min, lat_max, lon_min, lon_max
                ),
                "cluster_id": ev.get("cluster_id", None),
            })
            continue

        # ✅ OK
        all_results.append({
            "event": event,
            "issue": "ok",
            "TIME_EVENT": ev_time,
            "LATITUDE": ev_lat,
            "LONGITUDE": ev_lon,
            "nc_file": nc_row["filename"],
        })

# ======================================================
# OUTPUT
# ======================================================
df_results = pd.DataFrame(all_results)

out_csv = os.path.join(REPORTS_BASE, "event_crop_matching_diagnostics.csv")
df_results.to_csv(out_csv, index=False)

print("\nSaved diagnostics:")
print(out_csv)

# Quick summary
print("\nSummary:")
print(
    df_results
    .groupby(["event", "issue"])
    .size()
    .unstack(fill_value=0)
)

#Plot histogram of distances for outside_crop issues
import matplotlib.pyplot as plt
df_outside = df_results[df_results["issue"] == "outside_crop"]
plt.hist(df_outside["distance_outside_km"], bins=30, edgecolor='black')
plt.title('Histogram of Distances Outside Crop')
plt.xlabel('Distance Outside Crop (km)')
plt.ylabel('Number of Events')
plt.grid(True)
hist_path = os.path.join(REPORTS_BASE, "outside_crop_distance_histogram.png")
plt.savefig(hist_path)
print(f"\nSaved histogram of distances outside crop:")

#Plot the hist of the counts of outside crops per cluster_id (y counts, x the number of outside crops)
plt.figure()
outside_counts = df_outside['cluster_id'].value_counts()
plt.hist(outside_counts, bins=range(1, outside_counts.max()+2), align='left', edgecolor='black')
plt.title('Histogram of Outside Crops per Cluster ID')
plt.xlabel('Number of Outside Crops')
plt.ylabel('Number of Cluster IDs')
plt.grid(True)
hist_cluster_path = os.path.join(REPORTS_BASE, "outside_crops_per_cluster_histogram.png")
plt.savefig(hist_cluster_path)
print(f"\nSaved histogram of outside crops per cluster ID:")
