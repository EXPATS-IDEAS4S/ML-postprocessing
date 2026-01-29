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
