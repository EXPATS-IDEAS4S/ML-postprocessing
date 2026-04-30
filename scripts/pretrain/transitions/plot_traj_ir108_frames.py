import io
import os
import re
import sys
from datetime import datetime

import boto3
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import binary_closing


# ==================================================
# CONFIGURATION
# ==================================================
RUN_NAME = "dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc"
OUTPUT_PATH = f"/data1/fig/{RUN_NAME}/epoch_800/test_traj"
PATHWAY_CSV = os.path.join(OUTPUT_PATH, "pathway_analysis/df_pathways_merged_no_dominance.csv")
MIDPOINT_CSV = os.path.join(OUTPUT_PATH, "pathway_analysis/crop_midpoints_from_nc.csv")

OUTPUT_DIR = os.path.join(OUTPUT_PATH, "trajectory_ir108_frames")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ESWD_CSV = os.path.join(OUTPUT_PATH, "eswd-v2-2012-2025_expats.csv")

VAR_NAME = "IR_108"
CMA_VAR_NAME = "cma"
FRAME_SIZE_PX = 100
DOMAIN_EXTENT = [5, 16, 42, 51.5]
OROG_PATH = "/data1/DEM_EXPATS_0.01x0.01.nc"

# Full-domain shading and crop-highlight shading
DOMAIN_ALPHA = 0.18
CROP_ALPHA = 0.95

BT_MIN = 240
BT_MAX = 320

S3_DOMAIN_ROOT = "/data/sat/msg/ml_train_crops/IR_108-WV_062-CMA_FULL_EXPATS_DOMAIN"

PATHWAY_CLASS_MAP = {1: "EC", 2: "DC", 4: "OA"}


def pathway_to_class_label(pathway_value):
    """Convert pathway string like '1->2' into class labels like 'EC->DC'."""
    tokens = [t.strip() for t in str(pathway_value).split("->")]
    class_tokens = []
    for token in tokens:
        try:
            num = int(token)
            class_tokens.append(PATHWAY_CLASS_MAP.get(num, token))
        except ValueError:
            class_tokens.append(token)
    return "->".join(class_tokens)


# ==================================================
# S3 CREDENTIALS
# ==================================================
sys.path.append("/home/Daniele/codes/VISSL_postprocessing/scripts/pretrain/")
from transitions.plot_utils import plot_orography_map  # noqa: E402

sys.path.append("/home/Daniele/codes/VISSL_postprocessing/")
from utils.plotting.class_colors import CLOUD_CLASS_INFO  # noqa: E402

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
    """Extract storm_id and datetime from crop filename."""
    basename = os.path.basename(str(crop_name))
    pattern = (
        r"^storm(?P<storm_id>\d+)_"
        r"(?P<datetime>\d{4}-\d{2}-\d{2}T\d{2}-\d{2})_"
    )
    match = re.match(pattern, basename)
    if match is None:
        return pd.Series(
            {
                "parsed_storm_id": pd.NA,
                "parsed_datetime": pd.NaT,
            }
        )

    dt = pd.to_datetime(
        match.group("datetime"),
        format="%Y-%m-%dT%H-%M",
        errors="coerce",
    )
    return pd.Series(
        {
            "parsed_storm_id": int(match.group("storm_id")),
            "parsed_datetime": dt,
        }
    )


def add_centers_from_midpoint_csv(df, midpoint_csv):
    """Add center_lat/center_lon by merging precomputed midpoint CSV on crop filename."""
    midpoint_df = pd.read_csv(midpoint_csv)
    midpoint_df["crop_key"] = midpoint_df["crop"].astype(str).map(os.path.basename)
    midpoint_df = midpoint_df.drop_duplicates(subset=["crop_key"], keep="last")

    df = df.copy()
    df["crop_key"] = df["crop"].astype(str).map(os.path.basename)
    merged = df.merge(
        midpoint_df[["crop_key", "center_lat", "center_lon", "status"]],
        how="left",
        on="crop_key",
    )
    return merged


def get_crop_bounds_from_center(ds, center_lat, center_lon, frame_size_px=100):
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


def get_frame_color_from_label(label_value):
    """Get frame color using CLOUD_CLASS_INFO dictionary."""
    if pd.isna(label_value):
        return "black"
    try:
        class_id = int(label_value)
    except (TypeError, ValueError):
        return "black"
    return CLOUD_CLASS_INFO.get(class_id, {}).get("color", "black")


def get_frame_linestyle_from_crop_type(crop_type):
    """Map crop_type to frame linestyle."""
    crop_type_str = str(crop_type).strip().lower()
    if crop_type_str == "extrapolated":
        return ":"
    if crop_type_str == "interpolated":
        return "--"
    if crop_type_str == "observed":
        return "-"
    return "-"


def apply_cma_mask_to_ir108(ir108_field, cma_field):
    """Apply 3x3 binary closing to cma and set non-cloud IR108 pixels to 320."""
    cma_binary = np.asarray(cma_field) > 0
    closed_cma = binary_closing(cma_binary, structure=np.ones((3, 3), dtype=bool))

    ir108_masked = np.array(ir108_field, copy=True)
    ir108_masked[~closed_cma] = BT_MAX
    return ir108_masked


def load_eswd_events(eswd_csv):
    """Load ESWD events and normalize types/timestamps for matching."""
    if not os.path.exists(eswd_csv):
        print(f"Warning: ESWD CSV not found: {eswd_csv}. Continuing without event overlays.")
        return pd.DataFrame(columns=["TIME_EVENT", "LATITUDE", "LONGITUDE", "TYPE_EVENT"])

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

    print(f"Loaded {len(events)} ESWD events from {eswd_csv}")
    return events


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
    other_events = events_df[
        (~events_df["TYPE_EVENT"].isin(["HAIL", "PRECIP"]))
    ]

    if not hail_events.empty:
        ax.scatter(
            hail_events["LONGITUDE"],
            hail_events["LATITUDE"],
            marker="^",
            s=70,
            color="yellow",
            edgecolors="black",
            linewidths=0.6,
            transform=ccrs.PlateCarree(),
            zorder=20,
        )

    if not precip_events.empty:
        ax.scatter(
            precip_events["LONGITUDE"],
            precip_events["LATITUDE"],
            marker="s",
            s=55,
            color="lightblue",
            edgecolors="black",
            linewidths=0.6,
            transform=ccrs.PlateCarree(),
            zorder=20,
        )

    if not other_events.empty:
        ax.scatter(
            other_events["LONGITUDE"],
            other_events["LATITUDE"],
            marker="o",
            s=45,
            color="white",
            edgecolors="black",
            linewidths=0.6,
            transform=ccrs.PlateCarree(),
            zorder=20,
        )


def plot_timestamp_frame(
    ds_day,
    ts,
    center_lat,
    center_lon,
    frame_color,
    frame_linestyle,
    events_df,
    output_file,
):
    ds_t = ds_day.sel(time=ts, method="nearest")

    lon = ds_day.lon.values
    lat = ds_day.lat.values

    field_ir108 = ds_t[VAR_NAME].values
    field_cma = ds_t[CMA_VAR_NAME].values
    field = apply_cma_mask_to_ir108(field_ir108, field_cma)

    ix0, ix1, iy0, iy1 = get_crop_bounds_from_center(
        ds_day,
        center_lat=center_lat,
        center_lon=center_lon,
        frame_size_px=FRAME_SIZE_PX,
    )

    # Subset used for highlight overlay
    lon_sub = lon[ix0:ix1]
    lat_sub = lat[iy0:iy1]
    field_sub = field[iy0:iy1, ix0:ix1]
    lon_min, lon_max = float(np.min(lon_sub)), float(np.max(lon_sub))
    lat_min, lat_max = float(np.min(lat_sub)), float(np.max(lat_sub))
    events_frame = filter_events_for_frame(
        events_df,
        frame_start=ts,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
    )

    dx = lon[1] - lon[0]
    dy = lat[1] - lat[0]
    rect_lon0 = lon[ix0]
    rect_lat0 = lat[iy0]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

    plot_orography_map(
        OROG_PATH,
        ax=ax,
        var_name="DEM",
        extent=DOMAIN_EXTENT,
        cmap="Greys",
        levels=30,
        alpha=0.6,
    )

    # Dim full domain
    ax.pcolormesh(
        lon,
        lat,
        field,
        cmap="gray_r",
        vmin=BT_MIN,
        vmax=BT_MAX,
        alpha=DOMAIN_ALPHA,
        transform=ccrs.PlateCarree(),
        shading="auto",
    )

    # Highlight crop area with stronger alpha
    ax.pcolormesh(
        lon_sub,
        lat_sub,
        field_sub,
        cmap="gray_r",
        vmin=BT_MIN,
        vmax=BT_MAX,
        alpha=CROP_ALPHA,
        transform=ccrs.PlateCarree(),
        shading="auto",
    )

    # Red frame around the 100x100 crop
    rect = plt.Rectangle(
        (rect_lon0, rect_lat0),
        FRAME_SIZE_PX * dx,
        FRAME_SIZE_PX * dy,
        linewidth=3.0,
        edgecolor=frame_color,
        linestyle=frame_linestyle,
        facecolor="none",
        transform=ccrs.PlateCarree(),
        zorder=10,
    )
    ax.add_patch(rect)

    plot_eswd_events(ax, events_frame)

    ax.coastlines("50m", linewidth=0.8, color="black")
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="black")

    ax.set_extent(DOMAIN_EXTENT, crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, alpha=0.4, linewidth=0.7, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 11}
    gl.ylabel_style = {"size": 11}

    ax.set_title(ts.strftime("%Y-%m-%d %H:%M UTC"), fontsize=13)
    plt.tight_layout()
    plt.savefig(output_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    print(f"Loading pathway dataframe: {PATHWAY_CSV}")
    pathway_df = pd.read_csv(PATHWAY_CSV, low_memory=False)
    print(pathway_df.columns.to_list())
    exit()
    events_df = load_eswd_events(ESWD_CSV)

    parsed_cols = pathway_df["crop"].apply(parse_crop_metadata)
    pathway_df = pd.concat([pathway_df, parsed_cols], axis=1)
    pathway_df = pathway_df.dropna(
        subset=["parsed_storm_id", "parsed_datetime"]
    )
    pathway_df["parsed_storm_id"] = pathway_df["parsed_storm_id"].astype(int)

    if not os.path.exists(MIDPOINT_CSV):
        raise FileNotFoundError(
            f"Midpoint CSV not found: {MIDPOINT_CSV}. "
            "Run build_crop_midpoint_csv.py first."
        )

    pathway_df = add_centers_from_midpoint_csv(pathway_df, midpoint_csv=MIDPOINT_CSV)
    pathway_df = pathway_df.dropna(subset=["center_lat", "center_lon"])

    # Avoid duplicate plotting for repeated crop references
    pathway_df = pathway_df.drop_duplicates(subset=["crop"])

    print(
        f"Parsed {len(pathway_df)} rows with center coords from crop nc across "
        f"{pathway_df['parsed_storm_id'].nunique()} trajectories"
    )

    s3 = make_s3_client()
    day_cache = {}

    # Count storms per pathway and sort by count (descending)
    storms_per_pathway = (
        pathway_df.groupby("pathway")["parsed_storm_id"]
        .nunique()
        .sort_values(ascending=False)
    )
    sorted_pathways = storms_per_pathway.index.tolist()

    print(f"\nPathways sorted by number of storms (descending):")
    for pw in sorted_pathways:
        print(f"  {pathway_to_class_label(pw)}: {storms_per_pathway[pw]} storms")

    # Iterate through pathways in order of storm count
    for pathway_id in sorted_pathways:
        pathway_data = pathway_df[pathway_df["pathway"] == pathway_id].sort_values(
            "parsed_datetime"
        )
        pathway_label = pathway_to_class_label(pathway_id)
        pathway_out = os.path.join(OUTPUT_DIR, f"pathway_{pathway_id}")
        os.makedirs(pathway_out, exist_ok=True)

        print(
            f"\nProcessing pathway {pathway_label} "
            f"({pathway_data['parsed_storm_id'].nunique()} storms)"
        )

        grouped_by_storm = pathway_data.groupby("parsed_storm_id")

        for storm_id, storm_df in grouped_by_storm:
            storm_out = os.path.join(pathway_out, f"storm_{storm_id}")
            os.makedirs(storm_out, exist_ok=True)

            print(f"  Storm {storm_id} ({len(storm_df)} timestamps)")

            for _, row in storm_df.iterrows():
                ts = row["parsed_datetime"].to_pydatetime()
                center_lat = float(row["center_lat"])
                center_lon = float(row["center_lon"])
                frame_color = get_frame_color_from_label(row.get("label", pd.NA))
                frame_linestyle = get_frame_linestyle_from_crop_type(row.get("crop_type", "observed"))

                day_key = build_daily_key(ts)
                if day_key not in day_cache:
                    try:
                        day_cache[day_key] = read_s3_nc(s3, day_key)
                        print(f"    Loaded: {day_key}")
                    except Exception as exc:
                        print(f"    Skipping {day_key}: {exc}")
                        day_cache[day_key] = None

                ds_day = day_cache[day_key]
                if ds_day is None:
                    continue

                out_file = os.path.join(
                    storm_out,
                    f"ir108_{ts.strftime('%Y-%m-%dT%H-%M')}.png",
                )

                try:
                    plot_timestamp_frame(
                        ds_day=ds_day,
                        ts=ts,
                        center_lat=center_lat,
                        center_lon=center_lon,
                        frame_color=frame_color,
                        frame_linestyle=frame_linestyle,
                        events_df=events_df,
                        output_file=out_file,
                    )
                except Exception as exc:
                    print(
                        f"    Failed storm {storm_id} at "
                        f"{ts.strftime('%Y-%m-%dT%H:%M')}: {exc}"
                    )

    print(f"Done. Outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()


#nohup 900349