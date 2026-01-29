"""
Add precipitation area metric to merged crop CSV.

precip_area =
    (# pixels with rain_rate > 0.1)
    --------------------------------
    (# pixels with BT < 320 K)

Stored as:
- column 'None'  -> precip_area
- columns 25,50,75,99 -> None
"""

import os
import io
import re
import sys
import pandas as pd
import numpy as np
import xarray as xr
import boto3
from botocore.exceptions import ClientError

# =====================
# CONFIG
# =====================
RUN_NAME = "dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc"
epoch = "epoch_800"
crop_sel = "all"

BT_CLEAR_SKY = 320.0
RAIN_THRESHOLD = 0.1
MIN_PIXEL_COUNT = 100
PERCENTILE_COLS = ["25", "50", "75", "99"]
S3_BUCKET_NAME = "expats-imerg-prec"
bucket_filename_prefix = "IMERG_daily_"
bucket_filename_suffix = ".nc"

path_to_dir = f"/data1/fig/{RUN_NAME}/{epoch}/{crop_sel}/"

csv_path = os.path.join(
    path_to_dir,
    "merged_crops_stats_all_cvc.csv"
)

output_path = os.path.join(
    path_to_dir,
    "merged_crops_stats_all_cvc_area_precip.csv"
)

# =====================
# S3 SETUP
# =====================
sys.path.append("/home/Daniele/codes/VISSL_postprocessing/utils/buckets/")
from credentials_buckets import (
    S3_ACCESS_KEY,
    S3_SECRET_ACCESS_KEY,
    S3_ENDPOINT_URL,
)

s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT_URL,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_ACCESS_KEY,
)

# =====================
# HELPERS
# =====================
def read_s3_nc(key):
    try:
        obj = s3.get_object(Bucket=S3_BUCKET_NAME, Key=key)
        return xr.open_dataset(io.BytesIO(obj["Body"].read()), engine="h5netcdf")
    except ClientError as e:
        print(f"❌ Failed to read {key}: {e}")
        return None


def compute_precip_area(ds_bt, ds_rain):
    """
    Compute precipitation area for a crop dataset.
    """
    bt = ds_bt["IR_108"]
    rain = ds_rain["precipitation"]

    cloudy_mask = bt < BT_CLEAR_SKY
    cloudy_pixels = cloudy_mask.sum().item()

    if cloudy_pixels <= MIN_PIXEL_COUNT:
        return 0.0

    rainy_pixels = ((rain >= RAIN_THRESHOLD) & cloudy_mask).sum().item()
    if rainy_pixels <= MIN_PIXEL_COUNT:
        return 0.0
    else: 
        return rainy_pixels / cloudy_pixels


# =====================
# LOAD CSV
# =====================
df = pd.read_csv(csv_path)
print(f"✅ Loaded CSV: {df.shape}")

crop_indices = df["crop_index"].unique()
print(f"Found {len(crop_indices)} unique crops")

new_rows = []

# =====================
# MAIN LOOP
# =====================
for crop_idx in crop_indices:

    row = df[df["crop_index"] == crop_idx].iloc[0]
    nc_key = row["path"]

    print(f"\n▶ Crop {crop_idx}")
    print(f"  File: {nc_key}")

    ds_bt = xr.open_dataset(row["path"], engine="h5netcdf")
    #print(ds_bt)
    
    if ds_bt is None:
        print("  ❌ Failed to open BT dataset, skipping")
        continue
    
    timestamp = ds_bt.time.values[0]
    timestamp = pd.to_datetime(str(timestamp))
    y = int(timestamp.year)
    m = int(timestamp.month)
    d = int(timestamp.day)
    print(f"  Timestamp: {timestamp}")
    print(f"  Date: {y}-{m:02d}-{d:02d}")
    
    lat = ds_bt["lat"].values
    lon = ds_bt["lon"].values

    bucket_filename = (
                f"{bucket_filename_prefix}{y:04d}-{m:02d}-{d:02d}"
                f"{bucket_filename_suffix}"
            )

    # Load variable and CMA
    ds_rain = read_s3_nc(bucket_filename)

    #get only the timestamp matching
    if ds_rain is None:
        print("  ❌ Failed to open rain dataset, skipping")
        continue
    ds_rain = ds_rain.sel(time=timestamp, method="nearest")
    
    # slice to crop area
    ds_rain = ds_rain.sel(
        lat=slice(lat.min(), lat.max()),
        lon=slice(lon.min(), lon.max())
    )

    #print(ds_rain)
    
    precip_area = compute_precip_area(ds_bt, ds_rain)
    print(f"  precip_area = {precip_area}")

    new_row = row.copy()
    new_row["var"] = "precip_area"
    new_row["None"] = precip_area

    for p in PERCENTILE_COLS:
        new_row[p] = None

    new_rows.append(new_row)

# =====================
# SAVE UPDATED CSV
# =====================
if new_rows:
    df_out = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df_out.to_csv(output_path, index=False)
    print(f"\n✅ Saved updated CSV with precip_area → {output_path}")
else:
    print("\n⚠️ No new rows added")

#2267428







