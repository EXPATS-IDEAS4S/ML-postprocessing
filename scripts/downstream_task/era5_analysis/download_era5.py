#!/usr/bin/env python
import os
import pandas as pd
import cdsapi

# ==========================
# User settings
# ==========================
csv_path = "/home/Daniele/codes/VISSL_postprocessing/scripts/downstream_task/val_unique_hours.csv"
output_base = "/sat_data/era5"

area = [52, 5, 42, 16]  # [North, West, South, East]
# ==========================

# Load datetimes from CSV
df = pd.read_csv(csv_path, parse_dates=["datetime"])
df["date"] = df["datetime"].dt.strftime("%Y-%m-%d")
df["hour"] = df["datetime"].dt.strftime("%H:00")

# Get unique date-hour combinations
df = df.drop_duplicates(subset=["date", "hour"]).reset_index(drop=True)
print(f"Found {len(df)} unique date-hour pairs")
print(df)


c = cdsapi.Client()

# --------------------------
# ERA5 variable names
# --------------------------
single_level_vars = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_dewpoint_temperature",
    "2m_temperature",
    "total_column_water",
    "total_column_water_vapour",
    "convective_available_potential_energy",
    "convective_inhibition",
    "k_index",
    "mean_sea_level_pressure",
    "total_precipitation",
    "total_cloud_cover",
    "convective_precipitation"
]

pressure_level_vars = [
    "specific_humidity",
    "relative_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "temperature",
    "potential_vorticity",
    "divergence"
]

pressure_levels = [
   "300", "400", "500", "550", "600", "650", "700", "750", "800", "825", "850", "875", "900", "925", "950", "975", "1000"
]

# --------------------------
# Loop over dates
# --------------------------
for date, group in df.groupby("date"):
    year, month, day = date.split("-")
    print(f"\nProcessing date: {date}")

    # Collect all hours for this date
    times = group["hour"].unique().tolist()
    print(times)

    # =====================
    # Single-level request
    # =====================
    out_dir_single = os.path.join(output_base, "single_level", year, month, day)
    os.makedirs(out_dir_single, exist_ok=True)

    single_file = os.path.join(out_dir_single, f"{date}_single_levels.zip")
    if not os.path.exists(single_file):
        print(f"Downloading single-level data for {date} ({times})...")
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": ["reanalysis"],
                "variable": single_level_vars,
                "year": year,
                "month": month,
                "day": day,
                "time": times,
                "data_format": "netcdf",
                "download_format": "unarchived",
                "area": area,
            },
            single_file,
        )
    else:
        print(f"Already exists: {single_file}")

    # =====================
    # Pressure-level request
    # =====================
    out_dir_pl = os.path.join(output_base, "pressure_levels", year, month, day)
    os.makedirs(out_dir_pl, exist_ok=True)

    pl_file = os.path.join(out_dir_pl, f"{date}_pressure_levels.zip")
    if not os.path.exists(pl_file):
        print(f"Downloading pressure-level data for {date} ({times})...")
        c.retrieve(
            "reanalysis-era5-pressure-levels",
            {
                "product_type": ["reanalysis"],
                "variable": pressure_level_vars,
                "pressure_level": pressure_levels,
                "year": year,
                "month": month,
                "day": day,
                "time": times,
                "data_format": "netcdf",
                "download_format":"unarchived",
                "area": area,
            },
            pl_file,
        )
    else:
        print(f"Already exists: {pl_file}")

#798003