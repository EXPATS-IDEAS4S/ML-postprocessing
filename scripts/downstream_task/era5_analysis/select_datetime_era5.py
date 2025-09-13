import os
import xarray as xr
import pandas as pd
from glob import glob

# Base directory
base_dir = "/sat_data/crops/2006-2023_4-9_areathresh30_res15min_5frames_gap15min_cropsize75_min5pix_IR108-cm/val"
output_dir = "/home/Daniele/codes/VISSL_postprocessing/scripts/downstream_task"

#if file already exists, open it and skip processing
if os.path.exists(os.path.join(output_dir, "val_unique_hours.csv")):
    print(f"File {os.path.join(output_dir, 'val_unique_hours.csv')} already exists. Skipping processing.")
    #open it with pandas
    df = pd.read_csv(os.path.join(output_dir, "val_unique_hours.csv"))
    
# Find all nc files in the hail and no_hail subfolders
filename_list = sorted(glob(base_dir + "/*/*.nc"))

datetimes = []

for file in filename_list:
    try:
        ds = xr.open_dataset(file, engine="h5netcdf")
        if "time" in ds.variables:
            times = ds["time"].values
            #take all the times and append them to the list
            for t in times:
                t = pd.to_datetime(t)
                datetimes.append(t)
            
            print(f"Extracted times from {file}: {times}")
        else:
            print(f"No 'time' variable found in {file}")
        ds.close()
    except Exception as e:
        print(f"Error reading {file}: {e}")

# Convert to DataFrame
df = pd.DataFrame({"datetime": datetimes})

# Extract date and hour only
df["date"] = df["datetime"].dt.date
df["hour"] = df["datetime"].dt.hour

# Keep unique date-hour pairs
df = df.drop_duplicates(subset=["date", "hour"])

# Sort
df = df.sort_values(by=["date", "hour"]).reset_index(drop=True)

# Save to CSV
csv_path = os.path.join(
    output_dir,
    "val_unique_hours.csv",
)
df.to_csv(csv_path, index=False)

print(f"\nSaved results to {csv_path}")

