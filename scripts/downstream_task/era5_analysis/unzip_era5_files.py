import os
import zipfile
import xarray as xr
import pandas as pd

# ===============================
# Paths
# ===============================
csv_file = "/home/Daniele/codes/VISSL_postprocessing/scripts/downstream_task/era5_analysis/val_unique_hours.csv"
era5_base_dir = "/sat_data/era5/"
era5_types = ["pressure_levels", "single_level"]
check_contents = False  # If True, open each extracted file and check times

# ===============================
# Load timestamps from CSV
# ===============================
df = pd.read_csv(csv_file, parse_dates=["datetime"])
valid_times = pd.to_datetime(df["datetime"]).dt.floor("H")  # round to hour
valid_times = sorted(valid_times.unique())
print(f"Loaded {len(valid_times)} unique target datetimes")

# ===============================
# Unzip ERA5 .zip file
# ===============================

#loop over the valid times and unzip the corresponding files\
for era5_type in era5_types:
    for time in valid_times:
        #extract year, month, day
        year = time.year
        month = f"{time.month:02d}"
        day = f"{time.day:02d}"
        #add s if not in the era5_type
        if not era5_type.endswith("s"):
            era5_type_name = era5_type + "s"
        else:
            era5_type_name = era5_type
        zip_file = os.path.join(era5_base_dir, f"{era5_type}/{year}/{month}/{day}/{year}-{month}-{day}_{era5_type_name}.zip")
        print(f'unzipping {zip_file}')

        #skip this if on the folder nc files already exist
        if os.path.exists(os.path.join(era5_base_dir, f"{era5_type}/{year}/{month}/{day}/")):
            existing_files = [f for f in os.listdir(os.path.join(era5_base_dir, f"{era5_type}/{year}/{month}/{day}/")) if f.endswith(".nc")]
            if len(existing_files) > 0:
                print(f"Folder {os.path.join(era5_base_dir, f'{era5_type}/{year}/{month}/{day}/')} already contains {len(existing_files)} .nc files. Skipping extraction.")
                continue
        
        try: 
            if era5_type == "single_level" and zipfile.is_zipfile(zip_file):
                with zipfile.ZipFile(zip_file, "r") as z:
                    z.extractall(f"{era5_base_dir}/{era5_type}/{year}/{month}/{day}/")
                    nc_files = [f for f in z.namelist() if f.endswith(".nc")]
                    print(nc_files)
            else:
                print(f"{zip_file} is not a real ZIP → treating as NetCDF directly")
                # Optionally just rename extension (no replacement, copy)
                new_name = zip_file.replace(".zip", ".nc")
                os.rename(zip_file, new_name)
                print(f"Renamed to {new_name}")
                nc_files = [new_name]
        except Exception as e:
            print(f"Error processing {zip_file}: {e}")
            continue
    

        print(f"Extracted {len(nc_files)} NetCDF files:")

        # ===============================
        # Inspect each file
        # ===============================
        if check_contents:
            for nc_file in nc_files:
                nc_path = os.path.join(f"{era5_base_dir}/{era5_type}/{year}/{month}/{day}/", nc_file)
                print(f"\n--- Opening {nc_path} ---")

                ds = xr.open_dataset(nc_path, engine="h5netcdf")
                print(ds)

                file_times = pd.to_datetime(ds["valid_time"].values).floor("H")
                overlap = sorted(set(file_times).intersection(valid_times))

                print(f"Variables: {list(ds.data_vars)}")
                print(f"File times: {file_times} ... total {len(file_times)}")
                print(f"Matching times in CSV: {overlap}")

                ds.close()
            exit()
