import xarray as xr
import pandas as pd
import glob
import os
import numpy as np

# Define folder and file pattern
run_name = 'dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA'
sampling_type = 'closest'
n_div = 8
output_path = f'/data1/fig/{run_name}/{sampling_type}/'

file_pattern = f"percentile_maps_res_{n_div}x{n_div}_label_*.nc"
output_csv = f"{output_path}variable_min_max.csv"

# Find all matching .nc files
file_list = sorted(glob.glob(os.path.join(output_path, file_pattern)))
print(file_list)

# Dictionary to store min/max values for each variable
variable_stats = {}

# Loop through each file
for file_path in file_list:
    print(f"Processing: {file_path}")
    
    # Open NetCDF file
    ds = xr.open_dataset(file_path, decode_times=False, engine="h5netcdf")
    
    # Process each variable
    for var in ds.data_vars:
        var_values = ds[var].values
        var_min = np.nanmin(var_values)  # Get scalar value
        var_max = np.nanmax(var_values)
        
        # Update global min/max for this variable
        if var not in variable_stats:
            variable_stats[var] = {"Min": var_min, "Max": var_max}
        else:
            variable_stats[var]["Min"] = min(variable_stats[var]["Min"], var_min)
            variable_stats[var]["Max"] = max(variable_stats[var]["Max"], var_max)

    # Close dataset
    ds.close()

# Convert dictionary to DataFrame
df = pd.DataFrame.from_dict(variable_stats, orient="index").reset_index()
df.columns = ["Variable", "Min", "Max"]

# Save to CSV
df.to_csv(output_csv, index=False)
print(f"CSV file saved: {output_csv}")
