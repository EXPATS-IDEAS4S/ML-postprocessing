import os
import glob
import xarray as xr
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

def load_era5(
    base_dir,
    era_type="single_level",   # "single_level" or "pressure_level"
    single_level_type=None,  # instant or accum , flag only if era_type is single_level
    start=None,                # datetime or string e.g. "2013-04-01 00:00"
    end=None,                  # datetime or string
    times=None,                # list of datetime objects or strings
    variables=None,             # list of variable names to subset
    extend=None #[latmin, latmax, lonmin, lonmax] to extend the area of interest
):
    """
    Load ERA5 NetCDF files based on filters.
    Returns a merged xarray.Dataset.
    """

    # Normalize inputs
    if isinstance(start, str): start = pd.to_datetime(start)
    if isinstance(end, str): end = pd.to_datetime(end)
    if times is not None:
        times = pd.to_datetime(times)

    filtered_datasets = []

    # Traverse ERA5 folder structure
    if start is not None:
        start_year = start.year
        start_month = start.month
    if end is not None:
        end_year = end.year
        end_month = end.month
    
    for year in sorted(os.listdir(os.path.join(base_dir, era_type))):
        year_dir = os.path.join(base_dir, era_type, year)
        if not os.path.isdir(year_dir): continue

        for month in sorted(os.listdir(year_dir)):
            month_dir = os.path.join(year_dir, month)
            if not os.path.isdir(month_dir): continue

            if start is not None:
                if (int(year) < start_year) or (int(year) == start_year and int(month) < start_month):
                    print(f"Skipping {date}, before start date {start}")
                    continue
            if end is not None:
                if (int(year) > end_year) or (int(year) == end_year and int(month) > end_month):
                    print(f"Skipping {date}, after end date {end}")
                    
                    if filtered_datasets:
                        combined_ds = xr.concat(filtered_datasets, dim="valid_time")
                        return combined_ds
                    else:
                        print("No data matched the criteria.")
                        return None

            for day in sorted(os.listdir(month_dir)):
                day_dir = os.path.join(month_dir, day)
                if not os.path.isdir(day_dir): continue
                date = f"{year}-{month}-{day}"
                print(f"Processing date: {date}")
                # skip data out the range, if given
                     
                # Find .nc files
                if single_level_type and era_type == "single_level":
                    files = glob.glob(os.path.join(day_dir, f"*{single_level_type}.nc"))
                else:
                    files = glob.glob(os.path.join(day_dir, "*.nc"))

                for f in files:
                    print(f"Processing {f}...")
                    try:
                        ds = xr.open_dataset(f, engine="h5netcdf")
                        
                        #filter by variables if specified
                        if variables is not None:
                            missing_vars = [var for var in variables if var not in ds.variables]
                            if missing_vars:
                                print(f"Skipping {f}, missing variables: {missing_vars}")
                                continue
                            ds = ds[variables]

                        if extend is not None:
                            latmin, latmax, lonmin, lonmax = extend
                            ds = ds.sel(latitude=slice(latmax, latmin), longitude=slice(lonmin, lonmax))

                        valid_times = pd.to_datetime(ds["valid_time"].values)
                        # Boolean mask for filtering
                        mask = np.ones(len(valid_times), dtype=bool)

                        if times is not None:
                            mask &= np.isin(valid_times, times)
                        if start is not None:
                            mask &= valid_times >= start
                        if end is not None:
                            mask &= valid_times <= end

                        if mask.any():
                            # Select only matching timestamps
                            filtered_ds = ds.sel(valid_time=ds["valid_time"][mask])
                            filtered_datasets.append(filtered_ds)
                            print(f"Included {f}, {mask.sum()} timestamps matched")
                        else:
                            print(f"Skipped {f}, no timestamps matched")
                            
                    except Exception as e:
                        print(f"Error reading {f}: {e}")

                # merge all selected datasets
                if filtered_datasets:
                    combined_ds = xr.concat(filtered_datasets, dim="valid_time")
                    #print(combined_ds)

    
    return combined_ds


# ====================
# Example usage
# ====================
if __name__ == "__main__":
    base_dir = "/sat_data/era5"
    output_dir = "/data1/fig/supervised_ir108-cm_75x75_5frames_12k_nc_r2dplus1/era5_analysis"
    os.makedirs(output_dir, exist_ok=True)

    ml_mode = "supervised"  # "supervised" or "selfsupervised"
    dataset = "train"  # "train" or "val"

    #path to csv file with the predictions
    pred_csv_path = "/data1/runs/supervised_ir108-cm_75x75_5frames_12k_nc_r2dplus1/features/epoch_50"
    filename = "val_filepaths_true_predicted_labels_features.csv"

    #Open the csv file with pandas
    df = pd.read_csv(os.path.join(pred_csv_path, filename))
    print(df.columns)
    #change column 'true_label (0=hail; 1=no_hail) to 'true_label' and 'predicted_label (0=hail; 1=no_hail)' to 'predicted_label'
    df = df.rename(columns={'true_label (0=hail; 1=no_hail)': 'true_label', 'predicted_label (0=hail; 1=no_hail)': 'predicted_label'})
    #print(df.head())

    # 1. Both 0 (true hail, predicted hail) -> True Positive
    df_true0_pred0 = df[(df['true_label'] == 0) & (df['predicted_label'] == 0)]
    print(f"True Positives: {len(df_true0_pred0)}")

    # 2. Both 1 (true no_hail, predicted no_hail) -> True Negative
    df_true1_pred1 = df[(df['true_label'] == 1) & (df['predicted_label'] == 1)]
    print(f"True Negatives: {len(df_true1_pred1)}")

    # 3. True 0, Predicted 1 (true hail, predicted no hail) -> False Negative
    df_true0_pred1 = df[(df['true_label'] == 0) & (df['predicted_label'] == 1)]
    print(f"False Negatives: {len(df_true0_pred1)}")

    # 4. True 1, Predicted 0 (true no_hail, predicted hail)  -> False Positive
    df_true1_pred0 = df[(df['true_label'] == 1) & (df['predicted_label'] == 0)]
    print(f"False Positives: {len(df_true1_pred0)}")
    
    #append them in a list to loop over
    df_list = [df_true0_pred0, df_true1_pred1, df_true0_pred1, df_true1_pred0]
    category_names = ["True Positive (hail)", "True Negative (no_hail)", "False Negative (missed hail)", "False Positive (false alarm)"]

    
    for i, (df_cat, cat_name) in enumerate(zip(df_list, category_names)):
        print(f"Analyzing {cat_name}...")

        print(df_cat.head())
        exit()

        # Load single-level data for all 2013
        ds = load_era5(
            base_dir,
            era_type="single_level",
            single_level_type="instant", # "instant" or "accum"
            start="2013-04-01",
            end="2013-09-30",
            variables=["cape"]
        )

        print(ds)
        #print(ds.variables)

        # plot distribution of CAPE values
        
        plt.hist(ds["cape"].values.flatten(), bins=100, range=(0, 5000))
