import os
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys

sys.path.append("/home/Daniele/codes/VISSL_postprocessing")
from scripts.downstream_task.era5_analysis.era5_load_utils import load_era5
from scripts.downstream_task.era5_analysis.era5_plot_utils import (
    plot_distribution,
    plot_grouped_distribution,
    plot_vertical_profile,
)

# ====================
# Example usage
# ====================
if __name__ == "__main__":
    base_dir = "/sat_data/era5"
    output_dir = "/data1/fig/supervised_ir108-cm_75x75_5frames_12k_nc_r2dplus1/era5_analysis"
    os.makedirs(output_dir, exist_ok=True)
    config_path = "/home/Daniele/codes/VISSL_postprocessing/configs/era5_vars.yaml"

    flatten = False  # if True → flatten variable arrays before analysis
    per_frame_mode = False  # if True → group distributions by frames
    vertical_profile = False  # if True → plot vertical profile
    n_frame = 8
    ml_mode = "supervised"
    dataset = "val"
    variable_to_analyze = "cape"  

    # --------------------
    # Load variable config
    # --------------------
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    var_cfg = config["era5_vars"].get(variable_to_analyze)
    if var_cfg is None:
        raise ValueError(f"Variable {variable_to_analyze} not found in config.")

    era_type = var_cfg.get("era5_type")
    single_level_type = var_cfg.get("single_level_type", None)
    print(
        f"Variable {variable_to_analyze} → era_type={era_type}, single_level_type={single_level_type}, "
        f"unit={var_cfg['unit']}, vmin={var_cfg['vmin']}, vmax={var_cfg['vmax']}"
    )

    # --------------------
    # Load predictions CSV
    # --------------------
    pred_csv_path = "/data1/runs/supervised_ir108-cm_75x75_5frames_12k_nc_r2dplus1/features/epoch_50"
    filename = "val_filepaths_true_predicted_labels_features.csv"
    df = pd.read_csv(os.path.join(pred_csv_path, filename))
    df = df.rename(
        columns={
            "true_label (0=hail; 1=no_hail)": "true_label",
            "predicted_label (0=hail; 1=no_hail)": "predicted_label",
        }
    )

    # Split into categories
    df_true0_pred0 = df[(df["true_label"] == 0) & (df["predicted_label"] == 0)]
    df_true1_pred1 = df[(df["true_label"] == 1) & (df["predicted_label"] == 1)]
    df_true0_pred1 = df[(df["true_label"] == 0) & (df["predicted_label"] == 1)]
    df_true1_pred0 = df[(df["true_label"] == 1) & (df["predicted_label"] == 0)]

    df_list = [df_true0_pred0, df_true1_pred1, df_true0_pred1, df_true1_pred0]
    category_names = [
        "True Positive (hail)",
        "True Negative (no_hail)",
        "False Negative (missed hail)",
        "False Positive (false alarm)",
    ]

    # --------------------
    # Loop over categories
    # --------------------
    for df_cat, cat_name in zip(df_list, category_names):
        if cat_name == "False Negative (missed hail)":
            print(f"Analyzing {cat_name}...")

            extracted_vars = []

            for _, row in df_cat.iterrows():
                ds = xr.open_dataset(row["file_path"], engine="h5netcdf")
                times = pd.to_datetime(ds["time"].values)
                start_date = times.min().strftime("%Y-%m-%d")
                end_date = (times.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                extend = [ds["lat"].values.min(), ds["lat"].values.max(),
                        ds["lon"].values.min(), ds["lon"].values.max()]

                ds = load_era5(
                    base_dir,
                    era_type=era_type,
                    single_level_type=single_level_type,
                    start=start_date,
                    end=end_date,
                    times=times,
                    variables=[variable_to_analyze],
                    extend=extend,
                )
                if ds is None:
                    continue

                if flatten and vertical_profile is False and per_frame_mode is False:
                    values = ds[variable_to_analyze].values.flatten()
                    if values.size > 0:
                        extracted_vars.append(values)
                else:
                    extracted_vars.append(ds[variable_to_analyze].values)

            if not extracted_vars:
                print(f"No data extracted for {cat_name}. Skipping.")
                continue

            all_values = np.concatenate(extracted_vars)
            print(all_values)
            exit()
            print(f"Total {len(all_values)} values for {variable_to_analyze} in {cat_name}")

            # --------------------
            # Flexible plotting
            # --------------------
            plt_path = os.path.join(
                output_dir,
                f"{ml_mode}_{dataset}_{cat_name.replace(' ', '_').replace('(', '').replace(')', '')}_{variable_to_analyze}.png",
            )

            if vertical_profile and era_type == "pressure_levels":
                # Build dict by level
                data_by_level = {
                    lvl: all_values[:, i] if all_values.ndim == 2 else all_values
                    for i, lvl in enumerate(config["pressure_levels"])
                }
                plot_vertical_profile(data_by_level, var_cfg, config["pressure_levels"], title=f"{cat_name} - {variable_to_analyze}")
                #add to filename to save the vertical_profile flag
                plt_path_vp = plt_path.replace(".png", "_vertical_profile.png")
                plt.savefig(plt_path_vp)

            elif per_frame_mode:
                plot_grouped_distribution(all_values, var_cfg, n_frame=n_frame, title=f"{cat_name} - {variable_to_analyze}")
                #add to filename to save the per_frame_mode flag
                plt_path_pf = plt_path.replace(".png", f"_per_{n_frame}frame.png")
                plt.savefig(plt_path_pf)

            else:
                plot_distribution(all_values, var_cfg, title=f"{cat_name} - {variable_to_analyze}")
                #add to filename to save the flatten flag
                plt_path_flat = plt_path.replace(".png", "_flatten.png")
                plt.savefig(plt_path_flat)

            plt.close()
            print(f"Saved plot to {plt_path}")

