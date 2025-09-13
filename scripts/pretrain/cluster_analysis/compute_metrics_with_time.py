"""
Compute time metrics from the per frame statistics

"""

import os
import pandas as pd
import logging
import sys
import ast
import re
import numpy as np

sys.path.append(os.path.abspath("/home/Daniele/codes/VISSL_postprocessing"))
from utils.configs import load_config
from utils.processing.features_utils import get_num_crop
from utils.processing.time_metrics import compute_mean_curves


def load_parsed_dataframe(csv_path: str) -> pd.DataFrame:
    df_raw = pd.read_csv(csv_path)
    dict_like = re.compile(r"^\s*\{.*\}\s*$")
    flat_records = []

    metadata_cols = ["crop_index", "path", "label", "distance"]

    for _, row in df_raw.iterrows():
        metadata = {col: row[col] for col in metadata_cols}

        for col_name in df_raw.columns:
            if col_name in metadata_cols:
                continue

            val = row[col_name]
            if pd.isna(val):
                continue

            # If already a dict, use it
            if isinstance(val, dict):
                val_dict = val
            # If looks like a dict string, parse safely
            elif isinstance(val, str) and dict_like.match(val):
                try:
                    val_dict = ast.literal_eval(val)
                    if not isinstance(val_dict, dict):
                        continue
                except Exception:
                    # Skip malformed entries
                    continue
            else:
                # Wrap numeric / string values in a dict
                try:
                    val_dict = {"value": float(val)}
                except Exception:
                    continue  # skip anything else

            # Merge with metadata
            val_dict.update(metadata)
            flat_records.append(val_dict)

    flat_df = pd.DataFrame(flat_records)
    # Optional: rename percentile columns if present
    flat_df = flat_df.rename(columns={25: "p25", 50: "p50", 75: "p75", 99: "p99"})
    return flat_df


import matplotlib.pyplot as plt
import seaborn as sns

def plot_mean_curves(curves: pd.DataFrame, stat="p50"):
    """
    Plot mean evolution over frames for each variable and label.
    """
    g = sns.FacetGrid(curves, col="var", hue="label", sharey=False, col_wrap=3, height=3)
    g.map(sns.lineplot, "frame", stat)
    g.add_legend()
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle(f"Evolution of {stat} over frames per class/variable")
    plt.show()




# Load configuration
config_path = "/home/Daniele/codes/VISSL_postprocessing/configs/process_run_config.yaml"
var_config_path = "/home/Daniele/codes/VISSL_postprocessing/configs/variables_metadata.yaml"
config = load_config(config_path)
var_config = load_config(var_config_path)

# --- Configure logging ---
log_level = config["logging"]["log_level"]
logs_path = config["logging"]["logs_path"]
logging.basicConfig(
    level=log_level,  
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(), 
        logging.FileHandler(os.path.join(logs_path, "processing_crops_stats_per_frame.log"), mode="w")
    ]
)
logger = logging.getLogger(__name__)

# --- Extract experiment settings ---
run_name = config["experiment"]["run_names"][0]
epoch = config["experiment"]["epoch"]

# Data configuration
crops_name = config["data"]["crops_name"]
data_base_path = config["data"]["data_base_path"]
file_extension = config["data"]["file_extension"]
sampling_type = config["data"].get("sampling_type", "all")
filter_daytime = config["data"]["filter_daytime"]
filter_imerg_minutes = config["data"]["filter_imerg"]
n_frames = config["data"]["n_frames"]
data_format = config["data"]["file_extension"]
output_root = config["experiment"]["path_out"]


# Stats configuration
use_parallel = config["statistics"]["use_parallel"]
n_cores = config["statistics"]["n_cores"]
percentiles = config["statistics"]["spatial"]["percentiles"]
sel_vars = config["statistics"]["spatial"]["sel_vars"]


# Build paths
image_crops_path = f"{data_base_path}/{crops_name}/{data_format}/1/"
output_path = f"{output_root}/{run_name}/epoch_{epoch}/{sampling_type}/"

# Load crop list
n_samples = get_num_crop(image_crops_path, extension=data_format)
n_subsample = n_samples if sampling_type == "all" else config["data"]["n_subsample"]

# --- Build output filename ---
stats_str = "-".join(map(str, percentiles))
vars_str = "-".join(sel_vars)

time_flag = "timedim" if config["statistics"].get("time_dimension", False) else None
geo_flag = "coords-datetime" if config["statistics"].get("include_geotime", False) else None

# Construct filter tags for filename
filter_tags = []
if filter_daytime:
    filter_tags.append("daytime")
if filter_imerg_minutes:
    filter_tags.append("imergmin")
filter_suffix = "_" + "_".join(filter_tags) if filter_tags else ""

parts = [
    "crops_stats",
    f"vars-{vars_str}",
    f"stats-{stats_str}",
    f"frames-{n_frames}",
    time_flag,
    geo_flag,
    run_name,
    sampling_type,
    str(n_subsample) + filter_suffix,  # assumes these are defined elsewhere
]
filename = "_".join([p for p in parts if p]) + ".csv"
output_file = os.path.join(output_path, filename)

csv_filename = f"{output_path}crop_list_{run_name}_{sampling_type}_{n_subsample}{filter_suffix}.csv"
df_labels = pd.read_csv(csv_filename)
#print(df_labels)
# import xarray as xr
# ds = xr.open_dataset("/sat_data/crops/2006-2023_4-9_areathresh30_res15min_5frames_gap15min_cropsize75_min5pix_IR108-cm/val/hail/20130519_0530_res15min_5frames_cropsize75_2_hail_initiation_graupel.nc", engine="h5netcdf")
# print(ds)
# exit()

# --- Open dataframe ---
try:
    #df = load_parsed_dataframe(output_file)
    df = pd.read_csv(output_file)
    logger.info(f"Loaded dataframe from {output_file} with shape {df.shape}")
    #print the first row completely
    print(df)
except FileNotFoundError:
    logger.error(f"File not found: {output_file}")
    df = pd.DataFrame()  # empty fallback




curves = compute_mean_curves(df, stat="50")
print(curves)


#plot_mean_curves(curves, stat="p50")