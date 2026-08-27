"""Convert NetCDF crop files to PNG images.

Usage
-----
Run with the default configuration::

    python 2_make_img_from_nc.py

Use a different YAML configuration::

    python 2_make_img_from_nc.py --config /path/to/process_run_config.yaml

The default configuration is ``model_eval_paper/process_run_config.yaml``.
The ``visualization.datasets`` list controls one or more conversion jobs. Each
job specifies its input/output directories, variable, file limit, colormap,
and image value range.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_closing
import matplotlib as mpl

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.configs import load_config


logger = logging.getLogger(__name__)


def configure_logging(config):
    """Configure console and file logging for the image conversion run."""
    logging_config = config.get("logging", {})
    level_name = str(logging_config.get("log_level", "INFO")).upper()
    log_level = getattr(logging, level_name, logging.INFO)
    log_dir = Path(logging_config.get("logs_path", REPO_ROOT / "logs"))
    if not log_dir.is_absolute():
        log_dir = REPO_ROOT / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    log_filename = logging_config.get(
        "visualization_log_filename", "nc_to_images.log"
    )
    log_path = log_dir / log_filename
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    logger.setLevel(log_level)
    logger.handlers.clear()
    logger.propagate = False

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(
        log_path, mode=logging_config.get("visualization_log_mode", "a")
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("Logging to %s", log_path)


def create_WV_IR_diff_colormap(vmin, center, vmax, diverg_cmap=mpl.cm.seismic):
    if vmin is None:
        vmin = -1
    if vmax is None:
        vmax = 1
    # get number of colors above and below center point representing the respective range percentages
    n_pos = int(265*(vmax-center)/(vmax-vmin)) if vmax > center else 1
    n_neg = int(265*(center-vmin)/(vmax-vmin)) if vmin < center else 1
    # sample colors
    colors_pos = diverg_cmap(np.linspace(0.7, 1, n_pos))
    colors_neg = diverg_cmap(np.linspace(0, 0.5, n_neg))
    # combine them and build a new colormap
    colors = np.vstack((colors_neg, colors_pos))
    
    return mpl.colors.LinearSegmentedColormap.from_list('recentered_cmap', colors)


def normalize_data(data, vmin=None, vmax=None):
    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)
    return np.clip((data - vmin) / (vmax - vmin), 0, 1)

def save_image(data, out_path, cmap='viridis', vmin=None, vmax=None):
    plt.imsave(out_path, data, cmap=cmap, vmax=vmax, vmin=vmin)
    logger.info("Saved image: %s", out_path)


def _time_values(ds):
    """Return time values as a 1D iterable even when the dataset stores a scalar timestamp."""
    if "time" not in ds:
        return []
    return np.atleast_1d(ds.time.values)


def _select_time_slice(data_array, timestamp):
    """Select a time slice when the array is time-indexed, otherwise return the squeezed data."""
    if "time" in data_array.dims:
        return data_array.sel(time=timestamp).squeeze().values
    return data_array.squeeze().values

def is_combined_channel(var_name):
    return "-" in var_name

def combine_channels(ds, var_names, time_index, method='subtract'):
    """Load and combine channels from a dataset using the given method."""
    if not all(var in ds for var in var_names):
        raise ValueError(f"One or more variables not found: {var_names}")

    data = []
    for var in var_names:
        img = ds[var].sel(time=ds.time[time_index]).squeeze().values
        data.append(img)

    if method == 'subtract':
        return data[0] - data[1]
    elif method == 'average':
        return np.mean(data, axis=0)
    else:
        raise ValueError(f"Unknown combine method: {method}")

def convert_nc_to_images(
    input_dir,
    output_dir,
    var_name="IR_108",
    cmap="plasma",
    vmin=None,
    vmax=None,
    max_files=None,
):
    os.makedirs(output_dir, exist_ok=True)
    nc_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".nc")])
    if max_files is not None:
        nc_files = nc_files[:max_files]
    logger.info("Found %d .nc files in %s", len(nc_files), input_dir)

    #is_combo = is_combined_channel(var_name)
    #var_list = var_name.split('-') if is_combo else [var_name]
    var_list = [var_name]
    logger.info("Processing variables: %s", var_list)

    for nc_file in nc_files:
        logger.info("Processing NetCDF file: %s", nc_file)
        nc_path = os.path.join(input_dir, nc_file)
        try:
            ds = xr.open_dataset(nc_path)#, engine='h5netcdf')
            time_values = _time_values(ds)
            logger.info("Found %d time step(s) in %s", len(time_values), nc_file)


            # Check all required variables exist
            if not all(var in ds for var in var_list):
                logger.warning(
                    "Missing required variable(s) in %s; skipping file", nc_file
                )
                continue

            for i, t in enumerate(time_values):
                logger.info("Processing %s time step %d: %s", nc_file, i, t)

                # if is_combo:
                #     img = combine_channels(ds, var_list, i, method='subtract')
                # else:
                img = _select_time_slice(ds[var_name], t)

                # Apply cloud mask if available
                if 'cma' in ds and 'IR_108' in var_name:
                    cma = _select_time_slice(ds['cma'], t)
                    cma = binary_closing(cma, structure=np.ones((3, 3), dtype=bool))
                    img[cma == 0] = vmax

                timestamp = str(np.datetime_as_string(t, unit='m')).replace(':', '-')
                out_name = f"{os.path.splitext(nc_file)[0]}_t{i}_{timestamp}.png"
                out_path = os.path.join(output_dir, out_name)
                save_image(img, out_path, cmap=cmap, vmin=vmin, vmax=vmax)

        except Exception as e:
            logger.exception("Error processing %s: %s", nc_file, e)


def convert_datasets(visualization_config):
    """Run all NetCDF-to-image jobs defined in the visualization config."""
    datasets = visualization_config.get("datasets", [])
    if not datasets:
        raise ValueError("visualization.datasets must contain at least one dataset")

    for dataset in datasets:
        name = dataset.get("name", dataset["input_dir"])
        logger.info("Starting dataset conversion: %s", name)
        started = time.perf_counter()
        convert_nc_to_images(
            input_dir=dataset["input_dir"],
            output_dir=dataset["output_dir"],
            var_name=dataset.get("var_name", "IR_108"),
            cmap=dataset.get("cmap", "plasma"),
            vmin=dataset.get("vmin"),
            vmax=dataset.get("vmax"),
            max_files=dataset.get("max_files"),
        )
        logger.info(
            "Finished dataset conversion: %s (%.1f seconds)",
            name,
            time.perf_counter() - started,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    config_path = args.config or Path(
        os.getenv(
            "PROCESS_RUN_CONFIG",
            str(REPO_ROOT / "model_eval_paper" / "process_run_config.yaml"),
        )
    )
    config = load_config(config_path)
    configure_logging(config)
    logger.info("Using configuration: %s", config_path)
    convert_datasets(config["visualization"])
    logger.info("Image conversion run completed")

#3112361