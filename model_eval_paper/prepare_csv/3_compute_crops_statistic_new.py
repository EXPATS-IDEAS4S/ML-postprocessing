"""Add crop statistics to the t-SNE feature CSV.

The input is the merged feature CSV produced by
``1_compute_tsne_train_test.py``. Only rows with ``source_type == SAT`` are
processed. The statistics are written to a separate one-row-per-crop CSV with
``filename`` as the merge key. A continuous variable gets one column per
configured percentile (for example, ``cot_p50``), while categorical variables
get one column (for example, ``cma``).

Training and test crop paths are resolved from the row's ``vector_type`` and
``test_set`` values using ``model_eval_paper/process_run_config.yaml``. Rows
without a crop file, such as centroids, are excluded by the SAT filter.
NetCDF data is downloaded once per variable/day and reused for all selected
timestamps of that crop.

Usage
-----
Run with the project configuration::

    python 3_compute_crops_statistic_new.py

Use another configuration or variables metadata file::

    python 3_compute_crops_statistic_new.py --config /path/to/process_run_config.yaml \
        --variables-config /path/to/variables_metadata.yaml
"""

import argparse
import io
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from scipy.ndimage import binary_closing

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.buckets.credentials_buckets import (
    S3_ACCESS_KEY,
    S3_ENDPOINT_URL,
    S3_SECRET_ACCESS_KEY,
)
from utils.buckets.get_data_from_buckets import Initialize_s3_client, read_file
from utils.configs import load_config
from utils.processing.stats_utils import (
    classify_day_night,
    compute_categorical_values,
    compute_category_fractions,
    compute_cumulative_precipitation,
    compute_percentile,
    compute_precipitation_area,
)


logger = logging.getLogger(__name__)


def configure_logging(config):
    """Configure timestamped console and file logging."""
    logging_config = config.get("logging", {})
    level_name = str(logging_config.get("log_level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    log_dir = Path(logging_config.get("logs_path", REPO_ROOT / "logs"))
    if not log_dir.is_absolute():
        log_dir = REPO_ROOT / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / logging_config.get(
        "statistics_log_filename", "crop_statistics.log"
    )
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False
    for handler in (
        logging.StreamHandler(),
        logging.FileHandler(
            log_path,
            mode=logging_config.get("statistics_log_mode", "w"),
        ),
    ):
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.info("Logging to %s", log_path)


def resolve_feature_csv(config):
    """Return the merged feature CSV produced by scripts 0 and 1."""
    experiment = config["experiment"]
    preparation = config["feature_preparation"]
    run_name = experiment["run_names"][0]
    output_dir = (
        Path(experiment["path_out"])
        / run_name
        / preparation.get("output_suffix", "all")
    )
    filename = f"{preparation.get('output_filename', 'features')}_{run_name}.csv"
    return output_dir / filename


def resolve_statistics_csv(config):
    """Return the separate output path for SAT crop statistics."""
    experiment = config["experiment"]
    run_name = experiment["run_names"][0]
    preparation = config["feature_preparation"]
    output_dir = (
        Path(experiment["path_out"])
        / run_name
        / preparation.get("output_suffix", "all")
    )
    return output_dir / f"crops_statistics_SAT_{run_name}.csv"


def resolve_crop_path(row, config):
    """Resolve a feature row to its NetCDF crop path, or return ``None``."""
    filename = row.get("filename")
    if not isinstance(filename, str) or not filename or filename.startswith("CENTROID_"):
        return None

    data_config = config["data"]
    extension = data_config.get("file_extension", "nc")
    if extension != "nc":
        raise ValueError("This statistics pipeline currently requires NetCDF crops.")

    vector_type = str(row.get("vector_type", "TRAIN")).upper()
    if vector_type == "TRAIN":
        directory = (
            Path(data_config["data_base_path"])
            / data_config["crops_name"]
            / extension
            / str(config["feature_preparation"]["train"].get("crop_split", "1"))
        )
    elif vector_type == "TEST":
        test_name = row.get("test_set")
        test_sets = config["feature_preparation"].get("test", {}).get("sets", [])
        test_config = next(
            (item for item in test_sets if item.get("name") == test_name), None
        )
        if test_config is None:
            raise ValueError(f"No test-set path configured for {test_name!r}.")
        directory = Path(test_config["crops_path"].replace(
            "{run_name}", config["experiment"]["run_names"][0]
        ))
    else:
        return None

    path = directory / filename
    return path if path.exists() else None


def crop_metadata(crop_path, nc_engine):
    """Read crop coordinates and timestamps with one NetCDF open."""
    with xr.open_dataset(crop_path, engine=nc_engine) as dataset:
        if "time" not in dataset:
            raise ValueError(f"Crop has no time coordinate: {crop_path}")
        times = np.atleast_1d(dataset.time.values)
        lat = np.asarray(dataset["lat"].values)
        lon = np.asarray(dataset["lon"].values)
    return {
        "times": times.astype("datetime64[ns]"),
        "lat_min": float(np.nanmin(lat)),
        "lat_max": float(np.nanmax(lat)),
        "lon_min": float(np.nanmin(lon)),
        "lon_max": float(np.nanmax(lon)),
    }


def day_groups(times, variable):
    """Group timestamps by UTC day and apply variable-specific time rules.

    EUCLID observations arrive every five minutes. Crop timestamps represent
    the start of a 15-minute crop window, so the three observations at
    ``t``, ``t + 5 min``, and ``t + 10 min`` are included.
    """
    selected = sorted(times)
    if variable == "euclid_msg_grid":
        selected = [
            timestamp + np.timedelta64(offset, "m")
            for timestamp in selected
            for offset in (0, 5, 10)
        ]
    if variable == "precipitation":
        selected = [
            timestamp for timestamp in selected
            if np.datetime64(timestamp, "m").astype(object).minute in (0, 30)
        ]
    groups = {}
    for timestamp in selected:
        groups.setdefault(str(np.datetime64(timestamp, "D")), []).append(timestamp)
    return groups


def bucket_key(day, variable_metadata):
    """Build the bucket object key for one variable and day."""
    year, month, day_number = map(int, day.split("-"))
    prefix = variable_metadata.get("bucket_filename_prefix", "")
    suffix = variable_metadata.get("bucket_filename_suffix", "")
    if variable_metadata.get("bucket_name") == "expats-euclid":
        return (
            f"{year:04d}/{month:02d}/{prefix}{year:04d}{month:02d}"
            f"{day_number:02d}{suffix}"
        )
    return f"{prefix}{year:04d}-{month:02d}-{day_number:02d}{suffix}"


def load_day_values(s3, day, variable, metadata, bounds, nc_engine):
    """Download and spatially subset one variable and its CMA mask once."""
    # NetCDF4 cannot open BytesIO objects; h5netcdf/scipy can.
    byte_stream_engine = nc_engine if nc_engine in {"h5netcdf", "scipy"} else "h5netcdf"
    variable_bytes = read_file(
        s3, bucket_key(day, metadata), metadata["bucket_name"]
    )
    cma_bytes = read_file(s3, f"MCP_{day}_regrid.nc", "expats-cmsaf-cloud")
    if variable_bytes is None or cma_bytes is None:
        return None, None

    lat_min, lat_max, lon_min, lon_max = bounds
    with xr.open_dataset(io.BytesIO(variable_bytes), engine=byte_stream_engine) as dataset:
        variable_data = dataset[variable].sel(
            lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)
        ).load()
    with xr.open_dataset(io.BytesIO(cma_bytes), engine=byte_stream_engine) as dataset:
        cma_data = dataset["cma"].sel(
            lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)
        ).load()
    if {"lat", "lon"}.issubset(variable_data.coords) and {"lat", "lon"}.issubset(cma_data.coords):
        cma_data = cma_data.interp_like(variable_data, method="nearest")
    return variable_data, cma_data


def load_crop_values(crop_path, variable, times, nc_engine):
    """Load a crop variable and align it to the crop timestamps."""
    with xr.open_dataset(crop_path, engine=nc_engine) as dataset:
        if variable not in dataset:
            raise ValueError(f"Variable {variable!r} missing from {crop_path}")
        data = dataset[variable].sel(time=times).load()
    return data


def mask_values(values, cma_values, variable):
    """Apply the existing CMA mask rules without modifying source arrays."""
    values = np.asarray(values).flatten()
    cma_values = np.asarray(cma_values).flatten()
    if variable == "cma":
        return values
    if variable in {"rain_rate", "precipitation"}:
        rainy_cloudy = (cma_values == 1) & (values > 0.1)
        return values[rainy_cloudy] if rainy_cloudy.any() else np.array([0.0])
    if variable == "euclid_msg_grid":
        return values[values > 0] if values.size else np.array([0])
    filtered = values[cma_values == 1]
    return filtered if filtered.size else np.array([0])


def compute_stat(values, statistic, variable):
    """Compute one configured statistic."""
    if statistic == "value":
        return compute_categorical_values(values, variable)
    return compute_percentile(values, int(statistic))


def compute_derived_statistics(variable, variable_metadata, source_values,
                               cma_values=None, infrared_values=None):
    """Compute one metadata-defined derived statistic."""
    derived_type = variable_metadata["derived_type"]
    if derived_type == "category_fractions":
        cloud_mask = cma_values == 1 if cma_values is not None else None
        return compute_category_fractions(
            source_values,
            variable_metadata["categories"],
            valid_mask=cloud_mask,
        )
    if derived_type == "day_night":
        return {variable: classify_day_night(
            source_values, variable_metadata.get("nan_fraction_threshold", 0.5)
        )}
    if derived_type == "precipitation_area":
        return {variable: compute_precipitation_area(
            source_values,
            infrared_values,
            variable_metadata.get("clear_sky_threshold", 320.0),
            variable_metadata.get("rain_threshold", 0.1),
            variable_metadata.get("min_pixel_count", 100),
        )}
    if derived_type == "cumulative_precipitation":
        return {variable: compute_cumulative_precipitation(
            source_values,
            infrared_values,
            variable_metadata.get("clear_sky_threshold", 320.0),
            variable_metadata.get("rain_threshold", 0.1),
            variable_metadata.get("min_pixel_count", 100),
            variable_metadata.get("accumulation_factor", 2.0),
        )}
    raise ValueError(f"Unknown derived statistic type: {derived_type}")


def process_row(row, config, variable_config, s3, crop_number=None, total_crops=None):
    """Compute all configured statistics for one feature-CSV row."""
    result = row.to_dict()
    progress = (
        f"crop {crop_number}/{total_crops}"
        if crop_number is not None and total_crops is not None
        else "crop"
    )
    crop_path = resolve_crop_path(row, config)
    if crop_path is None:
        logger.warning(
            "%s (%s): no crop file; keeping statistics as missing",
            progress,
            row.get("filename"),
        )
        return result

    stats_config = config["statistics"]["spatial"]
    variables = stats_config["sel_vars"]
    percentiles = stats_config.get("percentiles", [])
    nc_engine = config["data"].get("nc_engine")
    metadata = crop_metadata(crop_path, nc_engine)
    bounds = tuple(metadata[key] for key in ("lat_min", "lat_max", "lon_min", "lon_max"))
    logger.info("Processing %s: %s", progress, crop_path.name)
    day_cache = {}
    derived_variables = [
        variable for variable in variables
        if variable_config["variables"][variable].get("derived_type")
    ]
    needs_infrared = any(
        variable_config["variables"][variable].get("derived_type")
        in {"precipitation_area", "cumulative_precipitation"}
        for variable in derived_variables
    )
    infrared_data = (
        load_crop_values(crop_path, "IR_108", metadata["times"], nc_engine)
        if needs_infrared else None
    )

    for variable in variables:
        variable_metadata = variable_config["variables"][variable]
        source_variable = variable_metadata.get("source_variable", variable)
        source_metadata = variable_config["variables"].get(source_variable)
        if source_metadata is None:
            raise ValueError(f"No metadata found for source variable {source_variable!r}.")
        derived_type = variable_metadata.get("derived_type")

        if derived_type:
            source_values = []
            mask_values_by_time = []
            infrared_values = []
            for day, timestamps in day_groups(metadata["times"], source_variable).items():
                cache_key = (day, source_variable)
                if cache_key not in day_cache:
                    day_cache[cache_key] = load_day_values(
                        s3, day, source_variable, source_metadata, bounds, nc_engine
                    )
                variable_data, cma_data = day_cache[cache_key]
                if variable_data is None:
                    logger.warning("Missing data for %s on %s (%s)", source_variable, day, crop_path.name)
                    continue
                for timestamp in timestamps:
                    try:
                        source_values.append(variable_data.sel(time=timestamp).values)
                        if cma_data is not None:
                            mask_values_by_time.append(cma_data.sel(time=timestamp).values)
                        if infrared_data is not None:
                            infrared_frame = infrared_data.sel(time=timestamp)
                            infrared_frame = infrared_frame.interp_like(
                                variable_data.sel(time=timestamp), method="nearest"
                            )
                            infrared_values.append(infrared_frame.values)
                    except KeyError:
                        logger.warning("Timestamp %s missing for %s", timestamp, crop_path.name)

            if source_values:
                source_array = np.concatenate([np.asarray(values).flatten() for values in source_values])
                cma_array = (
                    np.concatenate([
                        binary_closing(
                            np.asarray(values) == 1,
                            structure=np.ones((3, 3)),
                        ).astype(int).flatten()
                        for values in mask_values_by_time
                    ])
                    if mask_values_by_time else None
                )
                infrared_array = (
                    np.concatenate([np.asarray(values).flatten() for values in infrared_values])
                    if infrared_values else None
                )
                derived_results = compute_derived_statistics(
                    variable, variable_metadata, source_array, cma_array, infrared_array
                )
            else:
                derived_results = {
                    name: np.nan
                    for name in variable_metadata.get("categories", {})
                }
                if derived_type == "day_night":
                    derived_results = {variable: np.nan}
                elif derived_type in {"precipitation_area", "cumulative_precipitation"}:
                    derived_results = {variable: np.nan}
            result.update(derived_results)
            logger.info(
                "%s | variable=%s | statistics=%s",
                progress,
                variable,
                derived_results,
            )
            continue

        statistics = ["value"] if variable_metadata.get("categorical") else percentiles
        values_by_day = []
        for day, timestamps in day_groups(metadata["times"], source_variable).items():
            cache_key = (day, variable)
            if cache_key not in day_cache:
                day_cache[cache_key] = load_day_values(
                    s3, day, source_variable, source_metadata, bounds, nc_engine
                )
            variable_data, cma_data = day_cache[cache_key]
            if variable_data is None:
                logger.warning("Missing data for %s on %s (%s)", variable, day, crop_path.name)
                continue
            for timestamp in timestamps:
                try:
                    values = variable_data.sel(time=timestamp).values
                    cma_values = cma_data.sel(time=timestamp).values
                except KeyError:
                    logger.warning("Timestamp %s missing for %s", timestamp, crop_path.name)
                    continue
                values_by_day.append(mask_values(values, cma_values, variable))

        all_values = (
            np.concatenate([np.atleast_1d(values) for values in values_by_day])
            if values_by_day else np.array([np.nan])
        )
        for statistic in statistics:
            output_variable = variable_metadata.get("output_name", variable)
            column = (
                output_variable
                if statistic == "value"
                else f"{output_variable}_p{statistic}"
            )
            result[column] = compute_stat(all_values, statistic, variable)
        logger.info(
            "%s | variable=%s | statistics=%s",
            progress,
            variable,
            {
                column: result[column]
                for column in result
                if column.startswith(f"{output_variable}_p") or column == output_variable
            },
        )
    return result


def main(config_path=None, variables_config_path=None):
    """Compute SAT crop statistics and save them in a separate CSV."""
    config_path = Path(config_path or REPO_ROOT / "model_eval_paper" / "process_run_config.yaml")
    variables_config_path = Path(
        variables_config_path or REPO_ROOT / "configs" / "variables_metadata.yaml"
    )
    config = load_config(config_path)
    variable_config = load_config(variables_config_path)
    configure_logging(config)

    input_csv = resolve_feature_csv(config)
    logger.info("Using configuration: %s", config_path)
    logger.info("Reading feature CSV: %s", input_csv)
    dataframe = pd.read_csv(input_csv, low_memory=False)
    logger.info("Loaded %d rows and %d columns", len(dataframe), len(dataframe.columns))
    if "source_type" not in dataframe.columns:
        raise ValueError("The feature CSV must contain a 'source_type' column.")
    sat_mask = dataframe["source_type"].astype("string").str.upper().eq("SAT")
    dataframe = dataframe.loc[sat_mask].copy()
    logger.info("Selected %d SAT rows for statistics", len(dataframe))

    use_parallel = config["statistics"].get("use_parallel", False)
    requested_cores = config["statistics"].get("n_cores", 1)
    worker_count = max(1, min(int(requested_cores), os.cpu_count() or 1))
    s3 = Initialize_s3_client(S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY)
    started = time.perf_counter()

    if use_parallel and worker_count > 1:
        logger.info("Processing crops with %d threads", worker_count)
        results = Parallel(n_jobs=worker_count, prefer="threads")(
            delayed(process_row)(row, config, variable_config, s3, index, len(dataframe))
            for index, (_, row) in enumerate(dataframe.iterrows(), start=1)
        )
    else:
        logger.info("Processing crops serially")
        results = []
        for index, (_, row) in enumerate(dataframe.iterrows(), start=1):
            results.append(process_row(row, config, variable_config, s3, index, len(dataframe)))

    new_columns = [
        column for result in results for column in result
        if column not in dataframe.columns
    ]
    output = pd.DataFrame(
        results,
        columns=list(dataframe.columns) + list(dict.fromkeys(new_columns)),
    )
    statistic_columns = [
        column for column in output.columns
        if column == "filename" or column in new_columns
    ]
    output = output[statistic_columns]
    statistics_csv = resolve_statistics_csv(config)
    output.to_csv(statistics_csv, index=False)
    logger.info(
        "Saved SAT crop statistics: %s (%d rows, %.1f seconds)",
        statistics_csv,
        len(output),
        time.perf_counter() - started,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Path to process_run_config.yaml")
    parser.add_argument(
        "--variables-config", type=Path, help="Path to variables_metadata.yaml"
    )
    arguments = parser.parse_args()
    main(arguments.config, arguments.variables_config)

#3133116