"""Build a unified feature dataset (TRAIN / TEST / CENTROID) from YAML config.

Usage
-----
Run with the default configuration stored next to this script's project::

    python 0_prepare_feature_dataset.py

Select another configuration explicitly with::

    python 0_prepare_feature_dataset.py --config /path/to/process_run_config.yaml

Configuration precedence is: ``--config`` command-line argument, then the
``PROCESS_RUN_CONFIG`` environment variable, then
``model_eval_paper/process_run_config.yaml``. The default configuration is
resolved relative to this script, so the command can be run from any directory.

All parameters are read from the selected YAML file under the
``feature_preparation`` section (or the path set via the PROCESS_RUN_CONFIG
environment variable).

Output columns
--------------
dim_1 … dim_N   : feature vector dimensions
filename        : basename of the source crop file
label           : nearest centroid index (cosine similarity for TEST; assignments.pt for TRAIN)
distance        : cosine similarity / assignment distance to the assigned centroid
label_2nd       : second-nearest centroid index (TEST only, when label_top_k >= 2)
source_type     : SAT | MODEL | <null>  (can be further refined by mark_case_study)
test_set        : test-set name for TEST rows; <null> for TRAIN / CENTROID
vector_type     : TRAIN | TEST | CENTROID
datetime        : timestamp YYYYmmdd_HHMM  (optional, controlled per-split in config)
lat_centre      : latitude of crop centre   (optional)
lon_centre      : longitude of crop centre  (optional)
<summary cols>  : extra columns from a storm-summary CSV (TEST only, optional)
"""

import gc
import argparse
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import sys
import time
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.configs import load_config
from utils.read.features_reader import (
    add_labels_to_test,
    extract_latlon_centre,
    extract_timestamp,
    load_centroids_to_df,
    load_summary,
    mark_case_study,
    merge_test_with_summary,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve(value: str, run_name: str) -> str:
    """Replace ``{run_name}`` placeholder in a path string."""
    return value.replace("{run_name}", run_name) if value else value


def _get(set_cfg: dict, global_cfg: dict, key: str, default=None):
    """Return set-level override if present, otherwise the global default."""
    return set_cfg.get(key, global_cfg.get(key, default))


def configure_logging(config: dict, run_name: str):
    """Configure timestamped console and file logging for one run."""
    logging_cfg = config.get("logging", {})
    log_level = getattr(
        logging,
        str(logging_cfg.get("log_level", "INFO")).upper(),
        logging.INFO,
    )
    log_dir = Path(logging_cfg.get("logs_path", REPO_ROOT / "logs"))
    if not log_dir.is_absolute():
        log_dir = REPO_ROOT / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    log_filename = logging_cfg.get(
        "feature_dataset_log_filename",
        f"feature_dataset_{run_name}.log",
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
        log_path,
        mode=logging_cfg.get("feature_dataset_log_mode", "a"),
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("Logging to %s", log_path)


def _extract_crop_metadata(path, include_timestamp, include_latlon, from_filename):
    """Extract requested metadata with at most one NetCDF open per crop."""
    timestamp = None
    latlon = None

    if from_filename:
        if include_timestamp:
            timestamp = extract_timestamp(path, from_filename=True)
        if include_latlon:
            latlon = extract_latlon_centre(path, from_filename=True)
        return timestamp, latlon

    with xr.open_dataset(path, engine="h5netcdf") as ds:
        if include_timestamp:
            timestamp = pd.to_datetime(ds.time[0].values).strftime("%Y%m%d_%H%M")
        if include_latlon:
            latlon = (
                round(float((ds.lat.values.min() + ds.lat.values.max()) / 2.0), 2),
                round(float((ds.lon.values.min() + ds.lon.values.max()) / 2.0), 2),
            )
    return timestamp, latlon


def _extract_metadata_for_crops(crop_paths, include_timestamp, include_latlon, from_filename, workers):
    if not include_timestamp and not include_latlon:
        return [(None, None)] * len(crop_paths)

    extract = lambda path: _extract_crop_metadata(
        path, include_timestamp, include_latlon, from_filename
    )
    if workers <= 1:
        return [extract(path) for path in crop_paths]
    logger.info("Reading metadata for %d crops with %d workers", len(crop_paths), workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(extract, crop_paths))


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def _load_set(
    set_cfg: dict,
    runtime: dict,
    fp_cfg: dict,
    is_train: bool = False,
) -> pd.DataFrame:
    """Load one split (train or a single test set) into a feature DataFrame.

    Parameters
    ----------
    set_cfg  : ``fp_cfg["train"]`` for the training split, or one entry from
               ``fp_cfg["test"]["sets"]`` for a test split.
    runtime  : resolved runtime paths from ``build_runtime_config``.
    fp_cfg   : full ``feature_preparation`` config dict.
    is_train : True when loading the training split.
    """
    global_test_cfg = fp_cfg.get("test", {})

    # ---- Resolve paths --------------------------------------------------------
    if is_train:
        feat_dir   = runtime["train_feat_dir"]
        crops_path = runtime["image_train_path"]
    else:
        feat_dir   = _resolve(set_cfg["feat_dir"],   runtime["run_name"])
        crops_path = _resolve(set_cfg["crops_path"], runtime["run_name"])

    file_ext  = runtime["file_extension"]
    inds_file = fp_cfg["feature_file_inds"]
    feat_file = fp_cfg["feature_file_features"]

    # ---- Load raw arrays -----------------------------------------------------
    indices  = np.load(os.path.join(feat_dir, inds_file))
    features = np.load(os.path.join(feat_dir, feat_file))

    # ---- Optional subsample --------------------------------------------------
    ref_cfg  = set_cfg if is_train else global_test_cfg
    subsample = _get(set_cfg, ref_cfg, "subsample")
    if subsample:
        indices  = indices[:subsample]
        features = features[:subsample]

    # ---- Crop paths ----------------------------------------------------------
    all_crop_paths = sorted(glob(os.path.join(crops_path, f"*.{file_ext}")))
    crop_paths = [all_crop_paths[i] for i in indices]
    logger.info("Found %d crop files; using %d", len(all_crop_paths), len(crop_paths))

    # ---- Build DataFrame -----------------------------------------------------
    df = pd.DataFrame(
        np.reshape(features, (len(indices), -1)),
        columns=[f"dim_{i+1}" for i in range(features.shape[1])],
    )
    df["filename"] = [os.path.basename(p) for p in crop_paths]

    # ---- Labels + distances --------------------------------------------------
    if is_train:
        assignments = torch.load(runtime["assignments_file"], map_location="cpu")[0].cpu().numpy()
        distances   = torch.load(runtime["distances_file"],   map_location="cpu")[0].cpu().numpy()
        df["label"]    = assignments[indices]
        df["distance"] = distances[indices]
    else:
        top_k = global_test_cfg.get("label_top_k", 2)
        df = add_labels_to_test(df, runtime["centroids_file"], top_k=top_k)

    # ---- source_type / test_set / vector_type -------------------------------
    if is_train:
        df["source_type"] = set_cfg.get("source_type", "SAT")
        df["test_set"]    = pd.NA
        df["vector_type"] = "TRAIN"
    else:
        source_type = set_cfg.get("source_type", "SAT")
        test_set    = set_cfg.get("name", "test")
        df["source_type"] = source_type
        df["test_set"]    = test_set
        df["vector_type"] = "TEST"

    # ---- Optional: case-study filename marking -------------------------------
    if not is_train and set_cfg.get("mark_case_study", False):
        df = mark_case_study(df)

    # ---- Optional: timestamps ------------------------------------------------
    ref_ts = set_cfg if is_train else global_test_cfg
    include_ts = _get(set_cfg, ref_ts, "include_timestamp", False)
    include_ll = _get(set_cfg, ref_ts, "include_latlon", False)
    if include_ts or include_ll:
        from_fn = _get(set_cfg, ref_ts, "extract_latlon_from_filename", False)
        metadata_workers = fp_cfg.get("metadata_workers", 1)
        metadata = _extract_metadata_for_crops(
            crop_paths, include_ts, include_ll, from_fn, metadata_workers
        )
    if include_ts:
        df["datetime"] = pd.to_datetime(
            [timestamp for timestamp, _ in metadata],
            format="%Y%m%d_%H%M",
        )

    # ---- Optional: lat / lon -------------------------------------------------
    if include_ll:
        latlon = np.array([coords for _, coords in metadata])
        df["lat_centre"] = latlon[:, 0]
        df["lon_centre"] = latlon[:, 1]

    # Optional 2-D projected features are intentionally kept out of this table and
    # should be added as a separate post-processing step when needed.

    # ---- Optional: storm-summary merge (test only) ---------------------------
    if not is_train:
        summary_file = set_cfg.get("storm_summary_file")
        summary_dir  = set_cfg.get("storm_summary_dir")
        if summary_file and summary_dir:
            df_sum = load_summary(summary_file, _resolve(summary_dir, runtime["run_name"]))
            df = merge_test_with_summary(df, df_sum)
            logger.info("Merged with storm summary: %s", summary_file)

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Runtime config builder
# ---------------------------------------------------------------------------

def build_runtime_config(config: dict) -> dict:
    """Extract and resolve all runtime paths from the YAML config."""
    exp  = config["experiment"]
    data = config["data"]
    fp   = config["feature_preparation"]

    run_name       = exp["run_names"][0]
    epoch          = exp["epoch"]
    base_path      = exp["base_path"]
    path_out       = exp["path_out"]
    checkpoints    = os.path.join(base_path, run_name, "checkpoints")
    file_ext       = data["file_extension"]
    crop_split     = str(fp["train"].get("crop_split", "1"))

    return {
        "run_name":          run_name,
        "train_feat_dir":    os.path.join(base_path, run_name, "features", f"epoch_{epoch}"),
        "image_train_path":  os.path.join(data["data_base_path"], data["crops_name"], file_ext, crop_split),
        "output_path":       os.path.join(path_out, run_name, fp.get("output_suffix", "all")),
        "file_extension":    file_ext,
        "assignments_file":  os.path.join(checkpoints, fp.get("assignments_file", "assignments.pt")),
        "distances_file":    os.path.join(checkpoints, fp.get("distances_file", "distances.pt")),
        "centroids_file":    os.path.join(checkpoints, fp.get("centroids_file", "centroids0.pt")),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to the YAML configuration file (overrides PROCESS_RUN_CONFIG).",
    )
    args = parser.parse_args()

    config_path = args.config or Path(
        os.getenv(
            "PROCESS_RUN_CONFIG",
            str(REPO_ROOT / "model_eval_paper" / "process_run_config.yaml"),
        )
    )

    config  = load_config(config_path)
    fp_cfg  = config["feature_preparation"]
    runtime = build_runtime_config(config)
    os.makedirs(runtime["output_path"], exist_ok=True)
    configure_logging(config, runtime["run_name"])
    logger.info("Using configuration: %s", config_path)
    logger.info("Output directory: %s", runtime["output_path"])
    gc.collect()

    frames = []

    # ---- TRAIN ---------------------------------------------------------------
    if fp_cfg["train"].get("enabled", True):
        logger.info("Loading training features...")
        started = time.perf_counter()
        df_train = _load_set(fp_cfg["train"], runtime, fp_cfg, is_train=True)
        logger.info("Train: %d samples (%.1f s)", len(df_train), time.perf_counter() - started)
        frames.append(df_train)

    # ---- CENTROIDS -----------------------------------------------------------
    if fp_cfg["centroids"].get("enabled", True):
        logger.info("Loading centroids...")
        df_c = load_centroids_to_df(runtime["centroids_file"])
        logger.info("Centroids: %d rows", len(df_c))
        frames.append(df_c)

    # ---- TEST SETS -----------------------------------------------------------
    if fp_cfg["test"].get("enabled", False):
        for ts_cfg in fp_cfg["test"].get("sets", []):
            name = ts_cfg.get("name", "test")
            logger.info("Loading test set: %s", name)
            started = time.perf_counter()
            df_ts = _load_set(ts_cfg, runtime, fp_cfg, is_train=False)
            logger.info("%s: %d samples (%.1f s)", name, len(df_ts), time.perf_counter() - started)

            per_set_csv = os.path.join(
                runtime["output_path"],
                f"features_test_{name}_{runtime['run_name']}.csv",
            )
            df_ts.to_csv(per_set_csv, index=False)
            logger.info("Saved per-set CSV: %s", per_set_csv)
            frames.append(df_ts)

    # ---- MERGE & SAVE --------------------------------------------------------
    df_final = pd.concat(frames, ignore_index=True)
    df_final = df_final.drop(columns=["split"], errors="ignore")
    logger.info("Final dataset shape: %s", df_final.shape)
    logger.info("Dataset breakdown:\n%s", df_final[["source_type", "test_set", "vector_type"]].value_counts().to_string())

    output_name = fp_cfg.get("output_filename", "features")
    output_csv  = os.path.join(runtime["output_path"], f"{output_name}_{runtime['run_name']}.csv")
    df_final.to_csv(output_csv, index=False)
    logger.info("Saved merged dataset: %s", output_csv)


if __name__ == "__main__":
    main()


#2785976