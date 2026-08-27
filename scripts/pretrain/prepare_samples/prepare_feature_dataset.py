"""Build a unified feature dataset (TRAIN / TEST / CENTROID) from YAML config.

All parameters are read from ``configs/process_run_config.yaml`` under the
``feature_preparation`` section (or the path set via the PROCESS_RUN_CONFIG
environment variable).

Output columns
--------------
dim_1 … dim_N   : feature vector dimensions
path            : absolute path to the source crop file
label           : nearest centroid index (cosine similarity for TEST; assignments.pt for TRAIN)
distance        : cosine similarity / assignment distance to the assigned centroid
label_2nd       : second-nearest centroid index (TEST only, when label_top_k >= 2)
split           : TRAIN | TEST | CENTROID
source_type     : SAT | MODEL | <null>  (can be further refined by mark_case_study)
event_type      : PRECIP | HAIL | TRAJ | … | <null>
vector_type     : convenience concat, e.g. SAT_PRECIP, MODEL_TRAJ, TRAIN, CENTROID
datetime        : timestamp YYYYmmdd_HHMM  (optional, controlled per-split in config)
lat_centre      : latitude of crop centre   (optional)
lon_centre      : longitude of crop centre  (optional)
2d_dim_1/2      : 2-D projection coordinates (optional, e.g. t-SNE)
<summary cols>  : extra columns from a storm-summary CSV (TEST only, optional)
"""

import gc
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parents[3]))

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve(value: str, run_name: str) -> str:
    """Replace ``{run_name}`` placeholder in a path string."""
    return value.replace("{run_name}", run_name) if value else value


def _get(set_cfg: dict, global_cfg: dict, key: str, default=None):
    """Return set-level override if present, otherwise the global default."""
    return set_cfg.get(key, global_cfg.get(key, default))


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
    print(f"  Found {len(all_crop_paths)} crop files; using {len(crop_paths)}.")

    # ---- Build DataFrame -----------------------------------------------------
    df = pd.DataFrame(
        np.reshape(features, (len(indices), -1)),
        columns=[f"dim_{i+1}" for i in range(features.shape[1])],
    )
    df["path"] = crop_paths

    # ---- Labels + distances --------------------------------------------------
    if is_train:
        assignments = torch.load(runtime["assignments_file"], map_location="cpu")[0].cpu().numpy()
        distances   = torch.load(runtime["distances_file"],   map_location="cpu")[0].cpu().numpy()
        df["label"]    = assignments[indices]
        df["distance"] = distances[indices]
    else:
        top_k = global_test_cfg.get("label_top_k", 2)
        df = add_labels_to_test(df, runtime["centroids_file"], top_k=top_k)

    # ---- Split / source_type / event_type / vector_type ----------------------
    if is_train:
        df["split"]       = "TRAIN"
        df["source_type"] = pd.NA
        df["event_type"]  = pd.NA
        df["vector_type"] = "TRAIN"
    else:
        source_type = set_cfg.get("source_type", "SAT")
        event_type  = set_cfg.get("event_type", "")
        df["split"]       = "TEST"
        df["source_type"] = source_type
        df["event_type"]  = event_type
        df["vector_type"] = f"{source_type}_{event_type}" if event_type else source_type

    # ---- Optional: case-study filename marking -------------------------------
    if not is_train and set_cfg.get("mark_case_study", False):
        df = mark_case_study(df)

    # ---- Optional: timestamps ------------------------------------------------
    ref_ts = set_cfg if is_train else global_test_cfg
    include_ts = _get(set_cfg, ref_ts, "include_timestamp", False)
    if include_ts:
        from_fn = _get(set_cfg, ref_ts, "extract_latlon_from_filename", False)
        df["datetime"] = pd.to_datetime(
            [extract_timestamp(p, from_filename=from_fn) for p in crop_paths],
            format="%Y%m%d_%H%M",
        )

    # ---- Optional: lat / lon -------------------------------------------------
    include_ll = _get(set_cfg, ref_ts, "include_latlon", False)
    if include_ll:
        from_fn = _get(set_cfg, ref_ts, "extract_latlon_from_filename", False)
        latlon = np.array([extract_latlon_centre(p, from_filename=from_fn) for p in crop_paths])
        df["lat_centre"] = latlon[:, 0]
        df["lon_centre"] = latlon[:, 1]

    # ---- Optional: 2-D projected features (e.g., t-SNE) ---------------------
    include_2d = _get(set_cfg, ref_ts, "include_2d_features", False)
    if include_2d:
        feat_2d_file = _get(set_cfg, ref_ts, "feat_2d_file")
        feat_2d_dir  = _get(set_cfg, ref_ts, "feat_2d_dir")
        if feat_2d_file and feat_2d_dir:
            f2d = np.load(os.path.join(_resolve(feat_2d_dir, runtime["run_name"]), feat_2d_file))
            if subsample:
                f2d = f2d[:subsample]
            for i in range(f2d.shape[1]):
                df[f"2d_dim_{i+1}"] = f2d[:, i]

    # ---- Optional: storm-summary merge (test only) ---------------------------
    if not is_train:
        summary_file = set_cfg.get("storm_summary_file")
        summary_dir  = set_cfg.get("storm_summary_dir")
        if summary_file and summary_dir:
            df_sum = load_summary(summary_file, _resolve(summary_dir, runtime["run_name"]))
            df = merge_test_with_summary(df, df_sum)
            print(f"  Merged with storm summary: {summary_file}")

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
    repo_root   = Path(__file__).resolve().parents[3]
    config_path = Path(os.getenv("PROCESS_RUN_CONFIG",
                                 str(repo_root / "configs" / "process_run_config.yaml")))

    config  = load_config(config_path)
    fp_cfg  = config["feature_preparation"]
    runtime = build_runtime_config(config)

    os.makedirs(runtime["output_path"], exist_ok=True)
    gc.collect()

    frames = []

    # ---- TRAIN ---------------------------------------------------------------
    if fp_cfg["train"].get("enabled", True):
        print("Loading training features...")
        df_train = _load_set(fp_cfg["train"], runtime, fp_cfg, is_train=True)
        print(f"  Train: {len(df_train)} samples.")
        frames.append(df_train)

    # ---- CENTROIDS -----------------------------------------------------------
    if fp_cfg["centroids"].get("enabled", True):
        print("Loading centroids...")
        df_c = load_centroids_to_df(runtime["centroids_file"])
        print(f"  Centroids: {len(df_c)} rows.")
        frames.append(df_c)

    # ---- TEST SETS -----------------------------------------------------------
    if fp_cfg["test"].get("enabled", False):
        for ts_cfg in fp_cfg["test"].get("sets", []):
            name = ts_cfg.get("name", "test")
            print(f"Loading test set: {name} ...")
            df_ts = _load_set(ts_cfg, runtime, fp_cfg, is_train=False)
            print(f"  {name}: {len(df_ts)} samples.")

            per_set_csv = os.path.join(
                runtime["output_path"],
                f"features_test_{name}_{runtime['run_name']}.csv",
            )
            df_ts.to_csv(per_set_csv, index=False)
            print(f"  Saved per-set CSV: {per_set_csv}")
            frames.append(df_ts)

    # ---- MERGE & SAVE --------------------------------------------------------
    df_final = pd.concat(frames, ignore_index=True)
    print(f"\nFinal dataset shape: {df_final.shape}")
    print(df_final[["split", "source_type", "event_type", "vector_type"]].value_counts().to_string())

    output_name = fp_cfg.get("output_filename", "features")
    output_csv  = os.path.join(runtime["output_path"], f"{output_name}_{runtime['run_name']}.csv")
    df_final.to_csv(output_csv, index=False)
    print(f"\nSaved merged dataset: {output_csv}")


if __name__ == "__main__":
    main()
