"""Readers for feature tensors and cluster metadata used by sample-selection scripts.

Public API
----------
load_features_to_df            simple train-only loader (used by select_convective_samples).
extract_timestamp              parse YYYYmmdd_HHMM from a crop file path or xarray dataset.
extract_latlon_centre          derive (lat, lon) centre from filename or xarray dataset.
add_labels_to_test             assign cluster labels via cosine similarity to centroids.
load_centroids_to_df           load centroid vectors as a labelled DataFrame.
mark_case_study                override source_type using 'msg'/'icon' filename markers.
load_summary                   load a storm-summary CSV with parsed date column.
merge_test_with_summary        row-index join between a test DataFrame and a summary table.
"""

import os
from glob import glob

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity

from utils.processing.datetime_utils import extract_timestamp  # noqa: F401 – re-exported
from utils.processing.coords_utils import extract_latlon_centre  # noqa: F401 – re-exported


# ---------------------------------------------------------------------------
# Simple loader – kept for backward compatibility with select_convective_samples
# ---------------------------------------------------------------------------

def load_features_to_df(
    feature_path,
    indices_file,
    features_file,
    assignments_file,
    distances_file,
    centroids_file,
    crops_path,
    file_extension,
):
    """Load model features, labels, and crop paths into a DataFrame (train only)."""
    indices = np.load(os.path.join(feature_path, indices_file))
    features = np.load(os.path.join(feature_path, features_file))

    crop_paths = sorted(glob(os.path.join(crops_path, f"*.{file_extension}")))

    df = pd.DataFrame(
        np.reshape(features, (len(indices), -1)),
        columns=[f"dim_{i+1}" for i in range(features.shape[1])],
        index=indices,
    )

    df["path"] = [crop_paths[i] for i in indices]

    assignments = torch.load(os.path.join(feature_path, assignments_file), map_location="cpu")
    distances = torch.load(os.path.join(feature_path, distances_file), map_location="cpu")

    df["label"] = assignments[0].cpu().numpy()
    df["distance"] = distances[0].cpu().numpy()

    centroids = torch.load(os.path.join(feature_path, centroids_file), map_location="cpu").numpy()

    return df.reset_index(drop=True), centroids


# ---------------------------------------------------------------------------
# Metadata helpers – implemented in processing utils, re-exported here for
# convenience so existing callers of features_reader keep working.
# extract_timestamp   → utils.processing.datetime_utils
# extract_latlon_centre → utils.processing.coords_utils
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Label assignment
# ---------------------------------------------------------------------------

def add_labels_to_test(df: pd.DataFrame, centroids_file: str, top_k: int = 2) -> pd.DataFrame:
    """Assign cluster labels to test samples via cosine similarity to centroids.

    Adds columns ``label``, ``distance``, and (when *top_k* >= 2) ``label_2nd``.
    """
    centroids = torch.load(centroids_file, map_location="cpu").cpu().numpy()
    features = df[[c for c in df.columns if c.startswith("dim_")]].values
    sim = cosine_similarity(features, centroids)
    top_k = min(top_k, sim.shape[1])
    top_idx = np.argpartition(sim, -top_k, axis=1)[:, -top_k:]
    top_idx = top_idx[np.arange(len(sim))[:, None], np.argsort(
        sim[np.arange(len(sim))[:, None], top_idx], axis=1
    )]

    df = df.copy()
    df["label"] = top_idx[:, -1]
    df["distance"] = sim[np.arange(len(df)), top_idx[:, -1]]
    if top_k >= 2:
        df["label_2nd"] = top_idx[:, -2]
    return df


def load_centroids_to_df(centroids_file: str) -> pd.DataFrame:
    """Load centroid vectors as a DataFrame with CENTROID metadata columns."""
    centroids = torch.load(centroids_file, map_location="cpu").cpu().numpy()
    df = pd.DataFrame(
        centroids,
        columns=[f"dim_{i+1}" for i in range(centroids.shape[1])],
    )
    df["label"] = np.arange(centroids.shape[0])
    df["distance"] = 1.0
    df["source_type"] = pd.NA
    df["test_set"] = pd.NA
    df["vector_type"] = "CENTROID"
    return df


# ---------------------------------------------------------------------------
# Case-study marking
# ---------------------------------------------------------------------------

def mark_case_study(df: pd.DataFrame) -> pd.DataFrame:
    """Override ``source_type`` using 'msg' (→SAT) or 'icon' (→MODEL) filename markers."""
    df = df.copy()
    filename_col = "filename" if "filename" in df.columns else "path"
    filenames = df[filename_col].apply(os.path.basename)
    df.loc[filenames.str.contains("icon", case=False, na=False), "source_type"] = "MODEL"
    df.loc[filenames.str.contains("msg", case=False, na=False), "source_type"] = "SAT"
    df["vector_type"] = df["vector_type"].fillna("TEST")
    return df


# ---------------------------------------------------------------------------
# Storm summary helpers
# ---------------------------------------------------------------------------

def load_summary(filename: str, directory: str, date_col: str = "date") -> pd.DataFrame:
    """Load a storm-summary CSV and parse its date column."""
    path = os.path.join(directory, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Summary file not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    return df


def merge_test_with_summary(df_test: pd.DataFrame, df_summary: pd.DataFrame) -> pd.DataFrame:
    """Join a test DataFrame with a storm-summary table by row index.

    Drops ``date`` and ``time`` from the summary to avoid column collisions.
    """
    df_summary = df_summary.drop(columns=["date", "time"], errors="ignore")
    return pd.merge(
        df_test.reset_index(drop=True),
        df_summary.reset_index(drop=True),
        left_index=True,
        right_index=True,
        how="left",
    )
