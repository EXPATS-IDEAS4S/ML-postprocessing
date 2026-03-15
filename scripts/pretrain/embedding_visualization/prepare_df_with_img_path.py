"""
Prepare dataframe with image paths for embedding visualization.

"""

import os
import pandas as pd
import sys
import re

sys.path.append('/home/Daniele/codes/VISSL_postprocessing/')
from utils.plotting.class_colors import CLOUD_CLASS_INFO

from scripts.pretrain.embedding_visualization.config_visualization import (
    OUTPUT_PATH,
    TRAIN_LIST_IMAGE_CROPS,
    TEST_LIST_IMAGE_CROPS,
    N_FRAMES,
    FILENAME_TSNE,
    FILENAME_LABELS,
)

expanded_csv = os.path.join(
        os.path.dirname(OUTPUT_PATH),
        f"merged_tsne_crop_list_with_img_path.csv"
    )


# =============================================================================
# FUNCTIONS
# =============================================================================
def load_labels(output_path, tsne_filename, labels_filename, labels, tsne_cols=["tsne_dim_1", "tsne_dim_2"]) -> pd.DataFrame:
    """Load precomputed labels and dimensionality-reduced features."""
    
    #Open labels csv
    df = pd.read_csv(os.path.join(output_path, labels_filename), low_memory=False)
    df["split"] = df["split"].fillna("CENTROID")
    print(df["split"].value_counts())

    # Exclude centroid rows from labels table.
    df = df[df["split"].astype(str).str.upper() != "CENTROID"].copy()
    df["split"] = df["split"].astype(str).str.upper()
   
    #open df 2 dim feature space csv
    df_features = pd.read_csv(f"{output_path}{tsne_filename}", low_memory=False)

    print(df_features)

    # Some files store centroids as "CENTROID", others as "None".
    centroid_mask = df_features["vector_type"].astype(str).str.upper().isin(["CENTROID", "NONE"])
    df_centroids = df_features[centroid_mask].dropna(subset=tsne_cols).copy()

    #add label column to centroids df
    df_centroids["label"] = labels[:len(df_centroids)]

    df_features = df_features[~centroid_mask].dropna(subset=tsne_cols).copy()

    #check if len df_fearure is equal to len df
    if len(df_features) != len(df):
        print(f"Warning: len df_features ({len(df_features)}) != len df ({len(df)})")
    print(f"len df_features: {len(df_features)}")
    print(f"len df: {len(df)}")

    # Match by filename first (most stable key across files) to bring label/path.
    if "filename" not in df_features.columns and "path" in df_features.columns:
        df_features["filename"] = df_features["path"].apply(
            lambda p: os.path.basename(p) if isinstance(p, (str, os.PathLike)) else ""
        )

    df["filename"] = df["path"].apply(
        lambda p: os.path.basename(p) if isinstance(p, (str, os.PathLike)) else ""
    )

    df_lookup = (
        df[df["filename"] != ""][["filename", "label", "path", "split", "distance"]]
        .drop_duplicates(subset=["filename"], keep="first")
    )

    df_features = df_features.merge(
        df_lookup,
        on="filename",
        how="left",
        suffixes=("", "_src"),
    )

    if "label_src" in df_features.columns:
        src_label = pd.to_numeric(df_features["label_src"], errors="coerce")
        if "label" in df_features.columns:
            base_label = pd.to_numeric(df_features["label"], errors="coerce")
            df_features["label"] = src_label.where(src_label.notna(), base_label)
        else:
            df_features["label"] = src_label
        if df_features["label"].notna().any():
            df_features["label"] = df_features["label"].astype("Int64")
        df_features = df_features.drop(columns=["label_src"])

    if "path_src" in df_features.columns:
        if "path" in df_features.columns:
            df_features["path"] = df_features["path_src"].where(df_features["path_src"].notna(), df_features["path"])
        else:
            df_features["path"] = df_features["path_src"]
        df_features = df_features.drop(columns=["path_src"])

    if "split_src" in df_features.columns:
        if "split" in df_features.columns:
            df_features["split"] = df_features["split_src"].where(df_features["split_src"].notna(), df_features["split"])
        else:
            df_features["split"] = df_features["split_src"]
        df_features = df_features.drop(columns=["split_src"])

    if "distance_src" in df_features.columns:
        src_distance = pd.to_numeric(df_features["distance_src"], errors="coerce")
        if "distance" in df_features.columns:
            base_distance = pd.to_numeric(df_features["distance"], errors="coerce")
            df_features["distance"] = src_distance.where(src_distance.notna(), base_distance)
        else:
            df_features["distance"] = src_distance
        df_features = df_features.drop(columns=["distance_src"])

    # Guarantee split exists for downstream matching.
    if "split" not in df_features.columns:
        df_features["split"] = df_features["vector_type"].astype(str).str.upper()
    else:
        df_features["split"] = df_features["split"].fillna(df_features["vector_type"].astype(str).str.upper())

    df_features["split"] = df_features["split"].astype(str).str.upper()

    missing_paths = int(df_features["path"].isna().sum()) if "path" in df_features.columns else len(df_features)
    if missing_paths:
        print(f"Warning: {missing_paths} rows still missing path after filename merge.")
     

    #add label column to features df
    print(df_features)
    print(df_centroids)

    return df_features, df_centroids


def _stem_without_ext(path_value: str) -> str:
    return os.path.splitext(os.path.basename(path_value))[0]


def _image_prefix_and_frame(img_path: str):
    """Return (<prefix_before_tN>, frame_idx) from an image filename.

    Example:
      2013-04-01T00:00:00_0_t0_2013-04-01T00-00.png
      -> (2013-04-01T00:00:00_0, 0)
    """
    stem = _stem_without_ext(img_path)
    m = re.search(r"_t(\d+)_", stem)
    if not m:
        return None, None
    frame_idx = int(m.group(1))
    prefix = stem[:m.start()]
    return prefix, frame_idx


def _datetime_key_from_row(row) -> str:
    """Return YYYY-mm-ddTHH-MM key from datetime column or path filename."""
    if "datetime" in row and pd.notna(row["datetime"]):
        ts = pd.to_datetime(row["datetime"], errors="coerce")
        if pd.notna(ts):
            return ts.strftime("%Y-%m-%dT%H-%M")

    p = row.get("path", None)
    if isinstance(p, (str, os.PathLike)):
        stem = _stem_without_ext(str(p))
        m = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", stem)
        if m:
            ts = pd.to_datetime(m.group(1), errors="coerce")
            if pd.notna(ts):
                return ts.strftime("%Y-%m-%dT%H-%M")
    return ""


def _build_lookups(image_paths):
    """Build fast lookups for exact-prefix and datetime fallback matching."""
    by_prefix = {}
    by_dt = {}

    for img_path in image_paths:
        prefix, frame_idx = _image_prefix_and_frame(img_path)
        if prefix is None:
            continue
        by_prefix[(prefix, frame_idx)] = img_path

        # Datetime fallback from the suffix after _tN_.
        stem = _stem_without_ext(img_path)
        m_dt = re.search(r"_t\d+_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2})", stem)
        if m_dt:
            dt_key = m_dt.group(1)
            by_dt.setdefault((dt_key, frame_idx), []).append(img_path)

    return by_prefix, by_dt


def _match_image_for_row(row, frame_idx, train_lookups, test_lookups):
    split = str(row.get("split", "")).upper()
    sample_path = row.get("path", None)
    if not isinstance(sample_path, (str, os.PathLike)):
        return None

    sample_stem = _stem_without_ext(str(sample_path))
    dt_key = _datetime_key_from_row(row)

    if split == "TRAIN":
        by_prefix, by_dt = train_lookups
    elif split == "TEST":
        by_prefix, by_dt = test_lookups
    else:
        return None

    # Exact stem-prefix match first.
    exact = by_prefix.get((sample_stem, frame_idx), None)
    if exact is not None:
        return exact

    # Fallback: datetime-based (requested for TRAIN; also useful for TEST if needed).
    if dt_key:
        candidates = by_dt.get((dt_key, frame_idx), [])
        if candidates:
            return candidates[0]

    return None



def substitute_paths(df_labels: pd.DataFrame, output_path=expanded_csv):
    """Attach image path per frame to TRAIN/TEST samples.

    Keeps the original crop path in column 'path' and writes image path in 'img_path'.
    """

    # Keep only rows with a real path string; merged tables can contain NaN paths.
    valid_path_mask = df_labels["path"].apply(lambda p: isinstance(p, (str, os.PathLike)) and str(p) != "")
    dropped = int((~valid_path_mask).sum())
    if dropped:
        print(f"Skipping {dropped} rows with invalid path values.")
    df_labels = df_labels[valid_path_mask].copy()

    # Centroids should never be here, but enforce exclusion defensively.
    if "split" in df_labels.columns:
        before = len(df_labels)
        df_labels = df_labels[df_labels["split"].astype(str).str.upper().isin(["TRAIN", "TEST"])].copy()
        if before != len(df_labels):
            print(f"Removed {before - len(df_labels)} non-sample rows (e.g., centroids).")

    train_lookups = _build_lookups(TRAIN_LIST_IMAGE_CROPS)
    test_lookups = _build_lookups(TEST_LIST_IMAGE_CROPS)
    print(f"TRAIN images indexed: {len(TRAIN_LIST_IMAGE_CROPS)}")
    print(f"TEST images indexed: {len(TEST_LIST_IMAGE_CROPS)}")

    train_by_prefix, train_by_dt = train_lookups
    test_by_prefix, test_by_dt = test_lookups

    # Build unified lookup dicts for fast vectorized mapping.
    prefix_lookup = {}
    dt_lookup = {}

    for (prefix, frame_idx), img_path in train_by_prefix.items():
        prefix_lookup[f"TRAIN|{prefix}|{frame_idx}"] = img_path
    for (prefix, frame_idx), img_path in test_by_prefix.items():
        prefix_lookup[f"TEST|{prefix}|{frame_idx}"] = img_path

    for (dt_key, frame_idx), img_paths in train_by_dt.items():
        if img_paths:
            dt_lookup[f"TRAIN|{dt_key}|{frame_idx}"] = img_paths[0]
    for (dt_key, frame_idx), img_paths in test_by_dt.items():
        if img_paths:
            dt_lookup[f"TEST|{dt_key}|{frame_idx}"] = img_paths[0]

    df_labels = df_labels.copy()
    df_labels["sample_stem"] = df_labels["path"].apply(_stem_without_ext)
    df_labels["datetime_key"] = df_labels.apply(_datetime_key_from_row, axis=1)

    all_frames = []
   
    for frame_idx in range(N_FRAMES):
        df_frame = df_labels.copy()
        df_frame["frame_idx"] = frame_idx

        # 1) Exact match using original sample stem.
        key_prefix = (
            df_frame["split"].astype(str)
            + "|"
            + df_frame["sample_stem"].astype(str)
            + "|"
            + str(frame_idx)
        )
        df_frame["img_path"] = key_prefix.map(prefix_lookup)

        # 2) Fallback by datetime key where exact prefix did not match.
        missing = df_frame["img_path"].isna() & df_frame["datetime_key"].astype(str).ne("")
        if missing.any():
            key_dt = (
                df_frame.loc[missing, "split"].astype(str)
                + "|"
                + df_frame.loc[missing, "datetime_key"].astype(str)
                + "|"
                + str(frame_idx)
            )
            df_frame.loc[missing, "img_path"] = key_dt.map(dt_lookup)

        df_frame = df_frame[df_frame["img_path"].notna()].copy()

        if df_frame.empty:
            print(f"No matches for frame {frame_idx}; skipping write.")
            continue

        df_frame = df_frame[df_frame["label"] != -100]
        df_frame = df_frame.drop(columns=["sample_stem", "datetime_key"])
        all_frames.append(df_frame)

        n_train = int((df_frame["split"] == "TRAIN").sum()) if "split" in df_frame.columns else 0
        n_test = int((df_frame["split"] == "TEST").sum()) if "split" in df_frame.columns else 0
        print(f"Frame {frame_idx}: matched {len(df_frame)} rows (TRAIN={n_train}, TEST={n_test}).")

    if not all_frames:
        print("No rows matched any frame; nothing to save.")
        return

    # Save all frames in one file.
    df_out = pd.concat(all_frames, ignore_index=True)
    print(f"Saving expanded dataframe to {output_path}...")
    df_out.to_csv(output_path, index=False)

           
# =============================================================================
# MAIN
# =============================================================================
def main():
    items = sorted(CLOUD_CLASS_INFO.items(), key=lambda x: x[1]["order"])
    print(items)
    labels = sorted([label for label, _ in items])
    print(labels)
    print(f"TRAIN image files: {len(TRAIN_LIST_IMAGE_CROPS)}")
    print(f"TEST image files: {len(TEST_LIST_IMAGE_CROPS)}")

    df_feat, df_centroids = load_labels(OUTPUT_PATH, FILENAME_TSNE, FILENAME_LABELS, labels)
    
    df_prepared = df_feat[df_feat["label"] != -100].copy()
    #df_prepared['year'] = df_prepared['path'].apply(lambda x: int(os.path.basename(x).split('-')[0]))
    print(f"n samples after filtering: {len(df_prepared)}")

    substitute_paths(df_prepared, expanded_csv)
    print("Paths substituted and expanded dataframe saved.")


if __name__ == "__main__":
    main()

#47648