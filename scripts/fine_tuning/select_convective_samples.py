"""Select convective-related samples from VISSL features using YAML-driven run settings."""

import os
import shutil
import sys
from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(str(Path(__file__).resolve().parents[3]))

from utils.configs import load_config
from utils.read.features_reader import load_features_to_df


def build_runtime_config(config: dict) -> dict:
    """Build script runtime configuration from process_run_config.yaml."""
    experiment_cfg = config["experiment"]
    data_cfg = config["data"]
    conv_cfg = config.get("convective_selection", {})

    run_name = experiment_cfg["run_names"][0]
    epoch = experiment_cfg["epoch"]
    base_path = experiment_cfg["base_path"]
    path_out = experiment_cfg["path_out"]

    data_base_path = data_cfg["data_base_path"]
    crops_name = data_cfg["crops_name"]
    file_extension = data_cfg.get("file_extension", "nc")

    crop_split = str(conv_cfg.get("crop_split", "1"))
    output_suffix = conv_cfg.get("output_suffix", "convective_selected")

    return {
        "run_name": run_name,
        "train_feat_dir": os.path.join(base_path, run_name, "features", f"epoch_{epoch}"),
        "output_path": os.path.join(path_out, run_name, output_suffix),
        "image_train_path": os.path.join(data_base_path, crops_name, file_extension, crop_split),
        "output_nc_dir": os.path.join(data_base_path, crops_name, f"{file_extension}_convective", crop_split),
        "feature_file_train_inds": conv_cfg.get("feature_file_train_inds", "rank0_chunk0_train_heads_inds.npy"),
        "feature_file_train_features": conv_cfg.get("feature_file_train_features", "rank0_chunk0_train_heads_features.npy"),
        "train_assignments_file": os.path.join(base_path, run_name, "checkpoints", conv_cfg.get("train_assignments_file", "assignments.pt")),
        "train_distances_file": os.path.join(base_path, run_name, "checkpoints", conv_cfg.get("train_distances_file", "distances.pt")),
        "train_centroids_file": os.path.join(base_path, run_name, "checkpoints", conv_cfg.get("train_centroids_file", "centroids0.pt")),
        "file_extension": file_extension,
        "convective_classes": conv_cfg.get("convective_classes", [6, 1, 5]),
        "distance_threshold": conv_cfg.get("distance_threshold", 0.5),
    }



def select_convective_samples(df, centroids, convective_classes, dist_thresh):
    """Select convective samples + samples close to convective centroids."""

    # ---- get convective centroids ----
    conv_centroids = centroids[convective_classes]  # shape (#conv, 128)

    feature_cols = [c for c in df.columns if c.startswith("dim_")]
    feat_matrix = df[feature_cols].values

    # ---- compute cosine distance to each convective centroid ----
    dist_matrix = cosine_similarity(feat_matrix, conv_centroids)
    #print(dist_matrix.shape)
    #print(dist_matrix[:, :])
 
    min_dist = dist_matrix.max(axis=1)
    closest_centroid_idx = dist_matrix.argmax(axis=1)

    df["dist_to_convective"] = min_dist
    df["closest_conv_centroid"] = [convective_classes[i] for i in closest_centroid_idx]

    # ---- select ----
    df_convective = df[
        (df["label"].isin(convective_classes)) |
        (df["dist_to_convective"] >= dist_thresh)
    ].copy()

    #remove invalid labels
    df_convective = df_convective[df_convective["label"] != -100]

    print("Convective class distribution after selection:")
    print(df_convective["label"].value_counts())

    return df_convective



def copy_nc_files(df, dest_dir):
    """Copy selected .nc files to destination folder."""
    for p in df["path"]:
        if os.path.exists(p):
            shutil.copy(p, dest_dir)



def main():
    repo_root = Path(__file__).resolve().parents[3]
    default_config_path = repo_root / "configs" / "process_run_config.yaml"
    config_path = Path(os.getenv("PROCESS_RUN_CONFIG", str(default_config_path)))

    config = load_config(config_path)
    runtime = build_runtime_config(config)

    os.makedirs(runtime["output_path"], exist_ok=True)
    os.makedirs(runtime["output_nc_dir"], exist_ok=True)

    print("Loading features...")
    df, centroids = load_features_to_df(
        feature_path=runtime["train_feat_dir"],
        indices_file=runtime["feature_file_train_inds"],
        features_file=runtime["feature_file_train_features"],
        assignments_file=runtime["train_assignments_file"],
        distances_file=runtime["train_distances_file"],
        centroids_file=runtime["train_centroids_file"],
        crops_path=runtime["image_train_path"],
        file_extension=runtime["file_extension"],
    )
    print(f"Loaded {len(df)} samples.")
    print(df.head())
    print(centroids.shape)

    #count how many samples belong to the convective classes
    print("Convective class distribution before selection:")
    print(df[df["label"].isin(runtime["convective_classes"])]["label"].value_counts())
    

    print("Selecting convective samples...")
    df_sel = select_convective_samples(
        df=df,
        centroids=centroids,
        convective_classes=runtime["convective_classes"],
        dist_thresh=runtime["distance_threshold"],
    )

    print(f"Selected {len(df_sel)} convective-related samples.")
    #print(df_sel.head())

    # ==== SAVE CSV ====
    output_csv = os.path.join(runtime["output_path"], f"convective_selected_{runtime['run_name']}.csv")
    df_sel.to_csv(output_csv, index=False)
    print(f"Saved selected samples: {output_csv}")

    # ==== COPY FILES ====
    print("Copying .nc files...")
    copy_nc_files(df_sel, runtime["output_nc_dir"])
    print(f"Done. Copied files to {runtime['output_nc_dir']}")


if __name__ == "__main__":
    main()
