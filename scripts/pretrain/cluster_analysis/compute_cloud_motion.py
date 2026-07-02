"""
Compute mean cloud motion for video NetCDF crops. It uses OpenCV phase correlation: Phase correlation estimates 
the translation needed to align frame 1 with frame 2.
OpenCV means Open Source Computer Vision Library. It is a widely used library for image and video processing.
the function cv2.phaseCorrelate(arr1, arr2, window) compares two image frames and estimates how much 
the image pattern moved between them. For your cloud videos, it estimates the bulk cloud displacement between consecutive frames.
So in this project, OpenCV is not doing machine learning. It is being used as an image-processing tool to calculate
How many pixels did the cloud field shift from frame t to frame t+1.
it is related but not exactly as optical flow in the fact that it uses phase correlation and estimates one bulk
translation vector between two frames, rather than a dense flow field and assumes that the cloud patterns mostly move 
as one coherent object. Optical flow instead can capture deformation, rotation, shear etc.. for this goal, it provides 
a mean cloud motion per video. 

It works as follows: The velocity is calculated in [utils/processing/cloud_motion.py] not directly in the wrapper script.
For each NetCDF crop:
The script loads the video variable, by default IR_108, as a 3D array: It compares each pair of consecutive frames:
So for 8 frames, it estimates motion for 7 frame pairs.
For each frame pair, it masks invalid pixels and normalizes the valid pixels. Invalid means NaN, and optionally 
invalid_value if you set it in the config. 
The valid pixels are normalized by subtracting their mean and dividing by their standard deviation
So it gives the displacement vector (dx, dy) for each frame pair. Positive dx means motion to the right. 
 Positive dy means motion downward in image coordinates. The code averages all valid pair displacements:
The output columns of the CSV file are:
mean_dx_pixels_per_frame      average x displacement
mean_dy_pixels_per_frame      average y displacement
mean_speed_pixels_per_frame   magnitude of average displacement vector
mean_speed_kmh                converted physical speed
mean_direction_to_deg         direction cloud is moving toward
mean_direction_from_deg       opposite direction, like wind-from direction
n_pairs_used                  number of valid frame pairs, usually 7
mean_response                 phase-correlation confidence/quality

Example
-------
Run using the settings in configs/process_run_GRL.yaml:

    conda run -n vissl python scripts/pretrain/cluster_analysis/compute_cloud_motion.py

pid 1941839
pid for test 2098405
"""

import csv
import os
import sys
from glob import glob
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT))

from utils.configs import load_config
from utils.processing.cloud_motion import compute_cloud_motion_from_nc

config_path = "/home/claudia/codes/ML_postprocessing/configs/process_run_GRL.yaml"
config = load_config(config_path)

cloud_motion_config = config["cloud_motion"]
output_files = config["output_files"]
data_config = config["data"]

mode = cloud_motion_config["mode"]
if mode == "train":
    CSV_PATH = cloud_motion_config["csv_train"]
    PATH_ROOT = cloud_motion_config["path_root_train"]
    OUTPUT_FILE_KEY = cloud_motion_config["output_file_key_train"]
    VIDEO_SUMMARY_KEY = cloud_motion_config["video_summary_key_train"]
elif mode == "test":
    CSV_PATH = cloud_motion_config["csv_test"]
    PATH_ROOT = cloud_motion_config["path_root_test"]
    OUTPUT_FILE_KEY = cloud_motion_config["output_file_key_test"]
    VIDEO_SUMMARY_KEY = cloud_motion_config["video_summary_key_test"]
else:
    raise ValueError(f"Invalid cloud_motion.mode: {mode}. Expected 'train' or 'test'.")

OUTPUT_CSV = output_files[OUTPUT_FILE_KEY]
VIDEO_SUMMARY_CSV = output_files[VIDEO_SUMMARY_KEY]

PATH_COLUMN = cloud_motion_config.get("path_column", "path")
LABEL_COLUMN = cloud_motion_config.get("label_column", "label")
SUMMARY_CROP_COLUMN = cloud_motion_config.get("summary_crop_column", "crop")
SUMMARY_LABEL_COLUMN = cloud_motion_config.get("summary_label_column", "label")

VARIABLE = cloud_motion_config.get("variable", "IR_108")
NC_ENGINE = cloud_motion_config.get("nc_engine", data_config.get("nc_engine"))
PIXEL_SIZE_KM = cloud_motion_config.get("pixel_size_km")
FRAME_INTERVAL_MINUTES = cloud_motion_config.get("frame_interval_minutes")
INVALID_VALUE = cloud_motion_config.get("invalid_value")
MIN_VALID_FRACTION = cloud_motion_config.get("min_valid_fraction", 0.05)
MAX_SHIFT_PIXELS = cloud_motion_config.get("max_shift_pixels")
LIMIT = cloud_motion_config.get("limit")

def crop_lookup_keys(crop_name):
    basename = os.path.basename(str(crop_name))
    stem, _ = os.path.splitext(basename)
    return {basename, stem}


def load_label_lookup():
    if not VIDEO_SUMMARY_CSV:
        return {}

    with open(VIDEO_SUMMARY_CSV, newline="") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = reader.fieldnames or []
        missing_columns = [
            column
            for column in (SUMMARY_CROP_COLUMN, SUMMARY_LABEL_COLUMN)
            if column not in fieldnames
        ]
        if missing_columns:
            raise KeyError(
                f"Column(s) {missing_columns!r} not found in {VIDEO_SUMMARY_CSV}. "
                f"Available columns: {fieldnames}"
            )

        label_lookup = {}
        for row in reader:
            crop_name = row.get(SUMMARY_CROP_COLUMN)
            if not crop_name:
                continue
            label = row.get(SUMMARY_LABEL_COLUMN, "")
            for key in crop_lookup_keys(crop_name):
                label_lookup[key] = label

    return label_lookup


def get_label_for_path(path, current_label, label_lookup):
    if current_label not in ("", None):
        return current_label
    for key in crop_lookup_keys(path):
        if key in label_lookup:
            return label_lookup[key]
    return ""


def build_crop_path_lookup(path_root):
    if not path_root:
        return {}

    crop_paths = glob(os.path.join(path_root, "**", "*.nc"), recursive=True)
    return {os.path.basename(path): path for path in crop_paths}


def resolve_crop_path(path, crop_path_lookup):
    if os.path.isabs(path) or not crop_path_lookup:
        return path
    return crop_path_lookup.get(os.path.basename(path), path)


def collect_crop_records():
    label_lookup = load_label_lookup()
    crop_path_lookup = build_crop_path_lookup(PATH_ROOT)

    with open(CSV_PATH, newline="") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = reader.fieldnames or []
        if PATH_COLUMN not in fieldnames:
            raise KeyError(
                f"Column {PATH_COLUMN!r} not found in {CSV_PATH}. "
                f"Available columns: {reader.fieldnames}"
            )
        has_label_column = LABEL_COLUMN in fieldnames
        records = [
            {
                "path": resolve_crop_path(row[PATH_COLUMN], crop_path_lookup),
                "label": get_label_for_path(
                    row[PATH_COLUMN],
                    row[LABEL_COLUMN] if has_label_column else "",
                    label_lookup,
                ),
            }
            for row in reader
            if row.get(PATH_COLUMN)
        ]

    if LIMIT is not None:
        records = records[:LIMIT]

    if not records:
        raise ValueError("No crop paths found to process.")

    return records


def main():
    crop_records = collect_crop_records()

    rows = []
    for idx, crop_record in enumerate(crop_records, start=1):
        path = crop_record["path"]
        print(f"[{idx}/{len(crop_records)}] {path}")
        row = {"path": path, "label": crop_record["label"]}
        try:
            result = compute_cloud_motion_from_nc(
                path,
                variable=VARIABLE,
                pixel_size_km=PIXEL_SIZE_KM,
                frame_interval_minutes=FRAME_INTERVAL_MINUTES,
                nc_engine=NC_ENGINE,
                invalid_value=INVALID_VALUE,
                min_valid_fraction=MIN_VALID_FRACTION,
                max_shift_pixels=MAX_SHIFT_PIXELS,
            )
            row.update(result.as_dict())
        except Exception as exc:
            row.update(
                {
                    "mean_dx_pixels_per_frame": float("nan"),
                    "mean_dy_pixels_per_frame": float("nan"),
                    "mean_speed_pixels_per_frame": float("nan"),
                    "mean_speed_kmh": float("nan"),
                    "mean_direction_to_deg": float("nan"),
                    "mean_direction_from_deg": float("nan"),
                    "n_pairs_used": 0,
                    "mean_response": float("nan"),
                    "error": repr(exc),
                }
            )
        rows.append(row)

    output_dir = os.path.dirname(OUTPUT_CSV)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fieldnames = [
        "path",
        "label",
        "mean_dx_pixels_per_frame",
        "mean_dy_pixels_per_frame",
        "mean_speed_pixels_per_frame",
        "mean_speed_kmh",
        "mean_direction_to_deg",
        "mean_direction_from_deg",
        "n_pairs_used",
        "mean_response",
        "error",
    ]
    with open(OUTPUT_CSV, "w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved cloud-motion results to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
