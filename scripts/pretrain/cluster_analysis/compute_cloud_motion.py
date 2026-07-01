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
Run on the first 20 GRL training crops:

    conda run -n vissl python scripts/pretrain/cluster_analysis/compute_cloud_motion.py \
        --limit 20 \

Run all crops using the settings in configs/process_run_GRL.yaml:

    conda run -n vissl python scripts/pretrain/cluster_analysis/compute_cloud_motion.py

pid 1941839

"""

import argparse
import csv
import os
import sys
from glob import glob
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT))

from utils.configs import load_config
from utils.processing.cloud_motion import compute_cloud_motion_from_nc

DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "process_run_GRL.yaml"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate mean cloud-motion speed and direction from video NetCDF crops."
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--crop-dir", default=None, help="Directory containing NetCDF crop files.")
    input_group.add_argument("--csv", default=None, help="CSV file containing a crop path column.")

    parser.add_argument("--output", default=None, help="Output CSV path.")
    parser.add_argument("--path-column", default=None, help="Crop path column when using --csv.")
    parser.add_argument("--label-column", default=None, help="Label column when using --csv.")
    parser.add_argument(
        "--video-summary-csv",
        default=None,
        help=(
            "Optional video summary CSV containing crop labels. If omitted, the "
            "path is read from output_files.training_video_summary in --config."
        ),
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="YAML config containing output_files.training_video_summary.",
    )
    parser.add_argument(
        "--video-summary-key",
        default=None,
        help="Key under output_files in --config to use when --video-summary-csv is omitted.",
    )
    parser.add_argument(
        "--summary-crop-column",
        default=None,
        help="Crop filename column in --video-summary-csv.",
    )
    parser.add_argument(
        "--summary-label-column",
        default=None,
        help="Label column in --video-summary-csv.",
    )
    parser.add_argument("--variable", default=None, help="NetCDF variable with masked IR frames.")
    parser.add_argument("--nc-engine", default=None, help="xarray NetCDF engine, e.g. h5netcdf or netcdf4.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N crops.")
    parser.add_argument("--pixel-size-km", type=float, default=None, help="Pixel size in km.")
    parser.add_argument(
        "--frame-interval-minutes",
        type=float,
        default=None,
        help="Minutes between consecutive frames.",
    )
    parser.add_argument(
        "--invalid-value",
        type=float,
        default=None,
        help="Additional value to mask as non-cloud, often 0 for masked imagery.",
    )
    parser.add_argument(
        "--min-valid-fraction",
        type=float,
        default=None,
        help="Minimum fraction of pixels valid in both frames.",
    )
    parser.add_argument(
        "--max-shift-pixels",
        type=float,
        default=None,
        help="Drop frame-pair vectors longer than this many pixels.",
    )
    return parser.parse_args()


def apply_config_defaults(args):
    config = load_config(args.config)
    cloud_motion_config = config.get("cloud_motion", {})
    output_files = config.get("output_files", {})
    features_config = config.get("features_preparation", {})
    data_config = config.get("data", {})

    mode = cloud_motion_config.get("mode", "train")
    output_file_key = cloud_motion_config.get(
        "output_file_key",
        "training_cloud_motion" if mode == "train" else "testing_cloud_motion",
    )

    if args.crop_dir is None and args.csv is None:
        args.csv = cloud_motion_config.get("csv")
        args.crop_dir = cloud_motion_config.get("crop_dir")
        if args.crop_dir is None and args.csv is None:
            args.crop_dir = (
                features_config.get("crops_test_path")
                if mode == "test"
                else features_config.get("crops_path")
            )

    if args.output is None:
        args.output = cloud_motion_config.get("output") or output_files.get(output_file_key)

    if args.video_summary_key is None:
        args.video_summary_key = cloud_motion_config.get(
            "video_summary_key",
            "testing_video_summary" if mode == "test" else "training_video_summary",
        )

    args.path_column = args.path_column or cloud_motion_config.get("path_column", "path")
    args.label_column = args.label_column or cloud_motion_config.get("label_column", "label")
    args.summary_crop_column = (
        args.summary_crop_column or cloud_motion_config.get("summary_crop_column", "crop")
    )
    args.summary_label_column = (
        args.summary_label_column or cloud_motion_config.get("summary_label_column", "label")
    )
    args.variable = args.variable or cloud_motion_config.get("variable", "IR_108")
    args.nc_engine = args.nc_engine or cloud_motion_config.get("nc_engine", data_config.get("nc_engine"))
    args.pixel_size_km = (
        args.pixel_size_km
        if args.pixel_size_km is not None
        else cloud_motion_config.get("pixel_size_km")
    )
    args.frame_interval_minutes = (
        args.frame_interval_minutes
        if args.frame_interval_minutes is not None
        else cloud_motion_config.get("frame_interval_minutes")
    )
    args.invalid_value = (
        args.invalid_value
        if args.invalid_value is not None
        else cloud_motion_config.get("invalid_value")
    )
    args.min_valid_fraction = (
        args.min_valid_fraction
        if args.min_valid_fraction is not None
        else cloud_motion_config.get("min_valid_fraction", 0.05)
    )
    args.max_shift_pixels = (
        args.max_shift_pixels
        if args.max_shift_pixels is not None
        else cloud_motion_config.get("max_shift_pixels")
    )
    args.limit = args.limit if args.limit is not None else cloud_motion_config.get("limit")

    if args.crop_dir is None and args.csv is None:
        raise ValueError(
            "No cloud-motion input configured. Set cloud_motion.crop_dir, "
            "cloud_motion.csv, or features_preparation.crops_path in the config."
        )
    if args.output is None:
        raise ValueError(
            "No cloud-motion output configured. Set cloud_motion.output or "
            f"output_files.{output_file_key} in the config."
        )

    return args


def resolve_video_summary_csv(args):
    if args.video_summary_csv:
        return args.video_summary_csv

    config = load_config(args.config)

    try:
        return config["output_files"][args.video_summary_key]
    except KeyError as exc:
        raise KeyError(
            f"Could not find output_files.{args.video_summary_key} in {args.config}."
        ) from exc


def crop_lookup_keys(crop_name):
    basename = os.path.basename(str(crop_name))
    stem, _ = os.path.splitext(basename)
    return {basename, stem}


def load_label_lookup(args):
    video_summary_csv = resolve_video_summary_csv(args)
    if not video_summary_csv:
        return {}

    with open(video_summary_csv, newline="") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = reader.fieldnames or []
        missing_columns = [
            column
            for column in (args.summary_crop_column, args.summary_label_column)
            if column not in fieldnames
        ]
        if missing_columns:
            raise KeyError(
                f"Column(s) {missing_columns!r} not found in {video_summary_csv}. "
                f"Available columns: {fieldnames}"
            )

        label_lookup = {}
        for row in reader:
            crop_name = row.get(args.summary_crop_column)
            if not crop_name:
                continue
            label = row.get(args.summary_label_column, "")
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


def collect_crop_records(args):
    label_lookup = load_label_lookup(args)

    if args.crop_dir:
        paths = sorted(glob(os.path.join(args.crop_dir, "*.nc")))
        records = [
            {"path": path, "label": get_label_for_path(path, "", label_lookup)}
            for path in paths
        ]
    else:
        with open(args.csv, newline="") as input_file:
            reader = csv.DictReader(input_file)
            fieldnames = reader.fieldnames or []
            if args.path_column not in fieldnames:
                raise KeyError(
                    f"Column {args.path_column!r} not found in {args.csv}. "
                    f"Available columns: {reader.fieldnames}"
                )
            has_label_column = args.label_column in fieldnames
            records = [
                {
                    "path": row[args.path_column],
                    "label": get_label_for_path(
                        row[args.path_column],
                        row[args.label_column] if has_label_column else "",
                        label_lookup,
                    ),
                }
                for row in reader
                if row.get(args.path_column)
            ]

    if args.limit is not None:
        records = records[: args.limit]

    if not records:
        raise ValueError("No crop paths found to process.")

    return records


def main():
    args = apply_config_defaults(parse_args())
    crop_records = collect_crop_records(args)

    rows = []
    for idx, crop_record in enumerate(crop_records, start=1):
        path = crop_record["path"]
        print(f"[{idx}/{len(crop_records)}] {path}")
        row = {"path": path, "label": crop_record["label"]}
        try:
            result = compute_cloud_motion_from_nc(
                path,
                variable=args.variable,
                pixel_size_km=args.pixel_size_km,
                frame_interval_minutes=args.frame_interval_minutes,
                nc_engine=args.nc_engine,
                invalid_value=args.invalid_value,
                min_valid_fraction=args.min_valid_fraction,
                max_shift_pixels=args.max_shift_pixels,
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

    output_dir = os.path.dirname(args.output)
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
    with open(args.output, "w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved cloud-motion results to {args.output}")


if __name__ == "__main__":
    main()
