"""
Analyze the 2025-06-16 case study and test warning-threshold behavior.

The script selects one test-data view for the case study (currently view002),
loads all video crops for that date and view, and computes precursor anomaly
time series for selected variables. Each anomaly is calculated as:

    test crop value - training mean for the same class label

The main output figure contains:
    - a top class-sequence strip with one colored interval per 2-hour video crop
    - precursor anomaly time series for the selected view
    - vertical markers for ESSL report timestamps located inside the selected
      view footprint
    - red x markers where finite differences in the anomaly time series exceed
      warning thresholds from the south-daytime lookup table

Optional plotting code can also create a class lookup table for the selected
date and georeferenced per-view MP4 videos. Temporary PNG frames are deleted
after each MP4 is created successfully.

Inputs come from process_run_GRL.yaml:
    - output_files.testing_video_summary
    - output_files.training_video_summary
    - output_files.warning_south_day
    - output_files.ancillary_files.essl_dataset
    - features_preparation.crops_test_path
"""


import sys
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import xarray as xr
from pathlib import Path
from matplotlib.patches import Patch, Rectangle

sys.path.append("/home/claudia/codes/ML_postprocessing")

from utils.configs import load_config
from utils.plotting.plot_class_analysis import plot_hourly_histogram, style_axis
from utils.plotting.class_colors import colors_per_class1_names

# read filename of csv files from config file process_run_GRL.yaml
config_path = "/home/claudia/codes/ML_postprocessing/configs/process_run_GRL.yaml"
config = load_config(config_path)
CSV_FILES = {
    "training": config["output_files"]["training_video_summary"],
    "testing": config["output_files"]["testing_video_summary"],
    "training_motion": config["output_files"]["training_cloud_motion"],
    "testing_motion": config["output_files"]["testing_cloud_motion"],
    "essl_reports": config["output_files"]["ancillary_files"]["essl_dataset"],

}

# create output directory if it doesn't exist
OUTPUT_DIR = Path(config["output_files"]["figures_dir"]) / "case_study_analysis"
VIEW_VIDEO_OUTPUT_DIR = OUTPUT_DIR / "view_day_videos"
CROP_ROOT = Path(config["features_preparation"]["crops_test_path"])
VIDEO_VARIABLE = "IR_108"
VIDEO_FPS = 3
TITLE_FONTSIZE = 18
AXIS_LABEL_FONTSIZE = 15
TICK_LABEL_FONTSIZE = 13
LEGEND_FONTSIZE = 12
CLASS_LABEL_FONTSIZE = 12
REPORT_MARKER_SPECS = {
    "HAIL": {"marker": "o", "color": "#f28e2b", "label": "HAIL report"},
    "PRECIP": {"marker": "s", "color": "#2b7cd2", "label": "PRECIP report"},
}
REPORT_MARKER_SIZE = 360
REPORT_DISPLAY_FRAMES = 4
DEM_PATH = "/data1/DEM_EXPATS_0.01x0.01.nc"
OROGRAPHY_VARIABLE_CANDIDATES = (
    "orography",
    "orog",
    "orography_m",
    "elevation",
    "altitude",
    "surface_altitude",
    "HSURF",
    "height",
    "dem",
    "DEM",
)


def plot_view_hourly_class_histogram(view_crops, ax, title):
    view_crops = view_crops.copy()
    view_crops["label"] = pd.to_numeric(view_crops["label"], errors="coerce")
    view_crops["time_start"] = pd.to_datetime(view_crops["time_start"], errors="coerce")
    view_crops = view_crops.dropna(subset=["label", "time_start"])
    view_crops = view_crops[view_crops["label"] != -100].copy()
    view_crops["label"] = view_crops["label"].astype(int)
    view_crops["hour"] = view_crops["time_start"].dt.hour

    hourly_counts = (
        view_crops.groupby(["hour", "label"]).size().unstack(fill_value=0)
    )
    hours = np.arange(24)
    hourly_counts = hourly_counts.reindex(hours, fill_value=0)

    for label in sorted(hourly_counts.columns):
        color = colors_per_class1_names.get(str(label), None)
        plot_hourly_histogram(
            ax,
            hours,
            hourly_counts[label].to_numpy(),
            color,
            f"Class {label}",
        )

    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Number of crops")
    ax.set_title(title)
    ax.set_xticks(hours)
    ax.legend(frameon=False, ncol=3)


def plot_views_class_lookup_table(test_crops_day, case_study_date, output_path):
    lookup_df = test_crops_day.copy()
    lookup_df["label"] = pd.to_numeric(lookup_df["label"], errors="coerce")
    lookup_df["time_start"] = pd.to_datetime(
        lookup_df["time_start"], errors="coerce"
    )
    lookup_df["time_end"] = pd.to_datetime(lookup_df["time_end"], errors="coerce")
    lookup_df = lookup_df.dropna(subset=["view", "time_start", "time_end", "label"])
    lookup_df["label"] = lookup_df["label"].astype(int)
    lookup_df = lookup_df[lookup_df["label"] != -100].copy()

    views = sorted(lookup_df["view"].unique())
    if not views or lookup_df.empty:
        print("No class labels available for view lookup table; skipping plot.")
        return

    start_time = lookup_df["time_start"].min()
    end_time = lookup_df["time_end"].max()
    duration_hours = max((end_time - start_time).total_seconds() / 3600, 1)
    fig_width = max(10, 0.55 * duration_hours)
    fig_height = max(3.5, 0.45 * len(views) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    ax.set_title(f"Class lookup table for {case_study_date}", fontsize=14)
    ax.set_xlabel("Time")
    ax.set_ylabel("View")

    view_to_y = {view: row_index for row_index, view in enumerate(views)}
    for _, row in lookup_df.iterrows():
        y_position = view_to_y[row["view"]]
        x_start = mdates.date2num(row["time_start"])
        x_end = mdates.date2num(row["time_end"])
        label = int(row["label"])
        color = colors_per_class1_names.get(str(label), "lightgray")
        rect = Rectangle(
            (x_start, y_position - 0.45),
            x_end - x_start,
            0.9,
            facecolor=color,
            edgecolor="white",
            linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(
            x_start + (x_end - x_start) / 2,
            y_position,
            str(label),
            ha="center",
            va="center",
            fontsize=9,
            color="black",
        )

    ax.set_xlim(start_time, end_time)
    ax.set_ylim(-0.5, len(views) - 0.5)
    ax.set_yticks(np.arange(len(views)))
    ax.set_yticklabels(views)
    ax.invert_yaxis()

    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.tick_params(axis="x", rotation=35)
    ax.set_yticks(np.arange(-0.5, len(views), 1), minor=True)
    ax.grid(axis="x", color="0.85", linestyle="--", linewidth=0.8)
    ax.grid(which="minor", axis="y", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    legend_labels = sorted(
        int(label)
        for label in pd.unique(lookup_df["label"])
        if str(int(label)) in colors_per_class1_names
    )
    legend_handles = [
        Patch(
            facecolor=colors_per_class1_names[str(label)],
            edgecolor="none",
            label=f"Class {label}",
        )
        for label in legend_labels
    ]
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            frameon=False,
            ncol=2,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved class lookup table to {output_path}")


def build_crop_path_lookup(crop_root, case_study_date):
    return {
        crop_path.name: crop_path
        for crop_path in crop_root.rglob(f"{case_study_date}*.nc")
    }


def get_crop_paths(crops_df, crop_path_lookup):
    crop_paths = []
    missing_crops = []
    for crop_name in crops_df["crop"]:
        crop_path = crop_path_lookup.get(Path(crop_name).name)
        if crop_path is None:
            missing_crops.append(crop_name)
        else:
            crop_paths.append(crop_path)

    if missing_crops:
        print(f"Missing {len(missing_crops)} crop files; first missing: {missing_crops[0]}")
    return crop_paths


def get_video_data_array(ds, variable=VIDEO_VARIABLE):
    if variable in ds:
        return ds[variable]
    if variable == "IR_108" and "IR_108_masked" in ds:
        return ds["IR_108_masked"]
    raise KeyError(
        f"{variable} not found in crop. Available variables: {list(ds.data_vars)}"
    )


def get_frame_count(ds, variable=VIDEO_VARIABLE):
    data_array = get_video_data_array(ds, variable)
    return data_array.sizes.get("time", 1)


def get_frame_values(ds, frame_index, variable=VIDEO_VARIABLE):
    data_array = get_video_data_array(ds, variable)
    if "time" in data_array.dims:
        data_array = data_array.isel(time=frame_index)
    return np.asarray(data_array.values, dtype=float)


def get_frame_lat_lon(ds, frame_index):
    lat = ds["lat"]
    lon = ds["lon"]
    if "time" in lat.dims:
        lat = lat.isel(time=frame_index)
    if "time" in lon.dims:
        lon = lon.isel(time=frame_index)
    return lat, lon


def find_orography_field(ds):
    lower_name_map = {name.lower(): name for name in ds.data_vars}
    for candidate in OROGRAPHY_VARIABLE_CANDIDATES:
        var_name = lower_name_map.get(candidate.lower())
        if var_name and {"lat", "lon"} <= set(ds[var_name].dims):
            orography = ds[var_name]
            if "time" in orography.dims:
                orography = orography.isel(time=0)
            return orography

    for _, data_array in ds.data_vars.items():
        if {"lat", "lon"} <= set(data_array.dims):
            if "time" in data_array.dims:
                data_array = data_array.isel(time=0)
            return data_array
    return None


def load_orography(path=DEM_PATH):
    if not Path(path).exists():
        print(f"DEM file not found at {path}; orography overlay will be skipped.")
        return None

    try:
        ds_dem = xr.open_dataset(path)
    except Exception as exc:
        print(f"Could not open DEM file {path}: {exc}; orography overlay will be skipped.")
        return None
    return find_orography_field(ds_dem)


def get_lat_lon_extent(lat, lon):
    lat_values = np.asarray(lat.values, dtype=float)
    lon_values = np.asarray(lon.values, dtype=float)
    return (
        float(np.nanmin(lon_values)),
        float(np.nanmax(lon_values)),
        float(np.nanmin(lat_values)),
        float(np.nanmax(lat_values)),
    )


def pad_extent(extent, pad_fraction=0.03):
    lon_min, lon_max, lat_min, lat_max = extent
    lon_pad = max((lon_max - lon_min) * pad_fraction, 0.05)
    lat_pad = max((lat_max - lat_min) * pad_fraction, 0.05)
    return (
        lon_min - lon_pad,
        lon_max + lon_pad,
        lat_min - lat_pad,
        lat_max + lat_pad,
    )


def plot_orography_contours(ax, orography, extent):
    if orography is None or not {"lat", "lon"} <= set(orography.dims):
        return

    lon_min, lon_max, lat_min, lat_max = extent
    oro = orography.where(
        (orography.lon >= lon_min)
        & (orography.lon <= lon_max)
        & (orography.lat >= lat_min)
        & (orography.lat <= lat_max),
        drop=True,
    )
    if oro.size == 0:
        return

    data = np.asarray(oro.values).squeeze()
    if data.ndim != 2 or np.all(np.isnan(data)):
        return

    levels = np.arange(0, 4001, 500)
    levels = levels[(levels >= np.nanmin(data)) & (levels <= max(4000, np.nanmax(data)))]
    if len(levels) < 2:
        return

    contours = ax.contour(
        oro.lon.values,
        oro.lat.values,
        data,
        levels=levels,
        colors="0.35",
        linewidths=0.7,
        alpha=0.65,
        zorder=3,
    )
    ax.clabel(contours, inline=True, fontsize=7, fmt="%d m")


def calculate_view_color_limits(view_crops, crop_path_lookup, variable=VIDEO_VARIABLE):
    sampled_values = []
    crop_paths = get_crop_paths(view_crops, crop_path_lookup)
    for crop_path in crop_paths:
        with xr.open_dataset(crop_path) as ds:
            values = np.asarray(get_video_data_array(ds, variable).values, dtype=float)
            finite_values = values[np.isfinite(values)]
            if finite_values.size:
                sampled_values.append(finite_values)

    if not sampled_values:
        return None, None

    all_values = np.concatenate(sampled_values)
    return np.nanpercentile(all_values, [2, 98])


def get_frame_time_window(row, frame_index, n_frames):
    time_start = pd.to_datetime(row["time_start"], errors="coerce")
    time_end = pd.to_datetime(row["time_end"], errors="coerce")
    if pd.isna(time_start) or pd.isna(time_end) or n_frames <= 0:
        return pd.NaT, pd.NaT

    crop_duration = time_end - time_start
    frame_start = time_start + crop_duration * frame_index / n_frames
    frame_end = time_start + crop_duration * (frame_index + 1) / n_frames
    return frame_start, frame_end


def select_reports_for_frame(reports, frame_start, frame_end, is_last_frame=False):
    if reports is None or reports.empty or pd.isna(frame_start) or pd.isna(frame_end):
        return pd.DataFrame()

    frame_duration = frame_end - frame_start
    visible_start = frame_start - frame_duration * max(REPORT_DISPLAY_FRAMES - 1, 0)
    if is_last_frame:
        time_mask = (
            (reports["time_event_naive"] >= visible_start)
            & (reports["time_event_naive"] <= frame_end)
        )
    else:
        time_mask = (
            (reports["time_event_naive"] >= visible_start)
            & (reports["time_event_naive"] < frame_end)
        )
    return reports.loc[time_mask].copy()


def normalize_report_category(report_type):
    report_type = str(report_type).upper()
    if "HAIL" in report_type:
        return "HAIL"
    if "PRECIP" in report_type or "RAIN" in report_type:
        return "PRECIP"
    return None


def add_report_markers_to_frame(ax, reports):
    if reports is None or reports.empty:
        return

    for report_type, spec in REPORT_MARKER_SPECS.items():
        type_reports = reports[reports["report_category"] == report_type]
        if type_reports.empty:
            continue

        ax.scatter(
            type_reports["LONGITUDE"],
            type_reports["LATITUDE"],
            marker=spec["marker"],
            s=REPORT_MARKER_SIZE,
            facecolor=spec["color"],
            edgecolor="black",
            linewidth=1.8,
            alpha=0.95,
            zorder=20,
            clip_on=False,
            label=spec["label"],
        )
        for _, report in type_reports.iterrows():
            ax.annotate(
                report["time_event_naive"].strftime("%H:%M"),
                (report["LONGITUDE"], report["LATITUDE"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=11,
                fontweight="bold",
                color="black",
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.75,
                },
                zorder=21,
            )

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper right", framealpha=0.8, fontsize=8)


def plot_crop_frame(
    row,
    crop_path,
    frame_index,
    n_frames,
    output_path,
    vmin,
    vmax,
    orography,
    reports=None,
):
    with xr.open_dataset(crop_path) as ds:
        frame = get_frame_values(ds, frame_index)
        lat_frame, lon_frame = get_frame_lat_lon(ds, frame_index)
        extent = pad_extent(get_lat_lon_extent(lat_frame, lon_frame))

    frame_start, frame_end = get_frame_time_window(row, frame_index, n_frames)
    frame_reports = select_reports_for_frame(
        reports,
        frame_start,
        frame_end,
        is_last_frame=frame_index == n_frames - 1,
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.grid(True, color="0.88", linewidth=0.8)
    image = ax.pcolormesh(
        lon_frame.values,
        lat_frame.values,
        frame,
        cmap="gray_r",
        vmin=vmin,
        vmax=vmax,
        shading="auto",
        zorder=1,
    )
    plot_orography_contours(ax, orography, extent)
    add_report_markers_to_frame(ax, frame_reports)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        (
            f"{row['view']} | {frame_start} to {frame_end}\n"
            f"frame {frame_index + 1} | class {int(row['label'])}"
        ),
        fontsize=11,
    )
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(VIDEO_VARIABLE)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def save_mp4_from_frames(frame_paths, output_path, fps=VIDEO_FPS):
    try:
        import cv2
    except ImportError:
        print("OpenCV is not available; frame PNGs were saved but MP4 was skipped.")
        return None

    if not frame_paths:
        return None

    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
        print(f"Could not read first frame {frame_paths[0]}; MP4 was skipped.")
        return None

    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"Skipping unreadable frame {frame_path}")
            continue
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)

    writer.release()
    print(f"Saved view video to {output_path}")
    return output_path


def delete_frame_pngs(frame_paths):
    deleted_count = 0
    for frame_path in frame_paths:
        try:
            frame_path.unlink()
            deleted_count += 1
        except FileNotFoundError:
            continue
    print(f"Deleted {deleted_count} temporary frame PNGs.")


def plot_view_day_video(
    view_crops,
    crop_path_lookup,
    case_study_date,
    view,
    orography,
    reports=None,
):
    view_crops = view_crops.sort_values("time_start").reset_index(drop=True)
    frame_output_dir = VIEW_VIDEO_OUTPUT_DIR / f"{case_study_date}_{view}_frames"
    frame_output_dir.mkdir(parents=True, exist_ok=True)

    vmin, vmax = calculate_view_color_limits(view_crops, crop_path_lookup)
    frame_paths = []
    frame_counter = 0
    for _, row in view_crops.iterrows():
        crop_path = crop_path_lookup.get(Path(row["crop"]).name)
        if crop_path is None:
            print(f"Missing crop file for {row['crop']}; skipping.")
            continue

        with xr.open_dataset(crop_path) as ds:
            n_frames = get_frame_count(ds)

        for frame_index in range(n_frames):
            output_path = frame_output_dir / f"{frame_counter:04d}.png"
            plot_crop_frame(
                row,
                crop_path,
                frame_index,
                n_frames,
                output_path,
                vmin,
                vmax,
                orography,
                reports=reports,
            )
            frame_paths.append(output_path)
            frame_counter += 1

    output_video_path = VIEW_VIDEO_OUTPUT_DIR / f"{case_study_date}_{view}.mp4"
    video_path = save_mp4_from_frames(frame_paths, output_video_path)
    if video_path is not None:
        delete_frame_pngs(frame_paths)


def plot_case_day_videos_by_view(test_crops_day, case_study_date):
    VIEW_VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    orography = load_orography()
    crop_path_lookup = build_crop_path_lookup(CROP_ROOT, case_study_date)
    if not crop_path_lookup:
        raise FileNotFoundError(
            f"No NetCDF crops found for {case_study_date} under {CROP_ROOT}"
        )

    for view in sorted(test_crops_day["view"].dropna().unique()):
        view_crops = test_crops_day[test_crops_day["view"] == view].copy()
        reports_in_view = load_case_reports_in_view(
            CSV_FILES["essl_reports"],
            case_study_date,
            view_crops,
        )
        print(f"Creating day-long video for {case_study_date} {view}")
        plot_view_day_video(
            view_crops,
            crop_path_lookup,
            case_study_date,
            view,
            orography,
            reports=reports_in_view,
        )


def build_class_anomaly_time_series_dataset(case_df, training_mean_df, variables):
    case_df = case_df.copy()
    case_df["label"] = pd.to_numeric(case_df["label"], errors="coerce")
    case_df = case_df.dropna(subset=["time_start", "label"])
    case_df["label"] = case_df["label"].astype(int)
    case_df = case_df[case_df["label"] != -100].copy()

    class_labels = sorted(
        label
        for label in training_mean_df["label"].dropna().astype(int).unique()
        if label != -100
    )
    time_starts = sorted(case_df["time_start"].unique())
    time_ends = (
        case_df.groupby("time_start")["time_end"]
        .first()
        .reindex(time_starts)
        .to_numpy()
    )

    coords = {
        "time_start": time_starts,
        "class_label": class_labels,
        "time_end": ("time_start", time_ends),
    }
    anomaly_ds = xr.Dataset(coords=coords)

    for variable in variables:
        if variable not in case_df.columns or variable not in training_mean_df.columns:
            print(f"{variable} not available; skipping anomaly time series.")
            continue

        training_mean_by_label = training_mean_df.set_index("label")[variable]
        rows = []
        for class_label in class_labels:
            class_df = case_df[case_df["label"] == class_label].copy()
            if class_df.empty:
                rows.append(pd.Series(index=time_starts, dtype=float))
                continue

            class_df[variable] = pd.to_numeric(class_df[variable], errors="coerce")
            class_mean = training_mean_by_label.get(class_label, np.nan)
            if pd.isna(class_mean):
                print(
                    f"No training mean for {variable} class {class_label}; "
                    "its anomaly series will be NaN."
                )

            anomaly_series = (
                class_df.groupby("time_start")[variable].mean() - class_mean
            ).reindex(time_starts)
            rows.append(anomaly_series)

        anomaly_values = np.vstack([row.to_numpy(dtype=float) for row in rows]).T
        anomaly_ds[f"{variable}_anomaly"] = (
            ("time_start", "class_label"),
            anomaly_values,
        )

    return anomaly_ds


def calculate_view_extent_from_crop_files(view_crops, case_study_date):
    crop_path_lookup = build_crop_path_lookup(CROP_ROOT, case_study_date)
    crop_paths = get_crop_paths(view_crops, crop_path_lookup)
    if not crop_paths:
        return None

    lat_min_values = []
    lat_max_values = []
    lon_min_values = []
    lon_max_values = []
    for crop_path in crop_paths:
        with xr.open_dataset(crop_path) as ds:
            lat, lon = get_frame_lat_lon(ds, frame_index=0)
            lon_min, lon_max, lat_min, lat_max = get_lat_lon_extent(lat, lon)
            lat_min_values.append(lat_min)
            lat_max_values.append(lat_max)
            lon_min_values.append(lon_min)
            lon_max_values.append(lon_max)

    return (
        min(lon_min_values),
        max(lon_max_values),
        min(lat_min_values),
        max(lat_max_values),
    )


def load_case_reports_in_view(reports_csv, case_study_date, view_crops):
    view_extent = calculate_view_extent_from_crop_files(view_crops, case_study_date)
    if view_extent is None:
        print("Could not determine selected-view extent; report markers skipped.")
        return pd.DataFrame()

    reports = pd.read_csv(reports_csv, low_memory=False)
    required_columns = {"TIME_EVENT", "LATITUDE", "LONGITUDE", "TYPE_EVENT"}
    missing_columns = required_columns.difference(reports.columns)
    if missing_columns:
        raise KeyError(
            f"Missing columns in ESSL report table: {sorted(missing_columns)}. "
            f"Available columns: {list(reports.columns)}"
        )

    reports = reports.copy()
    reports["time_event_naive"] = (
        pd.to_datetime(reports["TIME_EVENT"], utc=True, errors="coerce")
        .dt.tz_convert(None)
    )
    reports["LATITUDE"] = pd.to_numeric(reports["LATITUDE"], errors="coerce")
    reports["LONGITUDE"] = pd.to_numeric(reports["LONGITUDE"], errors="coerce")
    reports = reports.dropna(subset=["time_event_naive", "LATITUDE", "LONGITUDE"])
    reports["report_category"] = reports["TYPE_EVENT"].apply(normalize_report_category)
    reports = reports.dropna(subset=["report_category"])

    lon_min, lon_max, lat_min, lat_max = view_extent
    start_time = view_crops["time_start"].min()
    end_time = view_crops["time_end"].max()
    case_reports = reports[
        (reports["time_event_naive"].dt.strftime("%Y%m%d") == case_study_date)
        & (reports["time_event_naive"] >= start_time)
        & (reports["time_event_naive"] <= end_time)
        & (reports["LONGITUDE"] >= lon_min)
        & (reports["LONGITUDE"] <= lon_max)
        & (reports["LATITUDE"] >= lat_min)
        & (reports["LATITUDE"] <= lat_max)
    ].copy()

    if "ID" in case_reports.columns:
        case_reports = case_reports.drop_duplicates("ID")
    else:
        case_reports = case_reports.drop_duplicates(
            ["TIME_EVENT", "LATITUDE", "LONGITUDE", "TYPE_EVENT"]
        )

    print(
        f"Found {len(case_reports)} reports on {case_study_date} "
        "inside the selected-view area."
    )
    if not case_reports.empty:
        print(
            "Report categories in selected-view area: "
            f"{case_reports['report_category'].value_counts().to_dict()}"
        )
    return case_reports.sort_values("time_event_naive")


def add_report_time_markers(ax, reports):
    if reports.empty:
        return

    y_min, y_max = ax.get_ylim()
    y_position = y_min + 0.06 * (y_max - y_min)
    used_labels = set()

    for report_type, spec in REPORT_MARKER_SPECS.items():
        report_times = reports.loc[
            reports["report_category"] == report_type,
            "time_event_naive",
        ]
        if report_times.empty:
            continue

        label = spec["label"]
        ax.scatter(
            report_times,
            np.full(len(report_times), y_position),
            marker=spec["marker"],
            color=spec["color"],
            edgecolor="black",
            linewidth=0.4,
            s=55,
            zorder=6,
            label=label if label not in used_labels else None,
        )
        used_labels.add(label)


def plot_class_interval_strip(ax, view_crops):
    strip_df = view_crops.copy()
    strip_df["label"] = pd.to_numeric(strip_df["label"], errors="coerce")
    strip_df["time_start"] = pd.to_datetime(strip_df["time_start"], errors="coerce")
    strip_df["time_end"] = pd.to_datetime(strip_df["time_end"], errors="coerce")
    strip_df = strip_df.dropna(subset=["label", "time_start", "time_end"])
    strip_df = strip_df[strip_df["label"] != -100].copy()

    for _, row in strip_df.iterrows():
        label = int(row["label"])
        x_start = mdates.date2num(row["time_start"])
        x_end = mdates.date2num(row["time_end"])
        color = colors_per_class1_names.get(str(label), "lightgray")
        rect = Rectangle(
            (x_start, -0.4),
            x_end - x_start,
            0.8,
            facecolor=color,
            edgecolor="white",
            linewidth=1.4,
        )
        ax.add_patch(rect)
        ax.text(
            x_start + (x_end - x_start) / 2,
            0,
            str(label),
            ha="center",
            va="center",
            fontsize=CLASS_LABEL_FONTSIZE,
            color="black",
        )

    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([0])
    ax.set_yticklabels(["class"])
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(axis="x", color="0.85", linestyle="--", linewidth=0.8)


def main():

    plotting = True  # Set to True to enable plotting
    selected_view = "view002"  # Specify the view for analysis
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # case study date
    case_study_date = "20250616"

    # read the rows of the test crops for the day from the test video summary csv file
    test_video_summary = pd.read_csv(CSV_FILES["testing"])

    # filter all rows where crop contains the date formatted as yyyymmdd in the first 8 characters of the crop name
    test_crops_day = test_video_summary[
        test_video_summary["crop"].str.startswith(case_study_date)
    ].copy()

    # select only the specified view for the case study analysis
    test_crops_day = test_crops_day[test_crops_day["crop"].str.contains(selected_view)].copy()
    print(f"Number of crops for the day {case_study_date} and {selected_view}: {len(test_crops_day)}")

    test_crops_day["time_start"] = pd.to_datetime(
        test_crops_day["time_start"], errors="coerce"
    )
    test_crops_day["time_end"] = pd.to_datetime(
        test_crops_day["time_end"], errors="coerce"
    )
    test_crops_day = test_crops_day.dropna(subset=["time_start", "time_end"])
    test_crops_day = test_crops_day.sort_values("time_start").reset_index(drop=True)
    reports_in_selected_view = load_case_reports_in_view(
        CSV_FILES["essl_reports"],
        case_study_date,
        test_crops_day,
    )

    
    # Load training dataframe and compute mean values for each class of all variables, to be used for calculating the precursor anomalies metrics
    training_df = pd.read_csv(CSV_FILES["training"])
    training_df["label"] = pd.to_numeric(training_df["label"], errors="coerce")
    training_mean_df = training_df.groupby("label").mean(numeric_only=True).reset_index()

    # define variables for precursor anomalies metrics
    precursor_variables = [
        "cot_mean",
        "cot30plus_mean",
        "cth_gradient",
        "euclid_msg_grid_mean",
        "precipitation_mean",
    ]
    anomaly_plot_labels = [
        "mean COT anomaly",
        "mean COT30+ anomaly",
        "grad(CTH) anomaly",
        "mean lightning count anomaly",
        "mean prec anomaly",
    ]

    n_precursor_variables = len(precursor_variables)

    # create a time serie of mean values taken from trainin_mean_df based on the class label of the test crops for the day

    # read labels from test_crops_day and for each label get the mean values from training_mean_df and create a new array with the mean values for each variable
    mean_val_matrix = np.zeros((len(test_crops_day), n_precursor_variables))
    # loop on the precursor variables
    for var_index, variable in enumerate(precursor_variables):

        # check if the variable is present in the training_mean_df columns
        if variable not in training_mean_df.columns:
            print(f"{variable} not found in training mean dataframe; filling with NaN.")
            mean_val_matrix[:, var_index] = np.nan
            continue

        # loop on the labels of the test crops for the day and get the mean values from training_mean_df
        for i in range(len(test_crops_day)):

            # read the label for the current video crop
            label = test_crops_day.iloc[i]["label"]

            # get the values for the precursor variables from training_mean_df for the given label
            mean_vals = training_mean_df[training_mean_df["label"] == label][variable].values.flatten()

            # calculate the mean of the values and store it in the mean_val_matrix
            mean_val_matrix[i, var_index] = np.nanmean(mean_vals)

    # calculate anomaly by subtracting the mean values from the test crops for the day
    anomaly_matrix = np.zeros((len(test_crops_day), n_precursor_variables))

    # loop on the precursor variables and calculate the anomaly for each variable
    for var_index, variable in enumerate(precursor_variables):

        # check if the variable is present in the test_crops_day columns
        if variable not in test_crops_day.columns:
            print(f"{variable} not found in test crops dataframe; filling with NaN.")
            anomaly_matrix[:, var_index] = np.nan
            continue

        # calculate the anomaly by subtracting the mean values from the test crops for the day
        anomaly_matrix[:, var_index] = (
            pd.to_numeric(test_crops_day[variable], errors="coerce").to_numpy(dtype=float)
            - mean_val_matrix[:, var_index]
        )

        print(anomaly_matrix[:, var_index])


    # extract the view from the crop name (it is given by view00x) and add it as a new column to the dataframe
    test_crops_day["view"] = test_crops_day["crop"].str.extract(r"(view\d{3})")

    test_crops_day["time_start"] = pd.to_datetime(
        test_crops_day["time_start"], errors="coerce"
    )
    test_crops_day["time_end"] = pd.to_datetime(
        test_crops_day["time_end"], errors="coerce"
    )

    # drop rows with NaN values in the columns view, time_start, or time_end
    test_crops_day = test_crops_day.dropna(subset=["view", "time_start", "time_end"])

    print(f"Number of crops for the day {case_study_date}: {len(test_crops_day)}")

    test_crops_day_sorted = test_crops_day.sort_values(by=["view", "time_start"])
    for view in sorted(test_crops_day_sorted["view"].unique()):
        view_crops = test_crops_day_sorted[test_crops_day_sorted["view"] == view]
        print(
            f"Number of crops for the day {case_study_date} and view {view}: "
            f"{len(view_crops)}"
        )

    if plotting: 
        plot_views_class_lookup_table(
            test_crops_day_sorted,
            case_study_date,
            OUTPUT_DIR / f"crops_{case_study_date}_views.png",
        )

        plot_case_day_videos_by_view(test_crops_day_sorted, case_study_date)

    # calculate now finite differences of the anomaly time series for each precursor variable and store them in a new matrix
    anomaly_diff_matrix = np.zeros((len(test_crops_day) - 1, n_precursor_variables))
    for var_index in range(n_precursor_variables):
        anomaly_diff_matrix[:, var_index] = np.diff(anomaly_matrix[:, var_index])   

    # load the warning thresholds for each precursor variable from the config file
    warning_thresholds = config["output_files"]["warning_south_day"]
    df_warning_thresholds = pd.read_csv(warning_thresholds)
    df_warning_thresholds["class_label"] = pd.to_numeric(
        df_warning_thresholds["class_label"], errors="coerce"
    )
    df_warning_thresholds = df_warning_thresholds.dropna(subset=["class_label"])
    df_warning_thresholds["class_label"] = df_warning_thresholds["class_label"].astype(int)

    # select rows in the  column from_temporal_sequence_name equal to > 4 h before event , 4h < t < 2h before event, < 2h before event
    ds_warning_before = df_warning_thresholds[df_warning_thresholds["from_temporal_sequence_name"].isin(
        ["> 4 h before event", "4h < t < 2h before event", "< 2h before event"]
    )].copy()   

    # compare now for each time stamp, as follows:
    # loop on time, if the class is a class for extreme, compare the finite difference of the anomaly time 
    # series for each precursor variable with the warning threshold for that variable in any of the time stamps before the event.
    # If the finite difference is above the warning threshold, draw a red x over the time stamp value in the plot  print a warning message with the time stamp, the class, and the precursor variable that triggered the warning.
    warning_array = np.zeros((len(test_crops_day) - 1, n_precursor_variables), dtype=bool)
    for time_index in range(len(test_crops_day) - 1):

        # read the class label for the next time stamp (the one after the current time index)
        current_class = test_crops_day.iloc[time_index + 1]["label"]

        # read the thresholds for the current class from the ds_warning_before
        thresholds_for_class = ds_warning_before[
            ds_warning_before["class_label"] == current_class
        ]

        # read the warning thresholds of the class for the 
        if current_class in df_warning_thresholds["class_label"].values:

            # loop on the precursor variables and compare the finite difference of the anomaly time series with any of the thresholds_for_class for that variable
            for var_index, variable in enumerate(precursor_variables):
                predictor = f"{variable}_anomaly"
                thresholds_for_predictor = thresholds_for_class[
                    thresholds_for_class["predictor"] == predictor
                ]["warning_threshold_finite_difference"]

                if thresholds_for_predictor.empty:
                    continue

                if any(
                    anomaly_diff_matrix[time_index, var_index] > threshold
                    for threshold in thresholds_for_predictor.values
                ):
                    warning_array[time_index, var_index] = True
                    print(
                        f"Warning: At time {test_crops_day.iloc[time_index + 1]['time_start']}, "
                        f"class {current_class}, precursor variable {variable} "
                        f"exceeded threshold with finite difference {anomaly_diff_matrix[time_index, var_index]:.3f} "
                        f"(threshold: {', '.join(map(str, thresholds_for_predictor.values))})"
                    )

    



    # plot the anomalies time series for the selected view, with reports and with crosses over the anomalies where the finite difference exceeded the warning threshold for that precursor variable
    fig, axes = plt.subplots(
        nrows=len(precursor_variables) + 1,
        ncols=1,
        figsize=(15, 2.5 * len(precursor_variables) + 1.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1] * len(precursor_variables) + [0.7]},
    )
    anomaly_axes = axes[:-1]
    class_axis = axes[-1]
    fig.suptitle(
        f"{selected_view} precursor anomalies and class sequence",
        fontsize=TITLE_FONTSIZE,
    )
  
    # for each suplot, plot a precursor variable as a function of time with resolution 2h. 
    for var_index, variable in enumerate(precursor_variables):
        anomaly_axes[var_index].plot(
            pd.to_datetime(test_crops_day["time_start"], errors="coerce"),
            anomaly_matrix[:, var_index],
            marker="o",
            linestyle="-",
            color="black",
            label=anomaly_plot_labels[var_index],
        )
        anomaly_axes[var_index].axhline(
            y=0, color="black", linestyle="--", linewidth=1, label="Training mean"
        )
        y_min, y_max = anomaly_axes[var_index].get_ylim()
        y_padding = 0.18 * (y_max - y_min if y_max > y_min else 1)
        anomaly_axes[var_index].set_ylim(y_min, y_max + y_padding)
        y_range = y_max - y_min if y_max > y_min else 1

        # plot crosses above the anomalies where finite differences exceeded thresholds
        warning_mask = warning_array[:, var_index]
        warning_times = pd.to_datetime(
            test_crops_day["time_start"], errors="coerce"
        ).iloc[1:][warning_mask]
        warning_values = np.full(len(warning_times), y_max + 0.08 * y_range)
        anomaly_axes[var_index].scatter(
            warning_times,
            warning_values,
            color="red",
            marker="x",
            s=120,
            linewidth=2.0,
            label="Warning",
            zorder=7,
        )
        add_report_time_markers(anomaly_axes[var_index], reports_in_selected_view)
        anomaly_axes[var_index].set_ylabel(
            anomaly_plot_labels[var_index],
            fontsize=AXIS_LABEL_FONTSIZE,
            rotation=35,
            ha="right",
            va="center",
        )
        anomaly_axes[var_index].grid(True, linestyle="--", alpha=0.5)

    plot_class_interval_strip(class_axis, test_crops_day)

    legend_handles = []
    legend_labels = []
    for ax in anomaly_axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label not in legend_labels:
                legend_handles.append(handle)
                legend_labels.append(label)
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fontsize=LEGEND_FONTSIZE,
        )
    
    # set x-axis label and format the x-axis to show time in HH:MM format
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    for ax in axes:
        style_axis(ax)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    axes[-1].set_xlabel("Time", fontsize=AXIS_LABEL_FONTSIZE)
    plt.tight_layout(rect=[0, 0, 0.86, 1])
    # store the figure in the output directory
    fig.savefig(
        OUTPUT_DIR / f"precursor_anomaly_time_series_{case_study_date}_{selected_view}.png",
        dpi=300,
        bbox_inches="tight",
    )   

    
if __name__ == "__main__":
    main()
