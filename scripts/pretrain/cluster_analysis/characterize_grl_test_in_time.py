"""
Characterize GRL test video crops relative to ESSL report timing.

For each date and eulerian view, the script finds ESSL reports that fall
inside the geographic bounds of that view. It then labels all video crops in
that date/view by their timing relative to the first and last report:

-4 : earlier than 4 h before the first report
-2 : between 4 h and 2 h before the first report
-1 : within 2 h before the first report
 0 : between the first and last report
 1 : within 2 h after the last report
 2 : between 2 h and 4 h after the last report
 4 : later than 4 h after the last report

Views without reports keep the default label -100.
"""

from pathlib import Path
import sys

import pandas as pd

sys.path.append("/home/claudia/codes/ML_postprocessing")

from utils.configs import load_config

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


config_path = "/home/claudia/codes/ML_postprocessing/configs/process_run_GRL.yaml"
config = load_config(config_path)

CSV_FILES = {
    "testing": config["output_files"]["testing_video_summary"],
    "essl_reports": config["output_files"]["ancillary_files"]["essl_dataset"],
}

OUTPUT_DIR = Path(CSV_FILES["testing"]).parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def add_view_column(video_stats_df):
    """Extract integer eulerian view id from crop filenames like view001."""
    required_columns = {"crop", "time_start", "lat_mid", "lon_mid"}
    missing_columns = required_columns.difference(video_stats_df.columns)
    if missing_columns:
        raise KeyError(
            f"Missing columns in test video summary: {sorted(missing_columns)}. "
            f"Available columns: {list(video_stats_df.columns)}"
        )

    video_stats_df = video_stats_df.copy()
    video_stats_df["view"] = (
        video_stats_df["crop"].astype(str).str.extract(r"view(\d{3})", expand=False)
    )
    if video_stats_df["view"].isna().any():
        missing_views = video_stats_df.loc[video_stats_df["view"].isna(), "crop"].unique()
        raise ValueError(f"Could not extract view from crop names: {missing_views[:5]}")

    video_stats_df["view"] = video_stats_df["view"].astype(int)
    return video_stats_df


def prepare_essl_reports(essl_reports_df):
    """Normalize ESSL columns to report_time, date, lat, and lon."""
    required_columns = {"TIME_EVENT", "LATITUDE", "LONGITUDE"}
    missing_columns = required_columns.difference(essl_reports_df.columns)
    if missing_columns:
        raise KeyError(
            f"Missing columns in ESSL reports: {sorted(missing_columns)}. "
            f"Available columns: {list(essl_reports_df.columns)}"
        )

    essl_reports_df = essl_reports_df.copy()
    essl_reports_df["report_time"] = (
        pd.to_datetime(essl_reports_df["TIME_EVENT"], utc=True, errors="coerce")
        .dt.tz_convert(None)
    )
    essl_reports_df["lat"] = pd.to_numeric(essl_reports_df["LATITUDE"], errors="coerce")
    essl_reports_df["lon"] = pd.to_numeric(essl_reports_df["LONGITUDE"], errors="coerce")
    essl_reports_df = essl_reports_df.dropna(subset=["report_time", "lat", "lon"])
    essl_reports_df["date"] = essl_reports_df["report_time"].dt.date
    return essl_reports_df


def bounds_from_centers(center_values):
    """Return lower/upper geographic bounds around sorted view centers."""
    centers = sorted(pd.Series(center_values).dropna().unique())
    if not centers:
        raise ValueError("No valid view centers found.")
    if len(centers) == 1:
        return {centers[0]: (centers[0], centers[0])}

    midpoints = [
        (centers[index] + centers[index + 1]) / 2
        for index in range(len(centers) - 1)
    ]
    first_half_width = midpoints[0] - centers[0]
    last_half_width = centers[-1] - midpoints[-1]

    bounds = {}
    for index, center in enumerate(centers):
        lower = center - first_half_width if index == 0 else midpoints[index - 1]
        upper = center + last_half_width if index == len(centers) - 1 else midpoints[index]
        bounds[center] = (min(lower, upper), max(lower, upper))

    return bounds


def build_view_specs_from_video_summary(video_stats_df):
    """Derive one geographic bounding box per view from lat_mid/lon_mid centers."""
    view_centers = (
        video_stats_df.groupby("view", as_index=False)
        .agg(lat_mid=("lat_mid", "first"), lon_mid=("lon_mid", "first"))
    )

    lat_bounds = bounds_from_centers(view_centers["lat_mid"])
    lon_bounds = bounds_from_centers(view_centers["lon_mid"])

    crop_specs = []
    for _, row in view_centers.iterrows():
        lat_min, lat_max = lat_bounds[row["lat_mid"]]
        lon_min, lon_max = lon_bounds[row["lon_mid"]]
        crop_specs.append(
            {
                "view_id": int(row["view"]),
                "lat_min": lat_min,
                "lat_max": lat_max,
                "lon_min": lon_min,
                "lon_max": lon_max,
            }
        )

    return crop_specs


def assign_temporal_sequence_label(crop_time, initial_report_time, final_report_time):
    if crop_time < initial_report_time - pd.Timedelta(hours=4):
        return -4
    if initial_report_time - pd.Timedelta(hours=4) <= crop_time < initial_report_time - pd.Timedelta(hours=2):
        return -2
    if initial_report_time - pd.Timedelta(hours=2) <= crop_time < initial_report_time:
        return -1
    if initial_report_time <= crop_time <= final_report_time:
        return 0
    if final_report_time < crop_time <= final_report_time + pd.Timedelta(hours=2):
        return 1
    if final_report_time + pd.Timedelta(hours=2) < crop_time <= final_report_time + pd.Timedelta(hours=4):
        return 2
    return 4


def iter_dates_with_progress(first_date, last_date):
    dates = list(pd.date_range(first_date, last_date))
    if tqdm is not None:
        return tqdm(dates, desc="Processing days", unit="day")
    return dates


def main():
    video_stats_df = pd.read_csv(CSV_FILES["testing"])
    print("Test video summary CSV file loaded successfully.")

    essl_reports_df = pd.read_csv(CSV_FILES["essl_reports"])
    print("ESSL reports CSV file loaded successfully.")

    video_stats_df = add_view_column(video_stats_df)
    essl_reports_df = prepare_essl_reports(essl_reports_df)

    video_stats_df["temporal_sequence_label"] = -100
    video_stats_df["time_start"] = pd.to_datetime(video_stats_df["time_start"])
    video_stats_df["date"] = video_stats_df["time_start"].dt.date

    grouped_video_stats = video_stats_df.groupby(["date", "view"])
    first_date = video_stats_df["date"].min()
    last_date = video_stats_df["date"].max()
    crop_specs = build_view_specs_from_video_summary(video_stats_df)

    dates = iter_dates_with_progress(first_date, last_date)
    total_days = len(dates) if tqdm is None else dates.total

    for day_index, date in enumerate(dates, start=1):
        if tqdm is None:
            print(f"Processing day {day_index}/{total_days}: {date.date()}")

        reports_on_date = essl_reports_df[essl_reports_df["date"] == date.date()]
        if reports_on_date.empty:
            continue

        for crop_spec in crop_specs:
            view_id = crop_spec["view_id"]
            reports_in_view = reports_on_date[
                (reports_on_date["lat"] >= crop_spec["lat_min"])
                & (reports_on_date["lat"] <= crop_spec["lat_max"])
                & (reports_on_date["lon"] >= crop_spec["lon_min"])
                & (reports_on_date["lon"] <= crop_spec["lon_max"])
            ]
            if reports_in_view.empty:
                continue

            group_key = (date.date(), view_id)
            if group_key not in grouped_video_stats.groups:
                continue

            initial_report_time = reports_in_view["report_time"].min()
            final_report_time = reports_in_view["report_time"].max()
            video_crops_in_view = grouped_video_stats.get_group(group_key)

            for index, row in video_crops_in_view.iterrows():
                video_stats_df.at[index, "temporal_sequence_label"] = (
                    assign_temporal_sequence_label(
                        row["time_start"],
                        initial_report_time,
                        final_report_time,
                    )
                )

    output_csv_path = OUTPUT_DIR / "crops_video_summary_with_temporal_sequence_labels.csv"
    video_stats_df.to_csv(output_csv_path, index=False)
    print(f"Updated test video summary CSV file saved to {output_csv_path}")


if __name__ == "__main__":
    main()
