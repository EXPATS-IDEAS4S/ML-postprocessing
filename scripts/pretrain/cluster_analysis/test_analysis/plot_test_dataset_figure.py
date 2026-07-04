"""
Plot GRL test-dataset overview and representative case maps.

The script first summarizes all ESSL cases used in the test dataset with a map,
case counts per year, and hail-vs-precipitation report counts. It then selects
the ESSL test cases with the highest number of reports. For each case, it finds
the eulerian test crop view that contains the largest number of reports, opens
one matching crop NetCDF file, and plots frame maps in the same style used by
ESSL_analysis/code/crop_generator/generate_test_crops_grl.py:

- IR_108 data from the selected view is drawn on the lon/lat map.
- all eulerian crop views are outlined and numbered.
- the selected/current view is highlighted in red.
- ESSL reports for the selected case are overlaid by report type.

How to run:
    python scripts/pretrain/cluster_analysis/test_analysis/plot_test_dataset_figure.py
"""

from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
import xarray as xr

sys.path.append("/home/claudia/codes/ML_postprocessing")

from utils.configs import load_config


CONFIG_PATH = "/home/claudia/codes/ML_postprocessing/configs/process_run_GRL.yaml"
DEM_PATH = "/data1/DEM_EXPATS_0.01x0.01.nc"
EXPATS_DOMAIN = (5.0, 16.0, 42.0, 51.5)
BT108_DISPLAY_MIN = 240.0
BT108_DISPLAY_MAX = 320.0
VIDEO_FPS = 2
N_TOP_CASES = 10
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


def read_config():
    config = load_config(CONFIG_PATH)
    output_files = config["output_files"]
    features_config = config.get("features_preparation", {})
    cloud_motion_config = config.get("cloud_motion", {})

    csv_files = {
        "testing": output_files["testing_video_summary"],
        "essl_reports": output_files["ancillary_files"]["essl_dataset"],
        "essl_cases": str(
            Path(output_files["ancillary_files"]["essl_dataset"]).with_name(
                "essl_cases_2025_grl.csv"
            )
        ),
    }
    crop_roots = [
        cloud_motion_config.get("path_root_test"),
        features_config.get("crops_test_path"),
        features_config.get("images_test_path"),
        "/sat_data/crops/test_grl_2026/1",
        "/sat_data/crops/GRL_testing_crops/run2",
        "/sat_data/crops/GRL_testing_crops",
    ]
    crop_roots = [Path(path) for path in crop_roots if path]

    output_dir = Path(output_files["figures_dir"]) / "test_dataset_case_map"
    output_dir.mkdir(parents=True, exist_ok=True)
    return csv_files, crop_roots, output_dir


def load_test_summary(csv_file):
    df = pd.read_csv(csv_file)
    required_columns = {"crop", "time_start", "lat_mid", "lon_mid"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise KeyError(
            f"Missing columns in test video summary: {sorted(missing_columns)}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.copy()
    df["time_start"] = pd.to_datetime(df["time_start"], errors="coerce")
    if "time_end" in df.columns:
        df["time_end"] = pd.to_datetime(df["time_end"], errors="coerce")
    df["lat_mid"] = pd.to_numeric(df["lat_mid"], errors="coerce")
    df["lon_mid"] = pd.to_numeric(df["lon_mid"], errors="coerce")
    df["view"] = pd.to_numeric(
        df["crop"].astype(str).str.extract(r"view(\d{3})", expand=False),
        errors="coerce",
    )
    return df.dropna(subset=["time_start", "lat_mid", "lon_mid", "view"])


def load_test_cases(csv_file, test_dates):
    cases = pd.read_csv(csv_file)
    required_columns = {
        "date",
        "num_reports",
        "num_precip",
        "num_hail",
        "start_time",
        "end_time",
        "case_type",
        "start_lat",
        "start_lon",
        "end_lat",
        "end_lon",
    }
    missing_columns = required_columns.difference(cases.columns)
    if missing_columns:
        raise KeyError(
            f"Missing columns in ESSL case list: {sorted(missing_columns)}. "
            f"Available columns: {list(cases.columns)}"
        )

    cases = cases.copy()
    for column in [
        "num_reports",
        "num_precip",
        "num_hail",
        "start_lat",
        "start_lon",
        "end_lat",
        "end_lon",
    ]:
        cases[column] = pd.to_numeric(cases[column], errors="coerce")

    cases = cases.dropna(subset=["num_reports"])
    if cases.empty:
        raise ValueError("No valid num_reports values found in the ESSL case list.")

    case_dates = pd.to_datetime(cases["date"], errors="coerce").dt.date
    cases = cases[case_dates.isin(test_dates)].copy()
    if cases.empty:
        raise ValueError(
            "No ESSL cases from the case list match dates present in the test "
            "video summary."
        )

    cases["case_date"] = pd.to_datetime(cases["date"]).dt.normalize()
    cases["start_time"] = (
        pd.to_datetime(cases["start_time"], utc=True).dt.tz_convert(None)
    )
    cases["end_time"] = (
        pd.to_datetime(cases["end_time"], utc=True).dt.tz_convert(None)
    )
    return cases.sort_values("case_date").reset_index(drop=True)


def load_top_report_cases(csv_file, test_dates, n_cases=N_TOP_CASES):
    cases = load_test_cases(csv_file, test_dates)
    return cases.sort_values("num_reports", ascending=False).head(n_cases).copy()


def load_reports_for_case(csv_file, case_row):
    reports = pd.read_csv(csv_file)
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

    return reports[
        (reports["time_event_naive"] >= case_row["start_time"])
        & (reports["time_event_naive"] <= case_row["end_time"])
    ].copy()


def load_reports_for_cases(csv_file, cases):
    reports = pd.read_csv(csv_file)
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

    lon_min, lon_max, lat_min, lat_max = EXPATS_DOMAIN
    reports = reports[
        (reports["LONGITUDE"] >= lon_min)
        & (reports["LONGITUDE"] <= lon_max)
        & (reports["LATITUDE"] >= lat_min)
        & (reports["LATITUDE"] <= lat_max)
    ]

    selected_reports = []
    for _, case in cases.iterrows():
        reports_for_case = reports[
            (reports["time_event_naive"] >= case["start_time"])
            & (reports["time_event_naive"] <= case["end_time"])
        ].copy()
        reports_for_case["case_date"] = case["case_date"]
        reports_for_case["case_type"] = case["case_type"]
        selected_reports.append(reports_for_case)

    if not selected_reports:
        return reports.iloc[0:0].copy()

    selected_reports_df = pd.concat(selected_reports, ignore_index=True)
    if "ID" in selected_reports_df.columns:
        selected_reports_df = selected_reports_df.drop_duplicates("ID")
    else:
        selected_reports_df = selected_reports_df.drop_duplicates(
            ["TIME_EVENT", "LATITUDE", "LONGITUDE", "TYPE_EVENT"]
        )

    return selected_reports_df


def load_reports_for_video_summary_days(csv_file, video_summary_dates):
    reports = pd.read_csv(csv_file)
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
    reports["date"] = reports["time_event_naive"].dt.date

    lon_min, lon_max, lat_min, lat_max = EXPATS_DOMAIN
    reports = reports[
        reports["date"].isin(video_summary_dates)
        & (reports["LONGITUDE"] >= lon_min)
        & (reports["LONGITUDE"] <= lon_max)
        & (reports["LATITUDE"] >= lat_min)
        & (reports["LATITUDE"] <= lat_max)
    ].copy()

    if "ID" in reports.columns:
        reports = reports.drop_duplicates("ID")
    else:
        reports = reports.drop_duplicates(
            ["TIME_EVENT", "LATITUDE", "LONGITUDE", "TYPE_EVENT"]
        )
    return reports


def style_overview_axis(ax):
    ax.grid(color="0.88", linestyle="--", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_video_summary_reports_map(reports, output_dir, orography=None):
    lon_min, lon_max, lat_min, lat_max = EXPATS_DOMAIN
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("ESSL reports on video-summary days")
    style_overview_axis(ax)
    plot_orography_contours(ax, orography, EXPATS_DOMAIN)

    if not reports.empty:
        hail = reports[reports["TYPE_EVENT"].astype(str).str.upper() == "HAIL"]
        precip = reports[reports["TYPE_EVENT"].astype(str).str.upper() == "PRECIP"]
        other = reports[~reports.index.isin(hail.index.union(precip.index))]
        if not precip.empty:
            ax.scatter(
                precip["LONGITUDE"],
                precip["LATITUDE"],
                s=11,
                color="#2b7cd2",
                marker="o",
                alpha=0.75,
                edgecolors="black",
                linewidths=0.3,
                label="PRECIP",
                zorder=4,
            )
        if not hail.empty:
            ax.scatter(
                hail["LONGITUDE"],
                hail["LATITUDE"],
                s=13,
                color="#e1761c",
                marker="^",
                alpha=0.8,
                edgecolors="black",
                linewidths=0.3,
                label="HAIL",
                zorder=4,
            )
        if not other.empty:
            ax.scatter(
                other["LONGITUDE"],
                other["LATITUDE"],
                s=9,
                color="0.4",
                marker="s",
                alpha=0.65,
                edgecolors="black",
                linewidths=0.3,
                label="Other report",
                zorder=4,
            )

    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()

    output_png = output_dir / "video_summary_days_reports_map.png"
    output_pdf = output_dir / "video_summary_days_reports_map.pdf"
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved video-summary report map to {output_png}")
    print(f"Saved video-summary report map to {output_pdf}")


def plot_video_summary_days_per_year(ax, video_summary_dates):
    dates = pd.to_datetime(pd.Series(sorted(video_summary_dates)))
    year_counts = dates.dt.year.value_counts().sort_index()
    ax.bar(year_counts.index.astype(str), year_counts.values, color="0.35")
    ax.set_title("Video-summary days per year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of days")
    ax.tick_params(axis="x", rotation=45)
    style_overview_axis(ax)


def plot_hail_precip_distribution(ax, reports):
    if reports.empty:
        ax.text(0.5, 0.5, "No reports found", ha="center", va="center")
        ax.set_axis_off()
        return

    report_types = reports["TYPE_EVENT"].astype(str).str.upper()
    report_counts = pd.Series({
        "Hail": int((report_types == "HAIL").sum()),
        "Precipitation": int((report_types == "PRECIP").sum()),
    })
    colors = ["#e1761c", "#2b7cd2"]
    ax.bar(report_counts.index, report_counts.values, color=colors)
    ax.set_title("Hail vs precipitation reports on video-summary days")
    ax.set_ylabel("Number of reports")
    style_overview_axis(ax)


def plot_video_summary_histograms(video_summary_dates, reports, output_dir):
    fig, (ax_year, ax_reports) = plt.subplots(2, 1, figsize=(7.5, 8.5))

    plot_video_summary_days_per_year(ax_year, video_summary_dates)
    plot_hail_precip_distribution(ax_reports, reports)

    fig.suptitle("Video-summary day and report distributions", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    output_png = output_dir / "video_summary_days_histograms.png"
    output_pdf = output_dir / "video_summary_days_histograms.pdf"
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved video-summary histograms to {output_png}")
    print(f"Saved video-summary histograms to {output_pdf}")


def get_frame_count(ds):
    if "time" in ds.dims:
        return int(ds.sizes["time"])
    if "time" in ds.coords:
        return int(ds["time"].size)
    return 1


def get_frame_time(ds, frame_index):
    if "time" not in ds.coords:
        return None
    return pd.to_datetime(ds["time"].values[frame_index])


def get_frame_lat_lon(ds, frame_index):
    lat = ds["lat"]
    lon = ds["lon"]
    if "time" in lat.dims:
        lat = lat.isel(time=frame_index)
    if "time" in lon.dims:
        lon = lon.isel(time=frame_index)
    return lat, lon


def get_crop_extent_from_file(crop_path):
    with xr.open_dataset(crop_path) as ds:
        lat, lon = get_frame_lat_lon(ds, 0)
        lat_values = np.asarray(lat.values, dtype=float)
        lon_values = np.asarray(lon.values, dtype=float)

    return {
        "center_lat": float(np.nanmean(lat_values)),
        "center_lon": float(np.nanmean(lon_values)),
        "lat_min": float(np.nanmin(lat_values)),
        "lat_max": float(np.nanmax(lat_values)),
        "lon_min": float(np.nanmin(lon_values)),
        "lon_max": float(np.nanmax(lon_values)),
    }


def build_view_specs_from_crop_files(video_stats_df, crop_roots, case_row):
    """
    Build eulerian view rectangles from the actual saved crop coordinates.

    This mirrors the output of define_eulerian_crop_centers() in
    generate_test_crops_grl.py more faithfully than rebuilding boxes from
    center points: each rectangle is the lat/lon extent of one real 100 x 100
    crop for that view.
    """
    case_date = case_row["case_date"].date()
    view_rows = (
        video_stats_df[video_stats_df["time_start"].dt.date == case_date]
        .sort_values("time_start")
        .drop_duplicates("view")
    )
    if view_rows.empty:
        view_rows = video_stats_df.sort_values("time_start").drop_duplicates("view")

    crop_specs = []
    for _, row in view_rows.sort_values("view").iterrows():
        crop_path = resolve_crop_path(row["crop"], crop_roots)
        crop_extent = get_crop_extent_from_file(crop_path)
        crop_specs.append(
            {
                "view_id": int(row["view"]),
                "crop_path": crop_path,
                **crop_extent,
            }
        )

    if not crop_specs:
        raise ValueError("Could not build any eulerian view specs from crop files.")

    return crop_specs


def select_view_with_most_reports(crop_specs, reports_df):
    if reports_df.empty:
        return crop_specs[0], 0

    report_counts = []
    for spec in crop_specs:
        reports_in_view = reports_df[
            (reports_df["LATITUDE"] >= spec["lat_min"])
            & (reports_df["LATITUDE"] <= spec["lat_max"])
            & (reports_df["LONGITUDE"] >= spec["lon_min"])
            & (reports_df["LONGITUDE"] <= spec["lon_max"])
        ]
        report_counts.append((len(reports_in_view), spec))

    max_count, selected_spec = max(
        report_counts,
        key=lambda item: (item[0], -item[1]["view_id"]),
    )
    return selected_spec, max_count


def get_reports_in_view(reports_df, view_spec):
    if reports_df.empty:
        return reports_df

    return reports_df[
        (reports_df["LATITUDE"] >= view_spec["lat_min"])
        & (reports_df["LATITUDE"] <= view_spec["lat_max"])
        & (reports_df["LONGITUDE"] >= view_spec["lon_min"])
        & (reports_df["LONGITUDE"] <= view_spec["lon_max"])
    ]


def select_crop_for_case_and_view(test_df, case_row, view_spec, reports_df):
    view_id = view_spec["view_id"]
    case_crops = test_df[
        (test_df["time_start"] >= case_row["start_time"])
        & (test_df["time_start"] <= case_row["end_time"])
        & (test_df["view"] == view_id)
    ].copy()

    if case_crops.empty:
        same_day = test_df[
            (test_df["time_start"].dt.date == case_row["case_date"].date())
            & (test_df["view"] == view_id)
        ].copy()
        if same_day.empty:
            raise ValueError(
                f"No test crops found for {case_row['date']} view {view_id:03d}."
            )
        case_crops = same_day

    reports_in_view = get_reports_in_view(reports_df, view_spec)
    if reports_in_view.empty:
        target_time = case_row["start_time"]
    else:
        target_time = reports_in_view["time_event_naive"].min()

    if "time_end" in case_crops.columns and case_crops["time_end"].notna().any():
        crops_covering_target = case_crops[
            (case_crops["time_start"] <= target_time)
            & (case_crops["time_end"] >= target_time)
        ]
        if not crops_covering_target.empty:
            return crops_covering_target.sort_values("time_start").iloc[0]

    case_crops["distance_to_target_time"] = (
        case_crops["time_start"] - target_time
    ).abs()
    return case_crops.sort_values("distance_to_target_time").iloc[0]


def resolve_crop_path(crop_value, crop_roots):
    crop_path = Path(str(crop_value))
    if crop_path.is_absolute() and crop_path.exists():
        return crop_path

    filename = crop_path.name
    date_folder = filename[:8]
    date_folder = f"{date_folder[:4]}-{date_folder[4:6]}-{date_folder[6:8]}"
    candidates = []
    for root in crop_roots:
        candidates.extend(
            [
                root / filename,
                root / date_folder / filename,
                root / "run2" / date_folder / filename,
                root / "1" / filename,
                root / "1" / date_folder / filename,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    existing_roots = [root for root in crop_roots if root.exists()]
    for root in existing_roots:
        matches = list(root.rglob(filename))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"Could not find crop file {filename}. Checked roots: "
        f"{[str(root) for root in crop_roots]}"
    )


def get_frame_ir108(ds, frame_index):
    if "IR_108" in ds:
        ir = ds["IR_108"]
    elif "IR_108_masked" in ds:
        ir = ds["IR_108_masked"]
    else:
        raise KeyError(
            "Selected crop does not contain IR_108 or IR_108_masked. "
            f"Available variables: {list(ds.data_vars)}"
        )

    if "time" in ir.dims:
        ir = ir.isel(time=frame_index)
    return ir


def get_frame_cma(ds, frame_index):
    if "cma" not in ds:
        print("No cma variable found in selected crop; no-cloud hatching skipped.")
        return None

    cma = ds["cma"]
    if "time" in cma.dims:
        cma = cma.isel(time=frame_index)
    return cma


def find_orography_field(ds):
    lower_name_map = {name.lower(): name for name in ds.data_vars}
    for candidate in OROGRAPHY_VARIABLE_CANDIDATES:
        var_name = lower_name_map.get(candidate.lower())
        if var_name and {"lat", "lon"} <= set(ds[var_name].dims):
            da = ds[var_name]
            if "time" in da.dims:
                da = da.isel(time=0)
            return da

    for _, da in ds.data_vars.items():
        if {"lat", "lon"} <= set(da.dims):
            if "time" in da.dims:
                da = da.isel(time=0)
            return da
    return None


def load_orography_from_dem(path):
    if not Path(path).exists():
        print(f"DEM file not found at {path}; orography overlay will be skipped.")
        return None
    try:
        ds_dem = xr.open_dataset(path)
    except Exception as exc:
        print(f"Could not open DEM file {path}: {exc}; orography overlay will be skipped.")
        return None
    return find_orography_field(ds_dem)


def plot_orography_contours(ax, orography, domain):
    if orography is None or not {"lat", "lon"} <= set(orography.dims):
        return

    lon_min, lon_max, lat_min, lat_max = domain
    oro = orography.where(
        (orography.lon >= lon_min)
        & (orography.lon <= lon_max)
        & (orography.lat >= lat_min)
        & (orography.lat <= lat_max),
        drop=True,
    )
    if oro.size == 0:
        return

    data = oro.values.squeeze()
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
    )
    ax.clabel(contours, inline=True, fontsize=7, fmt="%d m")


def plot_reports_on_axis(ax, reports_df, domain):
    if reports_df is None or reports_df.empty:
        return

    lon_min, lon_max, lat_min, lat_max = domain
    reports = reports_df[
        (reports_df["LONGITUDE"] >= lon_min)
        & (reports_df["LONGITUDE"] <= lon_max)
        & (reports_df["LATITUDE"] >= lat_min)
        & (reports_df["LATITUDE"] <= lat_max)
    ]
    if reports.empty:
        return

    precip = reports[reports["TYPE_EVENT"].astype(str).str.upper() == "PRECIP"]
    hail = reports[reports["TYPE_EVENT"].astype(str).str.upper() == "HAIL"]
    other = reports[~reports.index.isin(precip.index.union(hail.index))]

    if not precip.empty:
        ax.scatter(
            precip["LONGITUDE"],
            precip["LATITUDE"],
            s=18,
            c="#2b7cd2",
            marker="o",
            edgecolors="black",
            linewidths=0.3,
            label="PRECIP",
            zorder=7,
        )
    if not hail.empty:
        ax.scatter(
            hail["LONGITUDE"],
            hail["LATITUDE"],
            s=28,
            c="#e1761c",
            marker="^",
            edgecolors="black",
            linewidths=0.3,
            label="HAIL",
            zorder=7,
        )
    if not other.empty:
        ax.scatter(
            other["LONGITUDE"],
            other["LATITUDE"],
            s=14,
            c="0.45",
            marker="s",
            edgecolors="black",
            linewidths=0.3,
            label="Other report",
            zorder=7,
        )


def plot_no_cloud_hatching(ax, ds_crop, lat_frame, lon_frame, frame_index):
    cma_frame = get_frame_cma(ds_crop, frame_index)
    if cma_frame is None:
        return False

    no_cloud = np.asarray(cma_frame.values.squeeze(), dtype=float) < 0.5
    if no_cloud.ndim != 2 or not np.any(no_cloud):
        return False

    with plt.rc_context({"hatch.color": "#ffd84d", "hatch.linewidth": 1.2}):
        hatch = ax.contourf(
            lon_frame.values,
            lat_frame.values,
            no_cloud.astype(int),
            levels=[0.5, 1.5],
            colors="none",
            hatches=["////"],
            zorder=2,
        )

    if hasattr(hatch, "collections"):
        for collection in hatch.collections:
            collection.set_edgecolor("#ffd84d")
            collection.set_linewidth(0.8)

    return True


def plot_selected_case_frame_map(
    ds_crop,
    crop_specs,
    current_spec,
    reports_df,
    case_row,
    crop_row,
    output_dir,
    frame_index,
    orography=None,
):
    lon_min, lon_max, lat_min, lat_max = EXPATS_DOMAIN
    frame_time = get_frame_time(ds_crop, frame_index)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, color="0.88", linewidth=0.8)

    ir_frame = get_frame_ir108(ds_crop, frame_index)
    lat_frame, lon_frame = get_frame_lat_lon(ds_crop, frame_index)
    ir_mesh = ax.pcolormesh(
        lon_frame.values,
        lat_frame.values,
        ir_frame.values.squeeze(),
        cmap="gray_r",
        vmin=BT108_DISPLAY_MIN,
        vmax=BT108_DISPLAY_MAX,
        shading="auto",
        alpha=0.85,
        zorder=1,
    )
    cbar = fig.colorbar(ir_mesh, ax=ax, pad=0.02, shrink=0.82)
    cbar.set_label("BT108 (K)")

    has_no_cloud_hatching = plot_no_cloud_hatching(
        ax,
        ds_crop,
        lat_frame,
        lon_frame,
        frame_index,
    )
    plot_orography_contours(ax, orography, EXPATS_DOMAIN)
    plot_reports_on_axis(ax, reports_df, EXPATS_DOMAIN)

    for spec in crop_specs:
        is_current = spec["view_id"] == current_spec["view_id"]
        edge_color = "#d62728" if is_current else "0.35"
        line_width = 2.4 if is_current else 1.0
        zorder = 5 if is_current else 3

        rect = Rectangle(
            (spec["lon_min"], spec["lat_min"]),
            spec["lon_max"] - spec["lon_min"],
            spec["lat_max"] - spec["lat_min"],
            edgecolor=edge_color,
            facecolor="none",
            linewidth=line_width,
            zorder=zorder,
        )
        ax.add_patch(rect)
        ax.text(
            spec["center_lon"],
            spec["center_lat"],
            str(spec["view_id"]),
            color=edge_color,
            ha="center",
            va="center",
            fontsize=8 if not is_current else 10,
            fontweight="bold" if is_current else "normal",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.65,
                "pad": 1.2,
            },
            zorder=zorder + 1,
        )

    ax.plot([], [], color="0.35", linewidth=1, label="Eulerian views")
    ax.plot(
        [],
        [],
        color="#d62728",
        linewidth=2.4,
        label=f"Current view {current_spec['view_id']}",
    )
    ax.plot([], [], color="0.35", linewidth=1, label="orography")
    handles, labels = ax.get_legend_handles_labels()
    if has_no_cloud_hatching:
        handles.append(
            Patch(
                facecolor="none",
                edgecolor="#ffd84d",
                hatch="////",
            )
        )
        labels.append("No cloud (CMA < 0.5)")
    ax.legend(handles, labels, loc="upper right", fontsize=8)

    title = (
        f"Highest-report ESSL test case: {case_row['date']} "
        f"({case_row['case_type']}, {int(case_row['num_reports'])} reports)\n"
        f"View {current_spec['view_id']:03d}, frame {frame_index:03d}"
    )
    if frame_time is not None:
        title = f"{title}, {frame_time}"
    ax.set_title(title)

    timestamp = (
        frame_time.strftime("%Y%m%d_%H%M")
        if frame_time is not None and not pd.isna(frame_time)
        else f"frame{frame_index:03d}"
    )
    output_path = (
        output_dir
        / (
            f"top_report_case_{case_row['date']}_"
            f"view{current_spec['view_id']:03d}_"
            f"frame{frame_index:03d}_{timestamp}_map.png"
        )
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved selected case frame map to {output_path}")
    return output_path


def quote_concat_path(path):
    return str(path).replace("'", "'\\''")


def save_frame_maps_video(output_paths, output_video_path, fps=VIDEO_FPS):
    if not output_paths:
        print("No frame maps were created; MP4 generation skipped.")
        return None

    concat_file = output_video_path.with_suffix(".ffmpeg_concat.txt")
    frame_duration = 1.0 / fps
    with concat_file.open("w", encoding="utf-8") as file_obj:
        for output_path in output_paths:
            file_obj.write(f"file '{quote_concat_path(output_path)}'\n")
            file_obj.write(f"duration {frame_duration:.6f}\n")
        file_obj.write(f"file '{quote_concat_path(output_paths[-1])}'\n")

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_video_path),
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        print("ffmpeg was not found; MP4 generation skipped.")
        return None
    except subprocess.CalledProcessError as exc:
        print(f"ffmpeg failed with exit code {exc.returncode}; MP4 generation skipped.")
        return None

    print(f"Saved selected case frame video to {output_video_path}")
    return output_video_path


def plot_selected_case_maps(
    ds_crop,
    crop_specs,
    current_spec,
    reports_df,
    case_row,
    crop_row,
    output_dir,
    orography=None,
):
    n_frames = get_frame_count(ds_crop)
    frame_output_dir = output_dir / (
        f"top_report_case_{case_row['date']}_view{current_spec['view_id']:03d}_frames"
    )
    frame_output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = []
    for frame_index in range(n_frames):
        output_paths.append(
            plot_selected_case_frame_map(
                ds_crop,
                crop_specs,
                current_spec,
                reports_df,
                case_row,
                crop_row,
                frame_output_dir,
                frame_index,
                orography=orography,
            )
        )

    print(f"Saved {len(output_paths)} frame maps to {frame_output_dir}")
    video_path = frame_output_dir / (
        f"top_report_case_{case_row['date']}_view{current_spec['view_id']:03d}_frames.mp4"
    )
    save_frame_maps_video(output_paths, video_path)
    return output_paths


def process_case(case_row, test_df, csv_files, crop_roots, output_dir, orography):
    reports_df = load_reports_for_case(csv_files["essl_reports"], case_row)

    crop_specs = build_view_specs_from_crop_files(test_df, crop_roots, case_row)
    current_spec, n_reports_in_view = select_view_with_most_reports(crop_specs, reports_df)
    crop_row = select_crop_for_case_and_view(test_df, case_row, current_spec, reports_df)
    crop_path = resolve_crop_path(crop_row["crop"], crop_roots)

    print(
        "Selected case "
        f"{case_row['date']} with {int(case_row['num_reports'])} reports; "
        f"view {current_spec['view_id']:03d} contains {n_reports_in_view} reports."
    )
    print(f"Using crop file: {crop_path}")

    with xr.open_dataset(crop_path) as ds_crop:
        plot_selected_case_maps(
            ds_crop,
            crop_specs,
            current_spec,
            reports_df,
            case_row,
            crop_row,
            output_dir,
            orography=orography,
        )


def main():
    csv_files, crop_roots, output_dir = read_config()
    test_df = load_test_summary(csv_files["testing"])
    test_dates = set(test_df["time_start"].dt.date)
    test_cases = load_test_cases(csv_files["essl_cases"], test_dates)
    test_reports = load_reports_for_video_summary_days(
        csv_files["essl_reports"],
        test_dates,
    )
    top_cases = test_cases.sort_values("num_reports", ascending=False).head(N_TOP_CASES)
    orography = load_orography_from_dem(DEM_PATH)

    plot_video_summary_reports_map(test_reports, output_dir, orography=orography)
    plot_video_summary_histograms(test_dates, test_reports, output_dir)

    print(f"Processing top {len(top_cases)} ESSL test cases by number of reports.")
    for case_index, (_, case_row) in enumerate(top_cases.iterrows(), start=1):
        print(f"\nProcessing case {case_index}/{len(top_cases)}")
        process_case(case_row, test_df, csv_files, crop_roots, output_dir, orography)


if __name__ == "__main__":
    main()
