"""
Code to plot figure 1 of the paper "Discovering Convective Storm Evolution Through Self-Supervised Learning of Satellite Imagery"
This code should produce a multipanel figure with the following panels:
First row should contain panel A and panel B, with Panel A occupying 1/3 of the width and Panel B occupying 2/3 of the width.
Second row should contain panel C, D, E and F with each panel occupying the same width (1/4 of the total width).
- Panel A: a map of the domain EXPATS with the orography overlaid in gray, and the 10.8 micron channel of the MSG satellite plotted in color.
 Also, one example crop of 100 x 100 pixels should be shown as a solid thick rectangle on the map with a colored line,
 and the corresponding crop of the MSG satellite image should be shown in Panel B.
- Panel B: the sequence of the 8 frames of a video crop, where we plot the 10.8 micron channel of the MSG satellite,
with cloud mask overlaid in hatched orange areas.
- Panel C: sequence of 8 frames of the same video crop for lightning density map, plotted slightly overimposed along a diagonal time development line,
- panel D: same as panel C but for the imerg precipitation rate, plotted slightly overimposed along a diagonal time development line.
- panel E: same as panel C but for the cloud top height, plotted slightly overimposed along a diagonal time development line.
- panel F: same as panel C but for the cloud optical thickness, plotted slightly overimposed along a diagonal time development line.

The script selects one representative crop automatically from the configured run, unless a
specific crop is passed via --crop. By default it chooses the crop closest to a cluster
centroid within the selected dataset split.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import ScaledTranslation
import numpy as np
import pandas as pd
import xarray as xr

try:
    import cmcrameri.cm as cmc
except ImportError:  # pragma: no cover - optional dependency
    cmc = None

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError:  # pragma: no cover - optional dependency
    ccrs = None
    cfeature = None

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from utils.buckets.credentials_buckets import (  # noqa: E402
    S3_ACCESS_KEY,
    S3_ENDPOINT_URL,
    S3_SECRET_ACCESS_KEY,
)
from utils.buckets.get_data_from_buckets import (  # noqa: E402
    Initialize_s3_client,
    read_file,
)
from utils.configs import load_config  # noqa: E402
from utils.plotting.class_colors import colors_per_class1_names  # noqa: E402

CONFIG_PATH = REPO_ROOT / "configs" / "process_run_GRL.yaml"
VARIABLE_METADATA_PATH = REPO_ROOT / "configs" / "variables_metadata.yaml"
DEM_PATH_CANDIDATES = (
    Path("/data1/DEM_EXPATS_0.01x0.01.nc"),
    Path("/data1/other_data/DEM_EXPATS_0.01x0.01.nc"),
)
EXPATS_DOMAIN = (5.0, 16.0, 42.0, 51.5)
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
PANEL_LABEL_SIZE = 21
TITLE_SIZE = 21
SMALL_TITLE_SIZE = 21
TIME_LABEL_SIZE = 14
COLORBAR_LABEL_SIZE = 18
COLORBAR_TICK_LABEL_SIZE = 18
OUTPUT_FILENAME = "figure1_data_methods.png"
MSG_VARIABLE_CANDIDATES = ("IR_108", "ir_108", "BT108", "bt108")
CMA_VARIABLE_CANDIDATES = ("cma", "CMA")
BT108_COLORMAP = cmc.romaO_r if cmc is not None else plt.get_cmap("RdBu_r")
LIGHTNING_COLORMAP = getattr(cmc, "lipari_r", plt.get_cmap("magma")) if cmc is not None else plt.get_cmap("magma")
PRECIPITATION_COLORMAP = getattr(cmc, "lapaz_r", plt.get_cmap("Blues")) if cmc is not None else plt.get_cmap("Blues")
CTH_COLORMAP = getattr(cmc, "batlow", plt.get_cmap("viridis")) if cmc is not None else plt.get_cmap("viridis")
COT_COLORMAP = getattr(cmc, "bam", plt.get_cmap("cividis")) if cmc is not None else plt.get_cmap("cividis")
HORIZONTAL_COLORBAR_BOUNDS = [0.1, -0.02, 0.8, 0.035]
PANEL_B_COLORBAR_BOUNDS = [0.12, 0.35, 0.76, 0.25]
TOP_VIEW_COUNT = 4
TOP_VIEW_COLORS = ("#e66101", "#1f78b4", "#33a02c", "#984ea3")
MAX_CROP_OVERLAP_FRACTION = 0.2
INCLUDE_ANCILLARY_PANELS = True
OUTPUT_DPI = 220
USE_REMOTE_DOMAIN_BACKGROUND = True

# Default in-script selection. Adjust these values when you want to generate a
# different Figure 1 example without passing command-line arguments.
DEFAULT_RUN_ARGUMENTS = {
    "dataset": "training",
    "label": None,
    "crop": None,
    "date": "2023-07-24",
    "start_time": "15:00",
    "end_time": "17:00",
    "lat_min": 44.0,
    "lat_max": 48.0,
    "lon_min": 11.0,
    "lon_max": 15.0,
    "output": "/sat_data/output/grl_2026_k10/figs/figure1_data_methods.png",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the Figure 1 data/methods overview plot.")
    parser.add_argument(
        "--dataset",
        choices=("training", "testing"),
        default=DEFAULT_RUN_ARGUMENTS["dataset"],
        help="Dataset split used to select the example crop.",
    )
    parser.add_argument(
        "--label",
        type=int,
        default=DEFAULT_RUN_ARGUMENTS["label"],
        help="Optional class label used to restrict the example-crop selection.",
    )
    parser.add_argument(
        "--crop",
        default=DEFAULT_RUN_ARGUMENTS["crop"],
        help="Specific crop filename or full path to use instead of automatic selection.",
    )
    parser.add_argument(
        "--date",
        default=DEFAULT_RUN_ARGUMENTS["date"],
        help="Optional UTC date used to filter candidate crops, formatted as YYYY-MM-DD.",
    )
    parser.add_argument(
        "--start-time",
        default=DEFAULT_RUN_ARGUMENTS["start_time"],
        help="Optional lower bound for crop start time on the selected date, formatted as HH:MM.",
    )
    parser.add_argument(
        "--end-time",
        default=DEFAULT_RUN_ARGUMENTS["end_time"],
        help="Optional upper bound for crop start time on the selected date, formatted as HH:MM.",
    )
    parser.add_argument(
        "--lat-min",
        type=float,
        default=DEFAULT_RUN_ARGUMENTS["lat_min"],
        help="Optional minimum crop-center latitude used to filter candidate crops.",
    )
    parser.add_argument(
        "--lat-max",
        type=float,
        default=DEFAULT_RUN_ARGUMENTS["lat_max"],
        help="Optional maximum crop-center latitude used to filter candidate crops.",
    )
    parser.add_argument(
        "--lon-min",
        type=float,
        default=DEFAULT_RUN_ARGUMENTS["lon_min"],
        help="Optional minimum crop-center longitude used to filter candidate crops.",
    )
    parser.add_argument(
        "--lon-max",
        type=float,
        default=DEFAULT_RUN_ARGUMENTS["lon_max"],
        help="Optional maximum crop-center longitude used to filter candidate crops.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_RUN_ARGUMENTS["output"],
        help="Optional custom output path. Defaults to the configured figures directory.",
    )
    return parser.parse_args()


def load_runtime_config() -> tuple[dict, dict]:
    config = load_config(str(CONFIG_PATH))
    variable_metadata = load_config(str(VARIABLE_METADATA_PATH)).get("variables", {})
    return config, variable_metadata


def find_existing_path(candidates: Iterable[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def open_dataset_any(source, *, from_bytes: bool = False) -> xr.Dataset:
    engines = (None, "h5netcdf", "netcdf4", "scipy")
    last_error = None

    for engine in engines:
        try:
            if from_bytes:
                dataset = xr.open_dataset(io.BytesIO(source)) if engine is None else xr.open_dataset(io.BytesIO(source), engine=engine)
            else:
                dataset = xr.open_dataset(source) if engine is None else xr.open_dataset(source, engine=engine)
            return dataset
        except Exception as exc:  # pragma: no cover - backend availability is environment-specific
            last_error = exc

    raise RuntimeError(f"Could not open dataset {source!r}: {last_error}")


def normalize_time_coordinate(ds: xr.Dataset) -> xr.Dataset:
    if "time" not in ds.indexes:
        return ds
    if isinstance(ds.indexes["time"], xr.CFTimeIndex):
        ds = ds.assign_coords(time=ds["time"].astype("datetime64[ns]"))
    return ds


def find_data_array(ds: xr.Dataset, candidates: Iterable[str]) -> xr.DataArray:
    lower_name_map = {name.lower(): name for name in ds.data_vars}
    for candidate in candidates:
        variable_name = lower_name_map.get(candidate.lower())
        if variable_name is not None:
            return ds[variable_name]
    raise KeyError(f"None of the candidate variables {tuple(candidates)} were found in dataset variables {tuple(ds.data_vars)}")


def find_orography_field(ds: xr.Dataset) -> xr.DataArray | None:
    lower_name_map = {name.lower(): name for name in ds.data_vars}
    for candidate in OROGRAPHY_VARIABLE_CANDIDATES:
        variable_name = lower_name_map.get(candidate.lower())
        if variable_name is not None and {"lat", "lon"} <= set(ds[variable_name].dims):
            orography = ds[variable_name]
            if "time" in orography.dims:
                orography = orography.isel(time=0)
            return orography

    for data_array in ds.data_vars.values():
        if {"lat", "lon"} <= set(data_array.dims):
            return data_array.isel(time=0) if "time" in data_array.dims else data_array
    return None


def load_orography() -> xr.DataArray | None:
    dem_path = find_existing_path(DEM_PATH_CANDIDATES)
    if dem_path is None:
        print("No DEM file found; Panel A will be plotted without orography.")
        return None

    try:
        ds_dem = open_dataset_any(dem_path)
    except Exception as exc:
        print(f"Could not open DEM file {dem_path}: {exc}")
        return None

    return find_orography_field(ds_dem)


def get_output_path(config: dict, requested_output: str | None) -> Path:
    if requested_output:
        return Path(requested_output)
    output_dir = Path(config["output_files"]["figures_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / OUTPUT_FILENAME


def load_selection_dataframe(config: dict, dataset: str) -> pd.DataFrame:
    features_key = "features_training" if dataset == "training" else "features_testing"
    summary_key = "training_video_summary" if dataset == "training" else "testing_video_summary"

    features_df = pd.read_csv(config["output_files"][features_key])
    summary_df = pd.read_csv(config["output_files"][summary_key])

    features_df = features_df.copy()
    features_df["crop"] = features_df["path"].map(lambda value: Path(value).name)
    features_df["crop_path"] = features_df["path"]

    merged = summary_df.merge(
        features_df[["crop", "crop_path", "label", "distance"]],
        on=["crop", "label"],
        how="inner",
    )
    merged = merged[merged["label"] != -100].copy()
    merged["time_start"] = pd.to_datetime(merged["time_start"], errors="coerce")
    merged["time_end"] = pd.to_datetime(merged["time_end"], errors="coerce")
    if merged.empty:
        raise ValueError("No crops were found after merging the feature CSV and video-summary CSV.")
    return merged


def filter_selection_by_time_window(
    df: pd.DataFrame,
    dataset: str,
    date_string: str | None,
    start_time_string: str | None,
    end_time_string: str | None,
) -> pd.DataFrame:
    if date_string is None and start_time_string is None and end_time_string is None:
        return df

    if date_string is None:
        raise ValueError("--date is required when using --start-time or --end-time.")

    selected_date = pd.Timestamp(date_string)
    filtered = df[df["time_start"].dt.date == selected_date.date()].copy()

    if start_time_string is not None:
        start_time = pd.Timestamp(f"{date_string} {start_time_string}")
        filtered = filtered[filtered["time_start"] >= start_time]

    if end_time_string is not None:
        end_time = pd.Timestamp(f"{date_string} {end_time_string}")
        filtered = filtered[filtered["time_start"] < end_time]

    if filtered.empty:
        start_label = start_time_string or "00:00"
        end_label = end_time_string or "24:00"
        raise ValueError(
            f"No {dataset} crops were found on {date_string} with start times between {start_label} and {end_label}."
        )

    return filtered


def filter_selection_by_lat_lon_bounds(
    df: pd.DataFrame,
    dataset: str,
    lat_min: float | None,
    lat_max: float | None,
    lon_min: float | None,
    lon_max: float | None,
) -> pd.DataFrame:
    if lat_min is None and lat_max is None and lon_min is None and lon_max is None:
        return df

    filtered = df.copy()
    if lat_min is not None:
        filtered = filtered[filtered["lat_mid"] >= lat_min]
    if lat_max is not None:
        filtered = filtered[filtered["lat_mid"] <= lat_max]
    if lon_min is not None:
        filtered = filtered[filtered["lon_mid"] >= lon_min]
    if lon_max is not None:
        filtered = filtered[filtered["lon_mid"] <= lon_max]

    if filtered.empty:
        raise ValueError(
            f"No {dataset} crops were found inside lat [{lat_min}, {lat_max}] and lon [{lon_min}, {lon_max}]."
        )

    return filtered


def select_example_crop(df: pd.DataFrame, dataset: str, label: int | None, crop: str | None) -> pd.Series:
    candidates = df.copy()
    if label is not None:
        candidates = candidates[candidates["label"] == label]
    if crop is not None:
        crop_name = Path(crop).name
        crop_matches = candidates[candidates["crop"] == crop_name].copy()
        if crop_matches.empty and Path(crop).exists():
            crop_matches = candidates[candidates["crop_path"] == str(Path(crop))].copy()
        if crop_matches.empty:
            raise ValueError(f"Crop {crop!r} was not found in the {dataset} selection table.")
        candidates = crop_matches

    if candidates.empty:
        qualifier = f" for label {label}" if label is not None else ""
        raise ValueError(f"No {dataset} crops are available{qualifier}.")

    candidates["distance"] = pd.to_numeric(candidates["distance"], errors="coerce")
    candidates = candidates.dropna(subset=["distance"]).sort_values("distance", ascending=True)
    return candidates.iloc[0]


def get_crop_root(config: dict, dataset: str) -> Path:
    features_config = config.get("features_preparation", {})
    if dataset == "training":
        return Path(features_config["crops_path"])
    return Path(features_config["crops_test_path"])


def resolve_crop_path(row: pd.Series, crop_root: Path) -> Path:
    crop_path = Path(row["crop_path"])
    if crop_path.exists():
        return crop_path

    candidate = crop_root / row["crop"]
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"Could not find crop NetCDF for {row['crop']} at {crop_path} or {candidate}")


def get_time_values(ds_crop: xr.Dataset) -> np.ndarray:
    if "time" not in ds_crop:
        raise KeyError("Selected crop does not contain a time coordinate.")
    return pd.to_datetime(ds_crop["time"].values).to_numpy(dtype="datetime64[ns]")


def get_frame_data(data_array: xr.DataArray, frame_index: int) -> xr.DataArray:
    return data_array.isel(time=frame_index) if "time" in data_array.dims else data_array


def get_frame_lat_lon(ds_crop: xr.Dataset, frame_index: int) -> tuple[xr.DataArray, xr.DataArray]:
    lat = ds_crop["lat"]
    lon = ds_crop["lon"]
    if "time" in lat.dims:
        lat = lat.isel(time=frame_index)
    if "time" in lon.dims:
        lon = lon.isel(time=frame_index)
    return lat, lon


def build_local_variable_frames(ds_crop: xr.Dataset, candidates: Iterable[str], times: np.ndarray) -> list[xr.DataArray]:
    data_array = find_data_array(ds_crop, candidates)
    frames: list[xr.DataArray] = []
    for frame_index in range(len(times)):
        frame = get_frame_data(data_array, frame_index)
        lat_frame, lon_frame = get_frame_lat_lon(ds_crop, frame_index)
        frames.append(frame.assign_coords(lat=lat_frame, lon=lon_frame))
    return frames


def get_crop_extent(ds_crop: xr.Dataset) -> tuple[float, float, float, float]:
    lat_values = np.asarray(ds_crop["lat"].values, dtype=float)
    lon_values = np.asarray(ds_crop["lon"].values, dtype=float)
    return (
        float(np.nanmin(lon_values)),
        float(np.nanmax(lon_values)),
        float(np.nanmin(lat_values)),
        float(np.nanmax(lat_values)),
    )


def extents_overlap(
    first_extent: tuple[float, float, float, float],
    second_extent: tuple[float, float, float, float],
) -> bool:
    return not (
        first_extent[1] <= second_extent[0]
        or second_extent[1] <= first_extent[0]
        or first_extent[3] <= second_extent[2]
        or second_extent[3] <= first_extent[2]
    )


def compute_overlap_fraction(
    first_extent: tuple[float, float, float, float],
    second_extent: tuple[float, float, float, float],
) -> float:
    overlap_lon = max(0.0, min(first_extent[1], second_extent[1]) - max(first_extent[0], second_extent[0]))
    overlap_lat = max(0.0, min(first_extent[3], second_extent[3]) - max(first_extent[2], second_extent[2]))
    overlap_area = overlap_lon * overlap_lat
    if overlap_area == 0.0:
        return 0.0

    first_area = max(0.0, first_extent[1] - first_extent[0]) * max(0.0, first_extent[3] - first_extent[2])
    second_area = max(0.0, second_extent[1] - second_extent[0]) * max(0.0, second_extent[3] - second_extent[2])
    min_area = min(first_area, second_area)
    if min_area == 0.0:
        return 1.0
    return overlap_area / min_area


def estimate_crop_extent_from_center(
    row: pd.Series,
    lon_span: float,
    lat_span: float,
) -> tuple[float, float, float, float]:
    lon_mid = float(row["lon_mid"])
    lat_mid = float(row["lat_mid"])
    half_lon_span = 0.5 * lon_span
    half_lat_span = 0.5 * lat_span
    return (
        lon_mid - half_lon_span,
        lon_mid + half_lon_span,
        lat_mid - half_lat_span,
        lat_mid + half_lat_span,
    )


def compute_center_distance(first_row: pd.Series, second_row: pd.Series) -> float:
    lat_delta = float(first_row["lat_mid"]) - float(second_row["lat_mid"])
    lon_delta = float(first_row["lon_mid"]) - float(second_row["lon_mid"])
    return float(np.hypot(lat_delta, lon_delta))


def build_bucket_map(variable_metadata: dict) -> dict[str, str]:
    return {
        "msg": "expats-msg-training",
        "crop": "expats-msg-training",
        "cmsaf": variable_metadata["cth"]["bucket_name"],
        "imerg": variable_metadata["precipitation"]["bucket_name"],
        "euclid": variable_metadata["euclid_msg_grid"]["bucket_name"],
    }


def build_remote_key(variable_name: str, timestamp: np.datetime64) -> tuple[str, str]:
    ts = pd.Timestamp(timestamp)
    if variable_name == "IR_108":
        key = (
            f"/data/sat/msg/ml_train_crops/IR_108-WV_062-CMA_FULL_EXPATS_DOMAIN/"
            f"{ts.year:04d}/{ts.month:02d}/merged_MSG_CMSAF_{ts.year:04d}-{ts.month:02d}-{ts.day:02d}.nc"
        )
        return "msg", key
    if variable_name == "precipitation":
        return "imerg", f"IMERG_daily_{ts.year:04d}-{ts.month:02d}-{ts.day:02d}.nc"
    if variable_name == "euclid_msg_grid":
        return "euclid", (
            f"{ts.year:04d}/{ts.month:02d}/"
            f"EUCLID_total_lightning_{ts.year:04d}{ts.month:02d}{ts.day:02d}.nc"
        )
    return "cmsaf", f"MCP_{ts.year:04d}-{ts.month:02d}-{ts.day:02d}_regrid.nc"


class RemoteDatasetCache:
    def __init__(self, bucket_map: dict[str, str]):
        self.bucket_map = bucket_map
        self.s3 = Initialize_s3_client(S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY)
        self._cache: dict[tuple[str, str], xr.Dataset] = {}

    def get_dataset(self, variable_name: str, timestamp: np.datetime64) -> xr.Dataset:
        bucket_key, object_key = build_remote_key(variable_name, timestamp)
        cache_key = (bucket_key, object_key)
        if cache_key not in self._cache:
            payload = read_file(self.s3, object_key, self.bucket_map[bucket_key])
            if payload is None:
                raise FileNotFoundError(
                    f"Could not load {object_key} from bucket {self.bucket_map[bucket_key]} for variable {variable_name}."
                )
            dataset = open_dataset_any(payload, from_bytes=True)
            self._cache[cache_key] = normalize_time_coordinate(dataset)
        return self._cache[cache_key]

    def close(self) -> None:
        for dataset in self._cache.values():
            try:
                dataset.close()
            except Exception:
                pass
        self._cache.clear()


def subset_spatial(da: xr.DataArray, lon_min: float, lon_max: float, lat_min: float, lat_max: float) -> xr.DataArray:
    lon_values = np.asarray(da["lon"].values, dtype=float)
    lat_values = np.asarray(da["lat"].values, dtype=float)

    lon_slice = slice(lon_min, lon_max) if lon_values[0] <= lon_values[-1] else slice(lon_max, lon_min)
    lat_slice = slice(lat_min, lat_max) if lat_values[0] <= lat_values[-1] else slice(lat_max, lat_min)
    return da.sel(lon=lon_slice, lat=lat_slice)


def extract_remote_frame(
    cache: RemoteDatasetCache,
    variable_name: str,
    timestamp: np.datetime64,
    extent: tuple[float, float, float, float],
) -> xr.DataArray:
    lon_min, lon_max, lat_min, lat_max = extent
    dataset = cache.get_dataset(variable_name, timestamp)
    data_array = find_data_array(dataset, (variable_name,))
    data_array = subset_spatial(data_array, lon_min, lon_max, lat_min, lat_max)

    if "time" in data_array.coords:
        data_array = data_array.sel(time=pd.Timestamp(timestamp), method="nearest")
    return data_array.squeeze()


def get_variable_norm(variable_name: str, variable_metadata: dict, values: list[np.ndarray]) -> Normalize:
    metadata = variable_metadata.get(variable_name, {})
    vmin = metadata.get("vmin")
    vmax = metadata.get("vmax")

    finite_batches = [value[np.isfinite(value)] for value in values if np.isfinite(value).any()]

    if variable_name == "IR_108":
        if not finite_batches:
            return Normalize(vmin=220.0, vmax=320.0)
        finite = np.concatenate(finite_batches)
        return Normalize(vmin=float(np.nanpercentile(finite, 2)), vmax=float(np.nanpercentile(finite, 98)))

    if variable_name == "cth":
        if vmin is None:
            vmin = 0.0
        if vmax is None:
            vmax = 12.0
        return Normalize(vmin=float(vmin), vmax=float(vmax))

    if vmin is not None and vmax is not None:
        return Normalize(vmin=float(vmin), vmax=float(vmax))

    if not finite_batches:
        return Normalize(vmin=0.0, vmax=1.0)

    finite = np.concatenate(finite_batches)
    upper = float(np.nanpercentile(finite, 98))
    upper = upper if upper > 0 else 1.0
    return Normalize(vmin=0.0, vmax=upper)


def scale_variable_for_display(variable_name: str, data_array: xr.DataArray) -> np.ndarray:
    values = np.asarray(data_array.values, dtype=float)
    if variable_name == "cth":
        values = values * 0.001
    return values


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.01,
        0.99,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.5},
        zorder=10,
    )


def add_panel_title(ax: plt.Axes, title: str, x_offset_cm: float = 0.0) -> None:
    title_transform = ax.transAxes + ScaledTranslation(x_offset_cm / 2.54, 0.0, ax.figure.dpi_scale_trans)
    ax.text(
        0.0,
        1.02,
        title,
        transform=title_transform,
        ha="left",
        va="bottom",
        fontsize=TITLE_SIZE,
        fontweight="bold",
    )


def add_horizontal_inset_colorbar(fig: plt.Figure, host_ax: plt.Axes, mesh, label: str):
    cbar_ax = host_ax.inset_axes(HORIZONTAL_COLORBAR_BOUNDS)
    colorbar = fig.colorbar(mesh, cax=cbar_ax, orientation="horizontal")
    colorbar.set_label(label, fontsize=COLORBAR_LABEL_SIZE)
    colorbar.ax.tick_params(labelsize=COLORBAR_TICK_LABEL_SIZE)
    return colorbar


def plot_orography_contours(ax: plt.Axes, orography: xr.DataArray | None, extent: tuple[float, float, float, float]) -> None:
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

    values = np.asarray(oro.values).squeeze()
    if values.ndim != 2 or np.all(np.isnan(values)):
        return

    levels = np.arange(0, 4001, 500)
    levels = levels[(levels >= np.nanmin(values)) & (levels <= max(4000, np.nanmax(values)))]
    if len(levels) < 2:
        return

    contours = ax.contour(
        oro.lon.values,
        oro.lat.values,
        values,
        levels=levels,
        colors="0.35",
        linewidths=0.7,
        alpha=0.7,
        transform=ccrs.PlateCarree() if ccrs is not None else ax.transData,
        zorder=3,
    )
    ax.clabel(contours, inline=True, fontsize=8, fmt="%d m")


def plot_panel_a(
    ax: plt.Axes,
    fig: plt.Figure,
    domain_ir: xr.DataArray | None,
    crop_extent: tuple[float, float, float, float],
    label_color: str,
    orography: xr.DataArray | None,
    panel_title: str,
) -> None:

    # Plot the MSG 10.8 micron channel over the EXPATS domain with orography contours and a rectangle for the crop extent
    lon_min, lon_max, lat_min, lat_max = EXPATS_DOMAIN
    use_geo_axes = ccrs is not None and hasattr(ax, "add_feature")
    mesh = None
    if domain_ir is not None:
        ir_values = np.asarray(domain_ir.values, dtype=float)
        finite = ir_values[np.isfinite(ir_values)]
        vmin = float(np.nanpercentile(finite, 2)) if finite.size else 220.0
        vmax = float(np.nanpercentile(finite, 98)) if finite.size else 320.0
        mesh = ax.pcolormesh(
            domain_ir["lon"].values,
            domain_ir["lat"].values,
            ir_values,
            cmap=BT108_COLORMAP,
            norm=Normalize(vmin=vmin, vmax=vmax),
            shading="auto",
            transform=ccrs.PlateCarree() if use_geo_axes else ax.transData,
            zorder=1,
        )
    else:
        ax.set_facecolor("0.96")
    # add orography contours if available
    plot_orography_contours(ax, orography, EXPATS_DOMAIN)

    if cfeature is not None and use_geo_axes:
        ax.add_feature(cfeature.BORDERS, edgecolor="black", linewidth=0.8, zorder=4)
        ax.add_feature(cfeature.COASTLINE, edgecolor="black", linewidth=0.6, zorder=4)

    rect = Rectangle(
        (crop_extent[0], crop_extent[2]),
        crop_extent[1] - crop_extent[0],
        crop_extent[3] - crop_extent[2],
        linewidth=2.8,
        edgecolor=label_color,
        facecolor="none",
        transform=ccrs.PlateCarree() if use_geo_axes else ax.transData,
        zorder=5,
    )
    ax.add_patch(rect)
    if use_geo_axes:
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    else:
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, color="0.88", linewidth=0.8)
    add_panel_title(ax, f"a) {panel_title}", x_offset_cm=-1.7)

    #add_panel_label(ax, "A")


def plot_cloud_mask_hatching(ax: plt.Axes, lon: np.ndarray, lat: np.ndarray, cma_values: np.ndarray) -> bool:
    if cma_values.ndim != 2:
        return False
    no_cloud = cma_values < 0.5
    if not np.any(no_cloud):
        return False

    with plt.rc_context({"hatch.color": "white", "hatch.linewidth": 1.0}):
        contour = ax.contourf(
            lon,
            lat,
            no_cloud.astype(int),
            levels=[0.5, 1.5],
            colors="none",
            hatches=["////"],
            zorder=3,
        )
    if hasattr(contour, "collections"):
        for collection in contour.collections:
            collection.set_edgecolor("white")
            collection.set_linewidth(0.8)
    return True


def plot_panel_b(parent_spec, fig: plt.Figure, ds_crop: xr.Dataset, times: np.ndarray, label_color: str) -> None:
    panel_ax = fig.add_subplot(parent_spec)
    panel_ax.set_axis_off()
    add_panel_title(panel_ax, "b) Video sequence for the crop")
    subgrid = parent_spec.subgridspec(4, len(times), height_ratios=[0.15, 0.5, 0.25, 0.15], hspace=0.0, wspace=0.02)
    ir_data = find_data_array(ds_crop, MSG_VARIABLE_CANDIDATES)
    cma_data = find_data_array(ds_crop, CMA_VARIABLE_CANDIDATES)
    ir_values = [np.asarray(get_frame_data(ir_data, frame_index).values, dtype=float) for frame_index in range(len(times))]
    norm = get_variable_norm("IR_108", {}, ir_values)
    hatch_shown = False
    axes = []
    mesh = None

    for frame_index, timestamp in enumerate(times):
        ax = fig.add_subplot(subgrid[1, frame_index])
        axes.append(ax)
        ax.set_box_aspect(1)
        #ax.set_anchor("N")
        ir_frame = get_frame_data(ir_data, frame_index)
        cma_frame = get_frame_data(cma_data, frame_index)
        lat_frame, lon_frame = get_frame_lat_lon(ds_crop, frame_index)
        mesh = ax.pcolormesh(
            lon_frame.values,
            lat_frame.values,
            np.asarray(ir_frame.values, dtype=float),
            cmap=BT108_COLORMAP,
            norm=norm,
            shading="auto",
            zorder=1,
        )
        ax.set_aspect("equal")

        hatch_shown = plot_cloud_mask_hatching(
            ax,
            lon_frame.values,
            lat_frame.values,
            np.asarray(cma_frame.values, dtype=float),
        ) or hatch_shown
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.8)
            spine.set_edgecolor(label_color)

        ax.set_title(pd.Timestamp(timestamp).strftime("%H:%M") + " UTC", fontsize=SMALL_TITLE_SIZE)

    if mesh is not None:
        cbar_host = fig.add_subplot(subgrid[2, :])
        cbar_host.set_axis_off()
        cbar_ax = cbar_host.inset_axes(PANEL_B_COLORBAR_BOUNDS)
        colorbar = fig.colorbar(mesh, cax=cbar_ax, orientation="horizontal")
        colorbar.set_label("MSG 10.8 µm brightness temperature (K)", fontsize=COLORBAR_LABEL_SIZE)
        colorbar.ax.tick_params(labelsize=COLORBAR_TICK_LABEL_SIZE)
    if hatch_shown:
        axes[-1].legend(
            handles=[Patch(facecolor="none", edgecolor="white", hatch="////", label="No-cloud pixels")],
            loc="lower right",
            fontsize=9,
            frameon=True,
        )


def compute_inset_positions(n_frames: int) -> list[tuple[float, float, float, float]]:
    size = 0.34
    x_start = 0.02
    y_start = 0.04
    x_step = 0.09
    y_step = 0.075
    return [(x_start + index * x_step, y_start + index * y_step, size, size) for index in range(n_frames)]


def plot_diagonal_panel(
    ax: plt.Axes,
    fig: plt.Figure,
    panel_label: str,
    title: str,
    variable_name: str,
    frames: list[xr.DataArray] | None,
    times: np.ndarray,
    variable_metadata: dict,
    cmap: str,
    hatch_frames: list[xr.DataArray] | None = None,
) -> None:
    ax.set_axis_off()
    add_panel_title(ax, title)
    if not frames:
        ax.text(
            0.5,
            0.5,
            "Data unavailable\nfor selected day",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=16,
            color="0.35",
        )
        return

    # 
    ax.annotate(
        "time",
        xy=(0.92, 0.9),
        xytext=(0.18, 0.18),
        textcoords="axes fraction",
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "linewidth": 1.4, "color": "0.3"},
        fontsize=12,
        color="0.3",
    )

    display_values = [scale_variable_for_display(variable_name, frame) for frame in frames]
    norm = get_variable_norm(variable_name, variable_metadata, display_values)
    positions = compute_inset_positions(len(frames))
    mesh = None
    text_color = "white" if variable_name in {"IR_108", "euclid_msg_grid"} else "black"
    text_box_color = "black" if text_color == "white" else "white"
    hatch_shown = False

    for frame_index, (frame, display_value, (x0, y0, width, height)) in enumerate(zip(frames, display_values, positions)):
        inset = ax.inset_axes([x0, y0, width, height])
        mesh = inset.pcolormesh(
            frame["lon"].values,
            frame["lat"].values,
            display_value,
            cmap=cmap,
            norm=norm,
            shading="auto",
        )
        if hatch_frames is not None:
            hatch_frame = hatch_frames[frame_index]
            hatch_shown = plot_cloud_mask_hatching(
                inset,
                frame["lon"].values,
                frame["lat"].values,
                np.asarray(hatch_frame.values, dtype=float),
            ) or hatch_shown
        inset.set_xticks([])
        inset.set_yticks([])
        timestamp = pd.Timestamp(times[frame_index]).strftime("%H:%M")
        inset.text(
            0.04,
            0.94,
            timestamp + " UTC",
            transform=inset.transAxes,
            ha="left",
            va="top",
            fontsize=TIME_LABEL_SIZE,
            color=text_color,
            bbox={"facecolor": text_box_color, "edgecolor": "none", "alpha": 0.45, "pad": 0.8},
        )

    if mesh is not None:
        metadata = variable_metadata.get(variable_name, {})
        label = metadata.get("long_name", variable_name)
        unit = metadata.get("unit")
        if unit:
            label = f"{label} ({unit})"
        add_horizontal_inset_colorbar(fig, ax, mesh, label)
    if hatch_shown:
        ax.legend(
            handles=[Patch(facecolor="none", edgecolor="white", hatch="////", label="No-cloud pixels")],
            loc="lower right",
            fontsize=9,
            frameon=True,
        )


def plot_panel_g(ax: plt.Axes) -> None:
    ax.set_axis_off()
    add_panel_title(ax, "g) Deep learning model")


def save_ir108_diagonal_figure(
    output_path: Path,
    ds_crop: xr.Dataset,
    times: np.ndarray,
    variable_metadata: dict,
) -> Path:
    diagonal_output_path = output_path.with_name(f"{output_path.stem}_ir108_diagonal{output_path.suffix}")
    ir_frames = build_local_variable_frames(ds_crop, MSG_VARIABLE_CANDIDATES, times)
    cma_frames = build_local_variable_frames(ds_crop, CMA_VARIABLE_CANDIDATES, times)
    if len(times) > 5:
        selected_indices = np.linspace(0, len(times) - 1, 5, dtype=int)
        selected_times = times[selected_indices]
        ir_frames = [ir_frames[index] for index in selected_indices]
        cma_frames = [cma_frames[index] for index in selected_indices]
    else:
        selected_times = times

    fig = plt.figure(figsize=(10, 8), constrained_layout=False)
    ax = fig.add_subplot(111)
    plot_diagonal_panel(
        ax,
        fig,
        "h",
        "10.8 µm channel",
        "IR_108",
        ir_frames,
        selected_times,
        variable_metadata,
        cmap=BT108_COLORMAP,
        hatch_frames=cma_frames,
    )
    fig.tight_layout(pad=0.2)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.96, bottom=0.06)
    fig.savefig(diagonal_output_path, dpi=OUTPUT_DPI)
    plt.close(fig)
    return diagonal_output_path


def prepare_remote_frames(
    cache: RemoteDatasetCache,
    variable_name: str,
    times: np.ndarray,
    crop_extent: tuple[float, float, float, float],
) -> list[xr.DataArray]:
    return [extract_remote_frame(cache, variable_name, timestamp, crop_extent) for timestamp in times]


def try_prepare_remote_frames(
    cache: RemoteDatasetCache,
    variable_name: str,
    times: np.ndarray,
    crop_extent: tuple[float, float, float, float],
) -> list[xr.DataArray] | None:
    try:
        return prepare_remote_frames(cache, variable_name, times, crop_extent)
    except Exception as exc:
        print(f"Skipping {variable_name} panels for this crop: {exc}")
        return None


def choose_first_runnable_crop(
    selection_df: pd.DataFrame,
    dataset: str,
    label: int | None,
    crop: str | None,
    crop_root: Path,
) -> pd.Series:
    if crop is not None:
        return select_example_crop(selection_df, dataset, label, crop)

    candidates = selection_df.copy()
    if label is not None:
        candidates = candidates[candidates["label"] == label]
    candidates["distance"] = pd.to_numeric(candidates["distance"], errors="coerce")
    candidates = candidates.dropna(subset=["distance"]).sort_values("distance", ascending=True)

    if candidates.empty:
        qualifier = f" for label {label}" if label is not None else ""
        raise ValueError(f"No {dataset} crops are available{qualifier}.")

    return candidates.iloc[0]


def choose_distinct_crops(
    selection_df: pd.DataFrame,
    dataset: str,
    label: int | None,
    crop: str | None,
    crop_root: Path,
    count: int,
    fallback_dfs: list[pd.DataFrame] | None = None,
) -> list[pd.Series]:
    if crop is not None:
        return [select_example_crop(selection_df, dataset, label, crop)]

    candidate_frames = [selection_df]
    if fallback_dfs is not None:
        candidate_frames.extend(fallback_dfs)

    runnable_rows: list[pd.Series] = []
    seen_crops: set[str] = set()
    for frame_priority, candidate_df in enumerate(candidate_frames):
        candidates = candidate_df.copy()
        if label is not None:
            candidates = candidates[candidates["label"] == label]
        candidates["distance"] = pd.to_numeric(candidates["distance"], errors="coerce")
        candidates = candidates.dropna(subset=["distance", "lat_mid", "lon_mid"]).sort_values("distance", ascending=True)

        for _, row in candidates.iterrows():
            crop_name = str(row["crop"])
            if crop_name in seen_crops:
                continue
            try:
                crop_path = resolve_crop_path(row, crop_root)
            except Exception as exc:
                print(f"Skipping crop {row['crop']} while building the 4-view panel: {exc}")
                continue
            row = row.copy()
            row["selection_priority"] = frame_priority
            row["crop_path"] = str(crop_path)
            runnable_rows.append(row)
            seen_crops.add(crop_name)

    if len(runnable_rows) < count:
        raise ValueError(
            f"Found only {len(runnable_rows)} runnable {dataset} crops after filtering, but {count} are required."
        )

    reference_dataset = None
    try:
        reference_dataset = normalize_time_coordinate(open_dataset_any(runnable_rows[0]["crop_path"]))
        reference_extent = get_crop_extent(reference_dataset)
    finally:
        if reference_dataset is not None:
            reference_dataset.close()

    lon_span = reference_extent[1] - reference_extent[0]
    lat_span = reference_extent[3] - reference_extent[2]
    for row in runnable_rows:
        row["crop_extent"] = estimate_crop_extent_from_center(row, lon_span, lat_span)

    selected_rows = [runnable_rows[0]]
    remaining_rows = runnable_rows[1:]

    while len(selected_rows) < count and remaining_rows:
        best_index = None
        best_score = -np.inf
        best_priority = np.inf
        best_distance = np.inf

        for row_index, row in enumerate(remaining_rows):
            if any(
                compute_overlap_fraction(row["crop_extent"], selected_row["crop_extent"]) > MAX_CROP_OVERLAP_FRACTION
                for selected_row in selected_rows
            ):
                continue
            min_center_distance = min(compute_center_distance(row, selected_row) for selected_row in selected_rows)
            row_distance = float(row["distance"])
            row_priority = int(row.get("selection_priority", 0))
            if (
                min_center_distance > best_score
                or (np.isclose(min_center_distance, best_score) and row_priority < best_priority)
                or (
                    np.isclose(min_center_distance, best_score)
                    and row_priority == best_priority
                    and row_distance < best_distance
                )
            ):
                best_index = row_index
                best_score = min_center_distance
                best_priority = row_priority
                best_distance = row_distance

        if best_index is None:
            break

        selected_rows.append(remaining_rows.pop(best_index))

    if len(selected_rows) < count:
        raise ValueError(
            f"Found only {len(selected_rows)} crops with at most {MAX_CROP_OVERLAP_FRACTION:.0%} overlap after filtering, but {count} are required."
        )

    return selected_rows


def main() -> None:

    # Parse command-line arguments and load configuration
    args = parse_args()
    config, variable_metadata = load_runtime_config()
    full_selection_df = load_selection_dataframe(config, args.dataset)
    time_filtered_df = filter_selection_by_time_window(
        full_selection_df,
        args.dataset,
        args.date,
        args.start_time,
        args.end_time,
    )
    selection_df = filter_selection_by_lat_lon_bounds(
        time_filtered_df,
        args.dataset,
        args.lat_min,
        args.lat_max,
        args.lon_min,
        args.lon_max,
    )
    crop_root = get_crop_root(config, args.dataset)
    cache = RemoteDatasetCache(build_bucket_map(variable_metadata))
    output_path = get_output_path(config, args.output)
    selected_row = choose_first_runnable_crop(selection_df, args.dataset, args.label, args.crop, crop_root)
    crop_path = resolve_crop_path(selected_row, crop_root)
    ds_crop = normalize_time_coordinate(open_dataset_any(crop_path))
    label_value = int(selected_row["label"])
    label_color = colors_per_class1_names.get(str(label_value), "crimson")
    times = get_time_values(ds_crop)
    crop_extent = get_crop_extent(ds_crop)

    print(
        f"Selected crop {selected_row['crop']} from {args.dataset} set "
        f"(label={label_value}, distance={selected_row['distance']})."
    )
    orography = load_orography()

    try:
        domain_ir = extract_remote_frame(cache, "IR_108", times[0], EXPATS_DOMAIN) if USE_REMOTE_DOMAIN_BACKGROUND else None
        if INCLUDE_ANCILLARY_PANELS:
            lightning_frames = try_prepare_remote_frames(cache, "euclid_msg_grid", times, crop_extent)
            precipitation_frames = try_prepare_remote_frames(cache, "precipitation", times, crop_extent)
            cth_frames = try_prepare_remote_frames(cache, "cth", times, crop_extent)
            cot_frames = try_prepare_remote_frames(cache, "cot", times, crop_extent)
            fig = plt.figure(figsize=(24, 20), constrained_layout=False)
            outer = GridSpec(3, 12, figure=fig, height_ratios=[1.0, 1.1, 1.0], hspace=0.18, wspace=0.12)
        else:
            lightning_frames = None
            precipitation_frames = None
            cth_frames = None
            cot_frames = None
            fig = plt.figure(figsize=(24, 15), constrained_layout=False)
            outer = GridSpec(2, 12, figure=fig, height_ratios=[1.0, 1.0], hspace=0.18, wspace=0.12)

        panel_a_projection = ccrs.PlateCarree() if ccrs is not None else None
        ax_a = fig.add_subplot(outer[0, 0:4], projection=panel_a_projection)
        plot_panel_a(
            ax_a,
            fig,
            domain_ir,
            crop_extent,
            label_color,
            orography,
            panel_title=f"Domain overview",
        )

        plot_panel_b(outer[0, 4:12], fig, ds_crop, times, label_color)

        if INCLUDE_ANCILLARY_PANELS:
            ax_c = fig.add_subplot(outer[2, 0:3])
            plot_diagonal_panel(ax_c, fig, "d", "d) Lightning density", "euclid_msg_grid", lightning_frames, times, variable_metadata, cmap=LIGHTNING_COLORMAP)

            ax_d = fig.add_subplot(outer[2, 3:6])
            plot_diagonal_panel(ax_d, fig, "e", "e) Precipitation rate", "precipitation", precipitation_frames, times, variable_metadata, cmap=PRECIPITATION_COLORMAP)

            ax_e = fig.add_subplot(outer[2, 6:9])
            plot_diagonal_panel(ax_e, fig, "f", "f) Cloud top height", "cth", cth_frames, times, variable_metadata, cmap=CTH_COLORMAP)

            ax_f = fig.add_subplot(outer[2, 9:12])
            plot_diagonal_panel(ax_f, fig, "g", "g) Cloud optical thickness", "cot", cot_frames, times, variable_metadata, cmap=COT_COLORMAP)

            ax_g = fig.add_subplot(outer[1, :])
            plot_panel_g(ax_g)
        else:
            ax_g = fig.add_subplot(outer[1, :])
            plot_panel_g(ax_g)

        fig.tight_layout(pad=0.2)
        fig.subplots_adjust(left=0.02, right=0.99, top=0.97, bottom=0.05)
        fig.savefig(output_path, dpi=OUTPUT_DPI)
        plt.close(fig)
        print(f"Saved figure to {output_path}")

        diagonal_output_path = save_ir108_diagonal_figure(output_path, ds_crop, times, variable_metadata)
        print(f"Saved diagonal IR figure to {diagonal_output_path}")
    finally:
        cache.close()
        ds_crop.close()


if __name__ == "__main__":
    main()