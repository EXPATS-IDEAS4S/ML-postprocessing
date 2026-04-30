import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import label


RUN_NAME = "dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc"
OUTPUT_PATH = f"/data1/fig/{RUN_NAME}/epoch_800/test_traj"
PATHWAY_CSV = os.path.join(OUTPUT_PATH, "pathway_analysis/df_pathways_merged_no_dominance.csv")
MIDPOINT_CSV = os.path.join(OUTPUT_PATH, "pathway_analysis/crop_midpoints_from_nc.csv")
ESWD_CSV = os.path.join(OUTPUT_PATH, "eswd-v2-2012-2025_expats.csv")
OUT_DIR = os.path.join(OUTPUT_PATH, "trajectory_quality_stats")

VAR_CANDIDATES = ["IR_108", "ir_108", "IR108", "BT", "bt"]
BT_THRESHOLD_K = 240.0
DOMAIN_EXTENT = [5.0, 16.0, 42.0, 51.5]  # [lon_min, lon_max, lat_min, lat_max]
EVENT_RADIUS_KM = 150.0
DEFAULT_HALF_SIDE_DEG = 0.5


def parse_crop_metadata(crop_name: str) -> pd.Series:
    basename = os.path.basename(str(crop_name))
    pattern = (
        r"^storm(?P<storm_id>\d+)_"
        r"(?P<datetime>\d{4}-\d{2}-\d{2}T\d{2}-\d{2})_"
    )
    match = re.match(pattern, basename)
    if match is None:
        return pd.Series({"parsed_storm_id": pd.NA, "parsed_datetime": pd.NaT})

    dt = pd.to_datetime(
        match.group("datetime"),
        format="%Y-%m-%dT%H-%M",
        errors="coerce",
    )
    return pd.Series({
        "parsed_storm_id": int(match.group("storm_id")),
        "parsed_datetime": dt,
    })


def choose_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def add_centers_from_midpoint_csv(df: pd.DataFrame, midpoint_csv: str) -> pd.DataFrame:
    if not os.path.exists(midpoint_csv):
        return df

    midpoint_df = pd.read_csv(midpoint_csv, low_memory=False)
    if "crop" not in midpoint_df.columns:
        return df

    midpoint_df = midpoint_df.copy()
    midpoint_df["crop_key"] = midpoint_df["crop"].astype(str).map(os.path.basename)
    midpoint_df = midpoint_df.drop_duplicates(subset=["crop_key"], keep="last")

    out = df.copy()
    if "crop" in out.columns:
        out["crop_key"] = out["crop"].astype(str).map(os.path.basename)
    else:
        return out

    keep_cols = [c for c in ["crop_key", "center_lat", "center_lon"] if c in midpoint_df.columns]
    if len(keep_cols) < 3:
        return out

    out = out.merge(midpoint_df[keep_cols], how="left", on="crop_key")
    return out


def load_pathway_df(pathway_csv: str, midpoint_csv: str) -> pd.DataFrame:
    df = pd.read_csv(pathway_csv, low_memory=False)

    if "crop" in df.columns:
        parsed = df["crop"].apply(parse_crop_metadata)
        df = pd.concat([df, parsed], axis=1)
    else:
        df["parsed_storm_id"] = pd.NA
        df["parsed_datetime"] = pd.NaT

    # Same behavior as plot_traj_ir108_frames.py:
    # keep only rows with storm id and datetime parsed from crop filename.
    df = df.dropna(subset=["parsed_storm_id", "parsed_datetime"])
    df["parsed_storm_id"] = df["parsed_storm_id"].astype(int)

    df = add_centers_from_midpoint_csv(df, midpoint_csv)
    df = df.dropna(subset=["center_lat", "center_lon"])

    df["dt"] = pd.to_datetime(df["parsed_datetime"], errors="coerce")
    df["trajectory_id"] = pd.to_numeric(df["parsed_storm_id"], errors="coerce").astype("Int64")
    df["center_lat"] = pd.to_numeric(df["center_lat"], errors="coerce")
    df["center_lon"] = pd.to_numeric(df["center_lon"], errors="coerce")

    if "label" in df.columns:
        df["label"] = pd.to_numeric(df["label"], errors="coerce")
    else:
        df["label"] = np.nan

    if "pathway" not in df.columns:
        df["pathway"] = "unknown"

    if "crop" not in df.columns:
        df["crop"] = pd.NA

    df = df.dropna(subset=["trajectory_id", "dt", "center_lat", "center_lon"])
    df["trajectory_id"] = df["trajectory_id"].astype(int)
    df = df.sort_values(["trajectory_id", "dt"]).reset_index(drop=True)

    return df


def normalize_eswd_columns(events: pd.DataFrame) -> pd.DataFrame:
    col_map = {c.upper(): c for c in events.columns}

    time_col = col_map.get("TIME_EVENT") or col_map.get("DATETIME") or col_map.get("TIME")
    lat_col = col_map.get("LATITUDE") or col_map.get("LAT")
    lon_col = col_map.get("LONGITUDE") or col_map.get("LON")
    type_col = col_map.get("TYPE_EVENT") or col_map.get("TYPE")

    if time_col is None or lat_col is None or lon_col is None:
        raise ValueError(
            "ESWD CSV requires time/lat/lon columns (e.g., TIME_EVENT, LATITUDE, LONGITUDE)."
        )

    out = pd.DataFrame({
        "TIME_EVENT": pd.to_datetime(events[time_col], errors="coerce", utc=True).dt.tz_convert(None),
        "LATITUDE": pd.to_numeric(events[lat_col], errors="coerce"),
        "LONGITUDE": pd.to_numeric(events[lon_col], errors="coerce"),
        "TYPE_EVENT": events[type_col].astype(str).str.upper().str.strip() if type_col else "UNKNOWN",
    })

    out = out.dropna(subset=["TIME_EVENT", "LATITUDE", "LONGITUDE"]).reset_index(drop=True)
    return out


def load_eswd_events(eswd_csv: str) -> pd.DataFrame:
    if not os.path.exists(eswd_csv):
        print(f"Warning: ESWD CSV not found: {eswd_csv}")
        return pd.DataFrame(columns=["TIME_EVENT", "LATITUDE", "LONGITUDE", "TYPE_EVENT"])

    events = pd.read_csv(eswd_csv, low_memory=False)
    out = normalize_eswd_columns(events)
    print(f"Loaded {len(out)} ESWD events")
    return out


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return r * c


def haversine_to_many_km(lat, lon, lat_arr, lon_arr):
    return haversine_km(lat, lon, np.asarray(lat_arr), np.asarray(lon_arr))


def bearing_deg(lat1, lon1, lat2, lon2):
    lat1r = np.radians(lat1)
    lat2r = np.radians(lat2)
    dlon = np.radians(lon2 - lon1)

    x = np.sin(dlon) * np.cos(lat2r)
    y = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    b = np.degrees(np.arctan2(x, y))
    return (b + 360.0) % 360.0


def circular_mean_deg(angles_deg: np.ndarray) -> float:
    if angles_deg.size == 0:
        return np.nan
    rad = np.radians(angles_deg)
    m_sin = np.mean(np.sin(rad))
    m_cos = np.mean(np.cos(rad))
    return (np.degrees(np.arctan2(m_sin, m_cos)) + 360.0) % 360.0


def circular_std_deg(angles_deg: np.ndarray) -> float:
    if angles_deg.size == 0:
        return np.nan
    rad = np.radians(angles_deg)
    m_sin = np.mean(np.sin(rad))
    m_cos = np.mean(np.cos(rad))
    r = np.sqrt(m_sin ** 2 + m_cos ** 2)
    if r <= 0:
        return np.nan
    return np.degrees(np.sqrt(-2.0 * np.log(r)))


def infer_bbox_from_center(center_lat: float, center_lon: float) -> Tuple[float, float, float, float]:
    return (
        center_lon - DEFAULT_HALF_SIDE_DEG,
        center_lon + DEFAULT_HALF_SIDE_DEG,
        center_lat - DEFAULT_HALF_SIDE_DEG,
        center_lat + DEFAULT_HALF_SIDE_DEG,
    )


def open_dataset_flexible(path: str) -> Optional[xr.Dataset]:
    if not isinstance(path, str) or not os.path.exists(path):
        return None

    engines = ["h5netcdf", "netcdf4", None]
    for engine in engines:
        try:
            if engine is None:
                return xr.open_dataset(path)
            return xr.open_dataset(path, engine=engine)
        except Exception:
            continue
    return None


def extract_ir108_and_bbox(crop_path: str) -> Tuple[float, Optional[Tuple[float, float, float, float]]]:
    ds = open_dataset_flexible(crop_path)
    if ds is None:
        return np.nan, None

    try:
        var_name = None
        for c in VAR_CANDIDATES:
            if c in ds.data_vars:
                var_name = c
                break
        if var_name is None:
            return np.nan, None

        arr = ds[var_name].values
        while arr.ndim > 2:
            arr = arr[0]
        if arr.ndim != 2:
            return np.nan, None

        conv_mask = np.isfinite(arr) & (arr < BT_THRESHOLD_K)
        n_cells = int(label(conv_mask, structure=np.ones((3, 3), dtype=int))[1])

        bbox = None
        if "lon" in ds.coords and "lat" in ds.coords:
            lon = ds["lon"].values
            lat = ds["lat"].values
            if lon.ndim == 1 and lat.ndim == 1:
                bbox = (float(np.min(lon)), float(np.max(lon)), float(np.min(lat)), float(np.max(lat)))
            elif lon.ndim == 2 and lat.ndim == 2:
                bbox = (
                    float(np.nanmin(lon)),
                    float(np.nanmax(lon)),
                    float(np.nanmin(lat)),
                    float(np.nanmax(lat)),
                )

        return float(n_cells), bbox
    finally:
        ds.close()


def is_touching_edge(bbox: Tuple[float, float, float, float], domain_extent: List[float], eps: float = 1e-6) -> bool:
    lon_min, lon_max, lat_min, lat_max = bbox
    d_lon_min, d_lon_max, d_lat_min, d_lat_max = domain_extent
    return (
        (lon_min <= d_lon_min + eps)
        or (lon_max >= d_lon_max - eps)
        or (lat_min <= d_lat_min + eps)
        or (lat_max >= d_lat_max - eps)
    )


def distance_to_domain_edge_km(lat: float, lon: float, domain_extent: List[float]) -> float:
    d_lon_min, d_lon_max, d_lat_min, d_lat_max = domain_extent
    d_w = haversine_km(lat, lon, lat, d_lon_min)
    d_e = haversine_km(lat, lon, lat, d_lon_max)
    d_s = haversine_km(lat, lon, d_lat_min, lon)
    d_n = haversine_km(lat, lon, d_lat_max, lon)
    return float(np.nanmin([d_w, d_e, d_s, d_n]))


def compute_run_lengths(values: np.ndarray) -> List[Tuple[float, int]]:
    if values.size == 0:
        return []

    runs = []
    current = values[0]
    run_len = 1
    for v in values[1:]:
        if v == current:
            run_len += 1
        else:
            runs.append((current, run_len))
            current = v
            run_len = 1
    runs.append((current, run_len))
    return runs


def series_stats(values: np.ndarray, prefix: str) -> Dict[str, float]:
    if values.size == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_min": np.nan,
        }
    return {
        f"{prefix}_mean": float(np.nanmean(values)),
        f"{prefix}_std": float(np.nanstd(values)),
        f"{prefix}_max": float(np.nanmax(values)),
        f"{prefix}_min": float(np.nanmin(values)),
    }


def compute_metrics_per_trajectory(df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    crop_cache: Dict[str, Tuple[float, Optional[Tuple[float, float, float, float]]]] = {}

    for traj_id, g in df.groupby("trajectory_id"):
        g = g.sort_values("dt").copy()
        lat = g["center_lat"].to_numpy(dtype=float)
        lon = g["center_lon"].to_numpy(dtype=float)
        n = len(g)

        if n >= 2:
            disp = haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
            direction = bearing_deg(lat[:-1], lon[:-1], lat[1:], lon[1:])
        else:
            disp = np.array([])
            direction = np.array([])

        row_stats = {}
        row_stats.update(series_stats(disp, "disp_km"))
        row_stats["disp_km_total"] = float(np.nansum(disp)) if disp.size else np.nan

        row_stats["direction_deg_circ_mean"] = circular_mean_deg(direction)
        row_stats["direction_deg_circ_std"] = circular_std_deg(direction)
        row_stats.update(series_stats(direction, "direction_deg"))

        labels = g["label"].to_numpy()
        valid_label_mask = np.isfinite(labels)
        labels = labels[valid_label_mask]
        if labels.size > 0:
            labels = labels.astype(int)
            transitions = int(np.sum(labels[1:] != labels[:-1])) if labels.size > 1 else 0
            transitions_norm = transitions / max(labels.size - 1, 1)
            runs = compute_run_lengths(labels)
            run_lengths = np.array([r[1] for r in runs], dtype=float)
            dominant_frac = float(pd.Series(labels).value_counts(normalize=True).iloc[0])
        else:
            transitions = 0
            transitions_norm = np.nan
            run_lengths = np.array([])
            dominant_frac = np.nan

        row_stats["class_transitions"] = transitions
        row_stats["class_transitions_norm"] = transitions_norm
        row_stats["class_persistence_mean_steps"] = float(np.nanmean(run_lengths)) if run_lengths.size else np.nan
        row_stats["class_persistence_max_steps"] = float(np.nanmax(run_lengths)) if run_lengths.size else np.nan
        row_stats["dominant_class_fraction"] = dominant_frac

        frag_counts = []
        edge_flags = []
        dist_edge = []

        nearby_events_sum = 0
        inside_events_sum = 0
        close_outside_sum = 0

        for _, row in g.iterrows():
            crop_path = row.get("crop", pd.NA)
            crop_key = str(crop_path)

            if crop_key not in crop_cache:
                crop_cache[crop_key] = extract_ir108_and_bbox(crop_key)
            n_cells, bbox = crop_cache[crop_key]

            if np.isfinite(n_cells):
                frag_counts.append(float(n_cells))

            if bbox is None:
                bbox = infer_bbox_from_center(float(row["center_lat"]), float(row["center_lon"]))

            edge_flags.append(float(is_touching_edge(bbox, DOMAIN_EXTENT)))
            dist_edge.append(distance_to_domain_edge_km(float(row["center_lat"]), float(row["center_lon"]), DOMAIN_EXTENT))

            if not events_df.empty:
                t0 = pd.Timestamp(row["dt"])
                t1 = t0 + pd.Timedelta(minutes=15)
                ev = events_df[(events_df["TIME_EVENT"] >= t0) & (events_df["TIME_EVENT"] < t1)]
                if not ev.empty:
                    d = haversine_to_many_km(
                        float(row["center_lat"]),
                        float(row["center_lon"]),
                        ev["LATITUDE"].values,
                        ev["LONGITUDE"].values,
                    )
                    nearby = d <= EVENT_RADIUS_KM
                    if np.any(nearby):
                        ev_near = ev.loc[nearby].copy()
                        lon_min, lon_max, lat_min, lat_max = bbox
                        inside = (
                            (ev_near["LONGITUDE"] >= lon_min)
                            & (ev_near["LONGITUDE"] <= lon_max)
                            & (ev_near["LATITUDE"] >= lat_min)
                            & (ev_near["LATITUDE"] <= lat_max)
                        )
                        nearby_events_sum += int(len(ev_near))
                        inside_events_sum += int(np.sum(inside))
                        close_outside_sum += int(np.sum(~inside))

        frag_counts_arr = np.asarray(frag_counts, dtype=float)
        edge_arr = np.asarray(edge_flags, dtype=float)
        dist_edge_arr = np.asarray(dist_edge, dtype=float)

        row_stats.update(series_stats(frag_counts_arr, "fragmentation_cells"))

        row_stats["edge_fraction"] = float(np.nanmean(edge_arr)) if edge_arr.size else np.nan
        row_stats["edge_count"] = int(np.nansum(edge_arr)) if edge_arr.size else 0
        row_stats["edge_longest_run"] = int(_longest_run_of_ones(edge_arr)) if edge_arr.size else 0
        row_stats["dist_to_edge_km_mean"] = float(np.nanmean(dist_edge_arr)) if dist_edge_arr.size else np.nan
        row_stats["dist_to_edge_km_min"] = float(np.nanmin(dist_edge_arr)) if dist_edge_arr.size else np.nan

        row_stats["events_nearby_150km"] = nearby_events_sum
        row_stats["events_inside_crop"] = inside_events_sum
        row_stats["events_close_outside_crop"] = close_outside_sum
        row_stats["events_close_outside_frac_nearby"] = (
            close_outside_sum / nearby_events_sum if nearby_events_sum > 0 else np.nan
        )

        row_stats["trajectory_id"] = int(traj_id)
        row_stats["n_timestamps"] = int(n)
        row_stats["pathway"] = str(g["pathway"].iloc[0])
        row_stats["start_time"] = g["dt"].iloc[0]
        row_stats["end_time"] = g["dt"].iloc[-1]

        records.append(row_stats)

    return pd.DataFrame(records)


def _longest_run_of_ones(arr: np.ndarray) -> int:
    max_run = 0
    current = 0
    for x in arr:
        if int(x) == 1:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


def summarize_metrics(df_metrics: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in df_metrics.columns if pd.api.types.is_numeric_dtype(df_metrics[c])]

    summary_rows = []
    for c in numeric_cols:
        vals = df_metrics[c].dropna().values
        summary_rows.append(
            {
                "metric": c,
                "count": int(len(vals)),
                "mean": float(np.mean(vals)) if len(vals) else np.nan,
                "std": float(np.std(vals)) if len(vals) else np.nan,
                "median": float(np.median(vals)) if len(vals) else np.nan,
                "min": float(np.min(vals)) if len(vals) else np.nan,
                "max": float(np.max(vals)) if len(vals) else np.nan,
            }
        )

    return pd.DataFrame(summary_rows)


def _plot_hist(ax, values: np.ndarray, title: str, xlabel: str, bins: int = 30):
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        return
    ax.hist(vals, bins=bins, color="#377eb8", edgecolor="black", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(alpha=0.25, linestyle="--")


def plot_distributions(df_metrics: pd.DataFrame, out_dir: str):
    plot_specs = [
        ("disp_km_mean", "Mean displacement", "km / step"),
        ("disp_km_max", "Max displacement", "km / step"),
        ("direction_deg_circ_std", "Direction variability", "deg"),
        ("edge_fraction", "Edge-contact fraction", "fraction"),
        ("fragmentation_cells_mean", "Mean convective cells", "count"),
        ("class_transitions_norm", "Class transitions normalized", "transitions / step"),
        ("class_persistence_mean_steps", "Class persistence mean", "steps"),
        ("events_close_outside_frac_nearby", "Nearby events outside crop", "fraction"),
        ("n_timestamps", "Trajectory length", "timestamps"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    axes = axes.ravel()

    for ax, (col, title, xlabel) in zip(axes, plot_specs):
        if col in df_metrics.columns:
            _plot_hist(ax, df_metrics[col].to_numpy(dtype=float), title, xlabel)
        else:
            ax.text(0.5, 0.5, f"Missing: {col}", ha="center", va="center")
            ax.set_title(title)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "trajectory_metrics_distributions.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    if "pathway" in df_metrics.columns and "disp_km_mean" in df_metrics.columns:
        pathways = sorted(df_metrics["pathway"].astype(str).unique())
        box_data = [
            df_metrics.loc[df_metrics["pathway"].astype(str) == p, "disp_km_mean"].dropna().values
            for p in pathways
        ]
        fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(pathways)), 5))
        ax.boxplot(box_data, labels=pathways, showfliers=False)
        ax.set_ylabel("mean displacement (km / step)")
        ax.set_xlabel("pathway")
        ax.grid(alpha=0.25, linestyle="--")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "disp_by_pathway_boxplot.png"), dpi=220, bbox_inches="tight")
        plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading pathways: {PATHWAY_CSV}")
    df = load_pathway_df(PATHWAY_CSV, MIDPOINT_CSV)
    print(f"Rows after preprocessing: {len(df)}")
    print(f"Trajectories: {df['trajectory_id'].nunique()}")

    events_df = load_eswd_events(ESWD_CSV)

    print("Computing metrics per trajectory...")
    df_metrics = compute_metrics_per_trajectory(df, events_df)

    metrics_csv = os.path.join(OUT_DIR, "trajectory_metrics.csv")
    summary_csv = os.path.join(OUT_DIR, "trajectory_metrics_summary.csv")

    df_metrics.to_csv(metrics_csv, index=False)
    summarize_metrics(df_metrics).to_csv(summary_csv, index=False)

    plot_distributions(df_metrics, OUT_DIR)

    print(f"Saved trajectory metrics: {metrics_csv}")
    print(f"Saved summary stats: {summary_csv}")
    print(f"Saved plots in: {OUT_DIR}")


if __name__ == "__main__":
    main()
