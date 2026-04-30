import os
from pathlib import Path
import argparse

import pandas as pd
import xarray as xr


RUN_NAME = "dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc"
OUTPUT_PATH = f"/data1/fig/{RUN_NAME}/epoch_800/test_traj"
PATHWAY_CSV = os.path.join(OUTPUT_PATH, "pathway_analysis/df_pathways_merged_no_dominance.csv")
DEFAULT_OUT_CSV = os.path.join(OUTPUT_PATH, "pathway_analysis/crop_midpoints_from_nc.csv")
DEFAULT_TRAJ_CSV = (
    "/data1/crops/test_case_essl_14-15-16-18-19-20-22-23-24_100x100_ir108_cma_traj/"
    "storm_trajectories_after_merge.csv"
)


def extract_midpoint_from_coord(coord):
    """Extract midpoint value from 1D or 2D coordinate arrays."""
    if coord.ndim == 1:
        return float(coord.values[len(coord) // 2])
    if coord.ndim == 2:
        ny, nx = coord.shape
        return float(coord.values[ny // 2, nx // 2])
    raise ValueError(f"Unsupported coordinate ndim={coord.ndim}")


def get_lat_lon_midpoint_from_nc(nc_path):
    """Open crop NetCDF and return midpoint latitude and longitude."""
    lat_candidates = ["lat", "latitude", "Latitude"]
    lon_candidates = ["lon", "longitude", "Longitude"]

    with xr.open_dataset(nc_path) as ds:
        lat_name = next((name for name in lat_candidates if name in ds.coords), None)
        lon_name = next((name for name in lon_candidates if name in ds.coords), None)

        if lat_name is None:
            lat_name = next((name for name in lat_candidates if name in ds.variables), None)
        if lon_name is None:
            lon_name = next((name for name in lon_candidates if name in ds.variables), None)

        if lat_name is None or lon_name is None:
            raise KeyError(
                f"Could not find lat/lon coordinates in {nc_path}. "
                f"Available coords: {list(ds.coords)}"
            )

        mid_lat = extract_midpoint_from_coord(ds[lat_name])
        mid_lon = extract_midpoint_from_coord(ds[lon_name])
    return mid_lat, mid_lon


def resolve_crop_path(crop_name, search_roots, resolved_cache):
    crop_key = os.path.basename(str(crop_name))
    if crop_key in resolved_cache:
        return resolved_cache[crop_key]

    path_obj = Path(str(crop_name))
    if path_obj.exists():
        resolved_cache[crop_key] = path_obj
        return path_obj

    for root in search_roots:
        candidate = root / crop_key
        if candidate.exists():
            resolved_cache[crop_key] = candidate
            return candidate

    for root in search_roots:
        if not root.exists():
            continue
        matches = list(root.rglob(crop_key))
        if matches:
            resolved_cache[crop_key] = matches[0]
            return matches[0]

    resolved_cache[crop_key] = None
    return None


def build_midpoint_csv(pathway_csv, traj_csv, out_csv, round_decimals=3):
    df = pd.read_csv(pathway_csv, low_memory=False)
    unique_crops = sorted({os.path.basename(str(c)) for c in df["crop"].dropna().tolist()})

    search_roots = [Path(pathway_csv).parent, Path(traj_csv).parent, Path("/data1/crops")]
    resolved_cache = {}

    records = []
    total = len(unique_crops)
    print(f"Building midpoint lookup for {total} unique crops")

    for idx, crop_name in enumerate(unique_crops, start=1):
        resolved_path = resolve_crop_path(crop_name, search_roots, resolved_cache)

        status = "ok"
        center_lat = pd.NA
        center_lon = pd.NA

        if resolved_path is None:
            status = "file_not_found"
        else:
            try:
                center_lat, center_lon = get_lat_lon_midpoint_from_nc(str(resolved_path))
                center_lat = round(center_lat, round_decimals)
                center_lon = round(center_lon, round_decimals)
            except Exception as exc:
                status = f"read_error:{type(exc).__name__}"

        records.append(
            {
                "crop": crop_name,
                "resolved_path": str(resolved_path) if resolved_path is not None else pd.NA,
                "center_lat": center_lat,
                "center_lon": center_lon,
                "status": status,
            }
        )

        if idx % 500 == 0 or idx == total:
            print(f"Processed {idx}/{total} crops")

    out_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    n_ok = (out_df["status"] == "ok").sum()
    n_missing = (out_df["status"] == "file_not_found").sum()
    n_errors = len(out_df) - n_ok - n_missing

    print(
        f"Saved midpoint CSV to {out_csv} | total={len(out_df)}, ok={n_ok}, "
        f"missing={n_missing}, read_errors={n_errors}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build crop midpoint CSV (center lat/lon from crop NetCDF files)."
    )
    parser.add_argument("--pathway-csv", default=PATHWAY_CSV, help="Path to df_pathways_merged_no_dominance.csv")
    parser.add_argument("--traj-csv", default=DEFAULT_TRAJ_CSV, help="Path to storm trajectories CSV (used as search root)")
    parser.add_argument("--out-csv", default=DEFAULT_OUT_CSV, help="Output CSV with crop midpoint lookup")
    parser.add_argument("--round-decimals", type=int, default=3, help="Rounding precision for center lat/lon")
    args = parser.parse_args()

    build_midpoint_csv(
        pathway_csv=args.pathway_csv,
        traj_csv=args.traj_csv,
        out_csv=args.out_csv,
        round_decimals=args.round_decimals,
    )
