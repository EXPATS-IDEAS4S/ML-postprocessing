import os
import re
from pathlib import Path
import argparse

import pandas as pd
import xarray as xr


DEFAULT_RUN_NAME = "dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc"
DEFAULT_OUTPUT_PATH = f"/data1/fig/{DEFAULT_RUN_NAME}/epoch_800/test_traj"
DEFAULT_CSV_PATH = (
    "/data1/crops/test_case_essl_14-15-16-18-19-20-22-23-24_100x100_ir108_cma_traj/"
    "storm_trajectories_after_merge.csv"
)
DEFAULT_PATHWAY_CSV = os.path.join(
    DEFAULT_OUTPUT_PATH,
    "pathway_analysis/df_pathways_merged_no_dominance.csv",
)


def parse_crop_metadata(crop_name):
    """Extract storm_id, datetime, lat, lon from crop filename."""
    basename = os.path.basename(str(crop_name))
    pattern = (
        r"^storm(?P<storm_id>\d+)_"
        r"(?P<datetime>\d{4}-\d{2}-\d{2}T\d{2}-\d{2})_"
        r"lat(?P<lat>-?\d+(?:\.\d+)?)_"
        r"lon(?P<lon>-?\d+(?:\.\d+)?)_"
    )
    match = re.match(pattern, basename)
    if match is None:
        return pd.Series(
            {
                "parsed_storm_id": pd.NA,
                "parsed_datetime": pd.NaT,
                "parsed_lat": pd.NA,
                "parsed_lon": pd.NA,
            }
        )

    dt = pd.to_datetime(match.group("datetime"), format="%Y-%m-%dT%H-%M", errors="coerce")
    return pd.Series(
        {
            "parsed_storm_id": int(match.group("storm_id")),
            "parsed_datetime": dt,
            "parsed_lat": float(match.group("lat")),
            "parsed_lon": float(match.group("lon")),
        }
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


def validate_parsed_lat_lon_against_nc_midpoint(
    df,
    csv_path,
    pathway_csv,
    output_path,
    tolerance_deg=0.02,
):
    """Validate parsed filename lat/lon against NetCDF midpoint lat/lon."""
    search_roots = [Path(csv_path).parent, Path(pathway_csv).parent, Path("/data1/crops")]
    resolved_cache = {}

    def resolve_crop_path(crop_name):
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

    unique_crops = df[["crop", "parsed_lat", "parsed_lon"]].drop_duplicates(subset=["crop"]).copy()
    records = []

    n_unique = len(unique_crops)
    for idx, (_, row) in enumerate(unique_crops.iterrows(), start=1):
        crop_name = row["crop"]
        parsed_lat = float(row["parsed_lat"])
        parsed_lon = float(row["parsed_lon"])
        resolved_path = resolve_crop_path(crop_name)

        status = "ok"
        nc_mid_lat = pd.NA
        nc_mid_lon = pd.NA
        lat_diff = pd.NA
        lon_diff = pd.NA
        mismatch = pd.NA

        if resolved_path is None:
            status = "file_not_found"
            print(
                f"[{idx}/{n_unique}] {crop_name} | "
                f"filename(lat,lon)=({parsed_lat:.4f},{parsed_lon:.4f}) | "
                "nc_mid(lat,lon)=(NA,NA) | status=file_not_found",
                flush=True,
            )
        else:
            try:
                nc_mid_lat, nc_mid_lon = get_lat_lon_midpoint_from_nc(str(resolved_path))
                lat_diff = abs(parsed_lat - nc_mid_lat)
                lon_diff = abs(parsed_lon - nc_mid_lon)
                mismatch = (lat_diff > tolerance_deg) or (lon_diff > tolerance_deg)
                if mismatch:
                    status = "mismatch"
                print(
                    f"[{idx}/{n_unique}] {crop_name} | "
                    f"filename(lat,lon)=({parsed_lat:.4f},{parsed_lon:.4f}) | "
                    f"nc_mid(lat,lon)=({nc_mid_lat:.4f},{nc_mid_lon:.4f}) | "
                    f"|dlat|={lat_diff:.5f}, |dlon|={lon_diff:.5f} | status={status}",
                    flush=True,
                )
            except Exception as exc:
                status = f"read_error:{type(exc).__name__}"
                print(
                    f"[{idx}/{n_unique}] {crop_name} | "
                    f"filename(lat,lon)=({parsed_lat:.4f},{parsed_lon:.4f}) | "
                    f"nc_mid(lat,lon)=(NA,NA) | status={status}",
                    flush=True,
                )

        records.append(
            {
                "crop": crop_name,
                "resolved_path": str(resolved_path) if resolved_path is not None else pd.NA,
                "parsed_lat": parsed_lat,
                "parsed_lon": parsed_lon,
                "nc_mid_lat": nc_mid_lat,
                "nc_mid_lon": nc_mid_lon,
                "abs_diff_lat": lat_diff,
                "abs_diff_lon": lon_diff,
                "mismatch": mismatch,
                "status": status,
            }
        )

    report_df = pd.DataFrame(records)
    report_path = os.path.join(output_path, "crop_latlon_midpoint_validation.csv")
    report_df.to_csv(report_path, index=False)

    n_total = len(report_df)
    n_ok = (report_df["status"] == "ok").sum()
    n_mismatch = (report_df["status"] == "mismatch").sum()
    n_missing = (report_df["status"] == "file_not_found").sum()
    n_errors = n_total - n_ok - n_mismatch - n_missing

    print(
        f"Lat/lon midpoint validation complete: total={n_total}, ok={n_ok}, "
        f"mismatch={n_mismatch}, missing={n_missing}, read_errors={n_errors}."
    )
    print(f"Validation report saved to {report_path}")

    return report_df


def main():
    parser = argparse.ArgumentParser(
        description="Check filename lat/lon vs midpoint lat/lon of corresponding NetCDF crop files."
    )
    parser.add_argument("--pathway-csv", default=DEFAULT_PATHWAY_CSV, help="Pathway analysis CSV path.")
    parser.add_argument("--traj-csv", default=DEFAULT_CSV_PATH, help="Trajectory CSV path.")
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH, help="Output directory for report CSV.")
    parser.add_argument(
        "--tolerance-deg",
        type=float,
        default=0.02,
        help="Absolute tolerance in degrees for midpoint mismatch check.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    pathway_df = pd.read_csv(args.pathway_csv)

    parsed_cols = pathway_df["crop"].apply(parse_crop_metadata)
    pathway_df = pd.concat([pathway_df, parsed_cols], axis=1)
    pathway_df = pathway_df.dropna(
        subset=["parsed_storm_id", "parsed_datetime", "parsed_lat", "parsed_lon"]
    )

    print(
        f"Loaded {len(pathway_df)} parsed points from "
        f"{pathway_df['parsed_storm_id'].nunique()} unique trajectories"
    )

    validate_parsed_lat_lon_against_nc_midpoint(
        pathway_df,
        csv_path=args.traj_csv,
        pathway_csv=args.pathway_csv,
        output_path=args.output_path,
        tolerance_deg=args.tolerance_deg,
    )


if __name__ == "__main__":
    main()
