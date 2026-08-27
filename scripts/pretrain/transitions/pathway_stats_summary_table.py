#!/usr/bin/env python3
"""Print pathway-level statistics sorted by number of storms.

The script reads the merged pathway CSV used by the transition analysis and
an external crop percentile CSV to recover CTH/COT medians. It computes a
compact summary table with:
- n_storms
- CTH50 median
- COT50 median
- CC mean +/- std
- CTH10+ mean +/- std
- COT30+ mean +/- std

Percent-like quantities are reported in percent units.
"""

import os

import pandas as pd

RUN_NAME = "dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc"
BASE_DIR = f"/data1/fig/{RUN_NAME}/epoch_800/test_traj"
OUT_DIR = f"{BASE_DIR}/pathway_analysis"
INPUT_CSV = os.path.join(OUT_DIR, "df_pathways_merged_no_dominance.csv")
PCT_CSV = os.path.join(BASE_DIR, "merged_crops_stats_cot_percentiles.csv")
OUTPUT_CSV = os.path.join(OUT_DIR, "pathway_stats_summary_table.csv")


def format_mean_std(mean_value, std_value, decimals=2):
    if pd.isna(std_value):
        std_value = 0.0
    return f"{mean_value:.{decimals}f} +/- {std_value:.{decimals}f}"


def load_crop_percentiles(pct_csv):
    if not os.path.exists(pct_csv):
        print(f"Crop percentiles file not found: {pct_csv}")
        return None, None

    try:
        df_crop_pct = pd.read_csv(pct_csv, low_memory=False)
        print(f"Loaded crop percentiles from: {pct_csv}")
        df_crop_cth_pct = (
            df_crop_pct[df_crop_pct["var"] == "cth"]
            .rename(columns={"25": "cth25_pct", "50": "cth50_pct", "75": "cth75_pct"})
            [["crop", "cth25_pct", "cth50_pct", "cth75_pct"]]
        )
        df_crop_cot_pct = (
            df_crop_pct[df_crop_pct["var"] == "cot_percentiles"]
            .rename(columns={"25": "cot25_pct", "50": "cot50_pct", "75": "cot75_pct"})
            [["crop", "cot25_pct", "cot50_pct", "cot75_pct"]]
        )
        return df_crop_cth_pct, df_crop_cot_pct
    except Exception as exc:
        print(f"Failed to load crop percentiles: {exc}")
        return None, None


def build_pathway_table(df):
    required_cols = [
        "pathway_id",
        "pathway",
        "storm_id",
        "cma",
        "cth_very_high",
        "cot_thick",
        "crop",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df_crop_cth_pct, df_crop_cot_pct = load_crop_percentiles(PCT_CSV)
    if df_crop_cth_pct is None or df_crop_cot_pct is None:
        raise FileNotFoundError("Could not load crop percentile CSV with CTH/COT medians.")

    df_pw = (
        df.merge(df_crop_cth_pct, on="crop", how="left")
          .merge(df_crop_cot_pct, on="crop", how="left")
    )

    missing_pct = [col for col in ["cth50_pct", "cot50_pct"] if col not in df_pw.columns]
    if missing_pct:
        raise KeyError(f"Missing merged percentile columns: {missing_pct}")

    storm_stats = (
        df_pw.groupby(["pathway_id", "pathway", "storm_id"], dropna=False).agg(
            cth50_median=("cth50_pct", "median"),
            cot50_median=("cot50_pct", "median"),
            cc_mean=("cma", "mean"),
            cc_std=("cma", "std"),
            cth10_mean=("cth_very_high", "mean"),
            cth10_std=("cth_very_high", "std"),
            cot30_mean=("cot_thick", "mean"),
            cot30_std=("cot_thick", "std"),
        )
        .reset_index()
    )

    table = (
        storm_stats.groupby(["pathway_id", "pathway"], dropna=False).agg(
            n_storms=("storm_id", "nunique"),
            cth50_median=("cth50_median", "mean"),
            cth50_std=("cth50_median", "std"),
            cot50_median=("cot50_median", "mean"),
            cot50_std=("cot50_median", "std"),
            cc_mean=("cc_mean", "mean"),
            cc_std=("cc_mean", "std"),
            cth10_mean=("cth10_mean", "mean"),
            cth10_std=("cth10_mean", "std"),
            cot30_mean=("cot30_mean", "mean"),
            cot30_std=("cot30_mean", "std"),
        )
        .reset_index()
    )

    table = table.sort_values(["n_storms", "pathway_id"], ascending=[False, True]).reset_index(drop=True)
    table["rank"] = range(1, len(table) + 1)

    table["cth50_median"] = table["cth50_median"] / 1000.0
    table["cth50_std"] = table["cth50_std"] / 1000.0
    table["cot50_median"] = table["cot50_median"]

    table["cc_mean"] = table["cc_mean"] * 100.0
    table["cc_std"] = table["cc_std"] * 100.0
    table["cth10_mean"] = table["cth10_mean"] * 100.0
    table["cth10_std"] = table["cth10_std"] * 100.0
    table["cot30_mean"] = table["cot30_mean"] * 100.0
    table["cot30_std"] = table["cot30_std"] * 100.0

    display = table[[
        "rank",
        "pathway_id",
        "pathway",
        "n_storms",
        "cth50_median",
        "cth50_std",
        "cot50_median",
        "cot50_std",
        "cc_mean",
        "cc_std",
        "cth10_mean",
        "cth10_std",
        "cot30_mean",
        "cot30_std",
    ]].copy()

    display["CTH50"] = display.apply(lambda row: format_mean_std(row["cth50_median"], row["cth50_std"]), axis=1)
    display["COT50"] = display.apply(lambda row: format_mean_std(row["cot50_median"], row["cot50_std"]), axis=1)
    display["CC (%)"] = display.apply(lambda row: format_mean_std(row["cc_mean"], row["cc_std"]), axis=1)
    display["CTH10+ (%)"] = display.apply(lambda row: format_mean_std(row["cth10_mean"], row["cth10_std"]), axis=1)
    display["COT30+ (%)"] = display.apply(lambda row: format_mean_std(row["cot30_mean"], row["cot30_std"]), axis=1)

    display = display[[
        "rank",
        "pathway_id",
        "pathway",
        "n_storms",
        "CTH50",
        "COT50",
        "CC (%)",
        "CTH10+ (%)",
        "COT30+ (%)",
    ]]

    display = display.rename(columns={
        "n_storms": "N storms",
    })

    return table, display


def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    table, display = build_pathway_table(df)

    print("\n" + "=" * 140)
    print("PATHWAY STATS TABLE (sorted by number of storms)")
    print("=" * 140)
    print(display.to_string(index=False))
    print("=" * 140 + "\n")

    os.makedirs(OUT_DIR, exist_ok=True)
    table.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved pathway summary table to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
