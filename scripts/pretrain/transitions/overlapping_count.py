import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# ==================================================
# CONFIGURATION
# ==================================================
RUN_NAME = "dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc"
OUTPUT_PATH = f"/data1/fig/{RUN_NAME}/epoch_800/test_traj"
CSV_PATH = os.path.join(OUTPUT_PATH, f"features_train_test_{RUN_NAME}_2nd_labels.csv")


def get_crop_bounds(lat, lon, half_size_deg, domain):
    """
    Returns (lon_min, lon_max, lat_min, lat_max) for a crop,
    shifted if necessary to stay within the domain.
    """
    lon_min = lon - half_size_deg
    lon_max = lon + half_size_deg
    lat_min = lat - half_size_deg
    lat_max = lat + half_size_deg

    # Shift longitude
    if lon_min < domain["lon_min"]:
        lon_max += domain["lon_min"] - lon_min
        lon_min = domain["lon_min"]
    if lon_max > domain["lon_max"]:
        lon_min -= lon_max - domain["lon_max"]
        lon_max = domain["lon_max"]

    # Shift latitude
    if lat_min < domain["lat_min"]:
        lat_max += domain["lat_min"] - lat_min
        lat_min = domain["lat_min"]
    if lat_max > domain["lat_max"]:
        lat_min -= lat_max - domain["lat_max"]
        lat_max = domain["lat_max"]

    return lon_min, lon_max, lat_min, lat_max

def compute_overlap_percentage(box1, box2):
    """
    box = (lon_min, lon_max, lat_min, lat_max)
    Returns overlap percentage relative to crop area.
    """
    lon_overlap = max(0, min(box1[1], box2[1]) - max(box1[0], box2[0]))
    lat_overlap = max(0, min(box1[3], box2[3]) - max(box1[2], box2[2]))

    overlap_area = lon_overlap * lat_overlap
    crop_area = (box1[1] - box1[0]) * (box1[3] - box1[2])

    return 100.0 * overlap_area / crop_area if crop_area > 0 else 0.0

def ensure_datetimes(df):
    for col in ["datetime"]:#, "start_time", "end_time"]:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    
    return df

def filter_by_event_window(df):
    return df[
        (df["datetime"] >= df["start_time"]) &
        (df["datetime"] <= df["end_time"])
    ]


def count_days_and_timestamps_per_type(df, exclude_train=True):
    """
    Count total timestamps and unique days per event type,
    considering only datetimes within each event's start/end time.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe with columns:
        ['datetime', 'start_time', 'end_time', 'vector_type']
    exclude_train : bool, optional
        Whether to exclude vector_type == 'TRAIN'

    Returns
    -------
    summary : pandas.DataFrame
        Columns:
        ['vector_type', 'n_timestamps', 'n_days']
    """

    df = df.copy()

    # --- Ensure datetime types ---
    df = ensure_datetimes(df)

    # --- Filter by event time window ---
    #df = filter_by_event_window(df)

    # --- Optionally remove TRAIN ---
    if exclude_train:
        df = df[df["vector_type"] != "TRAIN"]

    # --- Extract day ---
    df["day"] = df["datetime"].dt.floor("D")

    # --- Aggregate ---
    summary = (
        df.groupby("vector_type")
        .agg(
            n_timestamps=("datetime", "count"),
            n_days=("day", "nunique"),
        )
        .reset_index()
    )

    return summary



def compute_precip_hail_overlap(
    df,
    mode="datetime",  # "datetime" or "day"
    crop_pixels=100,
    resolution_deg=0.04,
    domain=dict(lon_min=5, lon_max=16, lat_min=42, lat_max=51.5),
    bins=np.linspace(0, 100, 21),
    plot=True,
    output_path=None,
    threshold=50.0
):
    """
    Computes overlap distribution between PRECIP and HAIL crops.

    Returns
    -------
    overlaps : list
        List of overlap percentages
    """

    df = df.copy()

    # --- Ensure datetimes ---
    df = ensure_datetimes(df)

    # --- Filter by event window ---
    #df = filter_by_event_window(df)

    # --- Remove TRAIN ---
    df = df[df["vector_type"] != "TRAIN"]

    # --- Define grouping ---
    if mode == "day":
        df["group"] = df["datetime"].dt.floor("D")
    elif mode == "datetime":
        df["group"] = df["datetime"]
    else:
        raise ValueError("mode must be 'datetime' or 'day'")

    half_size = crop_pixels * resolution_deg / 2.0
    overlaps = []

    # --- Loop over groups ---
    for _, g in df.groupby("group"):

        precip = g[g["vector_type"] == "PRECIP"]
        hail   = g[g["vector_type"] == "HAIL"]

        if precip.empty or hail.empty:
            continue

        # ============================
        # DAY MODE → ONE overlap only
        # ============================
        if mode == "day":

            # representative crop (space is constant within a day)
            p = precip.iloc[0]
            h = hail.iloc[0]

            box_p = get_crop_bounds(
                p["lat_centre"], p["lon_centre"], half_size, domain
            )
            box_h = get_crop_bounds(
                h["lat_centre"], h["lon_centre"], half_size, domain
            )

            overlaps.append(
                compute_overlap_percentage(box_p, box_h)
            )

        # ============================
        # DATETIME MODE → pairwise
        # ============================
        else:
            for _, p in precip.iterrows():
                box_p = get_crop_bounds(
                    p["lat_centre"], p["lon_centre"], half_size, domain
                )

                for _, h in hail.iterrows():
                    box_h = get_crop_bounds(
                        h["lat_centre"], h["lon_centre"], half_size, domain
                    )

                    overlaps.append(
                        compute_overlap_percentage(box_p, box_h)
                    )

    if plot:
        plotting_overlaps(overlaps, bins=bins, mode=mode, output_path=output_path)

    n_total = len(overlaps)
    n_above = np.sum(np.array(overlaps) >= threshold)
    fraction_above = n_above / n_total if n_total > 0 else np.nan

    
    return overlaps, n_above, n_total, fraction_above



def plotting_overlaps(overlaps, bins, mode="datetime", output_path=None):
    """
    # --- Plot histogram ---
    """    
    plt.figure(figsize=(4, 2))
    plt.hist(overlaps, bins=bins, edgecolor="black")
    plt.xlabel("Crop overlap (%)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title(
        "PRECIP–HAIL overlap distribution\n"
        f"({'per datetime' if mode=='datetime' else 'per day'})", fontsize=12, fontweight="bold"
    )
    plt.grid(alpha=0.3)
    #set y ticks labels and fontsize
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    #I want only 5 ticks on y axis
    plt.yticks(np.linspace(0, plt.yticks()[0].max(), 5))
    plt.savefig(
        f"{output_path}/overlap_distribution_{mode}.png",
        dpi=300,
        bbox_inches="tight"
    )

    


if __name__ == "__main__":
    #take only these columns: datetime, lat_centre, lon_centre, vector_type, start_time, end_time
    THRESHOLD = 50.0  # percentage
    df = pd.read_csv(CSV_PATH, low_memory=False)#, usecols=["datetime", "lat_centre", "lon_centre", "vector_type", "start_time", "end_time"])
    print(df.head())
    summury = count_days_and_timestamps_per_type(df)
    print("Summary of timestamps and days per event type:")
    print(summury)
    
    
    overlaps_dt, n_above_dt, n_total_dt, fraction_above_dt = compute_precip_hail_overlap(df, mode="datetime", output_path=OUTPUT_PATH, threshold=THRESHOLD)
    overlaps_day, n_above_day, n_total_day, fraction_above_day = compute_precip_hail_overlap(df, mode="day", output_path=OUTPUT_PATH, threshold=THRESHOLD)
    print(f"Computed {len(overlaps_dt)} overlaps (datetime mode) and {n_above_dt} above {THRESHOLD} threshold")
    print(f"Computed {len(overlaps_day)} overlaps (day mode) and {n_above_day} above {THRESHOLD} threshold")