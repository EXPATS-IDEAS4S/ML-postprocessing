import os
import pandas as pd
import numpy as np
import xarray as xr


def filter_rows_in_event_window(df):
    """
    Keep only rows where:
        start_time <= datetime <= end_time
    Assumes all datetimes are in UTC.
    """

    # Work on a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Convert to datetime if not already
    for col in ["datetime", "start_time", "end_time"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Ensure all are tz-aware (UTC) to avoid comparison issues
    for col in ["datetime", "start_time", "end_time"]:
        if df[col].dt.tz is None:
            df[col] = df[col].dt.tz_localize("UTC")

    # Drop rows with missing timestamps
    df = df.dropna(subset=["datetime", "start_time", "end_time"])

    # Keep only rows where datetime is inside the event window
    mask_in_window = (df["datetime"] >= df["start_time"]) & (df["datetime"] <= df["end_time"])
    df_filtered = df[mask_in_window]

    return df_filtered.reset_index(drop=True)




def load_data(path, filter_event_window=True):
    """
    Load data from CSV and filter rows within event windows.
    """
    df = pd.read_csv(path, low_memory=False)

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    
    if filter_event_window:
        df = filter_rows_in_event_window(df)
    return df



def stratifiy_by_latitude(df, lat_column: str, LAT_DIVISION: float):
    """
    Splits dataframe into NORTH and SOUTH based on latitude division.
    Returns two dataframes: df_north, df_south
    """
    df_north = df[df[lat_column] >= LAT_DIVISION].copy()
    df_south = df[df[lat_column] < LAT_DIVISION].copy()
    print(f"NORTH samples: {len(df_north)}, SOUTH samples: {len(df_south)}")
    
    return df_north, df_south


def split_by_region(df, lat_column: str, LAT_DIVISION: float = 47.0):
    north, south = stratifiy_by_latitude(df, lat_column, LAT_DIVISION)
    return {"NORTH": north, "SOUTH": south}



def extract_bt_from_path(path, var):
    if not isinstance(path, str):
        return None
    try:
        with xr.open_dataset(path, engine='hdf5netcdf') as ds:
            return ds[var].values.ravel()
    except Exception:
        return None
    

def compute_bt_stats(bt_values):
    if bt_values is None or len(bt_values) == 0:
        return {"p50": np.nan, "p01": np.nan, "iqr": np.nan}
    return {
        "p50": np.nanpercentile(bt_values, 50),
        "p01": np.nanpercentile(bt_values, 1),
        "iqr": np.nanpercentile(bt_values, 75) - np.nanpercentile(bt_values, 25),
    }


def update_df_with_bt_stats(df, path_col, var):
    """
    Update dataframe with BT statistics.
    """
    stats = df[path_col].apply(lambda x: compute_bt_stats(extract_bt_from_path(x, var)))
    stats_df = pd.DataFrame(stats.tolist())
    df = pd.concat([df.reset_index(drop=True), stats_df], axis=1)
    return df



def build_event_groups(
    df,
    percentile: int,
    intensity_col: str = "max_intensity",
    vector_col: str = "vector_type",
):
    """
    Create boolean masks for event groups and optionally plot intensity distributions.

    Groups:
      - ALL
      - RAIN_ALL
      - RAIN_<percentile>th
      - HAIL_ALL
      - HAIL_<percentile>th

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    percentile : int
        Percentile used for intensity thresholding.
    intensity_col : str
        Column containing intensity values.
    vector_col : str
        Column identifying event type.

    Returns
    -------
    dict
        Dictionary of boolean masks.
    """

    # -------------------------------------------------
    # 1. Identify rain vs hail
    # -------------------------------------------------
    rain_mask = df[vector_col].str.contains("PRECIP", case=False, na=False)
    hail_mask = df[vector_col].str.contains("HAIL", case=False, na=False)

    overlap = (rain_mask & hail_mask).sum()
    if overlap > 0:
        print(f"Warning: {overlap} overlapping rain/hail events")

    # -------------------------------------------------
    # 2. Compute percentiles
    # -------------------------------------------------
    rain_vals = df.loc[rain_mask, intensity_col].dropna()
    hail_vals = df.loc[hail_mask, intensity_col].dropna()

    rain_perc = (
        np.percentile(rain_vals, percentile)
        if len(rain_vals) > 0 else np.nan
    )
    hail_perc = (
        np.percentile(hail_vals, percentile)
        if len(hail_vals) > 0 else np.nan
    )

    # -------------------------------------------------
    # 4. Return masks
    # -------------------------------------------------
    return {
        "ALL": rain_mask | hail_mask,
        "RAIN_ALL": rain_mask,
        f"RAIN_{percentile}th": rain_mask & (df[intensity_col] >= rain_perc),
        "HAIL_ALL": hail_mask,
        f"HAIL_{percentile}th": hail_mask & (df[intensity_col] >= hail_perc),
    }




def count_events_per_hour(
    df,
    time_col="TIME_EVENT",
):
    """
    Count number of events per hour of day.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing event timestamps.
    time_col : str
        Column name of the event timestamp.

    Returns
    -------
    hourly_counts : pd.Series
        Number of events per hour (index = 0–23).
    """

    df = df.copy()

    # --- Ensure datetime ---
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")

    # --- Drop invalid timestamps ---
    df = df.dropna(subset=[time_col])

    if df.empty:
        return pd.Series(0, index=range(24))

    # --- Extract hour ---
    df["hour"] = df[time_col].dt.hour

    # --- Count events per hour ---
    hourly_counts = (
        df["hour"]
        .value_counts()
        .sort_index()
        .reindex(range(24), fill_value=0)
    )

    return hourly_counts






def compute_variable(df, variable=None, labels=None, variable_stat=None):
    if variable == "label_freq":
        df = df.copy()
        if variable_stat:
            df_freq = compute_diurnal_stats(df,labels,hours=range(24), variable=variable_stat)
        else:
            df_freq = compute_diurnal_frequency(df, labels, variable_stat=variable_stat, percentile=np.percentile)
        #print(df_freq)
        return df_freq
    elif variable == 'hourly_occurrences':
        df = df.copy()
        hourly_counts = count_events_per_hour(df)
        return hourly_counts
    elif variable == 'hourly_intensities':
        df = df.copy()
        hour_intensity = compute_hourly_intensity_for_boxplot(
            df,
            intensity_col=variable_stat, 
            time_col="datetime"
        )
        return hour_intensity
    else:
        raise ValueError(f"Unknown variable: {variable}")
        



def compute_diurnal_frequency(df, labels, hours=range(24)):
    """
    Compute relative frequency per hour per label.
    Ensures all labels and all hours are present.
    """
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True, errors="coerce")
    df["hour"] = df["datetime"].dt.hour
    print(df.columns.tolist())
    n_samples = len(df)

    counts = (
        df.groupby(["hour", "label"])
        .size()
        .rename("count")
        .reset_index()
    )

    # Normalize over all counts
    if n_samples > 0:
        counts["relative_freq"] = counts["count"] / n_samples
    else:  
        counts["relative_freq"] = 0.0
    
    #counts["relative_freq"] = counts.groupby("hour")["count"].transform(lambda x: x / x.sum() if x.sum() > 0 else 0.0)


    # Pivot
    freq = counts.pivot(
        index="label",
        columns="hour",
        values="relative_freq"
    )

    # Force full grid and fill Nan with zeros
    freq = freq.reindex(
        index=labels,
        columns=hours,
        fill_value=0.0
    )

    return freq.fillna(0.0)


def compute_diurnal_stats(
    df,
    labels,
    hours=range(24),
    variable: str = None,
    time_col: str = "datetime",
    label_col: str = "label",
):
    """
    Compute diurnal statistics (mean, std, optional percentile)
    per hour per label.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    labels : list
        Ordered list of labels to enforce in output.
    hours : iterable
        Hours to enforce (default: 0–23).
    variable : str
        Column name of variable to aggregate.
    percentile : float, optional
        Percentile to compute (0–100).
    time_col : str
        Datetime column name.
    label_col : str
        Label column name.

    Returns
    -------
    dict
        Dictionary with keys:
          - "mean"
          - "std"
          - "percentile" (if requested)
        Each value is a DataFrame [label × hour].
    """

    df = df.copy()

    # --- Time handling
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df["hour"] = df[time_col].dt.hour

    # --- Base groupby
    gb = df.groupby(["hour", label_col])[variable]

    # --- Mean & std
    stats = gb.agg(["mean", "std"]).reset_index()

    # --- Pivot helper
    def pivot_stat(col):
        out = stats.pivot(
            index=label_col,
            columns="hour",
            values=col,
        )
        return out.reindex(
            index=labels,
            columns=hours,
        )

    out = {
        "mean": pivot_stat("mean"),
        "std": pivot_stat("std"),
    }

    return out



def compute_hourly_intensity_for_boxplot(
    df,
    intensity_col="mean_intensity", 
    time_col="datetime"
):
    """
    Compute hourly intensity distributions for boxplot visualization.

    Parameters
    ----------
    df_events : pd.DataFrame
        Events dataframe
    df_selected_clusters : pd.DataFrame or None
        Optional dataframe containing selected cluster IDs
    intensity_col : str
        Column containing intensity values
        (e.g. "mean_intensity" or "max_intensity")
    cluster_col : str
        Column name of cluster id
    time_col : str
        Datetime column for event time

    Returns
    -------
    data_by_hour : list of pd.Series
        List of length 24; each element contains intensity values for that hour
        (empty Series if no data)
    """

    df = df.copy()

    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")

    # --- Keep valid intensity values only ---
    df = df.dropna(subset=[intensity_col])

    if df.empty:
        return [pd.Series(dtype=float) for _ in range(24)]

    # --- Extract hour ---
    df["hour"] = df[time_col].dt.hour

    # --- Build boxplot-ready structure ---
    data_by_hour = [
        df.loc[df["hour"] == h, intensity_col]
        for h in range(24)
    ]

    return data_by_hour



def compute_transitions_and_persistence(df):
    """Compute transition probabilities and persistence per label."""
    labels = df['label'].tolist()

    # Compute blocks of consecutive identical labels
    blocks = []
    prev_label = None
    block_len = 0
    for lbl in labels:
        if lbl == prev_label:
            block_len += 1
        else:
            if prev_label is not None:
                blocks.append((prev_label, block_len))
            block_len = 1
            prev_label = lbl
    blocks.append((prev_label, block_len))

    blocks = pd.DataFrame(blocks, columns=['label', 'length'])
    #convert lenght in duration in minutes
    blocks['duration_min'] = blocks['length'] * 15  # 15-min timesteps → minutes
    #blocks['duration_h'] = blocks['length'] * 0.25  # 15-min timesteps → hours

    # Persistence = mean duration per label (round to integer) 
    persistence = blocks.groupby('label')['duration_min'].mean().round().astype(int).to_dict()

    # Compute transitions between blocks
    from_labels = blocks['label'][:-1].values
    to_labels = blocks['label'][1:].values
    trans_df = pd.DataFrame({'from': from_labels, 'to': to_labels})
    transition_counts = pd.crosstab(trans_df['from'], trans_df['to'])
    transition_probs = transition_counts.div(transition_counts.sum(axis=1), axis=0).fillna(0)

    # Insert persistence on diagonal
    all_labels = sorted(set(df['label'].unique()))
    transition_matrix = pd.DataFrame(0, index=all_labels, columns=all_labels, dtype=float)

    for i in all_labels:
        for j in all_labels:
            if i == j:
                transition_matrix.loc[i, j] = persistence.get(i, 0)
            else:
                transition_matrix.loc[i, j] = transition_probs.loc[i, j] if i in transition_probs.index and j in transition_probs.columns else 0

    return transition_matrix, persistence

  