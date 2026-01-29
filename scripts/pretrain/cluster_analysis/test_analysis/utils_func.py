import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.colors import ListedColormap
import re
import os


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



def stratifiy_by_latitude(df, lat_col: str, LAT_DIVISION: float):
    """
    Splits dataframe into NORTH and SOUTH based on latitude division.
    Returns two dataframes: df_north, df_south
    """
    df_north = df[df[lat_col] >= LAT_DIVISION].copy()
    df_south = df[df[lat_col] < LAT_DIVISION].copy()
    print(f"NORTH samples: {len(df_north)}, SOUTH samples: {len(df_south)}")
    
    return df_north, df_south



# ================================================
# GROUP BUILDER
# ================================================

def build_event_groups(
    df,
    percentile: int,
    plot: bool = False,
    intensity_col: str = "max_intensity",
    vector_col: str = "vector_type",
    out_dir: str = None,
    region: str = None,
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
    plot : bool, optional
        If True, plot intensity histograms for rain and hail.
    intensity_col : str
        Column containing intensity values.
    vector_col : str
        Column identifying event type.
    bins : int
        Number of bins for histograms.
    ax : array-like of matplotlib axes, optional
        Two axes [ax_rain, ax_hail]. Created automatically if None.

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
    # 3. Optional plotting
    # -------------------------------------------------
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(4, 1), sharey=True)

        # --- Rain histogram
        if len(rain_vals) > 0:
            bin_width = 20 #mm
            bins = np.arange(0, 420 + bin_width, bin_width)
            ax[0].hist(rain_vals, bins=bins)
            ax[0].axvline(rain_perc, linestyle="--", color="red")
            #ax[0].set_title("Precipitation amount (mm)")
            ax[0].set_xlabel("Precipitation \n amount (mm)", fontsize=10)
            ax[0].tick_params(axis='y', labelsize=10)
            ax[0].tick_params(axis='x', labelsize=10)
            #ax[0].yaxis.set_major_locator(plt.MaxNLocator(5))
            ax[0].set_yscale('log')
            ax[0].xaxis.set_major_locator(plt.MaxNLocator(4))

        # --- Hail histogram
        if len(hail_vals) > 0:
            bin_width = 1 #cm
            bins = np.arange(0, 20 + bin_width, bin_width)
            ax[1].hist(hail_vals, bins=bins)
            ax[1].axvline(hail_perc, linestyle="--", color="red")
            #ax[1].set_title("Hail max diameter (cm)")
            ax[1].set_xlabel("Hail max \n diameter (cm)", fontsize=10)
            ax[1].tick_params(axis='y', labelsize=10)
            ax[1].tick_params(axis='x', labelsize=10)
            #set number of y tickes to 5
            #ax[1].yaxis.set_major_locator(plt.MaxNLocator(5))
            ax[1].set_yscale('log')
            ax[1].xaxis.set_major_locator(plt.MaxNLocator(4))

        if region=='North':
            #remove x ticks and labels 
            ax[0].set_xticks([])
            ax[1].set_xticks([])
            ax[0].set_xlabel("")
            ax[1].set_xlabel("")
        
        ax[0].set_ylabel("Count", fontsize=10)
        plt.suptitle(
            f"Intensity distributions - {region if region is not None else 'ALL'}",
            fontsize=10, fontweight="bold", y=1.03
        )

        # Save plot if output directory provided
        if out_dir is not None:
            out_path = f"{out_dir}/intensity_distributions_{region}.png"
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"Saved intensity distribution plot: {out_path}")

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


def build_storm_event_groups(
    df,
    storm_id_col: str = "merged_storm_id",
    cluster_type_col: str = "cluster_event_type",
):
    """
    Build boolean masks for storm-level event groups using cluster_event_type.

    Storm classification:
      - RAIN  → only PRECIP in cluster_event_type
      - HAIL  → only HAIL in cluster_event_type
      - MIXED → both PRECIP and HAIL present

    Parameters
    ----------
    df : pd.DataFrame
        Trajectory dataframe.
    storm_id_col : str
        Column identifying storms (default: merged_storm_id).
    cluster_type_col : str


    Returns
    -------
    dict
        Dictionary of boolean masks indexed like df.
    """

    df = df.copy()

    # -------------------------------------------------
    # 1. Infer storm type from cluster_event_type
    # -------------------------------------------------
    def infer_storm_class(types):
        types = set(types.dropna().str.upper())

        has_precip = any("PRECIP" in t for t in types)
        has_hail = any("HAIL" in t for t in types)
        has_mixed = any("MIXED" in t for t in types)

        if has_precip and has_hail:
            return "MIXED"
        elif has_precip:
            return "RAIN"
        elif has_hail:
            return "HAIL"
        elif has_mixed:
            return "MIXED"
        else:
            return "UNDEFINED"

    storm_class = (
        df.groupby(storm_id_col)[cluster_type_col]
        .apply(infer_storm_class)
    )

    df["storm_class"] = df[storm_id_col].map(storm_class)

    # -------------------------------------------------
    # 2. Base masks
    # -------------------------------------------------
    masks = {
        "ALL": df[storm_id_col].notna(),
        "RAIN": df["storm_class"] == "RAIN",
        "HAIL": df["storm_class"] == "HAIL",
        "MIXED": df["storm_class"] == "MIXED",
        "UNDEFINED": df["storm_class"] == "UNDEFINED",
    }

    return masks, df




def count_storms_and_crops_from_filenames(
    base_dir="/data1/crops/test_case_essl_14-15-16-18-19-20-22-23-24_100x100_ir108_cma_traj/nc/1",
    lat_split=47.0,
):
    """
    Count storms and crops directly from crop filenames.

    Returns
    -------
    dict
        Dictionary with multiple summary DataFrames.
    """

    pattern = re.compile(
        r"storm(?P<storm_id>\d+)_"
        r"(?P<date>\d{4}-\d{2}-\d{2})T(?P<time>\d{2}-\d{2})_"
        r"lat(?P<lat>-?\d+\.?\d*)_"
        r"lon(?P<lon>-?\d+\.?\d*)_"
        r"(?P<storm_type>PRECIP|HAIL|MIXED)_"
        r"(?P<crop_type>observed|interpolated|extrapolated)\.nc",
        re.IGNORECASE,
    )

    records = []

    for fname in os.listdir(base_dir):
        if not fname.endswith(".nc"):
            continue

        match = pattern.match(fname)
        if not match:
            print(f"⚠️ Skipping unmatched filename: {fname}")
            continue

        d = match.groupdict()
        d["storm_id"] = int(d["storm_id"])
        d["lat"] = float(d["lat"])
        d["lon"] = float(d["lon"])
        d["storm_type"] = d["storm_type"].upper()
        d["crop_type"] = d["crop_type"].lower()
        d["region"] = "NORTH" if d["lat"] >= lat_split else "SOUTH"

        records.append(d)

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError("No valid crop files found.")

    # -------------------------------------------------
    # 1. Number of storms per storm type (by region)
    # -------------------------------------------------
    storms_per_type = (
        df[["storm_id", "storm_type", "region"]]
        .drop_duplicates()
        .groupby(["region", "storm_type"])
        .size()
        .rename("n_storms")
        .reset_index()
    )

    # -------------------------------------------------
    # 2. Total number of crops per storm ID
    # -------------------------------------------------
    crops_per_storm = (
        df.groupby(["region", "storm_id", "storm_type"])
        .size()
        .rename("n_crops")
        .reset_index()
    )

    # -------------------------------------------------
    # 3. Number of crops per storm type
    # -------------------------------------------------
    crops_per_storm_type = (
        df.groupby(["region", "storm_type"])
        .size()
        .rename("n_crops")
        .reset_index()
    )

    # -------------------------------------------------
    # 4. Number of crops per crop type per storm type
    # -------------------------------------------------
    crops_per_crop_type = (
        df.groupby(["region", "storm_type", "crop_type"])
        .size()
        .rename("n_crops")
        .reset_index()
    )

    # Sum over regions for overall stats
    overall_storms_per_type = (
        storms_per_type
        .groupby("storm_type")["n_storms"]
        .sum()
        .reset_index()
    )

    overall_crops_per_storm_type = (
        crops_per_storm_type
        .groupby("storm_type")["n_crops"]
        .sum()
        .reset_index()
    )

    overall_crops_per_crop_type = (
        crops_per_crop_type
        .groupby(["storm_type", "crop_type"])["n_crops"]
        .sum()
        .reset_index()
    )

    return {
        "storms_per_type": storms_per_type,
        "crops_per_storm": crops_per_storm,
        "crops_per_storm_type": crops_per_storm_type,
        "crops_per_crop_type": crops_per_crop_type,
        "overall_storms_per_type": overall_storms_per_type,
        "overall_crops_per_storm_type": overall_crops_per_storm_type,
        "overall_crops_per_crop_type": overall_crops_per_crop_type,
        "raw": df,
    }





def compute_temporal_popularity(df, time_unit="hour"):
    """
    Compute relative frequency of events over time.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'datetime'
    time_unit : str
        'year', 'month', or 'hour'

    Returns
    -------
    pd.Series
        Relative frequency indexed by time unit
    """
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])

    if time_unit == "year":
        t = df["datetime"].dt.year
    elif time_unit == "month":
        t = df["datetime"].dt.month
    elif time_unit == "hour":
        t = df["datetime"].dt.hour
    else:
        raise ValueError("time_unit must be 'year', 'month', or 'hour'")

    counts = t.value_counts().sort_index()
    return counts / counts.sum()


def plot_temporal_barplot(ax, freq, time_unit):
    """
    Plot temporal popularity as barplot.

    Parameters
    ----------
    ax : matplotlib axis
    freq : pd.Series
        Relative frequency indexed by time unit
    time_unit : str
    """
    ax.bar(freq.index, freq.values)

    if time_unit == "hour":
        ax.set_xticks(range(0, 24, 3))
        ax.set_xlabel("Hour (UTC)")
    elif time_unit == "month":
        ax.set_xticks(range(1, 13))
        ax.set_xlabel("Month")
    elif time_unit == "year":
        ax.set_xlabel("Year")

    ax.set_ylabel("Relative frequency")
    ax.set_ylim(0, freq.max() * 1.1)


def compute_class_kde_grid(
    df,
    xcol="2d_dim_1",
    ycol="2d_dim_2",
    label_col="label",
    grid_size=300,
    bandwidth=0.2,
    min_samples=20,
):
    """
    Compute KDE dominance grid per class.

    Returns
    -------
    dominant_class : 2D np.ndarray
        Index of dominant class per grid cell
    labels : list
        Sorted class labels corresponding to indices
    extent : tuple
        (xmin, xmax, ymin, ymax) for imshow
    """

    x = df[xcol].values
    y = df[ycol].values

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, grid_size),
        np.linspace(ymin, ymax, grid_size),
    )

    grid = np.vstack([xx.ravel(), yy.ravel()])

    labels = sorted(df[label_col].unique())
    density_stack = []

    for lbl in labels:
        subset = df[df[label_col] == lbl]

        if len(subset) < min_samples:
            density_stack.append(np.zeros(grid.shape[1]))
            continue

        kde = gaussian_kde(
            np.vstack([subset[xcol], subset[ycol]]),
            bw_method=bandwidth,
        )

        density_stack.append(kde(grid))

    density_stack = np.vstack(density_stack)
    dominant_class = np.argmax(density_stack, axis=0)
    extent = (xmin, xmax, ymin, ymax)

    return dominant_class.reshape(xx.shape), extent


def plot_feature_space_dots(
    ax,
    df,
    xcol="2d_dim_1",
    ycol="2d_dim_2",
    label_col="label",
    class_colors=None,
    s=4,
    alpha=0.05,
    rasterized=True,
):
    """
    Plot dot-based feature space colored by class label.

    Parameters
    ----------
    ax : matplotlib axis
    df : DataFrame (TRAIN samples recommended)
    class_colors : dict {str(label): color}
    """

    if class_colors is None:
        raise ValueError("class_colors dictionary must be provided")

   
    #add a column color to df based on the label_col and class_colors dict
    colors = []
    #loop over dataframe rows
    for _, row in df.iterrows():
        label = row[label_col]
        color = class_colors.get(label, {}).get("color", "gray")
        colors.append(color)


    ax.scatter(
        df[xcol],
        df[ycol],
        c=colors,
        s=s,
        alpha=alpha,
        rasterized=rasterized,
        linewidths=0,
    )



def plot_class_kde_background(
    ax,
    dominant_class,
    labels,
    extent,
    class_colors,
    alpha=0.35,
):
    """
    Plot precomputed KDE class dominance background.
    """

    cmap = ListedColormap(
        [class_colors.get(str(lbl), "gray") for lbl in labels]
    )

    ax.imshow(
        dominant_class,
        origin="lower",
        extent=extent,
        cmap=cmap,
        alpha=alpha,
        interpolation="bilinear",
        aspect="auto",
    )







def print_sample_counts(df, groups):
    """
    Print sample counts per group, considering unique events (datetime + lat_centre + lon_centre).
    """
    print("Samples per group:")
    df["date"] = pd.to_datetime(df["datetime"]).dt.date
    for name, mask in groups.items():
        df_g = df[mask]

        # unique events per day+lat+lon
        n_events = df_g.groupby(["date", "lat_centre", "lon_centre", "vector_type"]).ngroup().nunique()

        print(f"{name:<12}: {n_events}")


def plot_event_trajectories(ax, df_event, cmap, norm, alpha=0.5, linewidth=0.5):
    """
    Plot one trajectory per event-day.
    Each trajectory is a sequence of samples within
    [start_time, end_time] for the same day.
    """
    df_event = df_event.copy()
    df_event["datetime"] = pd.to_datetime(df_event["datetime"])
    df_event["start_time"] = pd.to_datetime(df_event["start_time"])
    df_event["end_time"] = pd.to_datetime(df_event["end_time"])

    # group by day
    for day, gday in df_event.groupby(df_event["datetime"].dt.floor("D")):

        # filter inside event window
        g = gday[
            (gday["datetime"] >= gday["start_time"].iloc[0]) &
            (gday["datetime"] <= gday["end_time"].iloc[0])
        ].sort_values("datetime")

        if len(g) < 2:
            continue  # nothing to plot

        hours = g["datetime"].dt.hour

        ax.plot(
            g["2d_dim_1"],
            g["2d_dim_2"],
            color=cmap(norm(hours.mean())),
            linewidth=linewidth,
            alpha=alpha,
        )

# ----------------------------------------------------
# Function: density contour for a group
# ----------------------------------------------------

def plot_density_contours(
    ax,
    df,
    xcol="2d_dim_1",
    ycol="2d_dim_2",
    percentiles=(50, 90),
    bandwidth=0.25,
    color="black",
    linewidths=(1.2, 2.0),
    alpha=1.0,
    linestyles_list=None,
    grid_size=300,
):
    """
    Plot KDE density contours using percentile-based levels.

    Parameters
    ----------
    ax : matplotlib axis
    df : DataFrame with feature space columns
    percentiles : tuple of percentiles (e.g. 50, 90)
    bandwidth : KDE bandwidth
    color : contour color
    linewidths : line widths for each percentile
    """

    if len(df) < 20:
        return

    x = df[xcol].values
    y = df[ycol].values

    # KDE
    kde = gaussian_kde(
        np.vstack([x, y]),
        bw_method=bandwidth,
    )

    # grid
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, grid_size),
        np.linspace(ymin, ymax, grid_size),
    )

    grid = np.vstack([xx.ravel(), yy.ravel()])
    density = kde(grid).reshape(xx.shape)

    # percentile-based contour levels
    levels = np.percentile(density, percentiles)

    #define different linestyle for each level
    #linesyles_list = ['solid', 'dashed','dotted', 'dashdot']
    if linestyles_list is not None and len(levels) <= len(linestyles_list):
        linestyles = linestyles_list[:len(levels)]
    else:
        linestyles = ['solid'] * len(levels)  # default to solid if too many levels

    ax.contour(
        xx,
        yy,
        density,
        levels=levels,
        colors=color,
        linewidths=linewidths,
        alpha=alpha,
        linestyles=linestyles
    )



def plot_mean_location(ax, df_events, x_dim, y_dim, name, color, marker):
    """
    Plot mean location of events on feature space.
    """
    ax.scatter(
            df_events[x_dim].mean(),
            df_events[y_dim].mean(),
            s=300,
            marker=marker,
            c=color,
            edgecolor="black",
            linewidth=1.2,
            zorder=5,
            label=f"{name} mean"
        )
    


def plot_distance_boxplot(
    ax,
    df,
    label_col="label",
    distance_col="distance",
    class_colors=None,
    showfliers=False,
    alpha=0.7,
):
    """
    Plot boxplots of cosine distance grouped by class label.

    Parameters
    ----------
    ax : matplotlib axis
        Axis where the boxplot is drawn.
    df : pd.DataFrame
        DataFrame containing class labels and cosine distance.
    label_col : str
        Column name for class labels.
    distance_col : str
        Column name for cosine distance.
    class_colors : dict
        Mapping {str(label): color}.
    showfliers : bool
        Whether to show outliers.
    alpha : float
        Transparency for box faces.
    """


    box_data = []
    box_positions = []
    box_colors = []

    for i, (lbl, color_info) in enumerate(class_colors, start=1):
        vals = pd.to_numeric(
            df[df[label_col] == lbl][distance_col],
            errors="coerce"
        ).dropna()

        if len(vals) <= 10:
            continue  # skip classes with too few samples

        box_data.append(vals.values)
        box_positions.append(i)
        box_colors.append(
            color_info["color"]
        )
    #print(box_positions)
    #print(box_colors)
    #print(box_data)

    bp = ax.boxplot(
        box_data,
        positions=box_positions,
        widths=0.6,
        patch_artist=True,
        showfliers=showfliers,
    )

    # --- styling ---
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(alpha)

    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    ax.set_xticks(box_positions)
    ax.grid(axis="y", alpha=0.3)
