import matplotlib.patches as patches
import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os
from datetime import datetime


COLORS_PER_CLASS = {
    '0': 'darkgray', '1': 'darkslategrey', '2': 'peru', '3': 'orangered',
    '4': 'lightcoral', '5': 'deepskyblue', '6': 'purple', '7': 'lightblue',
    '8': 'green', '9': 'goldenrod', '10': 'magenta', '11': 'dodgerblue',
    '12': 'darkorange', '13': 'olive', '14': 'crimson'
}


def build_trajectory_id(df, time_column="date", lat_column="lat_centre", lon_column="lon_centre"):
    df = df.copy()
    #rename dt column to YYYY-MM-DD_HH:MM format
    df[time_column] = pd.to_datetime(df[time_column]).dt.strftime("%Y-%m-%d")
    df["trajectory_id"] = (
        df[time_column].astype(str) + "_" +
        df[lat_column].round(3).astype(str) + "_" +
        df[lon_column].round(3).astype(str)
    )
    return df

def assign_region(df, lat_division=47):
    df = df.copy()
    df["region"] = np.where(df["lat_centre"] >= lat_division, "NORTH", "SOUTH")
    return df


def select_extreme_cases(
    df,
    intensity_column="max_intensity",
    traj_id_column="trajectory_id",
    n_cases=5,
    agg="max",
):
    """
    Select the N most extreme trajectories based on intensity.

    Parameters
    ----------
    df : pd.DataFrame
        Full events dataframe
    intensity_column : str
        Column used to define extremeness
    traj_id_column : str
        Trajectory identifier
    n_cases : int
        Number of extreme cases to select
    agg : str
        Aggregation over trajectory ("max", "mean", "p95")

    Returns
    -------
    cases : list of pd.DataFrame
        One DataFrame per selected trajectory
    summary : pd.DataFrame
        Trajectory-level summary ranked by intensity
    """

    if agg == "max":
        traj_stat = df.groupby(traj_id_column)[intensity_column].max()
    elif agg == "mean":
        traj_stat = df.groupby(traj_id_column)[intensity_column].mean()
    elif agg == "p95":
        traj_stat = df.groupby(traj_id_column)[intensity_column].quantile(0.95)
    else:
        raise ValueError(f"Unsupported aggregation: {agg}")

    summary = (
        traj_stat
        .rename("extreme_value")
        .reset_index()
        .sort_values("extreme_value", ascending=False)
    )

    top_ids = summary.head(n_cases)[traj_id_column].tolist()

    cases = [
        df[df[traj_id_column] == traj_id].sort_values("datetime")
        for traj_id in top_ids
    ]

    return cases, top_ids




def add_class_frame(ax, label, color_map, lw=4):
    color = color_map.get(label, "black")
    rect = patches.Rectangle(
        (0, 0), 1, 1,
        transform=ax.transAxes,
        fill=False,
        edgecolor=color,
        linewidth=lw
    )
    ax.add_patch(rect)

import os
from datetime import datetime, timedelta


def build_crop_filename(
    path,
    lat,
    lon,
):
    """
    Build crop filename from original path and metadata.

    Parameters
    ----------
    path : str
        Original file path ending with .nc
    lat : float
        Latitude centre
    lon : float
        Longitude centre

    Returns
    -------
    filename : str
        Formatted PNG filename
    """
    dt_str = os.path.basename(path).split("_")[0]

    # Robust parsing (handles seconds if present)
    dt = pd.to_datetime(dt_str)

    # Format components
    date_str = dt.strftime("%Y-%m-%dT%H:%M")
    compact_time = dt.strftime("%Y%m%dT%H%M")

    filename = (
        f"{date_str}_"
        f"{lat:.2f}_"
        f"{lon:.2f}_"
        f"{compact_time}.png"
    )

    return filename



def plot_feature_space_with_trajectory(
    ax,
    df_traj,
    df_train,
    df_centroids,
    xcol="tsne_dim_1",
    ycol="tsne_dim_2",
    label_col="label",
    time_col="datetime",
    class_colors=None,
):
    """
    Plot training feature space, centroids, and a trajectory on top.
    """

    # --- training feature space ---
    ax.scatter(
        df_train[xcol],
        df_train[ycol],
        c=df_train[label_col],
        cmap=mcolors.ListedColormap(
            [class_colors[k] for k in sorted(class_colors)]
        ),
        s=4,
        alpha=0.08,
        rasterized=True,
        zorder=1,
    )

    # --- centroids ---
    ax.scatter(
        df_centroids[xcol],
        df_centroids[ycol],
        c=df_centroids[label_col],
        cmap=mcolors.ListedColormap(
            [class_colors[k] for k in sorted(class_colors)]
        ),
        marker="^",
        s=90,
        edgecolor="white",
        linewidth=0.6,
        zorder=4,
    )

    # --- trajectory prep ---
    df_traj = df_traj.sort_values(time_col).copy()
    df_traj["hour"] = (
        df_traj[time_col].dt.hour +
        df_traj[time_col].dt.minute / 60.0
    )

    norm = mcolors.Normalize(
        vmin=df_traj["hour"].min(),
        vmax=df_traj["hour"].max()
    )

    # --- trajectory line ---
    ax.plot(
        df_traj[xcol],
        df_traj[ycol],
        color="black",
        linewidth=1.5,
        alpha=0.6,
        zorder=3,
    )

    # --- trajectory points ---
    sc = ax.scatter(
        df_traj[xcol],
        df_traj[ycol],
        c=df_traj["hour"],
        cmap=cm.viridis,
        norm=norm,
        s=60,
        edgecolor="black",
        linewidth=0.4,
        zorder=5,
    )

    cbar = plt.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label("Hour (UTC)", fontsize=11)

    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.set_title("Feature-space trajectory", fontsize=12)




def show_crop_table_with_gaps(
    img_dir,
    df_traj,
    time_col="datetime",
    crop_col="crop_filename",
    label_col="label",
    class_colors=None,
    freq="15min",        # "15min" or "1H"
    cols=8,
    max_rows=12,
):
    """
    Show crop images on a fixed time grid.
    Missing timestamps are shown as empty cells (gaps).

    Parameters
    ----------
    img_dir : str
        Directory with crop images
    df_traj : pd.DataFrame
        Trajectory dataframe (single event)
    freq : str
        Time resolution ("15min" or "1H")
    """

    df = df_traj.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True).dt.tz_convert(None)


    # --------------------------------------------------
    # Build full-day time index
    # --------------------------------------------------
    date = df[time_col].dt.normalize().iloc[0]
    start = date
    end = start + pd.Timedelta(days=1) - pd.Timedelta(freq)

    full_times = pd.date_range(start, end, freq=freq)


    # --------------------------------------------------
    # Map timestamps -> (filename, label)
    # --------------------------------------------------
    time_map = {}
    for _, row in df.iterrows():
        ts = row[time_col].floor(freq)
        time_map[ts] = (row[crop_col], row[label_col])

    # --------------------------------------------------
    # Grid layout
    # --------------------------------------------------
    n_slots = len(full_times)
    rows = min(int(np.ceil(n_slots / cols)), max_rows)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * 2.3, rows * 2.3),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).flatten()

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    for ax, ts in zip(axes, full_times):
        ax.axis("off")
        #print(ts)

        if ts in time_map:
            fname, label = time_map[ts]
            img_path = os.path.join(img_dir, fname)
            #print(f"Loading image: {img_path}")

            if os.path.exists(img_path):
                #print(f"Loading image: {img_path}")
                img = imread(img_path)
                ax.imshow(img, cmap="gray")

                # Time label
                ax.set_title(ts.strftime("%H:%M"), fontsize=10)

                # Colored frame
                if class_colors is not None:
                    rect = patches.Rectangle(
                        (0, 0), 1, 1,
                        transform=ax.transAxes,
                        fill=False,
                        linewidth=5,
                        edgecolor=class_colors.get(int(label), "black"),
                    )
                    ax.add_patch(rect)
        else:
            # Gap → keep empty but show time for clarity
            ax.set_title(ts.strftime("%H:%M"), fontsize=10)# color="gray")

    # Hide unused axes
    # for ax in axes[len(full_times):]:
    #     ax.axis("off")

    fig.suptitle(
        f"Crop evolution with temporal gaps – {date}",
        fontsize=16,
        fontweight="bold",
    )
    # make background transparent
    fig.patch.set_alpha(0.0)
    return fig



def plot_feature_trajectory(df_traj):
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.plot(
        df_traj["x"],
        df_traj["y"],
        "-o",
        color="black",
        alpha=0.6
    )

    sc = ax.scatter(
        df_traj["x"],
        df_traj["y"],
        c=df_traj["label"],
        cmap="tab20",
        s=60,
        zorder=3
    )

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("Feature-space trajectory")

    return fig
