import matplotlib.patches as patches
import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os
from datetime import datetime
import glob
import cmcrameri.cm as cmc

import sys
sys.path.append("/home/Daniele/codes/VISSL_postprocessing/")
from scripts.pretrain.cluster_analysis.test_analysis.utils_func import  plot_feature_space_dots


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
        c=df_train["color"],
        #cmap=class_colors,
        s=4,
        alpha=0.08,
        rasterized=True,
        zorder=1,
    )

    # # --- centroids ---
    # ax.scatter(
    #     df_centroids[xcol],
    #     df_centroids[ycol],
    #     c=df_centroids[label_col],
    #     cmap=mcolors.ListedColormap(class_colors),
    #     marker="^",
    #     s=90,
    #     edgecolor="black",
    #     linewidth=0.6,
    #     zorder=4,
    # )

    # --- trajectory prep ---
    df_traj[time_col] = pd.to_datetime(df_traj[time_col], utc=True).dt.tz_convert(None)
    df_traj = df_traj.sort_values(time_col).copy()

    t0 = df_traj[time_col].iloc[len(df_traj) // 2]  # midpoint reference
    df_traj["t_aligned"] = (
        (df_traj[time_col] - t0).dt.total_seconds() / 3600.0
    )

    tvals = df_traj["t_aligned"].values
    vmax = np.nanmax(np.abs(tvals))
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

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
        c=tvals,
        cmap=cmc.lisbon,
        norm=norm,
        s=150,
        edgecolor="black",
        linewidth=0.4,
        zorder=5,
    )

    cbar = plt.colorbar(sc, ax=ax, pad=0.01, fraction=0.03)
    cbar.set_label("Aligned time (hours)", fontsize=14)
    #inclrease fontsize of colorbar ticks
    cbar.ax.tick_params(labelsize=14)


    #ax.set_xlabel("t-SNE dim 1")
    #ax.set_ylabel("t-SNE dim 2")
    #ax.set_title("Feature-space trajectory", fontsize=12)
    #remove axis frame, ticks labels sttructure
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)




def show_crop_table(
    img_dir,
    df_traj,
    storm_id = None,
    time_col="datetime",
    path_col="path",     # NC filename or full path
    label_col="label",
    class_colors=None,
    cols=8,
    max_rows=12,
):
    """
    Plot all crop images found for a trajectory (no temporal gaps).
    Images are matched by NC filename prefix and shown in grayscale.
    """

    df = df_traj.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True).dt.tz_convert(None)
    df = df.sort_values(time_col)

    images = []

    # --------------------------------------------------
    # Find PNGs matching NC filenames
    # --------------------------------------------------
    for _, row in df.iterrows():
        nc_base = os.path.splitext(os.path.basename(row[path_col]))[0]
        matches = glob.glob(os.path.join(img_dir, f"{nc_base}*.png"))

        if len(matches) > 0:
            images.append({
                "path": matches[0],
                "time": row[time_col],
                "label": row[label_col],
                'color': class_colors[int(row[label_col])]
            })


    if len(images) == 0:
        raise ValueError("No matching PNG images found.")

    # --------------------------------------------------
    # Grid layout
    # --------------------------------------------------
    n_imgs = len(images)
    rows = min(int(np.ceil(n_imgs / cols)), max_rows)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * 2.3, rows * 2.3),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).flatten()

    # --------------------------------------------------
    # Plot images
    # --------------------------------------------------
    for ax, item in zip(axes, images):
        ax.axis("off")

        img = imread(item["path"])
        ax.imshow(img, cmap="gray")

        ax.set_title(
            item["time"].strftime("%H:%M"),
            fontsize=10
        )

        if class_colors is not None:
            rect = patches.Rectangle(
                (0, 0), 1, 1,
                transform=ax.transAxes,
                fill=False,
                linewidth=5,
                edgecolor=item['color'],#"black"),
            )
            ax.add_patch(rect)

    # hide unused axes
    for ax in axes[len(images):]:
        ax.axis("off")

    #date = images[0]["time"].date()
    fig.suptitle(
        f"Crop evolution – {storm_id}",
        fontsize=16,
        fontweight="bold",
    )

    fig.patch.set_alpha(0.0)
    return fig








def show_crop_table_with_gaps(
    img_dir,
    df_traj,
    time_col="datetime",
    path_col="path",        # NC filename or full path
    label_col="label",
    class_colors=None,
    freq="15min",
    cols=8,
    max_rows=12,
):
    """
    Show crop images on a fixed time grid.
    Images are found by matching the NC filename prefix to PNG files.
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
    # Map timestamps -> (png_path, label)
    # --------------------------------------------------
    time_map = {}

    for _, row in df.iterrows():
        ts = row[time_col].floor(freq)

        # extract base filename (without extension)
        nc_base = os.path.splitext(os.path.basename(row[path_col]))[0]

        # find matching PNG
        matches = glob.glob(os.path.join(img_dir, f"{nc_base}*.png"))

        if len(matches) > 0:
            time_map[ts] = (matches[0], row[label_col])

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

        if ts in time_map:
            img_path, label = time_map[ts]

            if os.path.exists(img_path):
                img = imread(img_path)
                ax.imshow(img)
                ax.set_title(ts.strftime("%H:%M"), fontsize=10)

                # colored frame
                if class_colors is not None:
                    rect = patches.Rectangle(
                        (0, 0), 1, 1,
                        transform=ax.transAxes,
                        fill=False,
                        linewidth=5,
                        edgecolor=class_colors[label]#, "black"),
                    )
                    ax.add_patch(rect)
        else:
            # gap
            ax.set_title(ts.strftime("%H:%M"), fontsize=10)

    fig.suptitle(
        f"Crop evolution with temporal gaps – {date.date()}",
        fontsize=16,
        fontweight="bold",
    )

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
