import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import gaussian_kde
from matplotlib.colors import ListedColormap
import seaborn as sns



def plot_bt_boxplot_by_event(
    df,
    value_col,
    event_col,
    event_order,
    ax,
):
    """
    Plot boxplots of a BT statistic grouped by event type.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame already filtered to a single BT statistic.
    value_col : str
        Column containing the BT values to plot.
    event_col : str
        Column containing event group labels.
    event_order : list of str
        Event group names in the desired x-axis order.
    ax : matplotlib.axes.Axes
        Axis to draw the boxplot on.
    ylabel : str, optional
        Y-axis label.
    title : str, optional
        Plot title.
    """

    if value_col not in df.columns:
        raise ValueError(f"'{value_col}' not found in DataFrame")

    if event_col not in df.columns:
        raise ValueError(f"'{event_col}' not found in DataFrame")

    # Collect values per event
    data = [
        df.loc[df[event_col] == event, value_col].dropna().values
        for event in event_order
    ]

    ax.boxplot(
        data,
        labels=event_order,
        showfliers=False
    )

    return ax




def plot_orography_map(
    path_nc,
    ax=None,
    var_name="orography",
    extent=None,
    cmap="Greys",
    levels=30,
    alpha=0.6
):
    """
    Plot orography from a NetCDF file on a Cartopy map.

    Parameters
    ----------
    path_nc : str
        Path to NetCDF file containing orography data.
    ax : cartopy.mpl.geoaxes.GeoAxes, optional
        Axis to plot on. If None, a new figure is created.
    extent : list, optional
        [lon_min, lon_max, lat_min, lat_max].
    cmap : str or Colormap
        Colormap for orography.
    levels : int
        Number of contour levels.
    alpha : float
        Transparency of the orography layer.
    """

    ds = xr.open_dataset(path_nc)

    if var_name not in ds:
        raise KeyError(f"Variable '{var_name}' not found in dataset")

    oro = ds[var_name]

    # Infer coordinate names
    lat_name = "lat" if "lat" in oro.coords else "latitude"
    lon_name = "lon" if "lon" in oro.coords else "longitude"

    lats = oro[lat_name]
    lons = oro[lon_name]

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(7, 6),
            subplot_kw={"projection": ccrs.PlateCarree()}
        )

    cf = ax.contourf(
        lons,
        lats,
        oro,
        levels=levels,
        cmap=cmap,
        alpha=alpha,
        transform=ccrs.PlateCarree()
    )

    ax.coastlines(resolution="50m", color="black", linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="black")
    #ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor="white", alpha=alpha)

    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    return ax, cf



def plot_event_density_map(
    df,
    lat_col="latitude",
    lon_col="longitude",
    ax=None,
    extent=None,              # [lon_min, lon_max, lat_min, lat_max]
    n_grid=200,
    cmap="viridis",
    levels=10,
    scatter=True,
    scatter_kwargs=None,
):
    """
    Plot event locations and filled KDE density on a Cartopy map.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    lat_col, lon_col : str
        Latitude and longitude column names.
    event_col : str or None
        Column containing event group names.
    event_name : str or None
        Event group to plot (used if event_col is provided).
    region : str or None
        "North" or "South" for latitude filtering.
    lat_division : float
        Latitude threshold.
    ax : cartopy.mpl.geoaxes.GeoAxes
        Axis to draw on.
    projection : cartopy CRS
        Map projection.
    extent : list
        [lon_min, lon_max, lat_min, lat_max]
    n_grid : int
        Grid resolution for KDE.
    cmap : str or Colormap
        Colormap for density.
    levels : int
        Number of filled contour levels.
    scatter : bool
        Overlay scatter points.
    scatter_kwargs : dict
        Custom scatter keyword arguments.
    title : str
        Axis title.
    """

    plot_df = df.copy()

    if plot_df.empty:
        ax.set_title("No data")
        return ax

    lats = plot_df[lat_col].values
    lons = plot_df[lon_col].values

    # Grid for KDE
    if extent is None:
        lon_min, lon_max = lons.min() - 1, lons.max() + 1
        lat_min, lat_max = lats.min() - 1, lats.max() + 1
    else:
        lon_min, lon_max, lat_min, lat_max = extent

    lon_grid, lat_grid = np.meshgrid(
        np.linspace(lon_min, lon_max, n_grid),
        np.linspace(lat_min, lat_max, n_grid)
    )

    kde = gaussian_kde(np.vstack([lons, lats]))
    density = kde(np.vstack([
        lon_grid.ravel(),
        lat_grid.ravel()
    ])).reshape(lon_grid.shape)

    # Filled density contours
    cf = ax.contourf(
        lon_grid,
        lat_grid,
        density,
        levels=levels,
        cmap=cmap,
        alpha=0.7,
        transform=ccrs.PlateCarree()
    )

    # Optional scatter points
    if scatter:
        scatter_kwargs = scatter_kwargs or {}
        ax.scatter(
            lons,
            lats,
            s=scatter_kwargs.get("s", 8),
            c=scatter_kwargs.get("c", "k"),
            alpha=scatter_kwargs.get("alpha", 0.4),
            transform=ccrs.PlateCarree(),
            zorder=3
        )

    # Map features
    #transparent land
    ax.add_feature(cfeature.LAND, facecolor="white", alpha=0)
    ax.add_feature(cfeature.OCEAN, facecolor="white", alpha=0)

    if extent:
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    return ax, cf



def plot_label_freq(ax, matrix, vmax=None, cmap=None):
    hm = sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap or "viridis",
        cbar=False,
        vmax=vmax,
        vmin=0,
    )
    #remove ticks and labels
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    return hm.collections[0]


def plot_distribution(ax, data, title=None):
    ax.hist(data, bins=40, density=False)
    ax.set_ylabel("Count")
    if title:
        ax.set_title(title, fontweight="bold")



def plot_transition_matrix(matrix, persistence, event, out_path):
    """Plot transition matrix with persistence values on diagonal."""
    off_diag = matrix.copy()
    np.fill_diagonal(off_diag.values, np.nan)

    plt.subplots(figsize=(7, 6))
    #2 decimal places only for off-diagonal
    ax =sns.heatmap(off_diag, annot=True, fmt=".2f", cmap="Blues",
                cbar_kws={'label': 'Transition Probability'}, vmax=0.4)
    #increase font size of colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label("Transition Probability", fontsize=16, fontweight="bold")
    # Overlay persistence (diagonal) with another color scale
    for i, label in enumerate(matrix.index):
        val = matrix.loc[label, label]
        #integer values in red bold font
        ax.text(i + 0.5, i + 0.5, f"{val.astype(int)}", ha='center', va='center',
                color='red', fontsize=14, fontweight='bold')

    plt.title(f"Class Transition Matrix ({event})", fontsize=16, fontweight='bold')
    plt.xlabel("To Class", fontsize=15)
    plt.ylabel("From Class", fontsize=15)
    plt.xticks(rotation=45, fontsize=15)
    plt.yticks(rotation=0, fontsize=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved transition matrix: {out_path}")

    # Save persistence values
    with open(out_path.replace(".png", "_persistence.txt"), "w") as f:
        f.write("Average class persistence (hours):\n")
        for k, v in persistence.items():
            f.write(f"Label {k}: {v:.2f} h\n")
    print(f"🕒 Saved persistence summary: {out_path.replace('.png', '_persistence.txt')}")