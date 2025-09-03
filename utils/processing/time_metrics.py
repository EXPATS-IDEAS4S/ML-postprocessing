"""
Cloud Evolution Metrics Extraction
==================================

This script provides functions to compute time-dependent metrics describing
the spatiotemporal development of clouds from geostationary satellite data
(e.g., MSG/SEVIRI at ~5 km resolution).

The metrics focus on the temporal evolution of cloud cover, vertical
development, and horizontal motion, and are designed to complement
self-supervised learning experiments where time invariances and dynamics
are important.

Usage
-----
- Functions can be called independently for analysis.
- They are designed to be integrated in a `process_row`-like pipeline
  that extracts time-resolved variables from satellite data crops.
- Input assumptions:
    - Time series are aligned with regular intervals.
    - Cloud mask and CTH are given as 3D arrays [time, lat, lon].
    - Time is provided as a list/array of np.datetime64 objects.

Notes
-----
- Pixel-to-km scaling should be adjusted depending on projection
  and latitude of the crop.
- Cloud detection / CMA filtering should be applied before metrics
  to avoid contamination from clear-sky or uncertain pixels.
"""

import numpy as np
from scipy.ndimage import center_of_mass


def change_in_cloud_cover(cloud_mask, times, pixel_area_km2=25.0):
    """
    Compute change in cloud cover area over time.
    
    Parameters
    ----------
    cloud_mask : np.ndarray [time, lat, lon]
        Binary mask (1 = cloudy, 0 = clear).
    times : list of np.datetime64
        Time steps corresponding to cloud_mask.
    pixel_area_km2 : float
        Area of one grid cell in km² (default 5km x 5km = 25 km²).
    
    Returns
    -------
    rate : float
        Normalized rate of CC change per hour (km²/hour).
    total_change : float
        Relative CC change (ΔCC / CC_init).
    """
    t0, t1 = times[0], times[-1]
    duration_h = (t1 - t0).astype("timedelta64[m]").astype(float) / 60.0
    
    cc_init = cloud_mask[0].sum() * pixel_area_km2
    cc_final = cloud_mask[-1].sum() * pixel_area_km2
    delta_cc = cc_final - cc_init
    
    if cc_init == 0:
        relative_change = np.nan
    else:
        relative_change = delta_cc / cc_init
    
    rate = delta_cc / duration_h if duration_h > 0 else np.nan
    return rate, relative_change



def vertical_growth(cth, times, method="max"):
    """
    Compute vertical cloud growth rate from Cloud Top Height (CTH).
    
    Parameters
    ----------
    cth : np.ndarray [time, lat, lon]
        Cloud top height (m).
    times : list of np.datetime64
        Time steps.
    method : str
        "max" = use maximum CTH per frame, "mean" = use mean CTH over cloud pixels.
    
    Returns
    -------
    growth_rate : float
        Growth rate in m/s.
    total_growth : float
        Total relative growth (ΔCTH / CTH_init).
    """
    t0, t1 = times[0], times[-1]
    duration_s = (t1 - t0).astype("timedelta64[s]").astype(float)
    
    if method == "max":
        cth_init, cth_final = np.nanmax(cth[0]), np.nanmax(cth[-1])
    else:
        cth_init, cth_final = np.nanmean(cth[0]), np.nanmean(cth[-1])
    
    delta_cth = cth_final - cth_init
    
    if cth_init == 0:
        relative_growth = np.nan
    else:
        relative_growth = delta_cth / cth_init
    
    growth_rate = delta_cth / duration_s if duration_s > 0 else np.nan
    return growth_rate, relative_growth



def horizontal_motion(cloud_mask, times, method="centroid"):
    """
    Estimate horizontal motion of cloud system.
    
    Parameters
    ----------
    cloud_mask : np.ndarray [time, lat, lon]
        Binary mask of cloudy pixels.
    times : list of np.datetime64
        Time steps.
    method : str
        "centroid" = use center of mass, "max" = use max CTH pixel (requires CTH input).
    
    Returns
    -------
    velocity : float
        Motion speed in pixels per hour.
    displacement : float
        Total displacement in pixels.
    vector : tuple
        (dx, dy) displacement in pixels.
    """
    
    t0, t1 = times[0], times[-1]
    duration_h = (t1 - t0).astype("timedelta64[m]").astype(float) / 60.0
    
    # Center of mass
    y0, x0 = center_of_mass(cloud_mask[0])
    y1, x1 = center_of_mass(cloud_mask[-1])
    
    dx, dy = x1 - x0, y1 - y0
    displacement = np.sqrt(dx**2 + dy**2)
    velocity = displacement / duration_h if duration_h > 0 else np.nan
    
    return velocity, displacement, (dx, dy)


def motion_direction(vector):
    """
    Compute direction of motion from displacement vector.
    
    Parameters
    ----------
    vector : tuple
        (dx, dy) displacement in pixels (dx east, dy north).
    
    Returns
    -------
    angle_deg : float
        Motion direction in degrees (0 = east, 90 = north).
    """
    dx, dy = vector
    angle_rad = np.arctan2(dy, dx)  # atan2(y, x)
    angle_deg = np.degrees(angle_rad) % 360
    return angle_deg
