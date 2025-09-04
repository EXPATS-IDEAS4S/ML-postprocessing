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


import numpy as np

def change_in_cloud_cover(cloud_mask, times, relative_change=False, rate=False, pixel_area_km2=None):
    """
    Compute change in cloud cover over time, with optional normalization and unit conversion.
    
    Parameters
    ----------
    cloud_mask : np.ndarray [time, lat, lon]
        Binary mask (1 = cloudy, 0 = clear).
    times : list of np.datetime64
        Time steps corresponding to cloud_mask.
    relative_change : bool, optional
        If True, compute relative change (ΔCC / CC_init).
    rate : bool, optional
        If True, compute rate of change (ΔCC / Δt).
    pixel_area_km2 : float or None, optional
        Area of one grid cell in km². If None, results are in pixel counts.
    
    Returns
    -------
    results : dict
        Dictionary containing requested metrics:
        - "absolute_change": ΔCC (km² or pixels)
        - "relative_change": ΔCC / CC_init (if requested)
        - "rate": ΔCC / Δt (km²/hour or pixels/hour, if requested)
    """
    results = {}

    # Duration in minutes
    t0, t1 = times[0], times[-1]
    duration = (t1 - t0).astype("timedelta64[m]").astype(float)

    # Initial and final cloud cover
    cc_init = cloud_mask[0].sum()
    cc_final = cloud_mask[-1].sum()

    # Convert to km² if requested
    if pixel_area_km2 is not None:
        cc_init *= pixel_area_km2
        cc_final *= pixel_area_km2

    delta_cc = cc_final - cc_init
    results["absolute_change"] = delta_cc

    # Relative change
    if relative_change:
        results["relative_change"] = np.nan if cc_init == 0 else delta_cc / cc_init

    # Rate of change
    if rate:
        results["rate"] = delta_cc / duration if duration > 0 else np.nan

    return results




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
