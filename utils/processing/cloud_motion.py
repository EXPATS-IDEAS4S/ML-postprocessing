"""
Utilities to estimate bulk cloud motion from short satellite image videos.

The main method uses OpenCV phase correlation to estimate one translation
between each consecutive pair of frames, then averages those displacement
vectors over the video.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class CloudMotionResult:
    """Mean cloud-motion estimate for one video."""

    mean_dx_pixels_per_frame: float
    mean_dy_pixels_per_frame: float
    mean_speed_pixels_per_frame: float
    mean_speed_kmh: float
    mean_direction_to_deg: float
    mean_direction_from_deg: float
    n_pairs_used: int
    mean_response: float
    pair_dx_pixels: list[float]
    pair_dy_pixels: list[float]
    pair_response: list[float]

    def as_dict(self) -> dict:
        return {
            "mean_dx_pixels_per_frame": self.mean_dx_pixels_per_frame,
            "mean_dy_pixels_per_frame": self.mean_dy_pixels_per_frame,
            "mean_speed_pixels_per_frame": self.mean_speed_pixels_per_frame,
            "mean_speed_kmh": self.mean_speed_kmh,
            "mean_direction_to_deg": self.mean_direction_to_deg,
            "mean_direction_from_deg": self.mean_direction_from_deg,
            "n_pairs_used": self.n_pairs_used,
            "mean_response": self.mean_response,
        }


def compute_cloud_motion_from_nc(
    nc_path: str,
    variable: str = "IR_108",
    pixel_size_km: float | None = None,
    frame_interval_minutes: float | None = None,
    nc_engine: str | None = None,
    invalid_value: float | None = None,
    min_valid_fraction: float = 0.05,
    max_shift_pixels: float | None = None,
) -> CloudMotionResult:
    """
    Load a NetCDF video crop and estimate mean cloud motion.

    Parameters
    ----------
    nc_path : str
        Path to a NetCDF crop with dimensions like time, y, x or time, lat, lon.
    variable : str, optional
        Data variable containing the masked IR images.
    pixel_size_km : float, optional
        Pixel spacing in km. If omitted, speed_kmh is NaN.
    frame_interval_minutes : float, optional
        Time between consecutive frames. If omitted, speed_kmh is NaN.
    nc_engine : str, optional
        xarray backend engine, e.g. "h5netcdf" or "netcdf4".
    invalid_value : float, optional
        Extra mask value to ignore, commonly 0 if non-cloud pixels are zero.
    min_valid_fraction : float, optional
        Minimum overlapping valid-pixel fraction required for a frame pair.
    max_shift_pixels : float, optional
        Drop pair estimates whose vector magnitude exceeds this value.

    Returns
    -------
    CloudMotionResult
        Mean displacement, speed, direction, and diagnostics.
    """

    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError(
            "xarray is required to load NetCDF crops. Run this inside the project "
            "environment, for example the conda env used by the other scripts."
        ) from exc

    open_kwargs = {}
    if nc_engine is not None:
        open_kwargs["engine"] = nc_engine

    with xr.open_dataset(nc_path, **open_kwargs) as ds:
        if variable not in ds:
            available = ", ".join(ds.data_vars)
            raise KeyError(f"Variable {variable!r} not found in {nc_path}. Available: {available}")
        video = ds[variable].values

    return compute_cloud_motion(
        video,
        pixel_size_km=pixel_size_km,
        frame_interval_minutes=frame_interval_minutes,
        invalid_value=invalid_value,
        min_valid_fraction=min_valid_fraction,
        max_shift_pixels=max_shift_pixels,
    )


def compute_cloud_motion(
    video: np.ndarray,
    pixel_size_km: float | None = None,
    frame_interval_minutes: float | None = None,
    invalid_value: float | None = None,
    min_valid_fraction: float = 0.05,
    max_shift_pixels: float | None = None,
) -> CloudMotionResult:
    """
    Estimate the mean bulk motion of a cloud video using phase correlation.

    The input must have shape (time, y, x). The returned dx/dy are in image
    coordinates: positive dx is rightward, positive dy is downward. Directions
    are geographic image bearings assuming image x is eastward and image y is
    southward: 0 deg means northward motion, 90 deg eastward motion.
    """

    frames = np.asarray(video)
    if frames.ndim != 3:
        raise ValueError(f"Expected video shape (time, y, x), got {frames.shape}")
    if frames.shape[0] < 2:
        raise ValueError("At least two frames are required to estimate motion.")

    pair_dx = []
    pair_dy = []
    pair_response = []

    for first, second in zip(frames[:-1], frames[1:]):
        result = phase_correlate_pair(
            first,
            second,
            invalid_value=invalid_value,
            min_valid_fraction=min_valid_fraction,
        )
        if result is None:
            continue

        dx, dy, response = result
        if max_shift_pixels is not None and np.hypot(dx, dy) > max_shift_pixels:
            continue

        pair_dx.append(float(dx))
        pair_dy.append(float(dy))
        pair_response.append(float(response))

    if not pair_dx:
        return _empty_result()

    mean_dx = float(np.mean(pair_dx))
    mean_dy = float(np.mean(pair_dy))
    speed_pixels = float(np.hypot(mean_dx, mean_dy))
    speed_kmh = _pixels_per_frame_to_kmh(speed_pixels, pixel_size_km, frame_interval_minutes)
    direction_to = _image_vector_to_bearing(mean_dx, mean_dy)
    direction_from = (direction_to + 180.0) % 360.0 if np.isfinite(direction_to) else np.nan

    return CloudMotionResult(
        mean_dx_pixels_per_frame=mean_dx,
        mean_dy_pixels_per_frame=mean_dy,
        mean_speed_pixels_per_frame=speed_pixels,
        mean_speed_kmh=speed_kmh,
        mean_direction_to_deg=direction_to,
        mean_direction_from_deg=direction_from,
        n_pairs_used=len(pair_dx),
        mean_response=float(np.mean(pair_response)),
        pair_dx_pixels=pair_dx,
        pair_dy_pixels=pair_dy,
        pair_response=pair_response,
    )


def phase_correlate_pair(
    frame1: np.ndarray,
    frame2: np.ndarray,
    invalid_value: float | None = None,
    min_valid_fraction: float = 0.05,
) -> tuple[float, float, float] | None:
    """
    Estimate one translation vector between two frames.

    Returns
    -------
    tuple or None
        (dx, dy, response), where dx/dy are pixels per frame. None is returned
        when too few valid pixels are available.
    """

    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required for phase correlation. Install/use an environment "
            "with opencv-python available."
        ) from exc

    arr1, arr2, valid = _prepare_pair(frame1, frame2, invalid_value)
    if valid.mean() < min_valid_fraction:
        return None

    window = cv2.createHanningWindow((arr1.shape[1], arr1.shape[0]), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(arr1, arr2, window)
    dx, dy = shift
    return float(dx), float(dy), float(response)


def compute_cloud_motion_for_paths(
    paths: Iterable[str],
    variable: str = "IR_108",
    pixel_size_km: float | None = None,
    frame_interval_minutes: float | None = None,
    nc_engine: str | None = None,
    invalid_value: float | None = None,
    min_valid_fraction: float = 0.05,
    max_shift_pixels: float | None = None,
) -> list[dict]:
    """Compute cloud-motion rows for a sequence of NetCDF crop paths."""

    rows = []
    for path in paths:
        result = compute_cloud_motion_from_nc(
            path,
            variable=variable,
            pixel_size_km=pixel_size_km,
            frame_interval_minutes=frame_interval_minutes,
            nc_engine=nc_engine,
            invalid_value=invalid_value,
            min_valid_fraction=min_valid_fraction,
            max_shift_pixels=max_shift_pixels,
        )
        row = {"path": path}
        row.update(result.as_dict())
        rows.append(row)
    return rows


def _prepare_pair(
    frame1: np.ndarray,
    frame2: np.ndarray,
    invalid_value: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr1 = np.asarray(frame1, dtype=np.float32)
    arr2 = np.asarray(frame2, dtype=np.float32)

    valid1 = np.isfinite(arr1)
    valid2 = np.isfinite(arr2)
    if invalid_value is not None:
        valid1 &= arr1 != invalid_value
        valid2 &= arr2 != invalid_value
    valid = valid1 & valid2

    arr1 = _normalize_for_phase_correlation(arr1, valid)
    arr2 = _normalize_for_phase_correlation(arr2, valid)
    return arr1, arr2, valid


def _normalize_for_phase_correlation(frame: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros(frame.shape, dtype=np.float32)
    values = frame[valid]
    if values.size == 0:
        return out

    mean = float(np.nanmean(values))
    std = float(np.nanstd(values))
    if not np.isfinite(std) or std == 0:
        std = 1.0

    out[valid] = (frame[valid] - mean) / std
    return out


def _pixels_per_frame_to_kmh(
    speed_pixels_per_frame: float,
    pixel_size_km: float | None,
    frame_interval_minutes: float | None,
) -> float:
    if pixel_size_km is None or frame_interval_minutes is None:
        return np.nan
    if frame_interval_minutes <= 0:
        raise ValueError("frame_interval_minutes must be positive.")
    return float(speed_pixels_per_frame * pixel_size_km / (frame_interval_minutes / 60.0))


def _image_vector_to_bearing(dx: float, dy: float) -> float:
    if not np.isfinite(dx) or not np.isfinite(dy):
        return np.nan
    if dx == 0 and dy == 0:
        return np.nan

    eastward = dx
    northward = -dy
    return float(np.degrees(np.arctan2(eastward, northward)) % 360.0)


def _empty_result() -> CloudMotionResult:
    return CloudMotionResult(
        mean_dx_pixels_per_frame=np.nan,
        mean_dy_pixels_per_frame=np.nan,
        mean_speed_pixels_per_frame=np.nan,
        mean_speed_kmh=np.nan,
        mean_direction_to_deg=np.nan,
        mean_direction_from_deg=np.nan,
        n_pairs_used=0,
        mean_response=np.nan,
        pair_dx_pixels=[],
        pair_dy_pixels=[],
        pair_response=[],
    )
