#!/usr/bin/env python3
"""Feature-space analysis on TeamX t-SNE vectors.

Creates a 2-panel figure (TEST msg and TEST icon) with:
- TRAIN feature-space overlay as faint background points
- CENTROID positions as stars
- Convective class (EC/DC/OA) density contours for TEST vectors
  - solid line: 50% highest-density region
  - dashed line: 90% highest-density region
"""

import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

import sys

sys.path.append('/home/Daniele/codes/VISSL_postprocessing/')
from utils.plotting.class_colors import CLOUD_CLASS_INFO


# ============================================================================
# CONFIG
# ============================================================================
RUN_NAME = "dcv2_resnet_k7_ir108_100x100_2013-2020_1xrandomcrops_1xtimestamp_cma_nc"
BASE_DIR = f"/data1/fig/{RUN_NAME}/epoch_800/teamx_test"
CSV_FILE = os.path.join(BASE_DIR, "tsne_all_vectors_with_centroids.csv")

OUT_DIR = os.path.join(BASE_DIR, "feature_space_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

FOCUS_SHORT_CLASSES = ["EC", "DC", "OA"]
VECTOR_TYPES = ["msg", "icon"]


# ============================================================================
# HELPERS
# ============================================================================
def infer_source_type_from_filename(filename_val: object) -> str:
	"""Infer source type (msg/icon) from filename."""
	if not isinstance(filename_val, str):
		return "unknown"

	s = filename_val.lower()
	if "icon" in s:
		return "icon"
	if "msg" in s:
		return "msg"
	return "unknown"


def find_tsne_columns(df: pd.DataFrame) -> Tuple[str, str]:
	"""Find t-SNE coordinate columns in a robust way."""
	preferred = [("tsne_dim_1", "tsne_dim_2"), ("tsne_1", "tsne_2"), ("TSNE1", "TSNE2")]
	for c1, c2 in preferred:
		if c1 in df.columns and c2 in df.columns:
			return c1, c2

	# Fallback: any two columns containing tsne and ending with 1/2.
	tsne_cols = [c for c in df.columns if "tsne" in c.lower()]
	if len(tsne_cols) >= 2:
		sorted_cols = sorted(tsne_cols, key=lambda c: c.lower())
		return sorted_cols[0], sorted_cols[1]

	raise ValueError("Could not find t-SNE columns in input CSV.")


def kde_grid_and_thresholds(
	x: np.ndarray,
	y: np.ndarray,
	xgrid: np.ndarray,
	ygrid: np.ndarray,
	bandwidth: float,
	quantiles: List[float],
) -> Tuple[np.ndarray, Dict[float, float]]:
	"""Estimate 2D KDE on grid and return density thresholds for HDR quantiles."""
	xy = np.column_stack([x, y])
	kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
	kde.fit(xy)

	grid_points = np.column_stack([xgrid.ravel(), ygrid.ravel()])
	log_dens = kde.score_samples(grid_points)
	dens = np.exp(log_dens).reshape(xgrid.shape)

	flat = dens.ravel()
	order = np.argsort(flat)[::-1]
	sorted_dens = flat[order]
	cumsum = np.cumsum(sorted_dens)
	total = cumsum[-1] if cumsum.size > 0 else 0.0

	thresholds: Dict[float, float] = {}
	if total <= 0:
		for q in quantiles:
			thresholds[q] = np.nan
		return dens, thresholds

	for q in quantiles:
		idx = np.searchsorted(cumsum, q * total)
		idx = min(idx, len(sorted_dens) - 1)
		thresholds[q] = float(sorted_dens[idx])

	return dens, thresholds


def build_class_metadata() -> Tuple[Dict[str, int], Dict[int, str], Dict[int, str]]:
	"""Return class id map by short name plus color/short lookups by class id."""
	id_by_short = {info["short"]: cid for cid, info in CLOUD_CLASS_INFO.items()}
	short_by_id = {cid: info["short"] for cid, info in CLOUD_CLASS_INFO.items()}
	color_by_id = {cid: info["color"] for cid, info in CLOUD_CLASS_INFO.items()}
	return id_by_short, short_by_id, color_by_id


# ============================================================================
# PLOTTING
# ============================================================================
def plot_feature_space_panels(df: pd.DataFrame, xcol: str, ycol: str, out_path: str) -> None:
	id_by_short, short_by_id, color_by_id = build_class_metadata()
	focus_ids = [id_by_short[s] for s in FOCUS_SHORT_CLASSES if s in id_by_short]
	order_by_id = {cid: info.get("order", 999) for cid, info in CLOUD_CLASS_INFO.items()}

	if len(focus_ids) == 0:
		raise RuntimeError("No convective class IDs found for EC/DC/OA in CLOUD_CLASS_INFO.")

	# Keep convective subset for TEST density contours only.
	df_conv = df[df["label"].isin(focus_ids)].copy()

	# Build subsets.
	train_df = df[df["split"] == "TRAIN"].copy()
	cent_df = df[df["split"] == "CENTROID"].copy()
	test_df = df_conv[df_conv["split"] == "TEST"].copy()

	if test_df.empty:
		raise RuntimeError("No TEST rows found for EC/DC/OA classes.")

	# Global axes limits based on all relevant rows (TRAIN/TEST/CENTROID).
	df_limits = df[df["split"].isin(["TRAIN", "TEST", "CENTROID"])].copy()
	x_pad = 0.05 * (df_limits[xcol].max() - df_limits[xcol].min() + 1e-12)
	y_pad = 0.05 * (df_limits[ycol].max() - df_limits[ycol].min() + 1e-12)
	x_min = df_limits[xcol].min() - x_pad
	x_max = df_limits[xcol].max() + x_pad
	y_min = df_limits[ycol].min() - y_pad
	y_max = df_limits[ycol].max() + y_pad

	# Shared KDE grid.
	nx, ny = 220, 220
	x_vals = np.linspace(x_min, x_max, nx)
	y_vals = np.linspace(y_min, y_max, ny)
	xgrid, ygrid = np.meshgrid(x_vals, y_vals)

	fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharex=True, sharey=True)

	for ax, vtype in zip(axes, VECTOR_TYPES):
		panel_test = test_df[test_df["source_type"] == vtype].copy()

		# TRAIN overlay (all classes, faint background).
		if not train_df.empty:
			ax.scatter(
				train_df[xcol].values,
				train_df[ycol].values,
				s=6,
				c="lightgray",
				alpha=0.08,
				linewidths=0,
				label="TRAIN overlay",
			)

		# Class-specific TEST density contours.
		for class_id in focus_ids:
			cls_df = panel_test[panel_test["label"] == class_id]
			if len(cls_df) < 20:
				continue

			x = cls_df[xcol].to_numpy(dtype=float)
			y = cls_df[ycol].to_numpy(dtype=float)

			# Bandwidth scaled on panel spread for stable contours.
			spread = max(np.std(x), np.std(y), 1e-3)
			bw = 0.18 * spread

			dens, th = kde_grid_and_thresholds(
				x=x,
				y=y,
				xgrid=xgrid,
				ygrid=ygrid,
				bandwidth=bw,
				quantiles=[0.50, 0.90],
			)

			color = color_by_id.get(class_id, "black")
			q50 = th.get(0.50, np.nan)
			q90 = th.get(0.90, np.nan)
			if np.isfinite(q50):
				ax.contour(
					xgrid,
					ygrid,
					dens,
					levels=[q50],
					colors=[color],
					linewidths=2.2,
					linestyles="solid",
				)
			if np.isfinite(q90):
				ax.contour(
					xgrid,
					ygrid,
					dens,
					levels=[q90],
					colors=[color],
					linewidths=1.9,
					linestyles="dashed",
				)

		# CENTROID stars for all classes (same centroids on both msg/icon panels).
		cent_panel = cent_df
		centroid_ids = sorted(
			cent_panel["label"].dropna().astype(int).unique().tolist(),
			key=lambda cid: (order_by_id.get(cid, 999), cid),
		)
		for class_id in centroid_ids:
			c_df = cent_panel[cent_panel["label"] == class_id]
			if c_df.empty:
				continue
			ax.scatter(
				c_df[xcol].values,
				c_df[ycol].values,
				s=260,
				marker="*",
				c=[color_by_id.get(class_id, "black")],
				edgecolors="black",
				linewidths=0.9,
				zorder=6,
			)

		ax.set_title(f"TEST {vtype.upper()}", fontsize=13, fontweight="bold")
		ax.set_xlim(x_min, x_max)
		ax.set_ylim(y_min, y_max)
		ax.grid(alpha=0.2)
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_xlabel("")
		ax.set_ylabel("")
		for spine in ax.spines.values():
			spine.set_visible(False)

	# Build legend: density contour lines only.
	legend_handles = []
	legend_labels = []

	ordered_focus_shorts = [s for s in FOCUS_SHORT_CLASSES if s in id_by_short]

	# First row: all 50% contour entries in EC, DC, OA order.
	for short in ordered_focus_shorts:
		class_id = id_by_short[short]
		color = color_by_id.get(class_id, "black")
		legend_handles.append(plt.Line2D([0], [0], color=color, linewidth=2.2, linestyle="solid"))
		legend_labels.append(f"{short} 50%")

	# Second row: all 90% contour entries in EC, DC, OA order.
	for short in ordered_focus_shorts:
		class_id = id_by_short[short]
		color = color_by_id.get(class_id, "black")
		legend_handles.append(plt.Line2D([0], [0], color=color, linewidth=1.9, linestyle="dashed"))
		legend_labels.append(f"{short} 90%")

	fig.legend(
		legend_handles,
		legend_labels,
		loc="lower center",
		ncol=max(1, len(focus_ids)),
		fontsize=9,
		frameon=False,
		bbox_to_anchor=(0.5, -0.02),
	)

	fig.suptitle(
		"Feature Space: Convective Classes Density Contours",
		fontsize=14,
		fontweight="bold",
	)
	plt.tight_layout(rect=[0, 0.08, 1, 0.96])
	plt.savefig(out_path, dpi=300)
	plt.close()


# ============================================================================
# MAIN
# ============================================================================
def main() -> None:
	if not os.path.exists(CSV_FILE):
		raise FileNotFoundError(f"Input CSV not found: {CSV_FILE}")

	print(f"Loading CSV: {CSV_FILE}")
	df = pd.read_csv(CSV_FILE, low_memory=False)
	print(f"Loaded shape: {df.shape}")

	xcol, ycol = find_tsne_columns(df)
	print(f"Using t-SNE columns: {xcol}, {ycol}")

	if "vector_type" not in df.columns:
		raise ValueError("CSV must contain 'vector_type' column.")
	if "filename" not in df.columns:
		raise ValueError("CSV must contain 'filename' column to infer msg/icon source type.")
	if "label" not in df.columns:
		raise ValueError("CSV must contain 'label' column.")

	# In this CSV, vector_type stores split (TRAIN/TEST/CENTROID).
	df["split"] = df["vector_type"].astype(str).str.upper()
	df["source_type"] = df["filename"].apply(infer_source_type_from_filename)
	df["label"] = pd.to_numeric(df["label"], errors="coerce")
	df = df.dropna(subset=[xcol, ycol, "label"]).copy()
	df["label"] = df["label"].astype(int)

	# Keep only rows relevant to requested analysis.
	df = df[df["split"].isin(["TRAIN", "TEST", "CENTROID"])].copy()
	test_unknown = ((df["split"] == "TEST") & (df["source_type"] == "unknown")).sum()
	if test_unknown > 0:
		print(f"Warning: {test_unknown} TEST rows have unknown source_type from filename and will not appear in msg/icon panels.")

	out_path = os.path.join(OUT_DIR, "feature_space_test_msg_icon_convective_density.png")
	plot_feature_space_panels(df=df, xcol=xcol, ycol=ycol, out_path=out_path)

	print("Done.")
	print(f"Figure saved to: {out_path}")


if __name__ == "__main__":
	main()
