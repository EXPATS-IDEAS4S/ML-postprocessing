#!/usr/bin/env python3
"""Case-study analysis for TeamX TEST on a selected day.

Outputs:
1) Feature-space plot with TRAIN background + CENTROIDS + selected-day TEST trajectories
   for msg and icon, colored by hour.
2) Evolution plot with shared lead time:
   - class assigned over lead time (msg and icon)
   - cosine distance between matched msg/icon feature vectors
   - optional crop strip (msg row and icon row) using merged crop CSV image paths
"""

import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from sklearn.metrics.pairwise import cosine_similarity

import sys

sys.path.append('/home/Daniele/codes/VISSL_postprocessing/')
from utils.plotting.class_colors import CLOUD_CLASS_INFO


# ============================================================================
# CONFIG
# ============================================================================
RUN_NAME = "dcv2_resnet_k7_ir108_100x100_2013-2020_1xrandomcrops_1xtimestamp_cma_nc"
BASE_DIR = f"/data1/fig/{RUN_NAME}/epoch_800/teamx_test"

MERGED_TSNE_CSV = os.path.join(BASE_DIR, "merged_tsne_crop_list_with_img_path.csv")
FEATURE_CSV = os.path.join(BASE_DIR, f"features_train_test_{RUN_NAME}_2nd_labels.csv")

SELECTED_DAY = "2025-06-30"
SELECTED_REGION = "central"

OUT_DIR = os.path.join(BASE_DIR, "case_study")
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================================
# HELPERS
# ============================================================================
def parse_datetime_from_filename(filename_val: object) -> pd.Timestamp:
	"""Parse datetime from MSG/ICON filename.

	Supported:
	- MSG_108_cma_YYYY-MM-DD_HH_region.nc
	- ICON500m_IR10.8_CLCT_YYYYMMDD_HH_YYYY-MM-DD_HH_region.nc
	  (use second datetime)
	"""
	if not isinstance(filename_val, str):
		return pd.NaT

	stem = os.path.splitext(os.path.basename(filename_val))[0]

	m_msg = re.search(
		r"^MSG_108_cma_(\d{4}-\d{2}-\d{2})_(\d{2})_(east|central|west)$",
		stem,
		re.IGNORECASE,
	)
	if m_msg:
		return pd.to_datetime(f"{m_msg.group(1)}_{m_msg.group(2)}", format="%Y-%m-%d_%H", errors="coerce")

	m_icon = re.search(
		r"^ICON500m_IR10\.8_CLCT_(\d{8})_(\d{2})_(\d{4}-\d{2}-\d{2})_(\d{2})_(east|central|west)$",
		stem,
		re.IGNORECASE,
	)
	if m_icon:
		return pd.to_datetime(f"{m_icon.group(3)}_{m_icon.group(4)}", format="%Y-%m-%d_%H", errors="coerce")

	return pd.NaT


def extract_region_from_filename(filename_val: object) -> str:
	if not isinstance(filename_val, str):
		return "unknown"
	stem = os.path.splitext(os.path.basename(filename_val))[0].lower()
	m = re.search(r"(?:_|-)(east|central|west)$", stem)
	return m.group(1) if m else "unknown"


def infer_source_from_filename(filename_val: object) -> str:
	if not isinstance(filename_val, str):
		return "unknown"
	s = filename_val.lower()
	if "icon" in s:
		return "icon"
	if "msg" in s:
		return "msg"
	return "unknown"


def to_lead_hour(ts: pd.Series) -> pd.Series:
	"""Map hour 00 to 24 for lead-time style plotting."""
	out = ts.dt.hour.astype(int)
	out[out == 0] = 24
	return out


def class_maps() -> Tuple[Dict[int, str], Dict[int, str], List[int]]:
	short_by_id = {cid: info["short"] for cid, info in CLOUD_CLASS_INFO.items()}
	color_by_id = {cid: info["color"] for cid, info in CLOUD_CLASS_INFO.items()}
	ordered_ids = [cid for cid, _ in sorted(CLOUD_CLASS_INFO.items(), key=lambda x: x[1]["order"])]
	return short_by_id, color_by_id, ordered_ids


def rowwise_cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
	if a.shape != b.shape:
		raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
	return 1.0 - np.diag(cosine_similarity(a, b))


def add_colored_trajectory(ax, df_src: pd.DataFrame, xcol: str, ycol: str, linestyle: str, label: str) -> None:
	"""Add trajectory segments colored by lead hour (single selected region expected)."""
	if df_src.empty:
		return

	dreg = df_src.sort_values("datetime")
	if len(dreg) >= 2:
		pts = dreg[[xcol, ycol]].to_numpy()
		segments = np.stack([pts[:-1], pts[1:]], axis=1)
		hours = dreg["lead_hour"].to_numpy()[:-1]

		lc = LineCollection(
			segments,
			cmap="winter",
			norm=plt.Normalize(1, 24),
			linestyle=linestyle,
			linewidths=3.2,
			alpha=0.95,
		)
		lc.set_array(hours)
		ax.add_collection(lc)

	# Overlay points for readability.
	ax.scatter(
		df_src[xcol].values,
		df_src[ycol].values,
		c=df_src["lead_hour"].values,
		cmap="winter",
		vmin=1,
		vmax=24,
		s=28,
		edgecolors="white",
		linewidths=0.3,
		alpha=0.95,
		label=label,
	)


def select_one_pair_per_hour(img_pairs: pd.DataFrame) -> pd.DataFrame:
	"""Pick one representative msg/icon pair per lead hour (prefer central region)."""
	if img_pairs.empty:
		return img_pairs

	rows = []
	for h, d in img_pairs.groupby("lead_hour"):
		d = d.sort_values(["datetime", "region"])
		central = d[d["region"] == "central"]
		rows.append((central.iloc[0] if not central.empty else d.iloc[0]))
	return pd.DataFrame(rows).sort_values("lead_hour")


def add_image_at(ax, img_path: str, x: float, y: float, zoom: float = 0.14) -> None:
	if not isinstance(img_path, str) or not os.path.exists(img_path):
		return
	try:
		img = plt.imread(img_path)
	except Exception:
		return
	ab = AnnotationBbox(OffsetImage(img, zoom=zoom), (x, y), frameon=False)
	ax.add_artist(ab)


# ============================================================================
# PLOT 1: FEATURE SPACE + TRAJECTORY
# ============================================================================
def plot_feature_space_trajectory(df_all: pd.DataFrame, df_day_test: pd.DataFrame, out_path: str) -> None:
	short_by_id, color_by_id, ordered_ids = class_maps()
	xcol, ycol = "tsne_dim_1", "tsne_dim_2"

	train_df = df_all[df_all["split"] == "TRAIN"].copy()
	cent_df = df_all[df_all["split"] == "CENTROID"].copy()

	fig, ax = plt.subplots(figsize=(8, 6))

	# TRAIN background with class color and high transparency.
	if not train_df.empty:
		for cid in ordered_ids:
			t = train_df[train_df["label"] == cid]
			if t.empty:
				continue
			ax.scatter(
				t[xcol].values,
				t[ycol].values,
				s=5,
				c=[color_by_id.get(cid, "lightgray")],
				alpha=0.05,
				linewidths=0,
			)

	# Centroids for all classes.
	for cid in ordered_ids:
		d = cent_df[cent_df["label"] == cid]
		if d.empty:
			continue
		ax.scatter(
			d[xcol].values,
			d[ycol].values,
			s=420,
			marker="*",
			c=[color_by_id.get(cid, "black")],
			edgecolors="black",
			linewidths=1.2,
			zorder=20,
		)

	msg_day = df_day_test[df_day_test["source_type"] == "msg"].copy()
	icon_day = df_day_test[df_day_test["source_type"] == "icon"].copy()

	add_colored_trajectory(ax, msg_day, xcol, ycol, "solid", "MSG")
	add_colored_trajectory(ax, icon_day, xcol, ycol, "dashed", "ICON")

	# Explicit legend: differentiate MSG/ICON by linestyle only.
	legend_handles = [
		Line2D([0], [0], color="black", linewidth=3.2, linestyle="solid", label="MSG"),
		Line2D([0], [0], color="black", linewidth=3.2, linestyle="dashed", label="ICON"),
	]

	# Colorbar for hour.
	sm = plt.cm.ScalarMappable(cmap="winter", norm=plt.Normalize(1, 24))
	sm.set_array([])
	cbar = plt.colorbar(sm, ax=ax, pad=0.01, aspect=40, shrink=0.78)
	cbar.set_label("Lead time", fontsize=14)
	cbar.ax.tick_params(labelsize=12)

	ax.set_title(f"{SELECTED_DAY} ({SELECTED_REGION})", fontsize=18, fontweight="bold")
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_xlabel("")
	ax.set_ylabel("")
	for spine in ax.spines.values():
		spine.set_visible(False)
	ax.grid(alpha=0.15)
	ax.legend(handles=legend_handles, loc="upper right", fontsize=12)
	plt.tight_layout()
	plt.savefig(out_path, dpi=300)
	plt.close()


# ============================================================================
# PLOT 2: CLASS + COS DIST + CROPS
# ============================================================================
def build_feature_pairs_for_day(day: str) -> pd.DataFrame:
	"""Load feature CSV and build matched msg/icon pairs for cosine distance."""
	if not os.path.exists(FEATURE_CSV):
		return pd.DataFrame()

	df = pd.read_csv(FEATURE_CSV, low_memory=False)
	df["split"] = df["split"].astype(str).str.upper()
	df = df[df["split"] == "TEST"].copy()
	df["vector_type"] = df["vector_type"].astype(str).str.lower()
	df = df[df["vector_type"].isin(["msg", "icon"])].copy()

	df["filename_only"] = df["path"].astype(str).map(os.path.basename)
	df["datetime"] = df["filename_only"].apply(parse_datetime_from_filename)
	df["region"] = df["filename_only"].apply(extract_region_from_filename)
	df = df.dropna(subset=["datetime"]).copy()
	df = df[df["datetime"].dt.strftime("%Y-%m-%d") == day].copy()
	df = df[df["region"] == SELECTED_REGION].copy()

	feature_cols = [c for c in df.columns if c.startswith("dim_")]
	if not feature_cols:
		return pd.DataFrame()

	if "distance" in df.columns:
		df = df.sort_values("distance", ascending=False)
	df = df.drop_duplicates(subset=["datetime", "region", "vector_type"], keep="first")

	msg = df[df["vector_type"] == "msg"][["datetime", "region", "label"] + feature_cols].copy()
	icon = df[df["vector_type"] == "icon"][["datetime", "region", "label"] + feature_cols].copy()

	msg = msg.rename(columns={"label": "label_msg"})
	icon = icon.rename(columns={"label": "label_icon"})
	pairs = msg.merge(icon, on=["datetime", "region"], suffixes=("_msg", "_icon"), how="inner")
	if pairs.empty:
		return pairs

	feat_msg = pairs[[f"{c}_msg" for c in feature_cols]].values.astype(float)
	feat_icon = pairs[[f"{c}_icon" for c in feature_cols]].values.astype(float)
	pairs["cosine_distance"] = rowwise_cosine_distance(feat_msg, feat_icon)
	pairs["lead_hour"] = to_lead_hour(pairs["datetime"])

	return pairs


def plot_evolution_and_crops(df_day_test: pd.DataFrame, out_path: str) -> None:
	short_by_id, _, ordered_ids = class_maps()

	# Class assignments from t-SNE merged CSV day slice.
	df_plot = df_day_test.copy()
	df_plot["lead_hour"] = to_lead_hour(df_plot["datetime"])

	# Cosine distance from full feature vectors.
	feat_pairs = build_feature_pairs_for_day(SELECTED_DAY)

	# Build msg/icon pairs for crop display and optional class pairing.
	msg = df_plot[df_plot["source_type"] == "msg"].copy()
	icon = df_plot[df_plot["source_type"] == "icon"].copy()
	msg = msg.sort_values("distance", ascending=False) if "distance" in msg.columns else msg
	icon = icon.sort_values("distance", ascending=False) if "distance" in icon.columns else icon
	msg = msg.drop_duplicates(subset=["datetime", "region"], keep="first")
	icon = icon.drop_duplicates(subset=["datetime", "region"], keep="first")

	class_pairs = msg[["datetime", "region", "lead_hour", "label", "img_path"]].rename(
		columns={"label": "label_msg", "img_path": "img_path_msg"}
	).merge(
		icon[["datetime", "region", "lead_hour", "label", "img_path"]].rename(
			columns={"label": "label_icon", "img_path": "img_path_icon"}
		),
		on=["datetime", "region", "lead_hour"],
		how="inner",
	)

	fig, axes = plt.subplots(
		3,
		1,
		figsize=(12, 8),
		sharex=True,
		gridspec_kw={"height_ratios": [0.7, 0.5, 1.]},
	)

	ax_cls, ax_cos, ax_img = axes

	# -------------------------------------------------------------------------
	# Top: class over lead time
	# -------------------------------------------------------------------------
	present_ids = [cid for cid in ordered_ids if cid in set(df_plot["label"].astype(int).unique())]
	class_to_pos = {cid: i for i, cid in enumerate(present_ids)}

	if not class_pairs.empty:
		class_pairs = class_pairs.copy()
		class_pairs["y_msg"] = class_pairs["label_msg"].map(class_to_pos)
		class_pairs["y_icon"] = class_pairs["label_icon"].map(class_to_pos)

		ax_cls.scatter(class_pairs["lead_hour"], class_pairs["y_msg"], s=52, c="black", alpha=0.75)
		ax_cls.scatter(class_pairs["lead_hour"], class_pairs["y_icon"], s=52, c="black", alpha=0.75)

		# Add hourly mode lines for readability.
		mode_msg = class_pairs.groupby("lead_hour")["label_msg"].agg(lambda x: x.mode().iloc[0]).reset_index()
		mode_icon = class_pairs.groupby("lead_hour")["label_icon"].agg(lambda x: x.mode().iloc[0]).reset_index()
		mode_msg["y"] = mode_msg["label_msg"].map(class_to_pos)
		mode_icon["y"] = mode_icon["label_icon"].map(class_to_pos)
		ax_cls.plot(mode_msg["lead_hour"], mode_msg["y"], color="black", linewidth=2.4, linestyle="solid", label="msg")
		ax_cls.plot(mode_icon["lead_hour"], mode_icon["y"], color="black", linewidth=2.4, linestyle="dashed", label="icon")

	# y ticks as class short names.
	if present_ids:
		ax_cls.set_yticks(list(range(len(present_ids))))
		ax_cls.set_yticklabels([short_by_id.get(cid, str(cid)) for cid in present_ids], fontsize=13)
	ax_cls.set_ylabel("Assigned class", fontsize=14)
	ax_cls.set_title(f"Class Evolution", fontsize=16, fontweight="bold")
	ax_cls.grid(alpha=0.25)
	ax_cls.legend(loc="lower right", fontsize=12)
	ax_cls.tick_params(axis="x", labelsize=12)
	ax_cls.tick_params(axis="y", labelsize=12)

	# -------------------------------------------------------------------------
	# Middle: cosine distance
	# -------------------------------------------------------------------------
	if not feat_pairs.empty:
		ax_cos.scatter(feat_pairs["lead_hour"], feat_pairs["cosine_distance"], s=36, c="gray", alpha=0.45)
		agg = feat_pairs.groupby("lead_hour")["cosine_distance"].agg(["mean", "std"]).reset_index()
		ax_cos.plot(agg["lead_hour"], agg["mean"], color="black", linewidth=2.2)
		lo = agg["mean"] - agg["std"].fillna(0)
		hi = agg["mean"] + agg["std"].fillna(0)
		ax_cos.fill_between(agg["lead_hour"], lo, hi, color="black", alpha=0.18)
	else:
		ax_cos.text(0.5, 0.5, "Feature CSV pairs not available", ha="center", va="center", fontsize=13, transform=ax_cos.transAxes)
	ax_cos.set_ylabel("Cosine distance", fontsize=14)
	ax_cos.set_title("msg/icon Feature Vectors Cosine Distance", fontsize=16, fontweight="bold")
	ax_cos.grid(alpha=0.25)
	ax_cos.tick_params(axis="x", labelsize=12)
	ax_cos.tick_params(axis="y", labelsize=12)

	# -------------------------------------------------------------------------
	# Bottom: crop strip (msg top row, icon bottom row)
	# -------------------------------------------------------------------------
	img_pairs = select_one_pair_per_hour(class_pairs)
	if img_pairs.empty:
		ax_img.text(0.5, 0.5, "No matched msg/icon crops found for selected day", ha="center", va="center", fontsize=13, transform=ax_img.transAxes)
	else:
		img_pairs = img_pairs.sort_values("lead_hour").iloc[::2].copy()
		for _, row in img_pairs.iterrows():
			h = float(row["lead_hour"])
			add_image_at(ax_img, row.get("img_path_msg", None), h, 1.0, zoom=0.60)
			add_image_at(ax_img, row.get("img_path_icon", None), h, 0.0, zoom=0.60)

	ax_img.set_ylim(-0.6, 1.6)
	ax_img.set_facecolor("#e6f2ff")
	ax_img.set_yticks([0.0, 1.0])
	ax_img.set_yticklabels(["icon", "msg"], fontsize=14)
	ax_img.set_title("Image Crops", fontsize=16, fontweight="bold")
	#ax_img.set_ylabel("Crops", fontsize=14)
	ax_img.grid(alpha=0.15, axis="x")

	# Shared x-axis lead-time formatting.
	lead_ticks = list(range(1, 24))
	lead_labels = [f"{h:02d}" for h in range(1, 24)] #+ ["00"]
	ax_img.set_xlim(0., 24.)
	ax_img.set_xticks(lead_ticks)
	ax_img.set_xticklabels(lead_labels, fontsize=14)
	ax_img.set_xlabel("Lead time hour", fontsize=14)
	ax_img.tick_params(axis="x", labelsize=13)

	fig.suptitle(f"Case Study Evolution - {SELECTED_DAY} ({SELECTED_REGION})", fontsize=18, fontweight="bold")
	plt.tight_layout(rect=[0, 0, 1, 0.97])
	plt.savefig(out_path, dpi=300)
	plt.close()


# ============================================================================
# MAIN
# ============================================================================
def main() -> None:
	if not os.path.exists(MERGED_TSNE_CSV):
		raise FileNotFoundError(f"Input CSV not found: {MERGED_TSNE_CSV}")

	print(f"Loading merged CSV: {MERGED_TSNE_CSV}")
	df = pd.read_csv(MERGED_TSNE_CSV, low_memory=False)
	print(f"Loaded shape: {df.shape}")

	required = ["tsne_dim_1", "tsne_dim_2", "vector_type", "label", "filename"]
	missing = [c for c in required if c not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns: {missing}")

	df["split"] = df["vector_type"].astype(str).str.upper()
	df["source_type"] = df["filename"].apply(infer_source_from_filename)
	df["datetime"] = df["filename"].apply(parse_datetime_from_filename)
	df["region"] = df["filename"].apply(extract_region_from_filename)
	df["label"] = pd.to_numeric(df["label"], errors="coerce")
	df = df.dropna(subset=["tsne_dim_1", "tsne_dim_2", "label"]).copy()
	df["label"] = df["label"].astype(int)

	# Full dataset subset for background + centroids.
	df_all = df[df["split"].isin(["TRAIN", "CENTROID", "TEST"])].copy()

	# Selected day TEST rows used for trajectories and evolution.
	df_day = df_all[(df_all["split"] == "TEST") & df_all["datetime"].notna()].copy()
	df_day = df_day[df_day["datetime"].dt.strftime("%Y-%m-%d") == SELECTED_DAY].copy()
	df_day = df_day[df_day["region"] == SELECTED_REGION].copy()
	df_day = df_day[df_day["source_type"].isin(["msg", "icon"])].copy()
	df_day["lead_hour"] = to_lead_hour(df_day["datetime"])

	if df_day.empty:
		raise RuntimeError(f"No TEST msg/icon rows found for selected day {SELECTED_DAY}.")

	print("Selected-day TEST counts by source:")
	print(df_day["source_type"].value_counts())

	out1 = os.path.join(OUT_DIR, f"feature_space_trajectory_{SELECTED_DAY}.png")
	out2 = os.path.join(OUT_DIR, f"case_study_evolution_{SELECTED_DAY}.png")

	plot_feature_space_trajectory(df_all=df_all, df_day_test=df_day, out_path=out1)
	plot_evolution_and_crops(df_day_test=df_day, out_path=out2)

	print("Done.")
	print(f"Saved: {out1}")
	print(f"Saved: {out2}")


if __name__ == "__main__":
	main()
