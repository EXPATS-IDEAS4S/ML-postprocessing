#!/usr/bin/env python3
"""Analyse TeamX TEST msg/icon consistency for class labels and feature similarity.

This script:
1) Loads TEST rows from the feature CSV.
2) Adds region (east/central/west) from filename.
3) Matches msg and icon rows by (datetime, region).
4) Computes:
   - class occurrence (msg vs icon)
   - overall msg-vs-icon classification accuracy (msg as reference)
   - diurnal accuracy (lead time 01..00)
   - cosine similarity / cosine distance by lead time
   - focused EC/DC/OA analyses overall, by lead time, and by region
5) Saves plots and summary tables.
"""

import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

import sys

sys.path.append('/home/Daniele/codes/VISSL_postprocessing/')
from utils.plotting.class_colors import CLOUD_CLASS_INFO


# ============================================================================
# CONFIG
# ============================================================================
RUN_NAME = "dcv2_resnet_k7_ir108_100x100_2013-2020_1xrandomcrops_1xtimestamp_cma_nc"
BASE_DIR = f"/data1/fig/{RUN_NAME}/epoch_800/teamx_test"
CSV_FILE = os.path.join(
	BASE_DIR,
	f"features_train_test_{RUN_NAME}_2nd_labels.csv",
)

OUT_DIR = os.path.join(BASE_DIR, "msg_icon_comparison")
os.makedirs(OUT_DIR, exist_ok=True)

FOCUS_SHORT_CLASSES = ["EC", "DC", "OA"]


# ============================================================================
# HELPERS
# ============================================================================
def parse_datetime_mixed(val) -> pd.Timestamp:
	"""Parse datetime from filename (preferred) or fallback mixed datetime strings.

	Supported filename patterns:
	- MSG_108_cma_YYYY-MM-DD_HH_region.nc
	- ICON500m_IR10.8_CLCT_YYYYMMDD_HH_YYYY-MM-DD_HH_region.nc
	  (for ICON, the second datetime is used as forecast/valid time)
	"""
	if pd.isna(val):
		return pd.NaT

	s = str(val).strip()
	base = os.path.splitext(os.path.basename(s))[0]

	# MSG pattern: MSG_108_cma_YYYY-MM-DD_HH_region.nc
	m_msg = re.search(
		r"^MSG_108_cma_(\d{4}-\d{2}-\d{2})_(\d{2})_(east|central|west)$",
		base,
		re.IGNORECASE,
	)
	if m_msg:
		date_str, hour_str = m_msg.group(1), m_msg.group(2)
		return pd.to_datetime(f"{date_str}_{hour_str}", format="%Y-%m-%d_%H", errors="coerce")

	# ICON pattern: ICON500m_IR10.8_CLCT_YYYYMMDD_HH_YYYY-MM-DD_HH_region.nc
	# Use the second datetime (forecast/valid time).
	m_icon = re.search(
		r"^ICON500m_IR10\.8_CLCT_(\d{8})_(\d{2})_(\d{4}-\d{2}-\d{2})_(\d{2})_(east|central|west)$",
		base,
		re.IGNORECASE,
	)
	if m_icon:
		date_str, hour_str = m_icon.group(3), m_icon.group(4)
		return pd.to_datetime(f"{date_str}_{hour_str}", format="%Y-%m-%d_%H", errors="coerce")

	# Pattern: 20130401_0000 or 20130401_2400 (24:00 = next-day 00:00)
	m = re.search(r"(\d{8})_(\d{4})", s)
	if m:
		date_str, time_str = m.group(1), m.group(2)
		if time_str == "2400":
			base = pd.to_datetime(date_str, format="%Y%m%d", errors="coerce")
			return pd.NaT if pd.isna(base) else base + pd.Timedelta(days=1)
		return pd.to_datetime(f"{date_str}_{time_str}", format="%Y%m%d_%H%M", errors="coerce")

	# Fallback to pandas parser for standard timestamps.
	return pd.to_datetime(s, errors="coerce")


def extract_lead_hour_from_path(path_val: str):
	"""Extract lead hour from filename.

	For ICON filenames, uses the second datetime hour (forecast/valid time).
	Hour 00 is mapped to 24.
	"""
	if not isinstance(path_val, str):
		return np.nan

	base = os.path.splitext(os.path.basename(path_val))[0]

	m_msg = re.search(
		r"^MSG_108_cma_(\d{4}-\d{2}-\d{2})_(\d{2})_(east|central|west)$",
		base,
		re.IGNORECASE,
	)
	if m_msg:
		hour = int(m_msg.group(2))
		return 24 if hour == 0 else hour

	m_icon = re.search(
		r"^ICON500m_IR10\.8_CLCT_(\d{8})_(\d{2})_(\d{4}-\d{2}-\d{2})_(\d{2})_(east|central|west)$",
		base,
		re.IGNORECASE,
	)
	if m_icon:
		hour = int(m_icon.group(4))
		return 24 if hour == 0 else hour

	return np.nan


def extract_region_from_path(path_val: str) -> str:
	"""Extract east/central/west from filename suffix."""
	if not isinstance(path_val, str):
		return "unknown"

	stem = os.path.splitext(os.path.basename(path_val))[0].lower()
	m = re.search(r"(?:_|-)(east|central|west)$", stem)
	if m:
		return m.group(1)
	return "unknown"


def rowwise_cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
	"""Compute row-wise cosine similarity between two matrices (N, D)."""
	if a.shape != b.shape:
		raise ValueError(f"Input shapes must match for row-wise similarity: {a.shape} vs {b.shape}")

	# cosine_similarity returns the full pairwise matrix; diagonal gives aligned row pairs.
	return np.diag(cosine_similarity(a, b))


def setup_class_metadata() -> Dict[str, object]:
	"""Build ordered class metadata and lookup maps from CLOUD_CLASS_INFO."""
	items = sorted(CLOUD_CLASS_INFO.items(), key=lambda x: x[1]["order"])
	labels_ordered = [cid for cid, _ in items]
	short_names = {cid: info["short"] for cid, info in items}
	long_names = {cid: info["name"] for cid, info in items}
	colors = {cid: info["color"] for cid, info in items}
	id_by_short = {info["short"]: cid for cid, info in items}

	return {
		"items": items,
		"labels_ordered": labels_ordered,
		"short_names": short_names,
		"long_names": long_names,
		"colors": colors,
		"id_by_short": id_by_short,
	}


def lead_ticks_and_labels() -> Tuple[List[int], List[str]]:
	ticks = list(range(1, 24))
	labels = [f"{h:02d}" for h in range(1, 24)]
	return ticks, labels


def add_lead_hour(df: pd.DataFrame, dt_col: str = "datetime") -> pd.DataFrame:
	out = df.copy()
	out["lead_hour"] = out[dt_col].dt.hour
	out.loc[out["lead_hour"] == 0, "lead_hour"] = 24
	return out


def aggregate_mean_std(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
	return (
		df.groupby(group_col)[value_col]
		.agg(["mean", "std", "count"])
		.reset_index()
		.sort_values(group_col)
	)


def aggregate_quantiles(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
	"""Aggregate median and interquartile range statistics by lead time."""
	if df.empty:
		return pd.DataFrame(columns=[group_col, "median", "q25", "q75", "count"])

	grp = df.groupby(group_col)[value_col]
	out = pd.DataFrame({
		group_col: grp.mean().index,
		"median": grp.median().values,
		"q25": grp.quantile(0.25).values,
		"q75": grp.quantile(0.75).values,
		"count": grp.count().values,
	})
	return out.sort_values(group_col).reset_index(drop=True)


# ============================================================================
# PLOTS
# ============================================================================
def plot_class_occurrence(test_df: pd.DataFrame, meta: Dict[str, object], out_path: str) -> None:
	labels_ordered = meta["labels_ordered"]
	short_names = meta["short_names"]
	colors = meta["colors"]

	msg_counts = (
		test_df[test_df["vector_type"] == "msg"]["label"]
		.value_counts()
		.reindex(labels_ordered, fill_value=0)
	)
	icon_counts = (
		test_df[test_df["vector_type"] == "icon"]["label"]
		.value_counts()
		.reindex(labels_ordered, fill_value=0)
	)

	x = np.arange(len(labels_ordered))
	width = 0.38

	plt.figure(figsize=(8, 3.6))
	plt.bar(
		x - width / 2,
		msg_counts.values,
		width=width,
		color=[colors[c] for c in labels_ordered],
		edgecolor="black",
		linewidth=0.5,
		label="msg",
	)
	plt.bar(
		x + width / 2,
		icon_counts.values,
		width=width,
		color=[colors[c] for c in labels_ordered],
		hatch="///",
		edgecolor="black",
		linewidth=0.5,
		alpha=0.9,
		label="icon",
	)

	plt.xticks(x, [short_names[c] for c in labels_ordered], fontsize=13)
	plt.ylabel("Count", fontsize=13)
	plt.title("TEST Class Occurrence: msg vs icon", fontsize=14, fontweight="bold")
	plt.grid(axis="y", alpha=0.3)
	plt.legend(fontsize=11)
	plt.tight_layout()
	plt.savefig(out_path, dpi=300)
	plt.close()


def plot_accuracy_diurnal(acc_df: pd.DataFrame, title: str, out_path: str) -> None:
	ticks, tick_labels = lead_ticks_and_labels()

	y = np.full(len(ticks), np.nan)
	lead_to_acc = dict(zip(acc_df["lead_hour"].astype(int), acc_df["accuracy"]))
	for i, t in enumerate(ticks):
		y[i] = lead_to_acc.get(t, np.nan)

	plt.figure(figsize=(10, 4))
	plt.plot(ticks, y * 100.0, marker="o", linewidth=2)
	plt.ylim(0, 100)
	plt.xlim(1, 23)
	plt.xticks(ticks, tick_labels, fontsize=10)
	plt.ylabel("Accuracy (%)", fontsize=12)
	plt.xlabel("Lead Time Hour", fontsize=12)
	plt.title(title, fontsize=13, fontweight="bold")
	plt.grid(alpha=0.3)
	plt.tight_layout()
	plt.savefig(out_path, dpi=300)
	plt.close()


def plot_distance_diurnal(dist_df: pd.DataFrame, title: str, out_path: str) -> None:
	ticks, tick_labels = lead_ticks_and_labels()
	mean_arr = np.full(len(ticks), np.nan)
	std_arr = np.full(len(ticks), np.nan)

	idx = dist_df["lead_hour"].astype(int).values
	for i, t in enumerate(ticks):
		m = dist_df.loc[idx == t, "mean"]
		s = dist_df.loc[idx == t, "std"]
		if not m.empty:
			mean_arr[i] = float(m.iloc[0])
			std_arr[i] = float(s.iloc[0]) if not s.empty else 0.0

	lo = mean_arr - std_arr
	hi = mean_arr + std_arr

	plt.figure(figsize=(10, 4))
	plt.plot(ticks, mean_arr, marker="o", linewidth=2, label="mean cosine distance")
	plt.fill_between(ticks, lo, hi, alpha=0.2, label="±1 std")
	plt.xlim(1, 23)
	plt.xticks(ticks, tick_labels, fontsize=10)
	plt.ylabel("Cosine Distance (1 - cosine similarity)", fontsize=12)
	plt.xlabel("Lead Time Hour", fontsize=12)
	plt.title(title, fontsize=13, fontweight="bold")
	plt.grid(alpha=0.3)
	plt.legend(fontsize=10)
	plt.tight_layout()
	plt.savefig(out_path, dpi=300)
	plt.close()


def _lead_series_from_df(df: pd.DataFrame, lead_col: str, value_col: str) -> np.ndarray:
	ticks, tick_labels = lead_ticks_and_labels()
	_ = tick_labels
	y = np.full(len(ticks), np.nan)
	if df is None or df.empty:
		return y

	lead_to_val = dict(zip(df[lead_col].astype(int), df[value_col]))
	for i, t in enumerate(ticks):
		y[i] = lead_to_val.get(t, np.nan)
	return y


def plot_accuracy_combined_with_occurrence(
	acc_all_df: pd.DataFrame,
	acc_focus_df: pd.DataFrame,
	acc_focus_class_df: Dict[str, pd.DataFrame],
	occ_focus_by_lead: pd.DataFrame,
	focus_colors: Dict[int, str],
	focus_short_names: Dict[int, str],
	out_path: str,
) -> None:
	"""Combined accuracy plot with per-class lines and EC/DC/OA occurrence bars."""
	ticks, tick_labels = lead_ticks_and_labels()

	acc_all = _lead_series_from_df(acc_all_df, "lead_hour", "accuracy") * 100.0
	acc_focus = _lead_series_from_df(acc_focus_df, "lead_hour", "accuracy") * 100.0

	plt.figure(figsize=(10, 4.5))
	ax1 = plt.gca()
	ax2 = ax1.twinx()

	# Stacked bars on the secondary axis (occurrence counts).
	bottom = np.zeros(len(ticks), dtype=float)
	for class_id in occ_focus_by_lead.columns:
		vals = occ_focus_by_lead[class_id].reindex(ticks, fill_value=0).values
		ax2.bar(
			ticks,
			vals,
			bottom=bottom,
			color=focus_colors.get(class_id, "gray"),
			alpha=0.28,
			width=0.75,
			edgecolor="none",
			label=f"Occ {focus_short_names.get(class_id, str(class_id))}",
		)
		bottom += vals

	# Accuracy lines on the main axis.
	ax1.plot(ticks, acc_all, linewidth=2.4, color="black", label="Acc All")
	# ax1.plot(
	# 	ticks,
	# 	acc_focus,
	# 	linewidth=2.4,
	# 	color="#d62728",
	# 	label="Acc EC/DC/OA",
	# )

	for short_name, class_df in acc_focus_class_df.items():
		class_id = None
		for cid, sname in focus_short_names.items():
			if sname == short_name:
				class_id = cid
				break
		acc_class = _lead_series_from_df(class_df, "lead_hour", "accuracy") * 100.0
		ax1.plot(
			ticks,
			acc_class,
			linewidth=2.0,
			linestyle="--",
			color=focus_colors.get(class_id, "gray"),
			label=f"Acc {short_name}",
		)

	ax1.set_ylim(0, 100)
	ax1.set_xlim(ticks[0] - 0.5, ticks[-1] + 0.5)
	ax1.set_ylabel("Accuracy (%)", fontsize=13)
	ax2.set_ylabel("Occurrence Count (msg EC/DC/OA)", fontsize=13)
	ax1.set_xlabel("Lead Time Hour", fontsize=13)
	ax1.set_xticks(ticks, fontsize=13)
	ax1.set_xticklabels(tick_labels, fontsize=13)
	ax1.grid(alpha=0.28)
	ax1.set_title(
		"Accuracy vs Lead Time with EC/DC/OA Occurrence",
		fontsize=14,
		fontweight="bold",
	)

	h1, l1 = ax1.get_legend_handles_labels()
	h2, l2 = ax2.get_legend_handles_labels()
	ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=10)

	plt.tight_layout()
	plt.savefig(out_path, dpi=300)
	plt.close()


def plot_confusion_matrix_heatmap(
	cm: np.ndarray,
	class_labels: List[str],
	title: str,
	out_path: str,
) -> None:
	"""Plot row-normalized confusion matrix heatmap with percentage annotations."""
	plt.figure(figsize=(6.5, 5.5))
	ax = plt.gca()
	im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
	cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
	cbar.set_label("Row-normalized Fraction", fontsize=14)
	cbar.ax.tick_params(labelsize=13)

	ax.set(
		xticks=np.arange(len(class_labels)),
		yticks=np.arange(len(class_labels)),
		xticklabels=class_labels,
		yticklabels=class_labels,
		ylabel="True label (msg)",
		xlabel="Predicted label (icon)",
		title=title,
	)
	ax.set_xlabel("Predicted label (icon)", fontsize=15)
	ax.set_ylabel("True label (msg)", fontsize=15)
	ax.set_title(title, fontsize=17, fontweight="bold")
	ax.tick_params(axis="both", which="major", labelsize=13)
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

	threshold = cm.max() / 2.0 if cm.size > 0 else 0
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(
				j,
				i,
				f"{cm[i, j]:.1f}",
				ha="center",
				va="center",
				color="white" if cm[i, j] > threshold else "black",
				fontsize=12,
			)

	plt.tight_layout()
	plt.savefig(out_path, dpi=300)
	plt.close()


def plot_distance_four_lines_quantiles(
	dist_overall_q: pd.DataFrame,
	dist_class_q: Dict[str, pd.DataFrame],
	class_line_colors: Dict[str, str],
	out_path: str,
) -> None:
	"""Distance plot with 4 median lines (overall, EC, DC, OA) and IQR bands."""
	ticks, tick_labels = lead_ticks_and_labels()

	plt.figure(figsize=(10, 4.5))
	ax = plt.gca()

	def _plot_one(label: str, stats_df: pd.DataFrame, color: str) -> None:
		med = _lead_series_from_df(stats_df, "lead_hour", "median")
		q25 = _lead_series_from_df(stats_df, "lead_hour", "q25")
		q75 = _lead_series_from_df(stats_df, "lead_hour", "q75")
		ax.plot(ticks, med, linewidth=2.2, color=color, label=f"{label} median")
		ax.fill_between(ticks, q25, q75, color=color, alpha=0.13, label=f"{label} IQR")

	_plot_one("Overall", dist_overall_q, "black")
	for short_name in ["EC", "DC", "OA"]:
		_plot_one(short_name, dist_class_q.get(short_name, pd.DataFrame()), class_line_colors.get(short_name, "gray"))

	ax.set_ylabel("Cosine Distance", fontsize=13)
	ax.set_xlabel("Lead Time Hour", fontsize=13)
	ax.set_xlim(1, 23)
	ax.set_xticks(ticks, fontsize=12)
	ax.set_xticklabels(tick_labels, fontsize=12)
	ax.grid(alpha=0.28)
	ax.set_title(
		"Cosine Distance vs Lead Time: Overall + EC/DC/OA",
		fontsize=14,
		fontweight="bold",
	)
	ax.legend(loc="upper left", fontsize=10, ncol=2)

	plt.tight_layout()
	plt.savefig(out_path, dpi=300)
	plt.close()


# ============================================================================
# MAIN
# ============================================================================
def main():
	if not os.path.exists(CSV_FILE):
		raise FileNotFoundError(f"Input CSV not found: {CSV_FILE}")

	meta = setup_class_metadata()
	id_by_short = meta["id_by_short"]
	colors = meta["colors"]
	short_names = meta["short_names"]
	focus_ids = [id_by_short[s] for s in FOCUS_SHORT_CLASSES if s in id_by_short]
	focus_colors = {cid: colors[cid] for cid in focus_ids}
	focus_short_names = {cid: short_names[cid] for cid in focus_ids}

	print(f"Loading CSV: {CSV_FILE}")
	df = pd.read_csv(CSV_FILE, low_memory=False)
	print(f"Loaded shape: {df.shape}")

	# Keep TEST only.
	df["split"] = df["split"].astype(str).str.upper()
	test_df = df[df["split"] == "TEST"].copy()
	print(f"TEST rows: {len(test_df)}")

	# Keep only msg/icon vectors.
	test_df["vector_type"] = test_df["vector_type"].astype(str).str.lower()
	test_df = test_df[test_df["vector_type"].isin(["msg", "icon"])].copy()
	print("vector_type counts:")
	print(test_df["vector_type"].value_counts())

	# Parse datetime from filename/path.
	test_df["datetime"] = test_df["path"].apply(parse_datetime_mixed)
	test_df["region"] = test_df["path"].apply(extract_region_from_path)
	test_df = test_df.dropna(subset=["datetime"]).copy()

	# Enforce lead-hour from datetime: 00 is represented as hour 24.
	test_df["lead_hour"] = test_df["datetime"].dt.hour.astype(int)
	test_df.loc[test_df["lead_hour"] == 0, "lead_hour"] = 24

	# Remove invalid labels.
	test_df = test_df[test_df["label"] != -100].copy()

	# Plot 1: class occurrence msg vs icon.
	plot_class_occurrence(
		test_df,
		meta,
		os.path.join(OUT_DIR, "class_occurrence_msg_vs_icon.png"),
	)

	# Prepare robust one-to-one matching for msg/icon using datetime+region.
	feature_cols = [c for c in test_df.columns if c.startswith("dim_")]

	# If duplicates exist for same key and vector_type, keep highest distance.
	if "distance" in test_df.columns:
		test_df = test_df.sort_values("distance", ascending=False)

	test_df = test_df.drop_duplicates(
		subset=["datetime", "region", "vector_type"], keep="first"
	)

	msg_cols = ["datetime", "region", "lead_hour", "label"] + feature_cols
	icon_cols = ["datetime", "region", "lead_hour", "label"] + feature_cols

	msg_df = test_df[test_df["vector_type"] == "msg"][msg_cols].copy()
	icon_df = test_df[test_df["vector_type"] == "icon"][icon_cols].copy()

	msg_df = msg_df.rename(columns={"label": "label_msg", "lead_hour": "lead_hour_msg"})
	icon_df = icon_df.rename(columns={"label": "label_icon", "lead_hour": "lead_hour_icon"})

	pairs = msg_df.merge(icon_df, on=["datetime", "region"], suffixes=("_msg", "_icon"), how="inner")

	if pairs.empty:
		raise RuntimeError("No msg/icon matched pairs found by (datetime, region).")

	# Enforce lead-hour from matched datetime: 00 is represented as hour 24.
	pairs["lead_hour"] = pairs["datetime"].dt.hour.astype(int)
	pairs.loc[pairs["lead_hour"] == 0, "lead_hour"] = 24

	# Classification accuracy (msg as reference truth).
	pairs["correct"] = (pairs["label_icon"].astype(int) == pairs["label_msg"].astype(int)).astype(int)
	overall_acc = pairs["correct"].mean()

	# Feature cosine similarity and distance.
	feat_msg = pairs[[f"{c}_msg" for c in feature_cols]].values.astype(np.float64)
	feat_icon = pairs[[f"{c}_icon" for c in feature_cols]].values.astype(np.float64)
	pairs["cosine_similarity"] = rowwise_cosine_similarity(feat_msg, feat_icon)
	pairs["cosine_distance"] = 1.0 - pairs["cosine_similarity"]

	# Save matched pairs table.
	pairs_out_cols = [
		"datetime",
		"region",
		"lead_hour",
		"label_msg",
		"label_icon",
		"correct",
		"cosine_similarity",
		"cosine_distance",
	]
	pairs[pairs_out_cols].to_csv(os.path.join(OUT_DIR, "matched_msg_icon_pairs.csv"), index=False)

	# Overall and diurnal accuracy.
	acc_by_lead = (
		pairs.groupby("lead_hour")["correct"].mean().reset_index().rename(columns={"correct": "accuracy"})
	)
	dist_by_lead = aggregate_mean_std(pairs, "lead_hour", "cosine_distance")
	dist_overall_q = aggregate_quantiles(pairs, "lead_hour", "cosine_distance")

	# Focus classes (EC, DC, OA) using msg label as truth class.
	focus_pairs = pairs[pairs["label_msg"].isin(focus_ids)].copy()
	focus_overall_acc = focus_pairs["correct"].mean() if not focus_pairs.empty else np.nan
	acc_focus_by_lead = (
		focus_pairs.groupby("lead_hour")["correct"]
		.mean()
		.reset_index()
		.rename(columns={"correct": "accuracy"})
	)
	acc_focus_each_by_lead: Dict[str, pd.DataFrame] = {}
	for short_name, class_id in zip(FOCUS_SHORT_CLASSES, focus_ids):
		class_pairs = pairs[pairs["label_msg"] == class_id].copy()
		acc_focus_each_by_lead[short_name] = (
			class_pairs.groupby("lead_hour")["correct"]
			.mean()
			.reset_index()
			.rename(columns={"correct": "accuracy"})
		)
	dist_focus_by_lead = aggregate_mean_std(focus_pairs, "lead_hour", "cosine_distance")

	# Per-class quantile stats for EC, DC, OA (msg reference class).
	dist_class_quantiles = {}
	for short_name, class_id in zip(FOCUS_SHORT_CLASSES, focus_ids):
		class_pairs = pairs[pairs["label_msg"] == class_id]
		dist_class_quantiles[short_name] = aggregate_quantiles(class_pairs, "lead_hour", "cosine_distance")

	# Stacked occurrence of EC/DC/OA by lead hour (msg reference class).
	if focus_pairs.empty:
		occ_focus_by_lead = pd.DataFrame(index=list(range(1, 25)), columns=focus_ids).fillna(0)
	else:
		occ_focus_by_lead = (
			focus_pairs.groupby(["lead_hour", "label_msg"]).size().unstack(fill_value=0)
		)
		for cid in focus_ids:
			if cid not in occ_focus_by_lead.columns:
				occ_focus_by_lead[cid] = 0
		occ_focus_by_lead = occ_focus_by_lead[focus_ids]

	# Combined lead-time plots with 01..23 and 00 (next day) as final point.
	plot_accuracy_combined_with_occurrence(
		acc_all_df=acc_by_lead,
		acc_focus_df=acc_focus_by_lead,
		acc_focus_class_df=acc_focus_each_by_lead,
		occ_focus_by_lead=occ_focus_by_lead,
		focus_colors=focus_colors,
		focus_short_names=focus_short_names,
		out_path=os.path.join(OUT_DIR, "accuracy_diurnal_combined_with_focus_occurrence.png"),
	)

	# Confusion matrices: all classes and convective subset.
	labels_all = meta["labels_ordered"]
	labels_all_short = [short_names[cid] for cid in labels_all]
	cm_all = confusion_matrix(
		pairs["label_msg"].astype(int),
		pairs["label_icon"].astype(int),
		labels=labels_all,
	)
	cm_all_row_sums = cm_all.sum(axis=1, keepdims=True)
	cm_all_norm = np.divide(cm_all, cm_all_row_sums, out=np.zeros_like(cm_all, dtype=float), where=cm_all_row_sums != 0)
	plot_confusion_matrix_heatmap(
		cm=cm_all_norm,
		class_labels=labels_all_short,
		title="Confusion Matrix - All Classes",
		out_path=os.path.join(OUT_DIR, "confusion_matrix_all_classes.png"),
	)
	pd.DataFrame(cm_all, index=labels_all_short, columns=labels_all_short).to_csv(
		os.path.join(OUT_DIR, "confusion_matrix_all_classes.csv")
	)
	pd.DataFrame(cm_all_norm, index=labels_all_short, columns=labels_all_short).to_csv(
		os.path.join(OUT_DIR, "confusion_matrix_all_classes_row_normalized.csv")
	)

	labels_focus = focus_ids
	labels_focus_short = [short_names[cid] for cid in labels_focus]
	focus_cm_pairs = pairs[
		pairs["label_msg"].isin(labels_focus) & pairs["label_icon"].isin(labels_focus)
	].copy()
	cm_focus = confusion_matrix(
		focus_cm_pairs["label_msg"].astype(int),
		focus_cm_pairs["label_icon"].astype(int),
		labels=labels_focus,
	)
	cm_focus_row_sums = cm_focus.sum(axis=1, keepdims=True)
	cm_focus_norm = np.divide(cm_focus, cm_focus_row_sums, out=np.zeros_like(cm_focus, dtype=float), where=cm_focus_row_sums != 0)
	plot_confusion_matrix_heatmap(
		cm=cm_focus_norm,
		class_labels=labels_focus_short,
		title="Confusion Matrix - EC/DC/OA",
		out_path=os.path.join(OUT_DIR, "confusion_matrix_convective_classes.png"),
	)
	pd.DataFrame(cm_focus, index=labels_focus_short, columns=labels_focus_short).to_csv(
		os.path.join(OUT_DIR, "confusion_matrix_convective_classes.csv")
	)
	pd.DataFrame(cm_focus_norm, index=labels_focus_short, columns=labels_focus_short).to_csv(
		os.path.join(OUT_DIR, "confusion_matrix_convective_classes_row_normalized.csv")
	)

	class_line_colors = {
		"EC": focus_colors.get(focus_ids[0], "#1f77b4") if len(focus_ids) > 0 else "#1f77b4",
		"DC": focus_colors.get(focus_ids[1], "#ff7f0e") if len(focus_ids) > 1 else "#ff7f0e",
		"OA": focus_colors.get(focus_ids[2], "#2ca02c") if len(focus_ids) > 2 else "#2ca02c",
	}
	plot_distance_four_lines_quantiles(
		dist_overall_q=dist_overall_q,
		dist_class_q=dist_class_quantiles,
		class_line_colors=class_line_colors,
		out_path=os.path.join(OUT_DIR, "cosine_distance_diurnal_overall_EC_DC_OA_quantiles.png"),
	)

	if not focus_pairs.empty:
		acc_focus_by_lead.to_csv(os.path.join(OUT_DIR, "accuracy_by_lead_focus_EC_DC_OA.csv"), index=False)
		dist_focus_by_lead.to_csv(os.path.join(OUT_DIR, "cosine_distance_by_lead_focus_EC_DC_OA.csv"), index=False)
		for short_name, acc_df in acc_focus_each_by_lead.items():
			acc_df.to_csv(os.path.join(OUT_DIR, f"accuracy_by_lead_{short_name}.csv"), index=False)

	# Save summary metrics.
	summary_lines = [
		f"Input CSV: {CSV_FILE}",
		f"TEST rows after filtering msg/icon and valid datetime: {len(test_df)}",
		f"Matched msg/icon pairs: {len(pairs)}",
		f"Overall accuracy (msg as reference): {overall_acc:.4f}",
		f"Focus classes: {FOCUS_SHORT_CLASSES} -> IDs {focus_ids}",
		f"Focus pairs count: {len(focus_pairs)}",
		(
			f"Focus overall accuracy (msg as reference): {focus_overall_acc:.4f}"
			if not np.isnan(focus_overall_acc)
			else "Focus overall accuracy (msg as reference): NaN"
		),
		"Region counts in matched pairs:",
		pairs["region"].value_counts().to_string(),
	]

	with open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
		f.write("\n".join(summary_lines))

	# Save per-lead tables.
	acc_by_lead.to_csv(os.path.join(OUT_DIR, "accuracy_by_lead_all_classes.csv"), index=False)
	dist_by_lead.to_csv(os.path.join(OUT_DIR, "cosine_distance_by_lead_all_classes.csv"), index=False)
	dist_overall_q.to_csv(os.path.join(OUT_DIR, "cosine_distance_quantiles_by_lead_overall.csv"), index=False)
	for short_name in FOCUS_SHORT_CLASSES:
		dist_class_quantiles.get(
			short_name,
			pd.DataFrame(columns=["lead_hour", "median", "q25", "q75", "count"]),
		).to_csv(
			os.path.join(OUT_DIR, f"cosine_distance_quantiles_by_lead_{short_name}.csv"),
			index=False,
		)
	occ_focus_by_lead.to_csv(os.path.join(OUT_DIR, "occurrence_focus_classes_by_lead.csv"))

	print("Done.")
	print(f"Results saved in: {OUT_DIR}")


if __name__ == "__main__":
	main()

