"""Plot distributions of selected video-summary variables for classes 6, 0, and 8."""

import argparse
from pathlib import Path
import sys
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
	sys.path.append(str(REPO_ROOT))

from utils.configs import load_config
from utils.plotting.class_colors import colors_per_class_codes_grl


CONFIG_PATH = REPO_ROOT / "configs" / "process_run_GRL.yaml"
OUTPUT_FILENAME = "figure_sup_distr_intermediate_classes.png"
OUTPUT_FILENAME_CUMULATIVE = "figure_sup_distr_intermediate_classes_cumulative.png"
CLASSES = [6, 0, 8]
DISTRIBUTION_COLUMNS = [
	"cth_mean",
	"cth10plus_mean",
	"cot_mean",
	"cot30plus_mean",
	"prec_fraction_mean",
	"precipitation_p95_mean",
]
FIGURE_SIZE = (18, 10)
LINE_WIDTH = 2.2


def load_distribution_data() -> Tuple[pd.DataFrame, Path]:
	config = load_config(str(CONFIG_PATH))
	csv_path = Path(config["output_files"]["training_video_summary_temporal"])
	data = pd.read_csv(csv_path, usecols=["label", *DISTRIBUTION_COLUMNS])
	data["label"] = pd.to_numeric(data["label"], errors="coerce")
	data = data[data["label"].isin(CLASSES)].copy()
	for column in DISTRIBUTION_COLUMNS:
		data[column] = pd.to_numeric(data[column], errors="coerce")
	return data, Path(config["output_files"]["figures_dir"])


def plot_distributions(data: pd.DataFrame, cumulative: bool = False) -> plt.Figure:
	figure, axes = plt.subplots(2, 3, figsize=FIGURE_SIZE)
	class_colors = {
		class_label: colors_per_class_codes_grl[str(class_label)]
		for class_label in CLASSES
	}

	for axis, column in zip(axes.flat, DISTRIBUTION_COLUMNS):
		all_values = data[column].to_numpy(dtype=float)
		all_values = all_values[np.isfinite(all_values)]
		if all_values.size == 0:
			continue
		bin_edges = np.histogram_bin_edges(all_values, bins=50)

		for class_label in CLASSES:
			values = data.loc[data["label"] == class_label, column].to_numpy(dtype=float)
			values = values[np.isfinite(values)]
			if values.size == 0:
				continue

			axis.hist(
				values,
				bins=bin_edges,
				density=True,
				cumulative=cumulative,
				histtype="step",
				color=class_colors[class_label],
				linewidth=LINE_WIDTH,
			)

		axis.set_title(column, loc="left", fontweight="bold")
		axis.set_xlabel(column)
		if cumulative:
			axis.set_ylim(0, 1.02)
		axis.grid(axis="both", linestyle="--", linewidth=0.6, alpha=0.45)
		axis.spines["top"].set_visible(False)
		axis.spines["right"].set_visible(False)

	for axis in axes[:, 0]:
		axis.set_ylabel("Cumulative proportion" if cumulative else "Probability density")

	legend_handles = [
		Line2D(
			[0],
			[0],
			color=class_colors[class_label],
			linewidth=LINE_WIDTH,
			label=f"Class {class_label}",
		)
		for class_label in CLASSES
	]
	figure.legend(
		handles=legend_handles,
		loc="upper center",
		bbox_to_anchor=(0.5, 0.99),
		ncol=len(CLASSES),
		frameon=False,
	)
	figure.suptitle(
		"Cumulative video-summary variable distributions"
		if cumulative
		else "Video-summary variable distributions",
		y=0.94,
		fontsize=16,
		fontweight="bold",
	)
	figure.tight_layout(rect=(0, 0, 1, 0.91))
	return figure


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Plot distributions of selected video-summary variables by class."
	)
	parser.add_argument(
		"--cumulative",
		action="store_true",
		help="Plot cumulative distributions instead of probability-density histograms.",
	)
	args = parser.parse_args()

	data, output_dir = load_distribution_data()
	output_dir.mkdir(parents=True, exist_ok=True)
	figure = plot_distributions(data, cumulative=args.cumulative)
	output_filename = (
		OUTPUT_FILENAME_CUMULATIVE if args.cumulative else OUTPUT_FILENAME
	)
	output_path = output_dir / output_filename
	figure.savefig(output_path, dpi=220, bbox_inches="tight")
	plt.close(figure)
	print(f"Saved distributions to {output_path}")


if __name__ == "__main__":
	main()


