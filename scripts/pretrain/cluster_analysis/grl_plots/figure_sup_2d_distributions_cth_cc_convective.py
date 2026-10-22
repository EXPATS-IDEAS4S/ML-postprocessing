"""Plot normalized cloud-cover/CTH distributions for selected classes."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
	import cmcrameri.cm as cmc
except ImportError:  # pragma: no cover - optional dependency
	cmc = None

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
	sys.path.append(str(REPO_ROOT))

from utils.configs import load_config


CONFIG_PATH = REPO_ROOT / "configs" / "process_run_GRL.yaml"
CONVECTIVE_CLASSES = [6, 5, 3, 9]
OTHER_CLASSES = [0, 1, 2, 4, 7, 8]
# (classes, subplot rows, subplot cols, output filename, figure title)
FIGURE_GROUPS = [
	(CONVECTIVE_CLASSES, 2, 2, "figure_sup_2d_distributions_cth_cc_convective.png", "Normalized cloud-cover / CTH distributions: convective classes"),
	(OTHER_CLASSES, 2, 3, "figure_sup_2d_distributions_cth_cc_other_classes.png", "Normalized cloud-cover / CTH distributions: other classes"),
]
PANEL_SIZE = (9, 7)
CLOUD_COVER_BINS = np.linspace(0, 100, 61)
CTH_BINS = np.linspace(0, 14000, 71)
LIGHTNING_COLORMAP = getattr(cmc, "lipari_r", plt.get_cmap("magma")) if cmc is not None else plt.get_cmap("magma")


def load_distribution_data(classes):
	config = load_config(str(CONFIG_PATH))
	csv_path = Path(config["output_files"]["training_video_summary_temporal"])
	data = pd.read_csv(csv_path, usecols=["label", "cma_mean", "cth_mean"])
	data["label"] = pd.to_numeric(data["label"], errors="coerce")
	data["cma_mean"] = pd.to_numeric(data["cma_mean"], errors="coerce")
	data["cth_mean"] = pd.to_numeric(data["cth_mean"], errors="coerce")
	data = data[data["label"].isin(classes)].dropna(
		subset=["cma_mean", "cth_mean"]
	)
	data = data[
		data["cma_mean"].between(0, 1) & data["cth_mean"].between(0, 14000)
	].copy()
	return data, Path(config["output_files"]["figures_dir"])


def enclosed_probability_threshold(probability_mass, enclosed_probability):
	sorted_mass = np.sort(probability_mass.ravel())[::-1]
	cumulative_mass = np.cumsum(sorted_mass)
	threshold_index = min(
		np.searchsorted(cumulative_mass, enclosed_probability), sorted_mass.size - 1
	)
	return sorted_mass[threshold_index]


def plot_distributions(data: pd.DataFrame, classes, n_rows, n_cols, title) -> plt.Figure:
	histograms = {}

	for class_label in classes:
		class_data = data[data["label"] == class_label]
		counts, _, _ = np.histogram2d(
			class_data["cma_mean"].to_numpy() * 100,
			class_data["cth_mean"].to_numpy(),
			bins=(CLOUD_COVER_BINS, CTH_BINS),
		)
		peak_bin = counts.max()
		normalized_density = counts / peak_bin if peak_bin else counts
		histograms[class_label] = (class_data, normalized_density)

	fig_width = PANEL_SIZE[0] * n_cols
	figure, axes = plt.subplots(
		n_rows,
		n_cols,
		figsize=(fig_width, PANEL_SIZE[1] * n_rows),
		sharex=True,
		sharey=True,
		squeeze=False,
	)
	mesh = None
	for axis, class_label in zip(axes.flat, classes):
		class_data, normalized_density = histograms[class_label]
		mesh = axis.pcolormesh(
			CLOUD_COVER_BINS,
			CTH_BINS,
			normalized_density.T,
			shading="flat",
			cmap=LIGHTNING_COLORMAP,
			vmin=0,
			vmax=1,
		)
		axis.set_title(
			f"Class {class_label} (n={len(class_data):,})",
			loc="left",
			fontweight="bold",
			fontsize=27,
		)
		axis.set_xlim(0, 100)
		axis.set_ylim(0, 14000)
		axis.tick_params(axis="both", labelsize=25)
		axis.spines["top"].set_visible(False)
		axis.spines["right"].set_visible(False)

	for axis in axes[-1, :]:
		axis.set_xlabel("Cloud cover (%)", fontsize=26)
	for axis in axes[:, 0]:
		axis.set_ylabel("Cloud-top height (m)", fontsize=26)

	left, right = 2.5 / fig_width, 0.98
	figure.subplots_adjust(left=left, right=right, top=0.90, bottom=0.25, wspace=0.18, hspace=0.24)
	colorbar_width = 8.3 / fig_width
	colorbar_axis = figure.add_axes([left + (right - left - colorbar_width) / 2, 0.105, colorbar_width, 0.035])
	colorbar = figure.colorbar(mesh, cax=colorbar_axis, orientation="horizontal")
	colorbar.set_label("Normalized bin density", fontsize=25, labelpad=12)
	colorbar.ax.tick_params(labelsize=25)
	figure.suptitle(
		title,
		y=0.97,
		fontsize=28,
		fontweight="bold",
	)
	return figure


def main() -> None:
	for classes, n_rows, n_cols, filename, title in FIGURE_GROUPS:
		data, output_dir = load_distribution_data(classes)
		output_dir.mkdir(parents=True, exist_ok=True)
		figure = plot_distributions(data, classes, n_rows, n_cols, title)
		output_path = output_dir / filename
		figure.savefig(output_path, dpi=220, bbox_inches="tight")
		plt.close(figure)
		print(f"Saved normalized distributions to {output_path}")


if __name__ == "__main__":
	main()
