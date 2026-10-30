
"""Compare class-centroid and video-level PCA for convective-class selection.

The input is the configured ``training_video_summary_temporal`` crops summary,
which contains all nine features below. Unlabeled rows (class ``-100``) are
excluded. For each domain, optional ``log1p`` transforms are applied to
observations first; transformed observations are averaged per class, and the
scaler and two-component PCA are fitted to those class means. Every usable
observation is then projected with that same scaler and PCA.

The structural domain contains cloud cover, cloud-top height, fraction above
10 km, optical thickness, and fraction above optical thickness 30. The
activity domain contains precipitation fraction, accumulated precipitation,
95th-percentile precipitation, and lightning counts. ``log1p`` is applied by
default to the latter three activity features.

The code performs 2 parallel PCA, to answer the following questions:
1. Which cloud classes exhibit convective characteristics? 
Identify the classes characterized by well-developed cloud structures and intense convective activity. 
2. Do the selected convective classes represent physically distinct convective regimes? 
Verify whether the selected classes actually differ in their physical properties.

To identify the classes (task 1),  we use the structure PCA and the avtivity PCA, the first
captures the structural and convective characteristics of each class, 
while the latter captures the convective activity (precipitation and electricity).

We use the diagram structure PC1 and Activity PC1 to identify convective classes, after having 
verified the physical meaning and the orientation of the components using the loadings.

To verify if the classes are different (task 2),  we use the scores of the individual videos that we calculate
For each class, we can represent all videos in the PCA space and analize:
1- separation between centroids of different classes in the PCA space
2- dispersion of individual videos around their class centroids
3- overlap between different classes in the PCA space

To quantify the separation we use the Bhattacharyya affinity, calculated for each pair of classes.
Assuming Gaussian distributions for the class scores, the Bhattacharyya affinity provides a measure of
 the overlap between the distributions of different classes in the PCA space.
B = 0  distributions well separated
B = 0.5 intermediate partial overlap
B = 1  distributions completely overlapping
The code calculaes the B affinity in the combined space of 4 dimentsions 
(structure PC1, structure PC2, Activity PC1, Activity PC2)

We do a PCA on centroids, and a PCA on individual videos

the code produces:
FOr both the analysis of centroids and individual videos, the code produces:
- Loadings and explained variance of the Structure and Activity PCAs.
  Loadings are scaled (eigenvector * sqrt(explained_variance_)), i.e. variable-PC
  correlations times sqrt(n/(n-1)) with n the number of fitted rows (10 class
  means in centroid_fit, so values reach about 1.05).
- PC1–PC2 plots with centroids and covariance ellipses.
- Combined Structure PC1 vs Activity PC1 plot.
- Candidate class tables and fractions of videos in the positive quadrant.
- Bhattacharyya affinity matrices in 2D and 4D, with corresponding heatmaps.
- Heatmap dedicated to candidate classes.
- classes dendrograms.


Run directly with python scripts/pretrain/cluster_analysis/grl_plots/figure_sup1_PCA_convection.py

"""

import sys
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.patches import Ellipse, FancyBboxPatch, Patch
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO_ROOT = (Path(__file__).resolve().parents[4]
             if len(Path(__file__).resolve().parents) > 4
             else Path(__file__).resolve().parent)
if str(REPO_ROOT) not in sys.path:
	sys.path.append(str(REPO_ROOT))

try:
    from utils.configs import load_config
    from utils.plotting.class_colors import colors_per_class_codes_grl
except ImportError:
    load_config = None
    colors_per_class_codes_grl = {}


CONFIG_PATH = REPO_ROOT / "configs" / "process_run_GRL.yaml"
OUTPUT_FILENAME = "figure_sup1_PCA_convection"
CLASS_COLUMN = "label"
GROUPS = {
	"structure": [
		"cma_mean",
		"cth_mean",
		"cth10plus_mean",
		"cot_mean",
		"cot30plus_mean",
	],
	"activity": [
		"prec_fraction_mean",
		"precipitation_mean",
		"precipitation_p95_mean",
		"euclid_msg_grid_mean",
	],
}
LOG1P_VARIABLES = {
	"precipitation_mean",
	"precipitation_p95_mean",
	"euclid_msg_grid_mean",
}
N_CLUSTERS = 3
MIN_COVARIANCE_OBSERVATIONS = 3


def clean_and_transform(data, variables, logs):
	values = data[variables].apply(pd.to_numeric, errors="coerce").astype(float)
	for variable in logs:
		if variable in variables:
			if (values[variable].dropna() < 0).any():
				raise ValueError(f"Negative values in log1p variable {variable}")
			values[variable] = np.log1p(values[variable])
	return values


def fit_project(data, group, variables, logs):
	values = clean_and_transform(data, variables, logs)
	good = pd.Series(np.isfinite(values.to_numpy()).all(axis=1), index=data.index)
	counts = data.loc[good].groupby(CLASS_COLUMN).size().sort_index()
	means = values.loc[good].groupby(data.loc[good, CLASS_COLUMN]).mean().sort_index()
	if len(means) < 3:
		raise ValueError(f"{group}: at least three classes with complete observations are required")

	constant = means.columns[means.std(ddof=0) == 0].tolist()
	if constant:
		raise ValueError(f"Constant variable(s) in {group}; remove them before PCA: {constant}")

	scaler = StandardScaler().fit(means)
	standardized_means = scaler.transform(means)
	pca = PCA(n_components=2).fit(standardized_means)
	orientation = np.ones(2)
	if pca.components_[0].sum() < 0:
		orientation[0] = -1
	components = pca.components_ * orientation[:, None]

	mean_scores = pca.transform(standardized_means) * orientation
	scores = pd.DataFrame(
		mean_scores,
		index=means.index,
		columns=[f"{group}_PC1", f"{group}_PC2"],
	)
	projected = pd.DataFrame(np.nan, index=data.index, columns=scores.columns)
	projected.loc[good] = pca.transform(scaler.transform(values.loc[good])) * orientation
	loadings = pd.DataFrame(
		components.T * np.sqrt(pca.explained_variance_),
		index=variables,
		columns=["PC1", "PC2"],
	)
	return projected, scores, loadings, pca.explained_variance_ratio_, counts, good, means


def class_color(class_id):
	return colors_per_class_codes_grl.get(str(int(class_id)), "#555555")


def ellipse(ax, center, covariance, color, nsig=1.0, alpha=0.15):
	eigenvalues, eigenvectors = np.linalg.eigh(covariance)
	order = np.argsort(eigenvalues)[::-1]
	eigenvalues = np.clip(eigenvalues[order], 0, None)
	major_vector = eigenvectors[:, order[0]]
	angle = np.degrees(np.arctan2(major_vector[1], major_vector[0]))
	patch = Ellipse(
		center,
		2 * nsig * np.sqrt(eigenvalues[0]),
		2 * nsig * np.sqrt(eigenvalues[1]),
		angle=angle,
		facecolor=color,
		edgecolor=color,
		alpha=alpha,
		lw=1.5,
	)
	ax.add_patch(patch)


def class_covariances(projected, score_columns):
	covariances = {}
	for class_id in sorted(projected[CLASS_COLUMN].unique()):
		values = projected.loc[projected[CLASS_COLUMN] == class_id, score_columns].dropna()
		if len(values) >= MIN_COVARIANCE_OBSERVATIONS:
			covariances[class_id] = (len(values), np.cov(values.to_numpy(), rowvar=False))
	return covariances


def save_covariances(covariances, score_columns, output_dir, name):
	for class_id, (_, covariance) in covariances.items():
		pd.DataFrame(covariance, index=score_columns, columns=score_columns).to_csv(
			output_dir / f"{OUTPUT_FILENAME}_{name}_class_{int(class_id)}_covariance.csv"
		)


def plot_domain(projected, scores, covariances, name, output_dir):
	fig, ax = plt.subplots(figsize=(9, 7))
	for class_id in scores.index:
		color = class_color(class_id)
		subset = projected.loc[projected[CLASS_COLUMN] == class_id, scores.columns].dropna()
		if len(subset) > 30000:
			subset = subset.sample(30000, random_state=42)
		ax.scatter(subset.iloc[:, 0], subset.iloc[:, 1], s=2, alpha=0.035, color=color, rasterized=True)
		center = scores.loc[class_id].to_numpy()
		if class_id in covariances:
			ellipse(ax, center, covariances[class_id][1], color)
		ax.scatter(*center, color=color, edgecolor="black", s=55, zorder=4)
		ax.annotate(str(int(class_id)), center, xytext=(5, 5), textcoords="offset points")
	ax.axhline(0, linewidth=0.6, color="gray")
	ax.axvline(0, linewidth=0.6, color="gray")
	ax.set_xlabel(f"{name.title()} PC1")
	ax.set_ylabel(f"{name.title()} PC2")
	ax.set_title(f"{name.title()}: class means and within-class observation dispersion")
	fig.tight_layout()
	fig.savefig(output_dir / f"{OUTPUT_FILENAME}_{name}_dispersion.png", dpi=200)
	plt.close(fig)


def gaussian_bhattacharyya(mean_a, covariance_a, mean_b, covariance_b):
	dimension = len(mean_a)
	scale = max((np.trace(covariance_a) + np.trace(covariance_b)) / (2 * dimension), 1e-8)
	ridge = 1e-6 * scale * np.eye(dimension)
	covariance_a = covariance_a + ridge
	covariance_b = covariance_b + ridge
	average = (covariance_a + covariance_b) / 2
	signs = [np.linalg.slogdet(covariance)[0] for covariance in (covariance_a, covariance_b, average)]
	if min(signs) <= 0:
		return np.nan
	log_determinants = [np.linalg.slogdet(covariance)[1] for covariance in (covariance_a, covariance_b, average)]
	delta = mean_a - mean_b
	distance = 0.125 * delta @ np.linalg.solve(average, delta)
	distance += 0.5 * (log_determinants[2] - 0.5 * (log_determinants[0] + log_determinants[1]))
	return float(np.exp(-max(0, distance)))


def pairwise_affinity(data, score_columns, classes, output_path):
	moments = {}
	for class_id in classes:
		values = data.loc[data[CLASS_COLUMN] == class_id, score_columns].dropna().to_numpy()
		if len(values) >= MIN_COVARIANCE_OBSERVATIONS:
			moments[class_id] = (len(values), values.mean(axis=0), np.cov(values, rowvar=False))

	rows = []
	class_ids = sorted(moments)
	for index, class_a in enumerate(class_ids):
		for class_b in class_ids[index + 1:]:
			count_a, mean_a, covariance_a = moments[class_a]
			count_b, mean_b, covariance_b = moments[class_b]
			rows.append({
				"class_a": class_a,
				"class_b": class_b,
				"observations_a": count_a,
				"observations_b": count_b,
				"gaussian_bhattacharyya_affinity": gaussian_bhattacharyya(
					mean_a, covariance_a, mean_b, covariance_b
				),
			})
	pd.DataFrame(rows, columns=[
		"class_a", "class_b", "observations_a", "observations_b",
		"gaussian_bhattacharyya_affinity",
	]).to_csv(output_path, index=False)


def plot_cluster(scores, output_path, nclusters):
	if not 2 <= nclusters <= len(scores):
		raise ValueError(f"--clusters must be between 2 and {len(scores)}")
	standardized = StandardScaler().fit_transform(scores)
	linked = linkage(standardized, method="ward")
	assignments = fcluster(linked, t=nclusters, criterion="maxclust")
	fig, ax = plt.subplots(figsize=(9, 5))
	dendrogram(linked, labels=[str(class_id) for class_id in scores.index], ax=ax)
	ax.set_xlabel("Cloud class")
	ax.set_ylabel("Ward distance (standardized four-PC scores)")
	ax.set_title("Hierarchical clustering of class centroids")
	fig.tight_layout()
	fig.savefig(output_path, dpi=200)
	plt.close(fig)
	return assignments




def fit_video_pca(data, group, variables, logs):
    """Fit scaler and PCA on individual complete videos (not class means)."""
    values = clean_and_transform(data, variables, logs)
    good = pd.Series(np.isfinite(values.to_numpy()).all(axis=1), index=data.index)
    if good.sum() < 3:
        raise ValueError(f"{group}: fewer than three complete videos")
    scaler = StandardScaler().fit(values.loc[good])
    standardized = scaler.transform(values.loc[good])
    pca = PCA(n_components=2).fit(standardized)
    # Orientation is conventional: choose PC1 with positive sum of coefficients.
    signs = np.ones(2)
    if pca.components_[0].sum() < 0:
        signs[0] = -1
    projected = pd.DataFrame(np.nan, index=data.index,
                             columns=[f"{group}_PC1", f"{group}_PC2"])
    projected.loc[good] = pca.transform(standardized) * signs
    loadings = pd.DataFrame((pca.components_ * signs[:, None]).T *
                            np.sqrt(pca.explained_variance_),
                            index=variables, columns=["PC1", "PC2"])
    return projected, loadings, pca.explained_variance_ratio_, good


def save_heatmap(affinity_csv, classes, target, title):
    df = pd.read_csv(affinity_csv)
    classes = sorted(classes)
    if len(classes) < 2:
        return
    matrix = np.eye(len(classes))
    lookup = {int(c): i for i, c in enumerate(classes)}
    for row in df.itertuples(index=False):
        a, b = int(row.class_a), int(row.class_b)
        if a in lookup and b in lookup:
            matrix[lookup[a], lookup[b]] = matrix[lookup[b], lookup[a]] = row.gaussian_bhattacharyya_affinity
    fig, ax = plt.subplots(figsize=(max(5, len(classes)*.7), max(4, len(classes)*.65)))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(classes)), labels=classes)
    ax.set_yticks(range(len(classes)), labels=classes)
    ax.set_xlabel("Class"); ax.set_ylabel("Class"); ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                    color="white" if matrix[i,j] < .55 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="Gaussian Bhattacharyya affinity")
    fig.tight_layout(); fig.savefig(target, dpi=200); plt.close(fig)


LOADING_LABELS = {
    "cma_mean": "CMA",
    "cth_mean": "CTH",
    "cth10plus_mean": "CTH > 10 km",
    "cot_mean": "COT",
    "cot30plus_mean": "COT > 30",
    "prec_fraction_mean": "Prec. fraction",
    "precipitation_mean": "Mean precip.*",
    "precipitation_p95_mean": "P95 precip.*",
    "euclid_msg_grid_mean": "Lightning*",
}
POSITIVE_COLOR, POSITIVE_DARK = "#A5473E", "#2A1410"
NEGATIVE_COLOR, NEGATIVE_DARK = "#3A5F95", "#141C2E"
LOADING_FULL_SCALE = 1.05
TILE_ASPECT = 0.5
SUMMARY_FONT = 22
TITLE_FONTSIZE = 23


def loading_tile_color(value):
    strength = min(abs(value) / LOADING_FULL_SCALE, 1.0)
    base, dark = (POSITIVE_COLOR, POSITIVE_DARK) if value >= 0 else (NEGATIVE_COLOR, NEGATIVE_DARK)
    return tuple(d + strength * (b - d) for b, d in zip(to_rgb(base), to_rgb(dark)))


def plot_loading_tiles(ax, loadings, title, max_rows):
    ax.set_xlim(-1.7, 3.7)
    ax.set_ylim(max_rows + 0.1, -1.1)
    ax.set_aspect(TILE_ASPECT)
    ax.axis("off")
    for column, component in enumerate(loadings.columns):
        ax.text(column + 0.5, -0.6, component, ha="center", va="center", fontsize=SUMMARY_FONT)
    for row, (variable, values) in enumerate(loadings.iterrows()):
        ax.text(-0.1, row + 0.5, LOADING_LABELS[variable], ha="right", va="center", fontsize=SUMMARY_FONT)
        for column, value in enumerate(values):
            ax.add_patch(FancyBboxPatch(
                (column + 0.04, row + 0.06), 0.92, 0.88,
                boxstyle="round,pad=0,rounding_size=0.08", mutation_aspect=TILE_ASPECT,
                facecolor=loading_tile_color(value), edgecolor="none",
            ))
            ax.text(column + 0.5, row + 0.5, f"{value:.2f}", color="white", fontweight="bold",
                    ha="center", va="center", fontsize=SUMMARY_FONT)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold", loc="left", pad=8)


def plot_centroid_scores(ax, centroids, name, explained, title):
    x_column, y_column = f"{name}_PC1", f"{name}_PC2"
    for class_id, row in centroids.iterrows():
        ax.scatter(row[x_column], row[y_column], s=450, color=class_color(class_id),
                   edgecolor="black", zorder=3)
        ax.annotate(str(int(class_id)), (row[x_column], row[y_column]), xytext=(10, 10),
                    textcoords="offset points", fontsize=SUMMARY_FONT)
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel(f"{name.title()} PC1 ({explained[0]:.0%})", fontsize=SUMMARY_FONT)
    ax.set_ylabel(f"{name.title()} PC2 ({explained[1]:.0%})", fontsize=SUMMARY_FONT)
    ax.set_title(title, fontsize=SUMMARY_FONT + 4, pad=18)
    ax.tick_params(labelsize=SUMMARY_FONT)
    ax.grid(color="lightgray", linestyle="--", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def plot_summary_figure(loadings_by_domain, centroids, explained_by_domain, output_path):
    fig = plt.figure(figsize=(18, 17))
    grid = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25, left=0.08, right=0.98, top=0.94, bottom=0.19)
    max_rows = max(len(loadings) for loadings in loadings_by_domain.values())
    plot_loading_tiles(fig.add_subplot(grid[0, 0]), loadings_by_domain["structure"], "a) Structure PCA", max_rows)
    plot_loading_tiles(fig.add_subplot(grid[0, 1]), loadings_by_domain["activity"], "b) Activity PCA", max_rows)
    plot_centroid_scores(fig.add_subplot(grid[1, 0]), centroids, "structure",
                         explained_by_domain["structure"], "c) Structure class centroids")
    plot_centroid_scores(fig.add_subplot(grid[1, 1]), centroids, "activity",
                         explained_by_domain["activity"], "d) Activity class centroids")
    fig.legend(handles=[Patch(facecolor=POSITIVE_COLOR, label="Positive"),
                        Patch(facecolor=NEGATIVE_COLOR, label="Negative")],
               title="Sign of scaled loading (shade = magnitude)", title_fontsize=SUMMARY_FONT,
               loc="lower center", ncol=2, frameon=False, fontsize=SUMMARY_FONT,
               bbox_to_anchor=(0.5, 0.04))
    fig.text(0.5, 0.01, "Scaled loading = eigenvector coefficient \u00d7 \u221aeigenvalue "
             "(\u2248 variable\u2013PC correlation; can slightly exceed 1)",
             ha="center", va="bottom", fontsize=SUMMARY_FONT - 3)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_pc1_centroids(centroids, explained_by_domain, output_path):
    fig, ax = plt.subplots(figsize=(11, 9))
    for class_id, row in centroids.iterrows():
        ax.scatter(row["structure_PC1"], row["activity_PC1"], s=600, color=class_color(class_id),
                   edgecolor="black", linewidth=1.4, zorder=3)
        ax.annotate(str(int(class_id)), (row["structure_PC1"], row["activity_PC1"]),
                    xytext=(12, 12), textcoords="offset points", fontsize=SUMMARY_FONT, fontweight="bold")
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel(f"Structure PC1 ({explained_by_domain['structure'][0]:.0%})", fontsize=SUMMARY_FONT)
    ax.set_ylabel(f"Activity PC1 ({explained_by_domain['activity'][0]:.0%})", fontsize=SUMMARY_FONT)
    ax.set_title("Structure PC1 vs Activity PC1: class centroids", fontsize=SUMMARY_FONT + 2,
                 fontweight="bold", loc="left")
    ax.tick_params(labelsize=SUMMARY_FONT)
    ax.grid(color="lightgray", linestyle="--", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_scores_figure(loadings_by_domain, centroids, explained_by_domain, output_path):
    fig = plt.figure(figsize=(18, 20))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], hspace=0.02, wspace=0.1,
                            left=0.08, right=0.98, top=0.95, bottom=0.17)
    max_rows = max(len(loadings) for loadings in loadings_by_domain.values())
    plot_loading_tiles(fig.add_subplot(grid[0, 0]), loadings_by_domain["structure"], "a) Structure PCA", max_rows)
    plot_loading_tiles(fig.add_subplot(grid[0, 1]), loadings_by_domain["activity"], "b) Activity PCA", max_rows)
    ax = fig.add_subplot(grid[1, :])
    for class_id, row in centroids.iterrows():
        ax.scatter(row["structure_PC1"], row["activity_PC1"], s=1100, color=class_color(class_id),
                   edgecolor="black", zorder=3)
        ax.annotate(str(int(class_id)), (row["structure_PC1"], row["activity_PC1"]),
                    xytext=(16, 16), textcoords="offset points", fontsize=SUMMARY_FONT)
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel(f"Structure PC1 ({explained_by_domain['structure'][0]:.0%})", fontsize=SUMMARY_FONT)
    ax.set_ylabel(f"Activity PC1 ({explained_by_domain['activity'][0]:.0%})", fontsize=SUMMARY_FONT)
    ax.set_title("c) Class centroids: Structure PC1 vs Activity PC1", fontsize=TITLE_FONTSIZE,
                 fontweight="bold", loc="left", pad=8)
    ax.tick_params(labelsize=SUMMARY_FONT)
    ax.grid(color="lightgray", linestyle="--", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.legend(handles=[Patch(facecolor=POSITIVE_COLOR, label="Positive"),
                        Patch(facecolor=NEGATIVE_COLOR, label="Negative")],
               title="Sign of scaled loading (shade = magnitude)", title_fontsize=SUMMARY_FONT,
               loc="lower center", ncol=2, frameon=False, fontsize=SUMMARY_FONT,
               bbox_to_anchor=(0.5, 0.03))
    fig.text(0.5, 0.01, "Scaled loading = eigenvector coefficient \u00d7 \u221aeigenvalue "
             "(\u2248 variable\u2013PC correlation; can slightly exceed 1)\n"
             "* log1p-transformed before PCA",
             ha="center", va="bottom", fontsize=SUMMARY_FONT - 3)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run_scores_figure(data, out, logs):
    centroids, explained_by_domain, loadings_by_domain = None, {}, {}
    for name, variables in GROUPS.items():
        _, centers, loadings, explained, _, _, _ = fit_project(data, name, variables, logs)
        explained_by_domain[name] = explained
        loadings_by_domain[name] = loadings
        centroids = centers if centroids is None else centroids.join(centers, how="inner")
    plot_scores_figure(loadings_by_domain, centroids, explained_by_domain, out / f"{OUTPUT_FILENAME}.png")


def run_analysis(data, out, logs, mode, save_scores=False):
    prefix = f"{OUTPUT_FILENAME}_{mode}"
    all_scores = pd.DataFrame({CLASS_COLUMN: data[CLASS_COLUMN]}, index=data.index)
    complete = pd.Series(True, index=data.index)
    all_centroids = None
    loadings_by_domain = {}
    explained_by_domain = {}
    for name, variables in GROUPS.items():
        if mode == "centroid_fit":
            projected, centers, loadings, explained, counts, good, means = fit_project(
                data, name, variables, logs)
            means.to_csv(out / f"{prefix}_{name}_transformed_class_means.csv")
        else:
            projected, loadings, explained, good = fit_video_pca(data, name, variables, logs)
            centers = projected.loc[good].groupby(data.loc[good, CLASS_COLUMN]).mean().sort_index()
        loadings_by_domain[name] = loadings
        explained_by_domain[name] = explained
        complete &= good
        all_scores = all_scores.join(projected)
        all_centroids = centers if all_centroids is None else all_centroids.join(centers, how="inner")
        loadings.to_csv(out / f"{prefix}_{name}_loadings.csv")
        pd.Series(explained, index=["PC1", "PC2"], name="explained_variance_ratio").to_csv(
            out / f"{prefix}_{name}_explained_variance.csv")
        sub = pd.concat([data[[CLASS_COLUMN]], projected], axis=1)
        cov = class_covariances(sub, list(centers.columns))
        save_covariances(cov, list(centers.columns), out, f"{mode}_{name}")
        plot_domain(sub, centers, cov, f"{mode}_{name}", out)
        print(f"{mode} {name} explained variance: {explained.round(3)}")
        print(loadings.round(3))
    all_centroids.index.name = "class_id"
    all_centroids.to_csv(out / f"{prefix}_class_centroid_scores.csv")
    plot_summary_figure(loadings_by_domain, all_centroids, explained_by_domain,
                        out / f"{prefix}_summary_four_panel.png")
    plot_pc1_centroids(all_centroids, explained_by_domain,
                       out / f"{prefix}_structure_vs_activity_PC1_centroids.png")
    joint = all_scores.loc[complete].dropna()
    if (~complete).sum():
        print(f"{mode}: {(~complete).sum():,} videos omitted from combined analysis")
    axes2 = ["structure_PC1", "activity_PC1"]
    axes4 = ["structure_PC1", "structure_PC2", "activity_PC1", "activity_PC2"]
    cov2 = class_covariances(joint, axes2)
    save_covariances(cov2, axes2, out, f"{mode}_joint_2D")
    save_covariances(class_covariances(joint, axes4), axes4, out, f"{mode}_joint_4D")
    fig, ax = plt.subplots(figsize=(9, 7))
    for cls in all_centroids.index:
        if cls not in cov2:
            continue
        color = class_color(cls)
        center = joint.loc[joint[CLASS_COLUMN] == cls, axes2].mean().to_numpy()
        ellipse(ax, center, cov2[cls][1], color)
        ax.scatter(*center, color=color, edgecolor="black", s=60, zorder=3)
        ax.annotate(str(cls), center, xytext=(5,5), textcoords="offset points")
    ax.axhline(0, color="gray", lw=.7); ax.axvline(0, color="gray", lw=.7)
    ax.set_xlabel("Structure PC1"); ax.set_ylabel("Activity PC1")
    ax.set_title(f"{mode}: class centroids and 1-SD covariance ellipses")
    fig.tight_layout(); fig.savefig(out / f"{prefix}_combined_class_dispersion.png", dpi=200)
    plt.close(fig)
    for dimension, axes in [("2D", axes2), ("4D", axes4)]:
        path = out / f"{prefix}_pairwise_gaussian_affinity_{dimension}.csv"
        pairwise_affinity(joint, axes, sorted(all_centroids.index), path)
        save_heatmap(path, all_centroids.index, out / f"{prefix}_affinity_{dimension}_heatmap.png",
                     f"{mode}: {dimension} class affinity")
    fractions = (joint.assign(in_positive_quadrant=(joint[axes2[0]] > 0) & (joint[axes2[1]] > 0))
                 .groupby(CLASS_COLUMN)["in_positive_quadrant"].agg(["mean", "sum", "count"]))
    fractions.columns = ["fraction_positive_quadrant", "n_positive_quadrant", "n_complete"]
    selection = all_centroids[axes2].join(fractions)
    selection["candidate_positive_centroid"] = (selection[axes2[0]] > 0) & (selection[axes2[1]] > 0)
    selection.to_csv(out / f"{prefix}_convective_candidate_screen.csv")
    selected = selection.index[selection["candidate_positive_centroid"]].tolist()
    save_heatmap(out / f"{prefix}_pairwise_gaussian_affinity_4D.csv", selected,
                 out / f"{prefix}_candidate_affinity_4D_heatmap.png",
                 f"{mode}: candidate class affinity (4 PCs)")
    if 2 <= len(all_centroids):
        assignments = plot_cluster(all_centroids[axes4],
                                   out / f"{prefix}_four_pc_dendrogram.png",
                                   min(N_CLUSTERS, len(all_centroids)))
        all_centroids.assign(cluster=assignments).to_csv(
            out / f"{prefix}_class_centroid_scores_and_clusters.csv")
    if save_scores:
        joint.to_csv(out / f"{prefix}_all_video_scores.csv", index=False)
    return selection, loadings_by_domain


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None,
                        help="Video-level CSV (default: repository config)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output directory (default: repository config)")
    parser.add_argument("--save-observation-scores", action="store_true")
    parser.add_argument("--log1p", nargs="*", default=sorted(LOG1P_VARIABLES),
                        choices=sorted(v for group in GROUPS.values() for v in group))
    args = parser.parse_args(argv)
    if args.input is None or args.out is None:
        if load_config is None:
            parser.error("Specify --input and --out when running outside the project repository")
        config = load_config(str(CONFIG_PATH))
    input_path = args.input if args.input is not None else Path(config["output_files"]["training_video_summary_temporal"])
    out = args.out if args.out is not None else Path(config["output_files"]["figures_dir"])
    out.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(input_path)
    required = {CLASS_COLUMN, *(v for group in GROUPS.values() for v in group)}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    labels = pd.to_numeric(data[CLASS_COLUMN], errors="coerce")
    if labels.isna().any() or not np.equal(labels, np.floor(labels)).all():
        raise ValueError("Invalid class labels")
    data[CLASS_COLUMN] = labels.astype(int)
    data = data.loc[data[CLASS_COLUMN] != -100].copy()
    if data.empty or not data[CLASS_COLUMN].between(0, 9).all():
        raise ValueError("Expected nonempty data with labels 0..9")
    print(f"Loaded {len(data):,} labeled videos, {data[CLASS_COLUMN].nunique()} classes")
    run_scores_figure(data, out, args.log1p)
    print(f"Saved {OUTPUT_FILENAME}.png to {out.resolve()}")


if __name__ == "__main__":
    main()
