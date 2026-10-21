"""Project cloud-summary observations into two PCAs fitted on class means.

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

Outputs include class means and PCA scores/loadings/variance, per-class
projected-observation covariance matrices in both domains and combined spaces,
dispersion plots, pairwise Gaussian Bhattacharyya affinities in two- and four-PC
spaces, a hierarchical clustering dendrogram and class assignments, and
optionally all observation scores. The four-PC centroids are standardized
before Ward clustering.

Run directly with ``python scripts/pretrain/cluster_analysis/grl_plots/figure_sup1_PCA_convection.py``
or through ``grl_pipeline.py --figures figure_sup1``. The configured figure
directory is used unless ``--out`` is provided.
"""

import sys
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
	sys.path.append(str(REPO_ROOT))

from utils.configs import load_config
from utils.plotting.class_colors import colors_per_class_codes_grl


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


def main(argv=None):
	config = load_config(str(CONFIG_PATH))
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--input",
		type=Path,
		default=Path(config["output_files"]["training_video_summary_temporal"]),
		help="Observation-level crops video-summary CSV",
	)
	parser.add_argument(
		"--out",
		type=Path,
		default=Path(config["output_files"]["figures_dir"]),
		help="Directory for PCA figures and tables",
	)
	parser.add_argument(
		"--log1p",
		nargs="*",
		default=sorted(LOG1P_VARIABLES),
		choices=sorted(column for group in GROUPS.values() for column in group),
		help="Variables to transform with log1p before class means are computed",
	)
	parser.add_argument("--clusters", type=int, default=N_CLUSTERS)
	parser.add_argument(
		"--save-observation-scores",
		action="store_true",
		help="Write the full per-observation PCA score table",
	)
	args = parser.parse_args(argv)
	input_path = args.input
	output_dir = args.out
	output_dir.mkdir(parents=True, exist_ok=True)

	video_stats = pd.read_csv(input_path)
	missing = sorted({CLASS_COLUMN, *[column for group in GROUPS.values() for column in group]} - set(video_stats.columns))
	if missing:
		raise ValueError(f"Missing required columns in observation summary: {missing}")
	video_stats[CLASS_COLUMN] = pd.to_numeric(video_stats[CLASS_COLUMN], errors="coerce")
	if video_stats[CLASS_COLUMN].isna().any():
		raise ValueError(f"Missing or non-numeric class assignments in {CLASS_COLUMN}")
	if not np.equal(video_stats[CLASS_COLUMN], np.floor(video_stats[CLASS_COLUMN])).all():
		raise ValueError("Class assignments must be integer labels")
	video_stats[CLASS_COLUMN] = video_stats[CLASS_COLUMN].astype(int)
	video_stats = video_stats.loc[video_stats[CLASS_COLUMN] != -100].copy()
	if video_stats.empty:
		raise ValueError("No labeled observations found in the input summary")
	if not video_stats[CLASS_COLUMN].between(0, 9).all():
		raise ValueError("Expected class labels in the range 0..9")
	print(f"Loaded {len(video_stats):,} labeled observations across {video_stats[CLASS_COLUMN].nunique()} classes")

	all_scores = pd.DataFrame({CLASS_COLUMN: video_stats[CLASS_COLUMN]})
	centroids = None
	complete = pd.Series(True, index=video_stats.index)
	for name, variables in GROUPS.items():
		projected, scores, loadings, explained, counts, good, means = fit_project(
			video_stats, name, variables, args.log1p
		)
		projected_data = pd.concat([video_stats[[CLASS_COLUMN]], projected], axis=1)
		covariances = class_covariances(projected_data, scores.columns)
		complete &= good
		all_scores = all_scores.join(projected)
		centroids = scores if centroids is None else centroids.join(scores, how="inner")
		means.to_csv(output_dir / f"{OUTPUT_FILENAME}_{name}_transformed_class_means.csv")
		counts.rename("observation_count").to_csv(output_dir / f"{OUTPUT_FILENAME}_{name}_class_counts.csv")
		loadings.to_csv(output_dir / f"{OUTPUT_FILENAME}_{name}_loadings.csv")
		pd.Series(explained, index=["PC1", "PC2"], name="explained_variance_ratio").to_csv(
			output_dir / f"{OUTPUT_FILENAME}_{name}_explained_variance.csv"
		)
		save_covariances(covariances, scores.columns, output_dir, name)
		plot_domain(projected_data, scores, covariances, name, output_dir)
		print(f"\n{name.upper()} explained variance: {np.round(explained, 3)}")
		print(f"{name.upper()} loadings:\n{loadings.round(3)}")

	centroids.index.name = "class_id"
	centroids.to_csv(output_dir / f"{OUTPUT_FILENAME}_class_centroid_scores.csv")
	joint = pd.concat([video_stats[[CLASS_COLUMN]], all_scores.drop(columns=CLASS_COLUMN)], axis=1)
	joint = joint.loc[complete].dropna()
	if not complete.all():
		print(f"WARNING: {(~complete).sum():,} observations missing at least one feature; omitted from joint analyses")

	joint_columns = ["structure_PC1", "activity_PC1"]
	combined_covariances_2d = class_covariances(joint, joint_columns)
	combined_covariances_4d = class_covariances(joint, list(centroids.columns))
	save_covariances(combined_covariances_2d, joint_columns, output_dir, "combined_2D")
	save_covariances(combined_covariances_4d, list(centroids.columns), output_dir, "combined_4D")
	fig, ax = plt.subplots(figsize=(9, 7))
	for class_id in centroids.index:
		subset = joint.loc[joint[CLASS_COLUMN] == class_id, joint_columns]
		if len(subset) < MIN_COVARIANCE_OBSERVATIONS:
			continue
		center = subset.mean().to_numpy()
		color = class_color(class_id)
		ellipse(ax, center, combined_covariances_2d[class_id][1], color)
		ax.scatter(*center, s=65, edgecolor="black", color=color, zorder=4)
		ax.annotate(str(int(class_id)), center, xytext=(5, 5), textcoords="offset points")
	ax.axhline(0, color="gray", linewidth=0.6)
	ax.axvline(0, color="gray", linewidth=0.6)
	ax.set_xlabel("Structural PC1")
	ax.set_ylabel("Activity PC1")
	ax.set_title("Joint class means and within-class 1-SD observation dispersion")
	fig.tight_layout()
	fig.savefig(output_dir / f"{OUTPUT_FILENAME}_combined_class_dispersion.png", dpi=200)
	plt.close(fig)

	classes = sorted(centroids.index)
	pairwise_affinity(
		joint, joint_columns, classes,
		output_dir / f"{OUTPUT_FILENAME}_pairwise_gaussian_affinity_2D.csv",
	)
	four_pc_columns = list(centroids.columns)
	pairwise_affinity(
		joint, four_pc_columns, classes,
		output_dir / f"{OUTPUT_FILENAME}_pairwise_gaussian_affinity_4D.csv",
	)
	assignments = plot_cluster(
		centroids[four_pc_columns],
		output_dir / f"{OUTPUT_FILENAME}_four_pc_dendrogram.png",
		args.clusters,
	)
	centroids.assign(cluster=assignments).to_csv(
		output_dir / f"{OUTPUT_FILENAME}_class_centroid_scores_and_clusters.csv"
	)
	if args.save_observation_scores:
		all_scores.to_csv(output_dir / f"{OUTPUT_FILENAME}_all_observation_scores.csv", index=False)
	print(f"\nSaved PCA outputs to {output_dir.resolve()}")


if __name__ == "__main__":
	main()
