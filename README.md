# Overview

This repository provides tools for analyzing and visualizing the training process and feature space of VISSL machine learning models.

## Pretraining Analysis

The repository includes scripts to evaluate self-supervised VISSL models and their feature representations:

### `check_training`

Utilities to inspect model training and convergence:

* **`plot_training_loss.py`** – Plots training loss (and optionally learning rate) across iterations and epochs.
* **`plot_loss_clustering_metrics_epochs.py`** – Visualizes clustering quality of features across epochs. Features should first be extracted at multiple epochs. Clustering metrics can be computed using `compute_cluster_metrics.py` in the `k_optimization` folder.
* Shared helper functions are in **`utils/plot_utils/check_training_utils.py`**.

### `k_optimization`

Subfolder `clustering_metrics` for analyzing cluster quality:

* **`compute_cluster_metrics_dim_red.py`** – Computes clustering metrics on dimensionally reduced features.
* **`compute_clustering_metrics_full_feat.py`** – Computes metrics on the full high-dimensional feature space.
* **`plot_clustering_metrics.py`** – Plots results from clustering metrics.
* Utility functions are stored in **`utils/analysis_utils/utils_clustering.py`**.
* *TODO:* Implement correlation analysis among features and class distribution distances for K optimization.

### `dim_reduction`

Scripts for reducing the dimensionality of feature vectors for visualization or further analysis.

### `prepare_features`

Scripts for organizing, cleaning, and merging features from training and inference:

* **`merge_train_test_features.py`** – Loads and merges high-dimensional feature vectors from training and test datasets, annotates them with metadata (e.g., case-study labels, vector types), and outputs a structured DataFrame ready for downstream analysis or dimensionality reduction. Ensures only relevant samples are included and preserves index and feature information.

---


# Suggestion of a Workflow 

### 1. Check model convergence

* Start by examining the **training loss curves** to ensure the model has properly converged.

### 2. Dimensionality reduction

* Compute a **dimensionality reduction** (commonly t-SNE, but other methods like Isomap can also be used).

### 3. Crop and label selection

* Generate a list of crops and labels according to your selection strategy:

  * all samples
  * random subset
  * samples close to cluster centroids
  * samples far from cluster centroids

### 4. Attach features to crops

* Combine the selected crops/labels with their **reduced feature components** (e.g., t-SNE embeddings).

### 5. Visualization of the feature space

* Plot the feature space as a **scatter plot of embeddings**.
* If images are available, visualize the feature space with **image crops overlaid**.
* For videos, additional preprocessing steps are required.

### 6. Compute sample statistics (optional)

* Enrich the analysis with **ancillary data** to compute sample-level statistics.

### 7. Cluster characterization plots

With the embeddings and statistics, different visualizations can be generated:

* Scatter plots (with or without additional statistics)
* Diurnal cycle plots
* Distributions of single variables (statistics integration in progress)


