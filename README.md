# Overview

This repository provides tools for analyzing and visualizing the training process and feature space of VISSL machine learning models.


## Configs

All configurable parameters are defined in **YAML files**:

* **`process_run_config.yaml`**
  Main runtime configuration (paths, variables to process, statistics options).
  ⚠️ Remember to update the path to this file in the main function of each script.
  *TODO*: allow passing the config path via command-line arguments.

* **`variables_metadata.yaml`**
  Contains the full list of variables from ancillary datasets, along with their metadata for computation and plotting.
  This makes it easy to extend it as new data sources are added.

### Planned Improvements

* Add a dedicated config file for **metric computation** (spatial & temporal).
* Create a separate config for **plotting/visualization**, decoupled from computation.
* Support multiple **dimensionality reduction methods** via a dedicated config.




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

* **compute_2d_embedding.py**: This script performs dimensionality reduction on high-dimensional feature data  to visualize it in a lower-dimensional space. It supports both Isomap and t-SNE.

* **visualize_embedding.py**: Plot feature space for an image-based and video-based samples.

* **visualize_embedding_case_studies.py**: Plot the feature space including case studies.

* **plot_embedding_utils.py**: Utils function for plotting the feature space

* **features_utils.py**: TODO, Utils functions for preparing the feature space data before plotting



## `prepare_features`

These scripts handle **feature extraction, cleaning, and organization** from training and inference datasets, preparing them for downstream analysis or dimensionality reduction.

* **`merge_train_test_features.py`** – Loads and merges high-dimensional feature vectors from training and test datasets. Annotates samples with metadata (e.g., case-study labels, vector types) and outputs a structured DataFrame ready for analysis. Ensures only relevant samples are included while preserving feature and index integrity.

* **`adapt_random_crops_filename.py`** – Adjusts or standardizes filenames for random crops to ensure consistency across datasets.

* **`create_crop_list_from_buckets.py`** – Selects and outputs a list of samples (crops) from cloud storage buckets for further processing.

* **`create_df_var_tsne.py`** – Extends the crop list by adding corresponding components from dimensionality reduction (e.g., t-SNE embeddings) to each sample. Planned merge with `create_crop_list_from_buckets.py` to streamline the workflow.


## `cluster_analysis`

These scripts focus on **characterizing clusters/classes in the feature space** using ancillary datasets (e.g., physical variables, satellite data, orography).

* **`compute_crops_statistics_new.py`** – Computes selected statistics (e.g., percentiles) for physical variables of interest for each sample, and appends them to the dataset.

* **`histogram_overlaid_per_class.py`** – Generates overlaid histograms of selected variables per cluster/class to visualize distributions.

* **`plot_class_var_cycle.py`** – Plots diurnal and seasonal cycles of physical variables per class to explore temporal patterns.

* **`plot_feature_space_with_physical_var_new.py`** – Visualizes the feature space embedding (e.g., t-SNE, PCA) colored by physical variable values.

* **`scatter_and_distribution_analysis.py`** – Performs scatter and distribution analyses for variables across classes or clusters.

* **`visualize_crops_with_phys_vars.py`** – Maps or plots samples/crops with associated physical variable values for exploratory analysis.

* **`percentile_maps`** – Computes spatial percentile maps for variables across grid cells and labels, and plots them using Cartopy for geospatial visualization. Includes optional parallel computation for efficiency.



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

### 6. Compute sample statistics 

* Enrich the analysis with **ancillary data** to compute sample-level statistics.

### 7. Cluster characterization plots

With the embeddings and statistics, different visualizations can be generated:

* Scatter plots (with or without additional statistics)
* Diurnal cycle plots
* Distributions of single variables (statistics integration in progress)


