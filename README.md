# Workflow Overview

This repository provides tools to analyze and visualize the training and feature space of machine learning models applied to crop data.

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


