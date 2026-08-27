repo for the analysis for the Model Evaluation paper.

After the VISSL run is done and features are extracted for both training and test sets, the following steps are required to prepapre the data for the final analysis and Figures.

Prepare the csv with the features.

- Gather all features
prepare_csv/0_prepare_feature_dataset.py for collecting together the features of diffrent datasets, like training centorids and possible test datasets that are analysed together.
All the configurable parmaeres are in process_run_config.yaml especially under feature_preparation.

- Tsne computation
Once the dataset with all the features to analyse is built, we need to compute tsne based feature space for visualization purpuses. For doing it properly we need all the feature vectors together, with training centrodis and test sets.
For this we use 1_compute_tsne_train_test.py, whic will take all feature vectors and reduce it to 2d space using tsne method. and save the 2 dimension in the same csv file

- Make crop images
Then, for actually visualize we need the tsne position but also the images, so in case nc format is used there is needed to also have the correspondent images (e.g. png files). In case you hav already process images files forthe ML traiing or already the images are computed, you can skip this step

- Compute crops statistics
Then for each crops, training and test, we can compute some cloud properties statistics and also precipitation. Select the variabels to compute in process_run_congi.yaml under 'statistics' section, all the variable metadata are in variables_metadata.yaml and the actual computation functions in stats_utils.py.

Once the csv files are process then we are ready to produce actual analysis and plots
So go nder the folder Figures:

