import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the cloud properties CSV
cloud_properties_path = '/home/Daniele/data/ir108_2013_128_germany/cloud_properties.csv'
cloud_properties = pd.read_csv(cloud_properties_path)

# Read the labels CSV
labels_path = 'path_to_labels.csv'
labels = pd.read_csv(labels_path)



# Merge the datasets on the 'filename' column
merged_df = pd.merge(cloud_properties, labels, on='filename')

# Display the first few rows of the merged dataframe
print(merged_df.head())

# Define the columns containing cloud properties and the class column
cloud_property_columns = [col for col in merged_df.columns if 'percentage' in col or 'percentile' in col]
class_column = 'class_label'  # replace with the actual name of your class label column

# Set up the plotting environment
sns.set(style="whitegrid")
num_columns = len(cloud_property_columns)
num_rows = (num_columns + 2) // 3

# Create a figure for the subplots
fig, axes = plt.subplots(num_rows, 3, figsize=(15, num_rows * 5))
axes = axes.flatten()

# Plot the distribution for each cloud property
for i, column in enumerate(cloud_property_columns):
    sns.histplot(data=merged_df, x=column, hue=class_column, kde=True, ax=axes[i], multiple="stack")
    axes[i].set_title(f'Distribution of {column}')

# Adjust layout
plt.tight_layout()
plt.show()
