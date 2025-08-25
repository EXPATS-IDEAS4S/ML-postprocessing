import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


run_name = '10th-90th' #['10th-90th', '10th-90th_CMA']
sampling_type = 'closest'  # Options: 'random', 'closest', 'farthest', 'all'
stat = '50' #None  # '50'
n_subsample = 1000  # Number of samples per cluster

# Path to fig folder for outputs
output_path = f'/home/Daniele/fig/cma_analysis/{run_name}/{sampling_type}/'

# Load the saved CSV into a new DataFrame
merged_df = pd.read_csv(f'{output_path}physical_feature_vectors_{run_name}_{sampling_type}_{n_subsample}_{stat}.csv')

# Step 1: Drop rows with NaN values if you haven't already
df_cleaned = merged_df.dropna()

# Step 2: Split the data into features (X) and labels (y)
# Assuming your label column is called 'label'
X = df_cleaned.drop(columns=['label'])  # Features
y = df_cleaned['label']  # Labels (truth)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Step 5: Make predictions and check accuracy
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy:.4f}")

# Step 6: Check feature importances
feature_importances = rf.feature_importances_

# Step 7: Create a DataFrame for the feature importances
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': feature_importances
})

# Step 8: Sort the features by importance
importance_df = importance_df.sort_values(by='importance', ascending=False)

# Step 9: Plot the feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=importance_df)#, palette='blue')
plt.title('Feature Importances from Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')

# Save the figure
output_fig_path = f'{output_path}RF_feature_importance_{run_name}_{sampling_type}_{n_subsample}_{stat}.png'
plt.savefig(output_fig_path, bbox_inches='tight', dpi=300)
print(f'fig saved in: {output_fig_path}')