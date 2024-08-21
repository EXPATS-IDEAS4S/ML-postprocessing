
import numpy as np
import os, glob
from sklearn.manifold import Isomap
import pandas as pd
import random

common_path = '/home/Daniele/codes/vissl/runs/dcv2_ir_128x128_k7_germany_70kcrops/features/'  ##

output_path = '/home/Daniele/fig/dcv2_ir_128x128_k7_germany_70kcrops/'

n_features = 128
n_epochs = 800
n_components = 2

filename1 = 'rank0_chunk0_train_heads_inds.npy' 
#filename2 = 'rank0_chunk0_train_heads_targets.npy'
filename3 = 'rank0_chunk0_train_heads_features.npy'

data1 = np.load(common_path+filename1)
#data2 = np.load(common_path+filename2)
data3 = np.load(common_path+filename3)

# Verify shapes
print(f'Index shape: {data1.shape}')
print(f'Feature shape: {data3.shape}')

n_crops = data1.shape[0] #70135

# Reshape data3 if necessary
data3 = np.reshape(data3,(n_crops,n_features))  #DC value 664332, it should be the train num samples
#print(f'Feature shape: {np.shape(data3)}')

# Randomly select some rows
n_random_samples = 20000  # Change this to the desired number of random samples
random_indices = random.sample(range(n_crops), n_random_samples)

# Extract the corresponding values from data3
selected_indices = data1[random_indices]
selected_features = data3[random_indices, :]

#isomap from sklearn:
#https://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html#sklearn.manifold.Isomap

#default parameters
#class sklearn.manifold.Isomap(*, n_neighbors=5, radius=None, n_components=2, 
#eigen_solver='auto', tol=0, max_iter=None, path_method='auto', neighbors_algorithm='auto', 
#n_jobs=None, metric='minkowski', p=2, metric_params=None)[source]

#tol: tolerance in the eigenvalue solver to attain convergence.
#While a lower value might result in a more accurate solution, it might also lengthen the computation time.

#max_iter: The maximum number of times the eigenvalue solver can run. 
#It continues if None is selected, unless convergence or additional stopping conditions are satisfied

print(selected_features.shape)
embedding = Isomap(n_components=n_components, metric='cosine', n_jobs=-1)
X_transformed = embedding.fit_transform(selected_features)
print(X_transformed.shape)
print(X_transformed)

#tsne = TSNE(n_components=2, random_state=5).fit_transform(data)

# Create a DataFrame with the Isomap-transformed data and the random indices
df = pd.DataFrame(X_transformed, columns=[f'Component_{i+1}' for i in range(n_components)])
df['Selected_Index'] = selected_indices
print(df)

# Save the DataFrame to a CSV file
df.to_csv(f'{output_path}isomap_{n_components}d_cosine_{n_epochs}ep_{n_random_samples}samples.csv', index=False)

#np.save(f'{output_path}isomap_cosine_{n_epochs}ep.npy', X_transformed)  ##

print('isomap calculated and saved')

