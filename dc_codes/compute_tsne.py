
import numpy as np
import os, glob
#from sklearn.manifold import TSNE
#from openTSNE import TSNE

import openTSNE

common_path = '/home/Daniele/codes/vissl/runs/dcv2_ir108_128x128_k9_germany_30kcrops_grey_5th-95th/features/'  ##

output_path = '/home/Daniele/fig/dcv_ir108_128x128_k9_30k_grey_5th-95th/'

n_crops = 33792

filename1 = 'rank0_chunk0_train_heads_inds.npy' 
filename2 = 'rank0_chunk0_train_heads_targets.npy'
filename3 = 'rank0_chunk0_train_heads_features.npy'

data1 = np.load(common_path+filename1)
#data2 = np.load(common_path+filename2)
data3 = np.load(common_path+filename3)

data = data3

data = np.reshape(data3,(n_crops,128))  #DC value 664332, it should be the train num samples

#tsne parameters here:
#https://opentsne.readthedocs.io/en/stable/api/sklearn.html#openTSNE.sklearn.TSNE
#%time
embedding_pca_cosine = openTSNE.TSNE(
    perplexity=30,
    initialization="pca",
    metric="cosine",
    n_jobs=-1,
    random_state=3,
).fit(data)

#tsne = TSNE(n_components=2, random_state=5).fit_transform(data)

np.save(output_path+'tsnegermany_pca_cosine_500ep.npy', embedding_pca_cosine)  ##

print('tsne calculated and saved')

"""
embedding_annealing = openTSNE.TSNE(
    perplexity=500, metric="cosine", initialization="pca", n_jobs=-1, random_state=3
).fit(data)

embedding_annealing.affinities.set_perplexities([50])

embedding_annealing = embedding_annealing.optimize(250)

np.save(output_path+'/tsnegermany_pca_cosine_500annealing50_800ep.npy', embedding_annealing) ##


affinities_multiscale_mixture = openTSNE.affinity.Multiscale(
    data,
    perplexities=[50, 500],
    metric="cosine",
    n_jobs=-1,
    random_state=3,
)

init = openTSNE.initialization.pca(data, random_state=42)

embedding_multiscale = openTSNE.TSNE(n_jobs=-1).fit(
    affinities=affinities_multiscale_mixture,
    initialization=init,
)

np.save(output_path+'tsnegermany_pca_cosine_500multiscale50_800ep.npy', embedding_multiscale)  ##
"""