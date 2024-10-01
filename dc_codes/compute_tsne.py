
import numpy as np
import os, glob
#from sklearn.manifold import TSNE
#from openTSNE import TSNE

import openTSNE

scales = ['10th-90th_CMA']
random_states = [0,3,16,23,57] #old was 3, already computed

n_crops = 33792

#filename1 = 'rank0_chunk0_train_heads_inds.npy' 
#filename2 = 'rank0_chunk0_train_heads_targets.npy'
filename3 = 'rank0_chunk0_train_heads_features.npy'

for scale in scales:
    for random_state in random_states:

        common_path = f'/data1/runs/dcv2_ir108_128x128_k9_germany_30kcrops_grey_{scale}/features/'  #

        output_path = f'/home/Daniele/fig/dcv_ir108_128x128_k9_30k_grey_{scale}/'

        #data1 = np.load(common_path+filename1)
        #data2 = np.load(common_path+filename2)
        data3 = np.load(common_path+filename3)
        #data = data3
        data = np.reshape(data3,(n_crops,128))  #DC value 664332, it should be the train num samples


        #tsne parameters here:
        #https://opentsne.readthedocs.io/en/stable/api/sklearn.html#openTSNE.sklearn.TSNE
        #%time
        embedding_pca_cosine = openTSNE.TSNE(
            perplexity=30,
            initialization="pca",
            metric="cosine",
            n_jobs=-1,
            random_state=random_state,
        ).fit(data)

        #tsne = TSNE(n_components=2, random_state=5).fit_transform(data)

        np.save(f'{output_path}tsne_pca_cosine_{scale}_{random_state}.npy', embedding_pca_cosine)  ##

        print(f'{output_path}tsne_pca_cosine_{scale}_{random_state}.npy calculated and saved')

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