import pandas as pd
import numpy as np
import os, glob
from sklearn.manifold import TSNE
import vissl
from vissl.utils.io import save_file
from vissl.utils.io import load_file

common_path = '/p/project/deepacf/kiste/DC/barbados/k7/leif_256_limited_aug/features/'

filename1 = 'rank0_chunk0_train_heads_inds.npy' 
filename2 = 'rank0_chunk0_train_heads_targets.npy'
filename3 = 'rank0_chunk0_train_heads_features.npy'

data1 = np.load(common_path+filename1)
#data2 = np.load(common_path+filename2)
data3 = np.load(common_path+filename3)

#print(data1.shape,data2.shape,data3.shape)

data = data3

tsne = TSNE(n_components=2, random_state=5).fit_transform(data)
save_file(tsne, "/p/project/deepacf/kiste/DC/barbados/k7/leif_256_limited_aug/tsne_barbados_800ep.npy")


#from cuml.manifold import TSNE
#tsne = TSNE(n_components=2, random_state=5).fit_transform(data)
#save_file(tsne, "/p/project/deepacf/kiste/DC/barbados/k6/leif_256/tsne_barbados_800ep_cuml.npy")
