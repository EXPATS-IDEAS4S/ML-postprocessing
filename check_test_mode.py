
import numpy as np
import os, glob
import pandas as pd


run_name = '/data1/runs/dcv2_ir108_200x200_k9_expats_70k_200-300K_closed-CMA'

feature_trainig = f'{run_name}/features/'

feature_test = f'{run_name}/evalaution/'

#filename1 = 'rank0_chunk0_train_heads_inds.npy' 
#filename2 = 'rank0_chunk0_train_heads_targets.npy'
filename3 = 'rank0_chunk0_train_heads_features.npy'