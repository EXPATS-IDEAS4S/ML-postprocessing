import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === CONFIG ===
RUN_NAME = "dcv2_vit_k10_ir108_100x100_2013-2020_3xrandomcrops_1xtimestamp_cma_nc"
crop_sel = "closest"
epoch = "epoch_800"
n_subsets = 1000 
path_to_dir = f"/data1/fig/{RUN_NAME}/{epoch}/{crop_sel}/"
merged_path = os.path.join(path_to_dir, f"merged_crops_stats_alltime_{crop_sel}_{n_subsets}.csv")