import os
import pandas as pd


# =========================
# PATHS
# =========================
run_name = "dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc"
base_path = f"/data1/fig/{run_name}/epoch_800/test_traj"
output_dir = os.path.join(base_path, "hypersphere_analysis")

neighbors_csv = os.path.join(
    output_dir,
    "test_vectors_local_label_composition.csv"
)

tsne_csv = os.path.join(
    base_path,
    "tsne_all_vectors_with_centroids.csv"
)

stats_csv = os.path.join(
    base_path,
    "merged_crops_stats_cot_cth_fractions.csv"
)


df = pd.read_csv(neighbors_csv)
df_tsne = pd.read_csv(tsne_csv)
df_tsne_test = df_tsne[(df_tsne["vector_type"] != "TRAIN") & (df_tsne["vector_type"] != "CENTROID")]

# merge tsne coords
df = df.merge(
    df_tsne_test[["filename", "tsne_dim_1", "tsne_dim_2"]],
    on="filename",
    how="left"
)

#open df_stats to get lat
df_stats = pd.read_csv(stats_csv, low_memory=False)
#rename crop column to filename in df_stats
df_stats = df_stats.rename(columns={"crop": "filename"})
print(df_stats)
print(df.columns.tolist())
#add columns to df related to the var values in df_stats (e.g columns var has the values 'precipitation', 'euclid_msg_grid' which valus are in column 'None')
#loop over the rows in df
for idx, row in df.iterrows():
    filename = row["filename"]
    #get the corresponding row in df_stats
    stats_rows = df_stats[df_stats["filename"] == filename]
    print(f"Processing {filename}, found {len(stats_rows)} stats rows")
    for _, stats_row in stats_rows.iterrows():
        var = stats_row["var"]
        if var in ['cth','precipitation']:
            for perc in ['25','50', '75', '99']:
                value = stats_row[perc]
                df.at[idx, var+perc] = value
                print(f"  Added {var+perc}: {value}")
                #print(df.at[idx, var+perc])
        else:
            value = stats_row['None']
            df.at[idx, var] = value
            print(f"  Added {var}: {value}")
            #print(df.at[idx, var])   


print(df.columns.tolist())

#save merged df 
output_csv = os.path.join(
    output_dir,
    "test_vectors_neigh_with_stats.csv"
)

df.to_csv(output_csv, index=False)
print(f"Saved merged dataframe to {output_csv}")

