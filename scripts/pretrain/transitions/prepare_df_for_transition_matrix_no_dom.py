import pandas as pd
import os
import numpy as np
import sys

sys.path.append("/home/Daniele/codes/VISSL_postprocessing/")
from scripts.pretrain.transitions.compute_transitions_utils import (
    check_continuous_timestamps
)

# NOTE: This version prepares data WITHOUT dominance features
# The wperc_label_* columns from the input CSV are not used for analysis

RUN_NAME = "dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc"
BASE_DIR = f"/data1/fig/{RUN_NAME}/epoch_800/test_traj"
OUT_DIR = f"{BASE_DIR}/pathway_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

path = f"{BASE_DIR}/hypersphere_analysis/test_vectors_neigh_with_stats.csv"
df = pd.read_csv(path, low_memory=False)
#print(df.columns.tolist())

#extract storm if from crop
df['storm_id'] = df['filename'].apply(lambda x: x.split('_')[0])

#extract datetime from filename column and create new column 'datetime' in df
df['datetime'] = df['filename'].apply(lambda x: x.split('_')[1])
df['datetime'] = pd.to_datetime(df['datetime'], format="%Y-%m-%dT%H-%M")

#check continuous timestamps for each storm_id
df_sum = check_continuous_timestamps(df, id_col="storm_id", time_col="datetime", freq='15min')
#print(df_sum)
#howm many entries have is_continuous == True
n_continuous = df_sum['is_continuous'].sum()
print(f"Number of continuous trajectories: {n_continuous} out of {len(df_sum)}")
#filter out non continuous trajectories
df_continuous = df[df['storm_id'].isin(df_sum[df_sum['is_continuous']==True]['storm_id'])]



#get list of unique datetime values
datetimes = df_continuous['datetime'].unique()
print(f"Unique datetimes: {datetimes}")

#path reports
report_dir = f"/home/Daniele/codes/ML_data_generator/test/essl/"
report_file = "storm_trajectories_after_merge.csv"
df_report = pd.read_csv(os.path.join(report_dir, report_file))
print(df_report.columns.tolist())

df_report["mean_precip_intensity"] = np.where(
    df_report["cluster_event_type"] == "PRECIP",
    df_report["mean_intensity"],
    np.nan,
)

df_report["mean_hail_intensity"] = np.where(
    df_report["cluster_event_type"] == "HAIL",
    df_report["mean_intensity"],
    np.nan,
)

df_report["max_precip_intensity"] = np.where(
    df_report["cluster_event_type"] == "PRECIP",
    df_report["max_intensity"],
    np.nan,
)

df_report["max_hail_intensity"] = np.where(
    df_report["cluster_event_type"] == "HAIL",
    df_report["max_intensity"],
    np.nan,
)


def resolve_storm_type(types):
    s = set(types.dropna())
    if len(s) == 1:
        return list(s)[0]
    return "MIXED"

def resolve_source(sources):
    s = set(sources.dropna())
    if "observed" in s:
        return "observed"
    if "interpolated" in s:
        return "interpolated"
    return "extrapolated"


collapsed = (
    df_report
    .groupby(["merged_storm_id", "time"])
    .agg(
        lat=("lat", "mean"),
        lon=("lon", "mean"),
        #storm_type=("storm_type", resolve_storm_type),
        cluster_event_type=("cluster_event_type", resolve_storm_type),
        source=("source", resolve_source),
        n_precip=("n_precip", "sum"),
        n_hail=("n_hail", "sum"),

        mean_precip_intensity=("mean_precip_intensity", "mean"),
        max_precip_intensity=("max_precip_intensity", "max"),

        mean_hail_intensity=("mean_hail_intensity", "mean"),
        max_hail_intensity=("max_hail_intensity", "max"),
    )
    .reset_index()
    .sort_values(["merged_storm_id", "time"])
)

print(f"Collapsed from {len(df_report)} → {len(collapsed)} rows")

collapsed["datetime"] = pd.to_datetime(collapsed["time"], utc=True)
collapsed["datetime"] = collapsed["datetime"].dt.tz_localize(None)
#filter out row with datetime not in datetimes
#collapsed["datetime"] = collapsed["datetime"].dt.floor("15min")

#list of unique datetimes
datetimes = collapsed['datetime'].unique()
print(f"Unique datetimes in collapsed: {datetimes}")

collapsed = collapsed[collapsed['datetime'].isin(datetimes)]
print(collapsed)
print(collapsed.columns.tolist())


collapsed["filename"] = (
    "storm" + collapsed["merged_storm_id"].astype(str)
    + "_" + collapsed["datetime"].dt.strftime("%Y-%m-%dT%H-%M")
    + "_lat" + collapsed["lat"].round(2).map(lambda v: f"{v:.2f}")
    + "_lon" + collapsed["lon"].round(2).map(lambda v: f"{v:.2f}")
    + "_" + collapsed["cluster_event_type"].astype(str)
    + "_" + collapsed["source"].astype(str)
    + ".nc"
)

#mergee df_report on df using filename (merge the columns: 'n_precip', 'n_hail', 'max_intensity', 'mean_intensity)
cols = ["filename", "n_precip", "n_hail", "max_precip_intensity", "mean_precip_intensity", "max_hail_intensity", "mean_hail_intensity"]
df = df_continuous.merge(
    collapsed[cols],
    on="filename",
    how="left",
)

#rename filename to crop
df = df.rename(columns={"filename": "crop"})
print(df)
print(df.columns.tolist())


aligned_times = []

for storm_id, g in df.groupby("storm_id"):
    g = g.sort_values("datetime")
    t0 = g["datetime"].iloc[0]
    t1 = g["datetime"].iloc[-1]
    #print(t0, t1)
    t_center = t0 + (t1 - t0) / 2

    aligned = (g["datetime"] - t_center).dt.total_seconds() / 3600.0
    aligned_times.append(aligned)

df["t_aligned"] = pd.concat(aligned_times).sort_index()

BIN_HOURS = 1.0
df["t_bin"] = (df["t_aligned"] / BIN_HOURS).round() * BIN_HOURS


#save the df to csv
output_path = os.path.join(OUT_DIR, "df_for_transition_matrix_no_dom.csv")
df.to_csv(output_path, index=False)
print(f"Saved prepared dataframe to {output_path}")





