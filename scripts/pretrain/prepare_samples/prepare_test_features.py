import numpy as np
import pandas as pd
import os
from glob import glob
import gc
import torch
from sklearn.metrics.pairwise import cosine_similarity
import xarray as xr

# ================= CONFIGURATION =================
run_name = 'dcv2_resnet_k8_ir108_100x100_2013-2020_1xrandomcrops_1xtimestamp_cma_nc_convective'

#features directories
feat_train_dir = f'/data1/runs/{run_name}/features/epoch_800/'
feat_test_dir = f'/data1/runs/{run_name}/test_features/epoch_800/'

feat_2d_train_dir = '/data1/fig/dcv2_resnet_k8_ir108_100x100_2013-2020_1xrandomcrops_1xtimestamp_cma_nc_convective/epoch_800/all'
feat_2d_test_dir = '/data1/fig/dcv2_resnet_k8_ir108_100x100_2013-2020_1xrandomcrops_1xtimestamp_cma_nc_convective/test'
feat_2d_train_file = 'tsne_opentsne_dcv2_resnet_k8_ir108_100x100_2013-2020_1xrandomcrops_1xtimestamp_cma_nc_convective_3_epoch_800.npy'
feat_2d_test_file = 'tsne_opentsne_dcv2_resnet_k8_ir108_100x100_2013-2020_1xrandomcrops_1xtimestamp_cma_nc_convective_3_epoch_800'  

# Output path
output_path = f'/data1/fig/{run_name}/test/'
os.makedirs(output_path, exist_ok=True)

# Paths to crops
image_train_path = '/data1/crops/ir108_100x100_2013-2020_1xrandomcrops_1xtimestamp_cma_nc_convective/nc/1/'
image_test_path = '/data1/crops/test_case_essl_2021-2025_100x100_ir108_cma/nc/1/'

# Summary directory
summury_events_dir = "/data1/crops/test_case_essl_2021-2025_100x100_ir108_cma"
event_types = ["PRECIP", "HAIL"]

# Feature files
feature_file_inds = 'rank0_chunk0_train_heads_inds.npy'
feature_file_features = 'rank0_chunk0_train_heads_features.npy'

#training assignemnts
train_assignments_file = f'/data1/runs/{run_name}/checkpoints/assignments.pt'
train_distances_file = f'/data1/runs/{run_name}/checkpoints/distances.pt'
train_centroids_file = f'/data1/runs/{run_name}/checkpoints/centroids0.pt'


# =========== FUNCTIONS =====================================

def extract_timestamp(path: str, extension: str = '.nc', from_filename: bool = True) -> str:
    """Extract timestamp from crop file path."""
    if from_filename:
        filename = os.path.basename(path)
        timestamp = filename.replace(extension, '').split('_')[0]
        #convert to datetime format
        timestamp = pd.to_datetime(timestamp).strftime('%Y%m%d_%H%M')
    else:
        ds = xr.open_dataset(path, engine ='h5netcdf')
        timestamp = pd.to_datetime(ds.time.values).strftime('%Y%m%d_%H%M')

    return timestamp

def extract_latlon_centre(path: str, from_filename: bool = False) -> str:
    """Extract lat and lon."""
    if from_filename:
        filename = os.path.basename(path)
        filename_without_ext = filename.replace('.nc', '')
        lat_str = filename_without_ext.split('_')[1]
        lon_str = filename_without_ext.split('_')[2]
        lat_centre = float(lat_str.replace('lat', ''))
        lon_centre = float(lon_str.replace('lon', ''))
    else:
        ds = xr.open_dataset(path, engine ='h5netcdf')
        lat = ds.lat.values
        lon = ds.lon.values
        #compute middle point
        lat_centre = (lat.min() + lat.max()) / 2.0
        lon_centre = (lon.min() + lon.max()) / 2.0
        #round to two decimals
        lat_centre = round(float(lat_centre), 2)
        lon_centre = round(float(lon_centre), 2)

    return lat_centre, lon_centre

def load_features_to_df(
    feature_path: str, indices_file: str, features_file: str, 
    assignments_file: str, distances_file: str, centroids_file: str, 
    vector_type: str, crops_path: str, include_timestamp: bool, 
    include_latlon: bool, subsample: int = None, extract_latlon_from_filename: bool = False,
    features_2d_file: str = None, features_2d_dir: str = None
) -> pd.DataFrame:
    """Load features and indices, with an optional simple first-N subsample."""

    # ----------------------------
    # 1. Load raw data
    # ----------------------------
    indices = np.load(os.path.join(feature_path, indices_file))
    features = np.load(os.path.join(feature_path, features_file))

    if subsample is not None and subsample > 0:
        # Apply subsample mask everywhere
        indices = indices[:subsample]
        features = features[:subsample]

    crop_test_paths = sorted(glob(os.path.join(crops_path, '*.nc')))
    #apply masto to crop_test_paths if subsample is set
    if subsample is not None and subsample > 0:
        crop_test_paths = crop_test_paths[:subsample]

    # ----------------------------
    # 3. Build DataFrame
    # ----------------------------
    df = pd.DataFrame(
        features,
        columns=[f"dim_{i+1}" for i in range(features.shape[1])]
    )
    df["path"] = crop_test_paths

    # ----------------------------
    # 4. Optional: timestamps
    # ----------------------------
    if include_timestamp:
        #apply subsample mask 
        all_timestamps = [extract_timestamp(p) for p in crop_test_paths]
        df["datetime"] = all_timestamps
        df["datetime"] = pd.to_datetime(df["datetime"], format="%Y%m%d_%H%M") 

    # ----------------------------
    # 5. Optional: lat/lon
    # ----------------------------
    if include_latlon:
        all_latlon = [extract_latlon_centre(p, extract_latlon_from_filename) for p in crop_test_paths]
        #transfom it to a numpy array
        all_latlon = np.array(all_latlon)
        df["lat_centre"] = all_latlon[:, 0] 
        df["lon_centre"] = all_latlon[:, 1] 

    # ----------------------------
    # 6. Add labels and distances
    # ----------------------------
    if vector_type == "TRAIN":
        assignments = torch.load(os.path.join(feature_path, assignments_file),
                                 map_location="cpu")[0].cpu().numpy()
        distances = torch.load(os.path.join(feature_path, distances_file),
                               map_location="cpu")[0].cpu().numpy()

        df["label"] = assignments[indices]
        df["distance"] = distances[indices]

    else:
        df = add_labels_to_test(df, centroids_file)

    df["vector_type"] = vector_type

    #add 2d features if required
    if features_2d_file is not None and features_2d_dir is not None:
        if vector_type != "TRAIN":
            features_2d_file = feat_2d_test_file + '_' + vector_type + '.npy'
        features_2d = np.load(os.path.join(features_2d_dir, features_2d_file))
        #print(f"2D features shape: {features_2d.shape}") #shape (n_samples, 2)
        if subsample is not None and subsample > 0:
            features_2d = features_2d[:subsample]
        for i in range(features_2d.shape[1]):
            df[f'2d_dim_{i+1}'] = features_2d[:, i]

    return df.reset_index(drop=True)


def add_labels_to_test(df: pd.DataFrame, centroids_file: str) -> pd.DataFrame:
    """Assign labels to test DataFrame based on cosine similarity to centroids."""
    # Load centroids
    centroids = torch.load(centroids_file, map_location="cpu").cpu().numpy()
    print(f"Centroids shape: {centroids.shape}")
    # Extract features from dataframe
    features = df[[f'dim_{i+1}' for i in range(centroids.shape[1])]].values
    print(f"Features shape: {features.shape}")
    
    # Compute cosine similarity: shape (n_samples, n_centroids)
    sim = cosine_similarity(features, centroids)
    print(f"Cosine similarity shape: {sim.shape}")
    
    # Assign label of most similar centroid
    df["label"] = np.argmax(sim, axis=1)
    
    # Optionally, also keep the similarity score for confidence
    df["distance"] = np.max(sim, axis=1)
    
    return df


def load_summary(event_type: str, summary_dir: str):
    """Load and clean ESSL summary CSV for an event type."""
    summary_path = os.path.join(summary_dir, f"{event_type}_summary.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    df_sum = pd.read_csv(summary_path, low_memory=False)

    # Standardize datetime column
    if "day_id" in df_sum.columns:
        df_sum["day_id"] = pd.to_datetime(df_sum["day_id"], errors="coerce")
    elif "date" in df_sum.columns:
        df_sum["day_id"] = pd.to_datetime(df_sum["date"], errors="coerce")

    return df_sum



def merge_test_with_summary(df_test: pd.DataFrame, df_summary: pd.DataFrame):
    """
    Merge df_test with df_summary using:
      1) SAME DATE
      2) NEAREST (lat_centre, lon_centre) when multiple summary entries exist.
      3) Both df_test AND df_summary may contain multiple cases per day.
      
    Cases are grouped by (date, lat_centre, lon_centre) on both sides.
    """

    # --- Normalize dates --- 
    #df_test["datetime"] = pd.to_datetime(df_test["datetime"], format="%Y%m%d_%H%M") 
    df_test["date"] = df_test["datetime"].dt.date 
    df_summary["date"] = pd.to_datetime(df_summary["day_id"], errors="coerce").dt.date

    merged_rows = []

    # Group test data by (date, lat, lon)
    for (date, lat_t, lon_t), df_group in df_test.groupby(["date", "lat_centre", "lon_centre"]):
        print(f"Merging for date: {date}, lat: {lat_t}, lon: {lon_t}")
        #print(df_group)

        # All summary rows on same date
        df_sum_day = df_summary[df_summary["date"] == date]
        #print(df_sum_day)

        # If no match at all → fill summary fields with NaN
        if df_sum_day.empty:
            df_out = df_group.copy()
            for col in df_summary.columns:
                if col not in df_out.columns:
                    df_out[col] = np.nan
            merged_rows.append(df_out)
            print(f"No summary match for date {date} in test data.")
            print(df_out)
            continue

        print(f"Found {len(df_sum_day)} summary entries for date {date}.")

        # Compute distance to all summary clusters on that day
        sum_lats = df_sum_day["cluster_lat"].values
        sum_lons = df_sum_day["cluster_lon"].values
        print(sum_lats, sum_lons)

        # squared distance
        dist = (lat_t - sum_lats)**2 + (lon_t - sum_lons)**2
        nearest_idx = np.argmin(dist)

        # Select the single matching summary row
        df_sum_match = df_sum_day.iloc[[nearest_idx]]

        # Merge the metadata into the group
        df_sum_match = df_sum_match.reset_index(drop=True)
        #remove some columns to avoid duplication
        df_sum_match = df_sum_match.drop(columns=["day_id", "cluster_lat", "cluster_lon", "cluster_id"])
        df_out = df_group.merge(df_sum_match, on="date", how="left")
        #delete the 'date' column to avoid duplication
        df_out = df_out.drop(columns=["date"])

        merged_rows.append(df_out)

    # Combine back
    return pd.concat(merged_rows, ignore_index=True)




def prepare_and_save_dataset():
    """Main function to load, merge, and save training and test datasets."""
    gc.collect()

    #Load train features
    df_train = load_features_to_df(
        feature_path=feat_train_dir,
        indices_file=feature_file_inds,
        features_file=feature_file_features,
        assignments_file=train_assignments_file,
        distances_file=train_distances_file,
        centroids_file=train_centroids_file,
        vector_type="TRAIN",
        crops_path=image_train_path,
        include_timestamp=True,
        include_latlon=True,
        subsample=None,
        extract_latlon_from_filename=False,
        features_2d_file=feat_2d_train_file,
        features_2d_dir=feat_2d_train_dir
    )
    #print(df_train.columns.to_list())
 

    df_test_list = []
    for event_type in event_types:
        image_test_path = f'{summury_events_dir}/{event_type}/nc/1/'
        test_feat_dir = f'/data1/runs/{run_name}/test_features/epoch_800/{event_type}/'

        # ---- Load TEST FEATURES ----
        df_test = load_features_to_df(
            feature_path=test_feat_dir,
            indices_file=feature_file_inds,
            features_file=feature_file_features,
            assignments_file=train_assignments_file,
            distances_file=train_distances_file,
            centroids_file=train_centroids_file,
            vector_type=event_type,
            crops_path=image_test_path,
            include_timestamp=True,
            include_latlon=True,
            subsample=None,
            extract_latlon_from_filename=True,
            features_2d_file=feat_2d_test_file,
            features_2d_dir=feat_2d_test_dir
        )
        #print(df_test)

        # ---- Load EVENT SUMMARY ----
        df_summary = load_summary(event_type, "/data1/crops/test_case_essl_2021-2025_100x100_ir108_cma")
        #print(df_summary)

        # ---- Merge EVENT METADATA ----
        df_test = merge_test_with_summary(df_test, df_summary)
        #print(df_test.columns.to_list())

        df_test_list.append(df_test)

    
    # Merge datasets
    df_final = pd.concat([df_train, *df_test_list], ignore_index=True)
    print(df_final.columns.to_list())
    print(f"Final dataset shape: {df_final.shape}")
    print(df_final)
    print(df_final['datetime'],df_final['start_time'], df_final['end_time'])
    
    # Save event-specific dataset
    output_test_csv = os.path.join(output_path, f'features_train_test_{run_name}.csv')
    df_final.to_csv(output_test_csv, index=False)
    print(f"Saved merged dataset with metadata to: {output_test_csv}")


if __name__ == "__main__":
    prepare_and_save_dataset()
