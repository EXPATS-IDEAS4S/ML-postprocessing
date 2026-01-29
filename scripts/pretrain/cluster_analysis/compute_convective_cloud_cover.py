
import os
import pandas as pd
import xarray as xr

# === CONFIG ===
RUN_NAME = "dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc"
crop_sel = "all"
epoch = "epoch_800"
n_subsets = 140207
BT_THRESHOLD = 240  # Brightness temperature threshold for convective cloud cover (ref sandwich product from EUMETSAT)
MIN_CLOUDY_PIXELS = 100  # Minimum number of cloudy pixels to compute convective cloud cover
path_to_dir = f"/data1/fig/{RUN_NAME}/{epoch}/{crop_sel}/"
merged_path = os.path.join(path_to_dir, f"merged_crops_stats_all_cvc.csv")

#check if merged_path exists
if not os.path.exists(merged_path):
    #open crop list
    crop_list_path = f"/data1/fig/{RUN_NAME}/{epoch}/{crop_sel}/crop_list_{RUN_NAME}_{crop_sel}_{n_subsets}.csv"
    df_crops = pd.read_csv(crop_list_path)
    print(df_crops.head())
    #extrect crop from path
    df_crops['crop'] = df_crops['path'].apply(lambda x: os.path.basename(x))
    #open stats file
    stats_path = f"/data1/fig/{RUN_NAME}/{epoch}/{crop_sel}/crops_stats_vars-cth-cma-cot-precipitation-euclid_msg_grid_stats-50-99-25-75_frames-1_coords-datetime_dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc_all_140207.csv"
    df_stats = pd.read_csv(stats_path)
    print(df_stats.head())
    #merge dataframes on crop
    df = pd.merge(df_crops, df_stats, on='crop', how='inner')
    #save merged dataframe
    print(f"Merged dataframe shape: {df.shape}")
    df.to_csv(merged_path, index=False)
    print(f"Saved merged dataframe to: {merged_path}")
    
# === LOAD DATA ===
df = pd.read_csv(merged_path)
print(f"✅ Loaded dataframe: {merged_path} ({df.shape})")
print(df)

#remove column 'convective_cloud_cover' if exists
if 'convective_cloud_cover' in df.columns:
    df = df.drop(columns=['convective_cloud_cover'])

print(df.columns)

#make a list of the unique crop_indexes
crop_indices = df['crop_index'].unique()
print(f"Number of unique crop indices: {len(crop_indices)}")

#loop ovet the indexes and compute convective cloud cover
for index in crop_indices:
    print(f"Crop {index}:")
    row = df[df['crop_index'] == index].iloc[0]
    print(f"Path: {row['path']}")
    #open the nc file
    ds = xr.open_dataset(row['path'])
    #print(ds)
    #select the ir108 variable
    ir108 = ds['IR_108']

    cloudy_pixels = ir108.where(ir108 < 320, drop=True)
    #count how many pixels are below the BT_THRESHOLD, normalized by total number of pixels
    total_pixels = ir108.size
    #print(f"Total pixels: {total_pixels}")
    convective_pixels = (cloudy_pixels < BT_THRESHOLD)
    convective_pixels_tot = (cloudy_pixels < BT_THRESHOLD).sum().item()

    #find cloudy pixels (BT<320K)
    cloud_pixels_tot = (ir108 < 320).sum().item()

    #count how many cloud pixels (BT>300K)
    #cloud_pixels = (ir108 < 300).sum().item()
    if (cloud_pixels_tot <= MIN_CLOUDY_PIXELS) or (convective_pixels_tot <= MIN_CLOUDY_PIXELS): # if less than 10 cloudy pixels, set convective cloud cover to 0
        convective_cloud_cover = 0.0
    else:
        convective_cloud_cover = convective_pixels_tot / cloud_pixels_tot   # percentage
    #round to second decimal
    convective_cloud_cover = round(convective_cloud_cover, 2)
    print(f"Convective cloud cover (% of pixels with BT < {BT_THRESHOLD}K): {convective_cloud_cover:.2f}%")

    #insert a new row in the dataframe with same values of the current row but not for
    # var',': 'ccv', '50'=None, '99'+None, '25'=None, '75'=None, 'None'= convective_cloud_cover
    new_row = row.copy()
    new_row['var'] = 'ccv'
    new_row['25'] = None
    new_row['50'] = None
    new_row['75'] = None
    new_row['99'] = None
    new_row['None'] = convective_cloud_cover
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    print(f"Added convective cloud cover row for crop index {index}")
    
    
#save the dataframe
df.to_csv(merged_path, index=False)
print(f"Saved updated dataframe with convective cloud cover to: {merged_path}")

