""""
Compute the BT percentile for each crop in the merged dataframe
""" 

import os
import pandas as pd
import xarray as xr

# === CONFIG ===
RUN_NAME = "dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc"
crop_sel = "all"
epoch = "epoch_800"
n_subsets = 1000 
PERCENTILES = [1, 25, 50, 75]
BT_CLEAR_SKY = 320.0  # BT value to assign when no cloudy pixels are found
path_to_dir = f"/data1/fig/{RUN_NAME}/{epoch}/{crop_sel}/"
merged_path = os.path.join(path_to_dir, f"merged_crops_stats_all_bt_perc.csv")

#check if merged_path exists
if not os.path.exists(merged_path):
    #open crop list
    crop_list_path = f"/data1/fig/{RUN_NAME}/{epoch}/{crop_sel}/crop_list_dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc_all_140207.csv"
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

#remove column 'brightness_temperature' if exists
if 'brightness_temperature' in df.columns:
    df = df.drop(columns=['brightness_temperature'])

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

    #get only the pixels with BT<320K (cloudy pixels)
    cloudy_ir108 = ir108.where(ir108 < BT_CLEAR_SKY, drop=True)
    cloudy_pixels = cloudy_ir108.size
    print(f"Cloudy pixels (BT<{BT_CLEAR_SKY}K): {cloudy_pixels}")

    #compute BT percentiles for the cloudy pixels
    new_row = row.copy()
    new_row['var'] = 'bt'
    new_row['None'] = None
    for perc in PERCENTILES:
        if cloudy_pixels == 0:
            bt_perc_value = BT_CLEAR_SKY
            #print(f"no cloudy pixels)")
        else:
            bt_perc_value = cloudy_ir108.quantile(perc/100).item()
        #round to second decimal
        bt_perc_value = round(bt_perc_value, 2)
        print(f"BT {perc}th percentile: {bt_perc_value:.2f}K")
        if perc == 1:
            new_row['99'] = bt_perc_value
        else:
            new_row[str(perc)] = bt_perc_value
        
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    print(f"Added BT {perc}th percentile row for crop index {index}")
    
    
#save the dataframe
df.to_csv(merged_path, index=False)
print(f"Saved updated dataframe with BT percentiles to: {merged_path}")

#672113