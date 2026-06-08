"""plot class population statistics for each cluster


"""



import os
import sys
import numpy as np
import pandas as pd
from glob import glob
import sys
import pdb

# === IMPORT HELPER FUNCTIONS ===
sys.path.append("/home/claudia/codes/ML_postprocessing")
from utils.processing.features_utils import load_tsne_coordinates
from utils.plotting.class_colors import colors_per_class1_names
from utils.configs import load_config



def main():

    # read the csv file containing the crop statistics for the run
    input_dir = "/sat_data/output/grl_2026/csv/"
    csv_filename = "crops_stats_var-cth_stats-50-95-25-75_frames-8_timedim_grl_2026_all_240216_imergmin.csv"
    csv_prec_data = "crops_stats_var-precipitation_stats-50-95-25-75_frames-8_timedim_grl_2026_all_240216_imergmin.csv"
    csv_cot_data = "crops_stats_var-cot_stats-50-95-25-75_frames-8_timedim_grl_2026_all_240216_imergmin.csv"


    # load the crop statistics CSV file
    crop_stats_df = pd.read_csv(os.path.join(input_dir, csv_filename))
    crop_stats_prec_df = pd.read_csv(os.path.join(input_dir, csv_prec_data))
    crop_stats_cot_df = pd.read_csv(os.path.join(input_dir, csv_cot_data))
    print("Crop statistics CSV files loaded successfully.")

    # read class numbers from colors_per_class1_names
    class_numbers = list(colors_per_class1_names.keys())    
    print(f"Class numbers: {class_numbers}")

    # for each class number, calculate the total amount of samples for that class in each file
    class_counts = {}
    for class_num in class_numbers:

        crop_tot = len(crop_stats_df[crop_stats_df["label"] == int(class_num)])
        # to find total number of cot values, check that columns 50 is not empty
        cot_tot = len(crop_stats_cot_df[(crop_stats_cot_df["label"] == int(class_num)) & (crop_stats_cot_df["50"].notna())])

        class_counts[class_num] = {
            "crop_stats": crop_tot,
            "cot": cot_tot,
        }
    # print the class counts
    for class_num, counts in class_counts.items():
        print(f"Class {class_num}:")
        print(f"  Crop Stats: {counts['crop_stats']} samples")
        print(f"  COT: {counts['cot']} samples")
    
    # plot a bar chart of the class counts for each file
    import matplotlib.pyplot as plt
    x = np.arange(len(class_numbers))  # the label locations
    width = 0.35  # the width of the bars for two side-by-side series
    fig, ax = plt.subplots(figsize=(10, 6))
    crop_stats_counts = [class_counts[class_num]["crop_stats"] for class_num in class_numbers]
    cot_counts = [class_counts[class_num]["cot"] for class_num in class_numbers]
    rects1 = ax.bar(x - width / 2, crop_stats_counts, width, label='Crop Stats')
    rects3 = ax.bar(x + width / 2, cot_counts, width, label='COT')

    ax.set_xlabel('Class')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Class Population Statistics')
    ax.set_xticks(x)
    ax.set_xticklabels(class_numbers)
    ax.legend(frameon=False)

    # set spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(axis='both', which='both', length=0)
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='gray')


    # save plot in figures directory
    output_dir = "/sat_data/output/grl_2026/figs/"
    output_filename = "class_population_statistics.png"
    plt.savefig(os.path.join(output_dir, output_filename))
    print(f"Class population statistics plot saved to {os.path.join(output_dir, output_filename)}") 



    # drop class with label -100
    crop_stats_df = crop_stats_df[crop_stats_df["label"] != -100]


    # counting now video classified for each class
    unique_crop_names = crop_stats_df["crop"].drop_duplicates()

    crop_stats_videos_df = crop_stats_df[crop_stats_df["crop"].isin(unique_crop_names)].drop_duplicates(subset="crop")
    video_class_counts = crop_stats_videos_df["label"].value_counts().sort_index()

    # make barplot of video class counts
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(video_class_counts.index.astype(str), video_class_counts.values, color=[colors_per_class1_names[str(int(class_num))] for class_num in video_class_counts.index])
    ax.set_xlabel('Class')
    ax.set_ylabel('Number of Videos')
    ax.set_title('Class Population Statistics for Videos')
    ax.set_xticks(video_class_counts.index.astype(str))
    ax.set_xticklabels(video_class_counts.index.astype(str))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)   
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(axis='both', which='both', length=0)
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='gray')
    output_filename = "class_population_statistics_videos.png"
    plt.savefig(os.path.join(output_dir, output_filename))
    print(f"Class population statistics for videos plot saved to directory {os.path.join(output_dir, output_filename)}")


if __name__ == "__main__":

    main()
