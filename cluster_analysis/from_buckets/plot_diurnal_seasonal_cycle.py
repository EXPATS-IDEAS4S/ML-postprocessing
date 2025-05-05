import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import cmcrameri.cm as cmc

from aux_functions_from_buckets import get_variable_info, extract_datetime

sys.path.append(os.path.abspath("/home/Daniele/codes/visualization/cluster_analysis"))

# Configuration
reduction_method = 'tsne'
run_name = 'dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA'
random_state = '3'
sampling_type = 'all'
output_path = f'/data1/fig/{run_name}/{sampling_type}/'

# Open dataframe
df = pd.read_csv(f'{output_path}merged_tsne_variables_{run_name}_{sampling_type}_{random_state}.csv')
print(df.columns)


def safe_parse_datetime(path):
    try:
        dt_dict = extract_datetime(os.path.basename(path))
        date_str = "{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:00".format(**dt_dict)
        return pd.to_datetime(date_str, errors='coerce')
    except Exception as e:
        print(f"Failed to parse datetime for {path}: {e}")
        return pd.NaT

# Extract datetime from the path
df.loc[:, 'datetime'] = df['path'].apply(safe_parse_datetime)

# Drop rows where datetime parsing failed
df = df.dropna(subset=['datetime'])
#print(df.datetime)

# Ensure 'hour' and 'month' columns are integers
df['hour'] = df['hour'].astype(int)
df['month'] = df['month'].astype(int)

# Filter months between April (4) and September (9)
df = df[df['month'].between(4, 9)]

# Dictionary to map months to names
month_names = {4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep"}
df['month'] = df['month'].map(month_names)


# Sort by timestamp to preserve chronological order
df['datetime'] = pd.to_datetime(df['datetime'])  # Ensure datetime exists
df = df.sort_values('datetime')

# Shift label and datetime for comparison
df['label_shifted'] = df['label'].shift()
df['time_shifted'] = df['datetime'].shift()
print(df['label_shifted'])
print(df['time_shifted'])
exit()

# Calculate time gap in minutes
df['gap'] = (df['datetime'] - df['time_shifted']).dt.total_seconds() / 60.0

# Define new block: label changed OR time gap too big (e.g. > 30 min)
df['is_new_block'] = (df['label'] != df['label_shifted']) | (df['gap'] > 30)

# Filter to get one row per block
df_filtered = df[df['is_new_block']].copy()


# Compute diurnal distribution (based on filtered non-repeating blocks)
hourly_counts = df_filtered.groupby(['hour', 'label']).size().unstack(fill_value=0)
hourly_percentage = hourly_counts.div(hourly_counts.sum(axis=1), axis=0) * 100

# Compute seasonal distribution
monthly_counts = df_filtered.groupby(['month', 'label']).size().unstack(fill_value=0)
monthly_percentage = monthly_counts.div(monthly_counts.sum(axis=1), axis=0) * 100


# --- PLOTTING SETTINGS ---
sns.set_context("talk")  # Set larger default sizes
fontsize_title = 20
fontsize_labels = 16
fontsize_ticks = 14

# Ensure output directories exist
diurnal_output_dir = os.path.join(output_path, "diurnal_lineplots")
seasonal_output_dir = os.path.join(output_path, "seasonal_lineplots")
os.makedirs(diurnal_output_dir, exist_ok=True)
os.makedirs(seasonal_output_dir, exist_ok=True)

# Get unique labels and their associated colors
label_colors = df[['label', 'color']].drop_duplicates().set_index('label')['color'].to_dict()

# 1. Diurnal Distribution - Separate Line Plots
for label in hourly_percentage.columns:
    plt.figure(figsize=(6, 4))
    
    plt.plot(hourly_percentage.index, hourly_percentage[label], 
             label=f'Class {label}', linewidth=2, color=label_colors.get(label, 'black'))  # Default to black if missing

    plt.title(f"Diurnal Distribution - Class {label}", fontsize=fontsize_title)
    plt.ylim(0, 40)
    plt.xlabel("Hour of Day", fontsize=fontsize_labels)
    plt.ylabel("Percentage", fontsize=fontsize_labels)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    
    #plt.legend(loc="upper right", fontsize=fontsize_ticks)
    plt.grid(True, linestyle="--", alpha=0.7)

    output_file = os.path.join(diurnal_output_dir, f"diurnal_distribution_label_{label}.png")
    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {output_file}")

# 2. Seasonal Distribution - Separate Line Plots
for label in monthly_percentage.columns:
    plt.figure(figsize=(6, 4))
    
    plt.plot(monthly_percentage.index, monthly_percentage[label], 
             label=f'Class {label}', linewidth=2, color=label_colors.get(label, 'black'))

    plt.title(f"Seasonal Distribution - Class {label}", fontsize=fontsize_title)
    plt.xlabel("Month", fontsize=fontsize_labels)
    plt.ylabel("Percentage", fontsize=fontsize_labels)
    plt.ylim(0, 40)

    plt.xticks(ticks=np.arange(len(month_names)), labels=list(month_names.values()), fontsize=fontsize_ticks, rotation=0)
    plt.yticks(fontsize=fontsize_ticks)

    #plt.legend(loc="upper right", fontsize=fontsize_ticks)
    plt.grid(True, linestyle="--", alpha=0.7)

    output_file = os.path.join(seasonal_output_dir, f"seasonal_distribution_label_{label}.png")
    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {output_file}")

# # 1. Diurnal Distribution - Heatmap
# plt.figure(figsize=(12, 6))
# ax = sns.heatmap(hourly_percentage.T.iloc[::-1], cmap=cmc.lapaz, annot=False, linewidths=0.5, 
#                  cbar_kws={'format': '%.0f%%'}, xticklabels=2)

# plt.title("Diurnal Distribution of Classes (%)", fontsize=fontsize_title)
# plt.xlabel("Hour of Day", fontsize=fontsize_labels)
# plt.ylabel("Class Label", fontsize=fontsize_labels)
# plt.xticks(fontsize=fontsize_ticks)
# plt.yticks(fontsize=fontsize_ticks)

# plt.savefig(f"{output_path}diurnal_distribution_heatmap.png", bbox_inches="tight")
# plt.show()

# # 2. Seasonal Distribution - Heatmap
# plt.figure(figsize=(12, 6))
# ax = sns.heatmap(monthly_percentage.T.iloc[::-1], cmap=cmc.lapaz, annot=False, linewidths=0.5, 
#                  cbar_kws={'format': '%.0f%%'})

# # Adjust x-axis ticks for months (centered)
# ax.set_xticks(np.arange(len(month_names)) + 0.5)
# ax.set_xticklabels(list(month_names.values()), fontsize=fontsize_ticks)

# plt.title("Seasonal Distribution of Classes (%)", fontsize=fontsize_title)
# plt.xlabel("Month", fontsize=fontsize_labels)
# plt.ylabel("Class Label", fontsize=fontsize_labels)
# plt.yticks(fontsize=fontsize_ticks)

# plt.savefig(f"{output_path}seasonal_distribution_heatmap.png", bbox_inches="tight")
# plt.show()

# # 3. Diurnal Distribution - Line Plot
# plt.figure(figsize=(12, 6))
# for label in hourly_percentage.columns:
#     plt.plot(hourly_percentage.index, hourly_percentage[label], label=f'Class {label}', linewidth=2)

# plt.title("Diurnal Distribution of Classes (%)", fontsize=fontsize_title)
# plt.xlabel("Hour of Day", fontsize=fontsize_labels)
# plt.ylabel("Percentage", fontsize=fontsize_labels)
# plt.xticks(fontsize=fontsize_ticks)
# plt.yticks(fontsize=fontsize_ticks)

# plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=fontsize_ticks)
# plt.grid(True, linestyle="--", alpha=0.7)

# plt.savefig(f"{output_path}diurnal_distribution_lineplot.png", bbox_inches="tight")
# plt.show()

# # 4. Seasonal Distribution - Line Plot
# plt.figure(figsize=(12, 6))
# for label in monthly_percentage.columns:
#     plt.plot(monthly_percentage.index, monthly_percentage[label], label=f'Class {label}', linewidth=2)

# plt.title("Seasonal Distribution of Classes (%)", fontsize=fontsize_title)
# plt.xlabel("Month", fontsize=fontsize_labels)
# plt.ylabel("Percentage", fontsize=fontsize_labels)

# # Adjust x-axis ticks for proper month alignment
# plt.xticks(ticks=np.arange(len(month_names)), labels=list(month_names.values()), fontsize=fontsize_ticks, rotation=0)
# plt.yticks(fontsize=fontsize_ticks)

# plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=fontsize_ticks)
# plt.grid(True, linestyle="--", alpha=0.7)

# plt.savefig(f"{output_path}seasonal_distribution_lineplot.png", bbox_inches="tight")
# plt.show()
