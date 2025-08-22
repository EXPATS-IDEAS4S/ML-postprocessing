import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
#import sys
#import cmcrameri.cm as cmc

from aux_functions_from_buckets import get_variable_info, extract_datetime

#sys.path.append(os.path.abspath("/home/Daniele/codes/visualization/cluster_analysis"))

# Configuration
reduction_method = 'tsne'
run_name = 'dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA'
random_state = '3'
sampling_type = 'all'
output_path = f'/data1/fig/{run_name}/{sampling_type}/'

diurnal_output_dir = os.path.join(output_path, "diurnal_lineplots")
seasonal_output_dir = os.path.join(output_path, "seasonal_lineplots")
variable_output_dir = os.path.join(output_path, "variable_lineplots")
os.makedirs(diurnal_output_dir, exist_ok=True)
os.makedirs(seasonal_output_dir, exist_ok=True)
os.makedirs(variable_output_dir, exist_ok=True)

# Plotting config
sns.set_context("talk")
fontsize_title = 18
fontsize_labels = 16
fontsize_ticks = 16
line_thickness = 3

colors_per_class1_names = {
    '0': 'darkgray', 
    '1': 'darkslategrey',
    '2': 'peru',
    '3': 'orangered',
    '4': 'lightcoral',
    '5': 'deepskyblue',
    '6': 'purple',
    '7': 'lightblue',
    '8': 'green'
}

VARIABLE_INFO = {
    'cot': {'long_name': 'Cloud Optical Thickness', 'units': None, 'dir': 'incr', 'log': True, 'limit': (0.1, 150)},
    'cth': {'long_name': 'Cloud Top Height', 'units': 'Km', 'dir': 'incr', 'log': False, 'limit': (0, 14)},
    'cma': {'long_name': 'Cloud Cover', 'units': None, 'dir': 'incr', 'log': False, 'limit': (0, 1)},
    'cph': {'long_name': 'Ice ratio', 'units': None, 'dir': 'incr', 'log': False, 'limit': (0, 1)},
    'precipitation': {'long_name': 'Rain Rate', 'units': 'mm/h', 'dir': 'incr', 'log': False, 'limit': (0, 50)}
}

month_names = {4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep"}

def safe_parse_datetime(path):
    try:
        dt_dict = extract_datetime(os.path.basename(path))
        date_str = "{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:00".format(**dt_dict)
        return pd.to_datetime(date_str, errors='coerce')
    except Exception as e:
        print(f"Failed to parse datetime for {path}: {e}")
        return pd.NaT

def prepare_dataframe(csv_path, apply_block_filter=True):
    df = pd.read_csv(csv_path)
    df['datetime'] = df['path'].apply(safe_parse_datetime)
    df.dropna(subset=['datetime'], inplace=True)
    df['hour'] = df['hour'].astype(int)
    df['month'] = df['month'].astype(int)
    df = df[df['month'].between(4, 9)]
    df['month'] = df['month'].map(month_names)
    df = df.sort_values('datetime')

    if apply_block_filter:
        df['label_shifted'] = df['label'].shift()
        df['time_shifted'] = df['datetime'].shift()
        df['gap'] = (df['datetime'] - df['time_shifted']).dt.total_seconds() / 60.0
        df['is_new_block'] = (df['label'] != df['label_shifted']) | (df['gap'] > 30)
        df = df[df['is_new_block']].copy()

    return df

def plot_diurnal_class_distribution(df):
    hourly_counts = df.groupby(['hour', 'label']).size().unstack(fill_value=0)
    hourly_percentage = hourly_counts.div(hourly_counts.sum(axis=1), axis=0) * 100

    for label in hourly_percentage.columns:
        plt.figure(figsize=(6, 4))
        label_str = str(label)
        plt.plot(
            hourly_percentage.index,
            hourly_percentage[label],
            label=f'Class {label}',
            linewidth=line_thickness,
            color=colors_per_class1_names.get(label_str, 'black')
        )
        plt.title(f"Diurnal Distribution - Class {label}", fontsize=fontsize_title)
        plt.ylim(0, 27)
        plt.xlabel("Hour of Day (UTC)", fontsize=fontsize_labels)
        plt.ylabel("Percentage", fontsize=fontsize_labels)
        plt.xticks(fontsize=fontsize_ticks)
        plt.yticks(fontsize=fontsize_ticks)
        plt.grid(True, linestyle="--", alpha=0.7)

        output_file = os.path.join(diurnal_output_dir, f"diurnal_distribution_label_{label}.png")
        plt.savefig(output_file, bbox_inches="tight", dpi=300, transparent=True)
        plt.close()

def plot_seasonal_class_distribution(df):
    monthly_counts = df.groupby(['month', 'label']).size().unstack(fill_value=0)
    monthly_percentage = monthly_counts.div(monthly_counts.sum(axis=1), axis=0) * 100

    for label in monthly_percentage.columns:
        plt.figure(figsize=(6, 4))
        label_str = str(label)
        plt.plot(
            monthly_percentage.index,
            monthly_percentage[label],
            label=f'Class {label}',
            linewidth=line_thickness,
            color=colors_per_class1_names.get(label_str, 'black')
        )
        plt.title(f"Seasonal Distribution - Class {label}", fontsize=fontsize_title)
        plt.xlabel("Month", fontsize=fontsize_labels)
        plt.ylabel("Percentage", fontsize=fontsize_labels)
        plt.ylim(0, 40)
        plt.xticks(fontsize=fontsize_ticks)
        plt.yticks(fontsize=fontsize_ticks)
        plt.grid(True, linestyle="--", alpha=0.7)

        output_file = os.path.join(seasonal_output_dir, f"seasonal_distribution_label_{label}.png")
        plt.savefig(output_file, bbox_inches="tight", dpi=300, transparent=True)
        plt.close()


def plot_diurnal_variable_cycle_by_class(df, variable, output_dir, colors_dict, fontsize_title=20, fontsize_labels=16, fontsize_ticks=16):
    """
    Plot the diurnal cycle (mean ± std) of a variable, split by class (label).
    """
    os.makedirs(output_dir, exist_ok=True)
    grouped = df.groupby(['hour', 'label'])[variable]

    stats = grouped.agg(['mean', 'std']).reset_index()

    for label in sorted(df['label'].unique()):
        label_stats = stats[stats['label'] == label]
        color = colors_dict.get(str(label), 'black')

        plt.figure(figsize=(6, 4))
        plt.plot(label_stats['hour'], label_stats['mean'], color=color, linewidth=line_thickness, label=f"Class {label}")
        plt.fill_between(label_stats['hour'],
                         label_stats['mean'] - label_stats['std'],
                         label_stats['mean'] + label_stats['std'],
                         color=color, alpha=0.3)

        var_name = variable.split('-')[0]
        unit = VARIABLE_INFO[var_name].get('units')
        y_label = VARIABLE_INFO[var_name]['long_name'] + (f" [{unit}]" if unit else "")
       
        plt.title(f"Diurnal Cycle - Class {label}", fontsize=fontsize_title)
        plt.xlabel("Hour of Day (UTC)", fontsize=fontsize_labels)
        plt.ylabel(y_label, fontsize=fontsize_labels)
        plt.xticks(fontsize=fontsize_ticks)
        plt.yticks(fontsize=fontsize_ticks)
        plt.grid(True, linestyle="--", alpha=0.7)

        plt.ylim(VARIABLE_INFO[var_name]['limit'])
        if VARIABLE_INFO[var_name]['log']: plt.yscale('log')
        if VARIABLE_INFO[var_name]['dir'] == 'decr': plt.yaxis()

        if var_name == 'cot':
            plt.xlim(5,16)

        output_file = os.path.join(output_dir, f"diurnal_{variable}_class_{label}.png")
        plt.savefig(output_file, bbox_inches="tight", dpi=300, transparent=True)
        plt.close()
        print(f"Saved: {output_file}")


def plot_seasonal_variable_cycle_by_class(df, variable, output_dir, month_order, colors_dict, fontsize_title=20, fontsize_labels=16, fontsize_ticks=16):
    """
    Plot the seasonal cycle (mean ± std) of a variable, split by class (label).
    """
    os.makedirs(output_dir, exist_ok=True)
    grouped = df.groupby(['month', 'label'])[variable]

    stats = grouped.agg(['mean', 'std']).reset_index()

    # Ensure correct month order
    stats['month'] = pd.Categorical(stats['month'], categories=month_order, ordered=True)
    stats = stats.sort_values('month')

    for label in sorted(df['label'].unique()):
        label_stats = stats[stats['label'] == label]
        color = colors_dict.get(str(label), 'black')

        plt.figure(figsize=(6, 4))
        plt.plot(label_stats['month'], label_stats['mean'], color=color, linewidth=line_thickness, label=f"Class {label}")
        plt.fill_between(label_stats['month'],
                         label_stats['mean'] - label_stats['std'],
                         label_stats['mean'] + label_stats['std'],
                         color=color, alpha=0.3)

        var_name = variable.split('-')[0]
        unit = VARIABLE_INFO[var_name].get('units')
        y_label = VARIABLE_INFO[var_name]['long_name'] + (f" [{unit}]" if unit else "")
        
        plt.title(f"Seasonal Cycle - Class {label}", fontsize=fontsize_title)
        plt.xlabel("Month", fontsize=fontsize_labels)
        plt.ylabel(y_label, fontsize=fontsize_labels)
        plt.xticks(ticks=np.arange(len(month_order)), labels=month_order, fontsize=fontsize_ticks)
        plt.yticks(fontsize=fontsize_ticks)
        plt.grid(True, linestyle="--", alpha=0.7)

        plt.ylim(VARIABLE_INFO[var_name]['limit'])
        if VARIABLE_INFO[var_name]['log']: plt.yscale('log')
        if VARIABLE_INFO[var_name]['dir'] == 'decr': plt.yaxis()

        output_file = os.path.join(output_dir, f"seasonal_{variable}_class_{label}.png")
        plt.savefig(output_file, bbox_inches="tight", dpi=300, transparent=True)
        plt.close()
        print(f"Saved: {output_file}")


# --- MAIN EXECUTION ---
csv_path = f'{output_path}crops_stats_{run_name}_{sampling_type}_33729_imergmin.csv'
df_filtered = prepare_dataframe(csv_path, False)

# if column cth-50 and cth-99 exixst, divide them by 1000
if 'cth-50' in df_filtered.columns and 'cth-99' in df_filtered.columns:
    df_filtered['cth-50'] /= 1000
    df_filtered['cth-99'] /= 1000

#plot_diurnal_class_distribution(df_filtered)
#plot_seasonal_class_distribution(df_filtered)

#variables_to_plot = ['cot-50', 'cot-99', 'cth-50', 'cth-99', 'precipitation-50', 'precipitation-99', 'cma-None', 'cph-None']
variables_to_plot = ['cot-50']#, 'cth-50']
for var in variables_to_plot:
    plot_diurnal_variable_cycle_by_class(df_filtered, 
                                         variable=var, 
                                         output_dir=os.path.join(output_path, "diurnal_variables"), 
                                         colors_dict=colors_per_class1_names)
    #plot_seasonal_variable_cycle_by_class(df_filtered, variable=var, output_dir=os.path.join(output_path, "seasonal_variables"), month_order=list(month_names.values()), colors_dict=colors_per_class1_names)





# # 1. Diurnal Distribution - Heatmap
# plt.figure(figsize=(12, 6))
# ax = sns.heatmap(hourly_percentage.T.iloc[::-1], cmap=cmc.lapaz, annot=False, linewidths=0.5, 
#                  cbar_kws={'format': '%.0f%%'}, xticklabels=2)

# plt.title("Diurnal Distribution of Classes (%)", fontsize=fontsize_title)
# plt.xlabel("Hour of Day", fontsize=fontsize_labels)
# plt.ylabel("Class Label", fontsize=fontsize_labels)
# plt.xticks(fontsize=fontsize_ticks)
# plt.yticks(fontsize=fontsize_ticks)

# plt.savefig(f"{diurnal_output_dir}/diurnal_distribution_heatmap.png", bbox_inches="tight")
# print("Diurnal distribution heatmap saved.")

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

# plt.savefig(f"{seasonal_output_dir}/seasonal_distribution_heatmap.png", bbox_inches="tight")
# print("Seasonal distribution heatmap saved.")

# # 3. Diurnal Distribution - Line Plot
# plt.figure(figsize=(10, 6))
# for label in hourly_percentage.columns:
#     plt.plot(
#         hourly_percentage.index,
#         hourly_percentage[label],
#         label=f'Class {label}',
#         linewidth=2,
#         color=colors_per_class1_names.get(str(label), 'black')  # fallback to black if key not found
#     )

# plt.title("Diurnal Distribution of Classes (%)", fontsize=fontsize_title)
# plt.xlabel("Hour of Day", fontsize=fontsize_labels)
# plt.ylabel("Percentage", fontsize=fontsize_labels)
# plt.xticks(fontsize=fontsize_ticks)
# plt.yticks(fontsize=fontsize_ticks)

# plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=fontsize_ticks)
# plt.grid(True, linestyle="--", alpha=0.7)

# plt.savefig(f"{diurnal_output_dir}/diurnal_distribution_lineplot.png", bbox_inches="tight")
# print("Diurnal distribution line plot saved.")


# # 4. Seasonal Distribution - Line Plot
# plt.figure(figsize=(10, 6))
# for label in monthly_percentage.columns:
#     plt.plot(
#         monthly_percentage.index,
#         monthly_percentage[label],
#         label=f'Class {label}',
#         linewidth=2,
#         color=colors_per_class1_names.get(str(label), 'black')
#     )

# plt.title("Seasonal Distribution of Classes (%)", fontsize=fontsize_title)
# plt.xlabel("Month", fontsize=fontsize_labels)
# plt.ylabel("Percentage", fontsize=fontsize_labels)

# plt.xticks(
#     ticks=np.arange(len(month_names)),
#     labels=list(month_names.values()),
#     fontsize=fontsize_ticks,
#     rotation=0
# )
# plt.yticks(fontsize=fontsize_ticks)

# plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=fontsize_ticks)
# plt.grid(True, linestyle="--", alpha=0.7)

# plt.savefig(f"{seasonal_output_dir}/seasonal_distribution_lineplot.png", bbox_inches="tight")
# print("Seasonal distribution line plot saved.")
