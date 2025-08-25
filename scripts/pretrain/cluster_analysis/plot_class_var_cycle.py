"""
Description:
------------
Plots diurnal and seasonal cycles of variables for each class, based on CSV statistics.

"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.buckets.aux_functions_from_buckets import get_variable_info, extract_datetime

# -----------------------------
# Configuration
# -----------------------------
RUN_NAME = 'dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA'
RANDOM_STATE = 3
SAMPLING_TYPE = 'all'
REDUCTION_METHOD = 'tsne'
OUTPUT_PATH = f'/data1/fig/{RUN_NAME}/{SAMPLING_TYPE}/'

# Plot directories
DIURNAL_DIR = os.path.join(OUTPUT_PATH, "diurnal_lineplots")
SEASONAL_DIR = os.path.join(OUTPUT_PATH, "seasonal_lineplots")
VARIABLE_DIR = os.path.join(OUTPUT_PATH, "variable_lineplots")
os.makedirs(DIURNAL_DIR, exist_ok=True)
os.makedirs(SEASONAL_DIR, exist_ok=True)
os.makedirs(VARIABLE_DIR, exist_ok=True)

# Plot styling
sns.set_context("talk")
LINEWIDTH = 3
FONTSIZE_TITLE = 18
FONTSIZE_LABELS = 16
FONTSIZE_TICKS = 16

# Class colors
CLASS_COLORS = {
    0: 'darkgray', 1: 'darkslategrey', 2: 'peru', 3: 'orangered',
    4: 'lightcoral', 5: 'deepskyblue', 6: 'purple', 7: 'lightblue', 8: 'green'
}

# Variable info
VARIABLE_INFO = {
    'cot': {'long_name': 'Cloud Optical Thickness', 'units': None, 'dir': 'incr', 'log': True, 'limit': (0.1, 150)},
    'cth': {'long_name': 'Cloud Top Height', 'units': 'Km', 'dir': 'incr', 'log': False, 'limit': (0, 14)},
    'cma': {'long_name': 'Cloud Cover', 'units': None, 'dir': 'incr', 'log': False, 'limit': (0, 1)},
    'cph': {'long_name': 'Ice ratio', 'units': None, 'dir': 'incr', 'log': False, 'limit': (0, 1)},
    'precipitation': {'long_name': 'Rain Rate', 'units': 'mm/h', 'dir': 'incr', 'log': False, 'limit': (0, 50)}
}

MONTH_NAMES = {4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep"}

# -----------------------------
# Helper functions
# -----------------------------
def safe_parse_datetime(path):
    """Parse datetime from filename safely."""
    try:
        dt = extract_datetime(os.path.basename(path))
        return pd.to_datetime(f"{dt['year']:04d}-{dt['month']:02d}-{dt['day']:02d}T{dt['hour']:02d}:{dt['minute']:02d}:00")
    except Exception as e:
        print(f"Failed to parse datetime for {path}: {e}")
        return pd.NaT

def prepare_dataframe(csv_path, filter_blocks=False):
    """Load CSV, parse datetime, filter months, optionally filter blocks."""
    df = pd.read_csv(csv_path)
    df['datetime'] = df['path'].apply(safe_parse_datetime)
    df.dropna(subset=['datetime'], inplace=True)
    df['hour'] = df['hour'].astype(int)
    df['month'] = df['month'].astype(int)
    df = df[df['month'].between(4, 9)]
    df['month'] = df['month'].map(MONTH_NAMES)
    df = df.sort_values('datetime')

    if filter_blocks:
        df['label_shifted'] = df['label'].shift()
        df['time_shifted'] = df['datetime'].shift()
        df['gap'] = (df['datetime'] - df['time_shifted']).dt.total_seconds() / 60
        df['is_new_block'] = (df['label'] != df['label_shifted']) | (df['gap'] > 30)
        df = df[df['is_new_block']].copy()
    return df

# -----------------------------
# Plotting functions
# -----------------------------
def plot_diurnal_variable(df, variable, output_dir, colors_dict=CLASS_COLORS):
    """Plot diurnal cycle (mean ± std) per class."""
    os.makedirs(output_dir, exist_ok=True)
    grouped = df.groupby(['hour', 'label'])[variable].agg(['mean', 'std']).reset_index()

    for label in sorted(df['label'].unique()):
        label_stats = grouped[grouped['label'] == label]
        color = colors_dict.get(label, 'black')

        plt.figure(figsize=(6,4))
        plt.plot(label_stats['hour'], label_stats['mean'], color=color, linewidth=LINEWIDTH)
        plt.fill_between(label_stats['hour'],
                         label_stats['mean'] - label_stats['std'],
                         label_stats['mean'] + label_stats['std'],
                         color=color, alpha=0.3)

        var_name = variable.split('-')[0]
        var_info = VARIABLE_INFO.get(var_name, {})
        y_label = var_info.get('long_name', var_name)
        if var_info.get('units'): y_label += f" [{var_info['units']}]"

        plt.title(f"Diurnal Cycle - Class {label}", fontsize=FONTSIZE_TITLE)
        plt.xlabel("Hour of Day (UTC)", fontsize=FONTSIZE_LABELS)
        plt.ylabel(y_label, fontsize=FONTSIZE_LABELS)
        plt.xticks(fontsize=FONTSIZE_TICKS)
        plt.yticks(fontsize=FONTSIZE_TICKS)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.ylim(var_info.get('limit', (0,1)))
        if var_info.get('log'): plt.yscale('log')

        # Optional x-limits for specific variables
        if var_name == 'cot': plt.xlim(5,16)

        output_file = os.path.join(output_dir, f"diurnal_{variable}_class_{label}.png")
        plt.savefig(output_file, bbox_inches="tight", dpi=300, transparent=True)
        plt.close()
        print(f"Saved: {output_file}")

def plot_seasonal_variable(df, variable, output_dir, month_order=list(MONTH_NAMES.values()), colors_dict=CLASS_COLORS):
    """Plot seasonal cycle (mean ± std) per class."""
    os.makedirs(output_dir, exist_ok=True)
    grouped = df.groupby(['month', 'label'])[variable].agg(['mean', 'std']).reset_index()
    grouped['month'] = pd.Categorical(grouped['month'], categories=month_order, ordered=True)
    grouped = grouped.sort_values('month')

    for label in sorted(df['label'].unique()):
        label_stats = grouped[grouped['label'] == label]
        color = colors_dict.get(label, 'black')

        plt.figure(figsize=(6,4))
        plt.plot(label_stats['month'], label_stats['mean'], color=color, linewidth=LINEWIDTH)
        plt.fill_between(label_stats['month'],
                         label_stats['mean'] - label_stats['std'],
                         label_stats['mean'] + label_stats['std'],
                         color=color, alpha=0.3)

        var_name = variable.split('-')[0]
        var_info = VARIABLE_INFO.get(var_name, {})
        y_label = var_info.get('long_name', var_name)
        if var_info.get('units'): y_label += f" [{var_info['units']}]"

        plt.title(f"Seasonal Cycle - Class {label}", fontsize=FONTSIZE_TITLE)
        plt.xlabel("Month", fontsize=FONTSIZE_LABELS)
        plt.ylabel(y_label, fontsize=FONTSIZE_LABELS)
        plt.xticks(ticks=np.arange(len(month_order)), labels=month_order, fontsize=FONTSIZE_TICKS)
        plt.yticks(fontsize=FONTSIZE_TICKS)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.ylim(var_info.get('limit', (0,1)))
        if var_info.get('log'): plt.yscale('log')

        output_file = os.path.join(output_dir, f"seasonal_{variable}_class_{label}.png")
        plt.savefig(output_file, bbox_inches="tight", dpi=300, transparent=True)
        plt.close()
        print(f"Saved: {output_file}")

# -----------------------------
# Main execution
# -----------------------------
csv_path = f'{OUTPUT_PATH}crops_stats_{RUN_NAME}_{SAMPLING_TYPE}_33729_imergmin.csv'
df = prepare_dataframe(csv_path, filter_blocks=False)

# Handle scaling for specific columns
for col in ['cth-50', 'cth-99']:
    if col in df.columns:
        df[col] /= 1000

# Example usage: diurnal plotting
variables_to_plot = ['cot-50']  # add more variables as needed
for var in variables_to_plot:
    plot_diurnal_variable(df, var, output_dir=os.path.join(OUTPUT_PATH, "diurnal_variables"))
    # plot_seasonal_variable(df, var, output_dir=os.path.join(OUTPUT_PATH, "seasonal_variables"))
