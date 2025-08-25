"""
Description:
------------
This script loads satellite/cloud dataset samples, processes physical variables 
(e.g., cloud optical thickness, cloud top height, precipitation), and produces 
visualizations including scatter plots, heatmaps, and per-class distributions.

Features:
- Optional percentile computation for each variable.
- Optional CMA filtering for cloud cover variables.
- Supports filtering by daytime or IMERG minute.
- Plots per-class scatterplots or all classes together.
- Caches computed percentiles for faster repeated runs.

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cmcrameri.cm as cmc
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from utils.buckets.aux_functions_from_buckets import (
    extract_variable_values, compute_categorical_values,
    filter_cma_values, extract_datetime
)
from utils.buckets.get_data_from_buckets import Initialize_s3_client
from utils.buckets.credentials_buckets import S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY, S3_ENDPOINT_URL

# -----------------------------
# Configuration
# -----------------------------
CONFIG = {
    "buckets": {
        'cmsaf': 'expats-cmsaf-cloud',
        'imerg': 'expats-imerg-prec',
        'crop': 'expats-msg-training'
    },
    "run_name": "dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA",
    "sample_type": "all",
    "n_subsamples": 33729,
    "use_heatmap": False,
    "apply_cma_filter": True,
    "n_samples": None,
    "plot_classes_together": False,
    "filter_daytime": False,
    "filter_imerg_minutes": True,
    "var_x": "cot",
    "var_y": "cth",
    "var_color": "precipitation",
    "percentiles": [50],
    "colormap": cmc.bamako,
    "use_cached": True
}

VARIABLE_INFO = {
    'cot': {'long_name': 'Cloud Optical Thickness', 'units': None, 'dir': 'incr', 'log': True, 'limit': (0.1, 150)},
    'cth': {'long_name': 'Cloud Top Height', 'units': 'Km', 'dir': 'incr', 'log': False, 'limit': (0, 14)},
    'cma': {'long_name': 'Cloud Cover', 'units': None, 'dir': 'incr', 'log': False, 'limit': (0, 1)},
    'cph': {'long_name': 'Ice ratio', 'units': None, 'dir': 'incr', 'log': False, 'limit': (0, 1)},
    'precipitation': {'long_name': 'Rain Rate', 'units': 'mm/h', 'dir': 'incr', 'log': False, 'limit': (0, 50)}
}

ADDITIONAL_VARS = {
    '6.2-10.8': {'long_name': 'WV 6.2 - IR 10.8 µm', 'units': 'K', 'dir': 'decr'},
    'precipitation': {'long_name': 'Rain Rate', 'units': 'mm/h', 'dir': 'incr'}
}

CATEGORICAL_VARS = ['cma', 'cph']

colors_per_class1_names = {
    '0': 'darkgray', '1': 'darkslategrey', '2': 'peru', '3': 'orangered',
    '4': 'lightcoral', '5': 'deepskyblue', '6': 'purple', '7': 'lightblue', '8': 'green'
}

# -----------------------------
# Helper Functions
# -----------------------------
def load_dataset(n_samples=None):
    """Load CSV dataset and optionally subsample."""
    filter_tags = []
    if CONFIG["filter_daytime"]:
        filter_tags.append("daytime")
    if CONFIG["filter_imerg_minutes"]:
        filter_tags.append("imergmin")
    filter_suffix = "_" + "_".join(filter_tags) if filter_tags else ""

    path = f'/data1/fig/{CONFIG["run_name"]}/{CONFIG["sample_type"]}/crop_list_{CONFIG["run_name"]}_{CONFIG["n_subsamples"]}_{CONFIG["sample_type"]}{filter_suffix}.csv'
    df = pd.read_csv(path)
    df = df[df['label'] != -100]

    if n_samples is not None:
        df = df.sample(n=n_samples, random_state=42)
    return df

def process_row(row, var_x, var_y, var_color, s3_client):
    """Extract variables from S3 and apply CMA filtering."""
    values_cma = extract_variable_values(row, 'cma', s3_client, **CONFIG["buckets"])
    vx = extract_variable_values(row, var_x, s3_client, **CONFIG["buckets"])
    vy = extract_variable_values(row, var_y, s3_client, **CONFIG["buckets"])
    vx = filter_cma_values(vx, values_cma, var_x, CONFIG["apply_cma_filter"])
    vy = filter_cma_values(vy, values_cma, var_y, CONFIG["apply_cma_filter"])

    if var_color == '6.2-10.8':
        v108 = extract_variable_values(row, 'IR_108', s3_client, **CONFIG["buckets"])
        v62 = extract_variable_values(row, 'WV_062', s3_client, **CONFIG["buckets"])
        vc = filter_cma_values(v62 - v108, values_cma, var_color, CONFIG["apply_cma_filter"])
    else:
        vc = extract_variable_values(row, var_color, s3_client, **CONFIG["buckets"])
        vc = filter_cma_values(vc, values_cma, var_color, CONFIG["apply_cma_filter"])
    return vx, vy, vc

def aggregate(values, var, percentile):
    """Aggregate variable using percentile or categorical counts."""
    if var in CATEGORICAL_VARS:
        return compute_categorical_values(values, var)
    return np.nanpercentile(values, percentile) if len(values) > 0 else np.nan

def is_night(path: str) -> bool:
    """Check if timestamp is during night (18:00–04:00)."""
    ts = extract_datetime(path)
    return ts['hour'] >= 18 or ts['hour'] < 4

def should_skip_precip_minute(path: str) -> bool:
    """Skip IMERG timestamps at :15 or :45."""
    ts = extract_datetime(path)
    return ts['minute'] in [15, 45]

def get_color_norm(var_color):
    """Return normalization object for color mapping."""
    if var_color == '6.2-10.8':
        return plt.Normalize(vmin=-40, vmax=10)
    elif var_color == 'precipitation':
        return plt.Normalize(vmin=0, vmax=20)
    else:
        return plt.Normalize(vmin=0, vmax=1)

# -----------------------------
# Plotting Functions
# -----------------------------
def plot_data(x, y, color, label, percentile):
    """Scatter or hexbin plot for one class."""
    fig, ax = plt.subplots(figsize=(6,5))
    if CONFIG["use_heatmap"]:
        hb = ax.hexbin(x, y, gridsize=40, cmap=cmc.buda, mincnt=1)
        plt.colorbar(hb, ax=ax, label='Counts')
    else:
        x = np.array(x)
        y = np.array(y)
        color = np.array(color)
        mask_zero = color == 0
        mask_pos = color > 0
        ax.scatter(x[mask_zero], y[mask_zero], c='red', s=30, alpha=1, label='Zero values')
        scatter = ax.scatter(x[mask_pos], y[mask_pos], c=color[mask_pos], cmap=CONFIG["colormap"], norm=get_color_norm(CONFIG["var_color"]), s=15, alpha=0.5)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(f"{ADDITIONAL_VARS[CONFIG['var_color']]['long_name']} [{ADDITIONAL_VARS[CONFIG['var_color']]['units']}]", fontsize=12)

    # Axis labels
    x_label = VARIABLE_INFO[CONFIG["var_x"]]['long_name'] + (f" [{VARIABLE_INFO[CONFIG['var_x']]['units']}]" if VARIABLE_INFO[CONFIG['var_x']]['units'] else "")
    y_label = VARIABLE_INFO[CONFIG["var_y"]]['long_name'] + (f" [{VARIABLE_INFO[CONFIG['var_y']]['units']}]" if VARIABLE_INFO[CONFIG['var_y']]['units'] else "")
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(f'Label {label}', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', labelsize=10)
    ax.set_xlim(VARIABLE_INFO[CONFIG['var_x']]['limit'])
    ax.set_ylim(VARIABLE_INFO[CONFIG['var_y']]['limit'])
    if VARIABLE_INFO[CONFIG['var_x']]['log']: ax.set_xscale('log')
    if VARIABLE_INFO[CONFIG['var_y']]['log']: ax.set_yscale('log')
    if VARIABLE_INFO[CONFIG['var_x']]['dir']=='decr': ax.invert_xaxis()
    if VARIABLE_INFO[CONFIG['var_y']]['dir']=='decr': ax.invert_yaxis()

    # Save figure
    out_dir = f'/data1/fig/{CONFIG["run_name"]}/{CONFIG["sample_type"]}/scatterplots/'
    os.makedirs(out_dir, exist_ok=True)
    suffix = 'heatmap' if CONFIG["use_heatmap"] else 'scatter'
    fname = f"{CONFIG['var_x']}_{CONFIG['var_y']}_perc_{percentile}_3rd-var_{CONFIG['var_color']}_label{label}_{CONFIG['run_name']}_{CONFIG['sample_type']}_{suffix}.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig)

# -----------------------------
# Main Execution
# -----------------------------
def main():
    s3_client = Initialize_s3_client(S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY)
    df = load_dataset(CONFIG["n_samples"])

    for percentile in CONFIG["percentiles"]:
        cache_file = f'/data1/fig/{CONFIG["run_name"]}/{CONFIG["sample_type"]}/crops_stats_{CONFIG["run_name"]}_{CONFIG["sample_type"]}_{CONFIG["n_subsamples"]}.csv'
        if CONFIG["use_cached"] and os.path.exists(cache_file):
            cached_df = pd.read_csv(cache_file)
        else:
            enriched_rows = []
            for label in sorted(df['label'].unique()):
                df_label = df[df['label']==label]
                for _, row in df_label.iterrows():
                    path = row['path'].split('/')[-1]
                    if 'cot' in [CONFIG['var_x'], CONFIG['var_y']] and is_night(path):
                        continue
                    if CONFIG['var_color']=='precipitation' and should_skip_precip_minute(path):
                        continue
                    vx, vy, vc = process_row(row, CONFIG['var_x'], CONFIG['var_y'], CONFIG['var_color'], s3_client)
                    if len(vc)==0 or np.all(np.isnan(vc)):
                        continue
                    row_data = row.copy()
                    row_data['percentile'] = percentile
                    row_data[f'{CONFIG["var_x"]}_val'] = aggregate(vx, CONFIG['var_x'], percentile)
                    row_data[f'{CONFIG["var_y"]}_val'] = aggregate(vy, CONFIG['var_y'], percentile)
                    row_data[f'{CONFIG["var_color"]}_val'] = np.nanpercentile(vc, percentile)
                    enriched_rows.append(row_data)
            cached_df = pd.DataFrame(enriched_rows).dropna(subset=[f'{CONFIG["var_x"]}_val', f'{CONFIG["var_y"]}_val', f'{CONFIG["var_color"]}_val'])
            cached_df.to_csv(cache_file, index=False)

        # Convert VAR_Y to km
        cached_df[f'{CONFIG["var_y"]}-{percentile}'] = cached_df[f'{CONFIG["var_y"]}_val'] / 1000

        if CONFIG["plot_classes_together"]:
            # TODO: Implement combined scatter plot
            pass
        else:
            for label in sorted(cached_df['label'].unique()):
                df_subset = cached_df[cached_df['label']==label]
                plot_data(
                    df_subset[f'{CONFIG["var_x"]}_val'].values,
                    df_subset[f'{CONFIG["var_y"]}_val'].values,
                    df_subset[f'{CONFIG["var_color"]}_val'].values,
                    label,
                    percentile
                )

if __name__=='__main__':
    main()
