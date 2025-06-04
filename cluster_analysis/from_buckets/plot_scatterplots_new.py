import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import cmcrameri.cm as cmc
from datetime import datetime

from aux_functions_from_buckets import extract_variable_values, compute_categorical_values, filter_cma_values, extract_datetime
from get_data_from_buckets import Initialize_s3_client
from credentials_buckets import S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY, S3_ENDPOINT_URL

# === CONFIGURATION ===
BUCKETS = {
    'cmsaf': 'expats-cmsaf-cloud',
    'imerg': 'expats-imerg-prec',
    'crop': 'expats-msg-training'
}

RUN_NAME = 'dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA'
SAMPLE_TYPE = 'all'
N_SUBSAMPLES = 67425
USE_HEATMAP = False
APPLY_CMA_FILTER = True
N_SAMPLES = None


VAR_X = 'cot'
VAR_Y = 'cth'
VAR_COLOR = 'precipitation'
PERCENTILES = [50, 95]

USE_CACHED = False  # Set to False to force recompute and overwrite the cache

# === VARIABLE METADATA ===
VARIABLE_INFO = {
    'cot': {'long_name': 'Cloud Optical Thickness', 'units': None, 'dir': 'incr', 'log': True, 'limit': (0.1, 150)},
    'cth': {'long_name': 'Cloud Top Height', 'units': 'm', 'dir': 'incr', 'log': False, 'limit': (0, 17000)},
    'cma': {'long_name': 'Cloud Cover', 'units': None, 'dir': 'incr', 'log': False, 'limit': (0, 1)},
    'cph': {'long_name': 'Ice ratio', 'units': None, 'dir': 'incr', 'log': False, 'limit': (0, 1)}
}

ADDITIONAL_VARS = {
    '6.2-10.8': {'long_name': 'WV 6.2 - IR 10.8 µm', 'units': 'K', 'dir': 'decr'},
    'precipitation': {'long_name': 'Rain Rate', 'units': 'mm/h', 'dir': 'incr'}
}

CATEGORICAL_VARS = ['cma', 'cph']

# === FUNCTIONS ===
def load_dataset(n_samples=None):
    #if n_samples is None, load all samples
    
    path = f'/data1/fig/{RUN_NAME}/{SAMPLE_TYPE}/crop_list_{RUN_NAME}_{N_SUBSAMPLES}_{SAMPLE_TYPE}.csv'
    df = pd.read_csv(path)
    df = df[df['label'] != -100]
    if n_samples is not None:
        df = df.sample(n=n_samples, random_state=42)

    return df

def process_row(row, var_x, var_y, var_color, s3):
    values_cma = extract_variable_values(row, 'cma', s3, **BUCKETS)
    vx = extract_variable_values(row, var_x, s3, **BUCKETS)
    vy = extract_variable_values(row, var_y, s3, **BUCKETS)
    vx = filter_cma_values(vx, values_cma, var_x, APPLY_CMA_FILTER)
    vy = filter_cma_values(vy, values_cma, var_y, APPLY_CMA_FILTER)

    if var_color == '6.2-10.8':
        v108 = extract_variable_values(row, 'IR_108', s3, **BUCKETS)
        v62 = extract_variable_values(row, 'WV_062', s3, **BUCKETS)
        vc = filter_cma_values(v62 - v108, values_cma, var_color, APPLY_CMA_FILTER)
    else:
        vc = extract_variable_values(row, var_color, s3, **BUCKETS)
        vc = filter_cma_values(vc, values_cma, var_color, APPLY_CMA_FILTER)

    return vx, vy, vc

def aggregate(values, var, percentile):
    if var in CATEGORICAL_VARS:
        return compute_categorical_values(values, var)
    return np.nanpercentile(values, percentile) if len(values) > 0 else np.nan

def get_color_norm(var_color, percentile):
    if var_color == '6.2-10.8':
        return plt.Normalize(vmin=-40, vmax=10)
    elif var_color == 'precipitation':
        if percentile == 50:
            return plt.Normalize(vmin=0, vmax=10)
        elif percentile == 95:
            return plt.Normalize(vmin=0, vmax=50)
        else:
            # Fallback for other percentiles
            return plt.Normalize(vmin=0, vmax=20)
    else:
        # Default normalization
        return plt.Normalize(vmin=0, vmax=1)

def plot_data(x, y, color, label, percentile):
    fig, ax = plt.subplots(figsize=(6, 5))

    if USE_HEATMAP:
        hb = ax.hexbin(x, y, gridsize=40, cmap=cmc.buda, mincnt=1)
        plt.colorbar(hb, ax=ax, label='Counts')
    else:
        norm = get_color_norm(VAR_COLOR, percentile)
        scatter = ax.scatter(x, y, c=color, cmap=cmc.batlowK, norm=norm, s=20, alpha=0.5)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(f"{ADDITIONAL_VARS[VAR_COLOR]['long_name']} [{ADDITIONAL_VARS[VAR_COLOR]['units']}]", fontsize=14)

    ax.set_xlabel(f"{VARIABLE_INFO[VAR_X]['long_name']} [{VARIABLE_INFO[VAR_X].get('units', '')}]", fontsize=14)
    ax.set_ylabel(f"{VARIABLE_INFO[VAR_Y]['long_name']} [{VARIABLE_INFO[VAR_Y].get('units', '')}]", fontsize=14)
    ax.set_title(f'Label {label}', fontsize=16, fontweight='bold')

    #increase font size of ticks
    ax.tick_params(axis='both', which='major', labelsize=14)
    #ax.tick_params(axis='both', which='minor', labelsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_aspect('equal', adjustable='box')
    
    ax.set_xlim(VARIABLE_INFO[VAR_X]['limit'])
    ax.set_ylim(VARIABLE_INFO[VAR_Y]['limit'])
    if VARIABLE_INFO[VAR_X]['log']: ax.set_xscale('log')
    if VARIABLE_INFO[VAR_Y]['log']: ax.set_yscale('log')
    if VARIABLE_INFO[VAR_X]['dir'] == 'decr': ax.invert_xaxis()
    if VARIABLE_INFO[VAR_Y]['dir'] == 'decr': ax.invert_yaxis()

    out_dir = f'/data1/fig/{RUN_NAME}/{SAMPLE_TYPE}/scatterplots/label_{label}/'
    os.makedirs(out_dir, exist_ok=True)
    suffix = 'heatmap' if USE_HEATMAP else 'scatter'
    fname = f"{VAR_X}_{VAR_Y}_perc_{percentile}_3rd-var_{VAR_COLOR}_label{label}_{RUN_NAME}_{SAMPLE_TYPE}_{suffix}.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches='tight')
    plt.close()


def is_night(path: str) -> bool:
    """Check if timestamp is during night (18:00 to 04:00)."""
    ts = extract_datetime(path)  # Assume filename contains timestamp
    hour = ts['hour']
    return hour >= 18 or hour < 4

def should_skip_precip_minute(path: str) -> bool:
    """Skip timestamps at :15 or :45 if var_color is precipitation."""
    ts = extract_datetime(path)
    return ts['minute'] in [15, 45]

# === MAIN EXECUTION ===
def main():
    s3 = Initialize_s3_client(S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY)
    df = load_dataset(N_SAMPLES)

    for percentile in PERCENTILES:
        CACHE_FILE = f'/data1/fig/{RUN_NAME}/{SAMPLE_TYPE}/crop_percentile_{percentile}_{RUN_NAME}_{N_SUBSAMPLES}_{SAMPLE_TYPE}.csv'
        if USE_CACHED and os.path.exists(CACHE_FILE):
            print(f"Loading cached data from {CACHE_FILE}...")
            cached_df = pd.read_csv(CACHE_FILE)
        else:
            print("Computing percentile data...")
            enriched_rows = []

            for percentile in PERCENTILES:
                for label in sorted(df['label'].unique()):
                    df_label = df[df['label'] == label]

                    for _, row in df_label.iterrows():
                        path = row['path'].split('/')[-1]  # Extract filename from path
                        print(path)
                        # === FILTER 1: Skip nighttime if COT is one of the variables ===
                        if 'cot' in [VAR_X, VAR_Y] and is_night(path):
                            print(f"Skipping nighttime data for {path}.")
                            continue
                        
                        # === FILTER 2: Skip certain minutes for precipitation ===
                        if VAR_COLOR == 'precipitation' and should_skip_precip_minute(path):
                            print(f"Skipping {VAR_COLOR} data for {path} at :15 or :45 minute.")
                            continue

                        vx, vy, vc = process_row(row, VAR_X, VAR_Y, VAR_COLOR, s3)
                        if len(vc) == 0 or np.all(np.isnan(vc)):
                            continue

                        # Copy the full original row
                        row_data = row.copy()
                        # Add percentile info and aggregated values
                        row_data['percentile'] = percentile
                        row_data[f'{VAR_X}_val'] = aggregate(vx, VAR_X, percentile)
                        row_data[f'{VAR_Y}_val'] = aggregate(vy, VAR_Y, percentile)
                        row_data[f'{VAR_COLOR}_val'] = np.nanpercentile(vc, percentile)

                        enriched_rows.append(row_data)

            cached_df = pd.DataFrame(enriched_rows)
            #delete rows with NaN values in any of the three variables
            cached_df = cached_df.dropna(subset=[f'{VAR_X}_val', f'{VAR_Y}_val', f'{VAR_COLOR}_val'])
            # Save to cache
            cached_df.to_csv(CACHE_FILE)
            print(f"Saved computed percentiles to {CACHE_FILE}.")

    # === PLOTTING ===
    for percentile in PERCENTILES:
        for label in sorted(cached_df['label'].unique()):
            df_subset = cached_df[(cached_df['label'] == label) & (cached_df['percentile'] == percentile)]
            x_vals = df_subset[f'{VAR_X}_val'].values
            y_vals = df_subset[f'{VAR_Y}_val'].values
            color_vals = df_subset[f'{VAR_COLOR}_val'].values

            plot_data(x_vals, y_vals, color_vals, label, percentile)

    print("All plots generated.")

if __name__ == '__main__':
    main()


#nohup 2895122