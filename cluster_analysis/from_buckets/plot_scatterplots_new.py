import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cmcrameri.cm as cmc
from datetime import datetime
from matplotlib.lines import Line2D

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
N_SAMPLES = None #Apply further random subsampling during testig the script
PLOT_CLASSES_TOGETHER = False  # If True, plot all classes together; if False, plot each class separately

filter_daytime = False       # Enable daytime filter (06–16 UTC)
filter_imerg_minutes = False  # Only keep timestamps with minutes 00 or 30

filter_tags = []
if filter_daytime:
    filter_tags.append("daytime")
if filter_imerg_minutes:
    filter_tags.append("imergmin")

FILTER_SUFFIX = "_" + "_".join(filter_tags) if filter_tags else ""

VAR_X = 'cot'
VAR_Y = 'cth'
VAR_COLOR = 'precipitation'
PERCENTILES = [50]

USE_CACHED = True  # Set to False to force recompute and overwrite the cache

# === VARIABLE METADATA ===
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

# === FUNCTIONS ===
def load_dataset(n_samples=None):
    #if n_samples is None, load all samples
    
    path = f'/data1/fig/{RUN_NAME}/{SAMPLE_TYPE}/crop_list_{RUN_NAME}_{N_SUBSAMPLES}_{SAMPLE_TYPE}{FILTER_SUFFIX}.csv'
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
        scatter = ax.scatter(x, y, c=color, cmap=cmc.hawaii_r, norm=norm, s=20, alpha=0.5)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(f"{ADDITIONAL_VARS[VAR_COLOR]['long_name']} [{ADDITIONAL_VARS[VAR_COLOR]['units']}]", fontsize=14)
        #increase ticks of colorbar
        cbar.ax.tick_params(labelsize=12)

    unit_x = VARIABLE_INFO[VAR_X].get('units')
    unit_y = VARIABLE_INFO[VAR_Y].get('units')
    x_label = VARIABLE_INFO[VAR_X]['long_name'] + (f" [{unit_x}]" if unit_x else "")
    y_label = VARIABLE_INFO[VAR_Y]['long_name'] + (f" [{unit_y}]" if unit_y else "")
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.set_title(f'Label {label}', fontsize=16, fontweight='bold')

    #increase font size of ticks
    ax.tick_params(axis='both', which='major', labelsize=14)
    #ax.tick_params(axis='both', which='minor', labelsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    ax.set_xlim(VARIABLE_INFO[VAR_X]['limit'])
    ax.set_ylim(VARIABLE_INFO[VAR_Y]['limit'])
    if VARIABLE_INFO[VAR_X]['log']: ax.set_xscale('log')
    if VARIABLE_INFO[VAR_Y]['log']: ax.set_yscale('log')
    if VARIABLE_INFO[VAR_X]['dir'] == 'decr': ax.invert_xaxis()
    if VARIABLE_INFO[VAR_Y]['dir'] == 'decr': ax.invert_yaxis()

    out_dir = f'/data1/fig/{RUN_NAME}/{SAMPLE_TYPE}/scatterplots/'
    os.makedirs(out_dir, exist_ok=True)
    suffix = 'heatmap' if USE_HEATMAP else 'scatter'
    fname = f"{VAR_X}_{VAR_Y}_perc_{percentile}_3rd-var_{VAR_COLOR}_label{label}_{RUN_NAME}_{SAMPLE_TYPE}_{suffix}.png"
    #save the plots transparent
    fig.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches='tight', transparent=True)
    plt.close()

def plot_single_variable_distribution_per_class(
    df_subset, label, variable, run_name, sample_type,
    bins=50, use_kde=False, log_scale=False
):
    """
    For each class (label), plots the distribution (histogram or KDE) of a single variable.

    Parameters:
    - df: pandas.DataFrame — must include 'label' and the variable
    - variable: str — the variable to plot
    - run_name: str — identifier for the output directory
    - sample_type: str — type of sample (used in folder name)
    - bins: int — number of histogram bins (ignored if use_kde=True)
    - use_kde: bool — use KDE instead of histogram
    - log_scale: bool — apply log scale on the x-axis
    """

    out_dir = f'/data1/fig/{run_name}/{sample_type}/distributions_per_class/'
    os.makedirs(out_dir, exist_ok=True)        

    plt.figure(figsize=(6, 4))

    #prepare dataframe to have the variable to plot
    if variable not in df_subset.columns:
        raise ValueError(f"Variable '{variable}' not found in DataFrame columns.")
    subset = df_subset[df_subset['label'] == label][variable].dropna()

    if use_kde:
        sns.kdeplot(subset, fill=True, linewidth=2, alpha=0.7)
    else:
        plt.hist(subset, bins=bins, density=True, alpha=0.7, edgecolor='black')

    # Set labels and title
    var_info = VARIABLE_INFO.get(variable, {})
    x_label = var_info.get('long_name', variable)
    units = var_info.get('units', '')
    x_label += f" [{units}]" if units else ""

    unit_x = VARIABLE_INFO[variable.split('-')[0]].get('units')
    x_label = VARIABLE_INFO[variable.split('-')[0]]['long_name'] + (f" [{unit_x}]" if unit_x else "")
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel("Density", fontsize=16)
    plt.title(f"Class {label}", fontsize=18, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim(0,0.5)
    plt.xlim(VARIABLE_INFO[variable.split('-')[0]]['limit'])

    

    if log_scale:
        plt.xscale('log')

    fname = f"{variable}_distribution_class_{label}_{run_name}_{sample_type}.png"
    plt.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"Saved: {fname}")


def plot_data_two_vars(df_label, label, var_x, var_y, run_name, sample_type, use_heatmap=False):
    
    #get x and y from df_label
    x = df_label[var_x].values
    y = df_label[var_y].values
    if var_x not in df_label.columns or var_y not in df_label.columns:
        raise ValueError(f"Variables '{var_x}' or '{var_y}' not found in DataFrame columns.")
    
    fig, ax = plt.subplots(figsize=(6, 5))

    if use_heatmap:
        hb = ax.hexbin(x, y, gridsize=40, cmap=cmc.buda, mincnt=1)
        plt.colorbar(hb, ax=ax, label='Counts')
    else:
        ax.scatter(x, y, s=20, alpha=0.5, color='steelblue')

    # Labels from VARIABLE_INFO
    unit_x = VARIABLE_INFO[var_x.split('-')[0]].get('units')
    unit_y = VARIABLE_INFO[var_y.split('-')[0]].get('units')
    x_label = VARIABLE_INFO[var_x.split('-')[0]].get('long_name', var_x) + (f" [{unit_x}]" if unit_x else "")
    y_label = VARIABLE_INFO[var_y.split('-')[0]].get('long_name', var_y) + (f" [{unit_y}]" if unit_y else "")

    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    ax.set_title(f'Label {label}', fontsize=18, fontweight='bold')

    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(True, linestyle='--', alpha=0.7)

    # Axis limits, log scale, direction
    if 'limit' in VARIABLE_INFO[var_x.split('-')[0]]: ax.set_xlim(VARIABLE_INFO[var_x.split('-')[0]]['limit'])
    if 'limit' in VARIABLE_INFO[var_y.split('-')[0]]: ax.set_ylim(VARIABLE_INFO[var_y.split('-')[0]]['limit'])
    if VARIABLE_INFO[var_x.split('-')[0]].get('log', False): ax.set_xscale('log')
    if VARIABLE_INFO[var_y.split('-')[0]].get('log', False): ax.set_yscale('log')
    if VARIABLE_INFO[var_x.split('-')[0]].get('dir') == 'decr': ax.invert_xaxis()
    if VARIABLE_INFO[var_y.split('-')[0]].get('dir') == 'decr': ax.invert_yaxis()

    # Save figure
    out_dir = f'/data1/fig/{run_name}/{sample_type}/scatterplots_two_vars/'
    os.makedirs(out_dir, exist_ok=True)

    suffix = 'hexbin' if use_heatmap else 'scatter'
    fname = f"{var_x}_{var_y}_label{label}_{run_name}_{sample_type}_{suffix}.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"Saved: {fname}")


def plot_all_labels_scatter(x, y, labels, third_var, percentile, label_colors, var_size_name):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Normalize dot size
    size_min, size_max = third_var.min(), third_var.max()
    size_norm = np.clip((third_var - size_min) / (size_max - size_min), 0, 1)
    sizes = 10 + 90 * size_norm

    # Plot each label cluster
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        ax.scatter(
            x[mask], y[mask],
            s=sizes[mask],
            color=label_colors.get(str(label), 'black'),
            alpha=0.6,
            label=f'Label {label}'
        )

    # Axis labeling
    unit_x = VARIABLE_INFO[VAR_X].get('units')
    unit_y = VARIABLE_INFO[VAR_Y].get('units')
    x_label = VARIABLE_INFO[VAR_X]['long_name'] + (f" [{unit_x}]" if unit_x else "")
    y_label = VARIABLE_INFO[VAR_Y]['long_name'] + (f" [{unit_y}]" if unit_y else "")
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.set_title(f'All Labels Scatterplot', fontsize=16, fontweight='bold')

    # Axis scaling and direction
    if VARIABLE_INFO[VAR_X].get('log'): ax.set_xscale('log')
    if VARIABLE_INFO[VAR_Y].get('log'): ax.set_yscale('log')
    if VARIABLE_INFO[VAR_X].get('dir') == 'decr': ax.invert_xaxis()
    if VARIABLE_INFO[VAR_Y].get('dir') == 'decr': ax.invert_yaxis()
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Adjust plot to leave space on the right
    fig.subplots_adjust(right=0.75)

    # Legend for labels (top right, outside plot)
    label_legend = ax.legend(
        loc='upper left', bbox_to_anchor=(1.02, 1.0),
        fontsize=11, title='Labels', title_fontsize=12
    )
    ax.add_artist(label_legend)

    # Size legend (bottom right, outside plot)
    example_values = [0.1, 0.5, 1, 2, 5, 10]
    size_legend_sizes = [10 + 90 * ((val - size_min) / (size_max - size_min)) for val in example_values]
    size_handles = [
        Line2D([], [], marker='o', color='gray', linestyle='None',
               markersize=np.sqrt(s), label=f'{val}')
        for val, s in zip(example_values, size_legend_sizes)
    ]
    size_legend_title = f'{VARIABLE_INFO[var_size_name].get("long_name", "")} [{VARIABLE_INFO[var_size_name].get("units", "")}]'

    size_legend = ax.legend(
        handles=size_handles,
        title=size_legend_title,
        loc='lower left', bbox_to_anchor=(1.02, 0.0),
        fontsize=10, title_fontsize=10
    )
    ax.add_artist(size_legend)

    # Save plot
    out_dir = f'/data1/fig/{RUN_NAME}/{SAMPLE_TYPE}/scatterplots/all_labels/'
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{VAR_X}_{VAR_Y}_perc_{percentile}_size_{var_size_name}_all_labels_{RUN_NAME}_{SAMPLE_TYPE}.png"
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

def check_min_max(cached_df, percentile):
    #get min and max values of the three variables
    #cached_df = cached_df[cached_df['percentile'] == percentile]
    min_x = cached_df[f'{VAR_X}-{percentile}'].min()
    max_x = cached_df[f'{VAR_X}-{percentile}'].max()
    min_y = cached_df[f'{VAR_Y}-{percentile}'].min()
    max_y = cached_df[f'{VAR_Y}-{percentile}'].max()
    min_c = cached_df[f'{VAR_COLOR}-{percentile}'].min()
    max_c = cached_df[f'{VAR_COLOR}-{percentile}'].max()
    print(f"Min {VAR_X}: {min_x}, Max {VAR_X}: {max_x}")
    print(f"Min {VAR_Y}: {min_y}, Max {VAR_Y}: {max_y}")
    print(f"Min {VAR_COLOR}: {min_c}, Max {VAR_COLOR}: {max_c}")

    return {
        'min_x': min_x,
        'max_x': max_x,
        'min_y': min_y,
        'max_y': max_y,
        'min_c': min_c,
        'max_c': max_c
    }
        

# === MAIN EXECUTION ===
def main():
    s3 = Initialize_s3_client(S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY)
    df = load_dataset(N_SAMPLES)

    for percentile in PERCENTILES:
        CACHE_FILE = f'/data1/fig/{RUN_NAME}/{SAMPLE_TYPE}/crops_stats_{RUN_NAME}_{SAMPLE_TYPE}_{N_SUBSAMPLES}{FILTER_SUFFIX}.csv'
        print(CACHE_FILE)
        if USE_CACHED and os.path.exists(CACHE_FILE):
            print(f"Loading cached data from {CACHE_FILE}...")
            cached_df = pd.read_csv(CACHE_FILE)
            print(cached_df)
            #check_min_max(cached_df, percentile)
        else:
            print("Computing percentile data...")
            enriched_rows = []
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
        # Convert CTH (VAR_Y) to km for better readability
        cached_df[f'{VAR_Y}-{percentile}'] = cached_df[f'{VAR_Y}-{percentile}'] / 1000  # Convert to km
        #remore invalid labels
        cached_df = cached_df[cached_df['label'] != -100]
        print(cached_df.columns)
        
        if PLOT_CLASSES_TOGETHER:
            # Plot all labels together
            plot_all_labels_scatter(
                cached_df[f'{VAR_X}-{percentile}'].values, 
                cached_df[f'{VAR_Y}-{percentile}'].values, 
                cached_df['label'].values,
                cached_df[f'{VAR_COLOR}-99'].values, 
                percentile,
                label_colors=colors_per_class1_names,
                var_size_name=VAR_COLOR
            )
        else:
            for label in sorted(cached_df['label'].unique()):
                df_subset = cached_df[(cached_df['label'] == label)]
                # #plot_data(df_subset[f'{VAR_X}-{percentile}'].values, 
                #           df_subset[f'{VAR_Y}-{percentile}'].values, 
                #           df_subset[f'{VAR_COLOR}-99'].values, 
                #           label, 
                #           percentile)
                # plot_single_variable_distribution_per_class(
                #     df_subset, label,
                #     f'{VAR_Y}-{percentile}', 
                #     RUN_NAME, 
                #     SAMPLE_TYPE, 
                #     bins=50, 
                #     use_kde=False, 
                #     log_scale=VARIABLE_INFO[VAR_Y].get('log', False)
                # )

                plot_data_two_vars(df_subset, 
                                   label, 
                                   'precipitation-99', 
                                   'cth-50', 
                                   RUN_NAME, 
                                   SAMPLE_TYPE, 
                                   use_heatmap=False)

    print("All plots generated.")

if __name__ == '__main__':
    main()


#nohup 