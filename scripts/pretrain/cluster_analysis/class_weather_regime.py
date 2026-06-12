"""
with this code, we assign to each video classified with VISSL the weather regime
label associated to the date of the video based on the method from ask Ilaria
The labels are contained in a txt file with 4 columns 
YYYY MM DD wrnum
where YYYY MM DD are the year, month and day of the video and wrnum is the weather regime label
We want to read all videos classified with VISSL contained in a csv file and listed per frame. 
First, we need to group the frames by crop name, then we need to read the date from the crop name 
(es: 20060401_0000_0345_IR_108_cma_0.nc -> date: 2006-04-01) and then we can match the date 
with the corresponding weather regime label in the txt file.,

author: Claudia Acquistapace
date: 2026-06-10
how to run:
conda activate venv_vissl
python class_weather_regime.py

"""
import os
import pandas as pd


CLOUD_CLASS_NAMES = {
    0: "broken low clouds",
    1: "Deep convection",
    2: "broken higher clouds",
    3: "decaying convection no prec",
    4: "overcast decaying",
    5: "early convection prec", 
    6: "convection growing no prec",
    7: "overcast growing",
    8: "decaying convection prec",
}


def read_weather_regime_labels(file_path):
    """ Reads the weather regime labels from a text file and returns a DataFrame.
    :param file_path: The path to the text file containing the weather regime labels.
    :return: A DataFrame with columns ['YYYY', 'MM', 'DD', 'wrnum'] if the file exists, otherwise None.
    """
    try:
        df = pd.read_csv(file_path, sep=r'\s+')
        df = df.rename(columns={'M': 'MM', 'D': 'DD', 'Wrnum': 'wrnum'})
        df[['YYYY', 'MM', 'DD', 'wrnum']] = df[['YYYY', 'MM', 'DD', 'wrnum']].apply(
            pd.to_numeric, errors='coerce'
        )
        df = df.dropna(subset=['YYYY', 'MM', 'DD', 'wrnum'])
        df[['YYYY', 'MM', 'DD', 'wrnum']] = df[['YYYY', 'MM', 'DD', 'wrnum']].astype(int)
        return df
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

def assign_weather_regime_label(video_date, weather_regime_df):
    """ Assigns a weather regime label to a video based on its date.
    :param video_date: A datetime object representing the date of the video.
    :param weather_regime_df: A DataFrame containing the weather regime labels with columns ['YYYY', 'MM', 'DD', 'wrnum'].
    :return: The weather regime label corresponding to the video date, or None if no match is found.
    """
    # Extract year, month, and day from the video date
    year = video_date.year
    month = video_date.month
    day = video_date.day

    # Find the matching weather regime label in the DataFrame
    match = weather_regime_df[
        (weather_regime_df['YYYY'] == year) &
        (weather_regime_df['MM'] == month) &
        (weather_regime_df['DD'] == day)
    ]

    if not match.empty:
        return match.iloc[0]['wrnum']
    else:
        print(f"No weather regime label found for date: {video_date}")
        return None


def extract_date_from_crop_name(crop_name):
    """Extract the date from crop filenames starting with YYYYMMDD."""
    date_str = os.path.basename(crop_name).split('_')[0]
    return pd.to_datetime(date_str, format='%Y%m%d')


def add_weather_regime_labels(videos_df, weather_regime_df):
    """
    Create one row per crop and add the matching weather-regime label by date.
    """
    output_df = videos_df[['crop', 'label']].drop_duplicates(subset='crop').copy()
    output_df = output_df.rename(columns={'crop': 'crop_name'})

    output_df['date_str'] = output_df['crop_name'].str.split('_').str[0]
    output_df['video_date'] = pd.to_datetime(output_df['date_str'], format='%Y%m%d', errors='coerce')

    invalid_dates = output_df['video_date'].isna().sum()
    if invalid_dates:
        print(f"Warning: {invalid_dates} crop names have dates that could not be parsed.")

    weather_regime_df = weather_regime_df.copy()
    weather_regime_df['video_date'] = pd.to_datetime(
        weather_regime_df[['YYYY', 'MM', 'DD']].rename(
            columns={'YYYY': 'year', 'MM': 'month', 'DD': 'day'}
        )
    )

    output_df = output_df.merge(
        weather_regime_df[['video_date', 'wrnum']],
        on='video_date',
        how='left'
    )
    output_df = output_df.rename(columns={'wrnum': 'wr_label'})

    missing_wr = output_df['wr_label'].isna().sum()
    if missing_wr:
        print(f"Warning: {missing_wr} crops have no matching weather-regime label.")

    output_df['cloud_class_name'] = output_df['label'].map(CLOUD_CLASS_NAMES)

    return output_df[['crop_name', 'label', 'cloud_class_name', 'wr_label']]



def main():

    # Path to the text file containing weather regime labels
    weather_regime_file = "/sat_data/output/grl_2026/csv/ERA5_pseudoPCs_labels_noreg_xr_4025.txt"

    # Read the weather regime labels into a DataFrame
    weather_regime_df = read_weather_regime_labels(weather_regime_file)
    if weather_regime_df is None:
        return
    
    # read the csv file containing the list of videos classified with VISSL
    videos_csv = "/sat_data/output/grl_2026/csv/crops_stats_var-cth_stats-50-95-25-75_frames-8_timedim_grl_2026_all_240216_imergmin.csv"
    print(f"Reading crop labels from {videos_csv}")
    videos_df = pd.read_csv(videos_csv, usecols=['crop', 'label'])
    print(f"Loaded {len(videos_df)} frame rows.")

    output_df = add_weather_regime_labels(videos_df, weather_regime_df)
    output_csv = "/sat_data/output/grl_2026/csv/crops_stats_var-cth_stats-50-95-25-75_frames-8_timedim_grl_2026_all_240216_imergmin_with_wr_labels.csv"
    output_df.to_csv(output_csv, index=False)
    print(f"Saved {len(output_df)} crop labels to {output_csv}")

    # plot now distribution of weather regimes for each VISSL class label using a bar plot
    # avoid plotting class -100 
    import matplotlib.pyplot as plt
    plot_df = output_df[output_df['label'] != -100]
    plot_df.groupby(['cloud_class_name', 'wr_label']).size().unstack(fill_value=0).plot(kind='bar', stacked=True)
    plt.xlabel('VISSL Cloud Class')
    plt.ylabel('Count')
    plt.title('Distribution of Weather Regimes for Each VISSL Cloud Class Label')
    plt.xticks(rotation=45, ha='right')
    plt.legend(
        title='Weather Regime Label',
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        borderaxespad=0,
    )
    plt.tight_layout(rect=[0, 0, 0.82, 1])

    # set axis invisbile and thicker
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(1.5)
    plt.gca().spines['bottom'].set_linewidth(1.5)
    plt.gca().tick_params(width=1.5)

    plot_file = "/sat_data/output/grl_2026/figs/weather_regime_distribution.png"
    os.makedirs(os.path.dirname(plot_file), exist_ok=True)
    plt.savefig(plot_file, bbox_inches='tight')
    print(f"Saved weather regime distribution plot to {plot_file}")
    
if __name__ == "__main__":
    main()
