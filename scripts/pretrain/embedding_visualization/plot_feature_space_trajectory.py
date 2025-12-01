import os
import pandas as pd
from plot_embedding_utils import plot_embedding_dots_iterative_test_msg_icon, plot_embedding_dots  # adjust import path if needed

# ---------- CONFIG ----------
CSV_PATH = "/data1/fig/dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA/test/teamx/features_dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA_with_tsne.csv"
OUTPUT_DIR = "/data1/fig/dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA/test/teamx/tsne_iterative/"
FILENAME_BASE = "tsne_feature_space"
csv_train = '/data1/fig/dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA/test/teamx/features_train_dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA_with_tsne.csv'  # Training features CSV for background

# Colors per class for the background
COLORS_PER_CLASS = {
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

# ---------- MAIN ----------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load dataframe
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df_train = pd.read_csv(csv_train, low_memory=False)
    print(df.columns)

    # Make sure color column exists for the background
    if 'color' not in df.columns:
        # If label column exists, map it to colors
        if 'label' in df.columns:
            df['color'] = df['label'].map(lambda x: COLORS_PER_CLASS.get(str(int(x)), 'gray'))
            df_train['color'] = df_train['label'].map(lambda x: COLORS_PER_CLASS.get(str(int(x)), 'gray'))
        else:
            df['color'] = 'gray'
            df_train['color'] = 'gray'

    # Separate full background embedding (all points) and the test case points
    df_background = df.copy()
    df_case_study = df[df['case_study'] == True].copy()

    if df_case_study.empty:
        print("No test case points found in the DataFrame.")
        return

    plot_embedding_dots(df_train, COLORS_PER_CLASS, 
                        "/data1/fig/dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA/test/teamx/", 
                        "features_train_dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA_with_tsne",
                        "comp_1", "comp_2",
                        df_subset2=None)
    print("Plotted full training set for background.")
    exit()
    # Call your iterative plotting function
    plot_embedding_dots_iterative_test_msg_icon(
        df_subset1=df_background,
        colors_per_class1_norm=COLORS_PER_CLASS,
        output_path=OUTPUT_DIR,
        filename=FILENAME_BASE + ".png",
        df_subset2=df_case_study,
        legend=False
    )

if __name__ == "__main__":
    main()
