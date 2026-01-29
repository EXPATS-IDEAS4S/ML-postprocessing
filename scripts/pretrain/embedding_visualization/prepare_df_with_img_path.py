"""
Prepare dataframe with image paths for embedding visualization.

"""

import os
import pandas as pd
import sys

sys.path.append('/home/Daniele/codes/VISSL_postprocessing/')
from utils.plotting.class_colors import CLOUD_CLASS_INFO

from scripts.pretrain.embedding_visualization.config_visualization import (
    OUTPUT_PATH,
    LIST_IMAGE_CROPS,
    N_FRAMES,
    FILENAME_TSNE,
    FILENAME_LABELS,
)

expanded_csv = os.path.join(
        os.path.dirname(OUTPUT_PATH),
        f"merged_tsne_crop_list_with_img_path.csv"
    )


# =============================================================================
# FUNCTIONS
# =============================================================================
def load_labels(output_path, tsne_filename, labels_filename, labels, tsne_cols=["tsne_dim_1", "tsne_dim_2"]) -> pd.DataFrame:
    """Load precomputed labels and dimensionality-reduced features."""
    
    #Open labels csv
    df = pd.read_csv(os.path.join(output_path, labels_filename), low_memory=False)
   
    #open df 2 dim feature space csv
    df_features = pd.read_csv(f"{output_path}{tsne_filename}", low_memory=False)

    print(df_features)

    df_centroids = df_features[
        (df_features["vector_type"] == "CENTROID")
    ].dropna(subset=tsne_cols)

    #add label column to centroids df
    df_centroids["label"] = labels[:len(df_centroids)]

    df_features = df_features[
        (df_features["vector_type"] != "CENTROID")
    ].dropna(subset=tsne_cols)

    #check if len df_fearure is equal to len df
    if len(df_features) != len(df):
        print(f"Warning: len df_features ({len(df_features)}) != len df ({len(df)})")
    print(f"len df_features: {len(df_features)}")
    print(f"len df: {len(df)}")

    #replace feature columns in df with those from df_features based on the vector_type and index
    for vtype in df["vector_type"].unique():
        mask = df["vector_type"] == vtype
        df_v = df[mask]
        df_features.loc[mask, 'label'] = df_v.loc[df.index[mask], 'label'].values.astype(int)
        df_features.loc[mask, 'path'] = df_v.loc[df.index[mask], 'path'].values
     

    #add label column to features df
    print(df_features)
    print(df_centroids)

    return df_features, df_centroids



def substitute_paths(df_labels: pd.DataFrame, output_path=expanded_csv):
    """Substitute image paths per frame and plot grids if expanded dataset is missing."""
   
    for frame_idx in range(N_FRAMES):
        frame_rows = []
        for _, row in df_labels.iterrows():
            video_stem = os.path.splitext(os.path.basename(row["path"]))[0]
            frame_str = f"t{frame_idx}_"
            matches = [p for p in LIST_IMAGE_CROPS if video_stem in p and frame_str in p]
            if not matches:
                continue
            new_row = row.copy()
            new_row["path"] = matches[0]
            new_row["frame_idx"] = frame_idx
            frame_rows.append(new_row)

        df_frame = pd.DataFrame(frame_rows)
        df_frame = df_frame[df_frame["label"] != -100]

        #save expanded dataframe
        expanded_csv = os.path.join(
            os.path.dirname(OUTPUT_PATH),
            f"merged_tsne_crop_list_with_img_path.csv"
        )
        print(f"Saving expanded dataframe to {expanded_csv}...")
        df_frame.to_csv(expanded_csv, index=False)

           
# =============================================================================
# MAIN
# =============================================================================
def main():
    items = sorted(CLOUD_CLASS_INFO.items(), key=lambda x: x[1]["order"])
    print(items)
    labels = sorted([label for label, _ in items])
    print(labels)
    print(f"n samples: {len(LIST_IMAGE_CROPS)}")

    df_feat, df_centroids = load_labels(OUTPUT_PATH, FILENAME_TSNE, FILENAME_LABELS, labels)
    
    df_prepared = df_feat[df_feat["label"] != -100].copy()
    #df_prepared['year'] = df_prepared['path'].apply(lambda x: int(os.path.basename(x).split('-')[0]))
    print(f"n samples after filtering: {len(df_prepared)}")

    substitute_paths(df_prepared, expanded_csv)
    print("Paths substituted and expanded dataframe saved.")


if __name__ == "__main__":
    main()

#47648