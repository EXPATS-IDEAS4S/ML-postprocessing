# =============================================================================
# CONFIGURATION
# =============================================================================


from glob import glob


RUN_NAME = "dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc"
CROPS_NAME = "ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc"
RANDOM_STATE = 3 #[0, 3, 16, 23, 57],
SAMPLING_TYPE = "test_traj"            # Options: "all", "subsample"
REDUCTION_METHOD = "tsne"        # Options: "tsne", "isomap"
PERPLEXITY = 50
EPOCH = 800
FILE_EXTENSION = "png"
VARIABLE_TYPE = "IR_108"      # e.g. "WV_062-IR_108"
VIDEO = True
N_FRAMES = 1
RANDOM_SEEDS = None #[0]# [0, 8, 16, 23, 42]  # For random selection in tables
YEAR = None

# Visualization settings
VMIN, CENTER, VMAX = -60, 0, 5
CMAP = "gray_r"  # or create_WV_IR_diff_colormap(VMIN, CENTER, VMAX)
OUTPUT_PATH = f"/data1/fig/{RUN_NAME}/epoch_{EPOCH}/{SAMPLING_TYPE}/"
#FILENAME = f"{REDUCTION_METHOD}_opentsne_{RUN_NAME}_{RANDOM_STATE}_epoch_{EPOCH}.npy"
FILENAME_TSNE = "tsne_all_vectors_with_centroids.csv"
FILENAME_LABELS = f"features_train_test_{RUN_NAME}_2nd_labels.csv"
FILENAME = f"{REDUCTION_METHOD}_embedding_{RUN_NAME}_epoch_{EPOCH}.png"

# Input data
IMAGE_CROPS_PATH = f"/data1/crops/{CROPS_NAME}/img/{VARIABLE_TYPE}/"
LIST_IMAGE_CROPS = sorted(glob(IMAGE_CROPS_PATH + "*." + FILE_EXTENSION))

SUBSTITUTE_PATH = True



