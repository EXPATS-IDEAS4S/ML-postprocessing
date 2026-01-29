import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from matplotlib.image import imread

# === CONFIG ===
RUN_NAME = 'dcv2_resnet_k8_ir108_100x100_2013-2020_1xrandomcrops_1xtimestamp_cma_nc_convective'
EVENT_TYPES = ["PRECIP", "HAIL"]
BASE_DIR = f"/data1/fig/{RUN_NAME}/test"
CROP_BASE_DIR = "/data1/crops/test_case_essl_2021-2025_100x100_ir108_cma"
SUMMARY_DIR = "/data1/crops/test_case_essl_2021-2025_100x100_ir108_cma"

# === SELECT CASE ===
SELECT_EVENT = "PRECIP"     # "HAIL" or "PRECIP"
SELECT_DATE = "2022-09-15"  # YYYY-MM-DD
N_CLASSES = 8              # Number of classes in the model
TARGET_LAT  = 43.51306666666667
TARGET_LON  = 12.9618

# === COLOR MAP ===
COLORS_PER_CLASS = {
    0: 'darkgray', 1: 'darkslategrey', 2: 'peru', 3: 'orangered',
    4: 'lightcoral', 5: 'deepskyblue', 6: 'purple', 7: 'lightblue',
    8: 'green', 9: 'goldenrod', 10: 'magenta', 11: 'dodgerblue',
    12: 'darkorange', 13: 'olive', 14: 'crimson'
}

def load_case(event):
    """Load and preprocess CSV for a given event."""
    path = f"{BASE_DIR}/features_train_test_{RUN_NAME}.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ File not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    #df = df[df['case_study'] == True].copy()
    df = df[df['vector_type'] == event]
    #print(df['datetime'])
    #print(df['datetime'])
    # df['datetime'] = (
    #     df['datetime'].astype(str)
    #     .str.extract(r"\['(\d{8}_\d{4})'\]")[0]
    # )
    #create date column
    df['datetime'] = pd.to_datetime(df['datetime'], format="%Y-%m-%d %H:%M:%S", errors='coerce')
    df['date'] = df['datetime'].dt.date
    #df = df.dropna(subset=['datetime'])
    #df = df.sort_values('datetime')
    return df

def plot_diurnal_labels(df_day, event, date, n_classes=15):
    """Plot the diurnal evolution of labels for a given day."""
    df_day['hour'] = df_day['datetime'].dt.hour + df_day['datetime'].dt.minute / 60.0

    plt.figure(figsize=(8, 4))
    #use custom colors per class
    sns.scatterplot(x='hour', y='label', data=df_day, hue='label', palette=COLORS_PER_CLASS, s=100, legend=False)
    plt.title(f"Class Evolution on {date} ({event})", fontsize=16, fontweight='bold')
    plt.xlabel("Hour (UTC)", fontsize=16)
    plt.ylabel("Label", fontsize=16)
    plt.xticks(range(0, 25, 1), fontsize=16, rotation=45, ha='right')
    plt.yticks(range(0, n_classes, 1), fontsize=16)
    #set ylim 
    plt.ylim(-0.5, n_classes - 0.5)
    plt.xlim(-0.5, 24)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    out_path = f"{BASE_DIR}/class_evolution_{event}_{date}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close()
    print(f"✅ Saved diurnal plot: {out_path}")

def show_image_table(event, date):
    """Display 1 image per hour for the given date (max 3x8 grid)."""
    img_dir = f"{CROP_BASE_DIR}/{event}/images/IR_108/png_vmin-vmax_greyscale_CMA"
    imgs = sorted([f for f in os.listdir(img_dir) if date.replace('-', '') in f and f.endswith('.png')])
    if not imgs:
        print(f"⚠️ No images found for {event} on {date}")
        return

    # --- Extract hour info and pick first per hour ---
    img_by_hour = {}
    for fname in imgs:
        # Example: 2025-09-27T23:45_45.8_9.19_20250927T2345.png
        try:
            ts_str = fname.split('_')[0]  # e.g., "2025-09-27T23:45"
            ts = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M")
            hour = ts.hour
            if hour not in img_by_hour:
                img_by_hour[hour] = fname
        except Exception:
            continue

    selected_imgs = [img_by_hour[h] for h in sorted(img_by_hour.keys())]
    n = len(selected_imgs)
    if n == 0:
        print(f"⚠️ No valid hourly images for {event} on {date}")
        return

    # --- Grid: 3 rows × 8 columns (fits up to 24 hours) ---
    cols = 8
    rows = (n + cols - 1) // cols
    rows = min(rows, 3)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < n:
            img_path = os.path.join(img_dir, selected_imgs[i])
            img = imread(img_path)
            ts_label = selected_imgs[i].split('_')[0].split('T')[1]  # show "HH:MM"
            ax.imshow(img, cmap='gray')
            ax.set_title(ts_label, fontsize=14)
            ax.axis('off')
        else:
            ax.axis('off')

    plt.suptitle(f"Satellite Crop Evolution (1 per hour) – {event}, {date}",
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    out_img_grid = f"{BASE_DIR}/crops_evolution_{event}_{date}_hourly.png"
    plt.savefig(out_img_grid, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"✅ Saved hourly image grid: {out_img_grid}")


def check_summary(event, date):
    """Find and print summary row for given event and date."""
    summary_path = f"{SUMMARY_DIR}/{event}_summary.csv"
    if not os.path.exists(summary_path):
        print(f"⚠️ No summary file found: {summary_path}")
        return
    df_sum = pd.read_csv(summary_path)
    print(df_sum.columns)
    print(df_sum.head())
    print(date)
   
    match = df_sum[df_sum['day_id'].astype(str).str.contains(date)]
    if match.empty:
        print(f"⚠️ No matching event found in summary for {date}")
    else:
        print(f"\n📄 Summary info for {event} {date}:")
        print(match)
        out_txt = f"{BASE_DIR}/summary_event_{event}_{date}.txt"
        match.to_csv(out_txt, index=False)
        print(f"✅ Saved summary to {out_txt}")

# === MAIN ===
if SELECT_EVENT not in EVENT_TYPES:
    raise ValueError(f"Invalid event '{SELECT_EVENT}', must be one of {EVENT_TYPES}")

print(f"\n🔍 Looking for {SELECT_EVENT} event on {SELECT_DATE} ...")

df = load_case(SELECT_EVENT)
#print(df['date'])
df_day = df[df['date'] == datetime.strptime(SELECT_DATE, "%Y-%m-%d").date()]
#check how many groups with unique date and lat and lon

#check if more thank 1 entries found
if len(df_day) > 96:
    df_day = df_day.sort_values('datetime')
    #select the one with the desired lon lat
    df_day = df_day.iloc[((df_day['lat_centre'] - TARGET_LAT)**2 + (df_day['lon_centre'] - TARGET_LON)**2).argsort()[:96]]

if df_day.empty:
    print(f"⚠️ No data found for {SELECT_EVENT} on {SELECT_DATE}")
else:
    print(f"✅ Found {len(df_day)} entries for {SELECT_EVENT} on {SELECT_DATE}")
    plot_diurnal_labels(df_day, SELECT_EVENT, SELECT_DATE, N_CLASSES)
    show_image_table(SELECT_EVENT, SELECT_DATE)
    check_summary(SELECT_EVENT, SELECT_DATE) #TODO fix this function

print("\n🎯 Done.")
