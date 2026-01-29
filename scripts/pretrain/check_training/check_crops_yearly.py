import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# ======================================================
# === CONFIGURATION ===
# ======================================================
img_dir = (
    "/data1/crops/ir108_100x100_2012-2016-2021-2025_2xrandomcrops_1xtimestamp_cma_nc/img/IR_108"
)

output_dir = (
    "/data1/fig/dcv2_resnet_k7_ir108_100x100_2012-2016-2021-2025_1xrandomcrops_1xtimestamp_cma_nc"
)

os.makedirs(output_dir, exist_ok=True)

# grid size
N_ROWS = 10
N_COLS = 15
N_SAMPLES = N_ROWS * N_COLS

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# ======================================================
# === COLLECT IMAGES BY YEAR ===
# ======================================================
images_per_year = defaultdict(list)

for fname in os.listdir(img_dir):
    if not fname.lower().endswith(".png"):
        continue

    # Example:
    # 2025-09-30T23:45:00_1_t0_2025-09-30T23-45.png
    try:
        year = int(fname.split("-")[0])
    except ValueError:
        print(f"Skipping file with unexpected name: {fname}")
        continue

    images_per_year[year].append(os.path.join(img_dir, fname))

years = sorted(images_per_year.keys())
print(f"Found years: {years}")


# ======================================================
# === PLOT TABLE PER YEAR ===
# ======================================================
for year in years:
    img_list = images_per_year[year]

    if len(img_list) == 0:
        continue

    # sample images (with replacement if not enough)
    if len(img_list) >= N_SAMPLES:
        sampled_imgs = random.sample(img_list, N_SAMPLES)
    else:
        sampled_imgs = random.choices(img_list, k=N_SAMPLES)

    fig, axes = plt.subplots(
        N_ROWS, N_COLS,
        figsize=(N_COLS * 1.2, N_ROWS * 1.2)
    )

    fig.suptitle(
        f"Random IR 10.8 μm Crops – Year {year}",
        fontsize=16,
        fontweight="bold"
    )

    for ax, img_path in zip(axes.flat, sampled_imgs):
        try:
            img = mpimg.imread(img_path)
            ax.imshow(img, cmap="gray")
        except Exception as e:
            print(f"Failed to read {img_path}: {e}")
            ax.axis("off")
            continue

        ax.axis("off")

    # hide any remaining axes (should not happen, but safe)
    for ax in axes.flat[len(sampled_imgs):]:
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    outname = f"random_examples_{year}.png"
    outfile = os.path.join(output_dir, outname)

    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {outfile}")


print("All yearly tables created.")
