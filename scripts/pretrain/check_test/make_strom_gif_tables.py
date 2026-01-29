import os
import re
import random
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt


# ==============================
# CONFIG
# ==============================
IMG_DIR = (
    "/data1/crops/test_case_essl_14-15-16-18-19-20-22-23-24_100x100_ir108_cma_traj/"
    "images/IR_108/png_vmin-vmax_greyscale_CMA"
)

OUT_DIR = "/data1/crops/test_case_essl_14-15-16-18-19-20-22-23-24_100x100_ir108_cma_traj/storm_gifs"
os.makedirs(OUT_DIR, exist_ok=True)

GRID_ROWS = 5
GRID_COLS = 8
N_GIFS = GRID_ROWS * GRID_COLS

GIF_DURATION_MS = 300
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


def parse_filename(fname):
    base = os.path.basename(fname).replace(".png", "")
    parts = base.split("_")

    storm_id = parts[0]

    # find datetime token
    time_token = next(p for p in parts if "T" in p)
    time = time_token.replace("-", ":")

    # storm type
    storm_type = parts[4]

    return storm_id, time, storm_type


storm_dict = {}

for f in glob(os.path.join(IMG_DIR, "*.png")):
    try:
        storm_id, time, storm_type = parse_filename(f)
       
    except Exception:
        continue

    storm_dict.setdefault(storm_id, {
        "type": storm_type,
        "frames": []
    })

    storm_dict[storm_id]["frames"].append((time, f))


gif_paths = []

for storm_id, info in storm_dict.items():
    frames = sorted(info["frames"], key=lambda x: x[0])

    imgs = [Image.open(f).convert("L") for _, f in frames]

    gif_path = os.path.join(
        OUT_DIR, f"{storm_id}_{info['type']}.gif"
    )

    imgs[0].save(
        gif_path,
        save_all=True,
        append_images=imgs[1:],
        duration=GIF_DURATION_MS,
        loop=0
    )

    gif_paths.append((gif_path, info["type"]))


def plot_gif_table(gifs, title, outname):
    fig, axes = plt.subplots(
        GRID_ROWS, GRID_COLS,
        figsize=(GRID_COLS * 2, GRID_ROWS * 2)
    )

    for ax, (gif_path, _) in zip(axes.flat, gifs):
        gif = Image.open(gif_path)
        ax.imshow(gif.convert("L"))
        ax.set_title(os.path.basename(gif_path).split("_")[0], fontsize=7)
        ax.axis("off")

    for ax in axes.flat[len(gifs):]:
        ax.axis("off")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(outname, dpi=200)
    plt.close()


for storm_type in ["PRECIP", "HAIL", "MIXED"]:
    selected = [g for g in gif_paths if g[1] == storm_type]

    if len(selected) < N_GIFS:
        print(f"⚠️ Not enough {storm_type} storms ({len(selected)})")
        continue

    sample = random.sample(selected, N_GIFS)

    plot_gif_table(
        sample,
        title=f"{storm_type} storm trajectories (random sample)",
        outname=os.path.join(OUT_DIR, f"{storm_type}_gif_table_rs-{RANDOM_SEED}.png")
    )
