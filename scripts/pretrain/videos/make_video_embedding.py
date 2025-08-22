import os
from glob import glob
from PIL import Image

def make_gif_from_folder(frames_dir, output_gif, duration=1000):
    """
    Create a GIF from all PNGs in frames_dir.
    
    Args:
        frames_dir (str): path to folder containing per-frame plots (pngs).
        output_gif (str): path to save the gif.
        duration (int): duration per frame in ms.
    """
    # Collect PNG files
    frame_files = sorted(glob(os.path.join(frames_dir, "*.png")))
    if not frame_files:
        raise ValueError(f"No PNG files found in {frames_dir}")

    print(f"Found {len(frame_files)} frames.")

    # Load frames
    frames = [Image.open(f).convert("RGB") for f in frame_files]

    # Save as GIF
    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )

    print(f"GIF saved to {output_gif}")


if __name__ == "__main__":
    frames_dir = "/data1/fig/dcv2_ir108-cm_100x100_8frames_k9_70k_nc_r2dplus1/epoch_800/all/frames"
    output_gif = os.path.join(os.path.dirname(frames_dir), "embedding_evolution.gif")

    make_gif_from_folder(frames_dir, output_gif, duration=1000)  # 1 sec per frame
