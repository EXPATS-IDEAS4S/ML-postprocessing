from __future__ import annotations

"""
Print the content of NumPy .npz files used by the GRL 2026 analysis.

How to run:
- Print every .npz file in the default directory:
    `python print_npz_content.py`

- Print one specific file:
    `python print_npz_content.py --file /sat_data/output/grl_2026/npz/mean_gradients_cth.npz`

- Print with longer arrays before truncation:
    `python print_npz_content.py --threshold 1000`
"""

import argparse
from pathlib import Path

import numpy as np


DEFAULT_NPZ_DIR = Path("/sat_data/output/grl_2026/npz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print keys and array contents from .npz files."
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Path to one .npz file. If omitted, all .npz files in --npz-dir are printed.",
    )
    parser.add_argument(
        "--npz-dir",
        type=Path,
        default=DEFAULT_NPZ_DIR,
        help=f"Directory containing .npz files. Default: {DEFAULT_NPZ_DIR}",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=200,
        help="Number of array elements NumPy prints before truncating. Use -1 for full arrays.",
    )
    return parser.parse_args()


def resolve_npz_files(file_path: Path | None, npz_dir: Path) -> list[Path]:
    if file_path is not None:
        if not file_path.exists():
            raise FileNotFoundError(f"Could not find .npz file: {file_path}")
        if file_path.suffix != ".npz":
            raise ValueError(f"Expected a .npz file, got: {file_path}")
        return [file_path]

    if not npz_dir.exists():
        raise FileNotFoundError(f"Could not find .npz directory: {npz_dir}")

    npz_files = sorted(npz_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in: {npz_dir}")
    return npz_files


def print_npz_file(npz_file: Path) -> None:
    print("=" * 80)
    print(f"File: {npz_file}")

    with np.load(npz_file, allow_pickle=True) as data:
        print(f"Keys: {data.files}")

        for key in data.files:
            value = data[key]
            print("-" * 80)
            print(f"Key: {key}")
            print(f"Shape: {value.shape}")
            print(f"Dtype: {value.dtype}")
            print("Content:")
            print(value)


def main() -> None:
    args = parse_args()
    threshold = None if args.threshold < 0 else args.threshold
    np.set_printoptions(threshold=threshold, linewidth=140)

    for npz_file in resolve_npz_files(args.file, args.npz_dir):
        print_npz_file(npz_file)


if __name__ == "__main__":
    main()
