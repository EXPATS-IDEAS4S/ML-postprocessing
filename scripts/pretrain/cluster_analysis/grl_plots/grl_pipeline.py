"""
GRL pipeline script to run all figure generation scripts in order.
Usage:
python scripts/pretrain/cluster_analysis/grl_plots/grl_pipeline.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path



SCRIPT_DIR = Path(__file__).resolve().parent
FIGURE_SCRIPTS = {
    "figure1": SCRIPT_DIR / "figure1_data_methods.py",
    "figure2": SCRIPT_DIR / "figure2_classes_representativity.py",
    "figure3": SCRIPT_DIR / "figure3_physics_interpretability.py",
    "figure4": SCRIPT_DIR / "figure4_convection_characterization.py",
}
DEFAULT_ORDER = ["figure1", "figure2", "figure3", "figure4"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full GRL figure generation pipeline."
    )
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=DEFAULT_ORDER,
        default=DEFAULT_ORDER,
        help="Subset of figures to run. Defaults to all figures in order.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running remaining figures if one figure script fails.",
    )
    return parser.parse_args()


def run_figure(figure_name: str) -> int:
    script_path = FIGURE_SCRIPTS[figure_name]
    command = [sys.executable, str(script_path)]
    print(f"Running {figure_name}: {' '.join(command)}")
    completed = subprocess.run(command, cwd=SCRIPT_DIR.parent.parent.parent.parent)
    return completed.returncode


def main() -> None:
    args = parse_args()
    failures: list[str] = []

    for figure_name in args.figures:
        return_code = run_figure(figure_name)
        if return_code == 0:
            print(f"Completed {figure_name}")
            continue

        failures.append(figure_name)
        print(f"Failed {figure_name} with exit code {return_code}")
        if not args.continue_on_error:
            raise SystemExit(return_code)

    if failures:
        raise SystemExit(
            f"GRL pipeline finished with failures: {', '.join(failures)}"
        )

    print("GRL pipeline completed successfully.")


if __name__ == "__main__":
    main()