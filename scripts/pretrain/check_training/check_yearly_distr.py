import os
import random
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset


# ======================================================
# === CONFIGURATION ===
# ======================================================
nc_dir = (
    "/data1/crops/ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc/nc/1"
)

output_dir = (
    "/data1/fig/dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc"
)
os.makedirs(output_dir, exist_ok=True)

VAR_NAME = "IR_108"
MAX_SAMPLES_PER_YEAR = 100000
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ======================================================
# === COLLECT FILES BY YEAR ===
# ======================================================
files_per_year = defaultdict(list)

for fname in os.listdir(nc_dir):
    if not fname.endswith(".nc"):
        continue

    # Example: 2025-09-30T23:45:00_1.nc
    try:
        year = int(fname.split("-")[0])
    except ValueError:
        print(f"Skipping malformed filename: {fname}")
        continue

    files_per_year[year].append(os.path.join(nc_dir, fname))

years = sorted(files_per_year.keys())
print(f"Found years: {years}")
#print count per year
for year in years:
    print(f"Year {year}: {len(files_per_year[year])} files")



# ======================================================
# === LOAD & SUBSAMPLE IR_108 VALUES PER YEAR ===
# ======================================================
values_per_year = {}

for year in years:
    all_values = []

    file_list = files_per_year[year]
    random.shuffle(file_list)

    for nc_path in file_list:
        try:
            with Dataset(nc_path, "r") as ds:
                if VAR_NAME not in ds.variables:
                    continue

                data = ds.variables[VAR_NAME][:]

                # flatten & remove invalid values
                data = np.asarray(data).ravel()
                data = data[np.isfinite(data)]

                all_values.append(data)

        except Exception as e:
            print(f"Error reading {nc_path}: {e}")
            continue

        # stop early if enough samples
        if sum(len(v) for v in all_values) >= MAX_SAMPLES_PER_YEAR:
            break

    if not all_values:
        continue

    all_values = np.concatenate(all_values)

    # random subsample
    if len(all_values) > MAX_SAMPLES_PER_YEAR:
        idx = np.random.choice(len(all_values), MAX_SAMPLES_PER_YEAR, replace=False)
        all_values = all_values[idx]

    values_per_year[year] = all_values

    print(f"Year {year}: {len(all_values)} samples")


# ======================================================
# === PLOT DISTRIBUTIONS ===
# ======================================================
plt.figure(figsize=(6, 4))

for year in years:
    if year not in values_per_year:
        continue

    plt.hist(
        values_per_year[year],
        bins=50,
        density=True,
        histtype="step",
        linewidth=2,
        label=str(year)
    )

plt.xlabel("IR 10.8 μm Brightness Temperature", fontsize=12)
plt.ylabel("Probability Density", fontsize=12)
plt.yscale("log")
plt.title("IR 10.8 μm Distribution by Year", fontweight="bold")
plt.grid(alpha=0.3)
plt.legend(title="Year")

plt.tight_layout()

outfile = os.path.join(output_dir, "ir108_distribution_by_year.png")
plt.savefig(outfile, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved distribution plot: {outfile}")
