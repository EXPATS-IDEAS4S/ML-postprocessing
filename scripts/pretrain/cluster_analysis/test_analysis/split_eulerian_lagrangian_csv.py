""""
this code aims at reading the csv file generated for the test dataset and split it into two csv files, one for the eulerian data and the other for the lagrangian data. the decision is 
made based on the filename reported in the column crop. if the filename contains the word "eulerian", then the row is written to the eulerian csv file, otherwise if it contains the word "lagrangian", it is written to the lagrangian csv file.
the csvs are stored in a dedicated folder named "test_csvs" to be created in the same directory as the original csv file. the output csv files are named with the string of the variable from the original csv and the ending "eulerian_test.csv" and "lagrangian_test.csv" respectively.
This is done for all csv files containing the string test_all in their name, which are located in the same directory as the original csv file.
input:
- csv folder /sat_data/output/grl_2026/csv/

output:
- csv folder /sat_data/output/grl_2026/csv/test_csvs/
    - eulerian_test.csv
    - lagrangian_test.csv

"""""

import os
import pandas as pd
from pathlib import Path


def main():
    input_dir = Path("/sat_data/output/grl_2026/csv/")
    output_dir = input_dir / "test_csvs"
    output_dir.mkdir(exist_ok=True)

    for csv_file in input_dir.glob("*test_all*.csv"):
        df = pd.read_csv(csv_file)
        eulerian_df = df[df["crop"].str.contains("eulerian", case=False, na=False)]
        lagrangian_df = df[df["crop"].str.contains("lagrangian", case=False, na=False)]

        # estract the string indicating the content of the file
        # crops_stats_var-cma_stats-50-95-25-75_frames-8_timedim_grl_2026_test_all_7045_imergmin.csv it is all what comes before "test_all" in the filename
        variable_name = csv_file.stem.split("_test_all")[0]

        # defining new output file names
        eulerian_output_file = output_dir / f"{variable_name}_eulerian_test.csv"
        lagrangian_output_file = output_dir / f"{variable_name}_lagrangian_test.csv"

        eulerian_df.to_csv(eulerian_output_file, index=False)
        lagrangian_df.to_csv(lagrangian_output_file, index=False)
        print(f"Processed {csv_file.name}:")
        print(f"  - Eulerian data: {len(eulerian_df)} rows -> {eulerian_output_file.name}")
        print(f"  - Lagrangian data: {len(lagrangian_df)} rows -> {lagrangian_output_file.name}")


if __name__ == "__main__":
    main()
    