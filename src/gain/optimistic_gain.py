import argparse
import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Optimistic gain method")
    parser.add_argument("-p", "--path", required=True, help="folder path")
    parser.add_argument("--pcons", type=int, required=True, help="consumed power")
    parser.add_argument("-o", "--output", required=True, help="output folder")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    csv_files = glob.glob(os.path.join(args.path, "*.csv"))
    P_E_carreg = 640 * (12 / 1.94384) / 0.63
    P_E_lastro = 475 * (12 / 1.94384) / 0.63

    for csv_file in tqdm(csv_files):
        df = pd.read_csv(csv_file)
        df["p_cons"] = np.ones(len(df)) * args.pcons
        df["p_e_rotor"] = df["p_cons"] + df["p_prop"]

        mid_idx = len(df) // 2
        df["gain"] = np.nan
        df.loc[: mid_idx - 1, "gain"] = (
            1 - df.loc[: mid_idx - 1, "p_e_rotor"] / P_E_carreg
        )
        df.loc[mid_idx:, "gain"] = 1 - df.loc[mid_idx:, "p_e_rotor"] / P_E_lastro

        df.to_csv(os.path.join(args.output, os.path.basename(csv_file)))


main()
