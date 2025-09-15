import argparse
import glob
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

def process_single_trip(trip_file):
    """Process a single trip file and return the mean gain"""
    df_trip = pd.read_csv(trip_file, index_col=0)
    return df_trip['Gain'].mean()

def calc_mean_gain_parallel(routes_csv_path, n_jobs=-1):
    csv_files = glob.glob(os.path.join(routes_csv_path, "*.csv"))
    print(f"[INFO] Calculating gain from {len(csv_files)} routes using {n_jobs} cores")
    
    # Parallel processing with progress bar
    gain_per_trip = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(process_single_trip)(trip) 
        for trip in tqdm(csv_files, desc="Processing routes")
    )
    
    return gain_per_trip

def total_gain(gains):
    return np.mean(gains)

def main():
    parser = argparse.ArgumentParser(description="Wind Route Creator")
    parser.add_argument("--ship", required=True, help="afra or suez")
    parser.add_argument("--rotation", required=True, help="100 or 180")

    args = parser.parse_args()
    ship = "abdias_suez" if args.ship == "suez" else "castro_alves_afra"

    routes_csv_path = f"../{ship}/routes_csv_rot{int(args.rotation)}"
    gain_per_trip = calc_mean_gain_parallel(routes_csv_path)
    final_gain = total_gain(gain_per_trip)
    print(f"[INFO] Total main: {final_gain}")
    return


if __name__ == "__main__":
    main()
