import argparse
import glob
import os
import pandas as pd

def calc_mean_gain(routes_csv_path):
    csv_files = glob.glob(os.path.join(routes_csv_path, "*.csv"))
    for trip in csv_files:
        df_trip = pd.read_csv(trip, index_col=0)
        gain_per_trip = df_trip['Gain'].mean()
        print(f"{int(100*gain_per_trip)}%")
    return


def main():
    parser = argparse.ArgumentParser(description="Wind Route Creator")
    parser.add_argument("--ship", required=True, help="afra or suez")
    parser.add_argument("--rotation", required=True, help="100 or 180")

    args = parser.parse_args()
    ship = "abdias_suez" if args.ship == "suez" else "castro_alves_afra"

    routes_csv_path = f"../{ship}/routes_csv_rot{int(args.rotation)}"
    calc_mean_gain(routes_csv_path)

    return


if __name__ == "__main__":
    main()
