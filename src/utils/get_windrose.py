import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from windrose import WindroseAxes
import argparse
import os
import glob
from tqdm import tqdm
import random


def get_windrose_from_route(
    route_df,
    output_name="route",
    ax=None,
    hide_cardinal_labels=False,
    invert_orientation=True,
):
    """
    Create windrose from route data with u10, v10 columns

    Parameters:
    route_df: DataFrame with columns ['LAT', 'LON', 'u10', 'v10']
    output_name: name for output file
    ax: matplotlib axis to plot on (for subplots)
    hide_cardinal_labels: if True, hide the compass labels (N, NE, E, ...)
    """
    # Extract u10 and v10 from the route dataframe
    u10 = route_df["u10"].values
    v10 = route_df["v10"].values
    import pdb

    pdb.set_trace()
    # Calculate wind speed and direction
    wind_speed = np.sqrt(u10**2 + v10**2)
    invert_orientation = True
    if invert_orientation:
        wind_direction = np.degrees(np.arctan2(-u10, -v10)) % 360
    else:
        wind_direction = np.degrees(np.arctan2(u10, v10)) % 360

    # Create windrose plot
    if ax is None:
        ax = WindroseAxes.from_ax()
    else:
        ax = WindroseAxes.from_ax(ax=ax)
    plt.setp(ax.get_xticklabels(), fontsize=16)  # Ângulos (N, NE, E, etc.)
    plt.setp(ax.get_yticklabels(), fontsize=14)  # Percentuais radiais

    if hide_cardinal_labels:
        # hide compass labels (N, NE, E, ...)
        plt.setp(ax.get_xticklabels(), visible=False)

    ax.bar(wind_direction, wind_speed, normed=True, opening=0.8, edgecolor="white")
    ax.set_legend(fontsize=12, title_fontsize=14)
    ax.set_title(f"{output_name}", fontsize=20)
    plt.show()
    return wind_speed, wind_direction, ax


def plot_6_windroses():
    """
    Create 6 windroses from specific CSV files
    """
    base_path = r"C:\Users\jung_\OneDrive\Documentos\Poli\TPN\CENPES Descarbonização\abdias_suez\csvs_volta"

    # Define the 6 specific files
    files_info = [
        ("wind_data_year_2020_month_1_day_1_hour_0.csv", "2020 - Jan"),
        ("wind_data_year_2020_month_5_day_1_hour_0.csv", "2020 - May"),
        ("wind_data_year_2020_month_9_day_1_hour_0.csv", "2020 - Oct"),
        ("wind_data_year_2021_month_1_day_1_hour_0.csv", "2021 - Jan"),
        ("wind_data_year_2021_month_5_day_1_hour_0.csv", "2021 - May"),
        ("wind_data_year_2021_month_9_day_1_hour_0.csv", "2021 - Oct"),
    ]

    # Create subplot layout
    fig = plt.figure(figsize=(18, 12))

    for i, (filename, title) in enumerate(files_info):
        file_path = os.path.join(base_path, filename)

        if not os.path.exists(file_path):
            print(f"File not found: {filename}")
            continue

        print(f"Processing {filename}")

        try:
            route_df = pd.read_csv(file_path)

            # Check if required columns exist
            if not all(col in route_df.columns for col in ["u10", "v10"]):
                print(f"Skipping {filename}: missing u10 or v10 columns")
                continue

            # Create subplot
            ax = fig.add_subplot(2, 3, i + 1, projection="windrose")

            # Create windrose
            wind_speed, wind_direction, _ = get_windrose_from_route(
                route_df, output_name=title, ax=ax
            )

            print(
                f"  - Points: {len(route_df)}, Avg wind speed: {np.mean(wind_speed):.2f} m/s"
            )

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    plt.tight_layout()
    os.makedirs("windroses", exist_ok=True)
    plt.savefig("windroses/6_windroses_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


def get_windrose_from_csvs_at_index(
    csv_paths, index, use_window=False, window=101, output_name=None, ax=None
):
    """
    Build a windrose from a list of CSV files by taking the same index
    from each CSV and using their `u_rel` and `v_rel` columns.

    Parameters:
    - csv_paths: iterable of file paths to CSV files
    - index: integer index to extract from each CSV
    - use_window: if True, for each CSV take a neighborhood of `window`
      samples around `index` (helps build a distribution). If False,
      only the single row at `index` from each CSV is used.
    - window: neighborhood size (only used when `use_window=True`)
    - output_name: title for the plot
    - ax: optional matplotlib axis

    Returns:
    - wind_speed, wind_direction, ax
    """
    u_list = []
    v_list = []
    for p in tqdm(csv_paths):
        if not os.path.exists(p):
            print(f"File not found, skipping: {p}")
            continue

        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"Error reading {p}: {e}")
            continue
        if not all(col in df.columns for col in ["u_rel", "v_rel"]):
            print(f"Skipping {p}: missing 'u_rel' or 'v_rel' columns")
            continue

        n = len(df)
        if use_window:
            if window < 1:
                raise ValueError("window must be >= 1")
            half = window // 2
            start = max(0, index - half)
            end = min(n, index + half + 1)
            sub = df.iloc[start:end]
            u_list.extend(sub["u_rel"].values.tolist())
            v_list.extend(sub["v_rel"].values.tolist())
        else:
            if index < 0 or index >= n:
                print(f"Index {index} out of range for {p}, skipping")
                continue
            row = df.iloc[index]
            u_list.append(row["u_rel"])
            v_list.append(row["v_rel"])

    if len(u_list) == 0:
        raise ValueError("No valid u_rel/v_rel samples collected from provided CSVs")

    tmp_df = pd.DataFrame({"u10": np.array(u_list), "v10": np.array(v_list)})

    if output_name is None:
        output_name = f"csvs_index_{index}_n{len(u_list)}"

    return get_windrose_from_route(tmp_df, output_name=output_name, ax=ax)


def get_windrose_from_csvs_all(
    csv_paths,
    cols=("u_rel", "v_rel"),
    output_name=None,
    ax=None,
    max_samples=None,
    shuffle=False,
    option="both",
):
    """
    Build a windrose aggregating ALL indices from a list of CSV routes.

    This collects the specified velocity columns from every CSV and
    constructs a single windrose from the pooled samples.

    Parameters:
    - csv_paths: iterable of CSV file paths
    - cols: tuple(name_u, name_v) columns to use (default 'u_rel','v_rel').
      If those are not present in a file, function will try 'u10','v10'.
    - output_name: optional plot title
    - ax: optional matplotlib axis
    - max_samples: optional int to limit total samples collected (keeps memory bounded)
    - shuffle: if True and max_samples set, samples are randomly chosen across the pool

    Returns:
    - wind_speed, wind_direction, ax
    """
    u_list = []
    v_list = []

    for p in tqdm(csv_paths):
        if not os.path.exists(p):
            print(f"File not found, skipping: {p}")
            continue
        try:
            df = pd.read_csv(p)
            total_n = len(df)
            if option == "both":
                pass
            elif option == "outbound":
                df = df.iloc[: int(total_n / 2)].reset_index(drop=True)
            else:
                df = df.iloc[int(total_n / 2) :].reset_index(drop=True)

        except Exception as e:
            print(f"Error reading {p}: {e}")
            continue

        # prefer provided cols, fallback to u10/v10
        if all(col in df.columns for col in cols):
            u = df[cols[0]].values
            v = df[cols[1]].values
        elif all(col in df.columns for col in ["u10", "v10"]):
            u = df["u10"].values
            v = df["v10"].values
        else:
            print(f"Skipping {p}: missing {cols} and u10/v10")
            continue

        mask = np.isfinite(u) & np.isfinite(v)
        if np.any(mask):
            u_list.extend(u[mask].tolist())
            v_list.extend(v[mask].tolist())

    if len(u_list) == 0:
        raise ValueError("No valid velocity samples found in provided CSVs")

    u_arr = np.array(u_list, dtype=float)
    v_arr = np.array(v_list, dtype=float)

    if max_samples is not None and len(u_arr) > int(max_samples):
        if shuffle:
            idx = np.random.choice(len(u_arr), int(max_samples), replace=False)
        else:
            idx = np.arange(int(max_samples))
        u_arr = u_arr[idx]
        v_arr = v_arr[idx]

    tmp_df = pd.DataFrame({"u10": u_arr, "v10": v_arr})

    if output_name is None:
        output_name = f"all_csvs_n{len(u_arr)}"
    if cols == ("u_rel", "v_rel"):
        invert_orientation = False
    else:
        invert_orientation = True
    return get_windrose_from_route(
        tmp_df, output_name=output_name, ax=ax, invert_orientation=invert_orientation
    )


def main():
    # parser = argparse.ArgumentParser(description="Process wind data.")
    # parser.add_argument("--route-path", help="Single route file path")
    # parser.add_argument(
    #     "--multiple", action="store_true", help="Create 6 windroses comparison"
    # )

    # args = parser.parse_args()
    files = glob.glob(r"D:\castro_alves_afra\route_csvs100\*.csv")
    # Shuffle files so the selection at the same index samples different routes
    random.shuffle(files)
    files = files[:500]
    get_windrose_from_csvs_at_index(files, 50)
    # get_windrose_from_csvs_all(files, option="outbound")
    # get_windrose_from_csvs_all(files, option="return")
    # if args.multiple:
    #     plot_6_windroses()
    # elif args.route_path:
    #     route_df = pd.read_csv(args.route_path)
    #     get_windrose_from_route(
    #         route_df, output_name=os.path.basename(args.route_path).replace(".csv", "")
    #     )
    # else:
    #     print("Please provide --route-path or use --multiple flag")
    #     print("Example: python get_windrose.py --multiple")


if __name__ == "__main__":
    main()
