import argparse
import glob
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import re


def process_single_trip(trip_file):
    """Process a single trip file and return the mean gain"""
    df_trip = pd.read_csv(trip_file, index_col=0)

    filename = os.path.basename(trip_file)
    pattern = (
        r"wind_data_year_(\d{4})_month_(\d{1,2})_day_(\d{1,2})_hour_(\d{1,2})\.csv"
    )
    match = re.search(pattern, filename)

    if match:
        year, month, day, hour = map(int, match.groups())
        timestamp = pd.Timestamp(year=year, month=month, day=day, hour=hour)
    else:
        print(f"[WARNING] Could not extract timestamp from {filename}")
        timestamp = pd.NaT  # Not a Time

    return {"gain": df_trip["gain"].mean(), "timestamp": timestamp}


def plot_gains_per_year(df_100, df_180, outputfolder):
    """
    Plot monthly average gains over time for both 100 RPM and 180 RPM

    Args:
        df_gain: DataFrame with columns [timestamp_100, gain_100, timestamp_180, gain_180]
        outputfolder: Folder to save the plot
    """
    # Separate data for 100 RPM and 180 RPM

    df_100 = df_100.dropna()
    df_100["timestamp"] = pd.to_datetime(df_100["timestamp"])

    df_180 = df_180.dropna()
    df_180["timestamp"] = pd.to_datetime(df_180["timestamp"])

    # Extract year-month for grouping
    df_100["year_month"] = df_100["timestamp"].dt.to_period("M")
    df_180["year_month"] = df_180["timestamp"].dt.to_period("M")

    # Calculate monthly average gains
    monthly_gains_100 = df_100.groupby("year_month")["gain"].mean()
    monthly_gains_180 = df_180.groupby("year_month")["gain"].mean()

    # Calculate overall means for the horizontal lines
    overall_mean_100 = df_100["gain"].mean()
    overall_mean_180 = df_180["gain"].mean()

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot lines with markers for 100 RPM
    ax.plot(
        monthly_gains_100.index.astype(str),
        100 * monthly_gains_100.values,
        marker="o",
        linewidth=2.5,
        markersize=8,
        color="blue",
        label="100 RPM",
        markerfacecolor="lightblue",
        markeredgecolor="blue",
        markeredgewidth=2,
    )

    # Plot lines with markers for 180 RPM
    ax.plot(
        monthly_gains_180.index.astype(str),
        100 * monthly_gains_180.values,
        marker="s",
        linewidth=2.5,
        markersize=8,
        color="red",
        label="180 RPM",
        markerfacecolor="lightcoral",
        markeredgecolor="red",
        markeredgewidth=2,
    )

    # Add horizontal dashed lines for overall means
    ax.axhline(
        y=overall_mean_100 * 100,
        color="blue",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label=f"Média 100 RPM: {100*overall_mean_100:.2f}",
    )
    ax.axhline(
        y=overall_mean_180 * 100,
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label=f"Média 180 RPM: {100*overall_mean_180:.2f}",
    )

    # Customize the plot
    ax.set_xlabel("Mês", fontsize=14, fontweight="bold")
    ax.set_ylabel("Ganho Médio Mensal", fontsize=14, fontweight="bold")
    ax.set_title(
        "Ganhos Médios Mensais: 100 RPM vs 180 RPM", fontsize=16, fontweight="bold"
    )

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right", fontsize=11)
    plt.yticks(fontsize=11)

    # Add grid
    ax.grid(True, alpha=0.4, linestyle=":", linewidth=0.8)

    # Add legend with better styling
    ax.legend(
        loc="best", fontsize=11, framealpha=0.9, fancybox=True, shadow=True, ncol=2
    )
    plt.tight_layout()

    # Save the plot
    os.makedirs("figures/gain", exist_ok=True)
    plt.savefig(
        os.path.join("figures/gain/monthly_gains_comparison.png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.show()

    # Print detailed monthly statistics
    print(f"\n{'='*60}")
    print(f"[GANHOS MENSAIS] 100 RPM:")
    print(f"{'='*60}")
    for month, gain in monthly_gains_100.items():
        print(f"  {month}: {gain:.4f} ({gain*100:.2f}%)")

    print(f"\n{'='*60}")
    print(f"[GANHOS MENSAIS] 180 RPM:")
    print(f"{'='*60}")
    for month, gain in monthly_gains_180.items():
        print(f"  {month}: {gain:.4f} ({gain*100:.2f}%)")

    # Print summary comparison
    print(f"\n{'='*60}")
    print(f"[RESUMO COMPARATIVO]")
    print(f"{'='*60}")
    print(f"Número de meses 100 RPM: {len(monthly_gains_100)}")
    print(f"Número de meses 180 RPM: {len(monthly_gains_180)}")
    print(
        f"Ganho médio geral 100 RPM: {overall_mean_100:.4f} ({overall_mean_100*100:.2f}%)"
    )
    print(
        f"Ganho médio geral 180 RPM: {overall_mean_180:.4f} ({overall_mean_180*100:.2f}%)"
    )
    print(f"Melhoria absoluta: {overall_mean_180 - overall_mean_100:.4f}")
    print(
        f"Melhoria relativa: {((overall_mean_180 - overall_mean_100) / abs(overall_mean_100) * 100):.2f}%"
    )

    return monthly_gains_100, monthly_gains_180


def plot_histogram(trip100, trip180, outputfolder):
    fig, ax = plt.subplots(figsize=(12, 8))
    # Convert to numpy arrays and filter out NaN values
    trip100 = np.array([gain for gain in trip100 if not np.isnan(gain)])
    trip180 = np.array([gain for gain in trip180 if not np.isnan(gain)])

    # Plot overlapping histograms with transparency
    ax.hist(
        trip100,
        bins=30,
        alpha=0.6,
        color="blue",
        edgecolor="black",
        label=f"100 RPM (n={len(trip100)})",
        density=True,
    )
    ax.hist(
        trip180,
        bins=30,
        alpha=0.6,
        color="red",
        edgecolor="black",
        label=f"180 RPM (n={len(trip180)})",
        density=True,
    )

    # Calculate statistics for both datasets
    mean_100 = np.mean(trip100)
    std_100 = np.std(trip100)
    mean_180 = np.mean(trip180)
    std_180 = np.std(trip180)

    # Add vertical lines for means
    ax.axvline(
        mean_100,
        color="blue",
        linestyle="--",
        linewidth=2,
        alpha=0.8,
        label=f"Mean 100 RPM: {mean_100:.3f}",
    )
    ax.axvline(
        mean_180,
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.8,
        label=f"Mean 180 RPM: {mean_180:.3f}",
    )

    # Set labels and title
    ax.set_xlabel("Gain (%)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        "Route Gain Distribution: 100 RPM vs 180 RPM", fontsize=14, fontweight="bold"
    )

    # Limit x-axis range to focus on data
    all_gains = np.concatenate([trip100, trip180])
    # Add legend
    ax.legend(loc="upper right")

    # Add grid
    ax.grid(True, alpha=0.3, linestyle="--")

    # Add statistics text box
    stats_text = (
        f"100 RPM Statistics:\n"
        f"  Mean: {mean_100:.4f}\n"
        f"  Std: {std_100:.4f}\n"
        f"  Count: {len(trip100)}\n\n"
        f"180 RPM Statistics:\n"
        f"  Mean: {mean_180:.4f}\n"
        f"  Std: {std_180:.4f}\n"
        f"  Count: {len(trip180)}"
    )

    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9),
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(outputfolder, "gain_comparison_histogram.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    # Print comparison summary
    print(f"\n[RESULTS] Gain Comparison Summary:")
    print(f"100 RPM - Mean: {mean_100:.4f}, Std: {std_100:.4f}, Count: {len(trip100)}")
    print(f"180 RPM - Mean: {mean_180:.4f}, Std: {std_180:.4f}, Count: {len(trip180)}")
    print(f"Difference in means: {mean_180 - mean_100:.4f}")
    print(f"Relative improvement: {((mean_180 - mean_100) / mean_100 * 100):.2f}%")


def calc_mean_gain_parallel(csv_files, n_jobs=-1):
    print(f"[INFO] Calculating gain from {len(csv_files)} routes using {n_jobs} cores")
    # Parallel processing with progress bar
    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(process_single_trip)(trip)
        for trip in tqdm(csv_files, desc="Processing routes")
    )

    return results


def process_single_trip_with_direction(trip_file):
    """Process a single trip file and return the mean gain for outbound and return"""
    df_trip = pd.read_csv(trip_file, index_col=0)

    filename = os.path.basename(trip_file)
    pattern = (
        r"wind_data_year_(\d{4})_month_(\d{1,2})_day_(\d{1,2})_hour_(\d{1,2})\.csv"
    )
    match = re.search(pattern, filename)

    if match:
        year, month, day, hour = map(int, match.groups())
        timestamp = pd.Timestamp(year=year, month=month, day=day, hour=hour)
    else:
        print(f"[WARNING] Could not extract timestamp from {filename}")
        timestamp = pd.NaT  # Not a Time

    # Separar ida (até linha 3693) e volta (3694 em diante)
    df_outbound = df_trip.iloc[:3693]  # Linha 0 até 3692 (3693 linhas)
    df_return = df_trip.iloc[3693:]  # Linha 3693 em diante

    return {
        "gain_outbound": df_outbound["Gain"].mean() if len(df_outbound) > 0 else np.nan,
        "gain_return": df_return["Gain"].mean() if len(df_return) > 0 else np.nan,
        "timestamp": timestamp,
    }


def calc_mean_gain_parallel_separated(csv_files, n_jobs=-1):
    print(
        f"[INFO] Calculating separated gains from {len(csv_files)} routes using {n_jobs} cores"
    )

    # Parallel processing with progress bar
    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(process_single_trip_with_direction)(trip)
        for trip in tqdm(csv_files, desc="Processing routes (separated)")
    )

    return results


def save_gains_separated_to_csv(gain_per_trip100, gain_per_trip180, ship):
    """
    Save separated gain data to two CSV files: outbound and return

    Args:
        gain_per_trip100: List of dictionaries with separated gains for 100 RPM
        gain_per_trip180: List of dictionaries with separated gains for 180 RPM
        ship: Ship name for file naming
    """

    # Extract outbound gains and timestamps
    gains_100_out = [
        item["gain_outbound"]
        for item in gain_per_trip100
        if not np.isnan(item["gain_outbound"])
    ]
    timestamps_100_out = [
        item["timestamp"]
        for item in gain_per_trip100
        if not np.isnan(item["gain_outbound"])
    ]

    gains_180_out = [
        item["gain_outbound"]
        for item in gain_per_trip180
        if not np.isnan(item["gain_outbound"])
    ]
    timestamps_180_out = [
        item["timestamp"]
        for item in gain_per_trip180
        if not np.isnan(item["gain_outbound"])
    ]

    # Extract return gains and timestamps
    gains_100_ret = [
        item["gain_return"]
        for item in gain_per_trip100
        if not np.isnan(item["gain_return"])
    ]
    timestamps_100_ret = [
        item["timestamp"]
        for item in gain_per_trip100
        if not np.isnan(item["gain_return"])
    ]

    gains_180_ret = [
        item["gain_return"]
        for item in gain_per_trip180
        if not np.isnan(item["gain_return"])
    ]
    timestamps_180_ret = [
        item["timestamp"]
        for item in gain_per_trip180
        if not np.isnan(item["gain_return"])
    ]

    # Create outbound DataFrame
    max_length_out = max(len(gains_100_out), len(gains_180_out))

    if len(gains_100_out) < max_length_out:
        gains_100_out.extend([np.nan] * (max_length_out - len(gains_100_out)))
        timestamps_100_out.extend([pd.NaT] * (max_length_out - len(timestamps_100_out)))

    if len(gains_180_out) < max_length_out:
        gains_180_out.extend([np.nan] * (max_length_out - len(gains_180_out)))
        timestamps_180_out.extend([pd.NaT] * (max_length_out - len(timestamps_180_out)))

    df_gains_outbound = pd.DataFrame(
        {
            "timestamp_100": timestamps_100_out,
            "gain_100": gains_100_out,
            "timestamp_180": timestamps_180_out,
            "gain_180": gains_180_out,
        }
    )

    # Create return DataFrame
    max_length_ret = max(len(gains_100_ret), len(gains_180_ret))

    if len(gains_100_ret) < max_length_ret:
        gains_100_ret.extend([np.nan] * (max_length_ret - len(gains_100_ret)))
        timestamps_100_ret.extend([pd.NaT] * (max_length_ret - len(timestamps_100_ret)))

    if len(gains_180_ret) < max_length_ret:
        gains_180_ret.extend([np.nan] * (max_length_ret - len(gains_180_ret)))
        timestamps_180_ret.extend([pd.NaT] * (max_length_ret - len(timestamps_180_ret)))

    df_gains_return = pd.DataFrame(
        {
            "timestamp_100": timestamps_100_ret,
            "gain_100": gains_100_ret,
            "timestamp_180": timestamps_180_ret,
            "gain_180": gains_180_ret,
        }
    )

    # Save CSVs
    outbound_file = f"../{ship}/total_gains_outbound.csv"
    return_file = f"../{ship}/total_gains_return.csv"

    df_gains_outbound.to_csv(outbound_file, index=False)
    df_gains_return.to_csv(return_file, index=False)

    print(f"[INFO] Outbound gains saved to {outbound_file}")
    print(f"[INFO] Return gains saved to {return_file}")
    print(
        f"[INFO] Outbound - 100 RPM: {len([x for x in gains_100_out if not np.isnan(x)])} entries"
    )
    print(
        f"[INFO] Outbound - 180 RPM: {len([x for x in gains_180_out if not np.isnan(x)])} entries"
    )
    print(
        f"[INFO] Return - 100 RPM: {len([x for x in gains_100_ret if not np.isnan(x)])} entries"
    )
    print(
        f"[INFO] Return - 180 RPM: {len([x for x in gains_180_ret if not np.isnan(x)])} entries"
    )

    return outbound_file, return_file


def plot_gains_separated(df_gains_outbound, df_gains_return, outputfolder):
    """
    Plot separated gains for outbound and return journeys
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    # Plot outbound gains
    df_100_out = df_gains_outbound[["timestamp_100", "gain_100"]].dropna()
    df_100_out = df_100_out.rename(
        columns={"timestamp_100": "timestamp", "gain_100": "gain"}
    )
    df_100_out["timestamp"] = pd.to_datetime(df_100_out["timestamp"])
    df_100_out["year_month"] = df_100_out["timestamp"].dt.to_period("M")

    df_180_out = df_gains_outbound[["timestamp_180", "gain_180"]].dropna()
    df_180_out = df_180_out.rename(
        columns={"timestamp_180": "timestamp", "gain_180": "gain"}
    )
    df_180_out["timestamp"] = pd.to_datetime(df_180_out["timestamp"])
    df_180_out["year_month"] = df_180_out["timestamp"].dt.to_period("M")

    monthly_gains_100_out = df_100_out.groupby("year_month")["gain"].mean()
    monthly_gains_180_out = df_180_out.groupby("year_month")["gain"].mean()

    ax1.plot(
        monthly_gains_100_out.index.astype(str),
        100 * monthly_gains_100_out.values,
        marker="o",
        linewidth=2.5,
        markersize=8,
        color="blue",
        label="100 RPM",
        markerfacecolor="lightblue",
        markeredgecolor="blue",
    )
    ax1.plot(
        monthly_gains_180_out.index.astype(str),
        100 * monthly_gains_180_out.values,
        marker="s",
        linewidth=2.5,
        markersize=8,
        color="red",
        label="180 RPM",
        markerfacecolor="lightcoral",
        markeredgecolor="red",
    )

    ax1.axhline(
        y=df_100_out["gain"].mean() * 100,
        color="blue",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
    )
    ax1.axhline(
        y=df_180_out["gain"].mean() * 100,
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
    )

    ax1.set_title(
        "Ganhos Médios Mensais - Viagem de Ida", fontsize=14, fontweight="bold"
    )
    ax1.set_ylabel("Ganho Médio Mensal (%)", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.4, linestyle=":")
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Plot return gains
    df_100_ret = df_gains_return[["timestamp_100", "gain_100"]].dropna()
    df_100_ret = df_100_ret.rename(
        columns={"timestamp_100": "timestamp", "gain_100": "gain"}
    )
    df_100_ret["timestamp"] = pd.to_datetime(df_100_ret["timestamp"])
    df_100_ret["year_month"] = df_100_ret["timestamp"].dt.to_period("M")

    df_180_ret = df_gains_return[["timestamp_180", "gain_180"]].dropna()
    df_180_ret = df_180_ret.rename(
        columns={"timestamp_180": "timestamp", "gain_180": "gain"}
    )
    df_180_ret["timestamp"] = pd.to_datetime(df_180_ret["timestamp"])
    df_180_ret["year_month"] = df_180_ret["timestamp"].dt.to_period("M")

    monthly_gains_100_ret = df_100_ret.groupby("year_month")["gain"].mean()
    monthly_gains_180_ret = df_180_ret.groupby("year_month")["gain"].mean()

    ax2.plot(
        monthly_gains_100_ret.index.astype(str),
        100 * monthly_gains_100_ret.values,
        marker="o",
        linewidth=2.5,
        markersize=8,
        color="blue",
        label="100 RPM",
        markerfacecolor="lightblue",
        markeredgecolor="blue",
    )
    ax2.plot(
        monthly_gains_180_ret.index.astype(str),
        100 * monthly_gains_180_ret.values,
        marker="s",
        linewidth=2.5,
        markersize=8,
        color="red",
        label="180 RPM",
        markerfacecolor="lightcoral",
        markeredgecolor="red",
    )

    ax2.axhline(
        y=df_100_ret["gain"].mean() * 100,
        color="blue",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
    )
    ax2.axhline(
        y=df_180_ret["gain"].mean() * 100,
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
    )

    ax2.set_title(
        "Ganhos Médios Mensais - Viagem de Volta", fontsize=14, fontweight="bold"
    )
    ax2.set_xlabel("Mês", fontsize=12)
    ax2.set_ylabel("Ganho Médio Mensal (%)", fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.4, linestyle=":")
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(
        os.path.join(outputfolder, "monthly_gains_separated.png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.show()


def total_gain(gains):
    return np.mean(gains)


def main():
    parser = argparse.ArgumentParser(description="Wind Route Creator")
    parser.add_argument("-s", "--ship", required=True, help="afra or suez")
    parser.add_argument("--total-gain", action="store_true")
    parser.add_argument("--gain-histograms", action="store_true")
    parser.add_argument(
        "--separated",
        action="store_true",
        help="Process separated outbound/return gains",
    )

    args = parser.parse_args()
    ship = "abdias_suez" if args.ship == "suez" else "castro_alves_afra"

    routes_csv_path_100 = f"D:/{ship}/route_csvs100"
    csv_files_100 = glob.glob(os.path.join(routes_csv_path_100, "*.csv"))

    routes_csv_path_180 = f"D:/{ship}/route_csvs180"
    csv_files_180 = glob.glob(os.path.join(routes_csv_path_180, "*.csv"))

    if args.separated:
        # Process separated gains
        outbound_file = f"../{ship}/total_gains_outbound.csv"
        return_file = f"../{ship}/total_gains_return.csv"

        if not os.path.exists(outbound_file) or not os.path.exists(return_file):
            gain_dict = calc_mean_gain_parallel_separated(csv_files_100)
            gain_dict = calc_mean_gain_parallel_separated(csv_files_180)
            save_gains_separated_to_csv(gain_dict, gain_dict, ship)

        # Load separated data
        df_gains_outbound = pd.read_csv(outbound_file)
        df_gains_return = pd.read_csv(return_file)

        if args.gain_histograms:
            plot_gains_separated(df_gains_outbound, df_gains_return, f"../{ship}")

    else:
        # Original combined processing
        if not os.path.exists(f"D:/{ship}/gain_100.csv"):
            print("[INFO] Creating gain csv for 100 RPM")
            gain_dict = calc_mean_gain_parallel(csv_files_100)
            df_100 = pd.DataFrame(gain_dict)
            df_100.to_csv(f"D:/{ship}/gain_100.csv")
        else:
            df_100 = pd.read_csv(f"D:/{ship}/gain_100.csv", index_col=0)
        if not os.path.exists(f"D:/{ship}gain_180.csv"):
            print("[INFO] Creating gain csv for 180 RPM")
            gain_dict = calc_mean_gain_parallel(csv_files_180)
            df_180 = pd.DataFrame(gain_dict)
            df_180.to_csv(f"D:/{ship}/gain_180.csv")
        else:
            df_180 = pd.read_csv(f"D:/{ship}/gain_180.csv", index_col=0)

        gain_100 = df_100["gain"]
        gain_180 = df_180["gain"]

        if args.total_gain:
            total_gain_100 = total_gain(gain_100)
            print(f"[INFO] Total gain for 100 RPM: {total_gain_100}")

            total_gain_180 = total_gain(gain_180)
            print(f"[INFO] Total gain for 180 RPM: {total_gain_180}")

        if args.gain_histograms:
            plot_gains_per_year(df_100, df_180, f"../{ship}")
            plot_histogram(gain_100, gain_180, f"../{ship}")


if __name__ == "__main__":
    main()
