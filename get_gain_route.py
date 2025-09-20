import argparse
import glob
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt


def process_single_trip(trip_file):
    """Process a single trip file and return the mean gain"""
    df_trip = pd.read_csv(trip_file, index_col=0)
    return df_trip['Gain'].mean()

def plot_histogram(trip100, trip180):
    fig, ax = plt.subplots(figsize=(12, 8))
    # Convert to numpy arrays and filter out NaN values
    trip100 = np.array([gain for gain in trip100 if not np.isnan(gain)])
    trip180 = np.array([gain for gain in trip180 if not np.isnan(gain)])
    
    # Plot overlapping histograms with transparency
    ax.hist(trip100, bins=30, alpha=0.6, color="blue", edgecolor="black", 
            label=f"100 RPM (n={len(trip100)})", density=True)
    ax.hist(trip180, bins=30, alpha=0.6, color="red", edgecolor="black", 
            label=f"180 RPM (n={len(trip180)})", density=True)
    
    # Calculate statistics for both datasets
    mean_100 = np.mean(trip100)
    std_100 = np.std(trip100)
    mean_180 = np.mean(trip180)
    std_180 = np.std(trip180)
    
    # Add vertical lines for means
    ax.axvline(mean_100, color="blue", linestyle="--", linewidth=2, alpha=0.8,
               label=f"Mean 100 RPM: {mean_100:.3f}")
    ax.axvline(mean_180, color="red", linestyle="--", linewidth=2, alpha=0.8,
               label=f"Mean 180 RPM: {mean_180:.3f}")
    
    # Set labels and title
    ax.set_xlabel("Gain (%)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Route Gain Distribution: 100 RPM vs 180 RPM", fontsize=14, fontweight='bold')
    
    # Limit x-axis range to focus on data
    all_gains = np.concatenate([trip100, trip180])
    x_min = np.percentile(all_gains, 1)  # 1st percentile
    x_max = np.percentile(all_gains, 99)  # 99th percentile
    ax.set_xlim(x_min, x_max)
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add statistics text box
    stats_text = (f"100 RPM Statistics:\n"
                 f"  Mean: {mean_100:.4f}\n"
                 f"  Std: {std_100:.4f}\n"
                 f"  Count: {len(trip100)}\n\n"
                 f"180 RPM Statistics:\n"
                 f"  Mean: {mean_180:.4f}\n"
                 f"  Std: {std_180:.4f}\n"
                 f"  Count: {len(trip180)}")
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9))
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f'gain_comparison_histogram.png', dpi=300, bbox_inches='tight')
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
    gain_per_trip = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(process_single_trip)(trip) 
        for trip in tqdm(csv_files, desc="Processing routes")
    )
    
    return gain_per_trip

def total_gain(gains):
    return np.mean(gains)

def save_gains_to_csv(gain_per_trip100, gain_per_trip180, outputfile):
    """
    Save gain data to CSV with two columns: 100 and 180
    
    Args:
        gain_per_trip100: List of gains for 100 RPM
        gain_per_trip180: List of gains for 180 RPM
        ship: Ship name for filename
    """
    # Convert to numpy arrays and filter out NaN values
    gains_100 = np.array([gain for gain in gain_per_trip100 if not np.isnan(gain)])
    gains_180 = np.array([gain for gain in gain_per_trip180 if not np.isnan(gain)])
    
    # Find the maximum length to pad shorter array with NaN
    max_length = max(len(gains_100), len(gains_180))
    
    # Pad shorter array with NaN
    if len(gains_100) < max_length:
        gains_100 = np.pad(gains_100, (0, max_length - len(gains_100)), 
                          constant_values=np.nan)
    if len(gains_180) < max_length:
        gains_180 = np.pad(gains_180, (0, max_length - len(gains_180)), 
                          constant_values=np.nan)
    
    # Create DataFrame
    df_gains = pd.DataFrame({
        '100': gains_100,
        '180': gains_180
    })
    
    # Save to CSV
    df_gains.to_csv(outputfile, index=False)
    
    print(f"[INFO] Gains saved to {outputfile}")
    print(f"[INFO] 100 RPM entries: {len([x for x in gains_100 if not np.isnan(x)])}")
    print(f"[INFO] 180 RPM entries: {len([x for x in gains_180 if not np.isnan(x)])}")
    
    return outputfile


def main():
    parser = argparse.ArgumentParser(description="Wind Route Creator")
    parser.add_argument("--ship", required=True, help="afra or suez")
    parser.add_argument("--total-gain", action="store_true") 
    parser.add_argument("--gain-histograms", action="store_true") 
    parser.add_argument("--save-csv", action="store_true", help="Save gains to CSV")
    
    args = parser.parse_args()
    ship = "abdias_suez" if args.ship == "suez" else "castro_alves_afra"

    routes_csv_path_100 = f"../{ship}/routes_csv_rot100"
    csv_files_100 = glob.glob(os.path.join(routes_csv_path_100, "*.csv"))

    routes_csv_path_180 = f"../{ship}/routes_csv_rot180"
    csv_files_180 = glob.glob(os.path.join(routes_csv_path_180, "*.csv"))
    
    gain_per_trip100 = calc_mean_gain_parallel(csv_files_100)
    gain_per_trip180 = calc_mean_gain_parallel(csv_files_180)
    
    if args.total_gain: 
        final_gain_100 = total_gain(gain_per_trip100)
        final_gain_180 = total_gain(gain_per_trip180)
        print(f"[INFO] Total gain 100 RPM: {final_gain_100:.4f}")
        print(f"[INFO] Total gain 180 RPM: {final_gain_180:.4f}")
        print(f"[INFO] Difference: {final_gain_180 - final_gain_100:.4f}")
    
    if args.gain_histograms:
        plot_histogram(gain_per_trip100, gain_per_trip180)
    
    if args.save_csv:
        save_gains_to_csv(gain_per_trip100, gain_per_trip180, ship)

if __name__ == "__main__":
    main()
def main():
    parser = argparse.ArgumentParser(description="Wind Route Creator")
    parser.add_argument("--ship", required=True, help="afra or suez")
    parser.add_argument("--total-gain", action="store_true") 
    parser.add_argument("--gain-histograms", action="store_true") 
    
    args = parser.parse_args()
    ship = "abdias_suez" if args.ship == "suez" else "castro_alves_afra"

    routes_csv_path_100 = f"../{ship}/routes_csv_rot100"
    csv_files_100 = glob.glob(os.path.join(routes_csv_path_100, "*.csv"))

    routes_csv_path_180 = f"../{ship}/routes_csv_rot180"
    csv_files_180 = glob.glob(os.path.join(routes_csv_path_180, "*.csv"))
    gain_per_trip100 = calc_mean_gain_parallel(csv_files_100)
    gain_per_trip180 = calc_mean_gain_parallel(csv_files_180)
    save_gains_to_csv(gain_per_trip100, gain_per_trip180, f"../{ship}/total_gains.csv")
    if args.total_gain: 
        final_gain = total_gain(gain_per_trip100)
        print(f"[INFO] Total main: {final_gain}")
    
    if args.gain_histograms:
        plot_histogram(gain_per_trip100, gain_per_trip180)
        pass


if __name__ == "__main__":
    main()
