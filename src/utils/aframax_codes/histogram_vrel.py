import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def plot_histogram(data, col, unit):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(data, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
    mean_val = np.mean(data)
    std_val = np.std(data)
    ax.axvline(mean_val, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.2f}")
    ax.set_xlabel(f"{col} ({unit})")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Histogram of {col}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    stats_text = f"Count: {len(data)}\nMean: {mean_val:.2f}\nStd: {std_val:.2f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    plt.tight_layout()
    plt.show()


def main():
    csvs_ida = glob.glob(os.path.join('../castro_alves_afra/csvs_ida', '*.csv'))
    csvs_volta = glob.glob(os.path.join('../castro_alves_afra/csvs_volta', '*.csv'))
    all_u_rel = []
    all_v_rel = []

    for csv_path in tqdm(csvs_ida + csvs_volta, total=len(csvs_ida)+len(csvs_volta)):
        df = pd.read_csv(csv_path)
        if 'u_rel' in df.columns and 'v_rel' in df.columns:
            all_u_rel.append(df['u_rel'].values)
            all_v_rel.append(df['v_rel'].values)

    # Concatenate all values
    u_rel = np.concatenate(all_u_rel)
    v_rel = np.concatenate(all_v_rel)
    mag = np.sqrt(u_rel**2 + v_rel**2)

    plot_histogram(mag, col='|Vrel|', unit='m/s')

if __name__ == "__main__":
    main()