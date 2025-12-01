import os
import re
import glob
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_timestamp_from_filename(fname):
    """Parse timestamp from filenames like
    wind_data_year_2020_month_1_day_1_hour_14.csv
    Returns pandas.Timestamp or None.
    """
    base = os.path.basename(fname)
    pattern = r"wind_data_year_(\d{4})_month_(\d{1,2})_day_(\d{1,2})_hour_(\d{1,2})"
    m = re.search(pattern, base)
    if not m:
        return None
    y, mo, d, h = map(int, m.groups())
    try:
        return pd.Timestamp(year=y, month=mo, day=d, hour=h)
    except Exception:
        return None


def compute_trip_means(csv_dir, max_files=None, leg="both", split_index=3693):
    """Read CSV files from a folder, parse timestamps from filenames and
    compute mean gain (in percent) per trip. Returns a DataFrame with
    columns ['file','timestamp','mean_gain_pct','month','hour'].
    """
    files = glob.glob(os.path.join(csv_dir, "*.csv"))
    files = sorted(files)
    if max_files:
        files = files[:max_files]

    records = []
    for f in tqdm(files, total=len(files)):
        try:
            ts = parse_timestamp_from_filename(f)
            df = pd.read_csv(f, index_col=0)
            # support both 'gain' and 'Gain' column names
            gain_col = None
            for c in ("gain", "Gain"):
                if c in df.columns:
                    gain_col = c
                    break
            if gain_col is None:
                continue

            # select leg slice if requested
            if leg == "both" or leg is None:
                mean_gain = df[gain_col].dropna().mean() * 100.0
            elif leg == "outbound":
                idx = min(split_index, len(df))
                mean_gain = df.iloc[:idx][gain_col].dropna().mean() * 100.0
            elif leg == "return":
                idx = min(split_index, len(df))
                mean_gain = df.iloc[idx:][gain_col].dropna().mean() * 100.0
            else:
                # unknown option, treat as both
                mean_gain = df[gain_col].dropna().mean() * 100.0
            records.append({"file": f, "timestamp": ts, "mean_gain_pct": mean_gain})
        except Exception as e:
            print(f"[WARN] skipping {f}: {e}")

    df_rec = pd.DataFrame(records)
    if df_rec.empty:
        raise RuntimeError(f"No valid trip files found in folder {csv_dir}")

    missing_ts = df_rec["timestamp"].isna().sum()
    if missing_ts > 0:
        print(
            f"[WARN] {missing_ts} files did not have a parsable timestamp and will be ignored for temporal grouping"
        )
        df_rec = df_rec.dropna(subset=["timestamp"]).reset_index(drop=True)

    df_rec["month"] = df_rec["timestamp"].dt.month
    df_rec["hour"] = df_rec["timestamp"].dt.hour
    return df_rec


def compute_trip_means_from_dir(csv_dir, max_files=None, leg="both", split_index=3693):
    return compute_trip_means(
        csv_dir, max_files=max_files, leg=leg, split_index=split_index
    )


def plot_monthly_boxplot_dual(
    df_a,
    df_b,
    label_a="100 RPM",
    label_b="180 RPM",
    out_png="figures/temporal/boxplot_monthly_dual.png",
    leg="both",
):
    """Create combined monthly boxplots for two datasets using year-month periods as x-axis.

    This function builds a sorted union of year-month periods present in either
    dataframe and draws side-by-side boxplots centered on each period tick.
    """
    # build Period index for a monthly grouping (year-month)
    per_a = df_a["timestamp"].dt.to_period("M")
    per_b = df_b["timestamp"].dt.to_period("M")
    periods = sorted(set(per_a.unique()).union(set(per_b.unique())))
    if len(periods) == 0:
        raise RuntimeError("No monthly periods found in input dataframes")

    data_a = [
        df_a.loc[df_a["timestamp"].dt.to_period("M") == p, "mean_gain_pct"].values
        for p in periods
    ]
    data_b = [
        df_b.loc[df_b["timestamp"].dt.to_period("M") == p, "mean_gain_pct"].values
        for p in periods
    ]

    n = len(periods)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.45), 6))
    # numeric positions for periods
    positions = np.arange(1, n + 1)
    half_offset = 0.15
    positions_a = positions - half_offset
    positions_b = positions + half_offset
    width = half_offset * 1.6

    bp_a = ax.boxplot(
        data_a, positions=positions_a, widths=width, patch_artist=True, showfliers=False
    )
    bp_b = ax.boxplot(
        data_b, positions=positions_b, widths=width, patch_artist=True, showfliers=False
    )

    for box in bp_a["boxes"]:
        box.set(facecolor="#4C72B0", alpha=0.6)
    for box in bp_b["boxes"]:
        box.set(facecolor="#C44E52", alpha=0.6)

    mean_a = [
        np.nanmean(d) if (len(d) and np.any(np.isfinite(d))) else np.nan for d in data_a
    ]
    mean_b = [
        np.nanmean(d) if (len(d) and np.any(np.isfinite(d))) else np.nan for d in data_b
    ]

    ax.plot(
        positions,
        mean_a,
        "-o",
        color="#1f77b4",
        label=f"Média {label_a}: {np.nanmean(mean_a):.2f}",
    )
    ax.plot(
        positions,
        mean_b,
        "-s",
        color="#d62728",
        label=f"Média {label_b}: {np.nanmean(mean_b):.2f}",
    )

    if np.isfinite(np.nanmean(mean_a)):
        ax.axhline(np.nanmean(mean_a), color="#1f77b4", linestyle="--", alpha=0.7)
    if np.isfinite(np.nanmean(mean_b)):
        ax.axhline(np.nanmean(mean_b), color="#d62728", linestyle="--", alpha=0.7)

    # set xticks and labels as year-month
    xticks = positions
    xticklabels = [p.strftime("%Y-%m") for p in periods]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=45, ha="right")
    ax.set_xlim(0.5, n + 0.5)
    ax.set_xlabel("Ano-Mês (2020–2021)")
    ax.set_ylabel("Ganho médio por viagem (%)")
    # adjust title depending on selected leg
    if leg == "outbound":
        title_leg = " (ida)"
    elif leg == "return":
        title_leg = " (volta)"
    else:
        title_leg = ""
    ax.set_title(f"Ganho médio mensal: {label_a} vs {label_b}{title_leg}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_monthly_boxplot(df_rec, out_png, leg="both"):
    # group by year-month periods (YYYY-MM)
    per = df_rec["timestamp"].dt.to_period("M")
    periods = sorted(per.unique())
    if len(periods) == 0:
        raise RuntimeError("No monthly periods found in dataframe")

    data = [
        df_rec.loc[df_rec["timestamp"].dt.to_period("M") == p, "mean_gain_pct"].values
        for p in periods
    ]
    plt.figure(figsize=(max(8, len(periods) * 0.4), 5))
    plt.boxplot(data, labels=[p.strftime("%Y-%m") for p in periods], showfliers=False)
    plt.xlabel("Ano-Mês (2020–2021)")
    plt.ylabel("Ganho médio por viagem (%)")
    if leg == "outbound":
        extra = " (ida)"
    elif leg == "return":
        extra = " (volta)"
    else:
        extra = ""
    plt.title(f"Distribuição do ganho médio por viagem por mês{extra} (2020–2021)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_hourly_boxplot(df_rec, out_png):
    plt.figure(figsize=(12, 5))
    order = list(range(0, 24))
    data = [df_rec.loc[df_rec["hour"] == h, "mean_gain_pct"].values for h in order]
    plt.boxplot(data, labels=[str(h) for h in order], showfliers=False)
    plt.xlabel("Hora de partida (UTC)")
    plt.ylabel("Ganho médio por viagem (%)")
    plt.title("Distribuição do ganho médio por viagem por hora de partida (2020–2021)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_stats_table(df_rec, out_csv):
    stats = (
        df_rec.groupby("month")["mean_gain_pct"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    stats.to_csv(out_csv, index=False)


def run(
    csv_dir, out_dir="figures/temporal", max_files=200, leg="both", split_index=3693
):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Reading CSVs from {csv_dir} (max {max_files}), leg={leg}")
    df_rec = compute_trip_means(
        csv_dir, max_files=max_files, leg=leg, split_index=split_index
    )
    print(f"Found {len(df_rec)} trips with timestamps")
    monthly_png = os.path.join(out_dir, "boxplot_monthly_gain.png")
    hourly_png = os.path.join(out_dir, "boxplot_hourly_gain.png")
    csv_stats = os.path.join(out_dir, "monthly_stats.csv")

    plot_monthly_boxplot(df_rec, monthly_png, leg=leg)
    plot_hourly_boxplot(df_rec, hourly_png)
    save_stats_table(df_rec, csv_stats)
    print(f"Saved plots to {monthly_png} and {hourly_png}")
    print(f"Saved monthly stats to {csv_stats}")


def run_compare(
    csv_dir_a,
    csv_dir_b,
    out_dir="figures/temporal",
    max_files=200,
    label_a="100 RPM",
    label_b="180 RPM",
    leg="both",
    split_index=3693,
):
    os.makedirs(out_dir, exist_ok=True)
    df_a = compute_trip_means_from_dir(
        csv_dir_a, max_files=max_files, leg=leg, split_index=split_index
    )
    df_b = compute_trip_means_from_dir(
        csv_dir_b, max_files=max_files, leg=leg, split_index=split_index
    )
    out_png = os.path.join(out_dir, f"boxplot_monthly_dual_{leg}.png")
    plot_monthly_boxplot_dual(
        df_a, df_b, label_a=label_a, label_b=label_b, out_png=out_png, leg=leg
    )
    df_a.groupby("month")["mean_gain_pct"].agg(
        ["count", "mean", "median", "std"]
    ).to_csv(
        os.path.join(out_dir, "monthly_stats_" + label_a.replace(" ", "") + ".csv")
    )
    df_b.groupby("month")["mean_gain_pct"].agg(
        ["count", "mean", "median", "std"]
    ).to_csv(
        os.path.join(out_dir, "monthly_stats_" + label_b.replace(" ", "") + ".csv")
    )
    print(f"Saved dual monthly boxplot to {out_png}")


def _build_argparser():
    p = argparse.ArgumentParser(
        description="Temporal analysis of trip gains (monthly/hourly)."
    )
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--dir", help="Folder with CSVs (single-mode)")
    grp.add_argument(
        "--dir-compare",
        nargs=2,
        metavar=("DIR_A", "DIR_B"),
        help="Two folders to compare (e.g. 100RPM 180RPM)",
    )
    p.add_argument("--out", default="figures/temporal", help="Output folder")
    p.add_argument(
        "--max-files",
        type=int,
        default=150000,
        help="Max number of files to process per folder",
    )
    p.add_argument(
        "--label-a", default="100 RPM", help="Label for first folder (used in compare)"
    )
    p.add_argument(
        "--label-b", default="180 RPM", help="Label for second folder (used in compare)"
    )
    p.add_argument(
        "--leg",
        choices=["both", "outbound", "return"],
        default="both",
        help="Which leg to aggregate: 'both' (default), 'outbound' or 'return'",
    )
    p.add_argument(
        "--split-index",
        type=int,
        default=3693,
        help="Row index to split outbound/return when CSVs contain both (default 3693)",
    )
    return p


if __name__ == "__main__":
    parser = _build_argparser()
    args = parser.parse_args()
    if args.dir:
        run(
            args.dir,
            out_dir=args.out,
            max_files=args.max_files,
            leg=args.leg,
            split_index=args.split_index,
        )
    else:
        a, b = args.dir_compare
        run_compare(
            a,
            b,
            out_dir=args.out,
            max_files=args.max_files,
            label_a=args.label_a,
            label_b=args.label_b,
            leg=args.leg,
            split_index=args.split_index,
        )
