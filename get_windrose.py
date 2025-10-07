import xarray as xr
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from windrose import WindroseAxes
import argparse
import os
import glob

def get_windrose_from_route(route_df, output_name="route", ax=None):
    """
    Create windrose from route data with u10, v10 columns
    
    Parameters:
    route_df: DataFrame with columns ['LAT', 'LON', 'u10', 'v10']
    output_name: name for output file
    ax: matplotlib axis to plot on (for subplots)
    """
    # Extract u10 and v10 from the route dataframe
    u10 = route_df['u10'].values
    v10 = route_df['v10'].values
    
    # Calculate wind speed and direction
    wind_speed = np.sqrt(u10**2 + v10**2)
    wind_direction = np.degrees(np.arctan2(-u10, -v10)) % 360
    
    # Create windrose plot
    if ax is None:
        ax = WindroseAxes.from_ax()
    else:
        ax = WindroseAxes.from_ax(ax=ax)
    plt.setp(ax.get_xticklabels(), fontsize=16)  # Ângulos (N, NE, E, etc.)
    plt.setp(ax.get_yticklabels(), fontsize=14)  # Percentuais radiais
    
    ax.bar(wind_direction, wind_speed, normed=True, opening=0.8, edgecolor='white')
    ax.set_legend(fontsize=12, title_fontsize=14)
    ax.set_title(f'{output_name}', fontsize=20)
    
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
        ("wind_data_year_2021_month_9_day_1_hour_0.csv", "2021 - Oct")
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
            if not all(col in route_df.columns for col in ['u10', 'v10']):
                print(f"Skipping {filename}: missing u10 or v10 columns")
                continue
            
            # Create subplot
            ax = fig.add_subplot(2, 3, i+1, projection="windrose")
            
            # Create windrose
            wind_speed, wind_direction, _ = get_windrose_from_route(
                route_df, 
                output_name=title,
                ax=ax
            )
            
            print(f"  - Points: {len(route_df)}, Avg wind speed: {np.mean(wind_speed):.2f} m/s")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    plt.tight_layout()
    os.makedirs("windroses", exist_ok=True)
    plt.savefig('windroses/6_windroses_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Process wind data.')
    parser.add_argument('--route-path', help='Single route file path')
    parser.add_argument('--multiple', action='store_true', help='Create 6 windroses comparison')
    
    args = parser.parse_args()
    
    if args.multiple:
        plot_6_windroses()
    elif args.route_path:
        route_df = pd.read_csv(args.route_path)
        get_windrose_from_route(route_df, output_name=os.path.basename(args.route_path).replace(".csv", ""))
    else:
        print("Please provide --route-path or use --multiple flag")
        print("Example: python get_windrose.py --multiple")

if __name__ == "__main__":
    main()