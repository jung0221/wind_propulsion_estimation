import xarray as xr
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from windrose import WindroseAxes
import argparse

def get_windrose_from_route(route_df, output_name="route"):
    """
    Create windrose from route data with u10, v10 columns
    
    Parameters:
    route_df: DataFrame with columns ['LAT', 'LON', 'u10', 'v10']
    output_name: name for output file
    """
    # Extract u10 and v10 from the route dataframe
    u10 = route_df['u10'].values
    v10 = route_df['v10'].values
    
    # Calculate wind speed and direction
    wind_speed = np.sqrt(u10**2 + v10**2)
    wind_direction = np.degrees(np.arctan2(-u10, -v10)) % 360
    
    # Create windrose plot
    ax = WindroseAxes.from_ax()
    ax.bar(wind_direction, wind_speed, normed=True, opening=0.8, edgecolor='white')
    ax.set_legend()
    plt.title(f'Wind Rose for Route: {output_name}')
    plt.savefig(f'figures/windrose_{output_name}.png')
    plt.show()
    
    return wind_speed, wind_direction


def get_windrose(u10, v10, lat, lon, month):
    # Calculate wind speed and direction from u10 and v10 components
    wind_speed = np.sqrt(u10**2 + v10**2)  # Wind speed magnitude
    wind_direction = np.degrees(np.arctan2(-u10, -v10)) % 360  # Wind direction (meteorological convention)

    # Create windrose plot
    ax = WindroseAxes.from_ax()
    ax.bar(wind_direction, wind_speed, normed=True, opening=0.8, edgecolor='white')
    ax.set_legend()
    plt.title(f'Wind Rose for Location: {lat}°, {lon}°')
    plt.savefig(f'figures/windrose_{lat}_{lon}_month_{month}.png')
    
    return wind_speed, wind_direction

def get_histogram(u10, v10, lat, lon, month):
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # U10 histogram with density
    ax1.hist(u10, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')
    u10_density = gaussian_kde(u10)
    u10_xs = np.linspace(u10.min(), u10.max(), 200)
    ax1.plot(u10_xs, u10_density(u10_xs), 'r-', linewidth=2, label='Density curve')
    ax1.set_xlabel('u10 (m/s)')
    ax1.set_ylabel('Density')
    ax1.set_title('Wind Component (u10)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # V10 histogram with density  
    ax2.hist(v10, bins=30, density=True, alpha=0.7, color='green', edgecolor='black')
    v10_density = gaussian_kde(v10)
    v10_xs = np.linspace(v10.min(), v10.max(), 200)
    ax2.plot(v10_xs, v10_density(v10_xs), 'r-', linewidth=2, label='Density curve')
    ax2.set_xlabel('v10 (m/s)')
    ax2.set_ylabel('Density')
    ax2.set_title('Wind Component (v10)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(f'figures/histogram_{lat}_{lon}_month_{month}.png')

def process_vels(ds, lat, lon, current_time):
    u10_all = ds.u10.sel(latitude=lat, longitude=lon, method="nearest")
    v10_all = ds.v10.sel(latitude=lat, longitude=lon, method="nearest")
    u10 = np.zeros(u10_all.shape[0])
    v10 = np.zeros(u10_all.shape[0])

    for i in tqdm(range(u10_all.shape[0])):
        u10[i] = u10_all.sel(time=current_time, method='nearest').values
        v10[i] = v10_all.sel(time=current_time, method='nearest').values
                
        current_time += pd.Timedelta(hours=1)
    return u10, v10

def main():
    parser = argparse.ArgumentParser(description='Process wind data.')
    parser.add_argument('--month', help='month')
    parser.add_argument('--lat', help='latitude')
    parser.add_argument('--lon', help='longitude')
    parser.add_argument('--route', help='longitude')
    
    args = parser.parse_args()

    print("[INFO] Reading dataset")
    all_u10 = []
    all_v10 = []
    
    for i in range(1, 13):
        current_time = pd.Timestamp(f"2020-{i}-01 00:00:00")
        ds = xr.open_dataset(f'../abdias_suez/gribs_2020/2020_{i}.grib', engine="cfgrib")
        lat = float(args.lat)
        lon = float(args.lon)
        u10, v10 = process_vels(ds, lat, lon, current_time)
        get_windrose(u10, v10, lat, lon, i)
        get_histogram(u10, v10, lat, lon, i)
        all_u10.append(u10)
        all_v10.append(v10)
    all_u10 = np.concatenate(all_u10)
    all_v10 = np.concatenate(all_v10)
    

    get_windrose_from_route(route_df, output_name="route")
    get_histogram(all_u10, all_v10, lat, lon, "all_year")
    

main()