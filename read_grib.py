import xarray as xr
import pandas as pd
import numpy as np
import folium
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def calc_dist(lat1, lon1, lat2, lon2):
    R = 6378100
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dlambda = np.radians(lon2 - lon1)
    hav = lambda angle: (1-np.cos(angle))/2
    h = hav(phi2 - phi1) + np.cos(phi1) * np.cos(phi2) * hav(dlambda)

    return R * 2 * np.arctan2(np.sqrt(h), np.sqrt(1 - h))

def calc_ang_deg(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    dlon = lon2 - lon1
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    y = np.sin(dlon) * np.cos(lat2)
    ang = np.arctan2(y, x)
    ang_deg = (np.degrees(ang) + 360) % 360  # Garante valor entre 0 e 360
    return ang_deg

def wind_dri(u, v):
    return (np.degrees(np.arctan2(u, v)) + 360) % 360

def create_map(df):
    lat = df['LAT'][:]
    lon = df['LON'][:]
    u10 = df['u10'][:] if 'u10' in df.columns else None
    v10 = df['v10'][:] if 'v10' in df.columns else None
    print(df)

    # Create a map centered around the first coordinate
    m = folium.Map(location=[lat.iloc[0], lon.iloc[0]], zoom_start=12)
    # Add the trajectory to the map
    i = 0
    for lat, lon in zip(lat, lon):
        tooltip = None

        if u10 is not None and v10 is not None:
            tooltip = f"u10: {u10.iloc[i]:.2f} m/s<br>v10: {v10.iloc[i]:.2f} m/s"
        folium.CircleMarker(location=[lat, lon], radius=1, color='blue', tooltip=tooltip).add_to(m)
        i += 1
    # Save the map to an HTML file
    m.save(f'trajectory_map_suez_v2.html')

def get_wind_properties(i, vs_ms, tempo_atual):
    lat1, lon1 = df.loc[i, "LAT"], df.loc[i, "LON"]
    lat2, lon2 = df.loc[i+1, "LAT"], df.loc[i+1, "LON"]
    diff = (df.loc[i+1, "Data"] - df.loc[i, "Data"]).total_seconds()
    dist = calc_dist(lat1, lon1, lat2, lon2)
    angulo = calc_ang_deg(lat1, lon1, lat2, lon2)

    vx_navio = vs_ms * np.sin(np.radians(angulo))
    vy_navio = vs_ms * np.cos(np.radians(angulo))
    
    dt = dist/vs_ms
    tempo_atual += pd.Timedelta(seconds=dt)
    u10_point1 = ds.u10.sel(latitude=lat1, longitude=-lon1, method="nearest")
    v10_point1 = ds.v10.sel(latitude=lat1, longitude=-lon1, method="nearest")
    u10_point2 = ds.u10.sel(latitude=lat2, longitude=lon2, method="nearest")
    v10_point2 = ds.v10.sel(latitude=lat2, longitude=lon2, method="nearest")
    u10_final = u10_point1.sel(time=tempo_atual, method='nearest').values
    v10_final = v10_point1.sel(time=tempo_atual, method='nearest').values
    ang_vento = wind_dri(u10_final, v10_final)

    # print(f"LAT1 {lat1} LON1 {lon1}")
    # print(f"LAT2 {lat2} LON2 {lon2}")
    # print("Velocidade horizontal do navio: ", vx_navio)
    # print("Velocidade vertical do navio: ", vy_navio)
    
    # # (0° = Norte, 90° = Leste, 180° = Sul, 270° = Oeste)
    # print("Angulo entre pontos (deg): ", lat1, lon1, angulo)
    # print("Tempo entre pontos (seg): ", dt)
    # print("Tempo atual: ", tempo_atual)

    # # Componente zonal (leste-oeste)
    # print("Velocidade de vento horizontal: ", u10)

    # # Componente meridional (norte-sul)
    # print("Velocidade de vento vertical: ", v10)

    # print("Angulo do vento: ", ang_vento)
    # print()
    # print("Time: ", time.process_time() - t_begin)

    return np.array([u10_final, v10_final, ang_vento])

if __name__ == "__main__":

    print("[INFO] Reading dataset")
    ds = xr.open_dataset("2024/2024.grib", engine="cfgrib")
    df = pd.read_csv("rota_suezmax.csv", sep=';')
    df["LAT"] = df["LAT"].str.replace(",", ".").astype(float)
    df["LON"] = df["LON"].str.replace(",", ".").astype(float)
    df["Data"] = pd.to_datetime(df["Data"].str.strip())

    vs_kns = 12 # Knots
    vs_ms = vs_kns / 1.94384
    tempo_atual = pd.Timestamp("2024-01-01 00:00:00")
    stamp_vento = 0
    total_range = np.arange(10000, 10200)
    

    def wrapper(i):
        return get_wind_properties(i, vs_ms, tempo_atual)
    
    t_begin = time.process_time()
    ang_vento = np.zeros(total_range.shape)
    u10 = np.zeros(total_range.shape)
    v10 = np.zeros(total_range.shape)


    parallel = False
    if not parallel:
        for j, i in enumerate(tqdm(total_range)):
            res = get_wind_properties(i, vs_ms, tempo_atual)
            u10[j] = res[0] 
            v10[j] = res[1]
            ang_vento[j] = res[2]
    else:
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(get_wind_properties, i, vs_ms, tempo_atual) for i in total_range]
            for i, future in enumerate(tqdm(futures)):
                res = future.result()
                u10[i] = res[0]
                v10[i] = res[1]
                ang_vento[i] = res[2]
                
    new_df = df.loc[total_range]
    new_df['u10'] = u10
    new_df['v10'] = u10
    new_df['angulo_vento'] = ang_vento
    print("[INFO] Ploting Map..")  
    create_map(new_df)
    
    print(f"[INFO] Saving informations with start time: {tempo_atual} ..")
    new_df.to_csv(f'infos_hour{tempo_atual.hour}_min{tempo_atual.minute}_sec{tempo_atual.hour}.csv')