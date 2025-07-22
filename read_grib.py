import xarray as xr
import pandas as pd
import numpy as np
import os
import folium
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

class ProcessMap:
    def __init__(self, timestamp, route_path, wind_path):
        self.timestamp = timestamp
        self.route_path = route_path
        self.wind_path = wind_path

    def load_data(self):
        
        print("[INFO] Reading dataset")
        self.ds = xr.open_dataset(self.wind_path, engine="cfgrib")
        self.df = pd.read_csv(self.route_path, sep=';')

        self.df["LAT"] = self.df["LAT"].str.replace(",", ".").astype(float)
        self.df["LON"] = self.df["LON"].str.replace(",", ".").astype(float)
        self.df["Data"] = pd.to_datetime(self.df["Data"].str.strip())

    def haversine(self, lat1, lon1, lat2, lon2):
        R = 6371e3
        phi1 = lat1 * np.pi/180
        phi2 = lat2 * np.pi/180
        del_phi = (lat2-lat1) * np.pi/180
        del_lambda = (lon2-lon1) * np.pi/180
        
        a = np.power(np.sin(del_phi/2), 2) + np.cos(phi1) * np.cos(phi2) * np.power(np.sin(del_lambda/2), 2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        d = R * c
        return d

    def bearing(self, lat1, lon1, lat2, lon2):
        phi1 = lat1 * np.pi/180
        phi2 = lat2 * np.pi/180
        lambda1 = lon1 * np.pi/180
        lambda2 = lon2 * np.pi/180
        y = np.sin(lambda2 - lambda1) * np.cos(phi2)
        x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(lambda2 - lambda1)
        theta = np.arctan2(y, x)
        brng = (theta*180/np.pi + 360) % 360
        return brng

    def wind_dri(self, u, v):
        return (np.degrees(np.arctan2(u, v)) + 360) % 360

    @staticmethod
    def create_map(filtered_df):
        lat = filtered_df['LAT'][:]
        lon = filtered_df['LON'][:]
        u10 = filtered_df['u10'][:] if 'u10' in filtered_df.columns else None
        v10 = filtered_df['v10'][:] if 'v10' in filtered_df.columns else None
        angle = filtered_df['angulo_vento'][:] if 'angulo_vento' in filtered_df.columns else None

        # Create a map centered around the first coordinate
        m = folium.Map(location=[lat.iloc[0], lon.iloc[0]], zoom_start=12)
        # Add the trajectory to the map
        i = 0
        for lat, lon in zip(lat, lon):
            tooltip = None

            if u10 is not None and v10 is not None:
                tooltip = f"lat: {lat}<br>lon: {lon}<br>u10: {u10.iloc[i]:.2f} m/s<br>v10: {v10.iloc[i]:.2f} m/s<br>angle: {angle.iloc[i]:.2f} deg"
            folium.CircleMarker(location=[lat, lon], radius=1, color='blue', tooltip=tooltip).add_to(m)
            i += 1
        # Save the map to an HTML file
        m.save(f'trajectory_map_suez_v2.html')

    def get_wind_properties(self, i, vs_ms, tempo_atual):
        lat1, lon1 = self.df.loc[i, "LAT"], self.df.loc[i, "LON"]
        lat2, lon2 = self.df.loc[i+1, "LAT"], self.df.loc[i+1, "LON"]
        diff = (self.df.loc[i+1, "Data"] - self.df.loc[i, "Data"]).total_seconds()
        dist = self.haversine(lat1, lon1, lat2, lon2)
        ang_navio = self.bearing(lat1, lon1, lat2, lon2)
    
        vx_navio = vs_ms * np.sin(np.radians(ang_navio))
        vy_navio = vs_ms * np.cos(np.radians(ang_navio))
        
        dt = dist/vs_ms
        tempo_atual += pd.Timedelta(seconds=dt)
        u10_point1 = self.ds.u10.sel(latitude=lat1, longitude=-lon1, method="nearest")
        v10_point1 = self.ds.v10.sel(latitude=lat1, longitude=-lon1, method="nearest")
        # u10_point2 = self.ds.u10.sel(latitude=lat2, longitude=lon2, method="nearest")
        # v10_point2 = self.ds.v10.sel(latitude=lat2, longitude=lon2, method="nearest")
        u10_final = u10_point1.sel(time=tempo_atual, method='nearest').values
        v10_final = v10_point1.sel(time=tempo_atual, method='nearest').values
        ang_vento = self.wind_dri(u10_final, v10_final)

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

        return np.array([u10_final, v10_final, ang_navio])

    def process_dataframe(self):
        vs_kns = 12 # Knots
        vs_ms = vs_kns / 1.94384
        stamp_vento = 0
        total_range = np.arange(10000, 10400)
        
        t_begin = time.process_time()
        ang_vento = np.zeros(total_range.shape)
        u10 = np.zeros(total_range.shape)
        v10 = np.zeros(total_range.shape)

        parallel = False
        if not parallel:
            for j, i in enumerate(tqdm(total_range)):
                res = self.get_wind_properties(i, vs_ms, self.timestamp)
                u10[j] = res[0] 
                v10[j] = res[1]
                ang_vento[j] = res[2]
        else:
            with ThreadPoolExecutor(max_workers=32) as executor:
                futures = [executor.submit(self.get_wind_properties, i, vs_ms, self.timestamp) for i in total_range]
                for i, future in enumerate(tqdm(futures)):
                    res = future.result()
                    u10[i] = res[0]
                    v10[i] = res[1]
                    ang_vento[i] = res[2]
                    
        self.filtered_df = self.df.loc[total_range]
        self.filtered_df['u10'] = u10
        self.filtered_df['v10'] = u10
        self.filtered_df['angulo_vento'] = ang_vento

        return self.filtered_df

if __name__ == "__main__":

    tempo_atual = pd.Timestamp("2024-01-01 00:00:00")
    map_processer = ProcessMap(timestamp=tempo_atual, route_path='rota_suezmax.csv', wind_path='2024/2024.grib')
    map_processer.load_data()

    if os.path.exists(f'infos_hour{tempo_atual.hour}_min{tempo_atual.minute}_sec{tempo_atual.hour}.csv'):
        new_df = pd.read_csv(f'infos_hour{tempo_atual.hour}_min{tempo_atual.minute}_sec{tempo_atual.hour}.csv', sep=',')
    else:
        new_df = map_processer.process_dataframe()

    print("[INFO] Ploting Map..")  
    map_processer.create_map(new_df)
    
    print(f"[INFO] Saving informations with start time: {tempo_atual} ..")
    new_df.to_csv(f'infos_hour{tempo_atual.hour}_min{tempo_atual.minute}_sec{tempo_atual.hour}.csv')