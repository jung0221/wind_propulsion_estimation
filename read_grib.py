import xarray as xr
import pandas as pd
import numpy as np
import os
import folium
from tqdm import tqdm
import concurrent
import time
import threading

class ProcessMap:
    def __init__(self, timestamp, route_path, wind_path):
        self.timestamp = timestamp
        self.route_path = route_path
        self.wind_path = wind_path
        self._lock = threading.Lock()  # Add thread lock
        self.lat = 0
        self.lon = 0
        

    def load_data(self):
        
        print("[INFO] Reading dataset")
        self.ds = xr.open_dataset(self.wind_path, engine="cfgrib")
        self.df = pd.read_csv(self.route_path, sep=';')

        self.df["LAT"] = self.df["LAT"].str.replace(",", ".").astype(float)
        self.df["LON"] = self.df["LON"].str.replace(",", ".").astype(float)
        self.df["Data"] = pd.to_datetime(self.df["Data"].str.strip())
        self.wind_u = np.zeros(8784)
        self.wind_v = np.zeros(8784)

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
        return (np.degrees(np.arctan2(-u, -v)) + 360) % 360

    def create_map(self, filtered_df):
        lat = filtered_df['LAT'][:]
        lon = filtered_df['LON'][:]
        u10 = filtered_df['u10'][:] if 'u10' in filtered_df.columns else None
        v10 = filtered_df['v10'][:] if 'v10' in filtered_df.columns else None
        angle_wind = filtered_df['angle_wind'][:] if 'angle_wind' in filtered_df.columns else None
        angle_ship = filtered_df['angle_ship'][:] if 'angle_ship' in filtered_df.columns else None

        # Create a map centered around the first coordinate
        m = folium.Map(location=[lat.iloc[0], lon.iloc[0]], zoom_start=12)
        # Add the trajectory to the map
        i = 0
        for lat, lon in zip(lat, lon):
            tooltip = None

            if u10 is not None and v10 is not None:
                tooltip = f"lat: {lat}<br>lon: {lon}<br>u10: {u10.iloc[i]:.2f} m/s<br>v10: {v10.iloc[i]:.2f} m/s<br>angle ship: {angle_ship.iloc[i]:.2f} deg<br>angle wind: {angle_wind.iloc[i]:.2f} deg"
            folium.CircleMarker(location=[lat, lon], radius=1, color='blue', tooltip=tooltip).add_to(m)
            i += 1
        # Save the map to an HTML file
        m.save(f'trajectory_map_{self.timestamp.year}_{self.timestamp.month}_{self.timestamp.day}_hour{self.timestamp.hour}.csv.html')
        
    def get_nearest_index(self, array, value):
        # Check if array contains datetime values
        if np.issubdtype(array.dtype, np.datetime64):
            # Convert value to numpy datetime64 if it's a pandas Timestamp
            if hasattr(value, 'to_numpy'):
                value = value.to_numpy()
            elif isinstance(value, (int, float)):
                value = np.datetime64(value, 'ns')
            # Calculate time differences and find minimum
            return np.argmin(np.abs((array - value).astype('timedelta64[ns]')))
        else:
            # For non-datetime arrays, use the original method
            return np.abs(array - value).argmin()
    def get_wind_properties(self, i, vs_ms):
        lat1, lon1 = self.df.loc[i, "LAT"], self.df.loc[i, "LON"]
        lat2, lon2 = self.df.loc[i+1, "LAT"], self.df.loc[i+1, "LON"]
        diff = (self.df.loc[i+1, "Data"] - self.df.loc[i, "Data"]).total_seconds()
        dist = self.haversine(lat1, lon1, lat2, lon2)
        ang_navio = self.bearing(lat1, lon1, lat2, lon2)

        vx_navio = vs_ms * np.sin(np.radians(ang_navio))
        vy_navio = vs_ms * np.cos(np.radians(ang_navio))
        
        dt = dist/vs_ms
        self.timestamp += pd.Timedelta(seconds=dt)

        if i % 7000 == 0: 
            self.ds = xr.open_dataset(self.wind_path, engine="cfgrib")

        u10_point1 = self.ds.u10.sel(latitude=lat1, longitude=lon1, method="nearest")
        v10_point1 = self.ds.v10.sel(latitude=lat1, longitude=lon1, method="nearest")
        # u10_point2 = self.ds.u10.sel(latitude=lat2, longitude=lon2, method="nearest")
        # v10_point2 = self.ds.v10.sel(latitude=lat2, longitude=lon2, method="nearest")
        try:
            u10_final = u10_point1.sel(time=self.timestamp, method='nearest').values
        except:
            u10_final = 0
        try:
            v10_final = v10_point1.sel(time=self.timestamp, method='nearest').values
        except:
            v10_final = 0
        ang_vento = self.wind_dri(u10_final, v10_final)
        return np.array([u10_final, v10_final, ang_navio, ang_vento])

    def save_wind_speeds(self, i):
        if np.round(self.df.loc[i, "LAT"]) == np.round(self.lat) and np.round(self.df.loc[i, "LON"]) == np.round(self.lon):
            self.lat, self.lon = self.df.loc[i, "LAT"], self.df.loc[i, "LON"]
            return self.wind_u, self.wind_v, self.lat, self.lon
        self.lat, self.lon = self.df.loc[i, "LAT"], self.df.loc[i, "LON"]
        
        u10_point = self.ds.u10.sel(latitude=self.lat, longitude=self.lon, method="nearest")
        v10_point = self.ds.v10.sel(latitude=self.lat, longitude=self.lon, method="nearest")
        self.wind_u = np.zeros(8784)
        self.wind_v = np.zeros(8784)
        
        for j in tqdm(range(8784)):
            try: 
                self.wind_u[j] = u10_point.sel(time=self.timestamp, method='nearest').values
            except: 
                pass
            try:
                self.wind_v[j] = v10_point.sel(time=self.timestamp, method='nearest').values
            except:
                pass

            self.timestamp += pd.Timedelta(hours=1)
        print()
        return self.wind_u, self.wind_v, self.lat, self.lon


    def process_dataframe(self):
        vs_kns = 12 # Knots
        vs_ms = vs_kns / 1.94384
        stamp_vento = 0
        total_range = np.arange(0, 24000)
        
        t_begin = time.process_time()
        ang_ship = np.zeros(total_range.shape)
        ang_vento = np.zeros(total_range.shape)
        u10 = np.zeros(total_range.shape)
        v10 = np.zeros(total_range.shape)

        parallel = False
        if not parallel:
            for j, i in enumerate(tqdm(total_range)):
                res = self.get_wind_properties(i, vs_ms)
                u10[j] = res[0] 
                v10[j] = res[1]
                ang_ship[j] = res[2]
                ang_vento[j] = res[3]
        else:
            def wrapper(i):
                return self.get_wind_properties(i, vs_ms)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
                results = list(tqdm(
                    executor.map(wrapper, total_range),
                    total=len(total_range),
                    desc="Processing in parallel"
                ))
                for j, res in enumerate(results):
                    u10[j] = res[0]
                    v10[j] = res[1]
                    ang_ship[j] = res[2]
                    ang_vento[j] = res[3]
                    
        filtered_df = self.df.loc[total_range]
        filtered_df['u10'] = u10
        filtered_df['v10'] = v10
        filtered_df['angle_ship'] = ang_ship
        filtered_df['angle_wind'] = ang_vento

        return filtered_df

if __name__ == "__main__":

    current_time = pd.Timestamp("2024-01-01 00:00:00")
    # Initialize lists to store all wind data
    all_wind_x = []
    all_wind_y = []
    all_positions = []
    
    map_processer = ProcessMap(timestamp=current_time, route_path='rota_suezmax.csv', wind_path='2024/2024.grib')
    map_processer.load_data()

    for i in tqdm(range(23000), desc="Processing positions"):

        if os.path.exists(f'infos_hour{current_time.hour}_min{current_time.minute}_sec{current_time.hour}.csv'):
            new_df = pd.read_csv(f'infos_hour{current_time.hour}_min{current_time.minute}_sec{current_time.hour}.csv', sep=',')
        else:
            # new_df = map_processer.process_dataframe()

            wind_x, wind_y, lat, lon = map_processer.save_wind_speeds(i)
            all_wind_x.append(wind_x)
            all_wind_y.append(wind_y)
            all_positions.append({'position_index': i, 'LAT': lat, 'LON': lon})

            # position_df = pd.DataFrame({
            #     'time_index': range(8784),
            #     'timestamp': [current_time + pd.Timedelta(hours=h) for h in range(8784)],
            #     'position_index': i,
            #     'LAT': lat,
            #     'LON': lon,
            #     'wind_x': wind_x,
            #     'wind_y': wind_y
            # })
            
            # Save individual position DataFrame
            # position_df.to_csv(f'./position_winds/wind_speeds_lat{lat}_lon{lon}', index=False)
            # print(f"[INFO] Saved position {i} data to {f'./position_winds/wind_speeds_lat{lat}_lon{lon}'}")
    all_wind_x = np.array(all_wind_x) 
    all_wind_y = np.array(all_wind_y) 
    
    positions_df = pd.DataFrame(all_positions)
    positions_df['wind_x_timeseries'] = list(all_wind_x)
    positions_df['wind_y_timeseries'] = list(all_wind_y)
    positions_df.to_csv('wind_data_by_position.csv', index=False)
    print(f"[INFO] Saved wind data for {len(positions_df)} positions to 'wind_data_by_position.csv'")
    


        # print("[INFO] Ploting Map..")  
        # map_processer.create_map(new_df)
        
        # print(f"[INFO] Saving informations with start time: {current_time} ..")
        # new_df.to_csv(f'wind_data_{current_time.year}_{current_time.month}_{current_time.day}_hour{current_time.hour}.csv')
        # current_time += pd.Timedelta(hours=1)