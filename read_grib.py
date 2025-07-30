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
        vs_kns = 12 # Knots
        self.vs_ms = vs_kns / 1.94384
        

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

    def create_map(self):
        lat = self.new_df['LAT'][:]
        lon = self.new_df['LON'][:]
        u10 = self.new_df['u10'][:] if 'u10' in self.new_df.columns else None
        v10 = self.new_df['v10'][:] if 'v10' in self.new_df.columns else None
        u_ship = self.new_df['u_ship'][:] if 'u_ship' in self.new_df.columns else None
        v_ship = self.new_df['v_ship'][:] if 'v_ship' in self.new_df.columns else None
        u_rel = self.new_df['u_rel'][:] if 'u_rel' in self.new_df.columns else None
        v_rel = self.new_df['v_rel'][:] if 'v_rel' in self.new_df.columns else None

        angle_wind = self.new_df['angle_wind'][:] if 'angle_wind' in self.new_df.columns else None
        angle_ship = self.new_df['angle_ship'][:] if 'angle_ship' in self.new_df.columns else None
        angle_rel = self.new_df['angle_rel'][:] if 'angle_ship' in self.new_df.columns else None
        times = self.new_df['time'][:] if 'time' in self.new_df.columns else None

        # Create a map centered around the first coordinate
        self.m = folium.Map(location=[lat.iloc[0], lon.iloc[0]], zoom_start=12)
        # Add the trajectory to the map
        i = 0
        for lat, lon in zip(lat, lon):
            tooltip = None

            if u10 is not None and v10 is not None:
        
                tooltip = (f"Time: {times.iloc[i]}<br>"
                    f"Lat: {lat}<br>"
                    f"Lon: {lon}<br>"
                    f"u_wind: {u10.iloc[i]:.2f} m/s<br>"
                    f"v_wind: {v10.iloc[i]:.2f} m/s<br>"
                    f"u_ship: {u_ship.iloc[i]:.2f} m/s<br>"
                    f"v_ship: {v_ship.iloc[i]:.2f} m/s<br>"
                    f"u_rel: {u_rel.iloc[i]:.2f} m/s<br>"
                    f"v_rel: {v_rel.iloc[i]:.2f} m/s<br>"
                    f"angle ship: {angle_ship.iloc[i]:.2f} deg<br>"
                    f"angle wind: {angle_wind.iloc[i]:.2f} deg<br>"
                    f"angle rel: {angle_rel.iloc[i]:.2f} deg")
                
            folium.CircleMarker(location=[lat, lon], radius=1, color='blue', tooltip=tooltip).add_to(self.m)
            i += 1
        # Save the map to an HTML file

        
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
        
    def get_wind_properties(self, i):
        lat1, lon1 = self.df.loc[i, "LAT"], self.df.loc[i, "LON"]
        lat2, lon2 = self.df.loc[i+1, "LAT"], self.df.loc[i+1, "LON"]
        
        dist = self.haversine(lat1, lon1, lat2, lon2)
        ang_navio = self.bearing(lat1, lon1, lat2, lon2)

        
        dt = dist/self.vs_ms
        self.timestamp += pd.Timedelta(seconds=dt)
        u10_point1 = self.ds.u10.sel(latitude=lat1, longitude=lon1, method="nearest")
        v10_point1 = self.ds.v10.sel(latitude=lat1, longitude=lon1, method="nearest")
        # u10_point2 = self.ds.u10.sel(latitude=lat2, longitude=lon2, method="nearest")
        # v10_point2 = self.ds.v10.sel(latitude=lat2, longitude=lon2, method="nearest")

        try:
            u10_final = u10_point1.sel(time=self.timestamp, method='nearest').values
        except:
            print("[ERROR] u10 not avilable")
            u10_final = 0
        try:
            v10_final = v10_point1.sel(time=self.timestamp, method='nearest').values
        except:
            print("[ERROR] v10 not avilable")
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
        total_range = np.arange(0, 24000)
        
        ang_ship = np.zeros(total_range.shape)
        ang_vento = np.zeros(total_range.shape)
        u10 = np.zeros(total_range.shape)
        v10 = np.zeros(total_range.shape)
        times = np.zeros(total_range.shape).astype(str)

        for i in tqdm(total_range):
            res = self.get_wind_properties(i)
            u10[i] = res[0] 
            v10[i] = res[1]
            ang_ship[i] = res[2]
            ang_vento[i] = res[3]
            times[i] = self.timestamp.strftime('%Y-%m-%d %X')

        filtered_df = self.df.loc[total_range]
        filtered_df['u10'] = u10
        filtered_df['v10'] = v10
        filtered_df['angle_ship'] = ang_ship
        filtered_df['angle_wind'] = ang_vento
        filtered_df['time'] = times

        return filtered_df

    def calculate_relative_velocity_and_angle(self):
        
        self.new_df['u_ship'] = -self.vs_ms * np.sin(np.radians(self.new_df['angle_ship']))
        self.new_df['v_ship'] = self.vs_ms * np.cos(np.radians(self.new_df['angle_ship']))

        self.new_df['u_rel'] = self.new_df['u_ship'] + self.new_df['u10']
        self.new_df['v_rel'] = self.new_df['v_ship'] + self.new_df['v10']
        self.new_df['angle_rel'] = np.degrees(np.arctan2(self.new_df['v_rel'], -self.new_df['u_rel']))

    def process_per_route(self):
        
        if os.path.exists(f'wind_data_year_{self.timestamp.year}_month_{self.timestamp.month}_day_{self.timestamp.day}_hour_{self.timestamp.hour}.csv'):
            self.new_df = pd.read_csv(f'wind_data_year_{self.timestamp.year}_month_{self.timestamp.month}_day_{self.timestamp.day}_hour_{self.timestamp.hour}.csv', sep=',')
        else:
            self.new_df = self.process_dataframe()

        self.calculate_relative_velocity_and_angle()

    def save_csv(self, timestamp):
        
        print(f"[INFO] Saving informations with start time: {timestamp} ..")
        self.new_df.to_csv(f'wind_data_year_{timestamp.year}_month_{timestamp.month}_day_{timestamp.day}_hour_{timestamp.hour}.csv')

        
    def save_map(self, timestamp):
        print("[INFO] Ploting Map..")  

        self.create_map()
        self.m.save(f'trajectory_map_year_{timestamp.year}_month_{timestamp.month}_day_{timestamp.day}_hour_{timestamp.hour}.html')


if __name__ == "__main__":

    current_time = pd.Timestamp("2020-01-01 00:00:00")
    for i in range(100):
        print("Time starts: ", current_time)
        map_processer = ProcessMap(timestamp=current_time, route_path='rota_suezmax.csv', wind_path='2020/2020.grib')
        map_processer.load_data()

        map_processer.process_per_route()
        map_processer.save_csv(current_time)
        map_processer.save_map(current_time)

        current_time += pd.Timedelta(hours=1)
        break