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
    def __init__(self, timestamp, route_path, wind_path, forces_path):
        self.timestamp = timestamp
        self.route_path = route_path
        self.wind_path = wind_path
        self.forces_path = forces_path
        self.folder = os.path.basename(self.route_path).split('.')[0]
        
        self._lock = threading.Lock()  # Add thread lock
        self.lat = 0
        self.lon = 0
        vs_kns = 12 # Knots
        self.vs_ms = vs_kns / 1.94384
        self.new_df = None
        self.df = None
        self.current_month = None
        self.ds = None  
        
    def load_data(self):
        
        print("[INFO] Loading wind data")
        self.ds = xr.open_dataset(self.wind_path, engine="cfgrib")
        if self.current_month != self.timestamp.month or self.ds is None:
            self.current_month = self.timestamp.month
            # Construct the path for the current month's GRIB file
            month_wind_path = f"{os.path.dirname(self.wind_path)}/2020_{int(self.current_month)}.grib"  # Adjust path format as needed
            self.ds = xr.open_dataset(month_wind_path, engine="cfgrib")
            print(f"[INFO] Loaded wind data for month {self.current_month}")
        

        if not self.df:
            print("[INFO] Loading route data")
            self.df = pd.read_csv(self.route_path, sep=';')

        # self.df["LAT"] = self.df["LAT"].str.replace(",", ".").astype(float)
        # self.df["LON"] = self.df["LON"].str.replace(",", ".").astype(float)
        self.df["Data"] = pd.to_datetime(self.df["Data"].str.strip())
        self.wind_u = None
        self.wind_v = None

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

    def wind_dri(self, u10, v10):
        return np.degrees(np.arctan2(u10, v10)) % 360

    def create_map(self):
        if np.any(self.new_df) == None:
            self.new_df = self.df.copy()
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

        self.dt = dist/self.vs_ms

        if self.current_month != self.timestamp.month:
            self.current_month = self.timestamp.month
            month_wind_path = f"{os.path.dirname(self.wind_path)}/2020_{int(self.current_month)}.grib"  # Adjust path format as needed
            self.ds = xr.open_dataset(month_wind_path, engine="cfgrib")
            print(f"[INFO] Loaded new wind data for month {self.current_month}")
        
        u10_point = self.ds.u10.sel(latitude=lat1, longitude=lon1, method="nearest")
        v10_point = self.ds.v10.sel(latitude=lat1, longitude=lon1, method="nearest")
        
        try:
            u10 = u10_point.sel(time=self.timestamp, method='nearest').values
        except:
            print("[ERROR] u10 not avilable")
            u10 = 0
        try:
            v10 = v10_point.sel(time=self.timestamp, method='nearest').values
        except:
            print("[ERROR] v10 not avilable")
            v10 = 0
        ang_vento = self.wind_dri(u10, v10)

        return np.array([u10, v10, ang_navio, ang_vento])

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
        total_range = np.arange(len(self.df)-1)
        
        ang_ship = np.zeros(total_range.shape)
        ang_vento = np.zeros(total_range.shape)
        u10 = np.zeros(total_range.shape)
        v10 = np.zeros(total_range.shape)
        times = np.zeros(total_range.shape).astype(str)
        pbar = tqdm(total_range)
        for i in pbar:
            pbar.set_description(f"[INFO] From time: {self.timestamp}")
            res = self.get_wind_properties(i)
            u10[i] = res[0] 
            v10[i] = res[1]
            ang_ship[i] = res[2]
            ang_vento[i] = res[3]
            times[i] = self.timestamp.strftime('%Y-%m-%d %X')
            self.timestamp += pd.Timedelta(seconds=self.dt)

        self.new_df = self.df.loc[total_range, ['LAT', 'LON']].copy()
        self.new_df['u10'] = u10
        self.new_df['v10'] = v10
        self.new_df['angle_ship'] = ang_ship
        self.new_df['angle_wind'] = ang_vento
        self.new_df['time'] = times

        return self.new_df

    def calculate_relative_velocity_and_angle(self):
        
        self.new_df['u_ship'] = self.vs_ms * np.sin(np.radians(self.new_df['angle_ship']))
        self.new_df['v_ship'] = self.vs_ms * np.cos(np.radians(self.new_df['angle_ship']))

        self.new_df['u_rel'] = self.new_df['u10'] - self.new_df['u_ship']
        self.new_df['v_rel'] = self.new_df['v10'] - self.new_df['v_ship']
        self.new_df['angle_rel'] = self.wind_dri(self.new_df['u_rel'], self.new_df['v_rel'])

    def process_per_route(self):
        csv_path = f'../{self.folder}/route_wind_data_csvs/wind_data_year_{self.timestamp.year}_month_{self.timestamp.month}_day_{self.timestamp.day}_hour_{self.timestamp.hour}.csv'
        
        if os.path.exists(csv_path):
            self.new_df = pd.read_csv(csv_path, sep=',')
        else:
            self.process_dataframe()

        self.calculate_relative_velocity_and_angle()

    def save_csv(self, timestamp):
        csv_path = f'../{self.folder}/route_wind_data_csvs/wind_data_year_{timestamp.year}_month_{timestamp.month}_day_{timestamp.day}_hour_{timestamp.hour}.csv'
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        if not os.path.exists(csv_path):
            print(f"[INFO] Saving informations with start time: {timestamp} ..")
            self.new_df.to_csv(csv_path)
        

        
    def save_map(self, timestamp):
        map_path = f'../{self.folder}/route_maps/'
        os.makedirs(os.path.dirname(map_path), exist_ok=True)
        if not os.path.exists(os.path.join(map_path, f"trajectory_map_year_{timestamp.year}_month_{timestamp.month}_day_{timestamp.day}_hour_{timestamp.hour}.html")):
            print("[INFO] Ploting Map..")  
            self.create_map()
            self.m.save(os.path.join(map_path, f"trajectory_map_year_{timestamp.year}_month_{timestamp.month}_day_{timestamp.day}_hour_{timestamp.hour}.html"))

    def load_forces(self):
        self.forces_df = pd.read_csv(self.forces_path)
        import pdb;pdb.set_trace()


if __name__ == "__main__":

    current_time = pd.Timestamp("2020-03-16 03:00:00")
    for i in range(8000):
        print("Time starts: ", current_time)
        map_processer = ProcessMap(timestamp=current_time, route_path='../castro_alves_afra/ais/castro_alves_afra.csv', wind_path='../castro_alves_afra/gribs_2020/2020_1.grib', forces_path='forces.csv')
        map_processer.load_data()

        map_processer.process_per_route()
        map_processer.save_csv(current_time)
        if i%100 == 0:
            map_processer.save_map(current_time)
        # map_processer.load_forces()
        current_time += pd.Timedelta(hours=1)