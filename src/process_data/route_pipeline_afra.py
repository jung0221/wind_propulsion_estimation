import xarray as xr
import pandas as pd
import numpy as np
import os
import gc
from tqdm import tqdm
import argparse
import threading
import bisect


class ProcessMap:
    _ds_path = None

    def __init__(
        self,
        timestamp,
        route_path,
        wind_path,
        forces_path,
        old_forces_path,
        ship,
        rotation,
        calculate_forces=True,
    ):
        self.calculate_forces = calculate_forces
        self.timestamp = timestamp
        self.route_path = route_path
        self.wind_path = wind_path
        self.ship = ship
        self.rotation = int(rotation)
        self.folder = os.path.dirname(self.route_path).split("/")[1]
        self.forces_path = forces_path
        self.old_forces_path = old_forces_path
        self._lock = threading.Lock()  # Add thread lock
        self.lat = 0
        self.lon = 0
        self.vs_kns = 12 if self.ship == "suez" else 14  # Knots
        self.vs_ms = self.vs_kns / 1.94384
        self.new_df = None
        self.df = None
        self.current_month = None
        self.RT = None

        self.ds = None
        self.df_forces = None
        self.draft = None
        self.Ax = 175
        self.Ay = 3300
        R_T = None
        self.eta_d = 0.7
        self.eta_rot = 0.9

    def load_data(self):

        print(
            f"[INFO] Loading wind data for ship {self.ship} with speed {self.vs_kns} knots"
        )
        self.ds = xr.open_dataset(self.wind_path, engine="cfgrib")
        # Only open dataset when needed (cache per-instance). Close previous ds if month changes.
        month_wind_path = f"{os.path.dirname(self.wind_path)}/{self.timestamp.year}_{int(self.timestamp.month)}.grib"
        if self.ds is None or ProcessMap._ds_path != month_wind_path:
            # close existing ds to avoid leaks
            try:
                if self.ds is not None:
                    self.ds.close()
            except Exception:
                pass
            try:
                self.ds = xr.open_dataset(month_wind_path, engine="cfgrib")
                ProcessMap._ds_path = month_wind_path
                self.current_month = self.timestamp.month
                print(
                    f"[INFO] Loaded wind data for month {self.current_month} from {month_wind_path}"
                )
            except Exception as e:
                print(f"[ERROR] Opening wind file {month_wind_path}: {e}")
                self.ds = None
                ProcessMap._ds_path = None
        if self.df is None:
            print("[INFO] Loading route data")
            df_original = pd.read_csv(self.route_path, sep=";")
            # Criar DataFrame com ida e volta
            df_ida = df_original.copy()
            df_volta = df_original.iloc[::-1].reset_index(drop=True)  # Inverte a ordem
            # Adicionar coluna indicando direção da viagem
            df_ida["direction"] = "outbound"
            df_volta["direction"] = "return"

            # Concatenar ida e volta
            self.df = pd.concat([df_ida, df_volta], ignore_index=True)

        if self.df_forces is None:
            try:
                print("[INFO] Loading thrust data")
                self.df_forces = pd.read_csv(self.forces_path)
                self.old_df_forces = pd.read_csv(self.old_forces_path)
                self.angle_list = self.df_forces["Angulo"].unique()
                self.angle_list = np.append(self.angle_list, 360)
            except FileNotFoundError as e:
                print(f"[ERROR] {e}")
                print("[WARNING] Continuing without return data")
            except pd.errors.EmptyDataError:
                print(f"[ERROR] Data file is empty")
            except Exception as e:
                print(f"[ERROR] Unexpected error loading return data: {e}")
        self.wind_u = None
        self.wind_v = None

    def haversine(self, lat1, lon1, lat2, lon2):
        R = 6371e3
        phi1 = lat1 * np.pi / 180
        phi2 = lat2 * np.pi / 180
        del_phi = (lat2 - lat1) * np.pi / 180
        del_lambda = (lon2 - lon1) * np.pi / 180

        a = np.power(np.sin(del_phi / 2), 2) + np.cos(phi1) * np.cos(phi2) * np.power(
            np.sin(del_lambda / 2), 2
        )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        d = R * c
        return d

    def bearing(self, lat1, lon1, lat2, lon2):
        phi1 = lat1 * np.pi / 180
        phi2 = lat2 * np.pi / 180
        lambda1 = lon1 * np.pi / 180
        lambda2 = lon2 * np.pi / 180
        y = np.sin(lambda2 - lambda1) * np.cos(phi2)
        x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(
            lambda2 - lambda1
        )
        theta = np.arctan2(y, x)
        brng = (theta * 180 / np.pi + 360) % 360
        return brng

    def wind_dri(self, u10, v10):
        return np.degrees(np.arctan2(-u10, -v10)) % 360

    def wind_rel(self, u, v):
        return np.degrees(np.arctan2(v, u)) % 360

    def extrapolate_v_wind(self, vel, height):
        return vel * np.power(height / 10, 1 / 9)

    def get_nearest_index(self, array, value):
        # Check if array contains datetime values
        if np.issubdtype(array.dtype, np.datetime64):
            # Convert value to numpy datetime64 if it's a pandas Timestamp
            if hasattr(value, "to_numpy"):
                value = value.to_numpy()
            elif isinstance(value, (int, float)):
                value = np.datetime64(value, "ns")
            # Calculate time differences and find minimum
            return np.argmin(np.abs((array - value).astype("timedelta64[ns]")))
        else:
            # For non-datetime arrays, use the original method
            return np.abs(array - value).argmin()

    def get_wind_properties(self, i, z_height):
        lat1, lon1 = self.df.loc[i, "LAT"], self.df.loc[i, "LON"]
        lat2, lon2 = self.df.loc[i + 1, "LAT"], self.df.loc[i + 1, "LON"]

        dist = self.haversine(lat1, lon1, lat2, lon2)
        ang_navio = self.bearing(lat1, lon1, lat2, lon2)
        self.dt = dist / self.vs_ms
        if self.current_month != self.timestamp.month:
            self.current_month = self.timestamp.month
            month_wind_path = f"{os.path.dirname(self.wind_path)}/{int(self.timestamp.year)}_{int(self.current_month)}.grib"  # Adjust path format as needed
            self.ds = xr.open_dataset(month_wind_path, engine="cfgrib")
            print(f"[INFO] Loaded new wind data for month {self.current_month}")

        u10_point = self.ds.u10.sel(latitude=lat1, longitude=lon1, method="nearest")
        v10_point = self.ds.v10.sel(latitude=lat1, longitude=lon1, method="nearest")

        try:
            u10 = u10_point.sel(time=self.timestamp, method="nearest").values
        except:
            print("[ERROR] u10 not avilable")
            u10 = 0
        try:
            v10 = v10_point.sel(time=self.timestamp, method="nearest").values
        except:
            print("[ERROR] v10 not avilable")
            v10 = 0
        u10 = self.extrapolate_v_wind(u10, z_height)
        v10 = self.extrapolate_v_wind(v10, z_height)
        wind_angle = self.wind_dri(u10, v10)

        return [u10, v10, ang_navio, wind_angle]

    def get_adjacent_angles(self, ang):

        pos = bisect.bisect_left(self.angle_list, ang)

        # Get floor and ceil
        if pos == 0:
            floor_angle = self.angle_list[0]
            ceil_angle = self.angle_list[0]
        elif pos == len(self.angle_list):
            floor_angle = self.angle_list[-1]
            ceil_angle = self.angle_list[-1]
        else:
            floor_angle = self.angle_list[pos - 1]
            ceil_angle = self.angle_list[pos]
        return floor_angle, ceil_angle

    def get_forces(self, ang, vel, force="fx", draft=None, index=0):

        spin_ratio = (self.rotation * np.pi / 30) * 2.5 / vel

        ang = float(ang) % 360.0
        # use provided draft if passed, otherwise use self.draft
        draft_local = self.draft if draft is None else draft
        floor_angle, ceil_angle = self.get_adjacent_angles(ang)

        # For angular interpolation treat ceil=0 as 360 (for distance calc) but selection uses 0
        ang_floor = float(floor_angle)
        ang_ceil_for_select = (
            0
            if float(ceil_angle) == 360 or float(ceil_angle) == 0
            else float(ceil_angle)
        )
        ang_ceil_for_interp = float(ceil_angle) if float(ceil_angle) != 0 else 360.0

        def get_sorted_rows(angle_sel, force_df):
            sel = force_df[
                (force_df["Angulo"] == angle_sel) & (force_df["Calado"] == draft_local)
            ]
            if sel.empty:
                raise ValueError(
                    f"No data for angle={angle_sel}, draft={draft_local}, rot={self.rotation}"
                )
            sel = sel.sort_values("Vw")
            return sel

        if ang > 120 or ang < 240:
            floor_sel = get_sorted_rows(int(ang_floor) % 360, self.old_df_forces)
            ceil_sel = get_sorted_rows(
                int(ang_ceil_for_select) % 360, self.old_df_forces
            )
        else:
            floor_sel = get_sorted_rows(int(ang_floor) % 360, self.df_forces)
            ceil_sel = get_sorted_rows(int(ang_ceil_for_select) % 360, self.df_forces)

        # Arrays for interpolation along Vw
        Sr_floor = floor_sel["Spin ratio"].to_numpy(dtype=float)
        F_floor = floor_sel[force].to_numpy(dtype=float)
        Sr_ceil = ceil_sel["Spin ratio"].to_numpy(dtype=float)
        F_ceil = ceil_sel[force].to_numpy(dtype=float)
        # Force for floor angle
        idx_f = np.argsort(Sr_floor)
        Sr_f_sorted = Sr_floor[idx_f]
        F_f_sorted = F_floor[idx_f]

        idx_c = np.argsort(Sr_ceil)
        Sr_c_sorted = Sr_ceil[idx_c]
        F_c_sorted = F_ceil[idx_c]

        f_floor_at_vel = np.interp(spin_ratio, Sr_f_sorted, F_f_sorted)
        # Force for ceil angle
        f_ceil_at_vel = np.interp(spin_ratio, Sr_c_sorted, F_c_sorted)

        # Angular interpolation (handle wrap-around)
        af = ang_floor
        ac = ang_ceil_for_interp
        # normalize so ac > af (if wrap occurred ac will be > af since ac==360)
        if ac <= af:
            ac += 360.0
        a = ang if ang >= af else ang + 360.0
        if ac == af:
            frac = 0.0
        else:
            frac = (a - af) / (ac - af)
            frac = np.clip(frac, 0.0, 1.0)
            f = f_floor_at_vel + frac * (f_ceil_at_vel - f_floor_at_vel)

        if spin_ratio >= 2.29 and spin_ratio <= 4.58:
            return f
        elif spin_ratio < 2.29:
            denom = 0.5 * 1.2 * (5.72**2) * self.Ax / 1000.0
            coef_x = f / denom if denom > 0 else np.nan
            # sr = 0: cx = 0
            coef_x_interp = (spin_ratio / (2.29)) * coef_x
        else:
            denom = 0.5 * 1.2 * (17.15**2) * self.Ax / 1000.0
            coef_x = f / denom if denom > 0 else np.nan
            coef_x_interp = coef_x + ((spin_ratio - 4.58) / (12 - 4.58)) * (0 - coef_x)

        # Voce vai pegar o spin ratio = 1: para sr = 0: cx = 0. Para sr = 2.29, você acha o cx de

        new_f = coef_x_interp * 0.5 * 1.2 * (vel**2) * self.Ax / 1000.0

        return new_f

    def get_power_rotor(self, ang, vel, moment="Mz_rotor", draft=None, index=0):
        # allow draft override for parallel workers
        draft_local = self.draft if draft is None else draft
        floor_angle, ceil_angle = self.get_adjacent_angles(ang)
        if ceil_angle == 360:
            ceil_angle = 0
        floor_moments = self.df_forces[
            (self.df_forces["Angulo"] == floor_angle)
            & (self.df_forces["Calado"] == draft_local)
        ]

        ceil_moments = self.df_forces[
            (self.df_forces["Angulo"] == ceil_angle)
            & (self.df_forces["Calado"] == draft_local)
        ]
        if vel >= 6 and vel <= 10:
            mz_ceil = ceil_moments[moment].iloc[0] + (
                vel - ceil_moments["Vw"].iloc[0]
            ) * (ceil_moments[moment].iloc[1] - ceil_moments[moment].iloc[0]) / (
                ceil_moments["Vw"].iloc[1] - ceil_moments["Vw"].iloc[0]
            )
            mz_floor = floor_moments[moment].iloc[0] + (
                vel - floor_moments["Vw"].iloc[0]
            ) * (floor_moments[moment].iloc[1] - floor_moments[moment].iloc[0]) / (
                floor_moments["Vw"].iloc[1] - floor_moments["Vw"].iloc[0]
            )
            m = mz_floor + (ang - floor_angle) * (mz_ceil - mz_floor) / (
                ceil_angle - floor_angle
            )

        elif vel > 10 and vel <= 12:
            mz_ceil = ceil_moments[moment].iloc[1] + (
                vel - ceil_moments["Vw"].iloc[1]
            ) * (ceil_moments[moment].iloc[2] - ceil_moments[moment].iloc[1]) / (
                ceil_moments["Vw"].iloc[2] - ceil_moments["Vw"].iloc[1]
            )
            mz_floor = floor_moments[moment].iloc[1] + (
                vel - floor_moments["Vw"].iloc[1]
            ) * (floor_moments[moment].iloc[2] - floor_moments[moment].iloc[1]) / (
                floor_moments["Vw"].iloc[2] - floor_moments["Vw"].iloc[1]
            )
            m = mz_floor + (ang - floor_angle) * (mz_ceil - mz_floor) / (
                ceil_angle - floor_angle
            )

        elif vel < 6:
            m = 0.5 * (ceil_moments[moment].iloc[0] + floor_moments[moment].iloc[0])

        elif vel > 12:
            m = 0.5 * (ceil_moments[moment].iloc[2] + floor_moments[moment].iloc[2])

        P_cons = m * self.rotation * np.pi / (30 * self.eta_rot)

        return P_cons

    def save_wind_speeds(self, i):
        if np.round(self.df.loc[i, "LAT"]) == np.round(self.lat) and np.round(
            self.df.loc[i, "LON"]
        ) == np.round(self.lon):
            self.lat, self.lon = self.df.loc[i, "LAT"], self.df.loc[i, "LON"]
            return self.wind_u, self.wind_v, self.lat, self.lon
        self.lat, self.lon = self.df.loc[i, "LAT"], self.df.loc[i, "LON"]

        u10_point = self.ds.u10.sel(
            latitude=self.lat, longitude=self.lon, method="nearest"
        )
        v10_point = self.ds.v10.sel(
            latitude=self.lat, longitude=self.lon, method="nearest"
        )
        self.wind_u = np.zeros(8784)
        self.wind_v = np.zeros(8784)
        for j in tqdm(range(8784)):
            try:
                self.wind_u[j] = u10_point.sel(
                    time=self.timestamp, method="nearest"
                ).values
            except:
                pass
            try:
                self.wind_v[j] = v10_point.sel(
                    time=self.timestamp, method="nearest"
                ).values
            except:
                pass

            self.timestamp += pd.Timedelta(hours=1)
        return self.wind_u, self.wind_v, self.lat, self.lon

    def calc_ship_speed(self, ang_ship):
        u_ship = self.vs_ms * np.sin(np.radians(ang_ship))
        v_ship = self.vs_ms * np.cos(np.radians(ang_ship))
        return u_ship, v_ship

    def rotate_vel(self, u, v, ang_ship):
        u_rel = u * np.sin(ang_ship) + v * np.cos(ang_ship)
        v_rel = -u * np.cos(ang_ship) + v * np.sin(ang_ship)
        return u_rel, v_rel

    def calc_relative_velocity_and_angle(self, u10, v10, angle_ship, u_ship, v_ship):
        u_rel_xy = u10 - u_ship
        v_rel_xy = v10 - v_ship
        u_rel, v_rel = self.rotate_vel(u_rel_xy, v_rel_xy, np.radians(angle_ship))
        # Angulo relativo em graus
        angle_rel = self.wind_rel(u_rel, v_rel)
        return u_rel, v_rel, angle_rel

    def process_dataframe(self):
        total_range = np.arange(len(self.df) - 1)
        ang_ship = []
        ang_vento = []
        u10 = []
        v10 = []
        u_ship = []
        v_ship = []
        u_rel = []
        v_rel = []
        angle_rel = []
        times = []
        force_x = []
        force_x_rotores = []
        p_cons = []
        pbar = tqdm(total_range, desc="Ida: Calado Carregado")
        z_height = 27.7
        self.draft = 16 if self.ship == "suez" else 15
        self.RT = 744
        R_T = np.zeros(total_range.shape[0])
        P_E = np.zeros(total_range.shape[0])

        cond = "carreg"
        for i in pbar:
            pbar.set_description(f"[INFO] From time: {self.timestamp}, {cond}")
            if cond == "carreg":
                R_T[i] = 640
            else:
                R_T[i] = 475
            P_E[i] = R_T[i] * self.vs_ms / 0.63
            if i == int(total_range[-1] / 2):
                z_height = 35.5
                self.draft = 8.5 if self.ship == "suez" else 6.9
                self.RT = 696
                cond = "lastro"
            res = self.get_wind_properties(i, z_height)

            # Wind parameters
            u10.append(res[0])
            v10.append(res[1])
            ang_vento.append(res[3])

            # Ship parameters
            ang_ship.append(res[2])
            u_ship_i, v_ship_i = self.calc_ship_speed(res[2])
            u_ship.append(u_ship_i)
            v_ship.append(v_ship_i)

            # Relative parameters
            u_rel_i, v_rel_i, angle_rel_i = self.calc_relative_velocity_and_angle(
                res[0], res[1], res[2], u_ship_i, v_ship_i
            )
            u_rel.append(u_rel_i)
            v_rel.append(v_rel_i)
            angle_rel.append(angle_rel_i)

            vel_mag = np.sqrt(np.power(u_rel_i, 2) + np.power(v_rel_i, 2))

            if vel_mag > 20:
                vel_mag = 20
            times.append(str(self.timestamp.strftime("%Y-%m-%d %X")))
            self.timestamp += pd.Timedelta(seconds=self.dt)

            if self.calculate_forces:
                fx_total = self.get_forces(angle_rel_i, vel_mag, "fx", index=i)
                fx_rotores = self.get_forces(
                    angle_rel_i, vel_mag, "fx_rotores", index=i
                )
                if fx_rotores > self.RT:
                    fx_rotores = self.RT
                if angle_rel_i > 120 and angle_rel_i < 240:
                    p_cons.append(0)
                else:
                    p_cons.append(
                        self.get_power_rotor(angle_rel_i, vel_mag, index=i) / 1000
                    )

                force_x.append(fx_total)
                force_x_rotores.append(fx_rotores)
        u10 = np.array(u10)
        v10 = np.array(v10)
        ang_ship = np.array(ang_ship)
        ang_vento = np.array(ang_vento)
        u_rel = np.array(u_rel)
        v_rel = np.array(v_rel)
        angle_rel = np.array(angle_rel)
        times = np.array(times)
        if self.calculate_forces:
            force_x = np.array(force_x)
            force_x_rotores = np.array(force_x_rotores)
            p_cons = np.array(p_cons)
            force_x_casco_sup = force_x - force_x_rotores
            p_prop = ((R_T - force_x_rotores) * self.vs_ms) / 0.63
            P_E_ = p_cons + p_prop
            P_E_[P_E_ < 0] = 0
            gain = 1 - P_E_ / P_E
        else:
            force_x = np.zeros(u10.shape)
            force_x_rotores = np.zeros(u10.shape)
            p_cons = np.zeros(u10.shape)
            force_x_casco_sup = np.zeros(u10.shape)
            p_prop = np.zeros(u10.shape)
            P_E_ = np.zeros(u10.shape)
            gain = np.zeros(u10.shape)

        self.new_df = self.df.loc[total_range, ["LAT", "LON"]].copy()
        self.new_df["time"] = times
        self.new_df["u10"] = u10
        self.new_df["v10"] = v10
        self.new_df["u_ship"] = u_ship
        self.new_df["v_ship"] = v_ship
        self.new_df["u_rel"] = u_rel
        self.new_df["v_rel"] = v_rel
        self.new_df["angle_rel"] = angle_rel
        self.new_df["force_x_total"] = force_x
        self.new_df["force_x_rotor"] = force_x_rotores
        self.new_df["force_x_casco_sup"] = force_x_casco_sup
        self.new_df["p_cons"] = p_cons
        self.new_df["p_prop"] = p_prop
        self.new_df["p_e_rotor"] = P_E_
        self.new_df["gain"] = gain
        return self.new_df

    def process_per_route(self):
        csv_path = f"D:/{ship}/route_csvs{self.rotation}/wind_data_year_{int(self.timestamp.year)}_month_{self.timestamp.month}_day_{self.timestamp.day}_hour_{self.timestamp.hour}.csv"

        if os.path.exists(csv_path):
            self.new_df = pd.read_csv(csv_path, sep=",")
        else:
            self.process_dataframe()

    def save_csv(self, timestamp, ship):
        csv_path = f"D:/{ship}/route_csvs{self.rotation}/wind_data_year_{timestamp.year}_month_{timestamp.month}_day_{timestamp.day}_hour_{timestamp.hour}.csv"
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        if not os.path.exists(csv_path):
            print(f"[INFO] Saving informations with start time: {timestamp} ..")
            self.new_df.to_csv(csv_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Wind Route Creator")
    parser.add_argument("-s", "--ship", help="afra or suez")
    parser.add_argument("--rotation", required=True, help="100 or 180")
    parser.add_argument("--start-month", required=True, help="Start month")
    parser.add_argument("--year", required=True, type=int, help="Year")

    parser.add_argument(
        "--no-forces", action="store_true", help="Do not calculate forces"
    )
    args = parser.parse_args()

    calculate_forces = False if args.no_forces else True

    ship = "abdias_suez" if args.ship == "suez" else "castro_alves_afra"
    year = args.year

    current_time = pd.Timestamp(f"{year}-{int(args.start_month)}-01 00:00:00")
    wind_csv = "data.csv"

    forces_path = f"../{ship}/forces_CFD_rotores_desligados.csv"
    old_forces_path = f"../{ship}/forces_CFD.csv"
    map_processer = ProcessMap(
        timestamp=current_time,
        route_path=f"../{ship}/ais/{wind_csv}",
        wind_path=f"../{ship}/gribs_2020/{year}_1.grib",
        forces_path=forces_path,
        old_forces_path=old_forces_path,
        ship=args.ship,
        rotation=args.rotation,
        calculate_forces=calculate_forces,
    )
    try:
        for i in range(744 * 3):
            print("Time starts: ", current_time)
            map_processer.timestamp = current_time
            map_processer.load_data()
            map_processer.process_per_route()
            map_processer.save_csv(current_time, ship)
            try:
                del map_processer.new_df
                map_processer.new_df = None
            except Exception:
                pass
            gc.collect()
            current_time += pd.Timedelta(hours=1)
    finally:
        # garantir fechamento do dataset aberto
        try:
            if map_processer.ds is not None:
                map_processer.ds.close()
                map_processer.ds = None
                ProcessMap._ds_path = None
        except Exception:
            pass
        gc.collect()
