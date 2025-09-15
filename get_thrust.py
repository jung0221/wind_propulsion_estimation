import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import argparse
import bisect
import folium
import matplotlib.pyplot as plt
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


class GetThrust:
    def __init__(
        self,
        outbound_csv_path: str,
        return_csv_path: str,
        forces_path: str,
        rotation: int,
        ship: str,
        no_rotor: bool
    ):
        self.outbound_csv_path = outbound_csv_path
        self.return_csv_path = return_csv_path
        self.forces_path = forces_path
        self.rotation = rotation
        self.ship = ship
        self.v_s = 12  # knots
        self.R_T = 744
        self.P_E = 4592
        self.eta_d = 0.7
        self.eta_rot = 0.9
        self.P_B = self.P_E / self.eta_d
        self.timestamp = None
        self.current_month = None
        self.df_outbound = None
        self.df_return = None
        self.df_forces = None
        self.hull_force = None
        self.draft = None
        self.Ax = 1130
        self.Ay = 3300
        self.force_x = None
        self.force_y = None
        self.new_df = None
        self.no_rotor = no_rotor

    def load_data(self):

        print(f"[INFO] Loading route for time: {self.timestamp}")
        print(self.current_month, self.timestamp.month)
        if self.current_month != self.timestamp.month:
            print(
                f"[INFO] Changing month from {self.current_month} to {self.timestamp.month}"
            )

        print("[INFO] Loading outbound route data")
        try:
            self.df_outbound = pd.read_csv(
                os.path.join(
                    self.outbound_csv_path,
                    f"wind_data_year_{self.timestamp.year}_month_{self.timestamp.month}_day_{self.timestamp.day}_hour_{self.timestamp.hour}.csv",
                ),
                sep=",",
            )
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            print("[WARNING] Continuing without return data")
        except pd.errors.EmptyDataError:
            print(f"[ERROR] Data file is empty")
        except Exception as e:
            print(f"[ERROR] Unexpected error loading return data: {e}")
        return_time = pd.Timestamp(self.df_outbound["time"].iloc[-1]) + pd.Timedelta(
            hours=1
        )
        print("[INFO] Loading return route data")
        try:
            self.df_return = pd.read_csv(
                os.path.join(
                    self.return_csv_path,
                    f"wind_data_year_{return_time.year}_month_{return_time.month}_day_{return_time.day}_hour_{return_time.hour}.csv",
                ),
                sep=",",
            )
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            print("[WARNING] Continuing without return data")
        except pd.errors.EmptyDataError:
            print(f"[ERROR] Data file is empty")
        except Exception as e:
            print(f"[ERROR] Unexpected error loading return data: {e}")

    def load_forces(self):
        if not self.df_forces:
            try:
                print("[INFO] Loading thrust data")
                self.df_forces = pd.read_csv(self.forces_path)
            except FileNotFoundError as e:
                print(f"[ERROR] {e}")
                print("[WARNING] Continuing without return data")
            except pd.errors.EmptyDataError:
                print(f"[ERROR] Data file is empty")
            except Exception as e:
                print(f"[ERROR] Unexpected error loading return data: {e}")

    def get_values(self, option="outbound"):
        if option == "outbound":
            df = self.df_outbound.copy()
            self.draft = 16

        else:
            df = self.df_return.copy()
            self.draft = 8.5

        self.angles = df["angle_rel"]
        self.vels = np.sqrt(np.power(df["u_rel"], 2) + np.power(df["v_rel"], 2))

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

    def get_forces(self, ang, vel, force="fx", coef_dir="cx"):
        floor_angle, ceil_angle = self.get_adjacent_angles(ang)
        if ceil_angle == 360:
            ceil_angle = 0
        floor_forces = self.df_forces[
            (self.df_forces["Angulo"] == floor_angle)
            & (self.df_forces["Calado"] == self.draft)
            & (self.df_forces["Rotacao"] == self.rotation)
        ]

        ceil_forces = self.df_forces[
            (self.df_forces["Angulo"] == ceil_angle)
            & (self.df_forces["Calado"] == self.draft)
            & (self.df_forces["Rotacao"] == self.rotation)
        ]
        if vel >= 6 and vel <= 10:
            fx_ceil = ceil_forces[force].iloc[0] + (vel - ceil_forces["Vw"].iloc[0]) * (
                ceil_forces[force].iloc[1] - ceil_forces[force].iloc[0]
            ) / (ceil_forces["Vw"].iloc[1] - ceil_forces["Vw"].iloc[0])
            fx_floor = floor_forces[force].iloc[0] + (
                vel - floor_forces["Vw"].iloc[0]
            ) * (floor_forces[force].iloc[1] - floor_forces[force].iloc[0]) / (
                floor_forces["Vw"].iloc[1] - floor_forces["Vw"].iloc[0]
            )
            if ceil_angle != floor_angle:
                f = fx_floor + (ang - floor_angle) * (fx_ceil - fx_floor) / (
                    ceil_angle - floor_angle
                )
            else:
                f = fx_floor

        elif vel > 10 and vel <= 12:
            fx_ceil = ceil_forces[force].iloc[1] + (vel - ceil_forces["Vw"].iloc[1]) * (
                ceil_forces[force].iloc[2] - ceil_forces[force].iloc[1]
            ) / (ceil_forces["Vw"].iloc[2] - ceil_forces["Vw"].iloc[1])
            fx_floor = floor_forces[force].iloc[1] + (
                vel - floor_forces["Vw"].iloc[1]
            ) * (floor_forces[force].iloc[2] - floor_forces[force].iloc[1]) / (
                floor_forces["Vw"].iloc[2] - floor_forces["Vw"].iloc[1]
            )
            if ceil_angle != floor_angle:
                f = fx_floor + (ang - floor_angle) * (fx_ceil - fx_floor) / (
                    ceil_angle - floor_angle
                )
            else:
                f = fx_floor

        elif vel < 6:
            coef_x_floor = floor_forces[floor_forces["Vw"] == 6][coef_dir].values[0]
            coef_x_ceil = ceil_forces[ceil_forces["Vw"] == 6][coef_dir].values[0]
            if ceil_angle != floor_angle:
                coef_x = coef_x_floor + (ang - floor_angle) * (
                    coef_x_ceil - coef_x_floor
                ) / (ceil_angle - floor_angle)
            else:
                coef_x = coef_x_floor
            f = coef_x * 0.5 * 1.2 * np.power(vel, 2) * self.Ax / 1000

        elif vel > 12:
            coef_x_floor = floor_forces[floor_forces["Vw"] == 12][coef_dir].values[0]
            coef_x_ceil = ceil_forces[ceil_forces["Vw"] == 12][coef_dir].values[0]
            if ceil_angle != floor_angle:
                coef_x = coef_x_floor + (ang - floor_angle) * (
                    coef_x_ceil - coef_x_floor
                ) / (ceil_angle - floor_angle)
            else:
                coef_x = coef_x_floor
            f = coef_x * 0.5 * 1.2 * np.power(vel, 2) * self.Ax / 1000

        return f

    def create_subset_df(self, option="outbound"):
        if option == "outbound":
            df = self.df_outbound
        else:
            df = self.df_return

        subset_df = df[["LAT", "LON", "time", "angle_rel"]].copy()

        subset_df["Vw"] = self.vels
        subset_df["Fx"] = self.force_x
        subset_df["Fy"] = self.force_y
        subset_df["Fx_rotores"] = self.force_x_rotores
        subset_df["Fy_rotores"] = self.force_y_rotores
        subset_df["Fx_casco_sup"] = self.force_x_casco_sup
        subset_df["Fy_casco_sup"] = self.force_y_casco_sup
        subset_df["P_cons"] = self.P_cons
        subset_df["P_prop"] = self.P_prop
        subset_df["P_E_"] = self.P_E_
        subset_df["Gain"] = self.gain

        return subset_df

    def concat_and_save_df(self, df_out, df_ret, output_path):
        self.new_df = pd.concat([df_out, df_ret], ignore_index=True)
        self.new_df.to_csv(output_path)
        return

    def create_map(self, df, option="outbound"):
        if np.any(df) == None:
            df = self.df.copy()
        lat = df["LAT"][:]
        lon = df["LON"][:]
        times = df["time"][:]
        Vw = df["Vw"][:]
        angle = df["angle_rel"][:]
        Fx = df["Fx"][:]
        Fy = df["Fy"][:]
        Fx_rotores = df["Fx_rotores"][:]
        Fy_rotores = df["Fy_rotores"][:]
        Fx_casco_sup = df["Fx_casco_sup"][:]
        Fy_casco_sup = df["Fy_casco_sup"][:]
        P_cons = df["P_cons"]
        P_E_ = df["P_E_"]
        gain = df["Gain"]

        # Create a map centered around the first coordinate
        self.m = folium.Map(location=[lat.iloc[0], lon.iloc[0]], zoom_start=12)
        # Add the trajectory to the map
        # Add custom CSS for better tooltips
        tooltip_css = """
        <style>
        .leaflet-tooltip {
            font-size: 13px !important;
            min-width: 200px !important;
            max-width: 350px !important;
            padding: 8px !important;
            border-radius: 6px !important;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3) !important;
            background-color: rgba(255,255,255,0.95) !important;
        }
        .leaflet-popup-content {
            margin: 8px 12px !important;
        }
        </style>
        """
        self.m.get_root().html.add_child(folium.Element(tooltip_css))

        # Color coding based on Fx values
        fx_min, fx_max = Fx.min(), Fx.max()

        def get_color(fx_value):
            """Return color based on Fx value"""
            if fx_value < 20:
                return "red"
            elif fx_value < 100:
                return "orange"
            else:
                return "green"

        # Add the trajectory to the map
        i = 0
        for lat, lon in zip(lat, lon):
            # Extended popup with charts or more details
            popup_html = f"""
            <div style="font-family: 'Segoe UI', Arial, sans-serif; font-size: 12px; width: 280px;">
                <div style="background: linear-gradient(90deg, #2E86AB, #A23B72); color: white; padding: 8px; margin: -8px -12px 8px -12px; border-radius: 4px 4px 0 0;">
                    <h4 style="margin: 0; text-align: center;">Data Point {i}</h4>
                </div>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="background-color: #f8f9fa;">
                        <td style="padding: 4px 8px; font-weight: bold; border-bottom: 1px solid #dee2e6;">Time</td>
                        <td style="padding: 4px 8px; border-bottom: 1px solid #dee2e6;">{times.iloc[i]}</td>
                    </tr>
                    <tr>
                        <td style="padding: 4px 8px; font-weight: bold; border-bottom: 1px solid #dee2e6;">Position</td>
                        <td style="padding: 4px 8px; border-bottom: 1px solid #dee2e6;">{lat:.4f}°, {lon:.4f}°</td>
                    </tr>
                    <tr style="background-color: #f8f9fa;">
                        <td style="padding: 4px 8px; font-weight: bold; border-bottom: 1px solid #dee2e6;">Relative Wind Speed</td>
                        <td style="padding: 4px 8px; border-bottom: 1px solid #dee2e6;">{Vw.iloc[i]:.2f} m/s</td>
                    </tr>
                    <tr>
                        <td style="padding: 4px 8px; font-weight: bold; border-bottom: 1px solid #dee2e6;">Relative Angle</td>
                        <td style="padding: 4px 8px; border-bottom: 1px solid #dee2e6;">{int(angle.iloc[i])}°</td>
                    </tr>
                    <tr style="background-color: #f8f9fa;">
                        <td style="padding: 4px 8px; font-weight: bold; border-bottom: 1px solid #dee2e6;">Fx Total</td>
                        <td style="padding: 4px 8px; border-bottom: 1px solid #dee2e6; color: {get_color(Fx.iloc[i])}; font-weight: bold;">{Fx.iloc[i]:.2f} kN</td>
                    </tr>
                    <tr>
                        <td style="padding: 4px 8px; font-weight: bold; border-bottom: 1px solid #dee2e6;">Fx Rotor</td>
                        <td style="padding: 4px 8px; border-bottom: 1px solid #dee2e6; color: {get_color(Fx_rotores.iloc[i])}; font-weight: bold;">{Fx_rotores.iloc[i]:.2f} kN</td>
                    </tr>
                        <td style="padding: 4px 8px; font-weight: bold; border-bottom: 1px solid #dee2e6;">Fx casco/sup</td>
                        <td style="padding: 4px 8px; border-bottom: 1px solid #dee2e6; color: {get_color(Fx_casco_sup.iloc[i])}; font-weight: bold;">{Fx_casco_sup.iloc[i]:.2f} kN</td>
                    </tr>
                    <tr>
                    <tr style="background-color: #f8f9fa;">
                        <td style="padding: 4px 8px; font-weight: bold; border-bottom: 1px solid #dee2e6;">Fx Total</td>
                        <td style="padding: 4px 8px; border-bottom: 1px solid #dee2e6; color: {get_color(Fy.iloc[i])}; font-weight: bold;">{Fy.iloc[i]:.2f} kN</td>
                    </tr>
                    <tr>
                        <td style="padding: 4px 8px; font-weight: bold; border-bottom: 1px solid #dee2e6;">Fx Rotor</td>
                        <td style="padding: 4px 8px; border-bottom: 1px solid #dee2e6; color: {get_color(Fy_rotores.iloc[i])}; font-weight: bold;">{Fy_rotores.iloc[i]:.2f} kN</td>
                    </tr>
                        <td style="padding: 4px 8px; font-weight: bold; border-bottom: 1px solid #dee2e6;">Fx casco/sup</td>
                        <td style="padding: 4px 8px; border-bottom: 1px solid #dee2e6; color: {get_color(Fy_casco_sup.iloc[i])}; font-weight: bold;">{Fy_casco_sup.iloc[i]:.2f} kN</td>
                    </tr>
                    <tr>
                    <tr style="background-color: #f8f9fa;">
                        <td style="padding: 4px 8px; font-weight: bold; border-bottom: 1px solid #dee2e6;">P_cons</td>
                        <td style="padding: 4px 8px; border-bottom: 1px solid #dee2e6; font-weight: bold;">{int(P_cons.iloc[i])} kW</td>
                    </tr>
                    <tr>
                        <td style="padding: 4px 8px; font-weight: bold; border-bottom: 1px solid #dee2e6;">P_E,w/o rotor</td>
                        <td style="padding: 4px 8px; border-bottom: 1px solid #dee2e6; font-weight: bold;">{int(self.P_E)} kW</td>
                    </tr>
                    <tr>
                        <td style="padding: 4px 8px; font-weight: bold; border-bottom: 1px solid #dee2e6;">P_E,w/ rotor</td>
                        <td style="padding: 4px 8px; border-bottom: 1px solid #dee2e6; font-weight: bold;">{int(P_E_.iloc[i])} kW</td>
                    </tr>
                        <td style="padding: 4px 8px; font-weight: bold; border-bottom: 1px solid #dee2e6;">Ganho</td>
                        <td style="padding: 4px 8px; border-bottom: 1px solid #dee2e6; font-weight: bold;">{int(100*gain.iloc[i])}% </td>
                    </tr>
                </table>
            </div>
            """

            # Compact tooltip for hover
            tooltip_text = f"""
            <div style="text-align: center;">
                <b>Point {i}</b><br>
                <span style="color: #666;">Lat: {lat:.3f}° | Lon: {lon:.3f}°</span><br>
                <span style="color: #2E86AB;">Relative Wind: {Vw.iloc[i]:.1f} m/s | Angle: {int(angle.iloc[i])}°</span><br>
                <span style="color: {get_color(Fx.iloc[i])}; font-weight: bold;">Fx: {Fx.iloc[i]:.1f} kN</span> | 
                <span style="color: {get_color(Fy.iloc[i])}; font-weight: bold;">Fy: {Fy.iloc[i]:.1f} kN</span>
            </div>
            """

            folium.CircleMarker(
                location=[lat, lon],
                radius=4,
                color=get_color(Fx.iloc[i]),
                fillColor=get_color(Fx.iloc[i]),
                fillOpacity=0.8,
                weight=2,
                popup=folium.Popup(popup_html, max_width=320, min_width=300),
                tooltip=folium.Tooltip(tooltip_text, sticky=True),
            ).add_to(self.m)
            i += 1

        image_path = "spiral.png"

        if os.path.exists(image_path):
            # Convert image to base64 for embedding
            import base64

            with open(image_path, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode()

            # Create HTML for the expandable image overlay
            html = f"""
            <style>
            .expandable-image {{
                position: fixed;
                top: 10px;
                right: 10px;
                width: 300px;
                height: 200px;
                background-color: white;
                border: 2px solid grey;
                z-index: 9999;
                font-size: 14px;
                padding: 5px;
                border-radius: 5px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.3);
                transition: all 0.3s ease-in-out;
                cursor: pointer;
            }}
            
            .expandable-image:hover {{
                width: 900px;
                height: 600px;
                transform-origin: top right;
            }}
            
            .expandable-image img {{
                width: 100%;
                height: auto;
                transition: all 0.3s ease-in-out;
            }}
            
            .expandable-image p {{
                margin: 0;
                text-align: center;
                font-weight: bold;
                padding-bottom: 5px;
            }}
            </style>
            
            <div class="expandable-image">
                <p>Fx Distribution (Hover to expand)</p>
                <img src="data:image/png;base64,{img_base64}" alt="Fx Distribution">
            </div>
            """
            self.m.get_root().html.add_child(folium.Element(html))

        calado = 16 if option == "outbound" else 8.5
        if self.no_rotor:
            out_path = (
                f"../{self.ship}/maps_{self.rotation}_no_rotor/{option}/calado_{calado}_rot_{self.rotation}/"
            )
        if not self.no_rotor:
            out_path = (
                f"../{self.ship}/maps_{self.rotation}/{option}/calado_{calado}_rot_{self.rotation}/"
            )
        
        filename = f"year_{self.timestamp.year}_month_{self.timestamp.month}_day_{self.timestamp.day}_hour_{self.timestamp.hour}.html"
        if not os.path.exists(out_path):
            os.makedirs(out_path, exist_ok=True)

        self.m.save(
            os.path.join(
                out_path,
                filename,
            )
        )
        
        print(f"[INFO] Map saved to {out_path}{filename}")

    def plot_histogram(self, value):
        """
        Plot histogram of DataFrame column(s)

        Args:
            value (str or list): Column name(s) to plot ('Fx', 'Fy', 'Vw', etc.)
                                Can be a single string or list of strings
        """
        if self.new_df is None:
            print("[ERROR] No DataFrame available. Run the analysis first.")
            return

        # Convert single value to list for uniform processing
        columns = [value] if isinstance(value, str) else value

        # Validate columns exist
        available_cols = self.new_df.columns.tolist()
        valid_columns = [col for col in columns if col in available_cols]

        if not valid_columns:
            print(f"[ERROR] None of the specified columns found in DataFrame.")
            print(f"Available columns: {available_cols}")
            return

        # Create subplots if multiple columns
        n_cols = len(valid_columns)
        if n_cols == 1:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            axes = [ax]
        else:
            fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))
            if n_cols == 1:
                axes = [axes]

        colors = ["skyblue", "lightgreen", "lightcoral", "gold", "lightpink"]
        unit_map = {
            "Fx": "kN",
            "Fy": "kN",
            "Vw": "m/s",
            "angle_rel": "°",
            "LAT": "°",
            "LON": "°",
            "Gain": "%",
            
        }

        for i, col in enumerate(valid_columns):
            data = self.new_df[col].dropna()

            if len(data) == 0:
                print(f"[WARNING] No valid data found for column '{col}'")
                continue

            # Plot histogram
            axes[i].hist(
                data,
                bins=30,
                alpha=0.7,
                color=colors[i % len(colors)],
                edgecolor="black",
            )

            # Add statistics
            mean_val = np.mean(data)
            std_val = np.std(data)

            axes[i].axvline(
                mean_val,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {mean_val:.2f}",
            )

            # Labels and title
            unit = unit_map.get(col, "")
            axes[i].set_xlabel(f"{col} ({unit})" if unit else col)
            axes[i].set_ylabel("Frequency")
            axes[i].set_title(f"Histogram of {col}")
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

            # Statistics text
            stats_text = f"Count: {len(data)}\nMean: {mean_val:.2f}\nStd: {std_val:.2f}"
            axes[i].text(
                0.02,
                0.98,
                stats_text,
                transform=axes[i].transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        plt.tight_layout()

        # Save the figure
        if not self.no_rotor:
            out_path = f"../{self.ship}/histograms_rot{self.rotation}/"
        else:
            out_path = f"../{self.ship}/histograms_rot{self.rotation}_no_rotor/"
        if not os.path.exists(out_path):
            os.makedirs(out_path, exist_ok=True)

        columns_str = "_".join(valid_columns)
        filename = f"{columns_str}_year_{self.timestamp.year}_month_{self.timestamp.month}_day_{self.timestamp.day}_hour_{self.timestamp.hour}.png"
        plt.savefig(os.path.join(out_path, filename), dpi=300, bbox_inches="tight")
        plt.close()

        print(f"[INFO] Histogram(s) saved to {out_path}{filename}")

    def get_power_rotor(self, ang, vel, moment="Mz_rotor", coef_dir="cz"):
        floor_angle, ceil_angle = self.get_adjacent_angles(ang)
        if ceil_angle == 360:
            ceil_angle = 0
        floor_moments = self.df_forces[
            (self.df_forces["Angulo"] == floor_angle)
            & (self.df_forces["Calado"] == self.draft)
            & (self.df_forces["Rotacao"] == self.rotation)
        ]

        ceil_moments = self.df_forces[
            (self.df_forces["Angulo"] == ceil_angle)
            & (self.df_forces["Calado"] == self.draft)
            & (self.df_forces["Rotacao"] == self.rotation)
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

    def process_single_point(self, ang, vel):
        fx_total = self.get_forces(ang, vel, "fx", "cx")
        fy_total = self.get_forces(ang, vel, "fy", "cy")
        fx_rotores = self.get_forces(ang, vel, "fx_rotores", "cx_rotores")
        fy_rotores = self.get_forces(ang, vel, "fy_rotores", "cy_rotores")
        
        fx_casco_sup = fx_total - fx_rotores
        fy_casco_sup = fy_total - fy_rotores
        p_cons = self.get_power_rotor(ang, vel)
        
        return {
            'fx_total': fx_total,
            'fy_total': fy_total,
            'fx_rotores': fx_rotores,
            'fy_rotores': fy_rotores,
            'fx_casco_sup': fx_casco_sup,
            'fy_casco_sup': fy_casco_sup,
            'p_cons': p_cons
        }

    def process_data_parallel(self, option='outbound', max_workers=None, use_threads=True):
        """
        Parallel version of process_data
        
        Args:
            option: 'outbound' or 'return'
            n_jobs: Number of parallel jobs (-1 uses all cores)
        """
        executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
        print(f"[INFO] Processing {len(self.angles)} points with {executor_class.__name__}")

        with executor_class(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = [
                executor.submit(self.process_single_point, ang, vel)
                for ang, vel in zip(self.angles, self.vels)
            ]
            
            # Collect results with progress bar
            results = []
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc=f'{option} (concurrent)'
            ):
                results.append(future.result())
        
        # Unpack results
        self.force_x = np.array([r['fx_total'] for r in results])
        self.force_y = np.array([r['fy_total'] for r in results])
        self.force_x_rotores = np.array([r['fx_rotores'] for r in results])
        self.force_y_rotores = np.array([r['fy_rotores'] for r in results])
        self.force_x_casco_sup = np.array([r['fx_casco_sup'] for r in results])
        self.force_y_casco_sup = np.array([r['fy_casco_sup'] for r in results])
        self.P_cons = np.array([r['p_cons'] for r in results]) / 1000
        
        self.P_prop = (self.R_T - self.force_x) * 0.5144 * self.v_s
        self.P_E_ = self.P_cons + self.P_prop
        self.gain = 1 - self.P_E_ / self.P_E
        
        print(f"[INFO] Parallel processing completed for {option}")  

    def process_data(self, option='outbound'):
        self.force_x = []
        self.force_y = []
        self.force_x_rotores = []
        self.force_y_rotores = []
        self.force_x_casco_sup = []
        self.force_y_casco_sup = []
        self.P_cons = []
        for ang, vel in tqdm(
            zip(self.angles, self.vels), total=len(self.angles), desc=option
        ):
            fx_total = self.get_forces(ang, vel, "fx", "cx")
            fy_total = self.get_forces(ang, vel, "fy", "cy")

            fx_rotores = self.get_forces(ang, vel, "fx_rotores", "cx_rotores")
            fy_rotores = self.get_forces(ang, vel, "fy_rotores", "cy_rotores")

            self.force_x.append(fx_total)
            self.force_y.append(fy_total)

            self.force_x_rotores.append(fx_rotores)
            self.force_y_rotores.append(fy_rotores)

            self.force_x_casco_sup.append(fx_total - fx_rotores)
            self.force_y_casco_sup.append(fy_total - fy_rotores)
            self.P_cons.append(self.get_power_rotor(ang, vel))

        if self.no_rotor:
            self.force_x = np.array(self.force_x_casco_sup)
            self.force_y = np.array(self.force_y_casco_sup)
            self.force_x_casco_sup = np.array(self.force_x_casco_sup)
            self.force_y_casco_sup = np.array(self.force_y_casco_sup)
            self.force_x_rotores = np.zeros(len(self.force_x_rotores))
            self.force_y_rotores = np.zeros(len(self.force_y_rotores))
            self.P_cons = np.zeros(len(self.P_cons))
            self.P_prop = (self.R_T - self.force_x) * 0.5144 * self.v_s
            self.P_E_ = self.P_cons + self.P_prop
            self.gain = 1 - self.P_E_ / self.P_E
            return
        self.force_x = np.array(self.force_x)
        self.force_y = np.array(self.force_y)
        self.force_x_casco_sup = np.array(self.force_x_casco_sup)
        self.force_y_casco_sup = np.array(self.force_y_casco_sup)
        self.force_x_rotores = np.array(self.force_x_rotores)
        self.force_y_rotores = np.array(self.force_y_rotores)
        self.P_cons = np.array(self.P_cons) / 1000
        self.P_prop = (self.R_T - self.force_x) * 0.5144 * self.v_s
        self.P_E_ = self.P_cons + self.P_prop
        self.gain = 1 - self.P_E_ / self.P_E
        return

    def run(self, timestamp):
        self.timestamp = timestamp
        self.load_data()
        if self.no_rotor:
            out_folder = f"../{self.ship}/routes_csv_rot{self.rotation}_no_rotor/"
        else:
            out_folder = f"../{self.ship}/routes_csv_rot{self.rotation}/"
        
        if not os.path.exists(out_folder):
            os.makedirs(out_folder, exist_ok=True)

        output_path = os.path.join(out_folder,
                f"wind_data_year_{self.timestamp.year}_month_{self.timestamp.month}_day_{self.timestamp.day}_hour_{self.timestamp.hour}.csv",
            )
        if not os.path.exists(output_path):
            self.angle_list = self.df_forces["Angulo"].unique()
            self.angle_list = np.append(self.angle_list, 360)

            self.get_values(option="outbound")
            self.process_data(option='outbound')
            # self.process_data_parallel(option='outbound', max_workers=4, use_threads=True)
            
            new_df_out = self.create_subset_df(option="outbound")

            self.get_values(option="return")
            self.process_data(option='return')
            # self.process_data_parallel(option='return', max_workers=4, use_threads=True)
            new_df_ret = self.create_subset_df(option="return")

            self.concat_and_save_df(new_df_out, new_df_ret, output_path)

            # Map and histogram creation
            if self.current_month != self.timestamp.month:
                if self.current_month:
                    print(f"[INFO] Month changed from {self.current_month}")
                print(f"[INFO] Saving Map from month {self.timestamp.month} and histogram")
                self.create_map(new_df_out, option="outbound")
                self.create_map(new_df_ret, option="return")

                self.plot_histogram("Fx")
                self.plot_histogram("Vw")
                self.plot_histogram("Gain")
                self.current_month = self.timestamp.month
        # TODO: Com relação a interpolação /extrapolação seria importante indicar no relatório como está foi feita (linear???). Talvez um gráfico 3D com ângulo de incidência X velocidade x força FX.


def main():
    parser = argparse.ArgumentParser(description="Wind Route Creator")
    parser.add_argument("--ship", required=True, help="afra or suez")
    parser.add_argument("--rotation", required=True, help="100 or 180")
    parser.add_argument("--no-rotor", action="store_true", help="100 or 180")
    
    args = parser.parse_args()
    ship = "abdias_suez" if args.ship == "suez" else "castro_alves_afra"
    if args.no_rotor:
        no_rotor=True
    else: 
        no_rotor=False

    current_time = pd.Timestamp("2020-01-01 00:00:00")
    outbound_csv_path = f"../{ship}/csvs_ida"
    return_csv_path = f"../{ship}/csvs_volta"
    forces_path = f"../{ship}/forces_V3.csv"
    get_thrust = GetThrust(
        outbound_csv_path=outbound_csv_path,
        return_csv_path=return_csv_path,
        forces_path=forces_path,
        rotation=int(args.rotation),
        ship=ship,
        no_rotor=no_rotor
    )

    get_thrust.load_forces()
    for i in range(2500):
        get_thrust.run(current_time)
        current_time += pd.Timedelta(hours=1)


if __name__ == "__main__":
    main()
