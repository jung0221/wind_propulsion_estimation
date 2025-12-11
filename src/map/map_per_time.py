import pandas as pd
import numpy as np
import os
import folium
import glob
import argparse
import random


class MapPerRoute:
    def __init__(
        self,
    ):
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
        self.hull_force = None
        self.draft = None
        self.Ax = 1130
        self.Ay = 3300
        self.force_x = None
        self.force_y = None
        self.new_df = None

    def create_map(self, df, out_path):
        if np.any(df) == None:
            df = self.df.copy()
        lat = df["LAT"][:]
        lon = df["LON"][:]
        times = df["time"][:]
        angle = df["angle_rel"][:]
        Fx = df["force_x_total"][:]
        Vw = (df["u_rel"] ** 2 + df["v_rel"] * 2) ** 0.5
        Fx_rotores = df["force_x_rotor"][:]
        Fx_casco_sup = df["force_x_casco_sup"][:]
        P_cons = df["p_cons"]
        P_E_ = df["p_e_rotor"]
        gain = df["gain"]
        u10, v10 = df["u10"], df["v10"]
        uship, vship = df["u_ship"], df["v_ship"]
        urel, vrel = df["u_rel"], df["v_rel"]

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
                    <tr>
                    <tr style="background-color: #f8f9fa;">
                        <td style="padding: 4px 8px; font-weight: bold; border-bottom: 1px solid #dee2e6;">Ganho</td>
                        <td style="padding: 4px 8px; border-bottom: 1px solid #dee2e6; font-weight: bold;">{int(100*gain.iloc[i])}% </td>
                    </tr>
                    <tr>
                        <td style="padding: 4px 8px; font-weight: bold; border-bottom: 1px solid #dee2e6;">U wind</td>
                        <td style="padding: 4px 8px; border-bottom: 1px solid #dee2e6; font-weight: bold;">{np.round(u10[i])} m/s</td>
                    </tr>
                    <tr>
                        <td style="padding: 4px 8px; font-weight: bold; border-bottom: 1px solid #dee2e6;">V wind</td>
                        <td style="padding: 4px 8px; border-bottom: 1px solid #dee2e6; font-weight: bold;">{np.round(v10[i])} m/s</td>
                    </tr>
                    <tr style="background-color: #f8f9fa;">
                        <td style="padding: 4px 8px; font-weight: bold; border-bottom: 1px solid #dee2e6;">U ship</td>
                        <td style="padding: 4px 8px; border-bottom: 1px solid #dee2e6; font-weight: bold;">{np.round(uship[i])} m/s</td>
                    </tr>
                    <tr>
                        <td style="padding: 4px 8px; font-weight: bold; border-bottom: 1px solid #dee2e6;">V ship</td>
                        <td style="padding: 4px 8px; border-bottom: 1px solid #dee2e6; font-weight: bold;">{np.round(vship[i])} m/s</td>
                    </tr>
                    <tr>
                        <td style="padding: 4px 8px; font-weight: bold; border-bottom: 1px solid #dee2e6;">U rel</td>
                        <td style="padding: 4px 8px; border-bottom: 1px solid #dee2e6; font-weight: bold;">{np.round(urel[i])} m/s</td>
                    </tr>
                    <tr>
                        <td style="padding: 4px 8px; font-weight: bold; border-bottom: 1px solid #dee2e6;">V rel</td>
                        <td style="padding: 4px 8px; border-bottom: 1px solid #dee2e6; font-weight: bold;">{np.round(vrel[i])} m/s</td>
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

        filename = f"map.html"

        if not os.path.exists(out_path):
            os.makedirs(out_path, exist_ok=True)

        self.m.save(os.path.join(out_path, filename))

        print(f"[INFO] Map saved to {out_path}{filename}")


def main():

    parser = argparse.ArgumentParser(description="Wind Route Creator")
    parser.add_argument("-d", "--data", help="route csv file")
    parser.add_argument("-o", "--output", help="output folder")

    args = parser.parse_args()
    map_creator = MapPerRoute()
    df = pd.read_csv(args.data)

    df = df.iloc[: int(len(df) / 2)]

    map_creator.create_map(df, args.output)


if __name__ == "__main__":
    main()
