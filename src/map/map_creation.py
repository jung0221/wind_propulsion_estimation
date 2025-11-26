import pandas as pd
import numpy as np
import os
import folium
import matplotlib.pyplot as plt
from windrose import WindroseAxes
import glob
import io
import base64
from joblib import Parallel, delayed
import re
from tqdm import tqdm
import argparse


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
        self.df_forces = None
        self.hull_force = None
        self.draft = None
        self.Ax = 1130
        self.Ay = 3300
        self.force_x = None
        self.force_y = None
        self.new_df = None

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

    def create_map(self, df, option="outbound"):
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
        out_path = f"../abdias_suez/maps_100/{option}/calado_{calado}_rot_100/"

        # filename = f"year_{self.timestamp.year}_month_{self.timestamp.month}_day_{self.timestamp.day}_hour_{self.timestamp.hour}.html"
        filename = f"map.html"

        if not os.path.exists(out_path):
            os.makedirs(out_path, exist_ok=True)

        self.m.save(
            os.path.join(
                "maps",
                filename,
            )
        )

        print(f"[INFO] Map saved to {out_path}{filename}")


class GlobalMap:
    def __init__(self):
        """Global map that aggregates wind data across many route CSVs and
        creates a folium map where each marker shows a windrose built from
        the u10/v10 values of all CSVs for that route index.
        """
        self.m = None

    def _make_windrose_datauri(self, u_arr, v_arr, title=""):
        """Create a windrose PNG from arrays of u10 and v10 and return a data URI."""
        # Build temporary dataframe expected by get_windrose_from_route
        tmp = pd.DataFrame({"u10": u_arr, "v10": v_arr})
        # Create figure
        fig = plt.figure(figsize=(6, 6), dpi=150)
        ax = WindroseAxes.from_ax(fig=fig)
        try:
            get_windrose_from_route(tmp, output_name=title, ax=ax)
        except Exception:
            # fallback: plot simple windrose directly
            wind_speed = np.sqrt(np.array(u_arr) ** 2 + np.array(v_arr) ** 2)
            wind_dir = np.degrees(np.arctan2(-np.array(u_arr), -np.array(v_arr))) % 360
            ax.bar(wind_dir, wind_speed, normed=True, opening=0.8, edgecolor="white")
            ax.set_title(title)

        buf = io.BytesIO()
        # WindroseAxes are not always compatible with plt.tight_layout();
        # use a conservative subplots_adjust to avoid the UserWarning while
        # keeping reasonable margins.
        try:
            fig.subplots_adjust(left=0.06, right=0.98, top=0.94, bottom=0.06)
        except Exception:
            pass
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode()
        return f"data:image/png;base64,{img_b64}"

    def _save_windrose_png(
        self,
        u_arr,
        v_arr,
        out_path,
        title="",
        hide_cardinal_labels=False,
        invert_orientation=False,
    ):
        """Save a windrose PNG for given u/v arrays to out_path. Returns True if saved.

        Supports passing precomputed `bins` by using the global `bins` variable
        injected by the caller (see create_global_map) via closure, or by
        inspecting provided arrays. To keep windroses comparable, callers
        should compute `bins` (common edges) and set `global_bins` in the
        surrounding scope before invoking this method.
        """
        try:
            import matplotlib

            matplotlib.use("Agg")
            from windrose import WindroseAxes
            import matplotlib.pyplot as plt

            u = np.asarray(u_arr, dtype=float)
            v = np.asarray(v_arr, dtype=float)
            # filter invalid values
            mask = np.isfinite(u) & np.isfinite(v)
            u = u[mask]
            v = v[mask]
            if len(u) == 0:
                return False

            fig = plt.figure(figsize=(4, 4), dpi=150)
            ax = WindroseAxes.from_ax(fig=fig)
            wind_speed = np.sqrt(u**2 + v**2)
            if invert_orientation:
                wind_dir = np.degrees(np.arctan2(-u, -v)) % 360
            else:
                wind_dir = np.degrees(np.arctan2(u, v)) % 360
            # Try to reuse a caller-provided `bins` (common across all windroses)
            bins = None
            try:
                bins = globals().get("_GLOBAL_WINDROSE_BINS", None)
            except Exception:
                bins = None

            if bins is not None:
                ax.bar(
                    wind_dir,
                    wind_speed,
                    bins=bins,
                    normed=True,
                    opening=0.8,
                    edgecolor="white",
                )
            else:
                ax.bar(
                    wind_dir, wind_speed, normed=True, opening=0.8, edgecolor="white"
                )
            # optionally hide the cardinal direction labels (N, N-E, E, ...)
            if hide_cardinal_labels:
                try:
                    import matplotlib.pyplot as _plt

                    _plt.setp(ax.get_xticklabels(), visible=False)
                except Exception:
                    try:
                        ax.set_xticklabels([])
                    except Exception:
                        pass
            # ensure the speed/legend is shown on saved windrose images
            try:
                ax.set_legend(fontsize=9, title_fontsize=10)
            except Exception:
                try:
                    ax.legend(fontsize=9)
                except Exception:
                    pass
            # make radial tick labels (speed percentages or values) readable
            try:
                import matplotlib.pyplot as _plt2

                _plt2.setp(ax.get_yticklabels(), fontsize=10)
            except Exception:
                pass
            ax.set_title(title, fontsize=10)
            try:
                fig.subplots_adjust(left=0.06, right=0.98, top=0.94, bottom=0.06)
            except Exception:
                pass
            # ensure parent dir exists before saving
            try:
                parent = os.path.dirname(out_path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
            except Exception:
                pass

            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            return True
        except Exception:
            import traceback

            print(
                "[WARNING] Exception while creating windrose PNG:\n",
                traceback.format_exc(),
            )
            return False

    def _save_windrose_pair(
        self,
        i,
        u_col,
        v_col,
        urel_col,
        vrel_col,
        out_u,
        out_rel,
        hide_cardinal_labels=False,
    ):
        """Save both ambient (u_col/v_col) and relative (urel_col/vrel_col) windrose PNGs for index i."""
        saved_u = False
        saved_rel = False

        # ambient/absolute wind (u10/v10)
        try:
            if u_col is not None and v_col is not None and len(u_col) > 0:
                # ambient/absolute wind should always show cardinal labels
                saved_u = self._save_windrose_png(
                    u_col,
                    v_col,
                    out_u,
                    title=f"Pt {i} - ambient",
                    hide_cardinal_labels=False,
                    invert_orientation=True,
                )
        except Exception:
            saved_u = False

        # relative wind (u_rel/v_rel)
        try:
            if urel_col is not None and vrel_col is not None and len(urel_col) > 0:
                saved_rel = self._save_windrose_png(
                    urel_col,
                    vrel_col,
                    out_rel,
                    title=f"Pt {i} - relative",
                    hide_cardinal_labels=hide_cardinal_labels,
                    invert_orientation=False,
                )
        except Exception:
            saved_rel = False

        return saved_u, saved_rel

    def _save_gain_hist_png(self, gains_arr, out_path, title=""):
        """Save a histogram PNG of gains (1D array-like) to out_path."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            gains = np.asarray(gains_arr, dtype=float)
            gains = gains[np.isfinite(gains)]
            # convert to percentage
            gains = gains * 100.0
            if gains.size == 0:
                return False

            fig, ax = plt.subplots(figsize=(4, 3), dpi=150)
            ax.hist(gains, bins=20, color="#2E86AB", edgecolor="white", alpha=0.9)
            ax.set_xlabel("Gain (%)")
            ax.set_ylabel("Frequency")
            # compute mean and std (in percent)
            mean_val = float(np.mean(gains))
            std_val = float(np.std(gains))

            # draw vertical dashed line at mean and shade ±1σ interval
            try:
                ax.axvline(mean_val, color="#222222", linestyle="--", linewidth=1.5)
                low = mean_val - std_val
                high = mean_val + std_val
                ax.axvspan(low, high, color="#222222", alpha=0.12)
            except Exception:
                pass

            # annotate mean and std inside the axes (top-right)
            try:
                ax.text(
                    0.98,
                    0.95,
                    f"μ={mean_val:.2f}%, σ={std_val:.2f}%",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                )
            except Exception:
                pass

            ax.set_title(f"{title} — mean={mean_val:.2f}%", fontsize=10)
            ax.grid(alpha=0.2)
            try:
                fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.12)
            except Exception:
                pass
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            return True
        except Exception:
            return False

    def create_global_map(
        self,
        csv_files,
        ship,
        windrose_folder=None,
        option="outbound",
        rotation=100,
        compute_gains=True,
        per_point_rel_windrose=False,
    ):
        """Create global folium map aggregating many CSV routes.

        Args:
            csv_files: list of csv file paths. Each CSV should contain columns
                ['LAT','LON','u10','v10', ...]. We will use the first half
                (outbound) of each CSV.
            out_map_path: output html file path for the map.
            windrose_folder: optional folder to save windrose PNGs (not required).
        """
        out_map_path = f"global_map_{ship}_{option}_{rotation}.html"
        # Read CSVs and take outbound halves
        dfs = []
        for f in tqdm(csv_files, total=len(csv_files)):
            try:
                df = pd.read_csv(f, index_col=0)
                if option == "outbound":
                    out = df.iloc[: int(len(df) / 2)].reset_index(drop=True)
                else:
                    out = df.iloc[int(len(df) / 2) :].reset_index(drop=True)
                dfs.append(out)
            except Exception as e:
                print(f"[WARNING] Skipping {f}: {e}")

        if len(dfs) == 0:
            raise ValueError("No valid CSVs provided")

        # Number of points we can aggregate: use minimum length across all outbound dfs
        n_points = min(len(d) for d in dfs)
        print(f"[INFO] Aggregating {len(dfs)} routes for {n_points} points each")

        # Use coordinates from first dataframe as marker positions
        ref_df = dfs[0]

        # Create folium map centered on first point
        center = [ref_df.loc[0, "LAT"], ref_df.loc[0, "LON"]]
        self.m = folium.Map(location=center, zoom_start=10)

        # Inject modal html / script (same as in MapPerRoute)
        modal_html = """
        <style>
        #windroseModal { display: none; position: fixed; z-index: 10000; left: 0; top: 0; width: 100%; height: 100%; overflow: auto; background-color: rgba(0,0,0,0.7); }
        /* increased modal content max-width so two windroses display comfortably */
        #windroseModalContent { margin: 3% auto; padding: 20px; width: 90%; max-width: 1600px; background: white; border-radius: 8px; text-align: center; }
        #windroseModalContent img { max-width: 100%; height: auto; }
        #windroseClose { position: absolute; right: 20px; top: 20px; color: white; font-size: 28px; font-weight: bold; cursor: pointer; }
        </style>
        <div id="windroseModal">
            <div id="windroseClose" onclick="document.getElementById('windroseModal').style.display='none'">&times;</div>
            <div id="windroseModalContent"><h3 id="windroseTitle">Windrose</h3><img id="windroseImg" src="" alt="Windrose"></div>
        </div>
        <script>
        function showWindrose(imgSrc, title) { var modal = document.getElementById('windroseModal'); var img = document.getElementById('windroseImg'); var t = document.getElementById('windroseTitle'); img.src = imgSrc; t.innerText = title || 'Windrose'; modal.style.display = 'block'; }
        window.onclick = function(event) { var modal = document.getElementById('windroseModal'); if (event.target == modal) { modal.style.display = 'none'; } }
        </script>
        """
        self.m.get_root().html.add_child(folium.Element(modal_html))

        # Indices to process
        indices = list(range(n_points))

        # Helper for gain images (defined even if compute_gains is False)
        gain_dir = windrose_folder or os.path.join(
            "figures", f"gain_histograms_{ship}_{rotation}"
        )
        os.makedirs(gain_dir, exist_ok=True)

        def _gain_path_for(i):
            return os.path.join(gain_dir, f"gain_pt_{i:04d}.png")

        # If requested, compute gains and prepare histograms + spatial maps
        if compute_gains:
            # Vectorize gains into matrix shape (n_files, n_points)
            gain_mat = np.vstack([d["gain"].values[:n_points] for d in dfs])
            # sanitize: replace non-finite with np.nan
            gain_mat = np.where(np.isfinite(gain_mat), gain_mat, np.nan)

            # Precompute per-point mean gain in percent to keep histogram and
            # popup numbers consistent (mean computed in _save_gain_hist_png
            # multiplies by 100). Use nanmean so points with all-NaN remain NaN.
            try:
                mean_gain_arr = np.nanmean(gain_mat * 100.0, axis=0)
            except Exception:
                mean_gain_arr = np.array([np.nan] * gain_mat.shape[1])

            # Generate histogram PNGs in parallel
            n_jobs = max(1, (os.cpu_count() or 2) - 1)
            print(
                f"[INFO] Generating {len(indices)} gain histograms using {n_jobs} jobs into {gain_dir}"
            )

            Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(self._save_gain_hist_png)(
                    gain_mat[:, i], _gain_path_for(i), title=f"Gain Pt {i}"
                )
                for i in tqdm(indices)
            )

            # Also produce spatial summary maps (std, probability >=10%, and count heatmap)
            try:
                self.create_spatial_maps(
                    gain_mat, ref_df, out_dir=gain_dir, option=option
                )
            except Exception as e:
                print(f"[WARNING] create_spatial_maps failed: {e}")
        else:
            # placeholder mean_gain_arr (all NaN) when not computing gains
            mean_gain_arr = np.array([np.nan] * n_points)

        # If requested, generate relative windrose PNGs per point (one image per index)
        rel_dir = None
        if per_point_rel_windrose:
            rel_dir = windrose_folder or os.path.join(
                "figures", f"windroses_{ship}_{rotation}"
            )
            os.makedirs(rel_dir, exist_ok=True)

            def _rel_path_for(i):
                return os.path.join(rel_dir, f"windrose_pt_{i:04d}_rel.png")

            # compute global vmax across all selected dfs so bins are consistent
            global_max = 0.0
            for d in dfs:
                try:
                    if "u_rel" in d.columns and "v_rel" in d.columns:
                        arr = np.sqrt(
                            d["u_rel"].to_numpy(dtype=float) ** 2
                            + d["v_rel"].to_numpy(dtype=float) ** 2
                        )
                    elif "u10" in d.columns and "v10" in d.columns:
                        arr = np.sqrt(
                            d["u10"].to_numpy(dtype=float) ** 2
                            + d["v10"].to_numpy(dtype=float) ** 2
                        )
                    else:
                        continue
                    mask = np.isfinite(arr)
                    if np.any(mask):
                        m = float(np.nanmax(arr[mask]))
                        if m > global_max:
                            global_max = m
                except Exception:
                    continue

            # fallback default
            if global_max <= 0.0:
                global_max = 12.0

            # number of speed bins (edges) — keeps the same intervals for all windroses
            num_bins = 6
            try:
                # create bin edges from 0..global_max
                bins = list(np.linspace(0.0, float(global_max), num_bins + 1))
            except Exception:
                bins = None

            # expose bins to _save_windrose_png via a globals slot so the saver
            # can pick them up without changing many call sites.
            try:
                globals()["_GLOBAL_WINDROSE_BINS"] = bins
            except Exception:
                pass

            print(
                f"[INFO] Generating relative windrose PNGs into {rel_dir} (bins={bins})"
            )
            saved_points = 0
            saved_rel_count = 0
            saved_abs_count = 0
            for i in tqdm(indices):
                # collect both relative (u_rel/v_rel) and ambient (u10/v10) across all dfs for point i
                u_rel_list = []
                v_rel_list = []
                u_abs_list = []
                v_abs_list = []
                for d in dfs:
                    if i < len(d):
                        # relative first
                        if "u_rel" in d.columns and "v_rel" in d.columns:
                            try:
                                val_u = d.iloc[i]["u_rel"]
                                val_v = d.iloc[i]["v_rel"]
                                if np.isfinite(val_u) and np.isfinite(val_v):
                                    u_rel_list.append(val_u)
                                    v_rel_list.append(val_v)
                            except Exception:
                                pass
                        # ambient / absolute wind
                        if "u10" in d.columns and "v10" in d.columns:
                            try:
                                val_u = d.iloc[i]["u10"]
                                val_v = d.iloc[i]["v10"]
                                if np.isfinite(val_u) and np.isfinite(val_v):
                                    u_abs_list.append(val_u)
                                    v_abs_list.append(val_v)
                            except Exception:
                                pass

                # skip if neither has samples
                if len(u_rel_list) == 0 and len(u_abs_list) == 0:
                    continue

                out_rel = _rel_path_for(i)
                out_abs = os.path.join(rel_dir, f"windrose_pt_{i:04d}_abs.png")

                try:
                    # ambient windrose should always show cardinal direction labels;
                    # hide_cardinal_labels applies only to the relative windrose
                    saved_u, saved_rel = self._save_windrose_pair(
                        i,
                        u_abs_list,
                        v_abs_list,
                        u_rel_list,
                        v_rel_list,
                        out_abs,
                        out_rel,
                        hide_cardinal_labels=True,
                    )
                    if saved_u:
                        saved_abs_count += 1
                    if saved_rel:
                        saved_rel_count += 1
                    if saved_u or saved_rel:
                        saved_points += 1
                except Exception:
                    # ignore failures per point
                    continue

            try:
                print(
                    f"[INFO] Windrose PNGs generated: {saved_points}/{len(indices)} points — rel={saved_rel_count}, abs={saved_abs_count} into {rel_dir}"
                )
            except Exception:
                pass

        # Add markers referencing saved images (use relative path from out_map_path dir)
        for i in indices:
            # choose image path: relative windrose preferred, else gain histogram if available
            img_path = None
            img_title = ""
            # prefer showing both relative and ambient windrose thumbnails when available
            if per_point_rel_windrose and rel_dir is not None:
                rel_candidate = os.path.join(rel_dir, f"windrose_pt_{i:04d}_rel.png")
                abs_candidate = os.path.join(rel_dir, f"windrose_pt_{i:04d}_abs.png")
                has_rel = os.path.exists(rel_candidate)
                has_abs = os.path.exists(abs_candidate)
            else:
                has_rel = has_abs = False

            img_html = ""
            if has_rel or has_abs:
                rel_rel = (
                    os.path.relpath(
                        rel_candidate, start=os.path.dirname(out_map_path) or "."
                    ).replace("\\", "/")
                    if has_rel
                    else None
                )
                rel_abs = (
                    os.path.relpath(
                        abs_candidate, start=os.path.dirname(out_map_path) or "."
                    ).replace("\\", "/")
                    if has_abs
                    else None
                )

                # two thumbnails side-by-side when both exist
                if has_rel and has_abs:
                    img_html = (
                        f'<div style="display:flex; gap:16px; justify-content:center; align-items:flex-start; margin-top:8px;">'
                        f"<a href=\"javascript:void(0)\" onclick=\"showWindrose('{rel_rel}','Pt {i} - relative')\">"
                        f'<img src="{rel_rel}" alt="Pt {i} - relative" style="max-width:420px; height:auto; border:1px solid #ddd; border-radius:4px;"/>'
                        f"</a>"
                        f"<a href=\"javascript:void(0)\" onclick=\"showWindrose('{rel_abs}','Pt {i} - ambient')\">"
                        f'<img src="{rel_abs}" alt="Pt {i} - ambient" style="max-width:420px; height:auto; border:1px solid #ddd; border-radius:4px;"/>'
                        f"</a></div>"
                    )
                else:
                    # single thumbnail
                    chosen = rel_rel if has_rel else rel_abs
                    title = f"Pt {i} - relative" if has_rel else f"Pt {i} - ambient"
                    img_html = (
                        f'<div style="text-align:center; margin-top:8px;">'
                        f"<a href=\"javascript:void(0)\" onclick=\"showWindrose('{chosen}','{title}')\">"
                        f'<img src="{chosen}" alt="{title}" style="max-width:780px; height:auto; border:1px solid #ddd; border-radius:4px;"/>'
                        f"</a></div>"
                    )
            else:
                # fallback: show gain histogram if available
                if compute_gains:
                    candidate = _gain_path_for(i)
                    if os.path.exists(candidate):
                        rel = os.path.relpath(
                            candidate, start=os.path.dirname(out_map_path) or "."
                        ).replace("\\", "/")
                        img_html = (
                            f'<div style="text-align:center; margin-top:6px;">'
                            f"<a href=\"javascript:void(0)\" onclick=\"showWindrose('{rel}','Gain Pt {i}')\">"
                            f'<img src="{rel}" alt="Gain Pt {i}" style="max-width:780px; height:auto; border:1px solid #ddd; border-radius:4px;"/>'
                            f"</a></div>"
                        )
                    else:
                        img_html = "<div style='text-align:center; color:#888; margin-top:6px;'>Imagem não disponível</div>"
                else:
                    img_html = "<div style='text-align:center; color:#888; margin-top:6px;'>Imagem não disponível</div>"

            # Marker location from reference df
            try:
                lat_i = float(ref_df.loc[i, "LAT"])
                lon_i = float(ref_df.loc[i, "LON"])
            except Exception:
                continue

            # Use precomputed mean gain (percent) if available
            try:
                mean_gain = float(mean_gain_arr[i])
            except Exception:
                mean_gain = float("nan")

            if np.isfinite(mean_gain):
                if mean_gain < 0.0:
                    mcolor = "#7e3fb2"
                elif mean_gain < 5.0:
                    mcolor = "#ff4d4d"
                elif mean_gain < 10.0:
                    mcolor = "#ff8c1a"
                elif mean_gain < 15.0:
                    mcolor = "#ffd24d"
                else:
                    mcolor = "#3ddc84"
                mean_line = f'<div style="margin-top:6px; font-weight:bold;">Mean gain: {mean_gain:.2f}%</div>'
            else:
                mcolor = "#888888"
                mean_line = (
                    '<div style="margin-top:6px; color:#888;">Mean gain: N/A</div>'
                )

            popup_html = (
                f"<div style='font-family: Arial; font-size:12px; width:1100px;'><b>Point {i}</b><br>"
                f"Lat: {lat_i:.4f} Lon: {lon_i:.4f}<br>"
                f"{mean_line}"
                f"{img_html}</div>"
            )

            folium.CircleMarker(
                location=[lat_i, lon_i],
                radius=6,
                color=mcolor,
                fill=True,
                fillColor=mcolor,
                fillOpacity=0.9,
                popup=folium.Popup(popup_html, max_width=900),
            ).add_to(self.m)

        # Add JS to adjust marker radius dynamically with zoom (makes points smaller when zoomed out)
        zoom_js = """
        <script>
        (function() {
            // find a Leaflet map instance on the page
            var map = Object.values(window).filter(function(v){return v instanceof L.Map;})[0];
            if(!map) return;
            function adjustMarkerRadius(){
                var z = map.getZoom();
                // radius scales with zoom; lower zoom -> smaller radius
                var r = Math.max(2, Math.round(z / 1.5));
                map.eachLayer(function(layer){
                    if(layer instanceof L.CircleMarker){
                        try{ layer.setRadius(r); }catch(e){}
                    }
                });
            }
            map.on('zoomend', adjustMarkerRadius);
            // run once after markers are added
            setTimeout(adjustMarkerRadius, 200);
        })();
        </script>
        """
        self.m.get_root().html.add_child(folium.Element(zoom_js))

        # Ensure output directory exists
        out_dir = os.path.dirname(out_map_path) or "."
        os.makedirs(out_dir, exist_ok=True)
        self.m.save(os.path.join(out_map_path))
        print(f"[INFO] Global map saved to {out_map_path}")

    def _colormap_hex(self, values, cmap_name="viridis", vmin=None, vmax=None):
        """Return hex colors for a 1D array of values using a matplotlib colormap.

        Non-finite values are mapped to a neutral grey `#888888`.
        """
        import matplotlib

        vals = np.asarray(values, dtype=float)
        # if all nan, return grey for all
        if np.all(~np.isfinite(vals)):
            return ["#888888"] * len(vals)

        if vmin is None:
            # ignore nan
            vmin = float(np.nanmin(vals))
        if vmax is None:
            vmax = float(np.nanmax(vals))

        try:
            cmap = matplotlib.colormaps.get_cmap(cmap_name)
        except Exception:
            # fallback for older matplotlib versions
            cmap = matplotlib.cm.get_cmap(cmap_name)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        hexes = []
        for v in vals:
            if not np.isfinite(v):
                hexes.append("#888888")
            else:
                rgba = cmap(norm(v))
                hexes.append(matplotlib.colors.to_hex(rgba))
        return hexes

    def create_spatial_maps(self, gain_mat, ref_df, out_dir=None, option="outbound"):
        """Create spatial visualizations per point:
        - standard deviation map of gain (uncertainty),
        - probability map (fraction of samples with gain >= 10%),
        - heatmap/contour of sample counts per point (coverage).

        Args:
            gain_mat: 2D array-like with shape (n_files, n_points) of gain values (linear, e.g. 0.12 for 12%).
            ref_df: reference dataframe with at least columns ['LAT','LON'] and length >= n_points.
            out_dir: directory to save HTML maps and CSV summary. Defaults to 'figures/spatial'.
            option: string used for filenames (e.g. 'outbound' or 'return').
        Produces:
            - HTML maps saved under out_dir
            - CSV summary `point_stats_{option}.csv` with mean/std/prob/count per point
        """
        import matplotlib
        from folium.plugins import HeatMap

        if out_dir is None:
            out_dir = os.path.join("figures", "spatial")
        os.makedirs(out_dir, exist_ok=True)

        arr = np.asarray(gain_mat, dtype=float)
        # compute per-point statistics ignoring nan
        mean_vals = np.nanmean(arr, axis=0)  # linear
        std_vals = np.nanstd(arr, axis=0)
        # probability of gain >= 10% (0.10)
        prob_vals = np.nanmean(arr >= 0.10, axis=0)
        counts = np.sum(np.isfinite(arr), axis=0)

        # convert to percent where useful
        mean_pct = mean_vals * 100.0
        std_pct = std_vals * 100.0
        prob_pct = prob_vals * 100.0

        n_points = mean_vals.shape[0]

        # Build CSV summary
        csv_rows = []
        for i in range(n_points):
            lat_i = float(ref_df.loc[i, "LAT"])
            lon_i = float(ref_df.loc[i, "LON"])
            csv_rows.append(
                {
                    "idx": i,
                    "lat": lat_i,
                    "lon": lon_i,
                    "mean_gain_pct": (
                        None if not np.isfinite(mean_pct[i]) else float(mean_pct[i])
                    ),
                    "std_gain_pct": (
                        None if not np.isfinite(std_pct[i]) else float(std_pct[i])
                    ),
                    "prob_gain_ge_10pct": (
                        None if not np.isfinite(prob_pct[i]) else float(prob_pct[i])
                    ),
                    "n_samples": int(counts[i]) if np.isfinite(counts[i]) else 0,
                }
            )

        stats_df = pd.DataFrame(csv_rows)
        stats_csv = os.path.join(out_dir, f"point_stats_{option}.csv")
        stats_df.to_csv(stats_csv, index=False)
        print(f"[INFO] Saved per-point summary to {stats_csv}")

        # === STD map ===
        m_std = folium.Map(
            location=[float(ref_df.loc[0, "LAT"]), float(ref_df.loc[0, "LON"])],
            zoom_start=9,
        )
        std_colors = self._colormap_hex(std_pct, cmap_name="plasma")
        for i in range(n_points):
            try:
                lat_i = float(ref_df.loc[i, "LAT"])
                lon_i = float(ref_df.loc[i, "LON"])
            except Exception:
                continue
            val = std_pct[i]
            col = std_colors[i]
            popup = (
                f"<b>Point {i}</b><br>Std gain: {('N/A' if not np.isfinite(val) else f'{val:.2f}%')}"
                + f"<br>Mean gain: {('N/A' if not np.isfinite(mean_pct[i]) else f'{mean_pct[i]:.2f}%')}"
                + f"<br>Samples: {int(counts[i])}"
            )
            folium.CircleMarker(
                location=[lat_i, lon_i],
                radius=5,
                color=col,
                fill=True,
                fillColor=col,
                fillOpacity=0.9,
                popup=folium.Popup(popup, max_width=300),
            ).add_to(m_std)

        # Add a colorbar legend showing min/max std (percent) in the corner
        try:
            finite_mask = np.isfinite(std_pct)
            if np.any(finite_mask):
                vmin = float(np.nanmin(std_pct[finite_mask]))
                vmax = float(np.nanmax(std_pct[finite_mask]))
            else:
                vmin = 0.0
                vmax = 0.0

            # avoid degenerate colorbar
            if vmax <= vmin:
                vmax = vmin + 1e-6

            # create a vertical colorbar image from the colormap
            import io, base64

            try:
                cmap = matplotlib.colormaps.get_cmap("plasma")
            except Exception:
                cmap = matplotlib.cm.get_cmap("plasma")
            gradient = np.linspace(0, 1, 256)[:, None]
            img = cmap(gradient)

            fig_cb = plt.figure(figsize=(0.5, 3.5), dpi=100)
            ax_cb = fig_cb.add_axes([0, 0, 1, 1])
            ax_cb.imshow(img, aspect="auto")
            ax_cb.set_axis_off()
            buf = io.BytesIO()
            fig_cb.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
            plt.close(fig_cb)
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode()

            legend_html = f"""
            <div style="position: fixed; bottom: 12px; left: 12px; z-index:9999; background: rgba(255,255,255,0.95); padding:8px; border-radius:6px; box-shadow:0 2px 6px rgba(0,0,0,0.3); font-family: Arial; font-size:12px;">
                <div style="font-weight:bold; margin-bottom:4px;">Std gain (%)</div>
                <div style="display:flex; align-items:center;">
                    <img src="data:image/png;base64,{img_b64}" style="height:140px; width:22px; display:block; margin-right:8px;"/>
                    <div style="display:flex; flex-direction:column; justify-content:space-between; height:140px;">
                        <span style="font-size:11px;">{vmax:.2f}%</span>
                        <span style="font-size:11px;">{vmin:.2f}%</span>
                    </div>
                </div>
            </div>
            """
            m_std.get_root().html.add_child(folium.Element(legend_html))
        except Exception as e:
            print(f"[WARNING] Could not add colorbar legend: {e}")

        std_path = os.path.join(out_dir, f"std_map_{option}.html")
        m_std.save(std_path)
        print(f"[INFO] Saved std deviation map to {std_path}")

        # === Probability map (gain >= 10%) ===
        m_prob = folium.Map(
            location=[float(ref_df.loc[0, "LAT"]), float(ref_df.loc[0, "LON"])],
            zoom_start=9,
        )
        prob_colors = self._colormap_hex(prob_pct, cmap_name="YlOrRd")
        for i in range(n_points):
            try:
                lat_i = float(ref_df.loc[i, "LAT"])
                lon_i = float(ref_df.loc[i, "LON"])
            except Exception:
                continue
            val = prob_pct[i]
            col = prob_colors[i]
            popup = (
                f"<b>Point {i}</b><br>Pr(gain ≥ 10%): {('N/A' if not np.isfinite(val) else f'{val:.1f}%')}"
                + f"<br>Mean gain: {('N/A' if not np.isfinite(mean_pct[i]) else f'{mean_pct[i]:.2f}%')}"
                + f"<br>Samples: {int(counts[i])}"
            )
            folium.CircleMarker(
                location=[lat_i, lon_i],
                radius=5,
                color=col,
                fill=True,
                fillColor=col,
                fillOpacity=0.9,
                popup=folium.Popup(popup, max_width=300),
            ).add_to(m_prob)

        prob_path = os.path.join(out_dir, f"prob_map_{option}.html")
        m_prob.save(prob_path)
        print(f"[INFO] Saved probability map to {prob_path}")

        # === Count heatmap ===
        m_count = folium.Map(
            location=[float(ref_df.loc[0, "LAT"]), float(ref_df.loc[0, "LON"])],
            zoom_start=9,
        )
        heat_data = []
        for i in range(n_points):
            try:
                lat_i = float(ref_df.loc[i, "LAT"])
                lon_i = float(ref_df.loc[i, "LON"])
            except Exception:
                continue
            w = int(counts[i])
            if w > 0:
                heat_data.append([lat_i, lon_i, w])

        try:
            HeatMap(heat_data, radius=9, blur=10, max_zoom=13).add_to(m_count)
        except Exception:
            # fallback: add circle markers sized by count
            max_c = max([c for c in counts if c > 0] + [1])
            for i in range(n_points):
                try:
                    lat_i = float(ref_df.loc[i, "LAT"])
                    lon_i = float(ref_df.loc[i, "LON"])
                except Exception:
                    continue
                w = int(counts[i])
                if w <= 0:
                    continue
                r = max(3, int(8 * (w / max_c)))
                popup = f"<b>Point {i}</b><br>Samples: {w}<br>Mean gain: {('N/A' if not np.isfinite(mean_pct[i]) else f'{mean_pct[i]:.2f}%')}"
                folium.CircleMarker(
                    location=[lat_i, lon_i],
                    radius=r,
                    color="#2E86AB",
                    fill=True,
                    fillColor="#2E86AB",
                    fillOpacity=0.6,
                    popup=folium.Popup(popup, max_width=300),
                ).add_to(m_count)

        count_path = os.path.join(out_dir, f"count_heatmap_{option}.html")
        m_count.save(count_path)
        print(f"[INFO] Saved count heatmap to {count_path}")


def calc_mean_gain_parallel(csv_files, n_jobs=-1):
    print(f"[INFO] Calculating gain from {len(csv_files)} routes using {n_jobs} cores")
    # Parallel processing with progress bar
    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(process_single_trip)(trip)
        for trip in tqdm(csv_files, desc="Processing routes")
    )

    return results


def get_windrose_from_route(route_df, output_name="route", ax=None):
    """
    Create windrose from route data with u10, v10 columns

    Parameters:
    route_df: DataFrame with columns ['LAT', 'LON', 'u10', 'v10']
    output_name: name for output file
    ax: matplotlib axis to plot on (for subplots)
    """
    # Extract u10 and v10 from the route dataframe
    u10 = route_df["u10"].values
    v10 = route_df["v10"].values

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

    ax.bar(wind_direction, wind_speed, normed=True, opening=0.8, edgecolor="white")
    ax.set_legend(fontsize=12, title_fontsize=14)
    ax.set_title(f"{output_name}", fontsize=20)

    return wind_speed, wind_direction, ax


def process_single_trip(trip_file):
    """Process a single trip file and return the mean gain"""
    df_trip = pd.read_csv(trip_file, index_col=0)

    filename = os.path.basename(trip_file)
    pattern = (
        r"wind_data_year_(\d{4})_month_(\d{1,2})_day_(\d{1,2})_hour_(\d{1,2})\.csv"
    )
    match = re.search(pattern, filename)

    if match:
        year, month, day, hour = map(int, match.groups())
        timestamp = pd.Timestamp(year=year, month=month, day=day, hour=hour)
    else:
        print(f"[WARNING] Could not extract timestamp from {filename}")
        timestamp = pd.NaT  # Not a Time

    return {"gain": df_trip["gain"].mean(), "timestamp": timestamp}


def main():

    parser = argparse.ArgumentParser(description="Wind Route Creator")
    parser.add_argument("-s", "--ship", help="afra or suez")
    parser.add_argument("--rotation", required=True, help="100 or 180")
    args = parser.parse_args()
    ship = "abdias_suez" if args.ship == "suez" else "castro_alves_afra"
    routes_csv_path = f"D:/{ship}/route_csvs{int(args.rotation)}"
    csv_files = glob.glob(os.path.join(routes_csv_path, "*.csv"))
    global_map = GlobalMap()
    global_map.create_global_map(
        csv_files,
        ship,
        option="outbound",
        rotation=int(args.rotation),
        compute_gains=False,
        per_point_rel_windrose=True,
    )
    global_map.create_global_map(
        csv_files,
        ship,
        option="return",
        rotation=int(args.rotation),
        compute_gains=False,
        per_point_rel_windrose=True,
    )

    return


if __name__ == "__main__":
    main()
