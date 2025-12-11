import pandas as pd
import numpy as np
import os
import folium
import matplotlib.pyplot as plt
from windrose import WindroseAxes
import glob
import re
from tqdm import tqdm
import argparse
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import random


class GlobalMap:
    def __init__(self):
        """Global map that aggregates wind data across many route CSVs and
        creates a folium map where each marker shows a windrose built from
        the u10/v10 values of all CSVs for that route index.
        """
        self.m = None

    def _save_windrose_pair(
        self,
        i,
        u_col,
        v_col,
        urel_col,
        vrel_col,
        out,
        center_image_path=None,
        center_image_zoom=0.18,
        center_image_alpha=0.85,
    ):
        """Save both ambient (u_col/v_col) and relative (urel_col/vrel_col) windrose PNGs for index i."""
        saved_u = False
        saved_rel = False

        try:
            # prepare arrays
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            u_abs = (
                np.asarray(u_col, dtype=float) if u_col is not None else np.array([])
            )
            v_abs = (
                np.asarray(v_col, dtype=float) if v_col is not None else np.array([])
            )
            u_rel = (
                np.asarray(urel_col, dtype=float)
                if urel_col is not None
                else np.array([])
            )
            v_rel = (
                np.asarray(vrel_col, dtype=float)
                if vrel_col is not None
                else np.array([])
            )

            mask_abs = np.isfinite(u_abs) & np.isfinite(v_abs)
            mask_rel = np.isfinite(u_rel) & np.isfinite(v_rel)
            u_abs = u_abs[mask_abs]
            v_abs = v_abs[mask_abs]
            u_rel = u_rel[mask_rel]
            v_rel = v_rel[mask_rel]

            # compute speeds and dirs
            s_abs = np.sqrt(u_abs**2 + v_abs**2) if u_abs.size > 0 else np.array([])
            d_abs = (
                (np.degrees(np.arctan2(-u_abs, -v_abs)) % 360)
                if u_abs.size > 0
                else np.array([])
            )
            s_rel = np.sqrt(u_rel**2 + v_rel**2) if u_rel.size > 0 else np.array([])
            d_rel = (
                (np.degrees(np.arctan2(-u_rel, -v_rel)) % 360)
                if u_rel.size > 0
                else np.array([])
            )

            # create combined fig
            fig = plt.figure(figsize=(20, 5), dpi=300)
            # top-left: relative windrose
            ax1 = fig.add_subplot(1, 4, 1, projection="windrose")
            # top-right: absolute windrose
            ax2 = fig.add_subplot(1, 4, 2, projection="windrose")
            # bottom-left: histogram u
            ax3 = fig.add_subplot(1, 4, 3)
            # bottom-left: histogram v
            ax4 = fig.add_subplot(1, 4, 4)

            # reuse global bins if available
            bins = globals().get("_GLOBAL_WINDROSE_BINS", None)

            if u_rel.size > 0:
                if bins is not None:
                    ax1.bar(
                        d_rel,
                        s_rel,
                        bins=bins,
                        normed=True,
                        opening=0.8,
                        edgecolor="white",
                    )
                else:
                    ax1.bar(d_rel, s_rel, normed=True, opening=0.8, edgecolor="white")
                # hide cardinal labels on relative
                try:
                    import matplotlib.pyplot as _plt

                    _plt.setp(ax1.get_xticklabels(), visible=False)
                except Exception:
                    try:
                        ax1.set_xticklabels([])
                    except Exception:
                        raise
                ax1.set_title(f"Pt {i} - relative", fontsize=10)
                try:
                    ax1.set_legend(fontsize=8, title_fontsize=10)
                except Exception:
                    try:
                        ax1.legend(fontsize=8)
                    except Exception:
                        raise
                # overlay image on relative windrose if provided
                if center_image_path and os.path.exists(center_image_path):
                    try:
                        img = plt.imread(center_image_path)
                        oi = OffsetImage(img, zoom=float(center_image_zoom))
                        try:
                            oi.set_alpha(float(center_image_alpha))
                        except Exception:
                            raise
                        ab = AnnotationBbox(
                            oi,
                            (0.5, 0.5),
                            frameon=False,
                            xycoords="axes fraction",
                            box_alignment=(0.5, 0.5),
                        )
                        ax1.add_artist(ab)
                    except Exception:
                        raise

            else:
                ax1.text(0.5, 0.5, "No relative samples", ha="center", va="center")

            if u_abs.size > 0:
                if bins is not None:
                    ax2.bar(
                        d_abs,
                        s_abs,
                        bins=bins,
                        normed=True,
                        opening=0.8,
                        edgecolor="white",
                    )
                else:
                    ax2.bar(d_abs, s_abs, normed=True, opening=0.8, edgecolor="white")
                ax2.set_title(f"Pt {i} - ambient", fontsize=10)
                try:
                    ax2.set_legend(fontsize=8, title_fontsize=10)
                except Exception:
                    try:
                        ax2.legend(fontsize=8)
                    except Exception:
                        raise
            else:
                ax2.text(0.5, 0.5, "No ambient samples", ha="center", va="center")
            if u_rel.size > 0:
                ax3.hist(
                    u_rel,
                    bins=20,
                    color="red",
                    edgecolor="white",
                    alpha=0.5,
                    label="Relative Speed (u)",
                )
                ax3.hist(
                    u_abs,
                    bins=20,
                    color="green",
                    edgecolor="white",
                    alpha=0.5,
                    label="Wind Speed (u)",
                )
                ax3.set_title("Horizontal speed distribution")
                ax3.set_xlabel("Speed (m/s)")
                ax3.legend(fontsize=8)
            else:
                ax3.text(0.5, 0.5, "No relative samples", ha="center", va="center")
            if v_rel.size > 0:
                ax4.hist(
                    v_rel,
                    bins=20,
                    color="red",
                    edgecolor="white",
                    alpha=0.5,
                    label="Relative Speed (v)",
                )
                ax4.hist(
                    v_abs,
                    bins=20,
                    color="green",
                    edgecolor="white",
                    alpha=0.5,
                    label="Wind Speed (v)",
                )
                ax4.set_title("Vertical speed distribution")
                ax4.set_xlabel("Speed (m/s)")
                ax4.legend(fontsize=8)
            else:
                ax4.text(0.5, 0.5, "No relative samples", ha="center", va="center")

            try:
                fig.subplots_adjust(hspace=0.35, wspace=0.25)
            except Exception:
                raise
            if out:
                try:
                    parent = os.path.dirname(out)
                    if parent:
                        os.makedirs(parent, exist_ok=True)
                    fig.savefig(out, bbox_inches="tight")
                except Exception:
                    raise
            plt.close(fig)
        except Exception:
            print("[ERROR] Error to create figures")

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
        per_point_rel_windrose=False,
        center_image_path=None,
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

        # If no center image path provided, attempt to auto-detect the user's
        # casco silhouette in the repo (common location used by the user).
        if center_image_path is None:
            candidate = r"C:\Users\jung_\OneDrive\Documentos\Poli\TPN\CENPES Descarbonização\casco.png"
            try:
                if os.path.exists(candidate):
                    center_image_path = candidate
            except Exception:
                center_image_path = None

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
        /* modal content: responsive width and a max-width to control very large screens; allow vertical scrolling for tall images */
        #windroseModalContent { margin: 3% auto; padding: 20px; width: 90%; max-width: 1400px; background: white; border-radius: 8px; text-align: center; overflow: auto; }
        /* ensure image scales but doesn't overflow viewport vertically */
        #windroseModalContent img { max-width: 100%; max-height: 75vh; height: auto; }
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

        # Helper for gain images (defined even if compute_gains is False).
        # Create separate folders for outbound/return by including `option` in the path.
        base_fig_dir = windrose_folder if windrose_folder else "figures"
        gain_dir = os.path.join(
            base_fig_dir, f"gain_histograms_{ship}_{rotation}_{option}"
        )
        os.makedirs(gain_dir, exist_ok=True)

        # If requested, generate relative windrose PNGs per point (one image per index)
        image_dir = None
        if per_point_rel_windrose:
            # Save windroses in a folder that includes ship, rotation and option
            base_fig_dir = windrose_folder if windrose_folder else "figures"
            image_dir = os.path.join(
                base_fig_dir, f"windroses_{ship}_{rotation}_{option}"
            )
            os.makedirs(image_dir, exist_ok=True)

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

            # can pick them up without changing many call sites.
            try:
                globals()["_GLOBAL_WINDROSE_BINS"] = bins
            except Exception:
                pass

            print(
                f"[INFO] Generating relative windrose PNGs into {image_dir} (bins={bins})"
            )
            saved_points = 0
            saved_rel_count = 0
            saved_abs_count = 0
            for i in tqdm(indices):
                u_rel_list = []
                v_rel_list = []
                u_abs_list = []
                v_abs_list = []
                for d in dfs:
                    if i < len(d):
                        # relative
                        if "u_rel" in d.columns and "v_rel" in d.columns:
                            try:
                                val_u = d.iloc[i]["u_rel"]
                                val_v = d.iloc[i]["v_rel"]
                                if np.isfinite(val_u) and np.isfinite(val_v):
                                    u_rel_list.append(val_u)
                                    v_rel_list.append(val_v)
                            except Exception:
                                print("[ERROR] Relative speed error")
                        # ambient / absolute wind
                        if "u10" in d.columns and "v10" in d.columns:
                            try:
                                val_u = d.iloc[i]["u10"]
                                val_v = d.iloc[i]["v10"]
                                if np.isfinite(val_u) and np.isfinite(val_v):
                                    u_abs_list.append(val_u)
                                    v_abs_list.append(val_v)
                            except Exception:
                                print("[ERROR] Wind speed error")
                # skip if neither has samples
                if len(u_rel_list) == 0 and len(u_abs_list) == 0:
                    continue

                output = os.path.join(image_dir, f"windrose_pt_{i:04d}.png")
                try:
                    saved_u, saved_rel = self._save_windrose_pair(
                        i,
                        u_abs_list,
                        v_abs_list,
                        u_rel_list,
                        v_rel_list,
                        output,
                        center_image_path=center_image_path,
                        center_image_zoom=0.1,
                    )
                    if saved_u:
                        saved_abs_count += 1
                    if saved_rel:
                        saved_rel_count += 1
                    if saved_u or saved_rel:
                        saved_points += 1
                except Exception:
                    raise

            try:
                print(
                    f"[INFO] Windrose PNGs generated: {saved_points}/{len(indices)} points — rel={saved_rel_count}, abs={saved_abs_count} into {image_dir}"
                )
            except Exception:
                pass

        # Add markers referencing saved images (use relative path from out_map_path dir)
        for i in indices:
            if per_point_rel_windrose and image_dir is not None:
                img_candidate = os.path.join(image_dir, f"windrose_pt_{i:04d}.png")
                has_img = os.path.exists(img_candidate)
            else:
                has_img = False

            img_html = ""
            if has_img:
                relative_img_path = (
                    os.path.relpath(
                        img_candidate, start=os.path.dirname(out_map_path) or "."
                    ).replace("\\", "/")
                    if has_img
                    else None
                )

                # two thumbnails side-by-side when both exist
                if has_img:
                    img_html = (
                        f'<div style="display:flex; gap:16px; justify-content:center; align-items:flex-start; margin-top:8px;">'
                        f"<a href=\"javascript:void(0)\" onclick=\"showWindrose('{relative_img_path}','Pt {i} - relative')\">"
                        f'<img src="{relative_img_path}" alt="Pt {i} - relative" style="max-width:1400px; max-height:80vh; height:auto; border:1px solid #ddd; border-radius:4px;"/>'
                        f"</a>"
                    )
            # Marker location from reference df
            try:
                lat_i = float(ref_df.loc[i, "LAT"])
                lon_i = float(ref_df.loc[i, "LON"])
            except Exception:
                continue

            mcolor = "#888888"
            mean_line = '<div style="margin-top:6px; color:#888;">Mean gain: N/A</div>'

            popup_html = (
                f"<div style='font-family: Arial; font-size:12px; width:90%; max-width:1400px; margin:0 auto;'><b>Point {i}</b><br>"
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
                popup=folium.Popup(popup_html, max_width=1400),
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
    parser.add_argument("--path", required=True, help="outbound or return")

    args = parser.parse_args()
    ship = "abdias_suez" if args.ship == "suez" else "castro_alves_afra"
    routes_csv_path = f"D:/{ship}/route_csvs{int(args.rotation)}"
    csv_files = glob.glob(os.path.join(routes_csv_path, "*.csv"))
    random.shuffle(csv_files)
    global_map = GlobalMap()
    global_map.create_global_map(
        csv_files,
        ship,
        option=args.path,
        rotation=int(args.rotation),
        per_point_rel_windrose=True,
    )

    return


if __name__ == "__main__":
    main()
