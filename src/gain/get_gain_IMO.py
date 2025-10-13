import pandas as pd
import numpy as np
import bisect
import argparse
import matplotlib.pyplot as plt
import os

class GainIMO:
    def __init__(self, imo_data_path, forces_data_path, rotation, draft, modified):
        self.imo_df = pd.read_csv(imo_data_path)
        if modified:
            # garantir dtype float nas colunas numéricas para evitar FutureWarning
            try:
                self.imo_df = self.imo_df.astype(float)
            except Exception:
                # se existirem colunas não-numéricas, converte apenas as numéricas
                num_cols = self.imo_df.select_dtypes(include=['number']).columns
                self.imo_df[num_cols] = self.imo_df[num_cols].astype(float)

            # shift all rows down by 3 (V=1 -> V=4), fill top 3 rows with zeros
            if len(self.imo_df) > 3:
                shifted = pd.DataFrame(
                    0.0,
                    index=self.imo_df.index,
                    columns=self.imo_df.columns,
                    dtype=float
                )
                # usar to_numpy(dtype=float) para garantir compatibilidade de tipos
                shifted.iloc[3:, :] = self.imo_df.iloc[:-3, :].to_numpy(dtype=float)
                self.imo_df = shifted
            else:
                # not enough rows -> zero numeric columns
                self.imo_df.loc[:, :] = 0.0

            print("[INFO] imo_df shifted +3 rows (first 3 rows set to 0) due to modified=True")

        self.thrust_df = pd.read_csv(forces_data_path)
        self.thrust_df["Angulo"] = np.where(
        self.thrust_df["Angulo"] <= 180, 
        180 - self.thrust_df["Angulo"], 
        540 - self.thrust_df["Angulo"]
        ) % 360
        self.moments = (
            self.thrust_df["Mz_popa_bom"]
            + self.thrust_df["Mz_popa_bor"]
            + self.thrust_df["Mz_proa_bom"]
            + self.thrust_df["Mz_proa_bor"]
        )
        angle_list = self.thrust_df["Angulo"].unique()
        self.angle_list = np.append(angle_list, 360)

        # 1. Define V_ref (ship velocity)
        self.V_ship = 15 * 0.5144  # knots to m/s
        self.eta_D = 0.7
        self.RT = 744 # kN
        self.draft = float(draft)  # meters
        self.rotation = int(rotation)  # RPM
        self.Ax = 1130
        self.Ay = 3300
        self.P_break = 6560  # kW
        self.P_eff = 4592  # kW
    
    def load_power_matrices_comparison(self, ship):
        """Load power matrices for both 100 and 180 RPM"""
        
        # Carregar dados para 100 RPM
        get_gain_100 = GainIMO(
            imo_data_path="../imo_guidance/global_prob_matrix.csv", 
            forces_data_path=f"../{ship}/forces_CFD.csv",
            rotation=100,
            draft=self.draft
        )
        get_gain_100.run()
        
        # Carregar dados para 180 RPM
        get_gain_180 = GainIMO(
            imo_data_path="../imo_guidance/global_prob_matrix.csv",
            forces_data_path=f"../{ship}/forces_CFD.csv", 
            rotation=180,
            draft=self.draft
        )
        get_gain_180.run()
        
        return get_gain_100, get_gain_180

    def plot_wind_profiles(self, v10_list=(5, 10, 15), z_min=1.0, z_max=120.0, npoints=300,
                        method='power', alpha=1/7, z0=0.03, z_ref=10.0,
                        cmap='viridis', out_folder='figures',
                        fname=None, show=True):
        """
        Plot wind speed profiles V(z) extrapolated from V(10m) for several V10 values.
        Uses power-law (default) or log-law. Marks rotor height extrapolated speeds
        using self.extrapolated_wind_vel for reference (if present).
        """
        import os
        os.makedirs(out_folder, exist_ok=True)

        heights = np.linspace(z_min, z_max, npoints)
        cmap_obj = plt.get_cmap(cmap)
        colors = [cmap_obj(i / max(1, len(v10_list) - 1)) for i in range(len(v10_list))]

        plt.style.use("seaborn-v0_8-muted")
        fig, ax = plt.subplots(figsize=(8, 6))

        # rotor height used by extrapolated_wind_vel (same logic)
        z_rotor = 27.7 if self.draft == 16.0 else 35.5

        for i, V10 in enumerate(v10_list):
            if method == 'power':
                Vz = V10 * (heights / z_ref) ** alpha
                label = "$V_{10m}=$" + f"{V10} m/s"
            else:  # log-law
                zz = np.maximum(heights, z0 * 1.0001)
                denom = np.log(z_ref / z0)
                Vz = V10 * np.log(zz / z0) / denom
                label = "$V_{10m}=$" + f"{V10} m/s"

            ax.plot(Vz, heights, color=colors[i], lw=2, label=label)

            # mark rotor height using class extrapolation if available
            try:
                V_rot_ex = float(self.extrapolated_wind_vel(np.array([V10])))
            except Exception:
                V_rot_ex = V10 * (z_rotor / z_ref) ** alpha if method == 'power' else V10 * np.log(z_rotor / z0) / np.log(z_ref / z0)

            ax.scatter([V_rot_ex], [z_rotor], color=colors[i], s=50, edgecolor='k', zorder=5)

        # optionally show mapping of self.V_wind -> self.V_wz if present (as markers)
        if hasattr(self, "V_wind"):
            try:
                Vwz = self.extrapolated_wind_vel(self.V_wind)
                ax.scatter(Vwz, np.full_like(Vwz, z_rotor * 0.98), c='k', s=10, alpha=0.6, label="Ponto médio do rotor ($T = 16m$)")
            except Exception:
                pass

        ax.axvspan(3.0, 11.0, color='grey', alpha=0.08, zorder=0)
        ax.set_xlabel("Velocidade do vento (m/s)")
        ax.set_ylabel("Altura (m)")
        ax.set_ylim(z_min, z_max)
        ax.set_title("Perfis de velocidade de vento $V_{10m}$ extrapolados")
        ax.grid(which='both', linestyle='--', alpha=0.5)
        ax.legend(fontsize=9, loc='best')

        if fname is None:
            fname = f"wind_profiles_calado_{self.draft}_method_{method}.png"
        out_path = os.path.join(out_folder, fname)
        fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        if show:
            plt.show()
        plt.close(fig)
        return out_path
    def plot_wind_extrapolation(self, v10_list, z_min=1.0, z_max=100.0, npoints=300,
                                method='power', alpha=1/7, z0=0.03, z_ref=10.0,
                                cmap='viridis', out_folder='figures',
                                fname='wind_profile_extrapolation.png', show=True):
        
        os.makedirs(out_folder, exist_ok=True)
        heights = np.linspace(z_min, z_max, npoints)
        cmap_obj = plt.get_cmap(cmap)
        colors = [cmap_obj(i / max(1, len(v10_list)-1)) for i in range(len(v10_list))]

        plt.style.use("seaborn-v0_8-muted")
        fig, ax = plt.subplots(figsize=(8, 6))

        for i, V10 in enumerate(v10_list):
            if method == 'power':
                Vz = V10 * (heights / z_ref) ** alpha
                label = f"V(10m)={V10} m/s — power (α={alpha:.3f})"
            else:  # log-law
                # evita divisão por zero: garantir heights>z0
                zz = np.maximum(heights, z0 * 1.0001)
                denom = np.log(z_ref / z0)
                Vz = V10 * np.log(zz / z0) / denom
                label = f"V(10m)={V10} m/s — log (z0={z0})"

            ax.plot(Vz, heights, color=colors[i], lw=2, label=label)
            ax.scatter([V10], [z_ref], color=colors[i], s=30, edgecolor='k', zorder=5)  # marca referência

        # destaque intervalo exemplo 3..11 m/s em grid vertical (opcional)
        ax.axvspan(3.0, 11.0, color='grey', alpha=0.08, zorder=0)

        ax.set_xlabel("Velocidade do vento (m/s)")
        ax.set_ylabel("Altura (m)")
        ax.set_ylim(z_min, z_max)
        ax.set_title("Perfis de velocidade extrapolados a partir de V(10m)")
        ax.grid(which='both', linestyle='--', alpha=0.5)
        ax.legend(fontsize=9, loc='best')
        plt.tight_layout()

        out_path = os.path.join(out_folder, fname)
        fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        if show:
            plt.show()
        plt.close(fig)
        return out_path
    def plot_power_comparison(self, ship):
        """Plot comprehensive comparison between 100 RPM and 180 RPM power consumption"""
        
        # Carregar dados para ambas as rotações
        gain_100, gain_180 = self.load_power_matrices_comparison(ship)
        
        useful_forces = self.forces_k[2:11]
        

        # Calcular matrizes de potência para ambas as rotações
        power_100 = np.zeros((len(self.V_wind), len(self.wind_angles)))
        power_180 = np.zeros((len(self.V_wind), len(self.wind_angles)))
        
        V_wz = self.extrapolated_wind_vel(self.V_wind)
        
        for i, rel_angle in enumerate(self.wind_angles):
            Vk = self.calculate_relative_ship_velocity(np.deg2rad(rel_angle), V_wz)
            
            if int(rel_angle) <= 180:
                cfd_angle = 180 - int(rel_angle)
            else:
                cfd_angle = 540 - int(rel_angle)
                
            for j, vel in enumerate(Vk):
                power_100[j][i] = gain_100.get_power(int(cfd_angle), vel)
                power_180[j][i] = gain_180.get_power(int(cfd_angle), vel)
        
        # Criar subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Heatmap 100 RPM
        im1 = axes[0,0].imshow(power_100, aspect='auto', cmap='viridis',
                            extent=[0, 360, self.V_wind[0], self.V_wind[-1]], origin='lower')
        axes[0,0].set_title('Potência Consumida - 100 RPM')
        axes[0,0].set_xlabel('Ângulo relativo (°)')
        axes[0,0].set_ylabel('Velocidade do vento (m/s)')
        plt.colorbar(im1, ax=axes[0,0], label='Potência (kW)')
        
        # 2. Heatmap 180 RPM  
        im2 = axes[0,1].imshow(power_180, aspect='auto', cmap='viridis',
                            extent=[0, 360, self.V_wind[0], self.V_wind[-1]], origin='lower')
        axes[0,1].set_title('Potência Consumida - 180 RPM')
        axes[0,1].set_xlabel('Ângulo relativo (°)')
        axes[0,1].set_ylabel('Velocidade do vento (m/s)')
        plt.colorbar(im2, ax=axes[0,1], label='Potência (kW)')
        
        # 3. Diferença absoluta
        power_diff = power_180 - power_100
        im3 = axes[0,2].imshow(power_diff, aspect='auto', cmap='RdBu_r',
                            extent=[0, 360, self.V_wind[0], self.V_wind[-1]], origin='lower')
        axes[0,2].set_title('Diferença (180 RPM - 100 RPM)')
        axes[0,2].set_xlabel('Ângulo relativo (°)')
        axes[0,2].set_ylabel('Velocidade do vento (m/s)')
        plt.colorbar(im3, ax=axes[0,2], label='Diferença de Potência (kW)')
        
        # 4. Comparação por velocidade média
        mean_power_100 = np.mean(power_100, axis=1)
        mean_power_180 = np.mean(power_180, axis=1)
        
        axes[1,0].plot(self.V_wind, mean_power_100, 'b-', linewidth=2, label='100 RPM', marker='o')
        axes[1,0].plot(self.V_wind, mean_power_180, 'r-', linewidth=2, label='180 RPM', marker='s')
        axes[1,0].set_xlabel('Velocidade do vento (m/s)')
        axes[1,0].set_ylabel('Potência média (kW)')
        axes[1,0].set_title('Potência Média por Velocidade')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Comparação por ângulo médio
        mean_power_angle_100 = np.mean(power_100, axis=0)
        mean_power_angle_180 = np.mean(power_180, axis=0)
        
        axes[1,1].plot(self.wind_angles, mean_power_angle_100, 'b-', linewidth=2, label='100 RPM')
        axes[1,1].plot(self.wind_angles, mean_power_angle_180, 'r-', linewidth=2, label='180 RPM')
        axes[1,1].set_xlabel('Ângulo relativo (°)')
        axes[1,1].set_ylabel('Potência média (kW)')
        axes[1,1].set_title('Potência Média por Ângulo')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Distribuição de potências
        axes[1,2].hist(power_100.flatten(), bins=50, alpha=0.7, label='100 RPM', color='blue', density=True)
        axes[1,2].hist(power_180.flatten(), bins=50, alpha=0.7, label='180 RPM', color='red', density=True)
        axes[1,2].set_xlabel('Potência (kW)')
        axes[1,2].set_ylabel('Densidade de Probabilidade')
        axes[1,2].set_title('Distribuição de Potências')
        axes[1,2].legend()
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle(f'Comparativo de Potência Consumida: 100 RPM vs 180 RPM - {ship.upper()}', 
                    y=1.02, fontsize=16)
        plt.show()
        
        # Estatísticas resumo
        print("\n=== ESTATÍSTICAS RESUMO ===")
        print(f"Potência média 100 RPM: {np.mean(power_100):.2f} kW")
        print(f"Potência média 180 RPM: {np.mean(power_180):.2f} kW")
        print(f"Diferença média: {np.mean(power_diff):.2f} kW")
        print(f"Aumento percentual: {100*np.mean(power_diff)/np.mean(power_100):.1f}%")
        print(f"Potência máxima 100 RPM: {np.max(power_100):.2f} kW")
        print(f"Potência máxima 180 RPM: {np.max(power_180):.2f} kW")

    def get_adjacent_angles(self, ang):
        # Convert angle to be within [0, 360) range
        ang = ang % 360
        
        # Remove the 360 from angle_list for searching (we only want 0-330)
        search_angles = np.sort(self.angle_list[:-1])  # [0, 30, 60, ..., 330]
        # Find position
        pos = bisect.bisect_left(search_angles, ang)
        
        if pos == 0:
            # ang is at or before the first angle (0)
            if ang == search_angles[0]:
                floor_angle = ang
                ceil_angle = ang
            else:
                # This shouldn't happen if ang >= 0
                floor_angle = search_angles[-1]  # 330
                ceil_angle = search_angles[0]    # 0
        elif pos == len(search_angles):
            # ang is after the last angle (330), so it wraps around
            floor_angle = search_angles[-1]  # 330
            ceil_angle = search_angles[0]    # 0 (wraps around)
        else:
            # Normal case: ang is between two angles
            if ang == search_angles[pos]:
                # Exact match
                floor_angle = ang
                ceil_angle = ang
            else:
                # Between two angles
                floor_angle = search_angles[pos - 1]
                ceil_angle = search_angles[pos]
        
        return floor_angle, ceil_angle

    def get_forces(self, ang, vel, force="fx", coef_dir="cx"):
        """
        Robust interpolation of force and coefficient for given angle (deg) and velocity (m/s).
        - Uses get_adjacent_angles to get floor/ceil angles
        - Selects rows for those angles (correct rotation/draft)
        - Interpolates force over Vw with np.interp
        - Interpolates force over angle with proper wrap-around handling
        Returns: (f (kN), coef_x)
        """
        ang = float(ang) % 360.0
        floor_angle, ceil_angle = self.get_adjacent_angles(ang)

        # For angular interpolation treat ceil=0 as 360 (for distance calc) but selection uses 0
        ang_floor = float(floor_angle)
        ang_ceil_for_select = 0 if float(ceil_angle) == 360 or float(ceil_angle) == 0 else float(ceil_angle)
        ang_ceil_for_interp = float(ceil_angle) if float(ceil_angle) != 0 else 360.0

        def get_sorted_rows(angle_sel):
            sel = self.thrust_df[
                (self.thrust_df["Angulo"] == angle_sel) &
                (self.thrust_df["Calado"] == self.draft) &
                (self.thrust_df["Rotacao"] == self.rotation)
            ]
            if sel.empty:
                raise ValueError(f"No data for angle={angle_sel}, draft={self.draft}, rot={self.rotation}")
            sel = sel.sort_values("Vw")
            return sel

        # select rows AFTER angles resolved
        floor_sel = get_sorted_rows(int(ang_floor) % 360)
        ceil_sel = get_sorted_rows(int(ang_ceil_for_select) % 360)

        # Arrays for interpolation along Vw
        Vw_floor = floor_sel["Vw"].to_numpy(dtype=float)
        F_floor = floor_sel[force].to_numpy(dtype=float)
        Vw_ceil = ceil_sel["Vw"].to_numpy(dtype=float)
        F_ceil = ceil_sel[force].to_numpy(dtype=float)

        # If vel outside the Vw range, np.interp returns edge value (that's fine here).
        f_floor_at_vel = np.interp(vel, Vw_floor, F_floor)
        f_ceil_at_vel = np.interp(vel, Vw_ceil, F_ceil)

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

        # compute coefficient consistent with how you use it (avoid divide-by-zero)
        rho = 1.2
        denom = 0.5 * rho * (vel ** 2) * self.Ax / 1000.0  # matches earlier scaling
        coef_x = f / denom if denom > 0 else np.nan

        return f, coef_x
    def get_power(self, ang, vel):
        floor_angle, ceil_angle = self.get_adjacent_angles(ang)
        if ceil_angle == 360:
            ceil_angle = 0
        floor_moments = self.thrust_df[
            (self.thrust_df["Angulo"] == floor_angle)
            & (self.thrust_df["Calado"] == self.draft)
            & (self.thrust_df["Rotacao"] == self.rotation)
        ]
        ceil_moments = self.thrust_df[
            (self.thrust_df["Angulo"] == ceil_angle)
            & (self.thrust_df["Calado"] == self.draft)
            & (self.thrust_df["Rotacao"] == self.rotation)
        ]
        if vel >= 6 and vel <= 10:
            P_ceil = ceil_moments["Pcons"].iloc[0] + (
                vel - ceil_moments["Vw"].iloc[0]
            ) * (ceil_moments["Pcons"].iloc[1] - ceil_moments["Pcons"].iloc[0]) / (
                ceil_moments["Vw"].iloc[1] - ceil_moments["Vw"].iloc[0]
            )
            P_floor = floor_moments["Pcons"].iloc[0] + (
                vel - floor_moments["Vw"].iloc[0]
            ) * (floor_moments["Pcons"].iloc[1] - floor_moments["Pcons"].iloc[0]) / (
                floor_moments["Vw"].iloc[1] - floor_moments["Vw"].iloc[0]
            )
            if ceil_angle != floor_angle:
                P = P_floor + (ang - floor_angle) * (P_ceil - P_floor) / (
                    ceil_angle - floor_angle
                )
            else:
                P = P_floor


        elif vel > 10 and vel <= 12:
            P_ceil = ceil_moments["Pcons"].iloc[1] + (
                vel - ceil_moments["Vw"].iloc[1]
            ) * (ceil_moments["Pcons"].iloc[2] - ceil_moments["Pcons"].iloc[1]) / (
                ceil_moments["Vw"].iloc[2] - ceil_moments["Vw"].iloc[1]
            )
            P_floor = floor_moments["Pcons"].iloc[1] + (
                vel - floor_moments["Vw"].iloc[1]
            ) * (floor_moments["Pcons"].iloc[2] - floor_moments["Pcons"].iloc[1]) / (
                floor_moments["Vw"].iloc[2] - floor_moments["Vw"].iloc[1]
            )
            if ceil_angle != floor_angle:
                P = P_floor + (ang - floor_angle) * (P_ceil - P_floor) / (
                    ceil_angle - floor_angle
                )
            else:
                P = P_floor

        elif vel < 6:
            P = np.mean([ceil_moments["Pcons"].iloc[0], floor_moments["Pcons"].iloc[0]])

        elif vel > 12:
            P = np.mean([ceil_moments["Pcons"].iloc[2], floor_moments["Pcons"].iloc[2]])
        return P
    
    def extrapolated_wind_vel(self, v_10):
        z = 27.7 if self.draft == 16.0 else 35.5 # 7.2 (Depth - Draft: 7.2 for design and 14.7 for ballast) + 3 (Base height) + 17.5 (Rotor height/2)
        alpha = 1 / 9
        return v_10 * np.power(z / 10, alpha)

    def polar_plot(self, x_axis, y_axis, z_axis, title):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)
        theta = np.deg2rad(x_axis)
        for i in range(int(len(y_axis) / 3)):
            ax.plot(
                theta, z_axis[int(i * 3), :], label=f"V = {y_axis[int(i*3)]:.1f} m/s"
            )
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_thetagrids(range(0, 360, 30))  # Add this line for 30-degree steps
        ax.set_title(title, va="bottom", fontsize=20)
        ax.legend(loc="center right", bbox_to_anchor=(1.2, 0.1))
        plt.show()

    def heatmap_plot(self, matrix, vels):
        plt.figure(figsize=(10, 6))
        plt.imshow(
            matrix,
            aspect="auto",
            cmap="viridis",
            extent=[0, 360, vels[0], vels[-1]],
            origin="lower",
        )

        plt.colorbar(label="Probabilidade")
        plt.xlabel("Ângulo de incidência do vento (°)", fontsize=10)
        plt.ylabel("Velocidade do vento (m/s)", fontsize=10)
        # plt.title("Mapa de calor da força em função da velocidade e ângulo")
        plt.show()

    def plot_velocity_comparison(self):
        """Plot comparison between relative velocity (Vk), extrapolated wind velocity, and absolute wind velocity"""
        
        sample_angles = [0, 45, 90, 135, 180, 225, 270, 315] 
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        V_wind_abs = self.V_wind  # Velocidade absoluta do vento
        V_wind_extrap = self.extrapolated_wind_vel(self.V_wind)  # Velocidade extrapolada
        
        for idx, angle in enumerate(sample_angles):
            ax = axes[idx]
            
            # Calcular velocidade relativa para este ângulo
            Vk, ang = self.calculate_relative_ship_velocity(np.deg2rad(angle), V_wind_extrap)
            # Plot das três velocidades
            ax.plot(V_wind_abs, V_wind_abs, 'b-', linewidth=2, label='V absoluta')
            ax.plot(V_wind_abs, V_wind_extrap, 'g--', linewidth=2, label=f'V extrapolada')
            ax.plot(V_wind_abs, Vk, 'r-', linewidth=2, label=f'V relativa')
            
            ax.set_xlabel('Velocidade do vento absoluta (m/s)')
            ax.set_ylabel('Velocidade (m/s)')
            ax.set_title(f'Ângulo absoluto do vento: {angle}°')
            ax.grid(True, alpha=0.3)
            ax.legend(loc="center right", bbox_to_anchor=(0.8, 0.1))
            # Adicionar informações estatísticas
            mean_diff_abs = np.mean(Vk - V_wind_abs)
            mean_diff_extrap = np.mean(Vk - V_wind_extrap)
            ax.text(0.05, 0.95, f'Δ abs: {mean_diff_abs:.1f}\nΔ extrap: {mean_diff_extrap:.1f}', 
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.suptitle('Comparação de Velocidades: Absoluta vs Extrapolada vs Relativa', y=1.02, fontsize=16)
        plt.show()

    def plot_velocity_summary(self):
        """Plot summary comparison for all angles"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        V_wind_abs = self.V_wind
        V_wind_extrap = self.extrapolated_wind_vel(self.V_wind)
        
        # Plot 1: Heatmap da velocidade relativa
        Vk_matrix = np.zeros((len(self.V_wind), len(self.wind_angles)))
        
        for i, angle in enumerate(self.wind_angles):
            Vk, ang = self.calculate_relative_ship_velocity(np.deg2rad(angle), V_wind_extrap)
            Vk_matrix[:, i] = Vk
        
        im1 = ax1.imshow(Vk_matrix, aspect='auto', cmap='viridis', 
                        extent=[0, 360, V_wind_abs[0], V_wind_abs[-1]], origin='lower')
        ax1.set_xlabel('Ângulo absoluto do vento (°)', fontsize=15)
        ax1.set_ylabel('Velocidade absoluta do vento (m/s)', fontsize=15)
        plt.colorbar(im1, ax=ax1, label='Velocidade relativa (m/s)')
        
        # Plot 2: Comparação das médias por velocidade
        mean_Vk_by_speed = np.mean(Vk_matrix, axis=1)
        
        ax2.plot(V_wind_abs, V_wind_abs, 'b-', linewidth=3, label='V absoluta')
        ax2.plot(V_wind_abs, V_wind_extrap, 'g--', linewidth=3, label=f'V extrapolada')
        ax2.plot(V_wind_abs, mean_Vk_by_speed, 'r-', linewidth=3, label='V relativa (média)')
        
        ax2.set_xlabel('Velocidade do vento absoluta (m/s)', fontsize=15)
        ax2.set_ylabel('Velocidade (m/s)', fontsize=15)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

    def plot_cx_vs_velocity(self, angles=None):
        """
        Plot Cx (self.coefs_xk) behaviour along wind speed.
        - If angles is None: plot mean +/- std across all angles.
        - If angles is a list of angles (degrees): plot Cx for those angles.
        """
        if not hasattr(self, "coefs_xk") or self.coefs_xk.size == 0:
            raise RuntimeError("coefs_xk not available. Run run() first.")

        V = self.V_wind  # wind speeds (m/s)
        C = self.coefs_xk  # shape (nV, nAngles)

        plt.figure(figsize=(8, 5))
        if angles is None:
            mean_cx = np.nanmean(C, axis=1)
            std_cx = np.nanstd(C, axis=1)
            plt.plot(V, mean_cx, color="black", lw=2, label="mean Cx (all angles)")
            plt.fill_between(V, mean_cx - std_cx, mean_cx + std_cx, color="gray", alpha=0.3,
                             label="±1 std")
        else:
            # find nearest available angle indices
            idxs = [int(np.argmin(np.abs(self.wind_angles - a))) for a in angles]
            for idx in idxs:
                plt.plot(V, C[:, idx], lw=2, label=f"angle {int(self.wind_angles[idx])}°")

        plt.xlabel("Wind speed (m/s)", fontsize=12)
        plt.ylabel("Cx (coeficient)", fontsize=12)
        plt.title("Cx vs Wind speed", fontsize=14)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def calculate_relative_ship_velocity(self, wind_angle, V_wz):
        """
        Calculate relative wind velocity using vector components.
        Positive values = headwind, Negative values = tailwind
        
        rel_angle: angle in radians where wind is coming from (0 = from ahead)
        V_wz: extrapolated wind velocity
        """
        u_ship = self.V_ship 
        
        u_wind = V_wz * -np.cos(wind_angle)
        v_wind = V_wz * -np.sin(wind_angle)
        
        u_rel = u_ship - u_wind 
        v_rel = -v_wind
        
        rel_ang = np.degrees(np.arctan2(v_rel, u_rel)) % 360

        return np.sqrt(np.power(u_rel, 2) + np.power(v_rel, 2)), rel_ang
    
    def run(self):
        self.V_wind = np.arange(1, 26)

        self.V_wz = self.extrapolated_wind_vel(self.V_wind)

        # Here, 0 are the wind coming from ship heading
        self.wind_angles = np.arange(0, 360, 5)
        first_term = 0
        second_term = 0
        third_term = 0
        self.forces_k = np.zeros((self.V_wz.shape[0], self.wind_angles.shape[0]))
        self.forces_rotor_k = np.zeros((self.V_wz.shape[0], self.wind_angles.shape[0]))
        self.coefs_xk = np.zeros((self.V_wz.shape[0], self.wind_angles.shape[0]))
        self.rel_angls = np.zeros((self.V_wz.shape[0], self.wind_angles.shape[0]))
        self.V_k_mat = np.zeros((self.V_wz.shape[0], self.wind_angles.shape[0]))
        
        power_k = np.zeros((self.V_wz.shape[0], self.wind_angles.shape[0]))
        self.W_k = np.zeros((self.V_wz.shape[0], self.wind_angles.shape[0]))


        soma_forces = 0
        for i, wind_angle in enumerate(self.wind_angles):
            self.Vk, ang = self.calculate_relative_ship_velocity(np.deg2rad(wind_angle), self.V_wz)
            # Fix the angle reference for CFD

            self.V_k_mat[:, i] = self.Vk
            self.rel_angls[:, i] = ang  
            
            for j, (vel, angle) in enumerate(zip(self.V_wind, ang)):
                Pk = self.get_power(angle, vel)
                Fxk, cxk = self.get_forces(angle, vel)
                Fxk_rotor, cxk_rotor = self.get_forces(angle, vel, force='fx_rotores')
                if Fxk >= self.RT:
                    Fxk = self.RT
                
                self.forces_k[j][i] = Fxk
                soma_forces += Fxk
                self.forces_rotor_k[j][i] = Fxk_rotor
                self.coefs_xk[j][i] = cxk
                power_k[j][i] = Pk
                self.W_k[j][i] = self.imo_df.loc[j, str(wind_angle)]
                first_term += self.W_k[j][i]
                second_term += (self.V_ship / self.eta_D) * Fxk * self.W_k[j][i]
                third_term += Pk * self.W_k[j][i]
        print(f"Soma de todas as forças: ", soma_forces)
        print(f"Força: {second_term}")
        print(f"Consumo: {third_term}")
        print("Soma de forças", np.sum(self.forces_k[2:11]))
        print(f"Razão do consumo pelo empuxo: {np.round(100*third_term/second_term)}%")
        f_eff_P_eff = (1/first_term) * (second_term - third_term)
        print(f"Primeiro termo: {first_term}")
        print(f"f_eff_p_eff: {f_eff_P_eff}")
        
        print(f"Potência efetiva: {np.round(self.P_break - f_eff_P_eff)} kW")
        print(f"Ganho: {np.round(100*f_eff_P_eff/self.P_break)}%")
        return self.forces_k
    
    def plot_graphics(self):
        # Cx, Cz plots
        self.plot_cx_vs_velocity(angles=[30, 60, 90])
        # self.polar_plot(self.wind_angles, self.V_wind, self.coefs_xk, "Curvas polares de C_x para diferentes velocidades")
        self.polar_plot(self.wind_angles, self.V_wind, self.forces_rotor_k, "Curvas polares da força para diferentes velocidades (só rotor)")
        self.heatmap_plot(self.W_k, self.V_wind)
        self.plot_velocity_comparison()
        self.plot_velocity_summary() 

    def compare_common_force(self, mats, target_angles=[0, 90, 180], vk_mats=None, ang_mats=None):
        """
        Plot total force vs wind speed for specific angles with relative velocity and angle annotations
        
        Args:
            mats: List of force matrices
            target_angles: List of angles in degrees to include in the sum
            vk_mats: List of relative velocity matrices (same order as mats)
            ang_mats: List of relative angle matrices (same order as mats)
        """
        speed_slice = slice(0, 25)
        V = np.asarray(self.V_wind)[speed_slice]

        # Find indices of target angles in wind_angles array
        angle_indices = []
        actual_angles = []
        for target_angle in target_angles:
            idx = np.argmin(np.abs(self.wind_angles - target_angle))
            angle_indices.append(idx)
            actual_angles.append(self.wind_angles[idx])
            print(f"[INFO] Target angle {target_angle}° -> Index {idx} (actual angle: {self.wind_angles[idx]}°)")

        # ensure numpy arrays
        mats = [np.asarray(m) for m in mats]
        if vk_mats is not None:
            vk_mats = [np.asarray(m) for m in vk_mats]
        if ang_mats is not None:
            ang_mats = [np.asarray(m) for m in ang_mats]

        # Verificar quantas matrizes temos e organizar os pares corretamente
        print(f"[DEBUG] Number of matrices: {len(mats)}")
        print(f"[DEBUG] Number of vk_mats: {len(vk_mats) if vk_mats is not None else 'None'}")
        print(f"[DEBUG] Number of ang_mats: {len(ang_mats) if ang_mats is not None else 'None'}")
        
        if len(mats) == 4:
            # 4 matrizes: draft 8.5 (100, 180) + draft 16 (100, 180)
            pairs = [(mats[0][speed_slice, :], mats[1][speed_slice, :]),  # draft 8.5
                    (mats[2][speed_slice, :], mats[3][speed_slice, :])]   # draft 16
            drafts = ["calado 8.5", "calado 16"]
            
            if vk_mats is not None and len(vk_mats) == 4:
                vk_pairs = [(vk_mats[0][speed_slice, :], vk_mats[1][speed_slice, :]),
                        (vk_mats[2][speed_slice, :], vk_mats[3][speed_slice, :])]
            else:
                vk_pairs = [None, None]
                
            if ang_mats is not None and len(ang_mats) == 4:
                ang_pairs = [(ang_mats[0][speed_slice, :], ang_mats[1][speed_slice, :]),
                            (ang_mats[2][speed_slice, :], ang_mats[3][speed_slice, :])]
            else:
                ang_pairs = [None, None]
                
        elif len(mats) == 2:
            # 2 matrizes: apenas um calado (100, 180)
            pairs = [(mats[0][speed_slice, :], mats[1][speed_slice, :])]
            drafts = ["calado 16"]  # Ajustar para o calado correto
            
            if vk_mats is not None and len(vk_mats) == 2:
                vk_pairs = [(vk_mats[0][speed_slice, :], vk_mats[1][speed_slice, :])]
            else:
                vk_pairs = [None]
                
            if ang_mats is not None and len(ang_mats) == 2:
                ang_pairs = [(ang_mats[0][speed_slice, :], ang_mats[1][speed_slice, :])]
            else:
                ang_pairs = [None]
        else:
            raise ValueError(f"Expected 2 or 4 matrices, got {len(mats)}. Matrix order should be: [draft1_100rpm, draft1_180rpm, draft2_100rpm, draft2_180rpm]")

        def integrate_between(Vvec, yvec, vmin, vmax, npoints=400):
            xs = np.linspace(vmin, vmax, npoints)
            ys = np.interp(xs, Vvec, yvec)
            return np.trapz(ys, xs)

        vmin, vmax = 3.0, 11.0

        # Criar subplots baseado no número de pares (calados)
        fig, axes = plt.subplots(1, len(pairs), figsize=(8 * len(pairs), 6), sharey=True)
        if len(pairs) == 1:
            axes = [axes]

        for ax_idx, ((mat100, mat180), draft_label) in enumerate(zip(pairs, drafts)):
            ax = axes[ax_idx]
            
            print(f"[DEBUG] Processing {draft_label}: mat100 shape {mat100.shape}, mat180 shape {mat180.shape}")
            
            # Sum only specific angles for each velocity
            sum100 = np.nansum(mat100[:, angle_indices], axis=1)
            sum180 = np.nansum(mat180[:, angle_indices], axis=1)

            # Calculate statistics
            area100 = integrate_between(V, sum100, vmin, vmax)
            area180 = integrate_between(V, sum180, vmin, vmax)
            
            # Plot main lines
            angles_str = ", ".join([f"{int(angle)}°" for angle in actual_angles])
            line1 = ax.plot(V, sum100, marker="o", lw=2.5, markersize=8,
                        label=f"100 RPM (área = {area100:.1f})", color="blue")
            line2 = ax.plot(V, sum180, marker="s", lw=2.5, markersize=8,
                        label=f"180 RPM (área = {area180:.1f})", color="red")

            # Corrigir a verificação das anotações - remover a verificação de ax_idx
            if vk_pairs is not None and ang_pairs is not None and ax_idx < len(vk_pairs) and vk_pairs[ax_idx] is not None and ang_pairs[ax_idx] is not None:
                print(f"[DEBUG] Adding annotations for {draft_label}")
                
                # Use the first matrix (100 RPM) since Vk and angles are the same for both
                vk_ref = vk_pairs[ax_idx][0]  # vk100 for this draft
                ang_ref = ang_pairs[ax_idx][0]  # ang100 for this draft
                
                print(f"[DEBUG] vk_ref shape: {vk_ref.shape}, ang_ref shape: {ang_ref.shape}")
                
                # Sample points for annotation
                sample_indices = range(4, len(V), 6)  # Começar em 4 e a cada 6 pontos
                
                for i in sample_indices:
                    if i < len(V) and i < vk_ref.shape[0]:
                        # Calculate average Vk and angle for the target angles
                        avg_vk = np.mean(vk_ref[i, angle_indices])
                        avg_ang = np.mean(ang_ref[i, angle_indices])
                        
                        print(f"[DEBUG] Point {i}: V={V[i]:.1f}, Vk={avg_vk:.1f}, ang={avg_ang:.1f}")
                        
                        # Add vertical dashed line
                        ax.axvline(x=V[i], ymin=0, ymax=1, color='gray', linestyle='--', 
                                alpha=0.6, linewidth=1.5, zorder=1)
                        
                        # Add text annotation for EVERY vertical line
                        # Find the y position at the top of the plot area
                        # Find the y position at the top of the plot area
                        y_plot_max = max(max(sum100), max(sum180))
                        y_plot_min = min(min(sum100), min(sum180))
                        y_range = y_plot_max - y_plot_min if y_plot_max != y_plot_min else abs(y_plot_max)
                        
                        # Position the text above the highest point
                        text_y = y_plot_max + 0.15 * abs(y_range)
                        
                        # Ensure axis top limit includes the annotation (extend if needed)
                        cur_ylim = ax.get_ylim()
                        new_top = max(cur_ylim[1], text_y + 0.05 * max(1.0, abs(y_range)))
                        if new_top != cur_ylim[1]:
                            ax.set_ylim(cur_ylim[0], new_top)
                        
                        # draw vertical dashed line
                        ax.axvline(x=V[i], ymin=0, ymax=1, color='gray', linestyle='--', 
                                alpha=0.6, linewidth=1.5, zorder=1)
                        
                        # annotate with clip disabled so text outside axis is visible
                        ax.annotate(
                            f'Vk={avg_vk:.1f}\nθ={avg_ang:.0f}°',
                            xy=(V[i], y_plot_max),               # arrow point (data coords)
                            xytext=(0, 8),                       # offset in points from xy
                            textcoords='offset points',
                            fontsize=8,
                            ha='center', va='bottom',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray',
                                    alpha=0.9, edgecolor='gray', linewidth=0.8),
                            arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7, lw=0.8),
                            clip_on=False,
                            zorder=5
                        )
            else:
                print(f"[DEBUG] No annotations for {draft_label}")
                print(f"[DEBUG] - vk_pairs: {vk_pairs}")
                print(f"[DEBUG] - ang_pairs: {ang_pairs}")
                print(f"[DEBUG] - ax_idx: {ax_idx}, len(vk_pairs): {len(vk_pairs) if vk_pairs else 'None'}")

            # Styling
            ax.axvspan(vmin, vmax, color="grey", alpha=0.15)
            ax.axhline(0.0, color="k", linestyle="--", linewidth=1, alpha=0.7)
            
            ax.set_xlabel("Velocidade do vento (m/s)", fontsize=12)
            ax.set_title(f"Forças ({angles_str}) — {draft_label}", fontsize=13)
            ax.grid(alpha=0.3)
            ax.legend(loc="lower right", bbox_to_anchor=(0.98, 0.02), fontsize=10, framealpha=0.9, borderaxespad=0.4)
        axes[0].set_ylabel("Soma das forças (kN)", fontsize=12)
        
        plt.suptitle(f"Força para ângulos específicos: {', '.join(map(str, target_angles))}°\n" +
                    f"(Linhas tracejadas: Vk = velocidade relativa, θrel = ângulo relativo)", 
                    fontsize=14)
        plt.tight_layout()
        
        os.makedirs("figures", exist_ok=True)
        out_path = f"figures/forces_angles_{'_'.join(map(str, target_angles))}_with_vertical_annotations.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor='white')
        plt.show()
        plt.close(fig)
        
        return angle_indices, actual_angles

def main():
    parser = argparse.ArgumentParser(description="Wind Route Creator")
    parser.add_argument("--ship", required=True, help="afra or suez")
    parser.add_argument("--plot", action="store_true", help="Plot graphics")
    parser.add_argument("--modified-matrix", action="store_true", help="Translate velocities +3 m/s from probability matrix")

    args = parser.parse_args()
    ship = "abdias_suez" if args.ship == "suez" else "castro_alves_afra"

    imo_data_path = "../imo_guidance/global_prob_matrix.csv"
    forces_data_path = f"../{ship}/forces_CFD.csv"
    rotations = [100, 180]
    drafts = [16]
    force_matrix = []
    vk_matrix = []
    ang_matrix = [] 
    for draft in drafts:
        for rotation in rotations:
            print(f"[INFO] Testing for draft = {draft} and rotation = {rotation}")
            get_gain = GainIMO(
                imo_data_path=imo_data_path,
                forces_data_path=forces_data_path,
                rotation=rotation,
                draft=draft,
                modified=args.modified_matrix
            )
            force_matrix.append(get_gain.run())
            vk_matrix.append(get_gain.V_k_mat)
            ang_matrix.append(get_gain.rel_angls)

            if args.plot: 
                get_gain.plot_graphics()
        
    # get_gain.plot_wind_profiles()
    
    # Passar as matrizes de Vk e ângulo para o método de comparação
    get_gain.compare_common_force(force_matrix, target_angles=[60], 
                                 vk_mats=vk_matrix, ang_mats=ang_matrix)
    get_gain.compare_common_force(force_matrix, target_angles=[90], 
                                 vk_mats=vk_matrix, ang_mats=ang_matrix)
    get_gain.compare_common_force(force_matrix, target_angles=[120], 
                                 vk_mats=vk_matrix, ang_mats=ang_matrix)

if __name__ == "__main__":
    main()
