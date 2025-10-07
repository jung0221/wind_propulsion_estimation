import pandas as pd
import numpy as np
import bisect
import argparse
import matplotlib.pyplot as plt
from matplotlib import cm

class GainIMO:
    def __init__(self, xls_data_path, imo_data_path, forces_data_path, rotation, draft):
        xls_csv = pd.read_excel(xls_data_path, sheet_name="Ganho")
        p_cons = xls_csv["P_rotor kW"]
        self.imo_df = pd.read_csv(imo_data_path)
        self.thrust_df = pd.read_csv(forces_data_path).iloc[:, 2:]
        self.thrust_df["Angulo"] = np.where(
        self.thrust_df["Angulo"] <= 180, 
        180 - self.thrust_df["Angulo"], 
        540 - self.thrust_df["Angulo"]
        ) % 360
        self.thrust_df["Pcons"] = p_cons
        self.moments = (
            self.thrust_df["Mz_popa_bom"]
            + self.thrust_df["Mz_popa_bor"]
            + self.thrust_df["Mz_proa_bom"]
            + self.thrust_df["Mz_proa_bor"]
        )
        angle_list = self.thrust_df["Angulo"].unique()
        self.angle_list = np.append(angle_list, 360)

        # 1. Define V_ref (ship velocity)
        self.V_ship = 12 * 0.5144  # knots to m/s
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
            xls_data_path="../abdias_suez/CFD_Jung.xlsx",
            imo_data_path="../imo_guidance/global_prob_matrix.csv", 
            forces_data_path=f"../{ship}/forces_CFD.csv",
            rotation=100,
            draft=self.draft
        )
        get_gain_100.run()
        
        # Carregar dados para 180 RPM
        get_gain_180 = GainIMO(
            xls_data_path="../abdias_suez/CFD_Jung.xlsx",
            imo_data_path="../imo_guidance/global_prob_matrix.csv",
            forces_data_path=f"../{ship}/forces_CFD.csv", 
            rotation=180,
            draft=self.draft
        )
        get_gain_180.run()
        
        return get_gain_100, get_gain_180

    def plot_power_comparison(self, ship):
        """Plot comprehensive comparison between 100 RPM and 180 RPM power consumption"""
        
        # Carregar dados para ambas as rotações
        gain_100, gain_180 = self.load_power_matrices_comparison(ship)
        
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
                floor_angle = search_angles[0]
                ceil_angle = search_angles[1]
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
                floor_angle = search_angles[pos]
                ceil_angle = search_angles[pos + 1] if pos + 1 < len(search_angles) else search_angles[0]
            else:
                # Between two angles
                floor_angle = search_angles[pos - 1]
                ceil_angle = search_angles[pos]
        
        return floor_angle, ceil_angle

    def get_forces(self, ang, vel, force="fx", coef_dir="cx"):
        floor_angle, ceil_angle = self.get_adjacent_angles(ang)

        floor_forces = self.thrust_df[
            (self.thrust_df["Angulo"] == floor_angle)
            & (self.thrust_df["Calado"] == self.draft)
            & (self.thrust_df["Rotacao"] == self.rotation)
        ]

        if ceil_angle == 360:
            ceil_ang = 0
        else:
            ceil_ang = ceil_angle
        ceil_forces = self.thrust_df[
            (self.thrust_df["Angulo"] == ceil_ang)
            & (self.thrust_df["Calado"] == self.draft)
            & (self.thrust_df["Rotacao"] == self.rotation)
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

    # def calculate_relative_ship_velocity(self, rel_angle, V_wz):
    #     return np.sqrt(
    #         np.power(V_wz, 2)
    #         + np.power(self.V_ship, 2)
    #         - 2 * V_wz * self.V_ship * np.cos(rel_angle)
    #     )

    def polar_plot(self, x_axis, y_axis, z_axis):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)
        theta = np.deg2rad(x_axis)
        for i in range(int(len(y_axis) / 5)):
            ax.plot(
                theta, z_axis[int(i * 5), :], label=f"V = {y_axis[int(i*5)]:.1f} m/s"
            )
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_thetagrids(range(0, 360, 30))  # Add this line for 30-degree steps
        ax.set_title("Curvas polares da força para diferentes velocidades", va="bottom", fontsize=20)
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
        
        # Relative wind components (wind relative to ship)
        u_rel = u_ship - u_wind 
        v_rel = v_wind
        
        rel_ang = np.degrees(np.arctan2(v_rel, u_rel)) % 360

        return np.sqrt(np.power(u_rel, 2) + np.power(v_rel, 2)), rel_ang
    def run(self):
        self.V_wind = np.arange(1, 26)

        V_wz = self.extrapolated_wind_vel(self.V_wind)

        # Here, 0 are the wind coming from ship heading
        self.wind_angles = np.arange(0, 360, 5)
        first_term = 0
        second_term = 0
        third_term = 0
        self.forces_k = np.zeros((V_wz.shape[0], self.wind_angles.shape[0]))
        self.forces_ki = np.zeros((V_wz.shape[0], self.wind_angles.shape[0]))
        
        power_k = np.zeros((V_wz.shape[0], self.wind_angles.shape[0]))
        self.W_k = np.zeros((V_wz.shape[0], self.wind_angles.shape[0]))
        sum_extrp = 0
        sum_rel = 0

        for i, wind_angle in enumerate(self.wind_angles):
            Vk, ang = self.calculate_relative_ship_velocity(np.deg2rad(wind_angle), V_wz)
            # Fix the angle reference for CFD
            for j, (vel, angle) in enumerate(zip(Vk, ang)):
                Pk = self.get_power(angle, vel)
                Fxk = self.get_forces(angle, vel)
                if Fxk >= self.RT:
                    Fxk = self.RT
                self.forces_k[j][i] = Fxk
                self.forces_ki[j][i] = self.get_forces(angle, V_wz[j])
                power_k[j][i] = Pk
                self.W_k[j][i] = self.imo_df.loc[j, str(wind_angle)]
                first_term += self.W_k[j][i]
                second_term += (self.V_ship / self.eta_D) * Fxk * self.W_k[j][i]
                third_term += Pk * self.W_k[j][i]
                # if j == 5: print(f"Vel {vel} angle {cfd_angle} force {Fxk}")
            
            sum_extrp += np.sum(self.forces_ki[:,i])
            sum_rel += np.sum(self.forces_k[:,i])
            
        print(f"Vel extrp: {sum_extrp}, Vel rel: {sum_rel}")
            
        f_eff_P_eff = (1/first_term) * (second_term - third_term)
        print(f"Potência efetiva: {np.round(self.P_eff - f_eff_P_eff)} kW")
        print(f"Ganho: {np.round(100*f_eff_P_eff/self.P_break)}%")
        return

    def plot_graphics(self):
        # self.polar_plot(self.wind_angles, self.V_wind, self.forces_k)
        # self.heatmap_plot(self.W_k, self.V_wind)
        self.plot_velocity_comparison()
        self.plot_velocity_summary() 

def main():
    parser = argparse.ArgumentParser(description="Wind Route Creator")
    parser.add_argument("--ship", required=True, help="afra or suez")
    parser.add_argument("--plot", action="store_true", help="Plot graphics")

    args = parser.parse_args()
    ship = "abdias_suez" if args.ship == "suez" else "castro_alves_afra"

    xls_data_path = "../abdias_suez/CFD_Jung.xlsx"
    imo_data_path = "../imo_guidance/global_prob_matrix.csv"
    forces_data_path = f"../{ship}/forces_CFD.csv"
    rotations = [100, 180]
    drafts = [8.5, 16]
    
    for rotation in rotations:
        for draft in drafts:
            print(f"[INFO] Testing for draft = {draft} and rotation = {rotation}")
            get_gain = GainIMO(
                xls_data_path=xls_data_path,
                imo_data_path=imo_data_path,
                forces_data_path=forces_data_path,
                rotation=rotation,
                draft=draft
            )
            get_gain.run()
            if args.plot: get_gain.plot_graphics()
    # get_gain.plot_power_comparison(ship)

if __name__ == "__main__":
    main()
