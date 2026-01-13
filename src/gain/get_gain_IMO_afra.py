import pandas as pd
import numpy as np
import bisect
import argparse
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.getcwd())
import src.utils.plot_functions as pf

# teste
class GainIMO:
    def __init__(self, imo_data_path, ship, forces_data_path, draft, modified):
        self.imo_df = pd.read_csv(imo_data_path)
        self.ship = ship

        # Probabilities translated to three rows
        if modified:
            try:
                self.imo_df = self.imo_df.astype(float)
            except Exception:
                num_cols = self.imo_df.select_dtypes(include=["number"]).columns
                self.imo_df[num_cols] = self.imo_df[num_cols].astype(float)
            # shift all rows down by 3 (V=1 -> V=4), fill top 3 rows with zeros
            if len(self.imo_df) > 3:
                shifted = pd.DataFrame(
                    0.0,
                    index=self.imo_df.index,
                    columns=self.imo_df.columns,
                    dtype=float,
                )
                # usar to_numpy(dtype=float) para garantir compatibilidade de tipos
                shifted.iloc[3:, :] = self.imo_df.iloc[:-3, :].to_numpy(dtype=float)
                self.imo_df = shifted
            else:
                # not enough rows -> zero numeric columns
                self.imo_df.loc[:, :] = 0.0

            print(
                "[INFO] imo_df shifted +3 rows (first 3 rows set to 0) due to modified=True"
            )

        self.thrust_df = pd.read_csv(forces_data_path)

        # Invert angles to IMO reference
        self.thrust_df["Angulo"] = (
            np.where(
                self.thrust_df["Angulo"] <= 180,
                180 - self.thrust_df["Angulo"],
                540 - self.thrust_df["Angulo"],
            )
            % 360
        )
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

        self.RT = 744 if self.ship == "suez" else 640  # kN
        self.draft = float(draft)  # meters
        if self.draft < 10:
            self.RT = 696 if self.ship == "suez" else 475
        self.Ax = 175
        self.Ay = 175
        self.P_break = self.RT * self.V_ship / 0.63

    def load_power_matrices_comparison(self, ship):
        """Load power matrices for both 100 and 180 RPM"""

        # Carregar dados para 100 RPM
        get_gain_100 = GainIMO(
            imo_data_path="../imo_guidance/global_prob_matrix.csv",
            forces_data_path=f"../{ship}/forces_CFD.csv",
            rotation=100,
            draft=self.draft,
        )
        get_gain_100.run()

        # Carregar dados para 180 RPM
        get_gain_180 = GainIMO(
            imo_data_path="../imo_guidance/global_prob_matrix.csv",
            forces_data_path=f"../{ship}/forces_CFD.csv",
            rotation=180,
            draft=self.draft,
        )
        get_gain_180.run()

        return get_gain_100, get_gain_180

    def plot_wind_profiles(
        self,
        v10_list=(5, 10, 15),
        z_min=1.0,
        z_max=120.0,
        npoints=300,
        method="power",
        alpha=1 / 7,
        z0=0.03,
        z_ref=10.0,
        cmap="viridis",
        out_folder="figures",
        fname=None,
        show=True,
    ):
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
            if method == "power":
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
                V_rot_ex = (
                    V10 * (z_rotor / z_ref) ** alpha
                    if method == "power"
                    else V10 * np.log(z_rotor / z0) / np.log(z_ref / z0)
                )

            ax.scatter(
                [V_rot_ex], [z_rotor], color=colors[i], s=50, edgecolor="k", zorder=5
            )

        # optionally show mapping of self.V_wind -> self.V_wz if present (as markers)
        if hasattr(self, "V_wind"):
            try:
                Vwz = self.extrapolated_wind_vel(self.V_wind)
                ax.scatter(
                    Vwz,
                    np.full_like(Vwz, z_rotor * 0.98),
                    c="k",
                    s=10,
                    alpha=0.6,
                    label="Ponto médio do rotor ($T = 16m$)",
                )
            except Exception:
                pass

        ax.axvspan(3.0, 11.0, color="grey", alpha=0.08, zorder=0)
        ax.set_xlabel("Velocidade do vento (m/s)")
        ax.set_ylabel("Altura (m)")
        ax.set_ylim(z_min, z_max)
        ax.set_title("Perfis de velocidade de vento $V_{10m}$ extrapolados")
        ax.grid(which="both", linestyle="--", alpha=0.5)
        ax.legend(fontsize=9, loc="best")

        if fname is None:
            fname = f"wind_profiles_calado_{self.draft}_method_{method}.png"
        out_path = os.path.join(out_folder, fname)
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        if show:
            plt.show()
        plt.close(fig)
        return out_path

    def plot_wind_extrapolation(
        self,
        v10_list,
        z_min=1.0,
        z_max=100.0,
        npoints=300,
        method="power",
        alpha=1 / 7,
        z0=0.03,
        z_ref=10.0,
        cmap="viridis",
        out_folder="figures",
        fname="wind_profile_extrapolation.png",
        show=True,
    ):

        os.makedirs(out_folder, exist_ok=True)
        heights = np.linspace(z_min, z_max, npoints)
        cmap_obj = plt.get_cmap(cmap)
        colors = [cmap_obj(i / max(1, len(v10_list) - 1)) for i in range(len(v10_list))]

        plt.style.use("seaborn-v0_8-muted")
        fig, ax = plt.subplots(figsize=(8, 6))

        for i, V10 in enumerate(v10_list):
            if method == "power":
                Vz = V10 * (heights / z_ref) ** alpha
                label = f"V(10m)={V10} m/s — power (α={alpha:.3f})"
            else:  # log-law
                # evita divisão por zero: garantir heights>z0
                zz = np.maximum(heights, z0 * 1.0001)
                denom = np.log(z_ref / z0)
                Vz = V10 * np.log(zz / z0) / denom
                label = f"V(10m)={V10} m/s — log (z0={z0})"

            ax.plot(Vz, heights, color=colors[i], lw=2, label=label)
            ax.scatter(
                [V10], [z_ref], color=colors[i], s=30, edgecolor="k", zorder=5
            )  # marca referência

        # destaque intervalo exemplo 3..11 m/s em grid vertical (opcional)
        ax.axvspan(3.0, 11.0, color="grey", alpha=0.08, zorder=0)

        ax.set_xlabel("Velocidade do vento (m/s)")
        ax.set_ylabel("Altura (m)")
        ax.set_ylim(z_min, z_max)
        ax.set_title("Perfis de velocidade extrapolados a partir de V(10m)")
        ax.grid(which="both", linestyle="--", alpha=0.5)
        ax.legend(fontsize=9, loc="best")
        plt.tight_layout()

        out_path = os.path.join(out_folder, fname)
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
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
        im1 = axes[0, 0].imshow(
            power_100,
            aspect="auto",
            cmap="viridis",
            extent=[0, 360, self.V_wind[0], self.V_wind[-1]],
            origin="lower",
        )
        axes[0, 0].set_title("Potência Consumida - 100 RPM")
        axes[0, 0].set_xlabel("Ângulo relativo (°)")
        axes[0, 0].set_ylabel("Velocidade do vento (m/s)")
        plt.colorbar(im1, ax=axes[0, 0], label="Potência (kW)")

        # 2. Heatmap 180 RPM
        im2 = axes[0, 1].imshow(
            power_180,
            aspect="auto",
            cmap="viridis",
            extent=[0, 360, self.V_wind[0], self.V_wind[-1]],
            origin="lower",
        )
        axes[0, 1].set_title("Potência Consumida - 180 RPM")
        axes[0, 1].set_xlabel("Ângulo relativo (°)")
        axes[0, 1].set_ylabel("Velocidade do vento (m/s)")
        plt.colorbar(im2, ax=axes[0, 1], label="Potência (kW)")

        # 3. Diferença absoluta
        power_diff = power_180 - power_100
        im3 = axes[0, 2].imshow(
            power_diff,
            aspect="auto",
            cmap="RdBu_r",
            extent=[0, 360, self.V_wind[0], self.V_wind[-1]],
            origin="lower",
        )
        axes[0, 2].set_title("Diferença (180 RPM - 100 RPM)")
        axes[0, 2].set_xlabel("Ângulo relativo (°)")
        axes[0, 2].set_ylabel("Velocidade do vento (m/s)")
        plt.colorbar(im3, ax=axes[0, 2], label="Diferença de Potência (kW)")

        # 4. Comparação por velocidade média
        mean_power_100 = np.mean(power_100, axis=1)
        mean_power_180 = np.mean(power_180, axis=1)

        axes[1, 0].plot(
            self.V_wind, mean_power_100, "b-", linewidth=2, label="100 RPM", marker="o"
        )
        axes[1, 0].plot(
            self.V_wind, mean_power_180, "r-", linewidth=2, label="180 RPM", marker="s"
        )
        axes[1, 0].set_xlabel("Velocidade do vento (m/s)")
        axes[1, 0].set_ylabel("Potência média (kW)")
        axes[1, 0].set_title("Potência Média por Velocidade")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 5. Comparação por ângulo médio
        mean_power_angle_100 = np.mean(power_100, axis=0)
        mean_power_angle_180 = np.mean(power_180, axis=0)

        axes[1, 1].plot(
            self.wind_angles, mean_power_angle_100, "b-", linewidth=2, label="100 RPM"
        )
        axes[1, 1].plot(
            self.wind_angles, mean_power_angle_180, "r-", linewidth=2, label="180 RPM"
        )
        axes[1, 1].set_xlabel("Ângulo relativo (°)")
        axes[1, 1].set_ylabel("Potência média (kW)")
        axes[1, 1].set_title("Potência Média por Ângulo")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # 6. Distribuição de potências
        axes[1, 2].hist(
            power_100.flatten(),
            bins=50,
            alpha=0.7,
            label="100 RPM",
            color="blue",
            density=True,
        )
        axes[1, 2].hist(
            power_180.flatten(),
            bins=50,
            alpha=0.7,
            label="180 RPM",
            color="red",
            density=True,
        )
        axes[1, 2].set_xlabel("Potência (kW)")
        axes[1, 2].set_ylabel("Densidade de Probabilidade")
        axes[1, 2].set_title("Distribuição de Potências")
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.suptitle(
            f"Comparativo de Potência Consumida: 100 RPM vs 180 RPM - {ship.upper()}",
            y=1.02,
            fontsize=16,
        )
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
                ceil_angle = search_angles[0]  # 0
        elif pos == len(search_angles):
            # ang is after the last angle (330), so it wraps around
            floor_angle = search_angles[-1]  # 330
            ceil_angle = search_angles[0]  # 0 (wraps around)
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

    def get_forces(self, ang, vel, force="fx"):
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
        ang_ceil_for_select = (
            0
            if float(ceil_angle) == 360 or float(ceil_angle) == 0
            else float(ceil_angle)
        )
        ang_ceil_for_interp = float(ceil_angle) if float(ceil_angle) != 0 else 360.0

        def get_sorted_rows(angle_sel, vel):
            if vel < 12.5:
                rotacao = 100
            else:
                rotacao = 180
            sel = self.thrust_df[
                (self.thrust_df["Angulo"] == angle_sel)
                & (self.thrust_df["Calado"] == self.draft)
                & (self.thrust_df["Rotacao"] == rotacao)
            ]
            if sel.empty:
                raise ValueError(f"No data for angle={angle_sel}, draft={self.draft}")
            sel = sel.sort_values("Vw")
            return sel

        # select rows AFTER angles resolved
        floor_sel = get_sorted_rows(int(ang_floor) % 360, vel)
        ceil_sel = get_sorted_rows(int(ang_ceil_for_select) % 360, vel)

        # Arrays for interpolation along Vw
        Vw_floor = floor_sel["Vw"].to_numpy(dtype=float)
        F_floor = floor_sel[force].to_numpy(dtype=float)
        Vw_ceil = ceil_sel["Vw"].to_numpy(dtype=float)
        F_ceil = ceil_sel[force].to_numpy(dtype=float)

        # If vel outside the Vw range, np.interp returns edge value (that's fine here).
        f_floor_at_vel = np.interp(vel, Vw_floor, F_floor)
        f_ceil_at_vel = np.interp(vel, Vw_ceil, F_ceil)

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
            denom = 0.5 * 1.2 * (vel**2) * self.Ax / 1000.0
            coef_x = f / denom if denom > 0 else np.nan
        if vel >= 6 and vel <= 12:
            return f, coef_x
        elif vel < 6:
            denom = 0.5 * 1.2 * (5.72**2) * self.Ax / 1000.0
        else:
            denom = 0.5 * 1.2 * (17.15**2) * self.Ax / 1000.0

        coef_x = f / denom if denom > 0 else np.nan
        new_f = coef_x * 0.5 * 1.2 * (vel**2) * self.Ax / 1000.0
        return new_f, coef_x

    def get_power(self, ang, vel):
        if ang < 60 or ang > 300:
            return 0
        floor_angle, ceil_angle = self.get_adjacent_angles(ang)
        if ceil_angle == 360:
            ceil_angle = 0
        if vel < 12.5:
            rotacao = 100
        else:
            rotacao = 180
        floor_moments = self.thrust_df[
            (self.thrust_df["Angulo"] == floor_angle)
            & (self.thrust_df["Calado"] == self.draft)
            & (self.thrust_df["Rotacao"] == rotacao)
        ]
        ceil_moments = self.thrust_df[
            (self.thrust_df["Angulo"] == ceil_angle)
            & (self.thrust_df["Calado"] == self.draft)
            & (self.thrust_df["Rotacao"] == rotacao)
        ]
        if vel < 12.5:
            P = np.mean([ceil_moments["Pcons"].iloc[0], floor_moments["Pcons"].iloc[0]])

        else:
            P = ceil_moments["Pcons"].iloc[0]

        return P

    def extrapolated_wind_vel(self, v_10):
        z = 26.5 if self.draft == 15.0 else 34.6
        alpha = 1 / 9
        return v_10 * np.power(z / 10, alpha)

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
        u_rel = u_wind - u_ship
        v_rel = v_wind
        # u_rel = u_rel * np.sin(wind_angle) + v_rel * np.cos(wind_angle)
        # v_rel = -u_rel * np.cos(wind_angle) + v_rel * np.sin(wind_angle)

        rel_ang = np.degrees(np.arctan2(-v_rel, -u_rel)) % 360

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
            self.Vk, ang = self.calculate_relative_ship_velocity(
                np.deg2rad(wind_angle), self.V_wz
            )
            # Fix the angle reference for CFD

            self.V_k_mat[:, i] = self.Vk
            self.rel_angls[:, i] = ang

            for j, (vel, angle) in enumerate(zip(self.V_wind, ang)):
                Pk = self.get_power(angle, vel)
                Fxk, coef_xk = self.get_forces(angle, vel, force="fx_rotores")
                if Fxk >= self.RT:
                    Fxk = self.RT

                self.forces_k[j][i] = Fxk
                self.coefs_xk[j][i] = coef_xk
                soma_forces += Fxk
                power_k[j][i] = Pk
                self.W_k[j][i] = self.imo_df.loc[j, str(wind_angle)]
                first_term += self.W_k[j][i]
                second_term += (self.V_ship / self.eta_D) * Fxk * self.W_k[j][i]
                third_term += Pk * self.W_k[j][i]

        f_eff_P_eff = (1 / first_term) * (second_term - third_term)

        print(f"Força: {second_term}")
        print(f"Consumo: {third_term}")
        print("Soma de forças", np.sum(self.forces_k[2:11]))
        print(f"f_eff_p_eff: {f_eff_P_eff}")

        print(f"Potência efetiva: {np.round(self.P_break - f_eff_P_eff)} kW")
        print(f"Ganho: {np.round(100*f_eff_P_eff/self.P_break)}%")
        return self.forces_k

    def plot_graphics(self):
        # Cx, Cz plots
        # pf.plot_cx_vs_velocity(self.V_wind, self.coefs_xk, angles=[30, 60, 90])
        # pf.polar_plot(self.wind_angles, self.V_wind, self.forces_rotor_k, "Curvas polares de c_x")
        pf.heatmap_plot(self.W_k, self.V_wind)
        pf.plot_velocity_comparison(self.draft)


def main():
    parser = argparse.ArgumentParser(description="Wind Route Creator")
    parser.add_argument("-s", "--ship", required=True, help="afra or suez")
    parser.add_argument("--plot", action="store_true", help="Plot graphics")
    parser.add_argument(
        "--modified-matrix",
        action="store_true",
        help="Translate velocities +3 m/s from probability matrix",
    )
    parser.add_argument("--cfd-data", required=True, help="CFD forces data path (csv)")

    args = parser.parse_args()
    ship = "abdias_suez" if args.ship == "suez" else "castro_alves_afra"

    imo_data_path = "data/global_prob_matrix.csv"
    forces_data_path = args.cfd_data
    drafts = [6.9, 15]
    force_matrix = []
    vk_matrix = []
    ang_matrix = []
    for draft in drafts:
        print(f"[INFO] Testing for draft = {draft}")
        get_gain = GainIMO(
            imo_data_path=imo_data_path,
            ship=ship,
            forces_data_path=forces_data_path,
            draft=draft,
            modified=args.modified_matrix,
        )
        force_matrix.append(get_gain.run())
        vk_matrix.append(get_gain.V_k_mat)
        ang_matrix.append(get_gain.rel_angls)

        if args.plot:
            get_gain.plot_graphics()

    get_gain.plot_wind_profiles()

    angles = np.arange(0, 150, 30)
    for angle in angles:
        pf.compare_forces_per_rotation(
            force_matrix, target_angles=[angle], vk_mats=vk_matrix, ang_mats=ang_matrix
        )


if __name__ == "__main__":
    main()
