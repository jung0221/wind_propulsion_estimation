import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

Ax = 175*4
Ay = 175*4
lpp = 264

def polar_plot(x_axis, y_axis, z_axis, forces, calado, rotacao, title, rmin=None, rmax=None, coef=True):
    plt.style.use("seaborn-v0_8-muted")

    fig, ax = plt.subplots(subplot_kw=dict(projection="polar"), figsize=(10, 8))
    theta = np.deg2rad(x_axis)

    cmap = plt.get_cmap("viridis")
    n = max(1, len(y_axis))
    colors = [cmap(i / (n - 1)) for i in range(n)]

    for i, V in enumerate(y_axis):
        forces_x = z_axis[z_axis["Vw"] == V][forces].to_numpy(dtype=float)

        if coef:
            if forces in ("mz", "Mz_rotor"):
                denom = 0.5 * 1.2 * (V ** 2) * lpp * Ay
            else:
                denom = 0.5 * 1.2 * (V ** 2) * Ax / 1000.0
            z_plot = (forces_x / denom).tolist()
        else:
            z_plot = forces_x.tolist()

        # garantir comprimento compatível com theta
        if len(z_plot) == len(theta) - 1:
            z_plot.append(z_plot[0])
        elif len(z_plot) != len(theta):
            # tenta interpolar pelos ângulos disponíveis para produzir valores para cada theta
            angs = z_axis[z_axis["Vw"] == V]["Angulo"].to_numpy(dtype=float) % 360
            if angs.size >= 2:
                order = np.argsort(angs)
                angs_ord = angs[order]
                vals_ord = np.array(z_plot)[order]
                degs = np.rad2deg(theta)
                # interpolar com periodo 360
                z_plot = np.interp(degs, np.concatenate([angs_ord - 360, angs_ord, angs_ord + 360]),
                                   np.tile(vals_ord, 3)).tolist()
            else:
                # fallback: repetir o primeiro valor
                z_plot = [z_plot[0]] * len(theta)

        alpha_val = float(rotacao) * np.pi * 2.5 / (V * 30)
        label = rf"$\alpha = {alpha_val:.2f}$"
        ax.plot(theta, z_plot, color=colors[i], linewidth=2, alpha=0.95, label=label)
        # ax.fill_between(theta, z_plot, color=colors[i], alpha=0.12)

    # limites radiais
    if rmin is not None or rmax is not None:
        low = rmin if rmin is not None else ax.get_ylim()[0]
        high = rmax if rmax is not None else ax.get_ylim()[1]
        ax.set_ylim(low, high)

    # estética
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(1)
    thetas = np.arange(0, 360, 30)
    ax.set_thetagrids(thetas, labels=[f"{int(t)}°" for t in thetas], fontsize=12)
    ax.set_rlabel_position(135)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.5)
    # ax.set_title(title, va="bottom", fontsize=16, pad=18)

    # legenda externa com fundo semi-transparente
    leg = ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05), framealpha=0.85, fontsize=10)
    leg.get_frame().set_linewidth(0.4)

    os.makedirs("figures", exist_ok=True)
    fname = f"figures/coef_{forces}_calado_{calado}_rotacao_{rotacao}_V1.png" if coef else f"figures/{forces}_calado_{calado}_rotacao_{rotacao}_V1.png"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)



def forces_polar_plot(x_axis, y_axis, z_axis, forces, calado, rotacao, title, rmin=None, rmax=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)
    theta = np.deg2rad(x_axis)
    for rot in rotacao:
        z_rot = z_axis[z_axis["Rotacao"] == rot]
        for i in range(int(len(y_axis))):
            forces_x = z_rot[z_rot["Vw"] == y_axis[i]][forces].to_numpy()
            z_plot = list(forces_x)
            z_plot.append(z_plot[0])

            ax.plot(
                theta, z_plot, label=f"Rot = {rot}RPM, V = {y_axis[int(i)]:.1f} m/s"
            )
    if rmin is not None:
        ax.set_ylim(bottom=rmin)
    if rmax is not None:
        ax.set_ylim(top=rmax)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetagrids(range(0, 360, 30))  # Add this line for 30-degree steps
    ax.set_title(title, va="bottom", fontsize=20)
    ax.legend(loc="center right", bbox_to_anchor=(1.2, 0.1))
    fig.savefig(f"figures/{forces}_calado_{calado}.png")

    
forces_data_path = f"../abdias_suez/forces_CFD.csv"
thrust_df = pd.read_csv(forces_data_path)
case_1 = thrust_df[
    (thrust_df["Calado"] == 8.5) &
    (thrust_df["Rotacao"] == 100)
]
case_2 = thrust_df[
    (thrust_df["Calado"] == 16.0) &
    (thrust_df["Rotacao"] == 100)
]
case_3 = thrust_df[
    (thrust_df["Calado"] == 8.5) &
    (thrust_df["Rotacao"] == 180)
]
case_4 = thrust_df[
    (thrust_df["Calado"] == 16.0) &
    (thrust_df["Rotacao"] == 180)
]
x_axis = np.arange(0, 390, 30)
y_axis = [6, 10, 12]


case_forces_1 = thrust_df[
    (thrust_df["Calado"] == 8.5)
]
case_forces_2 = thrust_df[
    (thrust_df["Calado"] == 16.0)
]

# forces_polar_plot(x_axis, y_axis, case_forces_1, 'fx_rotores', 8.5, [100,180], f"$F_x$ (rotores) para calado {8.5}m e rotação {100}RPM")
# forces_polar_plot(x_axis, y_axis, case_forces_2, 'fx_rotores', 16, [100,180],  f"$F_x$ (rotores) para calado {16}m e rotação {100}RPM")

polar_plot(x_axis, y_axis, case_1, 'fx_rotores', 8.5, 100, f"C_x (rotores) para T = {8.5}m e omega = 100 RPM", 0, 9)
polar_plot(x_axis, y_axis, case_2, 'fx_rotores', 16, 100,  f"C_x (rotores) para T = {16}m  e omega = 100 RPM", 0, 9)
polar_plot(x_axis, y_axis, case_3, 'fx_rotores', 8.5, 180, f"C_x (rotores) para T = {8.5}m e omega = 180 RPM", 0, 9)
polar_plot(x_axis, y_axis, case_4, 'fx_rotores', 16, 180,  f"C_x (rotores) para T = {16}m  e omega = 180 RPM", 0, 9)
polar_plot(x_axis, y_axis, case_1, 'fy_rotores', 8.5, 100, f"C_y (rotores) para T = {8.5}m e omega = 100 RPM", 0, 9)
polar_plot(x_axis, y_axis, case_2, 'fy_rotores', 16, 100,  f"C_y (rotores) para T = {16}m  e omega = 100 RPM", 0, 9)
polar_plot(x_axis, y_axis, case_3, 'fy_rotores', 8.5, 180, f"C_y (rotores) para T = {8.5}m e omega = 180 RPM", 0, 9)
polar_plot(x_axis, y_axis, case_4, 'fy_rotores', 16, 180,  f"C_y (rotores) para T = {16}m  e omega = 180 RPM", 0, 9)
# polar_plot(x_axis, y_axis, case_1, 'Mz_rotor', 8.5, 100,   f"C_Z (rotores) para T = {8.5}m e omega = 100 RPM")
# polar_plot(x_axis, y_axis, case_2, 'Mz_rotor', 16, 100,    f"C_Z (rotores) para T = {16}m  e omega = 100 RPM")
# polar_plot(x_axis, y_axis, case_3, 'Mz_rotor', 8.5, 180,   f"C_Z (rotores) para T = {8.5}m e omega = 180 RPM")
# polar_plot(x_axis, y_axis, case_4, 'Mz_rotor', 16, 180,    f"C_Z (rotores) para T = {16}m  e omega = 180 RPM")


# polar_plot(x_axis, y_axis, case_1, 'fx', 8.5, 100, f"$C_x$ (total) para calado {8.5}m e rotação {100}RPM", -3, 5)
# polar_plot(x_axis, y_axis, case_2, 'fx', 16, 100,  f"$C_x$ (total) para calado {16}m e rotação {100}RPM" , -3, 5)
# polar_plot(x_axis, y_axis, case_3, 'fx', 8.5, 180, f"$C_x$ (total) para calado {8.5}m e rotação {180}RPM", -3, 5)
# polar_plot(x_axis, y_axis, case_4, 'fx', 16, 180,  f"$C_x$ (total) para calado {16}m e rotação {180}RPM" , -3, 5)
# polar_plot(x_axis, y_axis, case_1, 'fy', 8.5, 100, f"$C_y$ (total) para calado {8.5}m e rotação {100}RPM", -10, 10)
# polar_plot(x_axis, y_axis, case_2, 'fy', 16, 100,  f"$C_y$ (total) para calado {16}m e rotação {100}RPM" , -10, 10)
# polar_plot(x_axis, y_axis, case_3, 'fy', 8.5, 180, f"$C_y$ (total) para calado {8.5}m e rotação {180}RPM", -10, 10)
# polar_plot(x_axis, y_axis, case_4, 'fy', 16, 180,  f"$C_y$ (total) para calado {16}m e rotação {180}RPM" , -10, 10)
# polar_plot(x_axis, y_axis, case_1, 'mz', 8.5, 100, f"$M_Z$ (total) para calado {8.5}m e rotação {100}RPM")
# polar_plot(x_axis, y_axis, case_2, 'mz', 16, 100,  f"$M_Z$ (total) para calado {16}m e rotação {100}RPM" )  
# polar_plot(x_axis, y_axis, case_3, 'mz', 8.5, 180, f"$M_Z$ (total) para calado {8.5}m e rotação {180}RPM")
# polar_plot(x_axis, y_axis, case_4, 'mz', 16, 180,  f"$M_Z$ (total) para calado {16}m e rotação {180}RPM" )