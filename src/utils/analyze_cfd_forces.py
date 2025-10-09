import pandas as pd
import numpy as np
import bisect
import argparse
import matplotlib.pyplot as plt

Ax = 1450
Ay = 5500
lpp = 264

def polar_plot(x_axis, y_axis, z_axis, forces, calado, rotacao, title, rmin=None, rmax=None, coef=True):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)
    theta = np.deg2rad(x_axis)
    for i in range(int(len(y_axis))):
        forces_x = z_axis[z_axis["Vw"] == y_axis[i]][forces].to_numpy()
        if coef: 
            if forces == 'mz' or forces == 'Mz_rotor': 
                z_plot = list(forces_x / (0.5 * 1.2 * np.power(y_axis[i], 2) * lpp * Ay))
            else:
                z_plot = list(forces_x / (0.5 * 1.2 * np.power(y_axis[i], 2) * Ax / 1000))
        else: 
            z_plot = list(forces_x)
        z_plot.append(z_plot[0])

        ax.plot(
            theta, z_plot, label=f"V = {y_axis[int(i)]:.1f} m/s"
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
    if coef:
        fig.savefig(f"figures/coef_{forces}_calado_{calado}_rotacao_{rotacao}.png")
    else: 
        fig.savefig(f"figures/{forces}_calado_{calado}_rotacao_{rotacao}.png")

        
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
thrust_df = pd.read_csv(forces_data_path).iloc[:, 2:]
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

forces_polar_plot(x_axis, y_axis, case_forces_1, 'fx_rotores', 8.5, [100,180], f"F_x (rotores) para calado {8.5}m e rotação {100}RPM")
forces_polar_plot(x_axis, y_axis, case_forces_2, 'fx_rotores', 16, [100,180],  f"F_x (rotores) para calado {16}m e rotação {100}RPM")

polar_plot(x_axis, y_axis, case_1, 'fx_rotores', 8.5, 100, f"C_x (rotores) para calado {8.5}m e rotação {100}RPM",  -3, 4)
polar_plot(x_axis, y_axis, case_2, 'fx_rotores', 16, 100,  f"C_x (rotores) para calado {16}m e rotação {100}RPM"  , -3, 4)
polar_plot(x_axis, y_axis, case_3, 'fx_rotores', 8.5, 180, f"C_x (rotores) para calado {8.5}m e rotação {180}RPM",  -3, 4)
polar_plot(x_axis, y_axis, case_4, 'fx_rotores', 16, 180,  f"C_x (rotores) para calado {16}m e rotação {180}RPM"  , -3, 4)
polar_plot(x_axis, y_axis, case_1, 'fy_rotores', 8.5, 100, f"C_y (rotores) para calado {8.5}m e rotação {100}RPM",  -10, 10)
polar_plot(x_axis, y_axis, case_2, 'fy_rotores', 16, 100,  f"C_y (rotores) para calado {16}m e rotação {100}RPM"  , -10, 10)
polar_plot(x_axis, y_axis, case_3, 'fy_rotores', 8.5, 180, f"C_y (rotores) para calado {8.5}m e rotação {180}RPM",  -10, 10)
polar_plot(x_axis, y_axis, case_4, 'fy_rotores', 16, 180,  f"C_y (rotores) para calado {16}m e rotação {180}RPM"  , -10, 10)
polar_plot(x_axis, y_axis, case_1, 'Mz_rotor', 8.5, 100,   f"C_Z (rotores) para calado {8.5}m e rotação {100}RPM"   )
polar_plot(x_axis, y_axis, case_2, 'Mz_rotor', 16, 100,    f"C_Z (rotores) para calado {16}m e rotação {100}RPM"    )
polar_plot(x_axis, y_axis, case_3, 'Mz_rotor', 8.5, 180,   f"C_Z (rotores) para calado {8.5}m e rotação {180}RPM"   )
polar_plot(x_axis, y_axis, case_4, 'Mz_rotor', 16, 180,    f"C_Z (rotores) para calado {16}m e rotação {180}RPM"    )

polar_plot(x_axis, y_axis, case_1, 'fx', 8.5, 100, f"C_x (total) para calado {8.5}m e rotação {100}RPM", -3, 5)
polar_plot(x_axis, y_axis, case_2, 'fx', 16, 100,  f"C_x (total) para calado {16}m e rotação {100}RPM" , -3, 5)
polar_plot(x_axis, y_axis, case_3, 'fx', 8.5, 180, f"C_x (total) para calado {8.5}m e rotação {180}RPM", -3, 5)
polar_plot(x_axis, y_axis, case_4, 'fx', 16, 180,  f"C_x (total) para calado {16}m e rotação {180}RPM" , -3, 5)
polar_plot(x_axis, y_axis, case_1, 'fy', 8.5, 100, f"C_y (total) para calado {8.5}m e rotação {100}RPM", -10, 10)
polar_plot(x_axis, y_axis, case_2, 'fy', 16, 100,  f"C_y (total) para calado {16}m e rotação {100}RPM" , -10, 10)
polar_plot(x_axis, y_axis, case_3, 'fy', 8.5, 180, f"C_y (total) para calado {8.5}m e rotação {180}RPM", -10, 10)
polar_plot(x_axis, y_axis, case_4, 'fy', 16, 180,  f"C_y (total) para calado {16}m e rotação {180}RPM" , -10, 10)
polar_plot(x_axis, y_axis, case_1, 'mz', 8.5, 100, f"M_Z (total) para calado {8.5}m e rotação {100}RPM")
polar_plot(x_axis, y_axis, case_2, 'mz', 16, 100,  f"M_Z (total) para calado {16}m e rotação {100}RPM" )  
polar_plot(x_axis, y_axis, case_3, 'mz', 8.5, 180, f"M_Z (total) para calado {8.5}m e rotação {180}RPM")
polar_plot(x_axis, y_axis, case_4, 'mz', 16, 180,  f"M_Z (total) para calado {16}m e rotação {180}RPM" )