import numpy as np
import matplotlib.pyplot as plt
import os

V_wind = np.arange(1, 26)
wind_angles = np.arange(0, 360, 5)
V_ship = 12 * 0.5144  # knots to m/s

def compare_forces_per_rotation(mats, target_angles=[0, 90, 180], vk_mats=None, ang_mats=None):
    """
    Plot total force vs wind speed for specific angles with relative velocity and angle annotations
    
    Args:
        mats: List of force matrices
        target_angles: List of angles in degrees to include in the sum
        vk_mats: List of relative velocity matrices (same order as mats)
        ang_mats: List of relative angle matrices (same order as mats)
    """
    speed_slice = slice(0, 25)
    V = np.asarray(V_wind)[speed_slice]

    # Find indices of target angles in wind_angles array
    angle_indices = []
    actual_angles = []
    for target_angle in target_angles:
        idx = np.argmin(np.abs(wind_angles - target_angle))
        angle_indices.append(idx)
        actual_angles.append(wind_angles[idx])
        print(f"[INFO] Target angle {target_angle}° -> Index {idx} (actual angle: {wind_angles[idx]}°)")

    mats = [np.asarray(m) for m in mats]
    if vk_mats is not None:
        vk_mats = [np.asarray(m) for m in vk_mats]
    if ang_mats is not None:
        ang_mats = [np.asarray(m) for m in ang_mats]

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

@staticmethod
def heatmap_plot(matrix, vels):
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
    plt.title("Mapa de calor da força em função da velocidade e ângulo")
    plt.show()    
    

# Não faz sentido usar curvas polares para velocidades de vento
@staticmethod
def polar_plot(x_axis, y_axis, z_axis, title):
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


def extrapolated_wind_vel(v_10, draft):
    z = 27.7 if draft == 16.0 else 35.5 # 7.2 (Depth - Draft: 7.2 for design and 14.7 for ballast) + 3 (Base height) + 17.5 (Rotor height/2)
    alpha = 1 / 9
    return v_10 * np.power(z / 10, alpha)


def calculate_relative_ship_velocity(wind_angle, V_wz):
    """
    Calculate relative wind velocity using vector components.
    Positive values = headwind, Negative values = tailwind
    
    rel_angle: angle in radians where wind is coming from (0 = from ahead)
    V_wz: extrapolated wind velocity
    """
    u_ship = V_ship 
    
    u_wind = V_wz * -np.cos(wind_angle)
    v_wind = V_wz * -np.sin(wind_angle)
    
    u_rel = u_ship - u_wind 
    v_rel = -v_wind
    
    rel_ang = np.degrees(np.arctan2(v_rel, u_rel)) % 360

    return np.sqrt(np.power(u_rel, 2) + np.power(v_rel, 2)), rel_ang

@staticmethod
def plot_velocity_comparison(draft):
    """Plot comparison between relative velocity (Vk), extrapolated wind velocity, and absolute wind velocity"""
    
    sample_angles = [0, 45, 90, 135, 180, 225, 270, 315] 
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    V_wind_extrap = extrapolated_wind_vel(V_wind, draft)  # Velocidade extrapolada
    
    for idx, angle in enumerate(sample_angles):
        ax = axes[idx]
        
        # Calcular velocidade relativa para este ângulo
        Vk, ang = calculate_relative_ship_velocity(np.deg2rad(angle), V_wind_extrap)
        # Plot das três velocidades
        ax.plot(V_wind, V_wind, 'b-', linewidth=2, label='V absoluta')
        ax.plot(V_wind, V_wind_extrap, 'g--', linewidth=2, label=f'V extrapolada')
        ax.plot(V_wind, Vk, 'r-', linewidth=2, label=f'V relativa')
        
        ax.set_xlabel('Velocidade do vento absoluta (m/s)')
        ax.set_ylabel('Velocidade (m/s)')
        ax.set_title(f'Ângulo absoluto do vento: {angle}°')
        ax.grid(True, alpha=0.3)
        ax.legend(loc="center right", bbox_to_anchor=(0.8, 0.1))
        # Adicionar informações estatísticas
        mean_diff_abs = np.mean(Vk - V_wind)
        mean_diff_extrap = np.mean(Vk - V_wind_extrap)
        ax.text(0.05, 0.95, f'Δ abs: {mean_diff_abs:.1f}\nΔ extrap: {mean_diff_extrap:.1f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.suptitle('Comparação de Velocidades: Absoluta vs Extrapolada vs Relativa', y=1.02, fontsize=16)
    plt.show()


@staticmethod
def plot_cx_vs_velocity(V_wind, coefs_xk, angles=None):
    """
    Plot Cx (self.coefs_xk) behaviour along wind speed.
    - If angles is None: plot mean +/- std across all angles.
    - If angles is a list of angles (degrees): plot Cx for those angles.
    """

    plt.figure(figsize=(8, 5))
    if angles is None:
        mean_cx = np.nanmean(coefs_xk, axis=1)
        std_cx = np.nanstd(coefs_xk, axis=1)
        plt.plot(V_wind, mean_cx, color="black", lw=2, label="mean Cx (all angles)")
        plt.fill_between(V_wind, mean_cx - std_cx, mean_cx + std_cx, color="gray", alpha=0.3,
                            label="±1 std")
    else:
        # find nearest available angle indices
        idxs = [int(np.argmin(np.abs(wind_angles - a))) for a in angles]
        for idx in idxs:
            plt.plot(V_wind, coefs_xk[:, idx], lw=2, label=f"angle {int(wind_angles[idx])}°")

    plt.xlabel("Wind speed (m/s)", fontsize=12)
    plt.ylabel("Cx (coeficient)", fontsize=12)
    plt.title("Cx vs Wind speed", fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()