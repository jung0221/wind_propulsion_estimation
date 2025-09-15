import pandas as pd
import numpy as np
import bisect
import argparse

class GainIMO:
    def __init__(self, xls_data_path, imo_data_path, forces_data_path, rotation):
        xls_csv = pd.read_excel(xls_data_path, sheet_name='Ganho') 
        p_cons = xls_csv['P_rotor kW']       
        self.imo_df = pd.read_csv(imo_data_path)
        self.thrust_df = pd.read_csv(forces_data_path).iloc[:,2:]
        self.thrust_df['Pcons'] = p_cons
        self.moments = self.thrust_df['Mz_popa_bom'] + self.thrust_df['Mz_popa_bor'] + self.thrust_df['Mz_proa_bom'] + self.thrust_df['Mz_proa_bor']
        angle_list = self.thrust_df['Angulo'].unique()
        self.angle_list = np.append(angle_list, 360)

        self.V_ref = 12 #knots
        self.eta_D = 0.7
        self.draft = 16.0 # meters
        self.rotation = int(rotation) #RPM
        self.Ax = 1130
        self.Ay = 3300
        self.P_eff = 6560 # kW

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


    def get_forces(self, ang, vel, force='fx', coef_dir='cx'):
        floor_angle, ceil_angle = self.get_adjacent_angles(ang)
        if ceil_angle == 360: ceil_angle = 0
        floor_forces = self.thrust_df[
            (self.thrust_df['Angulo'] == floor_angle) &
            (self.thrust_df['Calado'] == self.draft) &
            (self.thrust_df['Rotacao'] == self.rotation)
        ]

        ceil_forces = self.thrust_df[
            (self.thrust_df['Angulo'] == ceil_angle) &
            (self.thrust_df['Calado'] == self.draft) &
            (self.thrust_df['Rotacao'] == self.rotation)
        ]
        if vel >= 6 and vel <= 10: 
            fx_ceil = ceil_forces[force].iloc[0] + (vel - ceil_forces['Vw'].iloc[0])*(ceil_forces[force].iloc[1] - ceil_forces[force].iloc[0])/(ceil_forces['Vw'].iloc[1] - ceil_forces['Vw'].iloc[0])
            fx_floor = floor_forces[force].iloc[0] + (vel - floor_forces['Vw'].iloc[0])*(floor_forces[force].iloc[1] - floor_forces[force].iloc[0])/(floor_forces['Vw'].iloc[1] - floor_forces['Vw'].iloc[0])
            if ceil_angle != floor_angle:
                f = fx_floor + (ang - floor_angle)*(fx_ceil - fx_floor)/(ceil_angle - floor_angle)
            else: f = fx_floor

        elif vel > 10 and vel <= 12: 
            fx_ceil = ceil_forces[force].iloc[1] + (vel - ceil_forces['Vw'].iloc[1])*(ceil_forces[force].iloc[2] - ceil_forces[force].iloc[1])/(ceil_forces['Vw'].iloc[2] - ceil_forces['Vw'].iloc[1])
            fx_floor = floor_forces[force].iloc[1] + (vel - floor_forces['Vw'].iloc[1])*(floor_forces[force].iloc[2] - floor_forces[force].iloc[1])/(floor_forces['Vw'].iloc[2] - floor_forces['Vw'].iloc[1])
            if ceil_angle != floor_angle:
                f = fx_floor + (ang - floor_angle)*(fx_ceil - fx_floor)/(ceil_angle - floor_angle)
            else: f = fx_floor

        elif vel < 6:
            coef_x_floor = floor_forces[floor_forces['Vw'] == 6][coef_dir].values[0]
            coef_x_ceil = ceil_forces[ceil_forces['Vw'] == 6][coef_dir].values[0]
            if ceil_angle != floor_angle:
                coef_x = coef_x_floor + (ang - floor_angle)*(coef_x_ceil - coef_x_floor)/(ceil_angle - floor_angle)
            else: coef_x = coef_x_floor
            f = coef_x * 0.5 * 1.2 * np.power(vel, 2) * self.Ax / 1000

        elif vel > 12:
            coef_x_floor = floor_forces[floor_forces['Vw'] == 12][coef_dir].values[0]
            coef_x_ceil = ceil_forces[ceil_forces['Vw'] == 12][coef_dir].values[0]
            if ceil_angle != floor_angle:
                coef_x = coef_x_floor + (ang - floor_angle)*(coef_x_ceil - coef_x_floor)/(ceil_angle - floor_angle)
            else: coef_x = coef_x_floor
            f = coef_x * 0.5 * 1.2 * np.power(vel, 2) * self.Ax / 1000
        return f

    def get_power(self, ang, vel):
        floor_angle, ceil_angle = self.get_adjacent_angles(ang)
        if ceil_angle == 360: ceil_angle = 0
        floor_moments = self.thrust_df[
            (self.thrust_df['Angulo'] == floor_angle) &
            (self.thrust_df['Calado'] == self.draft) &
            (self.thrust_df['Rotacao'] == self.rotation)
        ]

        ceil_moments = self.thrust_df[
            (self.thrust_df['Angulo'] == ceil_angle) &
            (self.thrust_df['Calado'] == self.draft) &
            (self.thrust_df['Rotacao'] == self.rotation)
        ]
        if vel >= 6 and vel <= 10: 
            P_ceil = ceil_moments['Pcons'].iloc[0] + (vel - ceil_moments['Vw'].iloc[0])*(ceil_moments['Pcons'].iloc[1] - ceil_moments['Pcons'].iloc[0])/(ceil_moments['Vw'].iloc[1] - ceil_moments['Vw'].iloc[0])
            P_floor = floor_moments['Pcons'].iloc[0] + (vel - floor_moments['Vw'].iloc[0])*(floor_moments['Pcons'].iloc[1] - floor_moments['Pcons'].iloc[0])/(floor_moments['Vw'].iloc[1] - floor_moments['Vw'].iloc[0])
            if ceil_angle != floor_angle:
                P = P_floor + (ang - floor_angle)*(P_ceil - P_floor)/(ceil_angle - floor_angle)
            else: P = P_floor

        elif vel > 10 and vel <= 12: 
            P_ceil = ceil_moments['Pcons'].iloc[1] + (vel - ceil_moments['Vw'].iloc[1])*(ceil_moments['Pcons'].iloc[2] - ceil_moments['Pcons'].iloc[1])/(ceil_moments['Vw'].iloc[2] - ceil_moments['Vw'].iloc[1])
            P_floor = floor_moments['Pcons'].iloc[1] + (vel - floor_moments['Vw'].iloc[1])*(floor_moments['Pcons'].iloc[2] - floor_moments['Pcons'].iloc[1])/(floor_moments['Vw'].iloc[2] - floor_moments['Vw'].iloc[1])
            if ceil_angle != floor_angle:
                P = P_floor + (ang - floor_angle)*(P_ceil - P_floor)/(ceil_angle - floor_angle)
            else: P = P_floor

        elif vel < 6:
            P = np.mean([ceil_moments['Pcons'].iloc[0], floor_moments['Pcons'].iloc[0]])

        elif vel > 12:
            P = np.mean([ceil_moments['Pcons'].iloc[2], floor_moments['Pcons'].iloc[2]])
        return P

    def calculate_effective_power(self):
        matrix_cols = self.imo_df.columns[1:]
        
        first_term = 0
        second_term = 0
        third_term = 0
        for i in range(26):
            vel = self.imo_df['vel'].iloc[i]
            for angle in matrix_cols:
                W_k = self.imo_df.loc[i, angle]
                first_term += W_k
                if int(angle) <= 180: angle_cfd = 180 - int(angle)
                elif int(angle) > 180: 540 - angle_cfd
                F = self.get_forces(int(angle_cfd), vel)
                second_term += (0.5144*self.V_ref/self.eta_D) * F * W_k 
                P = self.get_power(int(angle_cfd), vel)
                third_term += P * W_k
                
        f_eff_P_eff = (1/first_term) * (second_term - third_term)
        print(f"Ganho: {np.round(100*f_eff_P_eff/self.P_eff)}%")
        
    
def main():
    parser = argparse.ArgumentParser(description="Wind Route Creator")
    parser.add_argument("--ship", required=True, help="afra or suez")
    parser.add_argument("--rotation", required=True, help="100 or 180")
    
    args = parser.parse_args()
    ship = "abdias_suez" if args.ship == "suez" else "castro_alves_afra"

    xls_data_path = "../abdias_suez/CFD_Jung.xlsx"
    imo_data_path = "../imo_guidance/global_prob_matrix.csv"
    forces_data_path = "../abdias_suez/forces_V3.csv"

    get_gain = GainIMO(xls_data_path=xls_data_path, imo_data_path=imo_data_path, forces_data_path=forces_data_path, rotation=args.rotation)
    get_gain.calculate_effective_power()

if __name__ == "__main__":
    main()
