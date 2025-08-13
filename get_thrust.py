import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import argparse
import bisect

class GetThrust:
    def __init__(self, outbound_csv_path: str, return_csv_path: str, forces_path: str, rotation: int):
        self.outbound_csv_path = outbound_csv_path
        self.return_csv_path = return_csv_path
        self.forces_path = forces_path
        self.rotation = rotation
        self.timestamp = None
        self.current_month = None
        self.df_outbound = None
        self.df_return = None
        self.df_forces = None
        self.hull_force = None
        self.draft = None
        self.Ax = 1130
        self.Ay = 3300
        


    def load_data(self):

        print(f'[INFO] Loading route for time: {self.timestamp}')
        if self.current_month != self.timestamp.month is None:
            print(f'[INFO] Changing month from {self.current_month} to {self.timestamp.month}')
            self.current_month = self.timestamp.month

        print('[INFO] Loading outbound route data')
        try:
            self.df_outbound = pd.read_csv(os.path.join(self.outbound_csv_path, f'wind_data_year_{self.timestamp.year}_month_{self.timestamp.month}_day_{self.timestamp.day}_hour_{self.timestamp.hour}.csv'), sep=',')
        except FileNotFoundError as e:
            print(f'[ERROR] {e}')
            print('[WARNING] Continuing without return data')
        except pd.errors.EmptyDataError:
            print(f'[ERROR] Data file is empty')
        except Exception as e:
            print(f'[ERROR] Unexpected error loading return data: {e}')
        return_time = pd.Timestamp(self.df_outbound['time'].iloc[-1]) + pd.Timedelta(hours=1)
        print('[INFO] Loading return route data')
        try:
            self.df_return = pd.read_csv(os.path.join(self.return_csv_path, f'wind_data_year_{return_time.year}_month_{return_time.month}_day_{return_time.day}_hour_{return_time.hour}.csv'), sep=',')
        except FileNotFoundError as e:
            print(f'[ERROR] {e}')
            print('[WARNING] Continuing without return data')
        except pd.errors.EmptyDataError:
            print(f'[ERROR] Data file is empty')
        except Exception as e:
            print(f'[ERROR] Unexpected error loading return data: {e}')

    def load_forces(self):
        if not self.df_forces:
            try:
                print("[INFO] Loading thrust data")
                self.df_forces = pd.read_csv(self.forces_path)
            except FileNotFoundError as e:
                print(f'[ERROR] {e}')
                print('[WARNING] Continuing without return data')
            except pd.errors.EmptyDataError:
                print(f'[ERROR] Data file is empty')
            except Exception as e:
                print(f'[ERROR] Unexpected error loading return data: {e}')

    def get_values(self, option='outbound'):

        if option == 'outbound':
            df = self.df_outbound.copy()     
        else: 
            df = self.df_return.copy()

        self.draft = 16
        self.angles = df['angle_rel']
        self.vels = np.sqrt(np.power(df['u_rel'], 2) + np.power(df['v_rel'], 2))

    def get_adjacent_angles(self, ang):
        # Find insertion position

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

    def get_forces(self, ang, vel, dir='x'):
        force_dir = 'fx' if dir == 'x' else 'fy'
        coef_dir = 'cx' if dir == 'x' else 'cy'
        floor_angle, ceil_angle = self.get_adjacent_angles(ang)
        if ceil_angle == 360: ceil_angle = 0
        floor_forces = self.df_forces[
            (self.df_forces['Angulo'] == floor_angle) &
            (self.df_forces['Calado'] == self.draft) &
            (self.df_forces['Rotacao'] == self.rotation)
        ]

        ceil_forces = self.df_forces[
            (self.df_forces['Angulo'] == ceil_angle) &
            (self.df_forces['Calado'] == self.draft) &
            (self.df_forces['Rotacao'] == self.rotation)
        ]
        if vel >= 6 and vel <= 10: 
            fx_ceil = ceil_forces[force_dir].iloc[0] + (vel - ceil_forces['Vw'].iloc[0])*(ceil_forces[force_dir].iloc[1] - ceil_forces[force_dir].iloc[0])/(ceil_forces['Vw'].iloc[1] - ceil_forces['Vw'].iloc[0])
            fx_floor = floor_forces[force_dir].iloc[0] + (vel - floor_forces['Vw'].iloc[0])*(floor_forces[force_dir].iloc[1] - floor_forces[force_dir].iloc[0])/(floor_forces['Vw'].iloc[1] - floor_forces['Vw'].iloc[0])
            f = fx_floor + (ang - floor_angle)*(fx_ceil - fx_floor)/(ceil_angle - floor_angle)

        elif vel > 10 and vel <= 12: 
            fx_ceil = ceil_forces[force_dir].iloc[1] + (vel - ceil_forces['Vw'].iloc[1])*(ceil_forces[force_dir].iloc[2] - ceil_forces[force_dir].iloc[1])/(ceil_forces['Vw'].iloc[2] - ceil_forces['Vw'].iloc[1])
            fx_floor = floor_forces[force_dir].iloc[1] + (vel - floor_forces['Vw'].iloc[1])*(floor_forces[force_dir].iloc[2] - floor_forces[force_dir].iloc[1])/(floor_forces['Vw'].iloc[2] - floor_forces['Vw'].iloc[1])
            f = fx_floor + (ang - floor_angle)*(fx_ceil - fx_floor)/(ceil_angle - floor_angle)

        elif vel < 6:
            coef_x_floor = floor_forces[floor_forces['Vw'] == 6][coef_dir].values
            coef_x_ceil = ceil_forces[ceil_forces['Vw'] == 6][coef_dir].values
            coef_x = coef_x_floor + (ang - floor_angle)*(coef_x_ceil - coef_x_floor)/(ceil_angle - floor_angle)
            f = coef_x * 0.5 * 1.2 * np.power(vel, 2) * self.Ax / 1000

        elif vel > 12:
            coef_x_floor = floor_forces[floor_forces['Vw'] == 12][coef_dir].values
            coef_x_ceil = ceil_forces[ceil_forces['Vw'] == 12][coef_dir].values
            coef_x = coef_x_floor + (ang - floor_angle)*(coef_x_ceil - coef_x_floor)/(ceil_angle - floor_angle)
            f = coef_x * 0.5 * 1.2 * np.power(vel, 2) * self.Ax / 1000

        return f

    def run(self, timestamp): 
        self.timestamp = timestamp
        self.load_data()
        
        self.angle_list = self.df_forces['Angulo'].unique()
        self.angle_list = np.append(self.angle_list, 360)
        
        self.get_values(option='outbound')
        for ang, vel in tqdm(zip(self.angles, self.vels), desc='outbound'):
            fx = self.get_forces(ang, vel, 'x')
            fy = self.get_forces(ang, vel, 'y')

        pbar = tqdm(len(self.angles))
        self.get_values(option='return')
        for ang, vel in tqdm(zip(self.angles, self.vels), desc='return'):
            fx = self.get_forces(ang, vel, 'x')
            fy = self.get_forces(ang, vel, 'y')
            


def main():
    parser = argparse.ArgumentParser(description='Wind Route Creator')
    parser.add_argument('--ship', help='afra or suez')
    parser.add_argument('--rotation', help='100 or 180')
    args = parser.parse_args()
    ship = 'abdias_suez' if args.ship == 'suez' else 'castro_alves_afra' 

    current_time = pd.Timestamp('2020-01-01 00:00:00')
    outbound_csv_path = f'../{ship}/csvs_ida'
    return_csv_path = f'../{ship}/csvs_volta'
    forces_path = f'../{ship}/forces_V2.csv'
    get_thrust = GetThrust(outbound_csv_path=outbound_csv_path,
                            return_csv_path=return_csv_path,
                            forces_path=forces_path,
                            rotation=int(args.rotation))
    
    get_thrust.load_forces()

    while current_time.year <= 2020:
        get_thrust.run(current_time)
        current_time += pd.Timedelta(hours=1)

if __name__ == '__main__':
    main()