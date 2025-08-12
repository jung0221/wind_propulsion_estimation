import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import argparse

class GetThrust:
    def __init__(self, timestamp, outbound_csv_path: str, return_csv_path: str, forces_path: str):
        self.timestamp = timestamp
        self.outbound_csv_path = outbound_csv_path
        self.return_csv_path = return_csv_path
        self.forces_path = forces_path
        self.current_month = None
        self.df_outbound = None
        self.df_return = None
        self.df_forces = None
        

    def load_data(self):

        print(f'[INFO] Loading route for time: {self.timestamp}')
        if self.current_month != self.timestamp.month or self.ds is None:
            print(f'[INFO] Changing month from {self.current_month} to {self.timestamp.month}')
            self.current_month = self.timestamp.month

        if not self.df_outbound:
            print('[INFO] Loading route data')
            self.df_outbound = pd.read_csv(os.path.join(self.outbound_csv_path, f'wind_data_year_{self.timestamp.year}_month_{self.timestamp.month}_day_{self.timestamp.day}_hour_{self.timestamp.hour}.csv'), sep=',')

        return_time = pd.Timestamp(self.df_outbound['time'].iloc[-1]) + pd.Timedelta(hours=1)
        if not self.df_return:
            print('[INFO] Loading route data')
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
        try:
            self.df_forces = pd.read_csv(self.forces_path)

    def run(self):
        self.load_data()


def main():
    parser = argparse.ArgumentParser(description='Wind Route Creator')
    parser.add_argument('--ship', help='afra or suez')
    args = parser.parse_args()
    ship = 'abdias_suez' if args.ship == 'suez' else 'castro_alves_afra' 

    current_time = pd.Timestamp('2020-01-01 00:00:00')
    outbound_csv_path = f'../{ship}/csvs_ida'
    return_csv_path = f'../{ship}/csvs_volta'
    forces_path = f'../{ship}forces.csv'
    while current_time.year <= 2020:

        get_thrust = GetThrust(timestamp=current_time,
                               outbound_csv_path=outbound_csv_path,
                               return_csv_path=return_csv_path,
                               forces_path=forces_path)
        get_thrust.run()
        current_time += pd.Timedelta(hours=1)

if __name__ == '__main__':
    main()