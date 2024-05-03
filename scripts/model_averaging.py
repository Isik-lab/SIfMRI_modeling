import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
import time
import argparse
from src import tools
import pandas as pd
import numpy as np
from src.stats import calculate_p
pd.set_option('display.max_columns', None)


def sum_of_arrays(series):
    # Stack arrays vertically and compute sum along the first axis (rows)
    return np.nansum(np.vstack(series), axis=0)


def compute_confidence_intervals(arr):
    lower = np.nanpercentile(arr, 2.5)
    upper = np.nanpercentile(arr, 97.5)
    return lower, upper


def calculate_p_df(row):
    r_value = row['test_score']  # The 'r' value for the current row
    r_null_array = row['r_null_dist']  # The 'r_null' array for the current row
    return calculate_p(r_null_array, r_value, n_perm_=len(r_null_array), H0_='greater')


def divide_df(df, n): 
    def divide_array(arr):
        return arr / n

    # Get the mean by averaging by the total number of models
    columns_to_divide = ['train_score', 'test_score', 'layer_relative_depth', 'r_null_dist', 'r_var_dist']
    for col in columns_to_divide:
        if isinstance(df[col][0], np.ndarray):
            # Apply division to arrays
            df[col] = df[col].apply(divide_array)
        else:
            # Apply division to numeric columns
            df[col] = df[col] / n
    return df


class ModelAveraging:
    def __init__(self, args):
        self.process = 'ModelAveraging'
        print('updated 11:01 AM 1 May 2024')
        self.user = args.user
        self.model_class = args.model_class
        self.model_subpath = args.model_subpath
        self.voxel_id = 9539 #test voxel in EVC
        self.top_dir = f'{args.top_dir}/data/interim'

        self.out_path = f'{self.top_dir}/{self.process}'
        Path(self.out_path).mkdir(exist_ok=True, parents=True)
        self.cols2keep = ['voxel_id', 'roi_name', 'layer_relative_depth',
                          'train_score', 'test_score',
                          'r_null_dist', 'r_var_dist']
        print(vars(self))
    
    def load_files(self, files):
        # Load the files and sum them up 
        df = None
        n_final_files = 0
        for file in tqdm(files, total=len(files), desc='Loading files'): 
            try: 
                pkl = pd.read_pickle(file)[self.cols2keep]
                n_final_files += 1

                # remove voxels not in roi
                if 'roi_name' in pkl.columns: 
                    pkl = pkl.loc[pkl.roi_name != 'none'].reset_index(drop=True)
                pkl.drop(columns=['roi_name'], inplace=True)

                if df is None: 
                    df = pkl
                else:
                    #After the first file has been loaded, concatenate the data and add it together
                    df = pd.concat([df, pkl])
                    df = df.groupby('voxel_id').agg({
                                                    'train_score': 'sum',
                                                    'test_score': 'sum',
                                                    'layer_relative_depth': 'sum',
                                                    'r_null_dist': sum_of_arrays,
                                                    'r_var_dist': sum_of_arrays
                                                    }).reset_index()
                    break
            except:
                print(f'{self.process} could not load {file.split('/')[-1]}')
        return df, n_final_files

    def run(self): 
        try:
            start_time = time.time()
            if self.model_subpath is not None: 
                file_path = f'{self.top_dir}/{self.model_class}/{self.model_subpath}/'
            else: 
                file_path = f'{self.top_dir}/{self.model_class}'

            files = glob(f'{file_path}/*.pkl.gz')
            n_files = len(files)
            print(f'{len(files)} files found')

            df, n_files = self.load_files(files)
            print(df.loc[df.voxel_id == self.voxel_id])

            df = divide_df(df, n_files)
            df['n_models'] = n_files # Add number info to the data
            print(df.loc[df.voxel_id == self.voxel_id])

            # calculate the confidence interval
            df[['lower_ci', 'upper_ci']] = df['r_var_dist'].apply(lambda arr: pd.Series(compute_confidence_intervals(arr)))
            print(df.loc[df.voxel_id == self.voxel_id])

            # calculate the p value
            df['p'] = df.apply(calculate_p_df, axis=1)
            print(df.loc[df.voxel_id == self.voxel_id])

            save_start = time.time()
            df.to_pickle(f'{self.out_path}/{self.model_class}_{self.model_subpath}.pkl.gz')
            save_time = time.time() - save_start
            elapsed = time.strftime("%H:%M:%S", time.gmtime(save_time))
            print(f'Saved in {elapsed}!')

            end_time = time.time()
            elapsed = end_time - start_time
            elapsed = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            print(f'Finished in {elapsed}!')
            tools.send_slack(f'Finished: {self.process} in {elapsed}', channel=self.user)
        except Exception as err:
            print(f'Error: {self.process} Error Msg = {err}')
            tools.send_slack(f'Error: {self.process} Error Msg = {err}', channel=self.user)


def main():
    parser = argparse.ArgumentParser()
    # Add arguments that are needed before setting the default for data_dir
    parser.add_argument('--user', type=str, default='emcmaho7')
    # Parse known args first to get the user
    args, remaining_argv = parser.parse_known_args()
    user = args.user  # Get the user from the parsed known args
    parser.add_argument('--model_class', type=str, default='VisionNeuralEncoding')
    parser.add_argument('--model_subpath', type=str, default=None)
    parser.add_argument('--top_dir', '-data', type=str,
                         default=f'/home/{user}/scratch4-lisik3/{user}/SIfMRI_modeling')
    args = parser.parse_args()
    ModelAveraging(args).run()


if __name__ == '__main__':
    main()
