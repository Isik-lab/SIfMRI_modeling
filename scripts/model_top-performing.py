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


def get_model_name(file_):
    if 'perturb' in file_:
        return file_.split('model-')[-1].split('_perturb')[0]
    else: 
        return file_.split('model-')[-1].split('.')[0]


def add_model_class(file_, df_):
    if 'Vision' in file_:
        df_['model_class'] = 'image'
    elif 'Video' in file_:
        df_['model_class'] = 'video'
    else:
        df_['model_class'] = 'language'
    return df_


class ModelTopPerforming:
    def __init__(self, args):
        self.process = 'ModelTopPerforming'
        print('updated 11:01 AM 1 May 2024')
        self.user = args.user
        self.model_class = args.model_class
        self.model_subpath = args.model_subpath
        self.voxel_id = 9539 #test voxel in EVC
        self.top_dir = f'{args.top_dir}/data/interim'
        print(vars(self))
        
        self.neural = True
        self.cols2keep = ['voxel_id', 'layer_relative_depth',
                        'train_score', 'test_score']
        self.out_path = f'{self.top_dir}/{self.process}'
        Path(self.out_path).mkdir(exist_ok=True, parents=True)
    
    def load_files(self, files):
        # Load the files and sum them up 
        df = None
        model_avg_performance = []
        n_final_files = 0
        for file in tqdm(files, total=len(files), desc='Loading files'): 
            try: 
                pkl = pd.read_pickle(file)
                pkl = pkl[['voxel_id', 'layer_relative_depth',
                        'train_score', 'test_score']]
                pkl = add_model_class(file, pkl)
                pkl['model_uid'] = get_model_name(file)
                model_avg_performance.append({'model_uid': get_model_name(file),
                                            'avg_score': pkl['test_score'].mean()})
                n_final_files += 1

                if df is None: 
                    df = pkl
                else:
                    #After the first file has been loaded, concatenate the data and add it together
                    df = pd.concat([df, pkl]).reset_index(drop=True)
                    idx = df.groupby('voxel_id')['train_score'].idxmax()
                    df = df.loc[idx].reset_index(drop=True)

            except KeyboardInterrupt:
                break
                print("Program was stopped by user.")
            except Exception as e:
                print(f"An unexpected error occurred loading {file.split('/')[-1]}: {e}")
        return df, n_final_files, pd.DataFrame(model_avg_performance)

    def run(self): 
        try:
            start_time = time.time()
            if self.model_subpath is not None: 
                file_path = f'{self.top_dir}/{self.model_class}/{self.model_subpath}/'
            else: 
                file_path = f'{self.top_dir}/{self.model_class}'

            files = glob(f'{file_path}/*.pkl.gz')
            print(f'{len(files)} files found')

            df, n_files, df_avg = self.load_files(files)
            print(f'{n_files} loaded successfully')

            save_start = time.time()
            df.to_csv(f'{self.out_path}/{self.model_class}_{self.model_subpath}.csv.gz', index=False)
            df_avg.to_csv(f'{self.out_path}/{self.model_class}_{self.model_subpath}_model-avg.csv', index=False)
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
    parser.add_argument('--model_class', type=str, default='VideoNeuralEncoding')
    parser.add_argument('--model_subpath', type=str, default=None)
    parser.add_argument('--top_dir', '-data', type=str,
                         default=f'/home/{user}/scratch4-lisik3/{user}/SIfMRI_modeling')
    args = parser.parse_args()
    ModelTopPerforming(args).run()


if __name__ == '__main__':
    main()

