import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
import time
import argparse
from src import tools
import pandas as pd
import numpy as np


def add_perturbation(file_, df_):
    if 'no_perturbation' in file_:
        df_['perturbation'] = 'original'
    elif 'mask_nonverbs' in file_:
        df_['perturbation'] = 'only verbs'
    elif 'mask_nonnouns' in file_:
        df_['perturbation'] = 'only nouns'
    elif 'mask_nouns' in file_:
        df_['perturbation'] = 'no nouns'
    elif 'mask_verbs' in file_:
        df_['perturbation'] = 'no verbs'
    elif 'shuffle' in file_:
        df_['perturbation'] = 'shuffled'
    return df_


def neural_max(df_, rois):
    # get average within subject first
    out_ = df_.groupby(['subj_id', 'roi_name', 'model_uid']).mean(numeric_only=True).reset_index()
    #now get average across subjects
    out_ = out_.groupby(['roi_name', 'model_uid']).mean(numeric_only=True).reset_index()
    out_ = out_.loc[out_.roi_name.isin(rois)].reset_index(drop=True)
    out_ = pd.melt(out_, id_vars=["model_uid", "layer_index", "layer_relative_depth", "roi_name", "reliability"], 
                        value_vars=["train_score", "test_score"], 
                        var_name="set", value_name="score")
    out_['roi_name'] = pd.Categorical(out_['roi_name'], categories=rois, ordered=True)
    out_['set'] = out_['set'].replace({'train_score': 'train', 'test_score': 'test'})
    out_['set'] = pd.Categorical(out_['set'], categories=['train', 'test'], ordered=True)
    out_['normalized_score'] = out_['score'] / out_['reliability']
    return out_


def add_model_class(file_, df_):
    if 'Vision' in file_:
        df_['model_class'] = 'image'
    elif 'Video' in file_:
        df_['model_class'] = 'video'
    else:
        df_['model_class'] = 'language'
    return df_


def get_model_name(file_):
    if 'perturb' in file_:
        return file_.split('model-')[-1].split('_perturb')[0]
    else: 
        return file_.split('model-')[-1].split('.')[0]


class ModelROISummary:
    def __init__(self, args):
        self.process = 'ModelROISummary'
        self.user = args.user
        self.model_class = args.model_class
        self.model_subpath = args.model_subpath
        self.voxel_id = 9539 #test voxel in EVC
        self.top_dir = f'{args.top_dir}/data/interim'
        self.rois = ['EVC', 'MT', 'EBA', 'LOC', 'pSTS', 'aSTS', 'FFA', 'PPA']


        self.out_path = f'{self.top_dir}/{self.process}'
        Path(self.out_path).mkdir(exist_ok=True, parents=True)
    
    def load_files(self, files, add_perturbation=False):
        # Load the files and sum them up 
        df = []
        model_avg_score = []
        for file in tqdm(files, total=len(files), desc='Loading files'): 
            try: 
                pkl = pd.read_pickle(file)
                pkl.drop(columns=['r_var_dist', 'r_null_dist'], inplace=True)
                pkl_mean = pkl.groupby('subj_id').mean(numeric_only=True).reset_index().mean(numeric_only=True)['test_score']
                model_avg_score.append({'model_uid': get_model_name(file), 'avg_score': pkl_mean})

                # Remove voxels not in an ROI
                if 'roi_name' in pkl.columns.tolist():
                    pkl = pkl.loc[pkl['roi_name'] != 'none'].reset_index(drop=True)
                
                # Add information about perturbation and model class
                if add_perturbation: 
                    pkl = add_perturbation(file, pkl)
                pkl = add_model_class(file, pkl)

                #Add to df list 
                df.append(pkl)
                n_final_files += 1 
            except:
                print(f'could not load {file}')
        return pd.DataFrame(pkl), pd.DataFrame(model_avg_score)

    def run(self): 
        try:
            start_time = time.time()
            if self.model_subpath is not None: 
                file_path = f'{self.top_dir}/{self.model_class}/{self.model_subpath}/'
            else: 
                file_path = f'{self.top_dir}/{self.model_class}'
        
            files = glob(f'{file_path}/*.pkl.gz')
            print(f'{len(files)} files found')

            if 'Language' in self.model_class: 
                df, model_avg_score = self.load_files(files, add_perturbation=True)
            else:
                df, model_avg_score = self.load_files(files)
            df = neural_max(df, self.rois)

            save_start = time.time()
            df.to_csv(f'{self.out_path}/{self.model_class}_{self.model_subpath}.csv.gz', index=False)
            model_avg_score.to_csv(f'{self.out_path}/{self.model_class}_{self.model_subpath}_avg-score.csv', index=False)
            save_time = time.time() - save_start
            elapsed = time.strftime("%H:%M:%S", time.gmtime(save_time))
            print(f'Saved in {elapsed}!')

            elapsed = time.time() - start_time
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
    ModelROISummary(args).run()


if __name__ == '__main__':
    main()
