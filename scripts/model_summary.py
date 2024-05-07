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


def get_max_score(group, col='train_score'):
    return group.loc[group[col].idxmax()]


def neural_max(df_, categories, category_col='roi_name'):
    # get average within subject first
    out_ = df_.groupby(['subj_id', category_col, 'model_uid']).mean(numeric_only=True).reset_index()
    #now get average across subjects
    out_ = out_.groupby([category_col, 'model_uid']).mean(numeric_only=True).reset_index()
    out_ = out_.loc[out_[category_col].isin(categories)].reset_index(drop=True)
    out_ = pd.melt(out_, id_vars=["model_uid", "layer_index", "layer_relative_depth", category_col, "reliability"], 
                        value_vars=["train_score", "test_score"], 
                        var_name="set", value_name="score")
    out_[category_col] = pd.Categorical(out_[category_col], categories=categories, ordered=True)
    out_['set'] = out_['set'].replace({'train_score': 'train', 'test_score': 'test'})
    out_['set'] = pd.Categorical(out_['set'], categories=['train', 'test'], ordered=True)
    out_['normalized_score'] = out_['score'] / out_['reliability']
    return out_


def behavior_max(df_, categories, formatting=True):
    out_ = df_.groupby(['feature', 'model_uid']).apply(get_max_score).reset_index(drop=True)
    if formatting:
        rename_mapping = {orig: orig.replace('rating-', '').replace('_', ' ') for orig in out_.feature.unique()}
        out_ = pd.melt(out_, id_vars=["model_uid", "model_layer", "model_layer_index", "feature"], 
                            value_vars=["train_score", "test_score"], 
                            var_name="set", value_name="score")
        out_['feature'] = out_['feature'].replace(rename_mapping)
        out_['feature'] = pd.Categorical(out_['feature'], categories=categories, ordered=True)
        out_['set'] = out_['set'].replace({'train_score': 'train', 'test_score': 'test'})
        out_['set'] = pd.Categorical(out_['set'], categories=['train', 'test'], ordered=True)
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


class ModelSummary:
    def __init__(self, args):
        self.process = 'ModelSummary'
        self.user = args.user
        self.model_class = args.model_class
        self.model_subpath = args.model_subpath
        self.voxel_id = 9539 #test voxel in EVC
        self.top_dir = f'{args.top_dir}/data/interim'
        self.category_col = args.category_col
        self.out_path = f'{self.top_dir}/{self.process}'
        if 'Neural' in self.model_class:
            self.neural = True
            if self.category_col is None:
                print('No category column passed for neural data. Setting to "roi_name"')
                self.category_col = 'roi_name'
                self.categories = ['EVC', 'MT', 'EBA', 'LOC', 'pSTS', 'aSTS', 'FFA', 'PPA']
            elif self.category_col == 'roi_name': 
                self.categories = ['EVC', 'MT', 'EBA', 'LOC', 'pSTS', 'aSTS', 'FFA', 'PPA']
            elif self.category_col == 'stream_name':
                self.categories = ['evc', 'mid_lateral', 'mid_ventral', 'mid_parietal',
                                   'high_lateral', 'high_ventral', 'high_parietal']
            self.out_name = f'{self.out_path}/{self.model_class}_{self.model_subpath}_{self.category_col}.csv.gz'
        else:
            self.neural = False
            self.categories = ['expanse', 'object', 
                               'agent distance', 'facingness', 'joint action', 
                               'communication', 'valence', 'arousal']
            self.out_name = f'{self.out_path}/{self.model_class}_{self.model_subpath}.csv.gz'
        print(vars(self))
        Path(self.out_path).mkdir(exist_ok=True, parents=True)
    
    def load_files(self, files, perturb_info=False):
        # Load the files and sum them up 
        df = []
        for file in tqdm(files, total=len(files), desc='Loading files'): 
            try: 
                model_uid = get_model_name(file)
                pkl = pd.read_pickle(file)
                if 'r_var_dist' in pkl.columns: 
                    pkl.drop(columns=['r_var_dist'], inplace=True)
                pkl.drop(columns=['r_null_dist'], inplace=True)

                # Remove voxels not in an ROI or in a visual stream
                if (self.category_col is not None) and (self.category_col in pkl.columns):
                    pkl = pkl.loc[pkl[self.category_col] != 'none'].reset_index(drop=True)
                
                # Add information about perturbation and model class
                if add_perturbation: 
                    pkl = add_perturbation(file, pkl)
                pkl = add_model_class(file, pkl)
                pkl['model_uid'] = model_uid

                #Add to df list 
                df.append(pkl)
            except KeyboardInterrupt:
                break
                print("Program was stopped by user.")
            except Exception as e:
                print(f"An unexpected error occurred loading {file.split('/')[-1]}: {e}")
        return pd.concat(df)

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
                df = self.load_files(files, perturb_info=True)
            else:
                df = self.load_files(files)

            if self.neural:
                df = neural_max(df, self.categories, category_col=self.category_col)
            else:
                df = behavior_max(df, self.categories)
            print(f'{df.head()=}')

            save_start = time.time()
            df.to_csv(self.out_name, index=False)
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
    parser.add_argument('--category_col', type=str, default=None)
    parser.add_argument('--top_dir', '-data', type=str,
                         default=f'/home/{user}/scratch4-lisik3/{user}/SIfMRI_modeling')
    args = parser.parse_args()
    ModelSummary(args).run()


if __name__ == '__main__':
    main()
