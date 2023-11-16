#/Applications/anaconda3/envs/nibabel/bin/python
from pathlib import Path
import argparse
import pandas as pd
import os
from src.mri import Benchmark
import torch


class fMRIDecoding:
    def __init__(self, args):
        self.process = 'fMRIDecoding'
        if 'u' not in args.sid:
            self.sid = f'subj{str(int(args.sid)).zfill(3)}'
        else:
            self.sid = args.sid
        self.regress_gaze = args.regress_gaze
        self.overwrite = args.overwrite
        print(vars(self))

        self.data_dir = args.data_dir
        self.figure_dir = args.figure_dir
        Path(f'{self.data_dir}/interim/{self.process}').mkdir(parents=True, exist_ok=True)
        Path(f'{self.figure_dir}/{self.process}').mkdir(parents=True, exist_ok=True)
        self.out_figure = f'{self.figure_dir}/{self.process}/{self.sid}_reg-gaze-{self.regress_gaze}_decoding.png'
        self.out_file = f'{self.data_dir}/interim/{self.process}/{self.sid}_reg-gaze-{self.regress_gaze}_decoding.csv'

        self.streams = ['EVC']
        self.streams += [f'{level}_{stream}' for level in ['mid', 'high'] for stream in ['ventral', 'lateral', 'parietal']]
        
        print(f'cuda is available {torch.cuda.is_available()}')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_fmri(self):
        metadata_ = pd.read_csv(f'{self.data_dir}/interim/ReorganziefMRI/metadata.csv')
        response_data_ = pd.read_csv(f'{self.data_dir}/interim/ReorganziefMRI/response_data.csv.gz')
        stimulus_data_ = pd.read_csv(f'{self.data_dir}/interim/ReorganziefMRI/stimulus_data.csv.gz')
        return Benchmark(metadata_, stimulus_data_, response_data_)
    
    def run(self):
        if os.path.exists(self.out_file) and not self.overwrite: 
            results = pd.read_csv(self.out_file)
        else:
            print('loading data...')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sid', type=str, default='1')
    parser.add_argument('--regress_gaze', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--data_dir', '-data', type=str,
                         default='/Users/emcmaho7/Dropbox/projects/SI_EEG/SIEEG_analysis/data')
    parser.add_argument('--figure_dir', '-figure', type=str,
                        default='/Users/emcmaho7/Dropbox/projects/SI_EEG/SIEEG_analysis/reports/figures')
    args = parser.parse_args()
    fMRIDecoding(args).run()


if __name__ == '__main__':
    main()