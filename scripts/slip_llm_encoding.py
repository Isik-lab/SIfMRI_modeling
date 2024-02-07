#/home/emcmaho7/.conda/envs/slip/bin/python
from pathlib import Path
import argparse
import pandas as pd
import os
from src.mri import Benchmark
import torch
from src import encoding, language_ops
import numpy as np


class SLIPEncoding:
    def __init__(self, args):
        self.process = 'SLIPEncoding'
        self.overwrite = args.overwrite
        self.backbone = args.backbone
        self.perturbation = args.perturbation
        self.top_dir = args.top_dir
        self.model_filepath = os.path.join(self.top_dir, 'SLIP', 'slip', 'models', self.backbone+'.pt')
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'
        print(vars(self))

        self.data_dir = args.data_dir
        if self.perturbation == 'none':
            self.stimulus_data_file = f'{self.data_dir}/interim/ReorganziefMRI/stimulus_data.csv'
        else:
            self.stimulus_data_file = f'{self.data_dir}/interim/SentenceDecomposition/{self.perturbation}.csv'
        self.out_file = f'{self.data_dir}/interim/{self.process}/model-{self.backbone}_perturbation-{self.perturbation}.csv'

    def load_fmri(self):
        metadata_ = pd.read_csv(f'{self.data_dir}/interim/ReorganziefMRI/metadata.csv')
        response_data_ = pd.read_csv(f'{self.data_dir}/interim/ReorganziefMRI/response_data.csv.gz')
        stimulus_data_ = pd.read_csv(self.stimulus_data_file)
        return Benchmark(metadata_, stimulus_data_, response_data_)

    def get_features(self, captions, reshape_dim):
        # Only perform feature extraction if needed
        features = language_ops.slip_feature_extraction(self.model_filepath, captions, device='cpu')
        print(f'{features.shape=}')
        features = features.reshape(reshape_dim + (-1,))
        print(f'{features.shape=}')
        return features.mean(axis=1)

    def run(self):
        if os.path.exists(self.out_file) and not self.overwrite: 
            print('Output file already exists. To run again pass --overwrite.')
        else:
            print('loading data...')
            benchmark = self.load_fmri()
            benchmark.filter_stimulus(stimulus_set='train')
            captions, reshape_dim = language_ops.captions_to_list(benchmark.stimulus_data.captions)

            features = self.get_features(captions, reshape_dim)

            print('running regressions')
            results = encoding.get_lm_encoded_training_benchmarking_results(benchmark, features)

            print('saving results')
            results.to_csv(self.out_file, index=False)
            print('Finished!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='clip_small_25ep')
    parser.add_argument('--perturbation', type=str, default='stripped_orig')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--top_dir', '-top', type=str,
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/')
    parser.add_argument('--data_dir', '-data', type=str,
                         default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIfMRI_modeling/data')                        
    args = parser.parse_args()
    SLIPEncoding(args).run()


if __name__ == '__main__':
    main()