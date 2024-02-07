#/home/emcmaho7/.conda/envs/deepjuice/bin/python
from pathlib import Path
import argparse
import pandas as pd
import os
from src.mri import Benchmark
from src import encoding
import torch
    

def get_features(captions, reshape_dim):
    features = encoding.glove_feature_extraction(captions)
    features = features.reshape(reshape_dim + (-1,))
    return features.mean(axis=1)


class GLoVeEncoding:
    def __init__(self, args):
        self.process = 'GLoVeEncoding'
        self.overwrite = args.overwrite
        self.perturbation = args.perturbation
        self.data_dir = args.data_dir

        if self.perturbation == 'none':
            self.stimulus_data_file = f'{self.data_dir}/interim/ReorganziefMRI/stimulus_data.csv'
        else:
            self.stimulus_data_file = f'{self.data_dir}/interim/SentenceDecomposition/{self.perturbation}.csv'
        
        self.out_file = f'{self.data_dir}/interim/{self.process}/model-glove_perturbation-{self.perturbation}.csv'
        print(vars(self))

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
    
    def load_fmri(self):
        metadata_ = pd.read_csv(f'{self.data_dir}/interim/ReorganziefMRI/metadata.csv')
        response_data_ = pd.read_csv(f'{self.data_dir}/interim/ReorganziefMRI/response_data.csv.gz')
        stimulus_data_ = pd.read_csv(f'{self.data_dir}/interim/ReorganziefMRI/stimulus_data.csv')
        return Benchmark(metadata_, stimulus_data_, response_data_)


    def run(self):
        if os.path.exists(self.out_file) and not self.overwrite: 
            print('Output file already exists. To run again pass --overwrite.')
        else:
            print('loading data...')
            benchmark = self.load_fmri()
            benchmark.filter_stimulus(stimulus_set='train')
            captions, reshape_dim = encoding.captions_to_list(benchmark.stimulus_data.captions)

            print('loading model...')
            features = get_features(captions, reshape_dim)
            print(f'{features.shape=}')

            print('running regressions')
            results = encoding.get_lm_encoded_training_benchmarking_results(benchmark, features)

            print('saving results')
            results.to_csv(self.out_file, index=False)
            print('Finished!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--perturbation', type=str, default='corrected_unmasked')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--data_dir', '-data', type=str,
                         default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIfMRI_modeling/data')                        
                        # default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_modeling/data')
    args = parser.parse_args()
    GLoVeEncoding(args).run()


if __name__ == '__main__':
    main()