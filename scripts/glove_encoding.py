#/Applications/anaconda3/envs/nibabel/bin/python
from pathlib import Path
import argparse
import pandas as pd
import os
from src.mri import Benchmark
from deepjuice.structural import flatten_nested_list # utility for list flattening
from src import encoding


class GLoVeEncoding:
    def __init__(self, args):
        self.process = 'GLoVeEncoding'
        self.overwrite = args.overwrite
        self.perturbation = args.perturbation
        self.data_dir = args.data_dir
        print(vars(self))

        Path(f'{self.data_dir}/interim/{self.process}').mkdir(parents=True, exist_ok=True)
        self.out_file = f'{self.data_dir}/interim/{self.process}/{self.perturbation}.csv'

        self.streams = ['EVC']
        self.streams += [f'{level}_{stream}' for level in ['mid', 'high'] for stream in ['ventral', 'lateral', 'parietal']]
    
    def load_fmri(self):
        metadata_ = pd.read_csv(f'{self.data_dir}/interim/ReorganziefMRI/metadata.csv')
        response_data_ = pd.read_csv(f'{self.data_dir}/interim/ReorganziefMRI/response_data.csv.gz')
        stimulus_data_ = pd.read_csv(f'{self.data_dir}/interim/ReorganziefMRI/stimulus_data.csv')
        return Benchmark(metadata_, stimulus_data_, response_data_)

    def get_captions(self):
        filename = f'{self.data_dir}/interim/CaptionData/{self.perturbation}.txt'
        with open(filename) as file:
           return [line.rstrip() for line in file]
    
    def run(self):
        if os.path.exists(self.out_file) and not self.overwrite: 
            # results = pd.read_csv(self.out_file)
            print('Output file already exists. To run again pass --overwrite.')
        else:
            print('loading data...')
            benchmark = self.load_fmri()
            benchmark.filter_stimulus(stimulus_set='train')
            captions = self.get_captions()
            print(f'caption length: {len(captions)}')

            print('loading model...')
            features = encoding.glove_feature_extraction(captions)
            print(features.shape)
            print(features[0].shape)

            # print('running regressions')
            # results = encoding.get_training_benchmarking_results(benchmark, features)

            # print('saving results')
            # results.to_csv(self.out_file, index=False)
            print('Finished!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--perturbation', type=str, default='nv_shuffled')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--data_dir', '-data', type=str,
                         default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIfMRI_modeling/data')                        
                        # default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_modeling/data')

    args = parser.parse_args()
    GLoVeEncoding(args).run()


if __name__ == '__main__':
    main()