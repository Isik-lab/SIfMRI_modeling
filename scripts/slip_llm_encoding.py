#/Applications/anaconda3/envs/nibabel/bin/python
from pathlib import Path
import argparse
import pandas as pd
import os
from src.mri import Benchmark
import torch
from src import encoding, language_ops


class SLIPEncoding:
    def __init__(self, args):
        self.process = 'SLIPEncoding'
        self.overwrite = args.overwrite
        self.backbone = args.backbone
        self.perturbation = args.perturbation
        self.data_dir = args.data_dir

        if self.perturbation == 'none':
            self.stimulus_data_file = f'{self.data_dir}/interim/ReorganziefMRI/stimulus_data.csv'
        else:
            self.stimulus_data_file = f'{self.data_dir}/interim/SentenceDecomposition/{self.perturbation}.csv'
        
        self.out_path = f'{self.data_dir}/interim/{self.process}/model-clip-{self.backbone}_perturbation-{self.perturbation}'
        self.out_file = f'{self.data_dir}/interim/{self.process}/model-clip-{self.backbone}_perturbation-{self.perturbation}.csv'
        print(vars(self))

        Path(self.out_path).mkdir(parents=True, exist_ok=True)
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
    
    def load_fmri(self):
        metadata_ = pd.read_csv(f'{self.data_dir}/interim/ReorganziefMRI/metadata.csv')
        response_data_ = pd.read_csv(f'{self.data_dir}/interim/ReorganziefMRI/response_data.csv.gz')
        stimulus_data_ = pd.read_csv(self.stimulus_data_file)
        return Benchmark(metadata_, stimulus_data_, response_data_)
    
    def run(self):
        if os.path.exists(self.out_file) and not self.overwrite: 
            print('Output file already exists. To run again pass --overwrite.')
        else:
            print('loading data...')
            benchmark = self.load_fmri()
            benchmark.filter_stimulus(stimulus_set='train')
            captions, _ = language_ops.captions_to_list(benchmark.stimulus_data.captions)

            print('loading model...')
            feature_extractor = language_ops.slip_extraction(captions, self.device)

            print('running regressions')
            results = encoding.get_training_benchmarking_results(benchmark, feature_extractor, self.out_path)

            print('saving results')
            results.to_csv(self.out_file, index=False)
            print('Finished!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='RN50')
    parser.add_argument('--perturbation', type=str, default='corrected_unmasked')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--data_dir', '-data', type=str,
                         default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIfMRI_modeling/data')                        
                        # default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_modeling/data')
    args = parser.parse_args()
    SLIPEncoding(args).run()


if __name__ == '__main__':
    main()