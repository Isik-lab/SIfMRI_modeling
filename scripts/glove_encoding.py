#/Applications/anaconda3/envs/nibabel/bin/python
from pathlib import Path
import argparse
import pandas as pd
import os
from src.mri import Benchmark
from src import lang_permute
from src import encoding
from deepjuice.structural import flatten_nested_list # utility for list flattening
    

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
        self.spell_check = args.spell_check
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

    def get_captions(self, benchmark):
        all_captions = benchmark.stimulus_data.captions.tolist() # list of strings
        captions = flatten_nested_list([eval(captions)[:5] for captions in all_captions])
        print(captions[:5])
        if self.spell_check:
            fix_spelling = lang_permute.load_spellcheck()
            return [cap['generated_text'] for cap in fix_spelling(captions)], (len(all_captions), 5)
        else:
            return captions, (len(all_captions), 5)

    def get_pos(self, captions):
        syntax_model = lang_permute.get_spacy_model()
        perturb = lang_permute.get_perturbation_data(self.perturbation)
        return lang_permute.pos_extraction(captions, syntax_model,
                                           perturb['pos'], lemmatize=True,
                                           shuffle=perturb['shuffle'], exclude=perturb['exclude'])

    def run(self):
        if os.path.exists(self.out_file) and not self.overwrite: 
            print('Output file already exists. To run again pass --overwrite.')
        else:
            print('loading data...')
            benchmark = self.load_fmri()
            benchmark.filter_stimulus(stimulus_set='train')
            captions, reshape_dim = self.get_captions(benchmark)
            if self.perturbation is not None and self.perturbation != 'none':
                captions = self.get_pos(captions)
            print(f'caption length: {len(captions)}')

            print('loading model...')
            features = get_features(captions, reshape_dim)
            print(features.shape)

            print('running regressions')
            results = encoding.get_glove_training_benchmarking_results(benchmark, features)

            print('saving results')
            results.to_csv(self.out_file, index=False)
            print('Finished!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--perturbation', type=str, default=None)
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--spell_check', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--data_dir', '-data', type=str,
                         default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIfMRI_modeling/data')                        
                        # default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_modeling/data')

    args = parser.parse_args()
    GLoVeEncoding(args).run()


if __name__ == '__main__':
    main()