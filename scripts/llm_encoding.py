#/Applications/anaconda3/envs/nibabel/bin/python
from pathlib import Path
import argparse
import pandas as pd
import os
from src.mri import Benchmark
from deepjuice.structural import flatten_nested_list # utility for list flattening
from src import encoding
from src import lang_permute


class LLMEncoding:
    def __init__(self, args):
        self.process = 'LLMEncoding'
        self.overwrite = args.overwrite
        self.model_uid = args.model_uid
        self.perturbation = args.perturbation
        self.spell_check = args.spell_check
        self.data_dir = args.data_dir
        print(vars(self))

        Path(f'{self.data_dir}/interim/{self.process}').mkdir(parents=True, exist_ok=True)
        model_name = self.model_uid.replace('/', '_')
        self.out_file = f'{self.data_dir}/interim/{self.process}/model-{model_name}_perturbation-{self.perturbation}.csv'

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
                                           perturb['pos'], lemmatize=perturb['lemmatize'],
                                           shuffle=perturb['shuffle'], exclude=perturb['exclude'])
    
    def run(self):
        if os.path.exists(self.out_file) and not self.overwrite: 
            # results = pd.read_csv(self.out_file)
            print('Output file already exists. To run again pass --overwrite.')
        else:
            print('loading data...')
            benchmark = self.load_fmri()
            benchmark.filter_stimulus(stimulus_set='train')
            captions, _ = self.get_captions(benchmark)
            if self.perturbation is not None and self.perturbation != 'none':
                captions = self.get_pos(captions)
            print('loading model...')
            feature_extractor = encoding.memory_saving_extraction(self.model_uid, captions)

            print('running regressions')
            results = encoding.get_training_benchmarking_results(benchmark, feature_extractor)

            print('saving results')
            results.to_csv(self.out_file, index=False)
            print('Finished!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_uid', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--perturbation', type=str, default=None)
    parser.add_argument('--spell_check', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--data_dir', '-data', type=str,
                         default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIfMRI_modeling/data')                        
                        # default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_modeling/data')

    args = parser.parse_args()
    LLMEncoding(args).run()


if __name__ == '__main__':
    main()