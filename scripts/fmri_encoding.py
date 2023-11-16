#/Applications/anaconda3/envs/nibabel/bin/python
from pathlib import Path
import argparse
import pandas as pd
import os
from src.mri import Benchmark
import torch
from deepjuice.structural import flatten_nested_list # utility for list flattening
from src import encoding


class fMRIDecoding:
    def __init__(self, args):
        self.process = 'fMRIDecoding'
        if 'u' not in args.sid:
            self.sid = f'subj{str(int(args.sid)).zfill(3)}'
        else:
            self.sid = args.sid
        self.overwrite = args.overwrite
        self.model_uid = args.model_uid
        print(vars(self))

        self.data_dir = args.data_dir
        Path(f'{self.data_dir}/interim/{self.process}').mkdir(parents=True, exist_ok=True)
        model_name = self.model_uid.replace('/', '_')
        self.out_file = f'{self.data_dir}/interim/{self.process}/{model_name}.csv'

        self.streams = ['EVC']
        self.streams += [f'{level}_{stream}' for level in ['mid', 'high'] for stream in ['ventral', 'lateral', 'parietal']]
        
        print(f'cuda is available {torch.cuda.is_available()}')
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
    
    def load_fmri(self):
        metadata_ = pd.read_csv(f'{self.data_dir}/interim/ReorganziefMRI/metadata.csv')
        response_data_ = pd.read_csv(f'{self.data_dir}/interim/ReorganziefMRI/response_data.csv.gz')
        stimulus_data_ = pd.read_csv(f'{self.data_dir}/interim/ReorganziefMRI/stimulus_data.csv')
        return Benchmark(metadata_, stimulus_data_, response_data_)

    def get_captions(self, benchmark):
        all_captions = benchmark.stimulus_data.captions.tolist() # list of strings
        # the listification and flattening of our 5 captions per image into one big list:
        captions = flatten_nested_list([eval(captions)[:5] for captions in all_captions])
        assert(len(captions) == 200 * 5) # assertion to ensure each image has 5 captions
        return captions
    
    def run(self):
        if os.path.exists(self.out_file) and not self.overwrite: 
            results = pd.read_csv(self.out_file)
        else:
            print('loading data...')
            benchmark = self.load_fmri()
            captions = self.get_captions(benchmark)
            feature_extractor = encoding.memory_saving_extraction(self.model_uid, captions)
            results = encoding.get_benchmarking_results(benchmark, feature_extractor, self.device)
            results.to_csv(self.out_file, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sid', type=str, default='1')
    parser.add_argument('--model_uid', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--data_dir', '-data', type=str,
                         default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIfMRI_modeling/data')
    args = parser.parse_args()
    fMRIDecoding(args).run()


if __name__ == '__main__':
    main()