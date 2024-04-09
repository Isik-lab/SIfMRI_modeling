#/Applications/anaconda3/envs/deepjuice/bin/python
from pathlib import Path
import argparse
import pandas as pd
import os
from src.mri import Benchmark
from src.neural_alignment import get_benchmarking_results
from src.language_ops import parse_caption_data, get_model, tokenize_captions
from src import tools
import time
import torch
from deepjuice.procedural.datasets import get_data_loader
from deepjuice.extraction import FeatureExtractor


class LanguageNeuralEncoding:
    def __init__(self, args):
        self.process = 'LanguageNeuralEncoding'
        print('working')
        self.user = args.user
        self.overwrite = args.overwrite
        self.model_uid = args.model_uid
        self.test_eval = args.test_eval
        self.memory_limit = args.memory_limit
        self.data_dir = f'{args.top_dir}/data'
        self.cache = f'{args.top_dir}/.cache'
        torch.hub.set_dir(self.cache)
        self.model_name = self.model_uid.replace('/', '_')
        Path(f'{self.data_dir}/interim/{self.process}').mkdir(parents=True, exist_ok=True)
        self.out_file = f'{self.data_dir}/interim/{self.process}/model-{self.model_name}.pkl.gz'
        # check hugging face cache location
        print("HF_HOME is set to:", os.environ['HF_HOME'])
        print("HUGGINGFACE_HUB_CACHE is set to:", os.environ['HUGGINGFACE_HUB_CACHE'])
        print("HF_DATASETS_CACHE is set to:", os.environ['HF_DATASETS_CACHE'])
        print(vars(self))
    
    def load_fmri(self):
        metadata_ = pd.read_csv(f'{self.data_dir}/interim/ReorganziefMRI/metadata.csv')
        response_data_ = pd.read_csv(f'{self.data_dir}/interim/ReorganziefMRI/response_data.csv.gz')
        stimulus_data_ = pd.read_csv(f'{self.data_dir}/interim/ReorganziefMRI/stimulus_data.csv')
        return Benchmark(metadata_, stimulus_data_, response_data_)

    def load_captions(self):
        file = f'{self.data_dir}/interim/CaptionData/captions.csv'
        return parse_caption_data(file)
    
    def run(self):
        try:
            if os.path.exists(self.out_file) and not self.overwrite:
                # results = pd.read_csv(self.out_file)
                print('Output file already exists. To run again pass --overwrite.')
            else:
                start_time = time.time()
                tools.send_slack(f'Started: {self.process} {self.model_name}...', channel=self.user)
                benchmark = self.load_fmri()
                captions = self.load_captions()

                # Get the model and dataloader
                model, tokenizer = get_model(self.model_uid)
                dataloader = get_data_loader(captions, tokenizer, input_modality='text',
                                             batch_size=16, data_key='caption', group_keys='video_name')


                # Reorganize the benchmark to the dataloader
                videos = list(dataloader.batch_data.groupby(by='video_name').groups.keys())
                benchmark.stimulus_data['video_name'] = pd.Categorical(benchmark.stimulus_data['video_name'],
                                                                       categories=videos, ordered=True)
                benchmark.stimulus_data = benchmark.stimulus_data.sort_values('video_name')
                stim_idx = list(benchmark.stimulus_data.index.to_numpy().astype('str'))
                benchmark.stimulus_data.reset_index(drop=True, inplace=True)
                benchmark.response_data = benchmark.response_data[stim_idx]

                print('running regressions')
                results = get_benchmarking_results(benchmark, model, dataloader,
                                                   model_name=self.model_name,
                                                   memory_limit=self.memory_limit,
                                                   test_eval=self.test_eval)
                print('saving results')
                results.to_pickle(self.out_file, compression='gzip')
                print('Finished!')

                end_time = time.time()
                elapsed = end_time - start_time
                elapsed = time.strftime("%H:%M:%S", time.gmtime(elapsed))
                print(f'Finished in {elapsed}!')
                tools.send_slack(f'Finished: {self.process} {self.model_name} in {elapsed}', channel=self.user)
        except Exception as err:
            print(f'Error: {self.process} {self.model_name} Error Msg = {err}')
            tools.send_slack(f'Error: {self.process} {self.model_name} Error Msg = {err}', channel=self.user)

def main():
    parser = argparse.ArgumentParser()
    # Add arguments that are needed before setting the default for data_dir
    parser.add_argument('--user', type=str, default='emcmaho7')
    # Parse known args first to get the user
    args, remaining_argv = parser.parse_known_args()
    user = args.user  # Get the user from the parsed known args
    parser.add_argument('--model_uid', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--memory_limit', type=str, default='70GB')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--test_eval', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--top_dir', '-data', type=str,
                         default=f'/home/{user}/scratch4-lisik3/{user}/SIfMRI_modeling')
    args = parser.parse_args()
    LanguageNeuralEncoding(args).run()


if __name__ == '__main__':
    main()
