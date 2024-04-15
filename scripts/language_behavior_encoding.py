#/Applications/anaconda3/envs/deepjuice/bin/python
from pathlib import Path
import argparse
import pandas as pd
import os
from src.mri import Benchmark
from src.behavior_alignment import get_benchmarking_results
from src.language_ops import parse_caption_data, get_model
from src.language_ablation import strip_sentence, Masking
from src import tools
import time
import torch
from deepjuice.procedural.datasets import get_data_loader
from deepjuice.extraction import FeatureExtractor
from tqdm import tqdm
tqdm.pandas()


def perturb_captions(df, func_name='none'):
    name_to_params = {'mask_nouns': {'POS': 'nouns', 'mask_else': False}, 
                    'mask_verbs': {'POS': 'verbs', 'mask_else': False}, 
                    'mask_adjectives': {'POS': 'adjectives', 'mask_else': False}, 
                    'mask_prepositions': {'POS': 'prepositions', 'mask_else': False}, 
                    'mask_nonnouns': {'POS': 'nouns', 'mask_else': True},
                    'mask_nonverbs': {'POS': 'verbs', 'mask_else': True}, 
                    'mask_nonadjectives': {'POS': 'adjectives', 'mask_else': True}, 
                    'mask_nonprepositions': {'POS': 'prepositions', 'mask_else': True}}
    mask_params = name_to_func[func_name]
    mask_func = Masking(mask_params['POS'],
                        mask_else=mask_params['mask_else'])
    
    df.reset_index(drop=True, inplace=True)
    df['caption'] = df['caption'].astype(object)
    if func_name != 'none':
        df['caption'] = df['caption'].progress_apply(lambda x: mask_func.run(strip_sentence(x)))
    else:
        df['caption'] = df['caption'].progress_apply(strip_sentence)


class LanguageBehaviorEncoding:
    def __init__(self, args):
        self.process = 'LanguageBehaviorEncoding'
        print('working')
        self.user = args.user
        self.overwrite = args.overwrite
        self.perturbation = args.perturbation
        self.perturb_func = args.perturb_func
        self.model_uid = args.model_uid
        self.memory_limit = args.memory_limit
        self.data_dir = f'{args.top_dir}/data'
        self.cache = f'{args.top_dir}/.cache'
        torch.hub.set_dir(self.cache)
        self.model_name = self.model_uid.replace('/', '_')

        if not self.perturbation: 
            self.out_file = f'{self.data_dir}/interim/{self.process}/no_perturbation/model-{self.model_name}.pkl.gz'
            self.input_file = f'{self.data_dir}/interim/{self.process}/no_perturbation/captions.csv'
        else:
            self.out_file = f'{self.data_dir}/interim/{self.process}/perturbation/{self.perturb_func}/model-{self.model_name}_perturb-{self.perturb_func}.pkl.gz'
            self.input_file = f'{self.data_dir}/interim/{self.process}/perturbation/{self.perturb_func}/{self.perturb_func}.csv'
        print(vars(self))
        # check hugging face cache location
        print("HF_HOME is set to:", os.environ['HF_HOME'])
        print("HUGGINGFACE_HUB_CACHE is set to:", os.environ['HUGGINGFACE_HUB_CACHE'])
        print("HF_DATASETS_CACHE is set to:", os.environ['HF_DATASETS_CACHE'])

        Path(f'{self.data_dir}/interim/{self.process}/no_perturbation').mkdir(parents=True, exist_ok=True)
        Path(f'{self.data_dir}/interim/{self.process}/perturbation/{self.perturb_func}').mkdir(parents=True, exist_ok=True)
    
    def load_data(self):
        return Benchmark(stimulus_data=f'{self.data_dir}/interim/ReorganziefMRI/stimulus_data.csv')

    def load_captions(self):
        if not os.path.exists(self.input_file): 
            file = f'{self.data_dir}/interim/CaptionData/captions.csv'
            df = parse_caption_data(file)
            if self.perturbation:
                perturb_captions(df, func_name=self.perturb_func)
            df.to_csv(self.input_file, index=False)
            return df 
        else:
            return pd.read_csv(self.input_file)

    def run(self):
        try:
            if os.path.exists(self.out_file) and not self.overwrite:
                # results = pd.read_csv(self.out_file)
                print('Output file already exists. To run again pass --overwrite.')
            else:
                start_time = time.time()
                tools.send_slack(f'Started: {self.process} {self.model_name}...', channel=self.user)
                benchmark = self.load_data()
                target_features = [col for col in benchmark.stimulus_data.columns if ('rating-' in col) and ('indoor' not in col)]
                captions = self.load_captions()

                # Get the model and dataloader
                model, tokenizer = get_model(self.model_uid)
                dataloader = get_data_loader(captions, tokenizer, input_modality='text',
                                             batch_size=16, data_key='caption', group_keys='video_name')

                # Reorganize the benchmark to the dataloader
                videos = list(dataloader.batch_data.groupby(by='video_name').groups.keys())
                benchmark.stimulus_data['video_name'] = pd.Categorical(benchmark.stimulus_data['video_name'],
                                                                       categories=videos, ordered=True)
                benchmark.stimulus_data = benchmark.stimulus_data.sort_values('video_name').reset_index(drop=True)

                print('running regressions')
                results = get_benchmarking_results(benchmark, model, dataloader,
                                                   target_features=target_features,
                                                   memory_limit=self.memory_limit,
                                                   model_name=self.model_name)
                print('saving results')
                results.to_pickle(self.out_file, compression='gzip')
                print('Finished!')

                end_time = time.time()
                elapsed = end_time - start_time
                elapsed = time.strftime("%H:%M:%S", time.gmtime(elapsed))
                print(f'Finished in {elapsed}!')
                tools.send_slack(f'Finished: {self.process} {self.model_name} in {elapsed}', channel=self.user)
        except Exception as err:
            print(f'Error: {self.process} {self.model_name}: Error Msg = {err}')
            tools.send_slack(f'Error: {self.process} {self.model_name}: Error Msg = {err}', channel=self.user)


def main():
    parser = argparse.ArgumentParser()
    # Add arguments that are needed before setting the default for data_dir
    parser.add_argument('--user', type=str, default='emcmaho7')
    # Parse known args first to get the user
    args, remaining_argv = parser.parse_known_args()
    user = args.user  # Get the user from the parsed known args
    parser.add_argument('--model_uid', type=str, default='sentence-transformers/paraphrase-MiniLM-L6-v2')
    parser.add_argument('--memory_limit', type=str, default='70GB')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--perturbation', action=argparse.BooleanOptionalAction, default=False) 
    parser.add_argument('--perturb_func', type=str, default='none')
    parser.add_argument('--top_dir', '-data', type=str,
                         default=f'/home/{user}/scratch4-lisik3/{user}/SIfMRI_modeling')

    args = parser.parse_args()
    LanguageBehaviorEncoding(args).run()


if __name__ == '__main__':
    main()
