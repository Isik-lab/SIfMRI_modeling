#/Applications/anaconda3/envs/deepjuice/bin/python
from pathlib import Path
import argparse
import pandas as pd
import os
from src.mri import Benchmark
from src.neural_alignment import get_benchmarking_results
from src import frame_ops as ops
import torch
from deepjuice.model_zoo.options import get_deepjuice_model
from deepjuice.procedural.datasets import get_data_loader
from deepjuice.extraction import FeatureExtractor


class VisionNeuralEncoding:
    def __init__(self, args):
        self.process = 'VisionNeuralEncoding'
        self.overwrite = args.overwrite
        self.test_eval = args.test_eval
        self.model_uid = args.model_uid
        self.grouping_func = args.grouping_func
        self.data_dir = f'{args.top_dir}/data'
        self.cache = f'{args.top_dir}/.cache'
        torch.hub.set_dir(self.cache)

        # check hugging face cache location
        print("HF_HOME is set to:", os.environ['HF_HOME'])
        print("HUGGINGFACE_HUB_CACHE is set to:", os.environ['HUGGINGFACE_HUB_CACHE'])
        print("HF_DATASETS_CACHE is set to:", os.environ['HF_DATASETS_CACHE'])

        self.video_path = f'{self.data_dir}/raw/videos/'
        self.frame_path = f'{self.cache}/frames/'
        print(vars(self))
        self.model_name = self.model_uid.replace('/', '_')
        Path(f'{self.data_dir}/interim/{self.process}/{self.grouping_func}').mkdir(parents=True, exist_ok=True)
        self.out_file = f'{self.data_dir}/interim/{self.process}/{self.grouping_func}/model-{self.model_name}.pkl.gz'
        self.frames = [0, 15, 30, 45, 60, 75, 89]

    def load_fmri(self):
        metadata_ = pd.read_csv(f'{self.data_dir}/interim/ReorganziefMRI/metadata.csv')
        response_data_ = pd.read_csv(f'{self.data_dir}/interim/ReorganziefMRI/response_data.csv.gz')
        stimulus_data_ = pd.read_csv(f'{self.data_dir}/interim/ReorganziefMRI/stimulus_data.csv')
        return Benchmark(metadata_, stimulus_data_, response_data_)
    
    def run(self):
        if os.path.exists(self.out_file) and not self.overwrite: 
            # results = pd.read_csv(self.out_file)
            print('Output file already exists. To run again pass --overwrite.')
        else:
            benchmark = self.load_fmri()
            # Break the videos into frames for averaging
            frame_data = ops.visual_events(benchmark.stimulus_data,
                                           self.video_path, self.frame_path,
                                           frame_idx=self.frames)

            # Get the model and dataloader
            model, preprocess = get_deepjuice_model(self.model_name)
            dataloader = get_data_loader(frame_data, preprocess, input_modality='image',
                                         batch_size=16, data_key='images', group_keys='video_name')

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
                                               test_eval=self.test_eval,
                                               grouping_func=self.grouping_func,
                                               memory_limit='30GB')
            print('saving results')
            results.to_pickle(self.out_file, compression='gzip')
            print('Finished!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_uid', type=str, default='torchvision_alexnet_imagenet1k_v1')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--test_eval', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--grouping_func', type=str, default='grouped_average')
    parser.add_argument('--top_dir', '-top', type=str,
                         default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIfMRI_modeling')  
    args = parser.parse_args()
    VisionNeuralEncoding(args).run()


if __name__ == '__main__':
    main()
