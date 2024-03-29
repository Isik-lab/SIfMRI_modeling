#/Applications/anaconda3/envs/deepjuice/bin/python
from pathlib import Path
import argparse
import pandas as pd
import os
from src.mri import Benchmark
from src.behavior_alignment import get_benchmarking_results
from src import frame_ops as ops
import torch
from deepjuice.model_zoo.options import get_deepjuice_model
from deepjuice.procedural.datasets import get_data_loader
from deepjuice.extraction import FeatureExtractor


class VisionBehaviorEncoding:
    def __init__(self, args):
        self.process = 'VisionBehaviorEncoding'
        print('working')
        self.overwrite = args.overwrite
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
    
    def load_data(self):
        return Benchmark(stimulus_data=f'{self.data_dir}/interim/ReorganziefMRI/stimulus_data.csv')
    
    def run(self):
        if os.path.exists(self.out_file) and not self.overwrite: 
            # results = pd.read_csv(self.out_file)
            print('Output file already exists. To run again pass --overwrite.')
        else:
            # Load data and sort
            benchmark = self.load_data()
            target_features = [col for col in benchmark.stimulus_data.columns if ('rating-' in col) and ('indoor' not in col)]
            # Break the videos into frames for averaging
            frame_data = ops.visual_events(benchmark.stimulus_data,
                                           self.video_path, self.frame_path,
                                           frame_idx=self.frames)

            # Get the model and dataloader
            model, preprocess = get_deepjuice_model(self.model_name)
            dataloader = get_data_loader(frame_data, preprocess, input_modality='image',
                                         batch_size=16, data_key='images', group_keys='video_name')

            # Reorganize the benchmark to the dataloader
            videos = dataloader.batch_data.groupby(by='video_name').groups.keys()
            print(videos)
            benchmark.stimulus_data['video_name'] = pd.Categorical(benchmark.stimulus_data['video_name'],
                                                                   categories=videos, ordered=True) 
            benchmark.stimulus_data = benchmark.stimulus_data.sort_values('video_name')

            # Perform all the regressions
            results = get_benchmarking_results(benchmark, model, dataloader,
                                               target_features=target_features,
                                               model_name=self.model_name,
                                               grouping_func=self.grouping_func)

            # Save
            print('saving results')
            results.to_pickle(self.out_file, compression='gzip')
            print('Finished!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_uid', type=str, default='torchvision_alexnet_imagenet1k_v1')
    parser.add_argument('--grouping_func', type=str, default='grouped_average')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--top_dir', '-data', type=str,
                         default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIfMRI_modeling')  
    args = parser.parse_args()
    VisionBehaviorEncoding(args).run()


if __name__ == '__main__':
    main()
