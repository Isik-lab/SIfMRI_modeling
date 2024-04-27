#/Applications/anaconda3/envs/deepjuice/bin/python
from pathlib import Path
import argparse
import pandas as pd
import os
from src.mri import Benchmark
from src.behavior_alignment import get_benchmarking_results
from src import frame_ops as ops
import torch
from src import tools
import time
from deepjuice.model_zoo.options import get_deepjuice_model
from deepjuice.procedural.datasets import get_data_loader
from deepjuice.extraction import FeatureExtractor
from deepjuice.systemops.devices import cuda_device_report


class VisionBehaviorEncoding:
    def __init__(self, args):
        self.process = 'VisionBehaviorEncoding'
        print('working')
        self.user = args.user
        self.overwrite = args.overwrite
        self.model_uid = args.model_uid
        self.memory_limit = args.memory_limit
        self.memory_limit_ratio = args.memory_limit_ratio
        frame_opts = ['first_frame', 'grouped_average', 'grouped_stack']
        if args.frame_handling not in frame_opts:
            raise ValueError("Invalid frame handling. Expected one of: %s" % frame_opts)
        else: 
            self.frame_handling = args.frame_handling
        self.data_dir = f'{args.top_dir}/data'
        self.cache = f'{args.top_dir}/.cache'
        torch.hub.set_dir(self.cache)

        # check hugging face cache location
        print("HF_HOME is set to:", os.environ['HF_HOME'])
        print("HUGGINGFACE_HUB_CACHE is set to:", os.environ['HUGGINGFACE_HUB_CACHE'])
        print("HF_DATASETS_CACHE is set to:", os.environ['HF_DATASETS_CACHE'])

        self.video_path = f'{self.data_dir}/raw/videos/'
        if self.frame_handling != 'first_frame':
            self.frame_path = f'{self.cache}/frames/'
            self.frames = [0, 15, 30, 45, 60, 75, 89]
            self.grouping_func = self.frame_handling
        else:
            self.frame_path = f'{self.cache}/first_frame/'
            self.frames = [0]
            self.grouping_func = None

        if self.memory_limit == 'none':
            # Calculate the memory limit and generate the feature_extractor
            total_memory_string = cuda_device_report(to_pandas=True).iloc[0]['Total Memory']
            total_memory = int(float(total_memory_string.split()[0]))
            memory_limit_int = int(total_memory * self.memory_limit_ratio)
            self.memory_limit = f'{memory_limit_int}GB'

        print(vars(self))
        self.model_name = self.model_uid.replace('/', '_')
        Path(f'{self.data_dir}/interim/{self.process}/{self.frame_handling}').mkdir(parents=True, exist_ok=True)
        self.out_file = f'{self.data_dir}/interim/{self.process}/{self.frame_handling}/model-{self.model_name}.pkl.gz'
    
    def load_data(self):
        return Benchmark(stimulus_data=f'{self.data_dir}/interim/ReorganziefMRI/stimulus_data.csv')
    
    def run(self):
        try:
            if os.path.exists(self.out_file) and not self.overwrite:
                # results = pd.read_csv(self.out_file)
                print('Output file already exists. To run again pass --overwrite.')
            else:
                start_time = time.time()
                tools.send_slack(f'Started: {self.process} {self.model_name}...', channel=self.user)
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
                print(dataloader.batch_data.head(20))

                # Perform all the regressions
                results = get_benchmarking_results(benchmark, model, dataloader,
                                                    target_features=target_features,
                                                    model_name=self.model_name,
                                                    grouping_func=self.grouping_func,
                                                    memory_limit=self.memory_limit)

                # Save
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

    parser.add_argument('--model_uid', type=str, default='torchvision_alexnet_imagenet1k_v1')
    parser.add_argument('--memory_limit', type=str, default='none')
    parser.add_argument('--memory_limit_ratio', type=float, default=.8)
    parser.add_argument('--frame_handling', type=str, default='grouped_average')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--top_dir', '-data', type=str,
                         default=f'/home/{user}/scratch4-lisik3/{user}/SIfMRI_modeling')
    args = parser.parse_args()
    VisionBehaviorEncoding(args).run()


if __name__ == '__main__':
    main()
