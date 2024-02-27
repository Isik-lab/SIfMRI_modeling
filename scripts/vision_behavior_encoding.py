#/Applications/anaconda3/envs/deepjuice/bin/python
from pathlib import Path
import argparse
import pandas as pd
import os
from src.mri import Benchmark
from src import behavior_alignment as align
import torch
from deepjuice.model_zoo.options import get_deepjuice_model
from deepjuice.procedural.datasets import get_image_loader
from deepjuice.extraction import FeatureExtractor


class BehaviorEncoding:
    def __init__(self, args):
        self.process = 'BehaviorEncoding'
        print('working')
        self.overwrite = args.overwrite
        self.model_uid = args.model_uid
        self.model_input = args.model_input
        self.data_dir = f'{args.top_dir}/data'
        self.cache = f'{args.top_dir}/.cache'
        torch.hub.set_dir(self.cache)

        # check hugging face cache location
        print("HF_HOME is set to:", os.environ['HF_HOME'])
        print("HUGGINGFACE_HUB_CACHE is set to:", os.environ['HUGGINGFACE_HUB_CACHE'])
        print("HF_DATASETS_CACHE is set to:", os.environ['HF_DATASETS_CACHE'])

        if self.model_input == 'videos':
            self.extension = 'mp4'
        else:
            self.extension = 'png'
        print(vars(self))
        self.model_name = self.model_uid.replace('/', '_')
        self.out_file = f'{self.data_dir}/interim/{self.process}/model-{self.model_name}.csv.gz'
    
    def load_data(self):
        return Benchmark(stimulus_data=f'{self.data_dir}/interim/ReorganziefMRI/stimulus_data.csv')
    
    def run(self):
        if os.path.exists(self.out_file) and not self.overwrite: 
            # results = pd.read_csv(self.out_file)
            print('Output file already exists. To run again pass --overwrite.')
        else:
            benchmark = self.load_data()
            stimulus_path = f'{self.data_dir}/raw/{self.model_input}/',
            benchmark.add_stimulus_path(data_dir=stimulus_path, extension=self.extension)
            target_features = [f for f in benchmark.stimulus_data.columns if 'rating-' in f]

            model, preprocess = get_deepjuice_model(model_name)
            images = pd.concat([benchmark.filter_stimulus('train').stimulus_data['stimulus_path'], 
                                benchmark.filter_stimulus('test').stimulus_data['stimulus_path']])
            dataloader = get_image_loader(images, preprocess)
            feature_map_extractor = FeatureExtractor(model, dataloader, memory_limit='10GB',
                                    flatten=True, progress=True, output_device='cuda:0')

            print('running regressions')
            results = align.get_training_benchmarking_results(benchmark, feature_map_extractor, 
                                                              target_features, model_name=self.model_name)
            print('saving results')
            results.to_csv(self.out_file, index=False, compression='gzip')
            print('Finished!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_uid', type=str, default='torchvision_alexnet_imagenet1k_v1')
    parser.add_argument('--model_input', type=str, default='images')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--top_dir', '-data', type=str,
                         default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIfMRI_modeling')  
    args = parser.parse_args()
    BehaviorEncoding(args).run()


if __name__ == '__main__':
    main()
