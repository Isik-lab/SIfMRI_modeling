#/Applications/anaconda3/envs/deepjuice/bin/python
from pathlib import Path
import argparse
import pandas as pd
import os
from src.mri import Benchmark
from src.neural_alignment import get_benchmarking_results
import torch
from deepjuice.model_zoo.options import get_deepjuice_model
from deepjuice.procedural.datasets import get_image_loader
from deepjuice.extraction import FeatureExtractor


class VisionNeuralEncoding:
    def __init__(self, args):
        self.process = 'VisionNeuralEncoding'
        print('working')
        self.overwrite = args.overwrite
        self.test_set_evaluation = args.test_set_evaluation
        self.model_uid = args.model_uid
        self.model_input = args.model_input
        self.data_dir = f'{args.top_dir}/data'
        self.cache = f'{args.top_dir}/.cache'
        torch.hub.set_dir(self.cache)
        self.stimulus_path = f'{self.data_dir}/raw/{self.model_input}/'

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
        Path(f'{self.data_dir}/interim/{self.process}').mkdir(parents=True, exist_ok=True)

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
            benchmark.add_stimulus_path(data_dir=self.stimulus_path, extension=self.extension)
            benchmark.sort_stimulus_values(col='stimulus_set')

            model, preprocess = get_deepjuice_model(self.model_name)
            dataloader = get_image_loader(benchmark.stimulus_data['stimulus_path'], preprocess)
            feature_extractor = FeatureExtractor(model, dataloader, max_memory_load='10GB',
                                                flatten=True, progress=True, output_device='cuda:0')

            print('running regressions')
            results = get_benchmarking_results(benchmark, feature_extractor, 
                                               test_set_evaluation=self.test_set_evaluation)
            print('saving results')
            results.to_csv(self.out_file, index=False)
            print('Finished!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_uid', type=str, default='torchvision_alexnet_imagenet1k_v1')
    parser.add_argument('--model_input', type=str, default='images')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--test_set_evaluation', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--top_dir', '-top', type=str,
                         default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIfMRI_modeling')  
    args = parser.parse_args()
    VisionNeuralEncoding(args).run()


if __name__ == '__main__':
    main()
