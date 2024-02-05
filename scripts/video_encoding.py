#/Applications/anaconda3/envs/nibabel/bin/python
from pathlib import Path
import argparse
import pandas as pd
import os
from src.mri import Benchmark
from src import encoding
import torch
from deepjuice.extraction import FeatureExtractor
from src.video import get_video_loader, get_transform


class VideoEncoding:
    def __init__(self, args):
        self.process = 'VideoEncoding'
        self.overwrite = args.overwrite
        self.model_name = args.model_name
        self.model_input = args.model_input
        self.data_dir = args.data_dir
        if self.model_input == 'videos':
            self.extension = 'mp4'
        else:
            self.extension = 'png'
        print(vars(self))
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.out_path = f'{self.data_dir}/interim/{self.process}/model-{self.model_name}'
        self.out_file = f'{self.data_dir}/interim/{self.process}/model-{self.model_name}.csv'
        Path(self.out_path).mkdir(parents=True, exist_ok=True)
    
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
            print('loading data...')
            benchmark = self.load_fmri()
            benchmark.add_stimulus_path(self.data_dir + f'/raw/{self.model_input}/', extension=self.extension)
            benchmark.filter_stimulus(stimulus_set='train')

            print('loading model...')
            model = torch.hub.load("facebookresearch/pytorchvideo",
                                   model=self.model_name, pretrained=True).to(self.device).eval()
            preprocess, clip_duration = get_transform(self.model_name)
            dataloader = get_video_loader(benchmark.stimulus_data['stimulus_path'], 
                                          preprocess, clip_duration)
            feature_map_extractor = FeatureExtractor(model, dataloader, max_memory_load='24GB',
                                                     initial_report=False, 
                                                     flatten=True, progress=True)
            
            print('running regressions')
            results = encoding.get_training_benchmarking_results(benchmark, feature_map_extractor, self.out_path)

            print('saving results')
            results.to_csv(self.out_file, index=False)
            print('Finished!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='slowfast_r50')
    parser.add_argument('--model_input', type=str, default='images')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--data_dir', '-data', type=str,
                         default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIfMRI_modeling/data')                        
                        # default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_modeling/data')

    args = parser.parse_args()
    VideoEncoding(args).run()


if __name__ == '__main__':
    main()




