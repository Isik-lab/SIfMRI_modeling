#/Applications/anaconda3/envs/nibabel/bin/python
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import os
import shutil
from src.mri import Benchmark
from deepjuice.model_zoo.options import get_deepjuice_model
from deepjuice.procedural.datasets import get_data_loader
from deepjuice.extraction import FeatureExtractor
from src.video_ops import visual_events
from src.encoding import get_vision_benchmarking_results, moving_grouped_average


class VisionEncoding:
    def __init__(self, args):
        self.process = 'VisionEncoding'
        self.overwrite = args.overwrite
        self.save_frames = args.save_frames
        self.model_uid = args.model_uid
        self.data_dir = args.data_dir
        self.key_frames = list(np.arange(1, 91, 30) - 1)
        self.device = args.device
        print(vars(self))

        model_name = self.model_uid.replace('/', '_')
        self.out_path = f'{self.data_dir}/interim/{self.process}/model-{model_name}'
        self.out_file = f'{self.data_dir}/interim/{self.process}/model-{model_name}.csv'
        Path(self.out_path).mkdir(parents=True, exist_ok=True)
    
    def load_fmri(self):
        metadata_ = pd.read_csv(f'{self.data_dir}/interim/ReorganziefMRI/metadata.csv')
        response_data_ = pd.read_csv(f'{self.data_dir}/interim/ReorganziefMRI/response_data.csv.gz')
        stimulus_data_ = pd.read_csv(f'{self.data_dir}/interim/ReorganziefMRI/stimulus_data.csv')
        return Benchmark(metadata_, stimulus_data_, response_data_)

    def get_frames(self, benchmark):
        if self.save_frames:
            shutil.rmtree(f'{self.data_dir}/raw/frames') # delete frames
        events = visual_events(stimulus_data=benchmark.stimulus_data, 
                                            video_dir=f'{self.data_dir}/raw/videos', 
                                            image_dir=f'{self.data_dir}/raw/frames',
                                            key_frames=self.key_frames,
                                            target_index=None)
        benchmark.update(events)
        return benchmark
    
    
    def run(self):
        if os.path.exists(self.out_file) and not self.overwrite: 
            # results = pd.read_csv(self.out_file)
            print('Output file already exists. To run again pass --overwrite.')
        else:
            benchmark = self.load_fmri()
            benchmark.filter_stimulus(stimulus_set='train')
            benchmark = self.get_frames(benchmark)

            # print('loading model...')
            model, preprocess = get_deepjuice_model(self.model_uid)
            dataloader = get_data_loader(benchmark.image_paths, preprocess)

            # define function to average over frames 
            skip = len(list(benchmark.group_indices.values())[0])
            def tensor_fn(tensor):
                return moving_grouped_average(tensor, skip)
            
            #define the feature extractor
            extractor = FeatureExtractor(model, dataloader, 
                                        tensor_fn=tensor_fn,
                                        initial_report=False, 
                                        batch_size=len(self.key_frames))
            extractor.modify_settings(flatten=True, batch_progress=True)
            
            # print('running regressions')
            results = get_vision_benchmarking_results(benchmark, extractor, self.out_path)

            print('saving results')
            results.to_csv(self.out_file, index=False)
            print('Finished!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_uid', type=str, default='torchvision_alexnet_imagenet1k_v1')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--save_frames', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--data_dir', '-data', type=str,
                         default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIfMRI_modeling/data')                        
                        # default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_modeling/data')

    args = parser.parse_args()
    VisionEncoding(args).run()


if __name__ == '__main__':
    main()