#/Applications/anaconda3/envs/nibabel/bin/python
from pathlib import Path
import argparse
import pandas as pd
import os
import shutil
# from src.mri import Benchmark
# from src import encoding
# from deepjuice.model_zoo.options import get_deepjuice_model
# from deepjuice.procedural.datasets import get_image_loader
# from deepjuice.extraction import FeatureExtractor
from src.video_ops import visual_events_benchmark, run_visual_event_pipeline


class VisionEncoding:
    def __init__(self, args):
        self.process = 'VisionEncoding'
        self.overwrite = args.overwrite
        self.model_uid = args.model_uid
        self.model_input = args.model_input
        self.data_dir = args.data_dir
        self.fresh_start = args.fresh_start
        self.key_frames = [0, 22, 45, 67, 89]
        self.device = 'cuda:0'
        print(vars(self))

        model_name = self.model_uid.replace('/', '_')
        self.out_path = f'{self.data_dir}/interim/{self.process}/model-{model_name}'
        self.out_file = f'{self.data_dir}/interim/{self.process}/model-{model_name}.csv'
        Path(self.out_path).mkdir(parents=True, exist_ok=True)
    
    def load_fmri(self):
        if os.path.exists(f'{self.data_dir}/raw/frames') and self.fresh_start:
            shutil.rmtree(f'{self.data_dir}/raw/frames') # delete dir

        benchmark = visual_events_benchmark(event_data=f'{self.data_dir}/raw/annotations/annotations.csv', 
                                            video_dir=f'{self.data_dir}/raw/videos', 
                                            image_dir=f'{self.data_dir}/raw/frames',
                                            key_frames=self.key_frames,
                                            target_index=None)
        return benchmark
        # metadata_ = pd.read_csv(f'{self.data_dir}/interim/ReorganziefMRI/metadata.csv')
        # response_data_ = pd.read_csv(f'{self.data_dir}/interim/ReorganziefMRI/response_data.csv.gz')
        # stimulus_data_ = pd.read_csv(f'{self.data_dir}/interim/ReorganziefMRI/stimulus_data.csv')
        # return Benchmark(metadata_, stimulus_data_, response_data_)
    
    def run(self):
        if os.path.exists(self.out_file) and not self.overwrite: 
            # results = pd.read_csv(self.out_file)
            print('Output file already exists. To run again pass --overwrite.')
        else:
            benchmark = self.load_fmri()
            results = run_visual_event_pipeline(self.model_uid, benchmark, self.device)
            
            # benchmark = self.load_fmri()
            # benchmark.add_stimulus_path(self.data_dir + f'/raw/{self.model_input}/', extension=self.extension)
            # benchmark.filter_stimulus(stimulus_set='train')

            # print('loading model...')
            # model, preprocess = get_deepjuice_model(self.model_uid)
            # dataloader = get_image_loader(benchmark.stimulus_data['stimulus_path'], preprocess)
            # feature_map_extractor = FeatureExtractor(model, dataloader, max_memory_load='24GB',
            #                         flatten=True, progress=True)
            
            # print('running regressions')
            # results = encoding.get_training_benchmarking_results(benchmark, feature_map_extractor, self.out_path)

            print('saving results')
            results.to_csv(self.out_file, index=False)
            print('Finished!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_uid', type=str, default='torchvision_alexnet_imagenet1k_v1')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--data_dir', '-data', type=str,
                         default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIfMRI_modeling/data')                        
                        # default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_modeling/data')

    args = parser.parse_args()
    VisionEncoding(args).run()


if __name__ == '__main__':
    main()