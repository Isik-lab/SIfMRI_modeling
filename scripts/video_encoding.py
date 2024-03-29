#/home/emcmaho7/.conda/envs/deepjuice_video/bin/python
from pathlib import Path
import argparse
import pandas as pd
import os
import time
from src.mri import Benchmark
from src import encoding
from src import tools
import torch
from deepjuice.extraction import FeatureExtractor
from src import video_ops
from transformers import AutoModel
from deepjuice.systemops.devices import cuda_device_report

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

    def get_model(self, model_name):
        if model_name in ['slowfast_r50', 'x3d_xs', 'x3d_s', 'x3d_m']:
            model = torch.hub.load("facebookresearch/pytorchvideo",
                                   model=self.model_name, pretrained=True).to(self.device).eval()
        elif model_name == 'xclip-base-patch32':
            model = AutoModel.from_pretrained(f"microsoft/{model_name}")
        else:
            raise Exception(f"{model_name} is not implemented!")
        return model
    
    def run(self):
        try:
            start_time = time.time()
            tools.send_slack(f'Started Rockfish job for {self.model_name}!', channel='kathy')
            if os.path.exists(self.out_file) and not self.overwrite:
                # results = pd.read_csv(self.out_file)
                print('Output file already exists. To run again pass --overwrite.')
            else:
                print('loading data...')
                benchmark = self.load_fmri()
                benchmark.add_stimulus_path(self.data_dir + f'/raw/{self.model_input}/', extension=self.extension)
                benchmark.filter_stimulus(stimulus_set='train')

                print(f'loading model {self.model_name}...')
                model = self.get_model(self.model_name)
                print(f"loaded model")

                preprocess, clip_duration = video_ops.get_transform(self.model_name)
                print(f'{preprocess}')
                print(f"Loading dataloader")
                dataloader = video_ops.get_video_loader(benchmark.stimulus_data['stimulus_path'],
                                                        clip_duration, preprocess, batch_size=5)
                print(f"loaded dataloader")

                def custom_forward(model, x):
                    return model(x)

                print(f"Creating feature extractor")
                # Calculate the memory limit and generate the feature_extractor
                total_memory_string = cuda_device_report(to_pandas=True)[0]['Total Memory']
                total_memory = int(float(total_memory_string.split()[0]))
                memory_limit = int(total_memory * 0.75)
                memory_limit_string = f'{memory_limit}GB'
                kwargs = {"forward_fn": custom_forward}
                feature_map_extractor = FeatureExtractor(model, dataloader, memory_limit=memory_limit_string, initial_report=True,
                                                         flatten=True, progress=True, **kwargs)
                print(f"loaded feature extractor")

                print('running regressions')
                results = encoding.get_training_benchmarking_results(benchmark, feature_map_extractor, self.out_path)

                print('saving results')
                results.to_csv(self.out_file, index=False)

                end_time = time.time()
                elapsed = end_time - start_time
                elapsed = time.strftime("%H:%M:%S", time.gmtime(elapsed))
                print(f'Finished in {elapsed}!')
                tools.send_slack(f'Finished for model = {self.model_name} in {elapsed}', channel='kathy')
        except Exception as err:
            print(err)
            tools.send_slack(f'ERROR! Failed for model = {self.model_name}: Error message = {err}', channel='kathy')
            raise err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='No_Model')
    parser.add_argument('--model_input', type=str, default='videos')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/home/kgarci18/scratch4-lisik3/SIfMRI_modeling/data')
                        # default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIfMRI_modeling/data')
                        # default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_modeling/data')

    args = parser.parse_args()
    VideoEncoding(args).run()


if __name__ == '__main__':
    main()
