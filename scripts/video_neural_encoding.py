#/home/emcmaho7/.conda/envs/deepjuice_video/bin/python
from pathlib import Path
import argparse
import pandas as pd
import os
import time
from src.mri import Benchmark
from src import neural_alignment, tools, video_ops
import torch
from deepjuice.extraction import FeatureExtractor
from deepjuice.systemops.devices import cuda_device_report

class VideoNeuralEncoding:
    def __init__(self, args):
        self.process = 'VideoNeuralEncoding'
        self.overwrite = args.overwrite
        self.model_name = args.model_name
        self.model_input = args.model_input
        self.data_dir = args.data_dir
        self.user = args.user
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
        self.out_file = f'{self.data_dir}/interim/{self.process}/model-{self.model_name}.pkl.gz'
        Path(self.out_path).mkdir(parents=True, exist_ok=True)
    
    def load_fmri(self):
        metadata_ = pd.read_csv(f'{self.data_dir}/interim/ReorganziefMRI/metadata.csv')
        response_data_ = pd.read_csv(f'{self.data_dir}/interim/ReorganziefMRI/response_data.csv.gz')
        stimulus_data_ = pd.read_csv(f'{self.data_dir}/interim/ReorganziefMRI/stimulus_data.csv')
        return Benchmark(metadata_, stimulus_data_, response_data_)
    
    def run(self):
        try:
            if os.path.exists(self.out_file) and not self.overwrite:
                # results = pd.read_csv(self.out_file)
                print('Output file already exists. To run again pass --overwrite.')
            else:
                start_time = time.time()
                tools.send_slack(f'Started: :video_camera: {self.process} - {self.model_name}...', channel=self.user)
                print('Loading data...')
                benchmark = self.load_fmri()
                benchmark.add_stimulus_path(self.data_dir + f'/raw/{self.model_input}/', extension=self.extension)
                # benchmark.filter_stimulus(stimulus_set='train')

                print(f'Loading model {self.model_name}...')
                model = video_ops.get_model(self.model_name)
                if self.model_name == 'xclip-base-patch32':
                    batch_size = 1
                else:
                    batch_size = 5

                preprocess, clip_duration = video_ops.get_transform(self.model_name)
                print(f'{preprocess}')
                print(f"Loading dataloader...")
                dataloader = video_ops.get_video_loader(benchmark.stimulus_data['stimulus_path'],
                                                        clip_duration, preprocess, batch_size=batch_size)

                def custom_forward(model, x):
                    return model(x)

                def xclip_forward(model, x):
                    return model(*x)

                def transform_forward(model, x):
                    return model(**x)

                if self.model_name == 'timesformer-base-finetuned-k400':
                    kwargs = {"forward_fn": transform_forward}
                elif self.model_name == 'xclip-base-patch32':
                    kwargs = {"forward_fn": xclip_forward}
                else:
                    kwargs = {"forward_fn": custom_forward}

                # Calculate the memory limit and generate the feature_extractor
                total_memory_string = cuda_device_report(to_pandas=True)[0]['Total Memory']
                total_memory = int(float(total_memory_string.split()[0]))
                memory_limit = int(total_memory * 0.75)
                memory_limit_string = f'{memory_limit}GB'

                print(f"Creating feature extractor with {memory_limit_string} batches...")
                feature_map_extractor = FeatureExtractor(model, dataloader, memory_limit=memory_limit_string, initial_report=True,
                                                         flatten=True, progress=True, **kwargs)

                print('Running regressions...')
                results = neural_alignment.get_video_benchmarking_results(benchmark, feature_map_extractor, devices=['cuda:0'], model_name=self.model_name, test_eval=True)

                print('Saving results')
                results.to_pickle(self.out_file, compression='gzip')

                end_time = time.time()
                elapsed = end_time - start_time
                elapsed = time.strftime("%H:%M:%S", time.gmtime(elapsed))
                print(f'Finished in {elapsed}!')
                tools.send_slack(f'Finished: :video_camera: {self.process} - {self.model_name} in {elapsed} :white_check_mark:', channel=self.user)
        except Exception as err:
            print(err)
            tools.send_slack(f'Error: :video_camera: {self.process} - {self.model_name} :x: Error = {err}', channel=self.user)
            raise err


def main():
    parser = argparse.ArgumentParser()
    # Add arguments that are needed before setting the default for data_dir
    parser.add_argument('--user', type=str, default='kgarci18')
    # Parse known args first to get the user
    args, remaining_argv = parser.parse_known_args()
    user = args.user  # Get the user from the parsed known args

    parser.add_argument('--model_name', type=str, default='No_Model')
    parser.add_argument('--model_input', type=str, default='videos')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--data_dir', '-data', type=str,
                        default=f'/home/{user}/scratch4-lisik3/{user}/SIfMRI_modeling/data')

    args = parser.parse_args(remaining_argv)
    VideoNeuralEncoding(args).run()


if __name__ == '__main__':
    main()
