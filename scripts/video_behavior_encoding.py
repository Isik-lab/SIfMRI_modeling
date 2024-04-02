# /Applications/anaconda3/envs/deepjuice/bin/python
import torch
import time
import argparse
import pandas as pd
import os
from src.mri import Benchmark
from src import video_ops, behavior_alignment, tools
from deepjuice.extraction import FeatureExtractor
from pathlib import Path
from deepjuice.systemops.devices import cuda_device_report
from transformers import AutoModel, VideoMAEModel, TimesformerForVideoClassification

class VideoBehaviorEncoding:
    def __init__(self, args):
        self.process = 'VideoBehaviorEncoding'
        self.overwrite = args.overwrite
        self.model_name = args.model_name
        self.model_input = args.model_input
        self.data_dir = args.data_dir
        self.user = args.user
        self.extension = 'mp4'
        print(vars(self))
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.out_path = f'{self.data_dir}/interim/{self.process}/model-{self.model_name}'
        self.out_file = f'{self.data_dir}/interim/{self.process}/model-{self.model_name}.pkl.gz'
        Path(self.out_path).mkdir(parents=True, exist_ok=True)

    def load_data(self):
        return Benchmark(stimulus_data=f'{self.data_dir}/interim/ReorganziefMRI/stimulus_data.csv')

    def get_model(self, model_name):
        if model_name in torch.hub.list('facebookresearch/pytorchvideo', force_reload=True):
            model = torch.hub.load("facebookresearch/pytorchvideo", model=self.model_name, pretrained=True).to(self.device).eval()
        elif model_name == 'xclip-base-patch32':
            model = AutoModel.from_pretrained(f"microsoft/{model_name}")
        elif model_name.lower() == 'videomae_base_short':
            model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
        elif model_name.lower() == 'timesformer-base-finetuned-k400':
            model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400")
        else:
            raise Exception(f"{model_name} is not implemented!")
        return model

    def run(self):
        try:
            if os.path.exists(self.out_file) and not self.overwrite:
                # results = pd.read_csv(self.out_file)
                print('Output file already exists. To run again pass --overwrite.')
            else:
                start_time = time.time()
                tools.send_slack(f'Started: {self.process} {self.model_name} on Rockfish...', channel=self.user)
                # Load data and sort
                benchmark = self.load_data()
                benchmark.add_stimulus_path(self.data_dir + f'/raw/{self.model_input}/', extension=self.extension)
                print(f'Loading target features...')
                target_features = [col for col in benchmark.stimulus_data.columns if
                                   ('rating-' in col) and ('indoor' not in col)]

                print(f'Loading model {self.model_name}...')
                model = self.get_model(self.model_name)

                preprocess, clip_duration = video_ops.get_transform(self.model_name)
                print(f'{preprocess}')
                print(f"Loading dataloader...")
                dataloader = video_ops.get_video_loader(benchmark.stimulus_data['stimulus_path'],
                                                        clip_duration, preprocess, batch_size=5)

                def custom_forward(model, x):
                    return model(x)

                def transform_forward(model, x):
                    return model(**x)

                if self.model_name == 'timesformer-base-finetuned-k400':
                    kwargs = {"forward_fn": transform_forward}
                else:
                    kwargs = {"forward_fn": custom_forward}

                # Calculate the memory limit and generate the feature_extractor
                total_memory_string = cuda_device_report(to_pandas=True)[0]['Total Memory']
                total_memory = int(float(total_memory_string.split()[0]))
                memory_limit = int(total_memory * 0.75)
                memory_limit_string = f'{memory_limit}GB'

                print(f"Creating feature extractor...")
                feature_map_extractor = FeatureExtractor(model, dataloader, memory_limit=memory_limit_string, initial_report=True,
                                                         flatten=True, progress=True, **kwargs)

                # Perform all the regressions
                print('Running regressions...')
                results = behavior_alignment.get_video_benchmarking_results(benchmark, feature_map_extractor, target_features=target_features, model_name=self.model_name, devices=['cuda:0'])
                print(results.head(20))
                print('Saving results...')
                results.to_pickle(self.out_file, compression='gzip')

                end_time = time.time()
                elapsed = end_time - start_time
                elapsed = time.strftime("%H:%M:%S", time.gmtime(elapsed))
                print(f'Finished in {elapsed}!')
                tools.send_slack(f'Finished: {self.process} {self.model_name} in {elapsed} :baby-yoda:', channel=self.user)
        except Exception as err:
            print(err)
            tools.send_slack(f'Error: {self.process} {self.model_name} Error = {err}', channel=self.user)
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
                        # default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIfMRI_modeling/data')
                        # default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_modeling/data')
    args = parser.parse_args(remaining_argv)
    VideoBehaviorEncoding(args).run()


if __name__ == '__main__':
    main()
