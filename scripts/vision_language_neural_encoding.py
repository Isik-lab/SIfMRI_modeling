# /Applications/anaconda3/envs/deepjuice/bin/python
from pathlib import Path
import argparse
import pandas as pd
import os
from src.mri import Benchmark
from src import neural_alignment, tools, video_ops
from src.language_ops import parse_caption_data, get_model
import src.multimodal_ops as mmops
from src.language_ablation import perturb_captions
from src import frame_ops as frameops
from src import tools
from deepjuice.extraction import FeatureExtractor
import torch
import ast
from deepjuice.systemops.devices import cuda_device_report


class VisionLanguageNeuralEncoding:
    def __init__(self, args):
        self.process = 'VisionLanguageNeuralEncoding'
        self.user = args.user
        self.overwrite = args.overwrite
        self.test_eval = args.test_eval
        self.model_uid = args.model_uid
        self.memory_limit = args.memory_limit
        self.memory_limit_ratio = args.memory_limit_ratio
        self.frame_handling = args.frame_handling
        self.modality = args.modality
        self.perturb_func = 'none'
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

        # Memory limit
        if self.memory_limit == 'none':
            # Calculate the memory limit and generate the feature_extractor
            total_memory_string = cuda_device_report(to_pandas=True).iloc[0]['Total Memory']
            total_memory = int(float(total_memory_string.split()[0]))
            memory_limit_int = int(total_memory * self.memory_limit_ratio)
            self.memory_limit = f'{memory_limit_int}GB'

        print(vars(self))
        self.model_name = self.model_uid.replace('/', '_')
        Path(f'{self.data_dir}/interim/{self.process}/{self.frame_handling}/{self.perturb_func}').mkdir(parents=True, exist_ok=True)
        self.out_file = f'{self.data_dir}/interim/{self.process}/{self.frame_handling}/{self.perturb_func}/model-{self.model_name}.parquet'
        self.input_file = f'{self.data_dir}/interim/{self.process}/{self.frame_handling}/{self.perturb_func}/{self.perturb_func}.csv'

    def load_fmri(self):
        metadata_ = pd.read_csv(f'{self.data_dir}/interim/ReorganziefMRI/metadata.csv')
        response_data_ = pd.read_csv(f'{self.data_dir}/interim/ReorganziefMRI/response_data.csv.gz')
        stimulus_data_ = pd.read_csv(f'{self.data_dir}/interim/ReorganziefMRI/stimulus_data.csv')
        return Benchmark(metadata_, stimulus_data_, response_data_)

    def load_captions(self):
        if not os.path.exists(self.input_file):
            file = f'{self.data_dir}/interim/CaptionData/captions.csv'
            df = parse_caption_data(file)
            perturb_captions(df, func_name=self.perturb_func)
            df.to_csv(self.input_file, index=False)
            return df
        else:
            return pd.read_csv(self.input_file)

    def run(self):
        try:
            if os.path.exists(self.out_file) and not self.overwrite:
                # results = pd.read_csv(self.out_file)
                print('Output file already exists. To run again pass --overwrite.')
            else:
                run_timer = tools.TimeBlock()
                run_timer.start()
                tools.send_slack(f'Started: {self.process} {self.model_name}...', channel=self.user)

                benchmark = self.load_fmri()
                captions = self.load_captions()
                # Break the videos into frames for averaging
                frame_data = frameops.visual_events(benchmark.stimulus_data,
                                               self.video_path, self.frame_path,
                                               frame_idx=self.frames)
                frame_data['captions'] = frame_data['captions'].apply(ast.literal_eval)

                model, preprocess = mmops.get_model(self.model_name, self.modality)

                print('Running dataloader...')
                dataloader = mmops.get_multimodal_loader(frame_data,
                                                         captions,
                                                         preprocess,
                                                         batch_size=16,
                                                         group_keys=None,
                                                         image_key='images',
                                                         caption_key='captions',
                                                         device='cuda')


                print(dataloader.batch_data.head(20))

                def forward_fn(model, inputs):
                    return model(**inputs)
                kwargs = {"forward_fn": forward_fn}

                print(f"Creating feature extractor with {self.memory_limit} batches...")
                feature_map_extractor = FeatureExtractor(model, dataloader, memory_limit=self.memory_limit, initial_report=True,
                                                         flatten=True, progress=True, exclude_oversize=True, **kwargs)
                benchmark_setup_elapsed = run_timer.elapse()

                print('Running regressions...')
                results = neural_alignment.get_video_benchmarking_results(benchmark, feature_map_extractor, devices=['cuda:0'], model_name=self.model_name, test_eval=True)

                print('Saving results')
                save_timer = tools.TimeBlock()
                save_timer.start()
                results.to_parquet(self.out_file, compression='gzip')
                save_elapsed = save_timer.elapse()

                timers = {}
                timers['Benchmark Setup Time'] = benchmark_setup_elapsed
                timers['File Save Time'] = save_elapsed
                elapsed = run_timer.elapse()

                tools.send_slack(
                    f'Finished: {self.process} {self.model_name} - Total time =  {elapsed} \nTimeBlock output:',
                    channel=self.user)
                for key, value in timers.items():
                    tools.send_slack(f'- {key.title()} time = {value}', channel=self.user)
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
    parser.add_argument('--memory_limit_ratio', type=float, default=.6)
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--test_eval', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--frame_handling', type=str, default='grouped_average')
    parser.add_argument('--modality', type=str, default='vision-language')
    parser.add_argument('--top_dir', '-top', type=str,
                        default=f'/home/{user}/scratch4-lisik3/{user}/SIfMRI_modeling')
    args = parser.parse_args()
    VisionLanguageNeuralEncoding(args).run()


if __name__ == '__main__':
    main()
