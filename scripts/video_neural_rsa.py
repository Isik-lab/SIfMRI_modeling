import argparse
from pathlib import Path
import pandas as pd
import os
from src.mri import Benchmark
from deepjuice.extraction import FeatureExtractor, get_feature_map_metadata
from src import neural_alignment, tools, video_ops
import time
import gc
from deepjuice.systemops.devices import cuda_device_report

class VideoNeuralRSA:
    """
    A class for conducting Representational Similarity Analysis (RSA) benchmarking
    of VIDEO based neural network models against fMRI data.

    Attributes:
        process (str): Name of the process, set to 'RSABenchmark'.
        overwrite (bool): Flag to overwrite existing results.
        model_uid (str): Unique identifier for the model being benchmarked.
        model_input (str): Type of input data for the model ('images' or 'videos').
        data_dir (str): Directory path where the data is stored.
        extension (str): File extension of the stimulus data.
        fmt_out_file (str): Path for saving formatted results.
        raw_out_file (str): Path for saving the raw un-aggregated kfold results.

    Methods:
        __init__(self, args): Initializes the benchmarking process.
        load_fmri(self): Loads fMRI data required for RSA.
        run(self): Conducts the RSA benchmarking and saves the results.
    """
    def __init__(self, args):
        """
         Initializes the VideoNeuralRSA class with necessary parameters for the RSA benchmarking process.

         Parameters:
             args: Argument parser outputs containing model UID, data directory, overwrite flag, and model input type.
        """
        self.process = 'VideoNeuralRSA'
        print(f'Starting process {self.process} with args:')
        self.overwrite = args.overwrite
        self.user = args.user
        self.model_uid = args.model_uid
        self.model_input = args.model_input
        self.data_dir = args.data_dir
        if self.model_input == 'videos':
            self.extension = 'mp4'
        else:
            self.extension = 'png'
        print(vars(self))
        model_name = self.model_uid.replace('/', '_')
        self.out_path = f'{self.data_dir}/interim/{self.process}'
        self.raw_out_file = f'{self.data_dir}/interim/{self.process}/model-{model_name}_raw.csv'
        self.fmt_out_file = f'{self.data_dir}/interim/{self.process}/model-{model_name}_fmt.csv'
        Path(self.out_path).mkdir(parents=True, exist_ok=True)

    def load_fmri(self) -> Benchmark:
        """
         Loads the fMRI data including metadata, response data, and stimulus data from the specified directory.

         Returns:
             Benchmark: An instance of the Benchmark class initialized with loaded fMRI data and ready for RSA analysis.
        """
        metadata_ = pd.read_csv(f'{self.data_dir}/interim/ReorganziefMRI/metadata.csv')
        response_data_ = pd.read_csv(f'{self.data_dir}/interim/ReorganziefMRI/response_data.csv.gz')
        stimulus_data_ = pd.read_csv(f'{self.data_dir}/interim/ReorganziefMRI/stimulus_data.csv')
        return Benchmark(metadata_, stimulus_data_, response_data_)

    def run(self):
        """
        Executes the RSA benchmarking process. This includes loading fMRI data, preparing stimulus data,
        loading and preparing the model for feature extraction, running RSA metrics, and saving the results.
        """
        start_time = time.time()
        if os.path.exists(self.raw_out_file) and not self.overwrite:
            print('Output file already exists. To run again pass --overwrite.')
            return
        else:
            try:
                start_time = time.time()
                tools.send_slack(f'Started :baby-yoda: : {self.process} {self.model_uid}...', channel=self.user)
                print('Loading data...')
                benchmark = self.load_fmri()
                stimulus_path = f'{self.data_dir}/raw/{self.model_input}/',
                benchmark.add_stimulus_path(data_dir=stimulus_path, extension=self.extension)
                # benchmark.filter_stimulus(stimulus_set='train')
                benchmark.generate_rdms()  # generates both train and test rdms and filters responses

                print(f'Loading model {self.model_uid}...')
                model = video_ops.get_model(self.model_uid)
                if self.model_uid == 'xclip-base-patch32':
                    batch_size = 1
                else:
                    batch_size = 5
                print('Model loaded!')

                preprocess, clip_duration = video_ops.get_transform(self.model_uid)
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

                if self.model_uid == 'timesformer-base-finetuned-k400':
                    kwargs = {"forward_fn": transform_forward}
                elif self.model_uid == 'xclip-base-patch32':
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

                print('Running rsa...')
                results = neural_alignment.get_video_rsa_benchmark_results(benchmark, feature_map_extractor, model_name=self.model_uid, test_eval=True, raw_output_file=self.raw_out_file)
                print('Finished RSA scoring!')

                print(f'Saving formatted results to {self.fmt_out_file}...')
                results.to_csv(self.fmt_out_file, index=False)

                end_time = time.time()
                elapsed = end_time - start_time
                elapsed = time.strftime("%H:%M:%S", time.gmtime(elapsed))
                print(f'Finished in {elapsed}!')
                tools.send_slack(f'Finished :white_check_mark: : {self.process} {self.model_uid} in {elapsed}', channel=self.user)
            except Exception as err:
                print(f'Error: {self.process} {self.model_uid}: Error Msg = {err}')
                tools.send_slack(f'Error :x: : {self.process} {self.model_uid}: Error Msg = {err}', channel=self.user)
                raise err

def main():
    parser = argparse.ArgumentParser()
    # Add arguments that are needed before setting the default for data_dir
    parser.add_argument('--user', type=str, default='kgarci18')
    # Parse known args first to get the user
    args, remaining_argv = parser.parse_known_args()
    user = args.user  # Get the user from the parsed known args

    parser.add_argument('--model_uid', type=str, default='slip_vit_s_yfcc15m')
    parser.add_argument('--model_input', type=str, default='images')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--data_dir', '-data', type=str,
                        default=f'/home/{user}/scratch4-lisik3/{user}/SIfMRI_modeling/data')

    args = parser.parse_args()
    VideoNeuralRSA(args).run()

if __name__ == '__main__':
    main()
