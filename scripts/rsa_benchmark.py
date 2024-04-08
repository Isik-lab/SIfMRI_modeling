import argparse
import sys
from pathlib import Path
from torch import hub
# Calculate the path to the root of the project
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
import pandas as pd
import os
import time
from src.mri import Benchmark
from src import neural_alignment
from src import tools
from deepjuice.model_zoo.options import get_deepjuice_model
from deepjuice.procedural.datasets import get_image_loader
from deepjuice.extraction import FeatureExtractor, get_feature_map_metadata
from deepjuice.model_zoo import get_model_options
from src import tools
import time


class RSABenchmark:
    """
    A class for conducting Representational Similarity Analysis (RSA) benchmarking
    of neural network models against fMRI data.

    Attributes:
        process (str): Name of the process, set to 'RSABenchmark'.
        overwrite (bool): Flag to overwrite existing results.
        model_uid (str): Unique identifier for the model being benchmarked.
        model_input (str): Type of input data for the model ('images' or 'videos').
        data_dir (str): Directory path where the data is stored.
        extension (str): File extension of the stimulus data.
        crsa_out_file (str): Path for saving CRSA results.
        ersa_out_file (str): Path for saving ERSA results.
        fmt_crsa_out_file (str): Path for saving formatted CRSA results.
        fmt_ersa_out_file (str): Path for saving formatted ERSA results.
        raw_out_file (str): Path for saving the raw un-aggregated kfold results.

    Methods:
        __init__(self, args): Initializes the benchmarking process.
        load_fmri(self): Loads fMRI data required for RSA.
        run(self): Conducts the RSA benchmarking and saves the results.
    """
    def __init__(self, args):
        """
         Initializes the RSABenchmark class with necessary parameters for the RSA benchmarking process.

         Parameters:
             args: Argument parser outputs containing model UID, data directory, overwrite flag, and model input type.
        """
        self.process = 'RSABenchmark'
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
        self.cache = f'{args.top_dir}/.cache'
        # set cache location
        hub.set_dir(self.cache)
        print(vars(self))
        model_name = self.model_uid.replace('/', '_')
        self.crsa_out_file = f'{self.data_dir}/interim/{self.process}/model-{model_name}_crsa.csv'
        self.ersa_out_file = f'{self.data_dir}/interim/{self.process}/model-{model_name}_ersa.csv'
        self.fmt_crsa_out_file = f'{self.data_dir}/interim/{self.process}/model-{model_name}_crsa_fmt.csv'
        self.fmt_ersa_out_file = f'{self.data_dir}/interim/{self.process}/model-{model_name}_ersa_fmt.csv'
        self.raw_out_file = f'{self.data_dir}/interim/{self.process}/model-{model_name}_raw.csv'

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
        if os.path.exists(self.crsa_out_file) and os.path.exists(self.ersa_out_file) and not self.overwrite:
            print('Output file already exists. To run again pass --overwrite.')
            return
        else:
            try:
                start_time = time.time()
                tools.send_slack(f'Started: {self.process} {self.model_name}...', channel=self.user)
                print('Loading data...')
                benchmark = self.load_fmri()
                stimulus_path = f'{self.data_dir}/raw/{self.model_input}/',
                benchmark.add_stimulus_path(data_dir=stimulus_path, extension=self.extension)
                benchmark.filter_stimulus(stimulus_set='train')
                benchmark.generate_rdms()

                print('Loading model...')
                model, preprocess = get_deepjuice_model(self.model_uid)
                dataloader = get_image_loader(benchmark.stimulus_data['stimulus_path'], preprocess)
                print('Extracting model features...')
                feature_map_extractor = FeatureExtractor(model, dataloader,
                                                         memory_limit='10GB',
                                                         flatten=True,
                                                         output_device='cuda',
                                                         show_progress=False,
                                                         exclude_oversize=True)
                print('Model loaded!')
                print('Running rsa...')
                results = neural_alignment.get_training_rsa_benchmark_results(benchmark, feature_map_extractor, model_uid=self.model_uid)
                print('Finished RSA scoring!')
                results.to_csv(self.raw_out_file, index=False)
                print(f'Raw results saved to {self.raw_out_file}')
                results = pd.read_csv(self.raw_out_file)
                print('Computing avg over folds')
                results_avg = results.groupby(['model_uid', 'metric', 'cv_split', 'region', 'subj_id', 'model_layer', 'model_layer_index']).mean(numeric_only=True).reset_index()
                crsa_results = results_avg[results_avg['metric'] == 'crsa']
                ersa_results = results_avg[results_avg['metric'] == 'ersa']
                print('Saving interim results...')
                crsa_results.to_csv(self.crsa_out_file, index=False)
                ersa_results.to_csv(self.ersa_out_file, index=False)
                print('Finished interim results!')

                print('Formatting results for plotting...')
                for metric in ['crsa', 'ersa']:
                    columns = ['model_uid', 'region', 'model_layer', 'model_layer_index', 'subj_id', 'score']
                    df_result = results_avg[results_avg['metric'] == metric].copy()
                    df_result = df_result[df_result['cv_split'] == 'test']
                    df_result = df_result[columns]
                    df_result['Model UID'] = df_result['model_uid']
                    df_all_models = get_model_options()
                    df_result['Model Name'] = df_all_models[df_all_models['model_uid'] == self.model_uid]['display_name'].values[0]
                    df_result['Model Name Short'] = df_all_models[df_all_models['model_uid'] == self.model_uid]['model_name'].values[0]
                    df_result['Architecture Type'] = df_all_models[df_all_models['model_uid'] == self.model_uid]['architecture_type'].values[0]
                    df_result['Architecture'] = df_all_models[df_all_models['model_uid'] == self.model_uid]['architecture'].values[0]
                    df_result['Train Task'] = df_all_models[df_all_models['model_uid'] == self.model_uid]['train_task_display'].values[0]
                    df_result['Train Data'] = df_all_models[df_all_models['model_uid'] == self.model_uid]['train_data_display'].values[0]
                    df_result['Task Cluster'] = df_all_models[df_all_models['model_uid'] == self.model_uid]['task_cluster'].values[0]
                    aggregation = {
                        'model_layer': 'first',
                        'score': 'mean',
                        'Model Name': 'first',
                        'Model Name Short': 'first',
                        'Architecture Type': 'first',
                        'Architecture': 'first',
                        'Train Task': 'first',
                        'Train Data': 'first',
                        'Task Cluster': 'first'
                    }
                    # average the subjects
                    df_result = df_result.groupby(['Model UID', 'region', 'model_layer_index']).agg(aggregation).reset_index()
                    # get the best score per region
                    idx = df_result.groupby(['Model UID', 'region'])['score'].idxmax()
                    df_result = df_result.loc[idx].reset_index(drop=True)
                    df_result['region'] = df_result['region'].str.lower()
                    # df_result = df_result[~df_result['region'].isin(['ffa', 'ppa', 'face'])]
                    custom_order = ['evc', 'mt', 'loc', 'eba', 'psts', 'face-psts', 'asts', 'ffa', 'ppa']
                    df_result['region'] = pd.Categorical(df_result['region'], categories=custom_order, ordered=True)
                    df_result = df_result.sort_values(by='region')
                    df_result = df_result.reset_index(drop=True)
                    df_result['region'] = df_result['region'].str.upper()

                    # Add feature map metadata
                    print('Extracting model layer metadata...')
                    feature_map_metadata = get_feature_map_metadata(model, dataloader, device='cuda', input_dim=0)
                    print('Model layers metadata loaded!')
                    df_result = df_result.merge(
                        feature_map_metadata[['output_uid', 'output_depth']].rename(
                            columns={'output_uid': 'model_layer', 'output_depth': 'Layer Depth'}),
                        on=['model_layer'],
                        how='left')
                    del feature_map_metadata
                    print('Saving formatted results...')
                    if metric == 'crsa':
                        df_result.to_csv(self.fmt_crsa_out_file, index=False)
                    elif metric == 'ersa':
                        df_result.to_csv(self.fmt_ersa_out_file, index=False)
                print('Finished formatted results!')

                end_time = time.time()
                elapsed = end_time - start_time
                elapsed = time.strftime("%H:%M:%S", time.gmtime(elapsed))
                print(f'Finished in {elapsed}!')
                tools.send_slack(f'Finished: {self.process} {self.model_name} in {elapsed}', channel=self.user)
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

    parser.add_argument('--model_uid', type=str, default='slip_vit_s_yfcc15m')
    parser.add_argument('--model_input', type=str, default='images')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--data_dir', '-data', type=str,
                        default=f'/home/{user}/scratch4-lisik3/{user}/SIfMRI_modeling')
    parser.add_argument('--top_dir', type=str, default=f'/home/{user}/scratch4-lisik3/{user}/SIfMRI_modeling')
    args = parser.parse_args()
    RSABenchmark(args).run()

if __name__ == '__main__':
    main()
