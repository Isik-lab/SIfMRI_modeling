import argparse
import sys
from pathlib import Path
# Calculate the path to the root of the project
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
import pandas as pd
import os
from src.mri import Benchmark
from src import encoding
from deepjuice.model_zoo.options import get_deepjuice_model
from deepjuice.procedural.datasets import get_image_loader
from deepjuice.extraction import FeatureExtractor, get_feature_map_metadata
from deepjuice.model_zoo import get_model_options

class RSABenchmark:
    def __init__(self, args):
        self.process = 'RSABenchmark'
        print(f'Starting process {self.process} with args:')
        self.overwrite = args.overwrite
        self.model_uid = args.model_uid
        self.model_input = args.model_input
        self.data_dir = args.data_dir
        if self.model_input == 'videos':
            self.extension = 'mp4'
        else:
            self.extension = 'png'
        print(vars(self))
        model_name = self.model_uid.replace('/', '_')
        self.out_path = f'{self.data_dir}/interim/{self.process}/model-{model_name}'
        self.crsa_out_file = f'{self.data_dir}/interim/{self.process}/model-{model_name}_crsa.csv'
        self.ersa_out_file = f'{self.data_dir}/interim/{self.process}/model-{model_name}_ersa.csv'
        self.fmt_crsa_out_file = f'{self.data_dir}/formatted/{self.process}/model-{model_name}_crsa_fmt.csv'
        self.fmt_ersa_out_file = f'{self.data_dir}/formatted/{self.process}/model-{model_name}_ersa_fmt.csv'
        Path(self.out_path).mkdir(parents=True, exist_ok=True)

    def load_fmri(self):
        metadata_ = pd.read_csv(f'{self.data_dir}/interim/ReorganziefMRI/metadata.csv')
        response_data_ = pd.read_csv(f'{self.data_dir}/interim/ReorganziefMRI/response_data.csv.gz')
        stimulus_data_ = pd.read_csv(f'{self.data_dir}/interim/ReorganziefMRI/stimulus_data.csv')
        return Benchmark(metadata_, stimulus_data_, response_data_, rdms=True)

    def run(self):
        if os.path.exists(self.crsa_out_file) and os.path.exists(self.crsa_out_file) and not self.overwrite:
            # results = pd.read_csv(self.out_file)
            print('Output file already exists. To run again pass --overwrite.')
        else:
            print('Loading data...')
            benchmark = self.load_fmri()
            stimulus_path = f'{self.data_dir}/raw/{self.model_input}/',
            benchmark.add_stimulus_path(data_dir=stimulus_path, extension=self.extension)
            benchmark.filter_stimulus(stimulus_set='train')

            print('Loading model...')
            model, preprocess = get_deepjuice_model(self.model_uid)
            dataloader = get_image_loader(benchmark.stimulus_data['stimulus_path'], preprocess)
            print('Extracting model features...')
            feature_map_extractor = FeatureExtractor(model, dataloader,
                                                     memory_limit='10GB',
                                                     flatten=True,
                                                     output_device='cuda:0',
                                                     show_progress=True,
                                                     exclude_oversize=True)
            print('Model loaded!')

            print('Running rsa...')
            results = encoding.get_training_rsa_benchmark_results(benchmark, feature_map_extractor, self.out_path, model_name=self.model_uid)
            print('Finished RSA scoring!')
            print('Computing avg over folds')
            results_avg = results.groupby(['method', 'cv_split', 'region', 'subj_id', 'model_layer', 'model_layer_index']).mean().reset_index()
            crsa_results = results_avg[results_avg['method'] == 'crsa']
            ersa_results = results_avg[results_avg['method'] == 'ersa']
            print('Saving interim results...')
            crsa_results.to_csv(self.crsa_out_file, index=False)
            ersa_results.to_csv(self.ersa_out_file, index=False)
            print('Finished interim results!')

            try:
                print('Formatting results for plotting...')
                for metric in ['crsa', 'ersa']:
                    columns = ['region', 'model_layer', 'model_layer_index', 'subj_id', 'score']
                    df_result = results_avg[results_avg['method'] == metric].copy()
                    df_result = df_result[df_result['cv_split'] == 'test']
                    df_result = df_result[columns]
                    df_result['Model UID'] = self.model_uid
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
                    df_result = df_result[~df_result['region'].isin(['ffa', 'ppa', 'face'])]
                    custom_order = ['evc', 'mt', 'loc', 'eba', 'psts', 'asts']
                    df_result['region'] = pd.Categorical(df_result['region'], categories=custom_order, ordered=True)
                    df_result = df_result.sort_values(by='region')
                    df_result = df_result.reset_index(drop=True)
                    df_result['region'] = df_result['region'].str.upper()

                    print('Extracting model layer metadata...')
                    feature_map_metadata = get_feature_map_metadata(model, dataloader, device='cuda:0', input_dim=0)

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
                print('!Finished RSA Benchmark!')
            except Exception as err:
                print(f'Failed during formatting for plots with error msg: {err}')
                print('Skippinf formatting for plots.')
                print('!Finished RSA Benchmark!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_uid', type=str, default='slip_vit_s_yfcc15m')
    parser.add_argument('--model_input', type=str, default='images')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--data_dir', '-data', type=str,
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIfMRI_modeling/data')
    args = parser.parse_args()
    RSABenchmark(args).run()

if __name__ == '__main__':
    main()
