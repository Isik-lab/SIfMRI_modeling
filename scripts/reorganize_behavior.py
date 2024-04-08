#/Applications/anaconda3/envs/nibabel/bin/python
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from glob import glob
from src import stats


class ReorganizeBehavior:
    def __init__(self, args):
        self.process = 'ReorganizeBehavior'
        self.data_dir = args.data_dir
        Path(f'{self.data_dir}/interim/{self.process}').mkdir(parents=True, exist_ok=True)
        print(vars(self))
        self.features_out = ['expanse', 'object', 'agent_distance', 'facingness', 'joint_action', 'communication', 'valence', 'arousal']
        self.features_in = ['expanse',  'object', 'distance', 'facingness', 'joint', 'communicating', 'valence', 'arousal']

    def generate_benchmark(self, video_names, possible_subjs=25):
        indiv_df = pd.read_csv(f'{self.data_dir}/raw/annotations/individual_subject_ratings.csv')

        metadata = []
        response_data = []
        for i_feature, (feature_in, feature_out) in enumerate(zip(self.features_in, self.features_out)):
            
            response_even = []
            response_odd = []
            for i_vid, video in enumerate(video_names):

                response = indiv_df.loc[(indiv_df.question_name == feature_in) & (indiv_df.video_name == video), 'likert_response'].to_numpy()
                subj_sample = np.random.default_rng(0).permutation(np.arange(len(response)))
                response_even.append(response[subj_sample[::2]].mean())
                response_odd.append(response[subj_sample[1::2]].mean())

            # Compute reliability
            r = stats.corr(np.array(response_even), np.array(response_odd))
            metadata.append({'feature': feature_out,
                            'reliability': r})
        
        metadata = pd.DataFrame(metadata)
        print(metadata.head(20))
        print(f'{metadata.shape=}')

        # Make the voxel ids unique so that there are no repeats across subjects
        return metadata
    
    def load_stimulus_data(self):
        stim_data = pd.read_csv(f'{self.data_dir}/interim/CaptionData/stimulus_data.csv')
        return stim_data, stim_data.video_name.tolist()

    def run(self):
        stimulus_data, video_names = self.load_stimulus_data()
        metadata = self.generate_benchmark(video_names)
        # stimulus_data.to_csv(f'{self.data_dir}/interim/{self.process}/stimulus_data.csv', index=False)
        metadata.to_csv(f'{self.data_dir}/interim/{self.process}/metadata.csv', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-data', type=str,
                         default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIfMRI_modeling/data')
    args = parser.parse_args()
    ReorganizeBehavior(args).run()


if __name__ == '__main__':
    main()
