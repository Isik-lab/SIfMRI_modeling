#
import re
import argparse
from glob import glob
import os
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer


def corr(x, y):
    x_m = x - np.nanmean(x)
    y_m = y - np.nanmean(y)
    numer = np.nansum(x_m * y_m)
    denom = np.sqrt(np.nansum(x_m * x_m) * np.nansum(y_m * y_m))
    if denom != 0:
        return numer / denom
    else:
        return np.nan


def noise_ceiling(rows):
    even = rows[rows.even].groupby('video_name').mean(numeric_only=True).reset_index().sort_values(by='video_name').likert_response.to_numpy()
    odd = rows[~rows.even].groupby('video_name').mean(numeric_only=True).reset_index().sort_values(by='video_name').likert_response.to_numpy()
    return corr(even, odd)


class CaptionData:
    def __init__(self, args):
        # save arg inputs into self
        self.top_dir = args.top_dir

        # Set up the directories
        self.process = 'CaptionData'
        self.figures_dir = f'{self.top_dir}/reports/figures/{self.process}'
        self.interim_dir = f'{self.top_dir}/data/interim'
        self.raw_dir = f'{self.top_dir}/data/raw'
        self.out_path = f'{self.interim_dir}/{self.process}'
        Path(self.out_path).mkdir(exist_ok=True, parents=True)
        Path(self.figures_dir).mkdir(exist_ok=True, parents=True)
        print(vars(self))
        # Set environment variables
        self.catch_trials = ['flickr-0-5-7-5-4-0-7-0-2605754070_54.mp4', 'yt-dfOVWymr76U_103.mp4']

    def load_all_data(self):
        all_data = []
        for path in glob(f'{self.raw_dir}/captions/*.csv'):
            match = re.search(r'sub-(.*?)_condition-(.*?)_(.*).csv', path)
            if match:
                sub_id = match.group(1)
                condition = match.group(2)
                date = match.group(3)

            if os.path.getsize(path) != 0:
                df = pd.read_csv(path, header=None)
                df.columns = ['url', 'caption']
                df['url'] = df['url'].str.extract("(https://[^']+)")
                df['video_name'] = df['url'].str.extract(r'/([^/]+\.mp4)')[0]
                df[['sub_id', 'condition', 'date']] = sub_id, condition, date
                all_data.append(df)
        return pd.concat(all_data)
    
    def get_complete_data(self, all_sub_data):
        incomplete_data = all_sub_data.groupby('sub_id').filter(lambda x: len(np.unique(x.video_name)) < 12)
        data = all_sub_data.groupby('sub_id').filter(lambda x: len(np.unique(x.video_name)) == 12)
        extra_data = all_sub_data.groupby('sub_id').filter(lambda x: len(np.unique(x.video_name)) > 12)
        extra_data = extra_data.drop_duplicates(subset=['sub_id', 'video_name'], keep='last')
        data = pd.concat([data, extra_data]).reset_index()

        returned_subs = list(incomplete_data.sub_id.unique())
        print(f'Number of incomplete subjects {len(returned_subs)}')

        all_sub_ids = list(data.sub_id.unique())
        print(f'Number of total complete subjects {len(all_sub_ids)}')
        return data

    def id_good_participants(self, data):
        llm = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v1')
        _, axes = plt.subplots(len(self.catch_trials), len(self.catch_trials))
        bad_subs = []
        for ax, catch_trial in zip(axes, self.catch_trials):
            catch_data = data.loc[data.video_name == catch_trial].reset_index(drop=True)    
            captions = catch_data.caption.to_list()
            embeddings = llm.encode(captions, normalize_embeddings=False)

            pairwise_dist = squareform(pdist(embeddings, metric='correlation'))
            ax[0].imshow(pairwise_dist)
            ax[0].set_title(catch_trial)
            
            pairwise_dist[pairwise_dist == 0] = np.nan
            distance = np.nanmean(pairwise_dist, axis=0)
            dist_mean = distance.mean()
            dist_std = distance.std()
            dist_threshold = dist_mean + (2.5 * dist_std)
            ax[1].hist(distance)
            ax[1].vlines(x=dist_threshold, ymin=0, ymax=10, color='red')
            ax[1].set_title(catch_trial)

            bad_subs = bad_subs + catch_data.loc[distance > dist_threshold, 'sub_id'].to_list()
        filtered_data = data[(~data['sub_id'].isin(bad_subs)) & (~data['video_name'].isin(self.catch_trials))].reset_index(drop=True)
        plt.savefig(f'{self.figures_dir}/data_quality_viz.pdf')

        print(f'Number of good participants {filtered_data.sub_id.nunique()}')
        return filtered_data

    def reorg_captions(self, filtered_data, annotations):
        caption_df = filtered_data[['video_name', 'caption']]
        caption_df['n_caption'] = caption_df.groupby('video_name').cumcount() + 1
        caption_df['n_caption'] = 'caption' + caption_df['n_caption'].astype('str').str.zfill(2)
        captions = caption_df.pivot(columns='n_caption', index='video_name', values='caption')
        captions.to_csv(f'{self.out_path}/captions.csv')

        missing_captions = captions.loc[np.invert(captions.isna().to_numpy()).sum(axis=1) < 5].reset_index().video_name.to_list()
        print(f'missing captions: {missing_captions}')

        cap_annot = annotations.merge(captions.reset_index(), on='video_name')
        caption_columns = [col for col in cap_annot.columns if col.startswith('caption')]
        cap_annot['captions'] = cap_annot[caption_columns].apply(lambda row: row.dropna().tolist(), axis=1)
        cap_annot = cap_annot.drop(columns=caption_columns)
        cap_annot.to_csv(f'{self.out_path}/stimulus_data.csv', index=False)
        
    def load_video_info(self):
        annotations = pd.read_csv(f'{self.raw_dir}/annotations/annotations.csv').drop(columns=['cooperation', 'dominance', 'intimacy'])
        test_videos = pd.read_csv(f'{self.raw_dir}/annotations/test.csv')['video_name'].to_list()

        rename_map = {col: 'rating-' + col.replace(' ', '_')  for col in annotations.columns if 'video_name' not in col}
        rename_map['transitivity'] = 'rating-object'
        annotations.rename(columns=rename_map, inplace=True)
        annotations['stimulus_set'] = 'train'
        annotations.loc[annotations.video_name.isin(test_videos), 'stimulus_set'] = 'test'
        return annotations

    def load_ratings_nc(self, annotations):
        # Load the ratings per subject
        individ_rating = pd.read_csv(f'{self.raw_dir}/annotations/individual_subject_ratings.csv')
        individ_rating = individ_rating[~individ_rating['question_name'].isin(['dominance', 'cooperation', 'relation'])]
        
        # Rename the questions
        # Manually edit some of the values so that it matches the convension in the annotations file
        rename_map = {q: 'rating-' + q.replace(' ', '_')  for q in individ_rating.question_name.unique()}
        rename_map['joint'] = 'rating-agent_distance'
        rename_map['distance'] = 'rating-joint_action'
        rename_map['communicating'] = 'rating-communication'

        individ_rating.replace(rename_map, inplace=True)
        individ_rating['rating_num'] = individ_rating.groupby(['question_name', 'video_name']).cumcount()
        individ_rating['even'] = False
        individ_rating.loc[(individ_rating.rating_num % 2) == 0, 'even'] = True

        for stimulus_set, stim_df in annotations.groupby('stimulus_set'):
            cur_df = individ_rating[individ_rating.video_name.isin(stim_df.video_name.to_list())]
            cur_df = cur_df.groupby('question_name').apply(noise_ceiling).reset_index()
            cur_df.rename(columns={0: 'nc'}, inplace=True)
            cur_df.to_csv(f'{self.out_path}/{stimulus_set}_rating_noise_ceiling.csv', index=False)

    def run(self):
        all_data = self.load_all_data()
        annotations = self.load_video_info()
        data = self.get_complete_data(all_data)
        filtered_data = self.id_good_participants(data)
        self.reorg_captions(filtered_data, annotations)
        self.load_ratings_nc(annotations)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_dir', '-top', type=str,
                        default='/scratch4/lisik3/emcmaho7/SIfMRI_modeling')
    args = parser.parse_args()
    CaptionData(args).run()

if __name__ == '__main__':
    main()