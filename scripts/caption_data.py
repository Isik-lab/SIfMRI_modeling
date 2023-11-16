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


class CaptionData:
    def __init__(self, args):
        # save arg inputs into self
        self.top_dir = args.top_dir

        # Set up the directories
        self.process = 'CaptionData'
        self.figures_dir = f'{self.top_dir}/reports/figures'
        self.interim_dir = f'{self.top_dir}/data/interim'
        self.raw_dir = f'{self.top_dir}/data/raw'
        self.out_path = f'{self.interim_dir}/{self.process}'
        Path(self.out_path).mkdir(exist_ok=True, parents=True)

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
                print(f"sub_id: {sub_id}, condition: {condition}, date: {date}")

            if os.path.getsize(path) == 0:
                print('oops that file is empty. moving on...')
            else:
                df = pd.read_csv(path, header=None)
                df.columns = ['url', 'caption']
                df['url'] = df['url'].str.extract("(https://[^']+)")
                df['video_name'] = df['url'].str.extract(r'/([^/]+\.mp4)')[0]
                df[['sub_id', 'condition', 'date']] = sub_id, condition, date
                all_data.append(df)
        return pd.concat(all_data)
    
    def get_complete_data(self, all_data):
        data = all_data.groupby('sub_id').filter(lambda x: len(x) == 12)
        extra_data = all_data.groupby('sub_id').filter(lambda x: len(x) > 12)
        print(extra_data.drop_duplicates(subset=['sub_id']).groupby(['condition']).count())
        print(data.drop_duplicates(subset=['sub_id']).groupby(['condition']).count())
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
        print(bad_subs)
        filtered_data = data[(~data['sub_id'].isin(bad_subs)) & (~data['video_name'].isin(self.catch_trials))].reset_index(drop=True)
        print(filtered_data.drop_duplicates(subset=['sub_id']).groupby(['condition']).count())
        plt.savefig(f'{self.figures_dir}/data_quality_viz.pdf')
        filtered_data.to_csv(f'{self.out_path}/filtered_data.csv', index=False)
        return filtered_data

    def reorg_captions(self, filtered_data):
        caption_df = filtered_data[['video_name', 'caption']]
        caption_df['n_caption'] = caption_df.groupby('video_name').cumcount() + 1
        caption_df['n_caption'] = 'caption' + caption_df['n_caption'].astype('str').str.zfill(2)
        captions = caption_df.pivot(columns='n_caption', index='video_name', values='caption')
        captions.to_csv(f'{self.out_path}/captions.csv', index=False)

        annotations = pd.read_csv(f'{self.raw_dir}/annotations/annotations.csv').drop(columns=['cooperation', 'dominance', 'intimacy'])
        annotations = annotations.merge(pd.read_csv(f'{self.raw_dir}/annotations/train.csv'), on='video_name').reset_index(drop=True)
        cap_annot = annotations.merge(captions.reset_index()[['video_name', 'caption01']], on='video_name')
        cap_annot.to_csv(f'{self.out_path}/captions_and_annotations.csv', index=False)

    def run(self):
        all_data = self.load_all_data()
        data = self.get_complete_data(all_data)
        filtered_data = self.id_good_participants(data)
        self.reorg_captions(filtered_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_dir', '-top', type=str,
                        default='/scratch4/lisik3/emcmaho7/SIfMRI_modeling')
    args = parser.parse_args()
    CaptionData(args).run()

if __name__ == '__main__':
    main()