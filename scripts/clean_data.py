#
import re
import argparse
from glob import glob
import os
from pathlib import Path
from tqdm import tqdm

import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns

from sentence_transformers import SentenceTransformer


def download_data(local, remote, sub_dir):
    import dropbox, getpass
    personal_access_token = getpass.getpass('Enter your Personal Access Token: ')
    dbx = dropbox.Dropbox(personal_access_token)

    list_folder_result = dbx.files_list_folder(path=f'{remote}/{sub_dir}')
    for entry in tqdm(list_folder_result.entries, total=len(list_folder_result.entries)):
        file = entry.path_lower.split('/')[-1]

        Path(f'{local}/videos/').mkdir(exist_ok=True, parents=True)
        dbx.files_download_to_file(f'{local}/{sub_dir}/{file}', entry.path_lower)


class CaptionData:
    def __init__(self, args):
        # save arg inputs into self
        self.local_path = args.local_path
        self.remote_path = args.remote_path
        self.download_videos = args.download_videos
        self.download_captions = args.download_captions
        self.download_annotations = args.download_annotations

        # Set up the directories
        self.figures_dir = f'{self.local_path}/reports/figures'
        self.interim_dir = f'{self.local_path}/data/interim'
        self.raw_dir = f'{self.local_path}/data/raw'
        self.out_path = f'{self.interim_dir}/CaptionData'
        Path(self.out_path).mkdir(exist_ok=True, parents=True)
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
        incomplete_data = all_data.groupby('sub_id').filter(lambda x: len(x) < 12)
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
        if ~os.path.exists(f'{self.raw_dir}/annotations') or self.download_annotations:
            download_data(self.raw_dir, self.remote_path, 'annotations')
        if ~os.path.exists(f'{self.raw_dir}/captions') or self.download_captions:
            download_data(self.raw_dir, self.remote_path, 'captions')
        if ~os.path.exists(f'{self.raw_dir}/videos') or self.download_videos:
            download_data(self.raw_dir, self.remote_path, 'videos')

        all_data = self.load_all_data()
        data = self.get_complete_data(all_data)
        filtered_data = self.id_good_participants(data)
        self.reorg_captions(filtered_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download_annotations', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--download_videos', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--download_captions', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--local_path', '-local', type=str,
                        default='/scratch4/lisik3/emcmaho7/SIfMRI_modeling')
    parser.add_argument('--remote_path', '-remote', type=str,
                    default='/projects/SI_fmri/SIfMRI_modeling/data/raw')
    args = parser.parse_args()
    CaptionData(args).run()

if __name__ == '__main__':
    main()