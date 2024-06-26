#/Applications/anaconda3/envs/nibabel/bin/python
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import nibabel as nib
from tqdm import tqdm
from src.mri import gen_mask
from glob import glob


class ReorganziefMRI:
    def __init__(self, args):
        self.process = 'ReorganziefMRI'
        self.data_dir = args.data_dir
        Path(f'{self.data_dir}/interim/{self.process}').mkdir(parents=True, exist_ok=True)
        print(vars(self))
        self.rois = ['EVC', 'MT', 'EBA', 'LOC', 'FFA',
                     'PPA', 'pSTS', 'face-pSTS', 'aSTS']
        self.streams = ['evc']
        self.streams += [f'{level}_{stream}' for level in ['mid', 'high'] for stream in ['ventral', 'lateral', 'parietal']]

    def generate_benchmark(self, sort_idx):
        all_rois = []
        all_betas = []
        for sub in tqdm(range(4)):
            sub = str(sub+1).zfill(2)
            reliability_mask = nib.load(f'{self.data_dir}/raw/reliability_mask/sub-{sub}_space-T1w_desc-test-fracridge_reliability-mask.nii.gz').get_fdata().astype('bool')
            reliability = nib.load(f'{self.data_dir}/raw/reliability_mask/sub-{sub}_space-T1w_desc-test-fracridge_stat-r_statmap.nii.gz').get_fdata()

            # Beta files
            betas_arr = []
            for dataset in ['train', 'test']:
                betas_file = f'{self.data_dir}/raw/fmri_betas/sub-{sub}_space-T1w_desc-{dataset}-fracridge_data.nii.gz'
                betas_arr.append(nib.load(betas_file).get_fdata())
            betas_arr = np.concatenate(betas_arr, axis=-1)
            betas_arr = betas_arr[..., sort_idx] #resort by the stimulus_data frame
            all_betas.append(betas_arr[reliability_mask].reshape((-1, betas_arr.shape[-1]))) # Only save the reliable voxels

            # voxel metadata arrays in the same shape as the brain
            roi_labels = np.zeros_like(reliability_mask, dtype='object')
            stream_labels = np.zeros_like(reliability_mask, dtype='object')

            # Add the roi labels
            for roi in self.rois:
                roi_mask = []
                for hemi in ['lh', 'rh']:
                    hemi_mask_file = glob(f'{self.data_dir}/raw/localizers/sub-{sub}/sub-{sub}*roi-{roi}*hemi-{hemi}*mask.nii.gz')[0]
                    hemi_mask = nib.load(hemi_mask_file).get_fdata().astype('bool')
                    hemi_reliability_mask = np.logical_and(reliability_mask, hemi_mask)
                    roi_labels[hemi_reliability_mask] = roi
                    roi_mask.append(hemi_reliability_mask[..., np.newaxis])
                print(f'{roi}: ', np.mean(reliability[np.any(np.concatenate(roi_mask, axis=-1), axis=-1)]**2))

            # Add the stream labels
            for stream in self.streams:
                stream_mask_file = glob(f'{self.data_dir}/raw/streams/sub-{sub}/sub-{sub}*roi-{stream}*mask.nii.gz')[0]
                stream_mask = nib.load(stream_mask_file).get_fdata().astype('bool')
                stream_reliability_mask = np.logical_and(reliability_mask, stream_mask)
                stream_labels[stream_reliability_mask] = stream

            # Add the subject's voxel metadata to list
            voxel_indices = np.array(np.where(reliability_mask)).T
            for (roi, stream), (r, index) in zip(zip(roi_labels[reliability_mask], stream_labels[reliability_mask]),
                                                 zip(reliability[reliability_mask], voxel_indices)):
                all_rois.append({'roi_name': roi, 'stream_name': stream, 'subj_id': sub, 'reliability': r,
                                 'i_index': index[0], 'j_index': index[1], 'k_index': index[-1]})

        # metadata
        metadata = pd.DataFrame(all_rois)
        metadata.loc[metadata.stream_name == 0, 'stream_name'] = 'none'
        metadata.loc[metadata.roi_name == 0, 'roi_name'] = 'none'
        # this makes a unique voxel_id for every voxel across all subjects
        metadata = metadata.reset_index().rename(columns={'index': 'voxel_id'})
        print(metadata.roi_name.unique())
        print(metadata.stream_name.unique())

        # response data
        response_data = []
        for i in range(4):
            response_data.append(pd.DataFrame(all_betas[i]))
        response_data = pd.concat(response_data, ignore_index=True)

        # Make the voxel ids unique so that there are no repeats across subjects
        return metadata, response_data
    
    def load_stimulus_data(self):
        stim_data = pd.read_csv(f'{self.data_dir}/interim/CaptionData/stimulus_data.csv')

        # get the sorting indices to combine the train and test fMRI data
        temp = stim_data[['video_name', 'stimulus_set']]
        temp['sort_stimulus_set'] = temp['stimulus_set'] == 'test'
        temp = temp.sort_values(by=['sort_stimulus_set', 'video_name']).reset_index(drop=True)
        return stim_data, temp.sort_values(by='video_name').reset_index()['index'].to_numpy()

    def run(self):
        stimulus_data, sort_idx = self.load_stimulus_data()
        metadata, response_data = self.generate_benchmark(sort_idx)
        # stimulus_data.to_csv(f'{self.data_dir}/interim/{self.process}/stimulus_data.csv', index=False)
        # metadata.to_csv(f'{self.data_dir}/interim/{self.process}/metadata.csv', index=False)
        # response_data.to_csv(f'{self.data_dir}/interim/{self.process}/response_data.csv.gz', index=False, compression='gzip')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-data', type=str,
                         default='/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_modeling/data')
    args = parser.parse_args()
    ReorganziefMRI(args).run()


if __name__ == '__main__':
    main()
