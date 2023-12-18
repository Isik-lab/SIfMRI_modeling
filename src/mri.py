import nibabel as nib
import numpy as np
import pandas as pd


def gen_mask(files, rel_mask=None):
    #Combine the two hemispheres
    roi = []
    for f in files:
        roi_hemi = nib.load(f).get_fdata().astype('bool')
        roi.append(roi_hemi)
    roi_mask = np.sum(roi, axis=0)
    #add the rel_mask if defined
    if rel_mask is not None:
        roi_mask += rel_mask
        return roi_mask > 1 #Equivalent to roi_mask and rel_mask
    else:
        return roi_mask.astype('bool')


class Benchmark:
    def __init__(self, metadata, stimulus_data, response_data):
        if type(metadata) is str:
            self.metadata = pd.read_csv(metadata)
        else:
            self.metadata = metadata

        if type(stimulus_data) is str:
            self.stimulus_data = pd.read_csv(stimulus_data)
        else:
            self.stimulus_data = stimulus_data

        if type(response_data) is str:
            self.response_data = pd.read_csv(response_data)
        else:
            self.response_data = response_data

    def add_image_path(self, data_dir):
        self.stimulus_data['image_path'] = data_dir + self.stimulus_data.video_name.str.replace('mp4', 'png')
        print(self.stimulus_data.head())

    def filter_rois(self, rois='none'):
        if rois != 'none':
            self.metadata = self.metadata.loc[self.metadata.roi_name.isin(rois)].reset_index()
            voxel_id = self.metadata['voxel_id'].to_numpy()
            self.metadata.drop(columns='index', inplace=True)
            self.response_data = self.response_data.iloc[voxel_id]
        else:
            self.metadata = self.metadata.loc[self.metadata.roi_name != 'none'].reset_index()
            voxel_id = self.metadata['voxel_id'].to_numpy()
            self.metadata.drop(columns='index', inplace=True)
            self.response_data = self.response_data.iloc[voxel_id]

    def filter_streams(self, streams='none'):
        if streams != 'none':
            self.metadata = self.metadata.loc[self.metadata.stream_name.isin(streams)].reset_index()
            voxel_id = self.metadata['voxel_id'].to_numpy()
            self.metadata.drop(columns='index', inplace=True)            
            self.response_data = self.response_data.iloc[voxel_id]
        else:
            self.metadata = self.metadata.loc[self.metadata.stream_name != 'none'].reset_index()
            voxel_id = self.metadata['voxel_id'].to_numpy()
            self.metadata.drop(columns='index', inplace=True)
            self.response_data = self.response_data.iloc[voxel_id]
    
    def filter_subjids(self, subj_ids):
        self.metadata = self.metadata.loc[self.metadata.subj_id.isin(subj_ids)].reset_index()
        voxel_id = self.metadata['voxel_id'].to_numpy()
        self.stimulus_data.drop(columns='index', inplace=True)
        self.response_data = self.response_data.iloc[voxel_id]

    def filter_stimulus(self, stimulus_set='train'):
        self.stimulus_data = self.stimulus_data[self.stimulus_data['stimulus_set'] == stimulus_set].reset_index()
        stim_idx = list(self.stimulus_data['index'].to_numpy().astype('str'))
        self.stimulus_data.drop(columns='index', inplace=True)
        self.response_data = self.response_data[stim_idx]

