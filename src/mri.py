import nibabel as nib
import numpy as np


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
        self.metadata = metadata
        self.stimulus_data = stimulus_data
        self.response_data = response_data

    def filter_rois(self, rois='none'):
        if rois != 'none':
            self.metadata = self.metadata.loc[self.metadata.roi_name.isin(rois)].reset_index()
            voxel_id = self.metadata['voxel_id'].to_numpy()
            self.response_data = self.response_data.iloc[voxel_id]
        else:
            self.metadata = self.metadata.loc[self.metadata.roi_name != 'none'].reset_index()
            voxel_id = self.metadata['voxel_id'].to_numpy()
            self.response_data = self.response_data.iloc[voxel_id]

    def filter_streams(self, streams='none'):
        if streams != 'none':
            self.metadata = self.metadata.loc[self.metadata.stream_name.isin(streams)].reset_index()
            voxel_id = self.metadata['voxel_id'].to_numpy()
            self.response_data = self.response_data.iloc[voxel_id]
        else:
            self.metadata = self.metadata.loc[self.metadata.stream_name != 'none'].reset_index()
            voxel_id = self.metadata['voxel_id'].to_numpy()
            self.response_data = self.response_data.iloc[voxel_id]
    
    def filter_subjids(self, subj_ids):
        self.metadata = self.metadata.loc[self.metadata.subj_id.isin(subj_ids)].reset_index()
        voxel_id = self.metadata['voxel_id'].to_numpy()
        self.response_data = self.response_data.iloc[voxel_id]

    def filter_stimulus(self, stimulus_set='train'):
        stim_data = self.stimulus_data[self.stimulus_data['stimulus_set'] == stimulus_set]
        stim_data.reset_index

