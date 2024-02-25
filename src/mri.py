import nibabel as nib
import numpy as np
import pandas as pd
from deepjuice.tensorops import convert_to_tensor
import torch


def gen_mask(files, rel_mask=None):
    #Combine the two hemispheres
    roi = []
    for f in files:
        roi_hemi = nib.load(f).get_fdata().astype('bool')
        if rel_mask is not None: 
             #add the rel_mask if defined
            roi.append(np.logical_and(roi_hemi, rel_mask))
        else:
            roi.append(roi_hemi)
    return np.logical_or.reduce(roi)


def generate_rdms(response_data, metadata, video_set='train'):
    """
    Generates representational dissimilarity matrices (RDMs) for a set of subjects and regions
    of interest (ROIs) based on brain imaging response data and associated metadata.

    This function iterates over each ROI in the metadata, applies the ROI mask to the response data,
    computes pairwise dissimilarities (using Pearson correlation) among the responses for different
    stimuli, and organizes the results into a structured format suitable for further analysis.

    Parameters:
        response_data (pandas.DataFrame): A DataFrame containing the brain imaging response data,
            with rows corresponding to stimuli and columns to neural measurements (e.g., voxel activations).
        metadata (pandas.DataFrame): A DataFrame containing metadata about the response data,
            including information on ROIs and subject IDs.
        video_set (str): Defaults to 'train'. Depending on if train or test is passed it will subset either the
            200 training video responses or 50 test video responses. If anything else is passed all 250 video
            responses will be used.

    Returns:
        custom_rdms (dict): A dictionary where keys are ROI names and values are dictionaries
            mapping subject IDs to their respective RDM numpy arrays.
        custom_rdm_indices (dict): A dictionary mapping each ROI to another dictionary,
            which maps subject IDs to the indices of the RDMs based on voxel IDs.
        custom_row_indices (dict): A dictionary mapping each ROI to another dictionary,
            which maps subject IDs to the indices in the response data corresponding to each ROI,
             facilitating direct access to ROI-specific response data.
    """
    subjects = [1, 2, 3, 4]
    # Build Brain RDM's
    custom_rdms = {}
    custom_rdm_indices = {}
    custom_roi_indices = {}
    response_data.index.name = 'voxel_id'
    for roi in metadata['roi_name'].unique()[1:]:
        sub_dict = {}
        custom_rdm_indices[roi] = {}
        for sub in subjects:
            # Applying ROI to Whole brain betas
            betas = response_data.loc[
                metadata[(metadata['roi_name'] == roi) & (metadata['subj_id'] == sub)].index]
            # if training videos only, take first 200 columns
            if video_set == 'train':
                betas = betas[betas.columns[:200]]
            elif video_set == 'test':
                betas = betas[betas.columns[200:]]

            # Correlating pairwise across 200 videos
            df_beta = pd.DataFrame(betas)
            df_pearson = 1 - df_beta.corr(method='pearson')
            sub_rdm = df_pearson.to_numpy()
            sub_dict[sub] = sub_rdm
            sub_rdm = (convert_to_tensor(sub_rdm).to(torch.float64).to('cpu'))
            sub_dict[sub] = sub_rdm

            # Populate the Indices
            custom_rdm_indices[roi][sub] = metadata.loc[
                (metadata['roi_name'] == roi) & (metadata['subj_id'] == sub)].voxel_id.to_list()

        # Populate the row Indices
        custom_row_indices = custom_rdm_indices # these are a direct match as the index in response data is numeric starting at 0, old code used - (response_data.reset_index().query('voxel_id==@roi_index').index.to_numpy())

        custom_rdms[roi] = sub_dict
    return custom_rdms, custom_rdm_indices, custom_row_indices


class Benchmark:
    def __init__(self, metadata, stimulus_data, response_data, rdms=False, video_set='train'):
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

        if rdms:
            self.rdms, self.rdm_indices, self.row_indices = generate_rdms(response_data, metadata, video_set)
            self.video_set = video_set

    def add_stimulus_path(self, data_dir, extension='png'):
        if extension != 'mp4': 
            self.stimulus_data['stimulus_path'] = data_dir + self.stimulus_data.video_name.str.replace('mp4', 'png')
        else:
            self.stimulus_data['stimulus_path'] = data_dir + self.stimulus_data.video_name
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

    def update(self, iterable):
        """
            iterable: dict
        """
        for key, value in iterable.items():
            setattr(self, key, value)
