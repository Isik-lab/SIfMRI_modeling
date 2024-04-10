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


class Benchmark:
    def __init__(self, metadata=None, stimulus_data=None, response_data=None):
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
        self.train_response_data = None
        self.test_response_data = None
        self.train_rdm_stimulus = None
        self.test_rdm_stimulus = None
        self.train_rdms = None
        self.test_rdms = None
        self.train_rdm_indices = None
        self.test_rdm_indices = None
        self.train_row_indices = None
        self.test_row_indices = None



    def add_stimulus_path(self, data_dir, extension='png'):
        if extension != 'mp4': 
            self.stimulus_data['stimulus_path'] = data_dir + self.stimulus_data.video_name.str.replace('mp4', 'png')
        else:
            self.stimulus_data['stimulus_path'] = data_dir + self.stimulus_data.video_name

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
        if self.response_data is not None:
            self.response_data = self.response_data[stim_idx]

    def sort_stimulus_values(self, col='stimulus_set'):
        self.stimulus_data = self.stimulus_data.sort_values(by=col).reset_index()
        stim_idx = list(self.stimulus_data['index'].to_numpy().astype('str'))
        self.stimulus_data.drop(columns='index', inplace=True)
        if self.response_data is not None: 
            self.response_data = self.response_data[stim_idx].reset_index(drop=True)


    def generate_rdms(self):
        """
        Generates representational dissimilarity matrices (RDMs) for a set of subjects and regions
        of interest (ROIs) based on brain imaging response data and associated metadata.

        This function iterates over each ROI in the metadata, applies the ROI mask to the response data,
        computes pairwise dissimilarities (using Pearson correlation) among the responses for different
        stimuli, and organizes the results into a structured format suitable for further analysis.

        Initializes:
            self.rdms (dict): A dictionary where keys are ROI names and values are dictionaries
                mapping subject IDs to their respective RDM numpy arrays.
            self.rdm_indices (dict): A dictionary mapping each ROI to another dictionary,
                which maps subject IDs to the indices of the RDMs based on voxel IDs.
            self.row_indices (dict): A dictionary mapping each ROI to another dictionary,
                which maps subject IDs to the indices in the response data corresponding to each ROI,
                 facilitating direct access to ROI-specific response data.
        """
        subjects = [1, 2, 3, 4]
        # Build Brain RDM's
        custom_train_rdms = {}
        custom_test_rdms = {}
        custom_train_rdm_indices = {}
        custom_test_rdm_indices = {}
        custom_train_row_indices = {}
        custom_test_row_indices = {}

        # train
        self.train_rdm_stimulus = self.stimulus_data[self.stimulus_data['stimulus_set'] == 'train'].reset_index()
        stim_idx = list(self.train_rdm_stimulus['index'].to_numpy().astype('str'))
        self.train_rdm_stimulus.drop(columns='index', inplace=True)
        if self.response_data is not None:
            self.train_response_data = self.response_data[stim_idx]
            self.train_response_data.index.name = 'voxel_id'

        for roi in self.metadata['roi_name'].unique()[1:]:
            sub_dict = {}
            custom_train_rdm_indices[roi] = {}
            for sub in subjects:
                # Applying ROI to Whole brain betas
                betas = self.train_response_data.loc[
                    self.metadata[(self.metadata['roi_name'] == roi) & (self.metadata['subj_id'] == sub)].index]
                # Correlating pairwise across 200 videos
                df_beta = pd.DataFrame(betas)
                df_pearson = 1 - df_beta.corr(method='pearson')
                sub_rdm = df_pearson.to_numpy()
                sub_dict[sub] = sub_rdm
                sub_rdm = (convert_to_tensor(sub_rdm).to(torch.float64).to('cpu'))
                sub_dict[sub] = sub_rdm

                # Populate the Indices
                custom_train_rdm_indices[roi][sub] = self.metadata.loc[
                    (self.metadata['roi_name'] == roi) & (self.metadata['subj_id'] == sub)].voxel_id.to_list()

            # Populate the row Indices
            custom_train_row_indices = custom_train_rdm_indices  # these are a direct match as the index in response data is numeric starting at 0, old code used - (response_data.reset_index().query('voxel_id==@roi_index').index.to_numpy())
            custom_train_rdms[roi] = sub_dict
        self.train_rdms = custom_train_rdms
        self.train_rdm_indices = custom_train_rdm_indices
        self.train_row_indices = custom_train_row_indices

        # test
        self.test_rdm_stimulus = self.stimulus_data[self.stimulus_data['stimulus_set'] == 'test'].reset_index()
        stim_idx = list(self.test_rdm_stimulus['index'].to_numpy().astype('str'))
        self.test_rdm_stimulus.drop(columns='index', inplace=True)
        if self.response_data is not None:
            self.test_response_data = self.response_data[stim_idx]
            self.test_response_data.index.name = 'voxel_id'

        for roi in self.metadata['roi_name'].unique()[1:]:
            sub_dict = {}
            custom_test_rdm_indices[roi] = {}
            for sub in subjects:
                # Applying ROI to Whole brain betas
                betas = self.test_response_data.loc[
                    self.metadata[(self.metadata['roi_name'] == roi) & (self.metadata['subj_id'] == sub)].index]
                # Correlating pairwise across 200 videos
                df_beta = pd.DataFrame(betas)
                df_pearson = 1 - df_beta.corr(method='pearson')
                sub_rdm = df_pearson.to_numpy()
                sub_dict[sub] = sub_rdm
                sub_rdm = (convert_to_tensor(sub_rdm).to(torch.float64).to('cpu'))
                sub_dict[sub] = sub_rdm

                # Populate the Indices
                custom_test_rdm_indices[roi][sub] = self.metadata.loc[
                    (self.metadata['roi_name'] == roi) & (self.metadata['subj_id'] == sub)].voxel_id.to_list()

            # Populate the row Indices
            custom_test_row_indices = custom_test_rdm_indices  # these are a direct match as the index in response data is numeric starting at 0, old code used - (response_data.reset_index().query('voxel_id==@roi_index').index.to_numpy())
            custom_test_rdms[roi] = sub_dict
        self.test_rdms = custom_test_rdms
        self.test_rdm_indices = custom_test_rdm_indices
        self.test_row_indices = custom_test_row_indices

    def update(self, iterable):
        """
            iterable: dict
        """
        for key, value in iterable.items():
            setattr(self, key, value)
