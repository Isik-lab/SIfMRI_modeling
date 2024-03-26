#
from deepjuice.alignment import TorchRidgeGCV, get_scoring_method, compute_rdm, compare_rdms
import torch
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
import pandas as pd
import gc
from itertools import product
from src import stats
from deepjuice.extraction import FeatureExtractor
from deepjuice.reduction import get_feature_map_srps
from deepjuice.systemops.devices import cuda_device_report
from deepjuice.procedural import pandas_query
from deepjuice.model_zoo.options import get_deepjuice_model
from deepjuice.procedural.datasets import get_data_loader
from deepjuice.extraction import FeatureExtractor
from deepjuice.procedural.cv_ops import CVIndexer
from deepjuice.alignment import TorchRidgeGCV
from deepjuice.reduction import compute_srp


def feature_scaler(train, test):
    mean_ = torch.mean(train)
    std_ = torch.std(train)
    return (train-mean_)/std_, (test-mean_)/std_


def get_benchmarking_results(benchmark, model, dataloader,
                                 target_features,
                                 layer_index_offset=0,
                                 devices=['cuda:0', 'cuda:1'],
                                 n_splits=4, random_seed=0,
                                 model_name=None,
                                 scale_y=True, memory_limit='30GB',
                                 grouping_func='grouped_average',
                                 alphas=[10.**power for power in np.arange(-5, 2)]):

    # Define a grouping function to average across the different captions
    def grouped_average(tensor, batch_iter=None, **kwargs):
        if batch_iter is None: return tensor # as is

        sub_data = dataloader.batch_data.query('batch_iter==@batch_iter')

        tensor_means = [] # fill with group tensor means
        for group in sub_data.group_index.unique():
            group_data = sub_data.query('group_index==@group')
            group_idx = group_data.batch_index.to_list()

            # convert index to tensor on device
            group_idx = (torch.LongTensor(group_idx)
                        .to(tensor.device))

            tensor_mean = tensor[group_idx].mean(dim=0)
            tensor_means += [tensor_mean.unsqueeze(0)]

        return torch.concat(tensor_means, dim=0) # as is for testing

        # Define a stacking function to concatenate across the different captions or frames
    def grouped_stack(tensor, batch_iter=None, **kwargs):
        if batch_iter is None: return tensor  # Return as is if no batch iteration is provided

        sub_data = dataloader.batch_data.query('batch_iter==@batch_iter')

        grouped_tensors = []  # To fill with concatenated tensors for each group
        for group in sub_data.group_index.unique():
            group_data = sub_data.query('group_index==@group')
            group_idx = group_data.batch_index.to_list()

            # Convert index to tensor on device
            group_idx = (torch.LongTensor(group_idx)
                        .to(tensor.device))

            # Concatenate tensors within the group along a new dimension (e.g., dim=1)
            group_tensor = torch.cat([tensor[idx].unsqueeze(0) for idx in group_idx], dim=0)

            # Add the concatenated group tensor to the list
            grouped_tensors.append(group_tensor.unsqueeze(0))

        # Concatenate all group tensors along dim=0
        return torch.cat(grouped_tensors, dim=0)

    # use a CUDA-capable device, if available, else: CPU
    print(cuda_device_report())
    print(f'Searching {len(alphas)} alpha values')

    # define the feature extractor object
    tensor_fn = globals().get(grouping_func)
    extractor = FeatureExtractor(model, dataloader, **{'device': devices[0], 'output_device': devices[0]},
                                tensor_fn=tensor_fn,
                                memory_limit=memory_limit,
                                batch_strategy='stack')
    extractor.modify_settings(flatten=True)

    # initialize pipe and kfold splitter
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    score_func = get_scoring_method('pearsonr')

    # divide responses
    indices = {'train': benchmark.stimulus_data[benchmark.stimulus_data.stimulus_set == 'train'].index,
            'test': benchmark.stimulus_data[benchmark.stimulus_data.stimulus_set == 'test'].index}
    y = {'train': torch.from_numpy(benchmark.stimulus_data[target_features].to_numpy()[indices['train']]).to(torch.float32).to(devices[-1]),  
         'test': torch.from_numpy(benchmark.stimulus_data[target_features].to_numpy()[indices['test']]).to(torch.float32).to(devices[-1])}

    layer_index = 0 # keeps track of depth
    results = []
    extractor_iterator = tqdm(extractor, desc = 'Extractor Steps')
    for batched_feature_maps in extractor_iterator:
        print(batched_feature_maps)
        feature_maps = batched_feature_maps.join_batches()
        feature_map_iterator = tqdm(feature_maps.items(), desc = 'Brain Mapping (Layer)', leave=False)
        for feature_map_uid, feature_map in feature_map_iterator:
            layer_index += 1 # one layer deeper in feature_maps

            # reduce dimensionality of feature_maps by sparse random projection
            feature_map = get_feature_map_srps(feature_map, device=devices[-1])
            feature_map_info = {'model_uid': model_name,
                                'model_layer': feature_map_uid,
                                'model_layer_index': layer_index + layer_index_offset}
            
            X = feature_map.detach().clone().squeeze().to(torch.float32).to(devices[-1])
            X = {'train': X[indices['train']], 'test': X[indices['test']]}
            del feature_map
            torch.cuda.empty_cache()

            # Memory saving
            pipe = TorchRidgeGCV(alphas=alphas, alpha_per_target=True, 
                                 device=devices[-1], scale_X=False)

            #### Fit CV in the train set ####
            y_cv_pred, y_cv_true = [], [] #Initialize lists
            for i, (cv_train_index, cv_test_index) in enumerate(cv.split(X['train'])):
                # Split the training set
                X_cv_train, X_cv_test = X['train'][cv_train_index].detach().clone(), X['train'][cv_test_index].detach().clone()
                y_cv_train, y_cv_test = y['train'][cv_train_index].detach().clone(), y['train'][cv_test_index].detach().clone()

                # Scale X and y
                X_cv_train, X_cv_test = feature_scaler(X_cv_train, X_cv_test)
                if scale_y: 
                    y_cv_train, y_cv_test = feature_scaler(y_cv_train, y_cv_test)

                # Fit the regression
                pipe.fit(X_cv_train, y_cv_train)
                y_cv_pred.append(pipe.predict(X_cv_test))
                y_cv_true.append(y_cv_test)
            scores_train = score_func(torch.cat(y_cv_pred), torch.cat(y_cv_true)) # Get the CV training scores 

            #### Fit the train/test split ####
            # Scale X and y
            X_train, X_test = feature_scaler(X['train'].detach().clone(), X['test'].detach().clone())
            y_train, y_test = y['train'].detach().clone(), y['test'].detach().clone()
            if scale_y: 
                y_train, y_test = feature_scaler(y_train, y_test)

            # Fit the regression
            pipe.fit(X_train, y_train)
            y_hat = pipe.predict(X_test)
            scores_test = score_func(y_hat, y_test)

            r_null = stats.perm_gpu(y_test, y_hat)
            r_var = stats.bootstrap_gpu(y_test, y_hat)
            # Add performance to the score sheet
            for target_feature, score_train, score_test, null, var in zip(target_features,
                                                                          scores_train.cpu().detach().numpy(),
                                                                          scores_test.cpu().detach().numpy(),
                                                                          r_null.cpu().detach().numpy().T,
                                                                          r_var.cpu().detach().numpy().T): 
                # add the scores to a "scoresheet"
                results.append({**feature_map_info,
                                'feature': target_feature,
                                'train_score': score_train,
                                'test_score': score_test,
                                'r_null_dist': null, 'r_var_dist': var})

            # Memory saving
            del pipe, scores_train, scores_test
            del X_cv_train, X_cv_test, y_cv_pred, y_cv_true
            del X_train, X_test, y_train, y_test
            del r_null, r_var
            gc.collect()
            torch.cuda.empty_cache()
    return pd.DataFrame(results)