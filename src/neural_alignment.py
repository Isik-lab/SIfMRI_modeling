#
from deepjuice.alignment import TorchRidgeGCV, get_scoring_method
import torch
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
import pandas as pd
import gc 
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
from deepjuice.alignment import compute_score


def feature_scaler(train, test):
    mean_ = torch.mean(train)
    std_ = torch.std(train)
    return (train-mean_)/std_, (test-mean_)/std_


def get_benchmarking_results(benchmark, model, dataloader,
                             layer_index_offset=0,
                             devices=['cuda:0', 'cuda:1'],
                             memory_limit='30GB', 
                             n_splits=4, random_seed=0,
                             model_name=None,
                             scale_y=True, 
                             test_eval=False,
                             grouping_func='grouped_average',
                             alphas=[10.**power for power in np.arange(-5, 2)]):

    # Define a grouping function to average across the different captions or frames
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
    y = {'train': torch.from_numpy(benchmark.response_data.to_numpy().T[indices['train']]).to(torch.float32).to(devices[-1]),  
         'test': torch.from_numpy(benchmark.response_data.to_numpy().T[indices['test']]).to(torch.float32).to(devices[-1])}

    print('\n\n\nStarting CV in the training set')
    layer_index = 0 # keeps track of depth
    scores_train_max = None
    results = []
    extractor_iterator = tqdm(extractor, desc = 'Extractor Steps')
    for batched_feature_maps in extractor_iterator:
        print(batched_feature_maps)
        feature_maps = batched_feature_maps.join_batches()
        feature_map_iterator = tqdm(feature_maps.items(), desc = 'Brain Mapping (Layer)', leave=False)
        for feature_map_uid, feature_map in feature_map_iterator:
            layer_index += 1 # one layer deeper in feature_maps
            
            print('line 117')
            # reduce dimensionality of feature_maps by sparse random projection
            feature_map = get_feature_map_srps(feature_map, device=devices[-1])
            
            print('line 121')
            X = feature_map.detach().clone().squeeze().to(torch.float32).to(devices[-1])
            X = {'train': X[indices['train']], 'test': X[indices['test']]}
            del feature_map
            torch.cuda.empty_cache()

            # Memory saving
            print('line 128')
            pipe = TorchRidgeGCV(alphas=alphas, alpha_per_target=True, 
                                 device=devices[-1], scale_X=False)

            #### Fit CV in the train set ####
            y_cv_pred, y_cv_true = [], [] #Initialize lists
            for i, (cv_train_index, cv_test_index) in enumerate(cv.split(X['train'])):
                # Split the training set
                print('line 136')
                X_cv_train, X_cv_test = X['train'][cv_train_index].detach().clone(), X['train'][cv_test_index].detach().clone()
                y_cv_train, y_cv_test = y['train'][cv_train_index].detach().clone(), y['train'][cv_test_index].detach().clone()

                # Scale X and y
                X_cv_train, X_cv_test = feature_scaler(X_cv_train, X_cv_test)
                if scale_y: 
                    y_cv_train, y_cv_test = feature_scaler(y_cv_train, y_cv_test)

                # Fit the regression
                print('line 146')
                pipe.fit(X_cv_train, y_cv_train)
                print('line 148')
                y_cv_pred.append(pipe.predict(X_cv_test))
                y_cv_true.append(y_cv_test)
            print('line 151')
            scores_train = score_func(torch.cat(y_cv_pred), torch.cat(y_cv_true)) # Get the CV training scores 
            scores_train = scores_train.cpu().detach().numpy()

            print('line 155')
            if scores_train_max is None:
                scores_train_max = scores_train.copy()
                model_layer_index_max = np.ones_like(scores_train_max, dtype='int') + layer_index_offset
                model_layer_max = np.zeros_like(scores_train_max, dtype='object')
                model_layer_max.fill(feature_map_uid)
            else:
                # replace the value in the output if the previous value is less than the current value
                idx = scores_train_max < scores_train
                scores_train_max[idx] = scores_train[idx]
                model_layer_index_max[idx] = layer_index + layer_index_offset
                model_layer_max[idx] = feature_map_uid

            # Memory saving
            del pipe, scores_train
            del X, X_cv_train, X_cv_test, y_cv_pred, y_cv_true
            gc.collect()
            torch.cuda.empty_cache()
    
    # Add training data to a dataframe
    results = benchmark.metadata.copy()
    results['layer_index'] = model_layer_index_max
    results['layer_relative_depth'] = model_layer_index_max / (layer_index + layer_index_offset)
    results['layer'] = model_layer_max
    results['train_score'] = scores_train_max
    results['model_uid'] = model_name

    if test_eval:
        print('\n\n\nRunning evaluation in the test set')
        scores_test_max = torch.from_numpy(np.zeros_like(scores_train_max)).to(torch.float32).to(devices[-1])
        layer_index = 0
        extractor_iterator = tqdm(extractor, desc = 'Extractor Steps')
        for batched_feature_maps in extractor_iterator:
            print(batched_feature_maps)
            feature_maps = batched_feature_maps.join_batches()
            feature_map_iterator = tqdm(feature_maps.items(), desc = 'Testing Mapping (Layer)', leave=False)
            for feature_map_uid, feature_map in feature_map_iterator:
                layer_index += 1 # one layer deeper in feature_maps

                if np.sum(model_layer_index_max == layer_index) > 0: 
                    # reduce dimensionality of feature_maps by sparse random projection
                    feature_map = get_feature_map_srps(feature_map, device=devices[-1])
                    
                    X = feature_map.detach().clone().squeeze().to(torch.float32).to(devices[-1])
                    X = {'train': X[indices['train']], 'test': X[indices['test']]}
                    del feature_map
                    torch.cuda.empty_cache()

                    # Memory saving
                    pipe = TorchRidgeGCV(alphas=alphas, alpha_per_target=True, 
                                        device=devices[-1], scale_X=False)

                    X_train, X_test = feature_scaler(X['train'].detach().clone(), X['test'].detach().clone())
                    if scale_y: 
                        y_train, y_test = feature_scaler(y['train'].detach().clone(), y['test'].detach().clone())
                    else:
                        y_train, y_test = y['train'].detach().clone(), y['test'].detach().clone()

                    pipe.fit(X_train, y['train'])
                    y_hat = pipe.predict(X_test)
                    scores_test = score_func(y_hat, y['test'])

                    #Save the test set scores to an array only if it is where performance was maximum in the training set 
                    idx = model_layer_index_max == (layer_index + layer_index_offset)
                    scores_test_max[idx] = scores_test[idx]

                    # Memory saving
                    del pipe, scores_test
                    del X, X_train, X_test, y_train
                    gc.collect()
                    torch.cuda.empty_cache()
                else: 
                    print(f'{feature_map_uid} (layer {layer_index}) is not a max layer in train set')
                    print('skipping test set regression')

        # Add test set results to the dataframe
        results['test_score'] = scores_test_max.cpu().detach().numpy()
        
        # Run permutation testing and bootstapping
        results['r_null_dist'] = np.nan
        results['r_var_dist'] = np.nan

        roi_indices = benchmark.metadata.index[benchmark.metadata.roi_name != 'none'].to_numpy()
        print(type(roi_indices))
        print(f'{y_test.shape=}')
        print(f'{y_hat.shape=}')
        r_null = stats.perm_gpu(y_test[:, roi_indices], y_hat[:, roi_indices], verbose=True)
        r_var = stats.bootstrap_gpu(y_test[:, roi_indices], y_hat[:, roi_indices], verbose=True)
        results.iloc[roi_indices]['r_null_dist'] = r_null.cpu().detach().numpy().T.tolist()
        results.iloc[roi_indices]['r_var_dist'] = r_var.cpu().detach().numpy().T.tolist()
    print(results.head(20))
    return results
