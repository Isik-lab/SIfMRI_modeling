#
from deepjuice.alignment import TorchRidgeGCV, get_scoring_method
import torch
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
import pandas as pd
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


def moving_grouped_average(outputs, skip=5, input_dim=0):
    from math import ceil as roundup # for rounding upwards
    return torch.stack([outputs[i*skip:i*skip+skip].mean(dim=input_dim) 
                        for i in range(roundup(outputs.shape[input_dim] / skip))])


def get_nearest_multiple(a, b):
    # Find the nearest multiple of b to a
    nearest_multiple = round(a / b) * b
    if nearest_multiple % 2 != 0:
        if (nearest_multiple - a) < (a - (nearest_multiple - b)):
            nearest_multiple += b
        else:
            nearest_multiple -= b
            
    return nearest_multiple # integer space


def get_training_benchmarking_results(benchmark, feature_extractor,
                                      file_path,
                                      layer_index_offset=0,
                                      device='cuda',
                                      n_splits=4, random_seed=0,
                                      alphas=[10.**power for power in np.arange(-5, 2)]):

    # use a CUDA-capable device, if available, else: CPU
    print(f'device: {device}')
    print(cuda_device_report())

    # initialize pipe and kfold splitter
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    alphas = [10.**power for power in np.arange(-5, 2)]
    score_func = get_scoring_method('pearsonr')
    pipe = TorchRidgeGCV(alphas=alphas, alpha_per_target=True,
                            device=device, scale_X=True,)

    # Send the neural data to the GPU
    y = torch.from_numpy(benchmark.response_data.to_numpy().T).to(torch.float32).to(device)

    layer_index = 0 # keeps track of depth
    scores_out = None
    for feature_maps in feature_extractor:
        feature_map_iterator = tqdm(feature_maps.items(), desc = 'Brain Mapping (Layer)', leave=False)
        for feature_map_uid, feature_map in feature_map_iterator:
            layer_index += 1 # one layer deeper in feature_maps

            # reduce dimensionality of feature_maps by sparse random projection
            feature_map = get_feature_map_srps(feature_map, device=device)
            
            # Avoiding "CUDA error: an illegal memory access was encountered"
            X = feature_map.detach().clone().squeeze().to(torch.float32)
            del feature_map
            torch.cuda.empty_cache()

            y_pred, y_true = [], [] #Initialize lists
            cv_iterator = tqdm(cv.split(X), desc='CV', total=n_splits)
            for train_index, test_index in cv_iterator:
                X_train, X_test = X[train_index].detach().clone(), X[test_index].detach().clone()
                y_train, y_test = y[train_index].detach().clone(), y[test_index].detach().clone()
                pipe.fit(X_train, y_train)
                y_pred.append(pipe.predict(X_test))
                y_true.append(y_test)
            
            scores = score_func(torch.cat(y_pred), torch.cat(y_true))

            # save the current scores to disk
            scores_arr = scores.cpu().detach().numpy()
            np.save(f'{file_path}/layer-{feature_map_uid}.npy', scores_arr)

            if scores_out is None:
                scores_out = scores_arr.copy()
                model_layer_index = np.ones_like(scores_out, dtype='int') + layer_index_offset
                model_layer = np.zeros_like(scores_out, dtype='object')
                model_layer.fill(feature_map_uid)
            else:
                # replace the value in the output if the previous value is less than the current value
                scores_out[scores_out < scores_arr] = scores_arr[scores_out < scores_arr]
                model_layer_index[scores_out < scores_arr] = layer_index + layer_index_offset
                model_layer[scores_out < scores_arr] = feature_map_uid

    # Make scoresheet based on the benchmark metadata
    results = []
    for i, row in benchmark.metadata.iterrows():
        row['layer_index'] = model_layer_index[i]
        row['layer'] = model_layer[i]
        row['score'] = scores_out[i]
        results.append(row)

    return pd.DataFrame(results)


def get_glove_training_benchmarking_results(benchmark, feature_map,
                                      device='cuda',
                                      n_splits=4, random_seed=0, 
                                      alphas=[10.**power for power in np.arange(-5, 2)]):
    # use a CUDA-capable device, if available, else: CPU
    print(f'device: {device}')
    print(cuda_device_report())

    # initialize pipe and kfold splitter
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    score_func = get_scoring_method('pearsonr')
    pipe = TorchRidgeGCV(alphas=alphas, alpha_per_target=True,
                            device=device, scale_X=True,)
            
    # Avoiding "CUDA error: an illegal memory access was encountered"
    feature_map = get_feature_map_srps(feature_map, device=device)
    X = feature_map.detach().clone().squeeze().to(torch.float32)
    del feature_map
    torch.cuda.empty_cache()

    # Send the neural data to the GPU
    y = torch.from_numpy(benchmark.response_data.to_numpy().T).to(torch.float32).to(device)

    y_pred, y_true = [], [] #Initialize lists
    cv_iterator = tqdm(cv.split(X), desc='CV', total=n_splits)
    for train_index, test_index in cv_iterator:
        X_train, X_test = X[train_index].detach().clone(), X[test_index].detach().clone()
        y_train, y_test = y[train_index].detach().clone(), y[test_index].detach().clone()
        pipe.fit(X_train, y_train)
        y_pred.append(pipe.predict(X_test))
        y_true.append(y_test)
    
    scores = score_func(torch.cat(y_pred), torch.cat(y_true)).cpu().detach().numpy()

    # Make scoresheet based on the benchmark metadata
    results = []
    for i, row in benchmark.metadata.iterrows():
        row['score'] = scores[i]
        results.append(row)

    return pd.DataFrame(results)


def get_vision_benchmarking_results(benchmark, feature_extractor, file_path,
                                    layer_index_offset=0, device='cuda',
                                    n_splits=4, random_seed=0):

    # use a CUDA-capable device, if available, else: CPU
    print(f'{device=}')
    print(f'{cuda_device_report()=}')

    # initialize pipe and kfold splitter
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    alphas = [10.**power for power in np.arange(-5, 2)]
    score_func = get_scoring_method('pearsonr')
    pipe = TorchRidgeGCV(alphas=alphas, alpha_per_target=True,
                            device=device, scale_X=True,)
    
    # Send the neural data to the GPU
    y = torch.from_numpy(benchmark.response_data.to_numpy().T).to(torch.float32).to(device)

    # Get the SRP matrix
    global_srp_matrix = feature_extractor.get_global_srp_matrix()
    global_srp_on_gpu = global_srp_matrix.clone().to(device)

    layer_index = 0 # keeps track of depth
    scores_out = None
    for feature_maps in feature_extractor:
        feature_map_iterator = tqdm(feature_maps.items(), desc = 'Brain Mapping (Layer)', leave=False)
        for feature_map_uid, feature_map in feature_map_iterator:
            layer_index += 1 # one layer deeper in feature_maps

            srp_kwargs = {'device': device+':2', 'srp_matrix': global_srp_on_gpu}
            feature_map = compute_srp(feature_map, **srp_kwargs)

            # Avoiding "CUDA error: an illegal memory access was encountered"
            X = feature_map.detach().clone().squeeze().to(torch.float32)
            del feature_map
            torch.cuda.empty_cache()

            y_pred, y_true = [], [] #Initialize lists
            cv_iterator = tqdm(cv.split(X), desc='CV', total=n_splits)
            for train_index, test_index in cv_iterator:
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                pipe.fit(X_train, y_train)
                y_pred.append(pipe.predict(X_test))
                y_true.append(y_test)
            
            scores = score_func(torch.cat(y_pred), torch.cat(y_true))

            # save the current scores to disk
            scores_arr = scores.cpu().detach().numpy()
            np.save(f'{file_path}/layer-{feature_map_uid}.npy', scores_arr)

            if scores_out is None:
                scores_out = scores_arr.copy()
                model_layer_index = np.ones_like(scores_out, dtype='int') + layer_index_offset
                model_layer = np.zeros_like(scores_out, dtype='object')
                model_layer.fill(feature_map_uid)
            else:
                # replace the value in the output if the previous value is less than the current value
                scores_out[scores_out < scores_arr] = scores_arr[scores_out < scores_arr]
                model_layer_index[scores_out < scores_arr] = layer_index + layer_index_offset
                model_layer[scores_out < scores_arr] = feature_map_uid

    # Make scoresheet based on the benchmark metadata
    results = []
    for i, row in benchmark.metadata.iterrows():
        row['layer_index'] = model_layer_index[i]
        row['layer'] = model_layer[i]
        row['score'] = scores_out[i]
        results.append(row)

    return pd.DataFrame(results)