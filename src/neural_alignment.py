#
from deepjuice.alignment import TorchRidgeGCV, get_scoring_method
import torch
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pandas as pd
import gc 
from src import stats
from src.stats import feature_scaler
from deepjuice.extraction import FeatureExtractor
from deepjuice.reduction import get_feature_map_srps
from deepjuice.systemops.devices import cuda_device_report
from deepjuice.procedural import pandas_query
from deepjuice.model_zoo.options import get_deepjuice_model
from deepjuice.procedural.datasets import get_data_loader
from deepjuice.extraction import FeatureExtractor
from deepjuice.procedural.cv_ops import CVIndexer
from deepjuice.alignment import TorchRidgeGCV
from deepjuice.tensorops import apply_tensor_op
from deepjuice.alignment import compute_rdm, compare_rdms

from deepjuice.reduction import compute_srp
from deepjuice.alignment import compute_score


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

    # define the feature extractor object
    if grouping_func == 'grouped_average':
        extractor = FeatureExtractor(model, dataloader, **{'device': devices[0], 'output_device': devices[0]},
                                    tensor_fn=grouped_average,
                                    memory_limit=memory_limit,
                                    batch_strategy='stack')
    elif grouping_func == 'grouped_stack':
        extractor = FeatureExtractor(model, dataloader, **{'device': devices[0], 'output_device': devices[0]},
                            tensor_fn=grouped_stack,
                            memory_limit=memory_limit,
                            batch_strategy='stack')
    else:
        extractor = FeatureExtractor(model, dataloader,
                                     **{'device': devices[0],
                                         'output_device': devices[0]},
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

    print('Starting CV in the training set')
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

            # reduce dimensionality of feature_maps by sparse random projection
            feature_map = get_feature_map_srps(feature_map, device=devices[-1])
            
            X = feature_map.detach().clone().squeeze().to(torch.float32).to(devices[-1])
            X = {'train': X[indices['train']], 'test': X[indices['test']]}
            del feature_map
            torch.cuda.empty_cache()

            # Memory saving
            pipe = TorchRidgeGCV(alphas=alphas, alpha_per_target=True, 
                                 device=devices[-1], scale_X=False)

            try:
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
                    y_cv_hat = pipe.predict(X_cv_test)
                    y_cv_pred.append(y_cv_hat)
                    y_cv_true.append(y_cv_test)
                scores_train = score_func(torch.cat(y_cv_pred), torch.cat(y_cv_true)) # Get the CV training scores
                scores_train = scores_train.cpu().detach().numpy()

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
            except:
                print(f'\nFitting failed to converge for {model_name} {feature_map_uid} ({layer_index + layer_index_offset})')
    
    # Add training data to a dataframe
    results = benchmark.metadata.copy()
    results['layer_index'] = model_layer_index_max
    results['layer_relative_depth'] = model_layer_index_max/ (layer_index + layer_index_offset)
    results['layer'] = model_layer_max
    results['train_score'] = scores_train_max
    results['model_uid'] = model_name

    if test_eval:
        print('Running evaluation in the test set')
        scores_test_max = np.zeros_like(scores_train_max)
        y_hat_max = torch.zeros_like(y_cv_hat)
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

                    try:
                        X_train, X_test = feature_scaler(X['train'].detach().clone(), X['test'].detach().clone())
                        if scale_y:
                            y_train, y_test = feature_scaler(y['train'].detach().clone(), y['test'].detach().clone())
                        else:
                            y_train, y_test = y['train'].detach().clone(), y['test'].detach().clone()

                        pipe.fit(X_train, y['train'])
                        y_hat = pipe.predict(X_test)
                        scores_test = score_func(y_hat, y['test']).cpu().detach().numpy()

                        #Save the test set scores to an array only if it is where performance was maximum in the training set
                        idx = model_layer_index_max == (layer_index + layer_index_offset)
                        scores_test_max[idx] = scores_test[idx]
                        y_hat_max[:, idx] = y_hat[:, idx]

                        # Memory saving
                        del pipe, scores_test
                        del X, X_train, X_test, y_train
                        gc.collect()
                        torch.cuda.empty_cache()
                    except:
                        print(f'\nFitting failed to converge for {model_name} {feature_map_uid} ({layer_index + layer_index_offset})')
                else: 
                    print(f'{feature_map_uid} (layer {layer_index}) is not a max layer in train set')
                    print('skipping test set regression')

        # Add test set results to the dataframe
        results['test_score'] = scores_test_max
        results['r_null_dist'] = np.nan
        results['r_var_dist'] = np.nan
        results['r_null_dist'] = results['r_null_dist'].astype('object')
        results['r_var_dist'] = results['r_var_dist'].astype('object')

        # Do permutation testing on voxels in ROIs
        roi_indices = benchmark.metadata.index[benchmark.metadata.roi_name != 'none'].to_numpy()
        print(type(roi_indices))
        print(f'{y_test.shape=}')
        print(f'{y_hat_max.shape=}')
        r_null = stats.perm_gpu(y_test[:, roi_indices],
                                y_hat_max[:, roi_indices],
                                verbose=True).cpu().detach().numpy().T.tolist()
        r_var = stats.bootstrap_gpu(y_test[:, roi_indices], 
                                    y_hat[:, roi_indices],
                                    verbose=True).cpu().detach().numpy().T.tolist()
        for idx, (r_null_val, r_var_val) in zip(roi_indices, zip(r_null, r_var)):
            results.at[idx, 'r_null_dist'] = r_null_val
            results.at[idx, 'r_var_dist'] = r_var_val

    print(results.head(20))
    return results


def get_video_benchmarking_results(benchmark, feature_extractor,
                             layer_index_offset=0,
                             devices=['cuda:0', 'cuda:1'],
                             n_splits=4, random_seed=0,
                             model_name=None,
                             scale_y=True,
                             test_eval=False,
                             alphas=[10. ** power for power in np.arange(-5, 2)]):

    # use a CUDA-capable device, if available, else: CPU
    print(cuda_device_report())

    # initialize pipe and kfold splitter
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    score_func = get_scoring_method('pearsonr')

    # divide responses
    indices = {'train': benchmark.stimulus_data[benchmark.stimulus_data.stimulus_set == 'train'].index,
               'test': benchmark.stimulus_data[benchmark.stimulus_data.stimulus_set == 'test'].index}
    y = {'train': torch.from_numpy(benchmark.response_data.to_numpy().T[indices['train']]).to(torch.float32).to(
        devices[-1]),
         'test': torch.from_numpy(benchmark.response_data.to_numpy().T[indices['test']]).to(torch.float32).to(
             devices[-1])}

    print('Starting CV in the training set')
    layer_index = 0  # keeps track of depth
    scores_train_max = None
    results = []
    for batch, feature_maps in enumerate(feature_extractor):
        print(f"Running batch: {batch+1}")
        feature_map_iterator = tqdm(feature_maps.items(), desc = 'Training Mapping (Layer)', leave=False)
        for feature_map_uid, feature_map in feature_map_iterator:
            layer_index += 1 # one layer deeper in feature_maps

            # reduce dimensionality of feature_maps by sparse random projection
            feature_map = get_feature_map_srps(feature_map, device=devices[-1])

            X = feature_map.detach().clone().squeeze().to(torch.float32).to(devices[-1])
            X = {'train': X[indices['train']], 'test': X[indices['test']]}
            del feature_map
            torch.cuda.empty_cache()

            # Memory saving
            pipe = TorchRidgeGCV(alphas=alphas, alpha_per_target=True,
                                 device=devices[-1], scale_X=False)

            #### Fit CV in the train set ####
            y_cv_pred, y_cv_true = [], []  # Initialize lists
            for i, (cv_train_index, cv_test_index) in enumerate(cv.split(X['train'])):
                # Split the training set
                X_cv_train, X_cv_test = X['train'][cv_train_index].detach().clone(), X['train'][
                    cv_test_index].detach().clone()
                y_cv_train, y_cv_test = y['train'][cv_train_index].detach().clone(), y['train'][
                    cv_test_index].detach().clone()

                # Scale X and y
                X_cv_train, X_cv_test = feature_scaler(X_cv_train, X_cv_test)
                if scale_y:
                    y_cv_train, y_cv_test = feature_scaler(y_cv_train, y_cv_test)

                # Fit the regression
                pipe.fit(X_cv_train, y_cv_train)
                y_cv_hat = pipe.predict(X_cv_test)
                y_cv_pred.append(y_cv_hat)
                y_cv_true.append(y_cv_test)
            scores_train = score_func(torch.cat(y_cv_pred), torch.cat(y_cv_true))  # Get the CV training scores
            scores_train = scores_train.cpu().detach().numpy()

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
        print('Running evaluation in the test set')
        scores_test_max = np.zeros_like(scores_train_max)
        y_hat_max = torch.zeros_like(y_cv_hat)
        layer_index = 0
        for batch, feature_maps in enumerate(feature_extractor):
            print(f"Running batch: {batch + 1}")
            feature_map_iterator = tqdm(feature_maps.items(), desc='Testing Mapping (Layer)', leave=False)
            for feature_map_uid, feature_map in feature_map_iterator:
                layer_index += 1  # one layer deeper in feature_maps

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
                    scores_test = score_func(y_hat, y['test']).cpu().detach().numpy()

                    # Save the test set scores to an array only if it is where performance was maximum in the training set
                    idx = model_layer_index_max == (layer_index + layer_index_offset)
                    scores_test_max[idx] = scores_test[idx]
                    y_hat_max[:, idx] = y_hat[:, idx]

                    # Memory saving
                    del pipe, scores_test
                    del X, X_train, X_test, y_train
                    gc.collect()
                    torch.cuda.empty_cache()
                else:
                    print(f'{feature_map_uid} (layer {layer_index}) is not a max layer in train set')
                    print('skipping test set regression')

        # Add test set results to the dataframe
        results['test_score'] = scores_test_max

        # Run permutation testing and bootstapping
        results['r_null_dist'] = np.nan
        results['r_var_dist'] = np.nan

        roi_indices = benchmark.metadata.index[benchmark.metadata.roi_name != 'none'].to_numpy()
        print(type(roi_indices))
        print(len(roi_indices))
        print(f'{y_test.shape=}')
        print(f'{y_hat_max.shape=}')
        r_null = stats.perm_gpu(y_test[:, roi_indices], y_hat_max[:, roi_indices], verbose=True)
        r_var = stats.bootstrap_gpu(y_test[:, roi_indices], y_hat[:, roi_indices], verbose=True)
        results.iloc[roi_indices]['r_null_dist'] = r_null.cpu().detach().numpy().T.tolist()
        results.iloc[roi_indices]['r_var_dist'] = r_var.cpu().detach().numpy().T.tolist()
    print(results.head(20))
    return results


def get_lm_encoded_training_benchmarking_results(benchmark, feature_map, device='cuda',
                                      n_splits=4, random_seed=0,
                                      alphas=[10.**power for power in np.arange(-5, 2)]):
    # use a CUDA-capable device, if available, else: CPU
    print(f'{device=}')
    print(f'{cuda_device_report()}')

    # initialize pipe and kfold splitter
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    score_func = get_scoring_method('pearsonr')
    pipe = TorchRidgeGCV(alphas=alphas, alpha_per_target=True,
                            device=device, scale_X=True,)

    # Get X
    feature_map = get_feature_map_srps(feature_map, device=device)
    X = feature_map.detach().clone().squeeze().to(torch.float32).to(device)
    print(f'{X.shape=}')

    # Send the neural data to the GPU
    y = torch.from_numpy(benchmark.response_data.to_numpy().T).to(torch.float32).to(device)
    print(f'{y.shape=}')

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

def get_training_rsa_benchmark_results(benchmark, feature_extractor,
                                      layer_index_offset=0,
                                      device='cuda',
                                      n_splits=5, random_seed=1,
                                      alphas=np.logspace(-1, 5, 7).tolist(),
                                      metrics = ['crsa', 'ersa'],
                                      feature_map_stats=None,
                                      model_uid = None,
                                      stack_final_results=True):
    """
    Benchmarks the performance of neural network feature extractors against brain response data,
    using Representational Similarity Analysis (RSA) metrics. This involves cross-validated comparison
    of neural network activations to human brain activity patterns recorded during similar tasks.

    Parameters:
        benchmark (Benchmark): An object containing brain response data and metadata necessary for RSA.
        feature_extractor: An iterable or generator that yields feature maps from a neural network model.
        layer_index_offset (int, optional): Offset for layer indexing, useful for models with skipped layers. Defaults to 0.
        device (str, optional): Computation device ('cuda' or 'cpu'). Defaults to 'cuda'.
        n_splits (int, optional): Number of splits for k-fold cross-validation. Defaults to 5.
        random_seed (int, optional): Seed for random number generation for reproducible splits. Defaults to 1.
        alphas (list, optional): List of alpha values for Ridge regression cross-validation.
        metrics (list of str, optional): List of RSA metrics to compute, such as 'crsa' and 'ersa'. Defaults to ['crsa', 'ersa'].
        feature_map_stats (dict, optional): Precomputed statistics of feature maps for normalization. Defaults to None.
        model_uid (str, optional): UID of the neural network model being benchmarked. Defaults to None.
        stack_final_results (bool, optional): Whether to stack results into a single DataFrame. Defaults to True.

    Returns:
        pandas.DataFrame or dict: The RSA benchmarking results. If `stack_final_results` is True, returns a single DataFrame
        containing the results; otherwise, returns a dictionary of DataFrames, one for each RSA metric.
    """

    # HELPER FUNCTIONS #
    def generate_fold_indices(n_splits=5, random_seed=1, n_stimuli=200) -> list:
        """
        Generates indices for training and testing splits based on k-fold cross-validation.

        Parameters:
            n_splits (int, optional): Number of folds for the cross-validation. Defaults to 5.
            random_seed (int, optional): Seed for the random number generator to ensure reproducibility. Defaults to 1.
            n_stimuli (int, optional): Number of stimuli to create a dummy map for splitting. Defaults to 200.

        Returns:
            list: A list of dictionaries, each containing 'train' and 'test' keys with arrays of indices for training and testing splits.
        """
        blank_map = np.ones((n_stimuli, 5000))
        ind_splits = []
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        for i, (train_index, test_index) in enumerate(kf.split(blank_map)):
            ind_split = {'train': train_index, 'test': test_index}
            ind_splits.append(ind_split)
        return ind_splits

    def get_kfold_xy_rdms(feature_map: np.array, responses: np.array, ind_splits: list) -> list:
        """
        Splits feature maps and responses into k-fold training and testing sets, applying standard scaling.

        Parameters:
            feature_map (np.array)(tensor): The feature map array to be split, it is expected it comes as a tensor from gpu.
            responses (np.array): The response array corresponding to the feature map.
            ind_splits (list): A list of dictionaries with 'train' and 'test' keys containing indices for training and testing splits.

        Returns:
            list: A list of dictionaries, each representing a data split with 'train' and 'test' keys. Each key maps to a dictionary
            containing 'X' (features) and 'y' (responses), which are scaled and split according to the provided indices.
        """
        scaling = StandardScaler()
        data_splits = []

        for i, indices in enumerate(ind_splits):
            data_split = {'train': {}, 'test': {}}
            # model splits
            feature_map = apply_tensor_op(feature_map, lambda x: x.to('cpu'))
            xtrain_scaled = scaling.fit_transform(feature_map[indices['train']])
            xtest_scaled = scaling.transform(feature_map[indices['test']])
            data_split['train']['X'] = xtrain_scaled
            data_split['test']['X'] = xtest_scaled
            # responses splits
            data_split['train']['y'] = responses.T[indices['train']]
            data_split['test']['y'] = responses.T[indices['test']]
            data_splits.append(data_split)
        return data_splits

    def get_rdm_splits(rdms: dict, ind_splits: list) -> list:
        """
        Splits representational dissimilarity matrices (RDMs) according to the provided k-fold cross-validation indices.

        Parameters:
            rdms (dict): A dictionary where keys are ROI names and values are subject-specific RDMs.
            ind_splits (list): A list of dictionaries with 'train' and 'test' keys containing indices for training and testing splits.

        Returns:
            list: A list of dictionaries, each representing the split RDMs for training and testing in each fold. The structure
            mirrors that of the input rdms, with an added layer for 'train' and 'test' splits.
        """
        rdm_splits = []
        for i, indices in enumerate(ind_splits):
            # rdm splits
            split_rdm = {}
            for roi_name in rdms:
                split_rdm[roi_name] = {}
                for subj_id in rdms[roi_name]:
                    empty_train = []
                    for row in rdms[roi_name][subj_id][indices['train']]:
                        empty_train.append(row[indices['train']])
                    train_rdm = torch.stack(empty_train, dim=0)
                    empty_test = []
                    for row in rdms[roi_name][subj_id][indices['test']]:
                        empty_test.append(row[indices['test']])
                    test_rdm = torch.stack(empty_test, dim=0)
                    split_rdm[roi_name][subj_id] = {'train': train_rdm,
                                                    'test': test_rdm}
            rdm_splits.append(split_rdm)
        return rdm_splits

    # use a CUDA-capable device, if available, else: CPU
    print(f'Running on device: {device}')
    print(cuda_device_report())

    # Send the neural data to the GPU
    Y = (apply_tensor_op(benchmark.response_data.to_numpy()).to(dtype=torch.float32, device=device))

    # initialize an empty list to record scores over layers
    scoresheet_lists = {metric: [] for metric in metrics}
    # if no feature_map_stats provided, make empty dict:
    if feature_map_stats is None: feature_map_stats = {}
    # get the voxel (neuroid) indices for each specified roi
    row_indices = benchmark.row_indices

    # get the kfold indices
    if benchmark.video_set == 'train':
        n_stimuli = 200
    elif benchmark.video_set == 'test':
        n_stimuli = 50
    else:
        n_stimuli = 250
    ind_splits = generate_fold_indices(n_splits, random_seed, n_stimuli)
    # use ind_splits to split the rdms
    rdm_splits = get_rdm_splits(benchmark.rdms, ind_splits)

    # now, we iterate over our extractor
    layer_index = 0  # keeps track of depth
    for feature_maps in feature_extractor:
        # dimensionality reduction of feature maps
        feature_maps = get_feature_map_srps(feature_maps, device='cuda')
        # now, we loop over our batch of feature_maps from the extractor...
        # ...starting by defining an iterator that will track our progress
        feature_map_iterator = tqdm(feature_maps.items(), desc='Brain Mapping (Layer)')

        for feature_map_uid, feature_map in feature_map_iterator:
            # index the 5 fold splits for this layer
            xy_folds = get_kfold_xy_rdms(feature_map, Y, ind_splits)

            layer_index += 1 # one layer deeper in feature_maps

            # loop over each fold in the kfold
            for i, fold in enumerate(xy_folds):
                # main data to add to our scoresheet per feature_map
                feature_map_info = {'model_uid': model_uid,
                                    'model_layer': feature_map_uid,
                                    # layer_index_offset is used here in case of subsetting
                                    'model_layer_index': layer_index + layer_index_offset,
                                    'k_fold': i+1}

                # now, our X Variable that we push onto the gpu:
                X = {'train': apply_tensor_op(fold['train']['X']).to(dtype=torch.float32, device='cuda'),
                     'test': apply_tensor_op(fold['test']['X']).to(dtype=torch.float32, device='cuda')}

                y = {'train': fold['train']['y'],
                      'test': fold['test']['y']}

                # initialize the regression, in this case ridge regression with LOOCV over alphas
                regression = TorchRidgeGCV(alphas=alphas, device='cuda', scale_X=True)
                regression.fit(X['train'], y['train']) # fit the regression on the train split
                # RidgeGCV gives us both internally generated LOOCV values for the train dataset
                # as well as the ability to predict our test set in the same way as any regressor
                y_pred = {'train': regression.cv_y_pred_, 'test': regression.predict(X['test'])}

                # loop over cRSA, eRSA...
                for metric in scoresheet_lists:

                    # encoding RSA score
                    if metric == 'ersa':
                        for split in ['train', 'test']:
                            for region in benchmark.rdms:
                                for subj_id in benchmark.rdms[region]:
                                    # get the relevant train-test split of the brain RDM
                                    target_rdm = rdm_splits[i][region][subj_id][split]
                                    # get the response_indices for current ROI group
                                    response_indices = row_indices[region][subj_id]
                                    # get predicted values for each response_index...
                                    y_pred_i = y_pred[split][:, response_indices]
                                    # ... and use them to calculate the weighted RDM
                                    model_rdm = compute_rdm(y_pred_i, 'pearson')
                                    # compare brain-reweighted model RDM to brain RDM
                                    # with our specified 2nd-order distance metric...
                                    score = compare_rdms(model_rdm, target_rdm, method='spearman')

                                    # add the scores to a "scoresheet"
                                    scoresheet = {**feature_map_info,
                                                  'region': region,
                                                  'subj_id': subj_id,
                                                  'cv_split': split,
                                                  'score': score}

                                    # append the scoresheet to our running list
                                    scoresheet_lists['ersa'].append(scoresheet)

                    elif metric == 'crsa':
                        for split in ['train', 'test']:
                            # get the relevant train-test split of the model RDM
                            model_rdm = compute_rdm(X[split], method='pearson', device=device)
                            for region in benchmark.rdms:
                                for subj_id in benchmark.rdms[region]:
                                    # get the relevant train-test split of the brain RDM
                                    target_rdm = rdm_splits[i][region][subj_id][split]
                                    # compare lower triangles of model + brain RDM
                                    # with our specified 2nd-order distance metric
                                    score = compare_rdms(model_rdm, target_rdm, method='spearman')

                                    # add the scores to a "scoresheet"
                                    scoresheet = {**feature_map_info,
                                                  'region': region,
                                                  'subj_id': subj_id,
                                                  'cv_split': split,
                                                  'score': score}

                                    # append the scoresheet to our running list
                                    scoresheet_lists['crsa'].append(scoresheet)

                # clean up tensors on gpu
                X = {key: tensor.to('cpu') for key, tensor in X.items()}
                del regression
                gc.collect()
                torch.cuda.empty_cache()

    results = {metric: pd.DataFrame(scores) for metric, scores in scoresheet_lists.items()}
    if stack_final_results:
        # if we do stack, results are a single concatenated dataframe
        # with only the common_columns of each (excluding method data)
        result_columns = pd.unique([col for results in results.values() for col in results.columns]).tolist()
        common_columns = [col for col in result_columns if all(col in result.columns for result in results.values())]
        common_columns = ['metric'] + common_columns # indicator
        results_list = []
        for metric, result in results.items():
            result.insert(0, 'metric', metric)
            results_list.append(result[common_columns])

    return pd.concat(results_list) if stack_final_results else results
