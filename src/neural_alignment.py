#
from deepjuice.alignment import TorchRidgeGCV, get_scoring_method
import torch
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pandas as pd
import gc
import time
from src import stats
from src.stats import feature_scaler
from deepjuice.alignment import TorchRidgeGCV, get_scoring_method, compute_rdm, compare_rdms
from deepjuice.reduction import get_feature_map_srps
from deepjuice.systemops.devices import cuda_device_report
from deepjuice.procedural import pandas_query
from deepjuice.model_zoo.options import get_deepjuice_model
from deepjuice.procedural.datasets import get_data_loader
from deepjuice.extraction import FeatureExtractor
from deepjuice.tensorops import apply_tensor_op, convert_to_tensor
from deepjuice.model_zoo import get_model_options
from deepjuice.procedural.cv_ops import CVIndexer
from deepjuice.alignment import TorchRidgeGCV
from deepjuice.tensorops import apply_tensor_op
from deepjuice.alignment import compute_rdm, compare_rdms
from deepjuice.reduction import compute_srp
from deepjuice.alignment import compute_score


def memory_stats(devices):
    def bit_to_gb(bit_):
        return bit_ / (8 * 1024 ** 3)

    for device in devices:
        memory_allocated = bit_to_gb(torch.cuda.memory_allocated(device))
        memory_cached = bit_to_gb(torch.cuda.memory_cached(device))
        print(f'{device} allocated = {memory_allocated:.2f} GB')
        print(f'{device} cached = {memory_cached:.2f} GB')


def get_benchmarking_results(benchmark, model, dataloader,
                             layer_index_offset=0,
                             devices=['cuda:0', 'cuda:1'],
                             memory_limit='30GB',
                             n_splits=4, random_seed=0,
                             model_name=None,
                             scale_y=True,
                             test_eval=False,
                             run_bootstrapping=False,
                             stream_statistics=False,
                             grouping_func='grouped_average',
                             batch_time=False,
                             alphas=[10.**power for power in np.arange(-5, 2)]):

    # Define a grouping function to average across the different captions
    def grouped_average(tensor, batch_iter=None, **kwargs):
        if batch_iter is None: return tensor  # as is

        sub_data = dataloader.batch_data.query('batch_iter==@batch_iter')

        tensor_means = []  # fill with group tensor means
        for group in sub_data.group_index.unique():
            group_data = sub_data.query('group_index==@group')
            group_idx = group_data.batch_index.to_list()

            # convert index to tensor on device
            group_idx = (torch.LongTensor(group_idx)
                         .to(tensor.device))

            tensor_mean = tensor[group_idx].mean(dim=0)
            tensor_means += [tensor_mean.unsqueeze(0)]

        return torch.concat(tensor_means, dim=0)  # as is for testing

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
    if grouping_func == 'grouped_stack':
        extractor = FeatureExtractor(model, dataloader,
                                     tensor_fn=grouped_stack,
                                     memory_limit=memory_limit,
                                     batch_strategy='stack', flatten=True,
                                     **{'device': devices[0], 'output_device': devices[0]})
    else:  # grouping_func == 'grouped_average':
        extractor = FeatureExtractor(model, dataloader,
                                     tensor_fn=grouped_average,
                                     memory_limit=memory_limit,
                                     batch_strategy='stack', flatten=True,
                                     **{'device': devices[0], 'output_device': devices[0]})

    # initialize pipe and kfold splitter
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    score_func = get_scoring_method('pearsonr')

    # divide responses
    indices = {'train': benchmark.stimulus_data[benchmark.stimulus_data.stimulus_set == 'train'].index,
               'test': benchmark.stimulus_data[benchmark.stimulus_data.stimulus_set == 'test'].index}
    y_train = torch.from_numpy(benchmark.response_data.to_numpy().T[indices['train']]).to(torch.float32).to(devices[-1])

    print('\n\n\n\n\nStarting CV in the training set')
    layer_index = 0  # keeps track of depth
    scores_train_max = None
    results = []
    extractor_iterator = tqdm(extractor, desc='Extractor Steps')
    for batched_feature_maps in extractor_iterator:
        print(batched_feature_maps)
        if batch_time:
            start_batch_time = time.time()
        feature_maps = batched_feature_maps.join_batches()
        feature_map_iterator = tqdm(feature_maps.items(), desc='CV Mapping Layer', leave=False)
        for feature_map_uid, feature_map in feature_map_iterator:
            layer_index += 1  # one layer deeper in feature_maps

            # reduce dimensionality of feature_maps by sparse random projection
            feature_map = get_feature_map_srps(feature_map[indices['train']], device=devices[-1])
            X_train = feature_map.detach().clone().squeeze().to(torch.float32).to(devices[-1])

            # Memory saving
            del feature_map
            gc.collect()
            torch.cuda.empty_cache()
            pipe = TorchRidgeGCV(alphas=alphas, alpha_per_target=True,
                                 device=devices[-1], scale_X=False)

            try:
                #### Fit CV in the train set ####
                y_cv_pred, y_cv_true = [], []  # Initialize lists
                for i, (cv_train_index, cv_test_index) in enumerate(cv.split(X_train)):
                    # Split the training set
                    X_cv_train, X_cv_test = X_train[cv_train_index], X_train[cv_test_index]
                    y_cv_train, y_cv_test = y_train[cv_train_index], y_train[cv_test_index]

                    # Scale X and y
                    X_cv_train, X_cv_test = feature_scaler(X_cv_train, X_cv_test)
                    if scale_y:
                        y_cv_train, y_cv_test = feature_scaler(y_cv_train, y_cv_test)

                    # Fit the regression
                    pipe.fit(X_cv_train, y_cv_train)
                    y_cv_hat = pipe.predict(X_cv_test)
                    y_cv_pred.append(y_cv_hat)
                    y_cv_true.append(y_cv_test)
                scores_train = score_func(torch.cat(y_cv_pred), torch.cat(y_cv_true)).cpu().detach().numpy() # Get the CV training scores 

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
                del X_train, X_cv_train, X_cv_test
                del y_cv_train, y_cv_test
                gc.collect()
                torch.cuda.empty_cache()
            except:
                print(f'\nFitting failed to converge for {model_name} {feature_map_uid} ({layer_index + layer_index_offset})')
        if batch_time:
            end_batch_time = time.time()
            elapsed = end_batch_time - start_batch_time
            elapsed = time.gmtime(elapsed)
            return elapsed

    # Add training data to a dataframe
    results = benchmark.metadata.copy()
    results['layer_index'] = model_layer_index_max
    results['layer_relative_depth'] = model_layer_index_max / (layer_index + layer_index_offset)
    results['layer'] = model_layer_max
    results['train_score'] = scores_train_max
    results['model_uid'] = model_name

    # Free up memory
    memory_stats(devices)
    y_hat_max = torch.zeros_like(y_cv_hat)
    del y_train, y_cv_hat, extractor
    gc.collect()
    torch.cuda.empty_cache()
    memory_stats(devices)

    if test_eval:
        print('\n\n\n\n\nRunning evaluation in the test set')
        print('resting extractor')
        # define the feature extractor object
        if grouping_func == 'grouped_stack':
            extractor = FeatureExtractor(model, dataloader,
                                        tensor_fn=grouped_stack,
                                        memory_limit=memory_limit,
                                        batch_strategy='stack', flatten=True,
                                        **{'device': devices[0], 'output_device': devices[0]})
        else:# grouping_func == 'grouped_average':
            extractor = FeatureExtractor(model, dataloader,
                                        tensor_fn=grouped_average,
                                        memory_limit=memory_limit,
                                        batch_strategy='stack', flatten=True,
                                        **{'device': devices[0], 'output_device': devices[0]})

        print('resetting y')
        y_train = torch.from_numpy(benchmark.response_data.to_numpy().T[indices['train']]).to(torch.float32).to(devices[-1])
        y_test = torch.from_numpy(benchmark.response_data.to_numpy().T[indices['test']]).to(torch.float32).to(devices[-1])
        if scale_y:
            y_train, y_test = feature_scaler(y_train, y_test)

        scores_test_max = np.zeros_like(scores_train_max)
        layer_index = 0
        extractor_iterator = tqdm(extractor, desc='Extractor Steps')
        for batched_feature_maps in extractor_iterator:
            print(batched_feature_maps)
            feature_maps = batched_feature_maps.join_batches()
            feature_map_iterator = tqdm(feature_maps.items(), desc='Testing Mapping Layer', leave=False)
            for feature_map_uid, feature_map in feature_map_iterator:
                layer_index += 1  # one layer deeper in feature_maps

                # reduce dimensionality of feature_maps by sparse random projection
                feature_map = get_feature_map_srps(feature_map, device=devices[-1])
                X = feature_map.detach().clone().squeeze().to(torch.float32).to(devices[-1])
                X_train, X_test = feature_scaler(X[indices['train']], X[indices['test']])

                # Memory saving
                del feature_map, X
                torch.cuda.empty_cache()
                pipe = TorchRidgeGCV(alphas=alphas, alpha_per_target=True,
                                     device=devices[-1], scale_X=False)

                try:
                    pipe.fit(X_train, y_train)
                    y_hat = pipe.predict(X_test)
                    scores_test = score_func(y_hat, y_test).cpu().detach().numpy()

                    # Save the test set scores to an array only if it is where performance was maximum in the training set
                    idx = model_layer_index_max == (layer_index + layer_index_offset)
                    scores_test_max[idx] = scores_test[idx]
                    y_hat_max[:, idx] = y_hat[:, idx]

                    # Memory saving
                    del pipe, scores_test
                    del X_train, X_test
                    gc.collect()
                    torch.cuda.empty_cache()
                except:
                    print(
                        f'\nFitting failed to converge for {model_name} {feature_map_uid} ({layer_index + layer_index_offset})')

        # Add test set results to the dataframe
        results['test_score'] = scores_test_max
        results['r_null_dist'] = np.nan
        results['r_null_dist'] = results['r_null_dist'].astype('object')

        # Do permutation testing on voxels in ROIs
        # If stream_statistics also run the statistics in the stream ROIs
        if stream_statistics:
            roi_indices = benchmark.metadata.index[(benchmark.metadata.stream_name != 'none') |
                                                    ( benchmark.metadata.roi_name != 'none')].to_numpy()
        else:
            roi_indices = benchmark.metadata.index[benchmark.metadata.roi_name != 'none'].to_numpy()
        print(type(roi_indices))
        print(f'{y_test.shape=}')
        print(f'{y_hat_max.shape=}')
        r_null = stats.perm_gpu(y_test[:, roi_indices],
                                y_hat_max[:, roi_indices],
                                verbose=True).cpu().detach().numpy().T.tolist()
        for idx, r_null_val in tqdm(zip(roi_indices, r_null), desc='Permutation results to pandas'):
            results.at[idx, 'r_null_dist'] = r_null_val

        # Run the bootstrapping only if specified
        if run_bootstrapping:
            results['r_var_dist'] = np.nan
            results['r_var_dist'] = results['r_var_dist'].astype('object')
            r_var = stats.bootstrap_gpu(y_test[:, roi_indices],
                                        y_hat[:, roi_indices],
                                        verbose=True).cpu().detach().numpy().T.tolist()
            for idx, r_var_val in zip(roi_indices, r_var):
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
                                   run_stats=True,
                                   alphas=[10. ** power for power in np.arange(-5, 2)]):
    # use a CUDA-capable device, if available, else: CPU
    print(cuda_device_report())

    # initialize pipe and kfold splitter
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    score_func = get_scoring_method('pearsonr')

    # divide responses
    indices = {'train': benchmark.stimulus_data[benchmark.stimulus_data.stimulus_set == 'train'].index,
               'test': benchmark.stimulus_data[benchmark.stimulus_data.stimulus_set == 'test'].index}
    y_train = torch.from_numpy(benchmark.response_data.to_numpy().T[indices['train']]).to(torch.float32).to(devices[-1])

    print('\n\n\n\n\nStarting CV in the training set')
    layer_index = 0  # keeps track of depth
    scores_train_max = None
    results = []
    extractor_iterator = tqdm(feature_extractor, desc='Extractor Steps')
    for feature_maps in extractor_iterator:
        feature_map_iterator = tqdm(feature_maps.items(), desc='CV Mapping Layer', leave=False)
        for feature_map_uid, feature_map in feature_map_iterator:
            layer_index += 1  # one layer deeper in feature_maps

            # reduce dimensionality of feature_maps by sparse random projection
            feature_map = get_feature_map_srps(feature_map[indices['train']], device=devices[-1])
            X_train = feature_map.detach().clone().squeeze().to(torch.float32).to(devices[-1])

            # Memory saving
            del feature_map
            gc.collect()
            torch.cuda.empty_cache()
            pipe = TorchRidgeGCV(alphas=alphas, alpha_per_target=True,
                                 device=devices[-1], scale_X=False)

            try:
                #### Fit CV in the train set ####
                y_cv_pred, y_cv_true = [], []  # Initialize lists
                for i, (cv_train_index, cv_test_index) in enumerate(cv.split(X_train)):
                    # Split the training set
                    X_cv_train, X_cv_test = X_train[cv_train_index], X_train[cv_test_index]
                    y_cv_train, y_cv_test = y_train[cv_train_index], y_train[cv_test_index]

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
                del X_train, X_cv_train, X_cv_test
                del y_cv_train, y_cv_test
                gc.collect()
                torch.cuda.empty_cache()
            except:
                print(
                    f'\nFitting failed to converge for {model_name} {feature_map_uid} ({layer_index + layer_index_offset})')

    # Add training data to a dataframe
    results = benchmark.metadata.copy()
    results['layer_index'] = model_layer_index_max
    results['layer_relative_depth'] = model_layer_index_max / (layer_index + layer_index_offset)
    results['layer'] = model_layer_max
    results['train_score'] = scores_train_max
    results['model_uid'] = model_name

    # Free up memory
    memory_stats(devices)
    y_hat_max = torch.zeros_like(y_cv_hat)
    del y_train, y_cv_hat
    gc.collect()
    torch.cuda.empty_cache()
    memory_stats(devices)

    if test_eval:
        print('\n\n\n\n\nRunning evaluation in the test set')
        print('resetting y')
        y_train = torch.from_numpy(benchmark.response_data.to_numpy().T[indices['train']]).to(torch.float32).to(
            devices[-1])
        y_test = torch.from_numpy(benchmark.response_data.to_numpy().T[indices['test']]).to(torch.float32).to(
            devices[-1])
        if scale_y:
            y_train, y_test = feature_scaler(y_train, y_test)

        scores_test_max = np.zeros_like(scores_train_max)
        layer_index = 0
        extractor_iterator = tqdm(feature_extractor, desc='Extractor Steps')
        for feature_maps in extractor_iterator:
            feature_map_iterator = tqdm(feature_maps.items(), desc='Testing Mapping Layer', leave=False)
            for feature_map_uid, feature_map in feature_map_iterator:
                layer_index += 1  # one layer deeper in feature_maps

                # reduce dimensionality of feature_maps by sparse random projection
                feature_map = get_feature_map_srps(feature_map, device=devices[-1])
                X = feature_map.detach().clone().squeeze().to(torch.float32).to(devices[-1])
                X_train, X_test = feature_scaler(X[indices['train']], X[indices['test']])

                # Memory saving
                del feature_map, X
                torch.cuda.empty_cache()
                pipe = TorchRidgeGCV(alphas=alphas, alpha_per_target=True,
                                     device=devices[-1], scale_X=False)

                try:
                    pipe.fit(X_train, y_train)
                    y_hat = pipe.predict(X_test)
                    scores_test = score_func(y_hat, y_test).cpu().detach().numpy()

                    # Save the test set scores to an array only if it is where performance was maximum in the training set
                    idx = model_layer_index_max == (layer_index + layer_index_offset)
                    scores_test_max[idx] = scores_test[idx]
                    y_hat_max[:, idx] = y_hat[:, idx]

                    # Memory saving
                    del pipe, scores_test
                    del X_train, X_test
                    gc.collect()
                    torch.cuda.empty_cache()
                except:
                    print(
                        f'\nFitting failed to converge for {model_name} {feature_map_uid} ({layer_index + layer_index_offset})')

        # Add test set results to the dataframe
        results['test_score'] = scores_test_max
        results['r_null_dist'] = np.nan
        results['r_var_dist'] = np.nan
        results['r_null_dist'] = results['r_null_dist'].astype('object')
        results['r_var_dist'] = results['r_var_dist'].astype('object')

        if run_stats:
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


def get_lm_encoded_training_benchmarking_results(benchmark, feature_map, device='cuda',
                                                 n_splits=4, random_seed=0,
                                                 alphas=[10. ** power for power in np.arange(-5, 2)]):
    # use a CUDA-capable device, if available, else: CPU
    print(f'{device=}')
    print(f'{cuda_device_report()}')

    # initialize pipe and kfold splitter
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    score_func = get_scoring_method('pearsonr')
    pipe = TorchRidgeGCV(alphas=alphas, alpha_per_target=True,
                         device=device, scale_X=True, )

    # Get X
    feature_map = get_feature_map_srps(feature_map, device=device)
    X = feature_map.detach().clone().squeeze().to(torch.float32).to(device)
    print(f'{X.shape=}')

    # Send the neural data to the GPU
    y = torch.from_numpy(benchmark.response_data.to_numpy().T).to(torch.float32).to(device)
    print(f'{y.shape=}')

    y_pred, y_true = [], []  # Initialize lists
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


def get_rsa_benchmark_results(benchmark, feature_extractor,
                              layer_index_offset=0,
                              metrics=['ersa', 'crsa'],
                              rdm_distance='pearson',
                              rsa_distance='spearman',
                              score_types=['spearmanr'],
                              format_final_results=True,
                              save_raw_results=True,
                              raw_output_file=None,
                              feature_map_stats={},
                              model_uid=None,
                              alpha_values=np.logspace(-1, 5, 7).tolist(),
                              device='cuda:0',
                              k_folds=4,
                              test_eval=True,
                              input_modal='images'):
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
        format_final_results (bool): Defaults to True, if results should be formatted, grouped and averaged for plotting
        save_raw_results (bool): Defaults to True, if results pre-formatting should be saved.
        raw_out_file (string, optional): The path to save the raw unformatted results
        feature_map_stats (dict, optional): Precomputed statistics of feature maps for normalization. Defaults to None.
        model_uid (str, optional): UID of the neural network model being benchmarked. Defaults to None.
        alpha_values (list, optional): List of alpha values for Ridge regression cross-validation.
        device (str, optional): Computation device ('cuda' or 'cpu'). Defaults to 'cuda'.
        k_folds (int, optional): Number of splits for k-fold cross-validation. Defaults to 5.
        test_eval (bool): Defaults to True, if we want to include the test set score.

    Returns:
        pandas.DataFrame or dict: The RSA benchmarking results. If `save_raw_results` is True, returns a single DataFrame
        containing the results; otherwise, returns a dictionary of DataFrames, one for each RSA metric.
    """

    # HELPER FUNCTIONS #
    def generate_fold_indices(k=4, benchmark=None):
        indices = {'train_set': benchmark.stimulus_data[benchmark.stimulus_data['stimulus_set'] == 'train'].index,
                   'test_set': benchmark.stimulus_data[benchmark.stimulus_data['stimulus_set'] == 'test'].index}
        train_ind_splits = []
        test_ind_splits = []
        kf = KFold(n_splits=k, shuffle=True, random_state=1)
        for i, (train_index, test_index) in enumerate(kf.split(indices['train_set'])):
            ind_split = {'train': indices['train_set'][train_index], 'test': indices['train_set'][test_index]}
            train_ind_splits.append(ind_split)

        for i, (train_index, test_index) in enumerate(kf.split(indices['test_set'])):
            ind_split = {'train': indices['test_set'][train_index], 'test': indices['test_set'][test_index]}
            test_ind_splits.append(ind_split)
        return train_ind_splits, test_ind_splits

    def get_kfold_xy_rdms(feature_map, responses, ind_splits):
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

    def get_rdm_splits(rdms, ind_splits):
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

    def get_rdm_set_splits(rdms, set_indices):
        split_rdm = {}
        for roi_name in rdms:
            split_rdm[roi_name] = {}
            for subj_id in rdms[roi_name]:
                empty_train = []
                for row in rdms[roi_name][subj_id][set_indices['train']]:
                    empty_train.append(row[set_indices['train']])
                train_rdm = torch.stack(empty_train, dim=0)
                empty_test = []
                for row in rdms[roi_name][subj_id][set_indices['test']]:
                    empty_test.append(row[set_indices['test']])
                test_rdm = torch.stack(empty_test, dim=0)
                split_rdm[roi_name][subj_id] = {'train': train_rdm,
                                                'test': test_rdm}
        return split_rdm

    def run_for_test_set(df_results):
        test_sheets = {metric: [] for metric in ['ersa', 'crsa']}
        scaling = StandardScaler()

        for i, row in df_results.iterrows():
            feature_map_uid = row['model_layer']
            feature_map = feature_maps_pointer[feature_map_uid]
            # Create X and y
            feature_map = apply_tensor_op(feature_map, lambda x: x.to('cpu'))
            x_train = scaling.fit_transform(feature_map[set_indices['train']])
            x_test = scaling.transform(feature_map[set_indices['test']])
            # responses splits
            y_train = Y.T[set_indices['train']]
            y_test = Y.T[set_indices['test']]
            # get the overall rdms sets
            set_rdms = get_rdm_set_splits(benchmark.rdms, set_indices)
            # now, our X Variable that we push onto the gpu:
            X = {'train': convert_to_tensor(x_train).to(dtype=torch.float32, device='cuda:0'),
                 'test': convert_to_tensor(x_test).to(dtype=torch.float32, device='cuda:0')}
            # create the y object
            y = {'train': y_train,
                 'test': y_test}
            # perform regression
            regression = TorchRidgeGCV(alphas=alpha_values, device='cuda:0', scale_X=True)
            regression.fit(X['train'], y['train'])
            y_pred = {'train': regression.cv_y_pred_, 'test': regression.predict(X['test'])}

            # loop over cRSA, eRSA...
            metric = row['metric']
            region = row['roi_name']
            # encoding RSA score
            if metric == 'ersa':
                sub_scores = []
                for subj_id in benchmark.rdms[region]:
                    target_rdm = set_rdms[region][subj_id]['test']
                    # get the response_indices for current ROI group
                    response_indices = roi_indices[region][subj_id]
                    # get predicted values for each response_index...
                    y_pred_i = y_pred['test'][:, response_indices]
                    # ... and use them to calculate the weighted RDM
                    model_rdm = compute_rdm(y_pred_i, rdm_distance)
                    # compare brain-reweighted model RDM to brain RDM
                    # with our specified 2nd-order distance metric...
                    score = compare_rdms(model_rdm, target_rdm, method=rsa_distance)
                    sub_scores.append(score)
                score = np.mean(sub_scores)
                # add the scores to a "scoresheet"
                scoresheet = {'model_layer': feature_map_uid,
                              'roi_name': region,
                              'test_score': score}
                # append the scoresheet to our running list
                test_sheets['ersa'].append(scoresheet)

            # classic RSA score
            elif metric == 'crsa':
                # get the relevant train-test split of the model RDM
                model_rdm = compute_rdm(X['test'], method=rdm_distance, device=device)
                for subj_id in benchmark.rdms[region]:
                    # get the relevant train-test split of the brain RDM
                    target_rdm = set_rdms[region][subj_id]['test']
                    # compare lower triangles of model + brain RDM
                    # with our specified 2nd-order distance metric
                    score = compare_rdms(model_rdm, target_rdm, method=rsa_distance)
                    sub_scores.append(score)
                score = np.mean(sub_scores)
                # add the scores to a "scoresheet"
                scoresheet = {'model_layer': feature_map_uid,
                              'roi_name': region,
                              'test_score': score}
                # append the scoresheet to our running list
                test_sheets['crsa'].append(scoresheet)

        # clean up tensors on gpu
        X = {key: tensor.to('cpu') for key, tensor in X.items()}
        del regression
        gc.collect()
        torch.cuda.empty_cache()

        results = {metric: pd.DataFrame(scores) for metric, scores in test_sheets.items()}
        results_list = []
        for metric, result in results.items():
            result.insert(0, 'metric', metric)
            results_list.append(result)
        results = pd.concat(results_list)

        df_results = df_results.merge(results, on=['model_layer', 'roi_name', 'metric'], how='left')
        return df_results

    feature_maps_device = None
    Y = (convert_to_tensor(benchmark.response_data.to_numpy()).to(dtype=torch.float32, device=device))

    # record key information about each method for reference
    method_info = {'regression': {'encoding_model': 'ridge'},
                   'rsa': {'rdm_distance': rdm_distance,
                           'rsa_distance': rsa_distance}}

    # initialize an empty list to record scores over layers
    scoresheet_lists = {metric: [] for metric in metrics}

    # get the voxel (neuroid) indices for each specified roi
    roi_indices = benchmark.row_indices

    # initialize a dictionary of scoring metrics to apply to the predicted outputs
    score_funcs = {score_type: get_scoring_method(score_type) for score_type in score_types}
    layer_index = 0  # keeps track of depth

    set_indices = {'train': benchmark.stimulus_data[benchmark.stimulus_data.stimulus_set == 'train'].index,
                   'test': benchmark.stimulus_data[benchmark.stimulus_data.stimulus_set == 'test'].index}

    train_ind_splits, test_ind_splits = generate_fold_indices(k=k_folds, benchmark=benchmark)
    train_rdm_splits = get_rdm_splits(benchmark.rdms, train_ind_splits)

    feature_maps_pointer = []

    # now, we iterate over our extractor
    for feature_maps in feature_extractor:
        # dimensionality reduction of feature maps
        feature_maps = get_feature_map_srps(feature_maps, device='cuda:0')
        # save a pointer to the feature_maps
        feature_maps_pointer.append(feature_maps)
        # now, we loop over our batch of feature_maps from the extractor...
        # ...starting by defining an iterator that will track our progress
        feature_map_iterator = tqdm(feature_maps.items(), desc='Train Brain Mapping (Layer)')

        for feature_map_uid, feature_map in feature_map_iterator:
            # index the 5 fold splits for this layer
            xy_folds = get_kfold_xy_rdms(feature_map, Y, train_ind_splits)

            layer_index += 1  # one layer deeper in feature_maps

            # loop over each fold in the kfold
            for i, fold in enumerate(xy_folds):
                # main data to add to our scoresheet per feature_map
                feature_map_info = {'model_uid': model_uid,
                                    'model_layer': feature_map_uid,
                                    # layer_index_offset is used here in case of subsetting
                                    'model_layer_index': layer_index + layer_index_offset,
                                    'k_fold': i + 1}

                # now, our X Variable that we push onto the gpu:
                X = {'train': convert_to_tensor(fold['train']['X']).to(dtype=torch.float32, device='cuda:0'),
                     'test': convert_to_tensor(fold['test']['X']).to(dtype=torch.float32, device='cuda:0')}

                y = {'train': fold['train']['y'],
                     'test': fold['test']['y']}

                # initialize the regression, in this case ridge regression with LOOCV over alphas
                regression = TorchRidgeGCV(alphas=alpha_values, device='cuda:0', scale_X=True)
                regression.fit(X['train'], y['train'])  # fit the regression on the train split
                # RidgeGCV gives us both internally generated LOOCV values for the train dataset
                # as well as the ability to predict our test set in the same way as any regressor
                y_pred = {'train': regression.cv_y_pred_, 'test': regression.predict(X['test'])}

                # loop over cRSA, SRPR, eRSA...
                for metric in scoresheet_lists:

                    # encoding RSA score
                    if metric == 'ersa':
                        for split in ['train', 'test']:
                            for region in benchmark.rdms:
                                for subj_id in benchmark.rdms[region]:
                                    # get the relevant train-test split of the brain RDM
                                    target_rdm = train_rdm_splits[i][region][subj_id][split]
                                    # get the response_indices for current ROI group
                                    response_indices = roi_indices[region][subj_id]
                                    # get predicted values for each response_index...
                                    y_pred_i = y_pred[split][:, response_indices]
                                    # ... and use them to calculate the weighted RDM
                                    model_rdm = compute_rdm(y_pred_i, rdm_distance)
                                    # compare brain-reweighted model RDM to brain RDM
                                    # with our specified 2nd-order distance metric...
                                    score = compare_rdms(model_rdm, target_rdm,
                                                         method=rsa_distance)
                                    # add the scores to a "scoresheet"
                                    scoresheet = {**feature_map_info,
                                                  'region': region,
                                                  'subj_id': subj_id,
                                                  'cv_split': split,
                                                  'score': score,
                                                  # **aux_stats[split],
                                                  **method_info['rsa']}
                                    # append the scoresheet to our running list
                                    scoresheet_lists['ersa'].append(scoresheet)

                    # classic RSA score
                    elif metric == 'crsa':
                        for split in ['train', 'test']:
                            # get the relevant train-test split of the model RDM
                            model_rdm = compute_rdm(X[split], method=rdm_distance, device=device)
                            for region in benchmark.rdms:
                                for subj_id in benchmark.rdms[region]:
                                    # get the relevant train-test split of the brain RDM
                                    target_rdm = train_rdm_splits[i][region][subj_id][split]
                                    # compare lower triangles of model + brain RDM
                                    # with our specified 2nd-order distance metric
                                    score = compare_rdms(model_rdm, target_rdm,
                                                         method=rsa_distance)
                                    # add the scores to a "scoresheet"
                                    scoresheet = {**feature_map_info,
                                                  'region': region,
                                                  'subj_id': subj_id,
                                                  'cv_split': split,
                                                  'score': score,
                                                  # **aux_stats[split],
                                                  **method_info['rsa']}
                                    # append the scoresheet to our running list
                                    scoresheet_lists['crsa'].append(scoresheet)

                # clean up tensors on gpu
                X = {key: tensor.to('cpu') for key, tensor in X.items()}
                del regression
                gc.collect()
                torch.cuda.empty_cache()

    feature_maps_pointer = {k: v for d in feature_maps_pointer for k, v in d.items()}

    results = {metric: pd.DataFrame(scores) for metric, scores in scoresheet_lists.items()}
    result_columns = pd.unique([col for results in results.values() for col in results.columns]).tolist()
    common_columns = [col for col in result_columns if all(col in result.columns for result in results.values())]
    common_columns = ['metric'] + common_columns  # indicator
    results_list = []
    for metric, result in results.items():
        result.insert(0, 'metric', metric)
        results_list.append(result[common_columns])
    results = pd.concat(results_list)

    if save_raw_results and raw_output_file:
        print(f'Saving raw results to {raw_output_file}...')
        results.to_csv(raw_output_file, index=False)

    if format_final_results:
        print('Formatting results...')
        df_all_models = get_model_options() if input_modal == 'images' else None
        formatted_results = []
        for metric in results['metric'].unique():
            df_metric = results[results['metric'] == metric]
            # avg subject score in each fold
            df_metric = df_metric.groupby(['model_layer', 'model_layer_index', 'region', 'subj_id']).mean(
                numeric_only=True).reset_index()
            # avg subject
            df_metric = \
            df_metric.groupby(['model_layer', 'model_layer_index', 'region']).mean(numeric_only=True).reset_index()[
                ['model_layer', 'model_layer_index', 'region', 'score']]
            # find the best layers per ROI
            best_layers = {region: {} for region in df_metric['region'].unique()}
            for region in df_metric['region'].unique():
                df_region = df_metric[df_metric['region'] == region]
                best_layers[region]['model_layer_index'] = df_region.loc[df_region['score'].idxmax()][
                    'model_layer_index']
                best_layers[region]['model_layer'] = df_region.loc[df_region['score'].idxmax()]['model_layer']
                best_layers[region]['score'] = df_region.loc[df_region['score'].idxmax()]['score']
            df_metric = pd.DataFrame(best_layers).T.reset_index()
            # Sort and order the dataframe
            df_metric.rename(columns={'index': 'roi_name'}, inplace=True)
            custom_order = ['EVC', 'MT', 'LOC', 'EBA', 'pSTS', 'face-pSTS', 'aSTS', 'FFA', 'PPA']
            df_metric['roi_name'] = pd.Categorical(df_metric['roi_name'], categories=custom_order, ordered=True)
            df_metric = df_metric.sort_values(by='roi_name')
            # Populate more columns
            df_metric['layer_relative_depth'] = df_metric['model_layer_index'] / results['model_layer_index'].max()
            df_metric['model_uid'] = results['model_uid'].unique()[0]
            df_metric['metric'] = metric
            # Grab model metadata from deepjuice if images
            if input_modal == 'images':
                model_metadata = df_all_models[df_all_models['model_uid'] == results['model_uid'].unique()[0]][
                    ['model_uid', 'architecture_type', 'train_task', 'train_data', 'task_cluster', 'modality',
                     'display_name']]
                df_metric = df_metric.merge(model_metadata, on='model_uid', how='left')
            # add to the greater frame
            formatted_results.append(df_metric)
        formatted_results = pd.concat(formatted_results)
        if test_eval:
            print('Running test set...')
            formatted_results.rename(columns={'score': 'train_score'}, inplace=True)
            formatted_results = run_for_test_set(formatted_results)
            print('Finished test set')
        return formatted_results
    else:
        # return the raw results
        return results
