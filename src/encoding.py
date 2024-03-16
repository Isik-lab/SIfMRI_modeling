import gc
from deepjuice.alignment import TorchRidgeGCV, get_scoring_method, compute_rdm, compare_rdms
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
from sklearn.preprocessing import StandardScaler
from deepjuice.tensorops import apply_tensor_op
from deepjuice.tensorops import convert_to_tensor


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
    Y = (convert_to_tensor(benchmark.response_data.to_numpy()).to(dtype=torch.float32, device=device))

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
                X = {'train': convert_to_tensor(fold['train']['X']).to(dtype=torch.float32, device='cuda'),
                     'test': convert_to_tensor(fold['test']['X']).to(dtype=torch.float32, device='cuda')}

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
