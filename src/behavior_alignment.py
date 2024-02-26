#
from deepjuice.alignment import TorchRidgeGCV, get_scoring_method, compute_rdm, compare_rdms
import torch
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
import pandas as pd
import gc
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
from itertools import product


def get_training_benchmarking_results(benchmark, feature_extractor,
                                      target_features,
                                      layer_index_offset=0,
                                      device='cuda:0',
                                      metrics=['ersa', 'crsa', 'encoding'],
                                      n_splits=4, random_seed=0,
                                      model_name=None,
                                      alphas=[10.**power for power in np.arange(-5, 2)]):

    # use a CUDA-capable device, if available, else: CPU
    print(f'device: {device}')
    print(cuda_device_report())

    # initialize pipe and kfold splitter
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    alphas = [10.**power for power in np.arange(-5, 2)]
    encoding_score_func = get_scoring_method('pearsonr')

    # Send the neural data to the GPU
    y = torch.from_numpy(benchmark.stimulus_data[target_features].to_numpy()).to(torch.float32).to(device)
    print(f'{benchmark.stimulus_data[target_features].to_numpy().shape=}')
    print(f'{y.shape=}')

    layer_index = 0 # keeps track of depth
    results = []
    for feature_maps in feature_extractor:
        feature_map_iterator = tqdm(feature_maps.items(), desc = 'Brain Mapping (Layer)', leave=False)
        for feature_map_uid, feature_map in feature_map_iterator:
            layer_index += 1 # one layer deeper in feature_maps

            # reduce dimensionality of feature_maps by sparse random projection
            feature_map = get_feature_map_srps(feature_map, device=device)
            
            X = feature_map.detach().clone().squeeze().to(torch.float32)
            del feature_map
            torch.cuda.empty_cache()

            # Memory saving
            pipe = TorchRidgeGCV(alphas=alphas, alpha_per_target=True, 
                                 device=device, scale_X=True)

            for i, (train_index, test_index) in enumerate(cv.split(X)):
                feature_map_info = {'model_uid': model_name,
                                    'model_layer': feature_map_uid,
                                    'model_layer_index': layer_index + layer_index_offset,
                                    'k_fold': i+1}

                X_train, X_test = X[train_index].detach().clone(), X[test_index].detach().clone()
                y_train, y_test = y[train_index].detach().clone(), y[test_index].detach().clone()
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)

                X_test_rdm = compute_rdm(X_test, 'pearson')
                feature_zip = zip(target_features, y_pred.T, y_test.T)
                iter_scores = tqdm(product(metrics, feature_zip), total=len(metrics)*len(target_features), desc='Scoring CV')
                for metric, (target_feature, target_pred, target_true) in iter_scores: 
                    if metric == 'encoding':
                        score = encoding_score_func(target_pred, target_true).detach().cpu().numpy()
                    elif metric == 'ersa':
                        target_pred = target_pred.unsqueeze(1)
                        pred_rdm = compute_rdm(target_pred, 'euclidean')
                        score = compare_rdms(X_test_rdm, pred_rdm, method='spearman')
                    elif metric == 'crsa':
                        target_true = target_true.unsqueeze(1)
                        true_rdm = compute_rdm(target_true, 'euclidean')
                        score = compare_rdms(X_test_rdm, true_rdm, method='spearman')

                    # add the scores to a "scoresheet"
                    scoresheet = {**feature_map_info,
                                    'feature': target_feature,
                                    'score': score,
                                    'metric': metric}
                    results.append(scoresheet)

            # Memory saving
            del pipe, y_pred
            gc.collect()
            torch.cuda.empty_cache()

    return pd.DataFrame(results)