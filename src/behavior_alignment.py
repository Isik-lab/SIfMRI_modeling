#
from deepjuice.alignment import TorchRidgeGCV, get_scoring_method, compute_rdm, compare_rdms
import torch
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
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


def scale_features(train, test):
    scaler = StandardScaler()
    train_ = scaler.fit_transform(train)
    return train_, scaler.transform(test)


def get_training_benchmarking_results(benchmark, feature_extractor
                                      target_features,
                                      layer_index_offset=0,
                                      device='cuda:0',
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

    # divide responses
    y = {'train': benchmark.filter_stimulus(stimulus_set='train'), 
         'test': benchmark.filter_stimulus(stimulus_set='test')}
    y = {key: torch.from_numpy(val[target_features].to_numpy()).to(torch.float32).to(device) for key, val in y.items()}
    print(f'{y['train'].shape=}')
    print(f'{y['test'].shape=}')

    layer_index = 0 # keeps track of depth
    results = []
    for feature_maps in feature_extractor:
        feature_map_iterator = tqdm(feature_maps.items(), desc = 'Brain Mapping (Layer)', leave=False)
        for feature_map_uid, feature_map in feature_map_iterator:
            layer_index += 1 # one layer deeper in feature_maps

            # reduce dimensionality of feature_maps by sparse random projection
            feature_map = get_feature_map_srps(feature_map, device=device)
            feature_map_info = {'model_uid': model_name,
                                'model_layer': feature_map_uid,
                                'model_layer_index': layer_index + layer_index_offset}
            
            X = feature_map.detach().clone().squeeze().to(torch.float32)
            print(f'{X.shape=}')
            X = {'train': X[:200], 'test': X[200:]}
            del feature_map
            torch.cuda.empty_cache()

            # Memory saving
            pipe = TorchRidgeGCV(alphas=alphas, alpha_per_target=True, 
                                 device=device, scale_X=False)

            #### Fit CV in the train set ####
            y_pred, y_true = [], [] #Initialize lists
            for i, (cv_train_index, cv_test_index) in enumerate(cv.split(X['train'])):
                # Split the training set
                X_cv_train, X_cv_test = X['train'][cv_train_index].detach().clone(), X['train'][cv_test_index].detach().clone()
                y_cv_train, y_cv_test = y['train'][cv_train_index].detach().clone(), y['train'][cv_test_index].detach().clone()

                # Scale X and y
                X_cv_train, X_cv_test = scale_features(X_cv_train, X_cv_test)
                y_cv_train, y_cv_test = scale_features(y_cv_train, y_cv_test)

                # Fit the regression
                pipe.fit(X_cv_train, y_cv_train)
                y_cv_pred.append(pipe.predict(X_cv_test))
                y_cv_true.append(y_cv_test)
            scores_train = score_func(torch.cat(y_cv_pred), torch.cat(y_cv_true)) # Get the CV training scores 

            #### Fit the train/test split ####
            # Scale X and y
            X_train, X_test = scale_features(X['train'].detach().clone(), X['test'].detach().clone())
            y_train, y_test = scale_features(y['train'].detach().clone(), y['test'].detach().clone())

            # Fit the regression
            pipe.fit(X_train, y_train)
            scores_test = score_func(pipe.predict(X_test), y_test)

            # Add performance to the score sheet
            for target_feature, score_train, score_test in zip(target_features,
                                                               scores_train.cpu().detach().numpy(),
                                                               scores_test.cpu().detach().numpy()): 
                # add the scores to a "scoresheet"
                results.append({**feature_map_info,
                                'feature': target_feature,
                                'train_score': score_train,
                                'test_score': score_test})

            # Memory saving
            del pipe, scores_train, scores_test,
                X_cv_train, X_cv_test, y_cv_pred, y_cv_true,
                X_train, X_test, y_train, y_test
            gc.collect()
            torch.cuda.empty_cache()
    return pd.DataFrame(results)