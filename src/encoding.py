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


def run_visual_event_pipeline(model_uid, benchmark, device, **kwargs):

    model, preprocess = get_deepjuice_model(model_uid)
    
    response_data = benchmark['response_data']
    image_paths = benchmark['image_paths']
    group_index = benchmark['group_indices']
    
    target_names = response_data.columns.tolist()
    n_inputs = len(response_data) # unique total

    dataloader = get_data_loader(image_paths, preprocess)
    extractor_desc = 'Global Progress (Extractor Batch)'
    
    scoresheet_list = [] # append scoresheets by layer / metric
    method_info = {'regression': {'encoding_model': 'RidgeCV'},
                   'cvfunction': {'method': '10-iter-5-fold'}}

    sample_group_index = list(group_index.values())[0]
    average_over_nmany = len(sample_group_index)

    stimulus_info = {'frame_set': None}

    if average_over_nmany >= 1:
        tensor_fn = None # pass
        
        stimulus_info['frame_set'] = 'middle_frame'

    if average_over_nmany >= 2:
        skip = average_over_nmany

        stimulus_info['frame_set'] = f'average_of_{skip}'

        def tensor_fn(tensor):
            return moving_grouped_average(tensor, skip)

        batch_size = dataloader.batch_size
        if kwargs.get('batch_size', None):
            batch_size = kwargs.pop('batch_size')
        
        batch_size = get_nearest_multiple(batch_size, skip)
        
        dataloader = get_data_loader(image_paths, preprocess,
                                     batch_size = batch_size)

    method_info['stimulus_set'] = stimulus_info.copy()

    extractor = FeatureExtractor(model, dataloader, 
                                 tensor_fn=tensor_fn,
                                 n_inputs=n_inputs,
                                 initial_report=False)
    
    extractor.modify_settings(flatten=True, batch_progress=True)
        
    cv_indexer = CVIndexer(200, iterations=10, random_state=0,
                           iterable_format='list')
    
    cv_iter_idx = cv_indexer.kfold_split(kfolds=5) # get kfolds

    global_srp_matrix = extractor.get_global_srp_matrix()
    global_srp_on_gpu = global_srp_matrix.clone().to(device)

    y_actual = torch.from_numpy(response_data.to_numpy())
    y_actual = y_actual.to(torch.float32).to(device) 
    y_cvsplit, y_heldout = y_actual[:200], y_actual[200:]
    
    regression = TorchRidgeGCV(alphas=np.logspace(-1,5,7).tolist(), 
                               device=device, scale_X=True)

    shape_report = kwargs.pop('print_shapes', False) # for debug

    scoresheet_list = [] # fill with results from each feature map

    for batch_index, feature_maps in enumerate(tqdm(extractor, desc=extractor_desc)):

        feature_map_iterator = tqdm(feature_maps.items(), desc='Social Event Annotation (Layer)')

        for layer_index, (model_layer, feature_map) in enumerate(feature_map_iterator):
            feature_map_info = {'model_uid': model_uid, 'model_layer': model_layer, 
                                'model_layer_index': layer_index+1}


            srp_kwargs = {'device': device, 'srp_matrix': global_srp_on_gpu}
            feature_map = compute_srp(feature_map, **srp_kwargs)


            feature_map = feature_map.squeeze().to(torch.float32).to(device)
        
            for cv_iter, kfold_split_idx in enumerate(cv_iter_idx):
                y_pred = torch.ones(y_cvsplit.shape, device=device)
                
                for kfold, cv_split_idx in kfold_split_idx.items():
                    X, y = {}, {} # fill with split + cv_idx
                    for split, cv_idx in cv_split_idx.items():
                        X[split] = feature_map[cv_idx, :]
                        y[split] = y_cvsplit[cv_idx]

                    if shape_report:
                        print(list(X['train'].shape), list(X['test'].shape),
                              list(y['train'].shape), list(y['test'].shape))
        
                    regression.fit(X['train'], y['train'])
                    
                    y_preds = {'train': regression.cv_y_pred_, 
                               'test': regression.predict(X['test'])}
        
                    y_pred[cv_split_idx['test']] = y_preds['test']
        
                for score_type in ['pearsonr']:
                    y_true = y_cvsplit.clone()
                    
                    scores = compute_score(y_true, y_pred, score_type)
                    
                    for target_index, target_name in enumerate(target_names):
                        score_val = scores[target_index].item()
                        
                        scoresheet = {**feature_map_info,
                                      'target': target_name,
                                      'cv_iter': cv_iter,
                                      'score': score_val}

                        for info_type in method_info:
                            scoresheet = {**scoresheet, **method_info[info_type]}
        
                        scoresheet_list.append(scoresheet)

    return pd.DataFrame(scoresheet_list)