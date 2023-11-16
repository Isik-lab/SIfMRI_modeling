#
from deepjuice.alignment import TorchRidgeGCV, get_scorer
import torch
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, TensorDataset
from deepjuice.extraction import FeatureExtractor
from deepjuice.reduction import get_feature_map_srps


def load_model(model_uid):
    model_ = AutoModel.from_pretrained(model_uid)
    tokenizer_ = AutoTokenizer.from_pretrained(model_uid)
    return model_, tokenizer_


def tokenize_captions(tokenizer_, captions_):
    return tokenizer_(captions_, return_tensors='pt', padding='max_length')


def moving_grouped_average(outputs, input_dim=0, skip=5):
    return torch.stack([outputs[i*skip:i*skip+skip].mean(dim=input_dim) 
                        for i in range(outputs.size(0) // skip)])


def memory_saving_extraction(model_uid, captions):
    model, tokenizer = load_model(model_uid)
    tokenized_captions = tokenize_captions(tokenizer, captions)
    tensor_dataset = TensorDataset(tokenized_captions['input_ids'],
                                    tokenized_captions['attention_mask'])
    dataloader = DataLoader(tensor_dataset, batch_size = 100)
    feature_extractor = FeatureExtractor(model, dataloader, remove_duplicates=False,
                                        keep=['Attention','BertModel'],
                                        tensor_fn=moving_grouped_average,
                                        sample_size=5, reduce_size_by=5,
                                        output_device='cpu', exclude_oversize=False)
    feature_extractor.modify_settings(flatten=True)
    return feature_extractor


def get_benchmarking_results(benchmark, feature_extractor, device,
                             layer_index_offset=0,
                             verbose=True, n_splits=4):
    # initialize pipe and kfold splitter
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    alphas = [10.**power for power in np.arange(-5, 2)]
    score_func = get_scorer('pearsonr')
    pipe = TorchRidgeGCV(alphas=alphas, alpha_per_target=True,
                            device=device, scale_X=True,)

    layer_index = 0 # keeps track of depth
    results = []
    for feature_maps in feature_extractor:
        if verbose:
            feature_map_iterator = tqdm(feature_maps.items(), desc = 'Brain Mapping (Layer)', leave=False)
        else:
            feature_map_iterator = feature_maps.items()

        for feature_map_uid, feature_map in feature_map_iterator:
            layer_index += 1 # one layer deeper in feature_maps

            # reduce dimensionality of feature_maps by sparse random projection
            feature_map = get_feature_map_srps(feature_map, device=device)
        
            # main data to add to our scoresheet per feature_map
            feature_map_info = {'model_layer': feature_map_uid, 
                                # layer_index_offset is used here in case of subsetting
                                'model_layer_index': layer_index + layer_index_offset}
            
            X = feature_map.squeeze().to(torch.float32).to(device)
            y = benchmark.response_data.to_numpy().T

            #send X and y to torch
            X = torch.from_numpy(X).to(torch.float32).to(device)
            y = torch.from_numpy(y).to(torch.float32).to(device)

            y_pred = []
            y_true = []

            if verbose:
                cv_iterator = tqdm(cv.split(X), desc='CV', total=n_splits)
            else:
                cv_iterator = cv.split(X)

            for train_index, test_index in cv_iterator:
                pipe.fit(X[train_index], y[train_index])
                y_pred.append(pipe.predict(X[test_index]))
                y_true.append(y[test_index])
            
            scores = score_func(torch.cat(y_pred), torch.cat(y_true))
            scores = scores.cpu().detach().numpy() #send to CPU
            print(f'scores shape {scores.shape}')

            for region in benchmark.metadata.stream_name.unique():
                for subj_id in benchmark.metadata.subj_id.unique():
                    voxel_id = benchmark.metadata.loc[(benchmark.metadata.subj_id == subj_id) &
                                                    (benchmark.metadata.stream_name == region), 'voxel_id'].to_numpy()
                    results.append({**feature_map_info,
                                    'stream_name': region,
                                    'subj_id': subj_id,
                                    'score': np.mean(scores[voxel_id]),
                                    'method': 'ridge'})
    return pd.DataFrame(results)