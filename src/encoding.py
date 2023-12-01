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
from deepjuice.tensorfy import get_device_name
from sentence_transformers import SentenceTransformer
from src import stats


def load_llm(model_uid):
    model_ = AutoModel.from_pretrained(model_uid)
    return model_


def load_glove():
    model = SentenceTransformer('sentence-transformers/average_word_embeddings_glove.6B.300d')
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    else:
        print('torch error')
    return model


def glove_feature_extraction(captions):
    model = load_glove()
    return model.encode(captions)


def moving_grouped_average(outputs, input_dim=0, skip=5):
    return torch.stack([outputs[i*skip:i*skip+skip].mean(dim=input_dim) 
                        for i in range(outputs.size(0) // skip)])


def tokenize_captions(model_uid, captions):
    if 'gpt' not in model_uid:
        tokenizer = AutoTokenizer.from_pretrained(model_uid)
        return tokenizer(captions, return_tensors='pt', padding='max_length')
    else: 
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(model_uid)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer(captions, return_tensors='pt', padding=True, truncation=True)


def memory_saving_extraction(model_uid, captions):
    model = load_llm(model_uid)
    tokenized_captions = tokenize_captions(model_uid, captions)
    print(tokenized_captions)
    tensor_dataset = TensorDataset(tokenized_captions['input_ids'],
                                    tokenized_captions['attention_mask'])
    dataloader = DataLoader(tensor_dataset, batch_size = 20)
    feature_extractor = FeatureExtractor(model, dataloader, remove_duplicates=True,
                                         memory_limit='15 GB',
                                        # keep=['Attention','BertModel'],
                                        tensor_fn=moving_grouped_average,
                                        sample_size=5, reduce_size_by=5,
                                        output_device='cuda:1', exclude_oversize=False)
    feature_extractor.modify_settings(flatten=True)
    return feature_extractor


def get_training_benchmarking_results(benchmark, feature_extractor,
                                      layer_index_offset=0,
                                      device='auto',
                                      n_splits=4, random_seed=0):
    # use a CUDA-capable device, if available, else: CPU
    if device == 'auto': device = get_device_name(device, index=2)
    print(f'device: {device}')

    # initialize pipe and kfold splitter
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    alphas = [10.**power for power in np.arange(-5, 2)]
    score_func = get_scorer('pearsonr')
    pipe = TorchRidgeGCV(alphas=alphas, alpha_per_target=True,
                            device=device, scale_X=True,)

    layer_index = 0 # keeps track of depth
    results = []
    for feature_maps in feature_extractor:
        feature_map_iterator = tqdm(feature_maps.items(), desc = 'Brain Mapping (Layer)', leave=False)
        for feature_map_uid, feature_map in feature_map_iterator:
            layer_index += 1 # one layer deeper in feature_maps

            # reduce dimensionality of feature_maps by sparse random projection
            feature_map = get_feature_map_srps(feature_map, device=device)
        
            # main data to add to our scoresheet per feature_map
            feature_map_info = {'model_layer': feature_map_uid, 
                                # layer_index_offset is used here in case of subsetting
                                'model_layer_index': layer_index + layer_index_offset}
            
            X = feature_map.squeeze().to(torch.float32).to(device)
            y = torch.from_numpy(benchmark.response_data.to_numpy().T).to(torch.float32).to(device)

            y_pred = []
            y_true = []

            cv_iterator = tqdm(cv.split(X), desc='CV', total=n_splits)
            for train_index, test_index in cv_iterator:
                pipe.fit(X[train_index], y[train_index])
                y_pred.append(pipe.predict(X[test_index]))
                y_true.append(y[test_index])
            
            y_pred, y_true = torch.cat(y_pred, dim=0), torch.cat(y_true, dim=0)
            print(f'y_pred shape: {y_pred.shape}')
            scores = score_func(y_pred, y_true)

            results.append({**feature_map_info,
                            'scores': scores.cpu().detach().numpy(), #send to CPU
                            'y_pred': y_pred.cpu().detach().numpy(),
                            'y_true': y_true.cpu().detach().numpy()})
    return results


def get_glove_training_benchmarking_results(benchmark, feature_map,
                                      device='auto',
                                      n_splits=4, random_seed=0):
    # use a CUDA-capable device, if available, else: CPU
    if device == 'auto': device = get_device_name(device)
    print(f'device: {device}')

    # initialize pipe and kfold splitter
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    alphas = [10.**power for power in np.arange(-5, 2)]
    pipe = TorchRidgeGCV(alphas=alphas, alpha_per_target=True,
                            device=device, scale_X=True,)

            
    X = torch.from_numpy(feature_map).to(torch.float32).to(device)
    y = torch.from_numpy(benchmark.response_data.to_numpy().T).to(torch.float32).to(device)

    y_pred = []
    y_true = []
    cv_iterator = tqdm(cv.split(X), desc='CV', total=n_splits)
    for train_index, test_index in cv_iterator:
        pipe.fit(X[train_index], y[train_index])
        y_pred.append(pipe.predict(X[test_index]))
        y_true.append(y[test_index])
    
    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)
    print(f'y_pred shape: {y_pred.shape}')
    scores, null_scores = stats.perm_gpu(y_pred, y_true)

    results = []
    for region in benchmark.metadata.stream_name.unique():
        for subj_id in benchmark.metadata.subj_id.unique():
            voxel_id = benchmark.metadata.loc[(benchmark.metadata.subj_id == subj_id) &
                                            (benchmark.metadata.stream_name == region), 'voxel_id'].to_numpy()
            results.append({'stream_name': region,
                            'subj_id': subj_id,
                            'score': torch.mean(scores[voxel_id]).cpu().detach().numpy(),
                            'score_null': torch.mean(null_scores[voxel_id], dim=0).cpu().detach().numpy(),
                            'method': 'ridge'})
    return pd.DataFrame(results)
