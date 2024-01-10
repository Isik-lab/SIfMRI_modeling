#
from deepjuice.alignment import TorchRidgeGCV, get_scoring_method
import torch
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, TensorDataset
from deepjuice.extraction import FeatureExtractor
from deepjuice.reduction import get_feature_map_srps
from deepjuice.systemops.devices import cuda_device_report
from sentence_transformers import SentenceTransformer


def load_llm(model_uid):
    model_ = AutoModel.from_pretrained(model_uid)
    tokenizer_ = AutoTokenizer.from_pretrained(model_uid)
    print(f'{tokenizer_.eos_token=}')
    print(f'{tokenizer_.eos_token_id=}')
    print(f'{tokenizer_.pad_token=}')
    print(f'{tokenizer_.pad_token_id=}')
    return model_, tokenizer_


def load_gpt():
    from transformers import GPT2TokenizerFast
    model_ = AutoModel.from_pretrained('gpt2')
    tokenizer_ = GPT2TokenizerFast.from_pretrained('gpt2')
    SPECIAL_TOKENS = {"bos_token": "<|endoftext|>", 
                      "eos_token": "<|endoftext|>",
                      "pad_token": "[PAD]",
                      "additional_special_tokens": ["[SYS]", "[USR]", "[KG]", "[SUB]", "[PRED]", "[OBJ]", "[TRIPLE]", "[SEP]", "[Q]","[DOM]", 'frankie_and_bennys', 'cb17dy']
                      }
    tokenizer_.add_special_tokens(SPECIAL_TOKENS)
    return model_, tokenizer_


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


def tokenize_captions(tokenizer_, captions_):
    tokenized_captions_ = tokenizer_(captions_, return_tensors='pt', padding='max_length') 
    print(f'{tokenized_captions_["input_ids"]=}')
    print(f'{tokenized_captions_["attention_mask"]=}')
    return tokenized_captions_


def moving_grouped_average(outputs, input_dim=0, skip=5):
    return torch.stack([outputs[i*skip:i*skip+skip].mean(dim=input_dim) 
                        for i in range(outputs.size(0) // skip)])


def memory_saving_extraction(model_uid, captions, device):
    if 'gpt2' not in model_uid:
        model, tokenizer = load_llm(model_uid)
    else:
        model, tokenizer = load_gpt()
    tokenized_captions = tokenize_captions(tokenizer, captions)
    tensor_dataset = TensorDataset(tokenized_captions['input_ids'],
                                    tokenized_captions['attention_mask'])
    print(f'{tensor_dataset=}')
    dataloader = DataLoader(tensor_dataset, batch_size = 20)
    print(f'{dataloader=}')
    feature_extractor = FeatureExtractor(model, dataloader, remove_duplicates=False,
                                        tensor_fn=moving_grouped_average,
                                        sample_size=5, reduce_size_by=5,
                                        output_device=device, exclude_oversize=False)
    feature_extractor.modify_settings(flatten=True)
    return feature_extractor


def get_training_benchmarking_results(benchmark, feature_extractor,
                                      file_path,
                                      layer_index_offset=0,
                                      device='cuda',
                                      n_splits=4, random_seed=0):
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


# def get_glove_training_benchmarking_results(benchmark, feature_map,
#                                       device='auto',
#                                       n_splits=4, random_seed=0):
#     # use a CUDA-capable device, if available, else: CPU
#     if device == 'auto': device = get_device_name(device)
#     print(f'device: {device}')

#     # initialize pipe and kfold splitter
#     cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
#     alphas = [10.**power for power in np.arange(-5, 2)]
#     score_func = get_scoring_method('pearsonr')
#     pipe = TorchRidgeGCV(alphas=alphas, alpha_per_target=True,
#                             device=device, scale_X=True,)

            
#     X = torch.from_numpy(feature_map).to(torch.float32).to(device)
#     y = torch.from_numpy(benchmark.response_data.to_numpy().T).to(torch.float32).to(device)

#     y_pred = []
#     y_true = []
#     cv_iterator = tqdm(cv.split(X), desc='CV', total=n_splits)
#     for train_index, test_index in cv_iterator:
#         pipe.fit(X[train_index], y[train_index])
#         y_pred.append(pipe.predict(X[test_index]))
#         y_true.append(y[test_index])
    
#     scores = score_func(torch.cat(y_pred), torch.cat(y_true))
#     scores = scores.cpu().detach().numpy() #send to CPU

#     results = []
#     for region in benchmark.metadata.stream_name.unique():
#         for subj_id in benchmark.metadata.subj_id.unique():
#             voxel_id = benchmark.metadata.loc[(benchmark.metadata.subj_id == subj_id) &
#                                             (benchmark.metadata.stream_name == region), 'voxel_id'].to_numpy()
#             results.append({'stream_name': region,
#                             'subj_id': subj_id,
#                             'score': np.mean(scores[voxel_id]),
#                             'method': 'ridge'})
#     return pd.DataFrame(results)
