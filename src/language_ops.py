#
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch import tensor
from sentence_transformers import SentenceTransformer
from deepjuice.structural import flatten_nested_list # utility for list flattening
from transformers import GPT2TokenizerFast
from deepjuice.extraction import FeatureExtractor
from src.tools import moving_grouped_average
import numpy as np
import pandas as pd

######SLIP###########

def slip_language_model(model_filepath, device='cuda'):
    from slip import models #SLIP models downloaded from https://github.com/facebookresearch/SLIP/blob/main/models.py
    from slip.tokenizer import SimpleTokenizer#custom tokenizer for SLIP https://github.com/facebookresearch/SLIP/blob/main/tokenizer.py
    from slip import utils #https://github.com/facebookresearch/SLIP/blob/main/utils.py
    from collections import OrderedDict

    ### following code is taken from https://github.com/facebookresearch/SLIP/blob/main/eval_zeroshot.py
    #load model information (weights + metadata)
    ckpt = torch.load(model_filepath, map_location=device)
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    # create model architecture and load weights
    old_args = ckpt['args']
    print("=> creating model: {}".format(old_args.model))
    model = getattr(models, old_args.model)(rand_embed=False,
        ssl_mlp_dim=old_args.ssl_mlp_dim, ssl_emb_dim=old_args.ssl_emb_dim)
    #put model in evaluation mode
    if 'cuda' in device:
        model.cuda()    
    model.load_state_dict(state_dict, strict=True)
    print("=> loaded epoch {}".format(ckpt['epoch']))
    
    model.eval()
    return model, SimpleTokenizer()


def slip_feature_extraction(model_filepath, captions, device):
    from slip import utils
    model, tokenizer = slip_language_model(model_filepath, device=device)
    print('encoding captions...')

    features = []
    caption_iterator = tqdm(captions, desc='Encoding captions', total=len(captions))
    with torch.no_grad():
        for caption in caption_iterator:
            tokenized_caption = tokenizer(caption)
            if 'cuda' in device:
                tokenized_caption = tokenized_caption.cuda(non_blocking=True)
            tokenized_caption = tokenized_caption.view(-1, 77).contiguous()
            encoded_caption = utils.get_model(model).encode_text(tokenized_caption)
            features.append(encoded_caption.detach().numpy().squeeze())
  
    return np.vstack(features)


######GLOVE###########

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


######General###########

def load_llm(model_uid, generative=False):
    if generative: 
        model_config = {'use_cache': False}
        model_ = AutoModel.from_pretrained(model_uid, **model_config)
    else:
        model_ = AutoModel.from_pretrained(model_uid)
    tokenizer_ = AutoTokenizer.from_pretrained(model_uid)
    tokenizer_.add_special_tokens({'pad_token': '[PAD]'})
    model_.resize_token_embeddings(len(tokenizer_))
    return model_, tokenizer_


def tokenize_captions(tokenizer_, captions_):
    tokenized_captions_ = tokenizer_(captions_, return_tensors='pt', padding='max_length') 
    print(f'{tokenized_captions_["input_ids"]=}')
    print(f'{tokenized_captions_["attention_mask"]=}')
    return tokenized_captions_


def parse_caption_data(dataset_file, format='long', output_file=None, **kwargs):

    captions_wide = pd.read_csv(dataset_file)

    def row_func(row):
        return [x for x in row['caption01':] if not pd.isna(x)]

    captions_list = captions_wide.apply(lambda row: row_func(row), axis=1)

    captions_dict = {captions_wide['video_name'].iloc[i]: 
                     captions_list.iloc[i] for i in range(len(captions_wide))}

    captions_long = pd.melt(captions_wide, id_vars=['video_name'], 
                            value_vars=[f'caption{i:02}' for 
                                        i in range(1, 12)],
                            var_name='caption_index', value_name='caption')
    
    captions_long['caption_index'] = (captions_long['caption_index'].str
                                      .replace('caption', '').astype(int))
    
    # Optional: removing rows where caption is NaN
    captions_long.dropna(subset=['caption'], inplace=True)

    caption_counts = (captions_long.groupby('video_name')['caption_index'].max()
                      .reset_index().rename(columns={'caption_index': 'caption_count'}))

    count_min = caption_counts['caption_count'].min()
    count_max = caption_counts['caption_count'].max()

    if kwargs.pop('show_counts', False):
        print(f'Caption Count Min-Max: ({count_min}, {count_max})')

    if format=='nested':
        data['captions'] = captions_list
        return data['video_name','captions']

    if format=='dict':
        return captions_dict

    if format=='wide':
        return captions_wide
        
    if format=='long':
        return captions_long

    raise ValueError('format must be one of {nested,dict,wide,list}')
