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


def captions_to_list(input_captions):
    all_captions = input_captions.tolist() # list of strings
    captions = flatten_nested_list([eval(captions)[:5] for captions in all_captions])
    print(captions[:5])
    return captions, (len(all_captions), 5)


def load_llm(model_uid):
    model_ = AutoModel.from_pretrained(model_uid)
    tokenizer_ = AutoTokenizer.from_pretrained(model_uid)
    print(f'{tokenizer_.eos_token=}')
    print(f'{tokenizer_.eos_token_id=}')
    print(f'{tokenizer_.pad_token=}')
    print(f'{tokenizer_.pad_token_id=}')
    return model_, tokenizer_


def load_gpt():
    model_ = AutoModel.from_pretrained('gpt2')
    tokenizer_ = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer_.add_special_tokens({'pad_token': '[PAD]'})
    model_.resize_token_embeddings(len(tokenizer_))
    return model_, tokenizer_


class CustomDataset(Dataset):
    def __init__(self, input_ids, attention_masks):
        self.input_ids = input_ids
        self.attention_masks = attention_masks

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': tensor(self.input_ids[idx], dtype=torch.long),
            'attention_mask': tensor(self.attention_masks[idx], dtype=torch.long)
        }


def gpt_extraction(captions, device):
    model, tokenizer = load_gpt()
    tokenized_captions = tokenize_captions(tokenizer, captions)
    tensor_dataset = CustomDataset(tokenized_captions['input_ids'],
                                   tokenized_captions['attention_mask'])
    dataloader = DataLoader(tensor_dataset, batch_size=5)
    feature_extractor = FeatureExtractor(model, dataloader, remove_duplicates=True,
                                        tensor_fn=moving_grouped_average,
                                        memory_limit='10GB',
                                        output_device=device, exclude_oversize=True)
    feature_extractor.modify_settings(flatten=True)
    return feature_extractor


def tokenize_captions(tokenizer_, captions_):
    tokenized_captions_ = tokenizer_(captions_, return_tensors='pt', padding='max_length') 
    print(f'{tokenized_captions_["input_ids"]=}')
    print(f'{tokenized_captions_["attention_mask"]=}')
    return tokenized_captions_


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


def memory_saving_extraction(model_uid, captions, device):
    model, tokenizer = load_llm(model_uid)
    tokenized_captions = tokenize_captions(tokenizer, captions)
    tensor_dataset = TensorDataset(tokenized_captions['input_ids'], tokenized_captions['attention_mask'])
    dataloader = DataLoader(tensor_dataset, batch_size=20)
    feature_extractor = FeatureExtractor(model, dataloader,
                                        memory_limit='20GB', output_device=device)
    feature_extractor.modify_settings(flatten=True)
    return feature_extractor


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
