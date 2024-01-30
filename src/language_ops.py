#
import clip
import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch import tensor
from sentence_transformers import SentenceTransformer
from deepjuice.structural import flatten_nested_list # utility for list flattening
from transformers import GPT2TokenizerFast
from deepjuice.extraction import FeatureExtractor
from src.encoding import moving_grouped_average



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
    dataloader = DataLoader(tensor_dataset, batch_size=20)
    feature_extractor = FeatureExtractor(model, dataloader, remove_duplicates=False,
                                        tensor_fn=moving_grouped_average,
                                        sample_size=5, reduce_size_by=5,
                                        output_device=device, exclude_oversize=False)
    feature_extractor.modify_settings(flatten=True)
    return feature_extractor


def clip_extraction(captions, backbone='RN50', device='cuda'):
    """
        input:
            captions: a list of strings of the captions of the images
        output: 
            feature_extractor: DeepJuice feature extractor object
    """
    model, _ = clip.load(backbone, device=device)
    model = model.token_embedding.eval() #select the language encoder
    tokenized_captions = clip.tokenize(captions)
    tensor_dataset = TensorDataset(tokenized_captions)
    print(f'{tensor_dataset=}')
    dataloader = DataLoader(tensor_dataset, batch_size = 20)
    feature_extractor = FeatureExtractor(model, dataloader, remove_duplicates=False,
                                        tensor_fn=moving_grouped_average,
                                        sample_size=5, reduce_size_by=5,
                                        output_device='cuda', exclude_oversize=False)
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
    tensor_dataset = TensorDataset(['input_ids'], tokenized_captions['attention_mask'])
    dataloader = DataLoader(tensor_dataset, batch_size=20)
    feature_extractor = FeatureExtractor(model, dataloader, remove_duplicates=False,
                                        tensor_fn=moving_grouped_average,
                                        sample_size=5, reduce_size_by=5,
                                        output_device=device, exclude_oversize=False)
    feature_extractor.modify_settings(flatten=True)
    return feature_extractor