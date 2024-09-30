from transformers import CLIPProcessor, CLIPModel
import pandas as pd
try:
    from deepjuice.procedural.datasets import CustomDataset as CustomDataset
except:
    from deepjuice.procedural.datasets import CustomData as CustomDataset
import torch
from PIL import Image
from torch.utils.data import BatchSampler
import numpy as np
from torch.utils.data import DataLoader

######General###########

class MultimodalData(CustomDataset):
    def __init__(self, image_paths, captions,
                 transforms=None, device='cuda'):
        self.images = image_paths
        self.texts = captions  # or docs
        self.device = device
        self.transforms = transforms

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        text = self.texts[index]
        text = [text]

        inputs = self.transforms(text=text, images=image, return_tensors="pt", padding=True)
        #inputs['pixel_values'] = inputs['pixel_values'].squeeze(0)
        #inputs['attention_mask'] = inputs['attention_mask'].squeeze(0)
        # Move to device
        inputs = inputs.to(self.device)
        return inputs

    def __len__(self):
        return len(self.images)

    def get_sample(self, index=None, show_original=False):
        index = self.get_sample_index(index, len(self))

        if show_original:
            print('not yet implemented')

        return self[index]  # the output of __getitem__


class SizeSampler(BatchSampler):
    def __init__(self, batch_sizes):
        self.batch_sizes = batch_sizes

    def __iter__(self):
        start = 0
        for batch_size in self.batch_sizes:
            indices = range(start, start + batch_size)
            yield indices
            start += batch_size

    def __len__(self):
        return len(self.batch_sizes)

    def __repr__(self):
        return self.get_report()

    def get_report(self, skip_header=False):
        lines = [f'BatchSampler with {len(self.batch_sizes)} batches.']

        lines += ['Batch sizes range from: ' +
                  str(min(self.batch_sizes)) +
                  ' to ' + str(max(self.batch_sizes))]

        lines += ['with an average size of: ~' +
                  str(int(np.mean(self.batch_sizes)))]

        if skip_header:
            lines = lines[1:]

        return '\n  '.join(lines)

def pad_collate(batch):
    # Safeguard: Ensure attention_mask and input_ids exist and have valid shape
    for item in batch:
        print(item['attention_mask'].shape)

    valid_attention_masks = [item['attention_mask'] for item in batch if 'attention_mask' in item and item['attention_mask'] is not None]
    valid_ids = [item['input_ids'] for item in batch if 'input_ids' in item and item['input_ids'] is not None]

    if len(valid_attention_masks) == 0:
        raise ValueError("No valid 'attention_mask' found in batch")
    if len(valid_ids) == 0:
        raise ValueError("No valid 'input_ids' found in batch")

    # Find the maximum size among the attention masks
    max_size_attention_mask = max([mask.shape[1] for mask in valid_attention_masks])
    print(max_size_attention_mask)

    # Pad each attention_mask to the max_size_attention_mask
    for item in batch:
        if 'attention_mask' in item and item['attention_mask'] is not None:
            pad_size = max_size_attention_mask - item['attention_mask'].shape[1]
            print(pad_size)
            if pad_size > 0:
                # Pad the attention_mask with zeros to match the max size
                item['attention_mask'] = torch.nn.functional.pad(item['attention_mask'], (0, pad_size), "constant", 0)
            # Ensure attention_mask is 2D by squeezing extra dimensions
            item['attention_mask'] = item['attention_mask'].squeeze()

    # Find the maximum size among the input_ids
    max_size_input_ids = max([id.shape[1] for id in valid_ids])
    print(max_size_input_ids)

    # Pad each input_ids to the max_size_input_ids
    for item in batch:
        if 'input_ids' in item and item['input_ids'] is not None:
            pad_size = max_size_input_ids - item['input_ids'].shape[1]
            print(pad_size)
            if pad_size > 0:
                # Pad the input_ids with zeros to match the max size
                item['input_ids'] = torch.nn.functional.pad(item['input_ids'], (0, pad_size), "constant", 0)
            # Ensure input_ids is 2D by squeezing extra dimensions
            item['input_ids'] = item['input_ids'].squeeze()

    # Debugging output for tensor shapes
    for item in batch:
        print(f"input_ids shape: {item['input_ids'].shape}")
        print(f"attention_mask shape: {item['attention_mask'].shape}")
        print(f"pixel_values shape: {item['pixel_values'].shape}")

def get_multimodal_loader(frame_data, captions, transforms, batch_size=16, group_keys=None, image_key='images', caption_key='captions', device='cuda',  **kwargs):
    frame_data[caption_key] = captions['caption']
    if group_keys is not None:
        batch_data = batch_by_group(frame_data, group_keys, batch_size)
        images = batch_data[image_key]  # batched data after sort
        captions = batch_data[caption_key]
        dataloader = DataLoader(MultimodalData(images, captions, transforms, device), batch_size, **kwargs)
        setattr(dataloader, 'batch_data', batch_data)
        return dataloader
    else:
        images = frame_data[image_key]
        captions = captions['caption']
        return DataLoader(MultimodalData(images, captions, transforms, device), batch_size, collate_fn=pad_collate, **kwargs)


def get_model(model_uid, modal='vision-language'):
    if model_uid == 'clip-vit-base-patch32':
        if modal == 'vision-language':
            model, processor = load_mm_clip(model_uid)
        elif modal == 'vision':
            model, processor = load_mm_clip(model_uid)
        elif modal == 'language':
            model, processor = load_mm_clip(model_uid)
    else:
        raise ValueError(f'Model UID - {model_uid} not implemented')
    return model, processor


def load_mm_clip(model_uid):
    model = CLIPModel.from_pretrained(f"openai/{model_uid}")
    processor = CLIPProcessor.from_pretrained(f"openai/{model_uid}")
    return model, processor

def load_vision_clip(model_uid):
    model = CLIPModel.from_pretrained(f"openai/{model_uid}")
    processor = CLIPProcessor.from_pretrained(f"openai/{model_uid}")
    return model, processor

def load_language_clip(model_uid):
    model = CLIPModel.from_pretrained(f"openai/{model_uid}")
    processor = CLIPProcessor.from_pretrained(f"openai/{model_uid}")
    return model, processor


def batch_by_group(df, group_vars, max_batch_size=64, add_batch_size=True):
    if not isinstance(group_vars, list):
        group_vars = [group_vars]

    group_counts = df.groupby(group_vars).size().reset_index(name='group_size')
    group_counts['group_index'] = range(len(group_counts))  # assign group index

    df = df.merge(group_counts[group_vars + ['group_index']], on=group_vars, how='left')

    batch_iter, batch_indices = 0, []
    current_batch_size = 0

    for _, row in group_counts.iterrows():
        group_size = row['group_size']
        if current_batch_size + group_size > max_batch_size:
            batch_iter += 1
            current_batch_size = 0
        batch_indices.append(batch_iter)
        current_batch_size += group_size

    group_to_batch = dict(zip(group_counts['group_index'], batch_indices))
    df['batch_iter'] = df['group_index'].map(group_to_batch)
    df['batch_index'] = df.groupby('batch_iter').cumcount()  # index in batch

    if add_batch_size:
        batch_sizes = df.groupby('batch_iter').size().reset_index(name='batch_size')
        df = df.merge(batch_sizes)  # concatenate batch sizes with main df

    return df  # updated dataframe with batches determined by specified groups