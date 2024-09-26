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
        text = text[:1]

        inputs = self.transforms(text=text, images=image, return_tensors="pt", padding=True)
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

def get_multimodal_loader(frame_data, transforms, batch_size=16, group_keys=None, image_key='images', caption_key='captions', device='cuda'):
    if group_keys is not None:
        batch_data = batch_by_group(frame_data, group_keys, batch_size)
        images = batch_data[image_key]  # batched images after sort
        captions = batch_data[caption_key]  # batched captions after sort
        dataloader = MultimodalData(images, captions, transforms, device)
        setattr(dataloader, 'batch_data', batch_data)
        return dataloader

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