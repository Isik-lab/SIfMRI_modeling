from transformers import CLIPProcessor, CLIPModel
import pandas as pd
try:
    from deepjuice.procedural.datasets import CustomDataset as CustomDataset
except:
    from deepjuice.procedural.datasets import CustomData as CustomDataset
import torch
from torch.utils.data import DataLoader
from PIL import Image

######General###########

class MultimodalData(CustomDataset):
    def __init__(self, image_paths, captions,
                 transforms=None, device='cuda', group_keys='video_name', **kwargs):
        self.images = image_paths
        self.texts = captions  # or docs
        self.device = device
        self.transforms = transforms
        self.group_keys = group_keys

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        text = self.texts[index]

        inputs = self.transforms(text=text, images=image, return_tensors="pt", padding=True)
        # Move to device
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.to(self.device)
        else:
            inputs = [x.to(self.device) for x in inputs]
        return inputs

    def __len__(self):
        return len(self.images)

    def get_sample(self, index=None, show_original=False):
        index = self.get_sample_index(index, len(self))

        if show_original:
            print('not yet implemented')

        return self[index]  # the output of __getitem__

def get_multimodal_loader(image_paths, captions, transforms, batch_size=64, group_keys='video_name', device='cuda', **kwargs):
    return DataLoader(MultimodalData(image_paths, captions, transforms, device, group_keys), batch_size, **kwargs)

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
