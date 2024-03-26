#
import torch
import pandas as pd
from deepjuice.procedural.datasets import CustomDataset
from torchvision.transforms import Compose, Lambda, Resize, Normalize, CenterCrop
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
)
from torch.autograd._functions import Resize
from torch.utils.data import DataLoader
from transformers import VideoMAEImageProcessor


class VideoData(CustomDataset):
    def __init__(self, video_paths, clip_duration,
                 transforms=None, device='cuda'):
        self.videos = video_paths
        self.clip_duration = clip_duration
        self.device = device
        self.transforms = transforms
        self.data = self.videos

    def __getitem__(self, index):
        video = EncodedVideo.from_path(self.videos[index])  # Initialize an EncodedVideo helper class
        video_data = video.get_clip(start_sec=0, end_sec=self.clip_duration)  # Load the desired clip
        video_data = self.transforms(video_data)  # Transform the
        inputs = video_data["video"]  # we only want video not audio
        inputs = [i.to(self.device)[None, ...] for i in inputs]  # Move to device
        return inputs
    
    def __len__(self):
        return len(self.videos)

    def get_sample(self, index=None, show_original=False):
        index = self.get_sample_index(index, len(self))
            
        if show_original:
            print('not yet implemented')
            
        return self[index] # the output of __getitem__
    

def get_video_loader(video_set, clip_duration, transforms, batch_size=64, **kwargs):
    if isinstance(video_set, pd.Series) or isinstance(video_set, list):
        return DataLoader(VideoData(video_set, clip_duration, transforms), batch_size, **kwargs)


def get_transform(model_name):
    if 'slowfast' in model_name:
        sampling_rate = 2
        frames_per_second = 30
        num_frames = 32
        clip_duration = (num_frames * sampling_rate) / frames_per_second
        return slowfast_transform(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225], crop_size=256, num_frames=num_frames, side_size=256), clip_duration

    elif 'x3d' in model_name:
        return x3d_transform(model_name)
    elif 'slow_r50' in model_name:
        return slow_r50_transform()
    elif 'c2d_r50' in model_name:
        return c2d_r50_transform()
    elif 'i3d_r50' in model_name:
        return i3d_r50_transform()
    elif 'csn_r101' in model_name:
        return csn_r101_transform()
    elif 'mvit' in model_name:
        return mvit_transform()
    elif 'videomae' in model_name:
        return videomae_transform()
    else:
        print(f'{model_name} model not yet implemented!')


####################
# SlowFast transform
####################

class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """
    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        alpha = 4
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


def slowfast_transform(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225], crop_size=256, num_frames=32, side_size=256):
    return ApplyTransformToKey(
        key="video",
        transform=Compose([
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x/255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(size=side_size),
            CenterCropVideo(crop_size),
            PackPathway()
        ]
        )
    )

####################
# X3D transform
####################

def x3d_transform(model_name):
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    frames_per_second = 30
    model_transform_params = {
        "x3d_xs": {
            "crop_size": 182,
            "num_frames": 4,
            "sampling_rate": 12,
        },
        "x3d_s": {
            "crop_size": 182,
            "num_frames": 13,
            "sampling_rate": 6,
        },
        "x3d_m": {
            "crop_size": 256,
            "num_frames": 16,
            "sampling_rate": 5,
        }
    }

    # Get transform parameters based on model
    transform_params = model_transform_params[model_name]
    return ApplyTransformToKey(
              key="video",
              transform=Compose(
                    [
                        UniformTemporalSubsample(transform_params["num_frames"]),
                        Lambda(lambda x: x/255.0),
                        Normalize(mean, std),
                        ShortSideScale(transform_params["crop_size"])
                    ]
                   )
    )

####################
# slow_r50 transform
####################
def slow_r50_transform():
    side_size = 256
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    crop_size = 256
    num_frames = 8
    sampling_rate = 8
    frames_per_second = 30
    # The duration of the input clip is also specific to the model.
    #clip_duration = (num_frames * sampling_rate) / frames_per_second

    # Note that this transform is specific to the slow_R50 model.
    return ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(
                    size=side_size
                ),
                CenterCropVideo(crop_size=(crop_size, crop_size))
            ]
        ),
    )

####################
# c2d_r50 transform
####################
def c2d_r50_transform():
    side_size = 256
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    crop_size = 256
    num_frames = 8

    return ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(size=side_size),
                CenterCropVideo(crop_size=(crop_size, crop_size))
            ]
        ),
    )

####################
# i3d_r50 transform
####################
def i3d_r50_transform():
    side_size = 256
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    crop_size = 256
    num_frames = 8

    return ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(size=side_size),
                CenterCropVideo(crop_size=(crop_size, crop_size))
            ]
        ),
    )

####################
# csn_r101 transform
####################
def csn_r101_transform():
    side_size = 256
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    crop_size = 224
    num_frames = 32
    return ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(num_frames),  # Subsamples 32 frames from the video uniformly
                Lambda(lambda x: x / 255.0),  # Normalize pixel values to [0, 1], assuming x is a tensor in the range [0, 255]
                ShortSideScale(size=side_size),  # Resize the short side of the frame to 256 pixels
                CenterCrop(crop_size),  # Crop the center 224x224 pixels from the frame
                Normalize(mean=mean, std=std),
            ]
        ),
    )

def mvit_transform():
    side_size = 256
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    crop_size = 224
    num_frames = 16
    return ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(num_frames),  # Subsample 16 frames uniformly across the video
                Lambda(lambda x: x / 255.0),  # Scale pixel values to [0, 1]
                Resize(side_size),  # Resize such that the smaller side is 256 pixels
                CenterCrop(crop_size),  # Crop the center to get 224x224 pixels
                Normalize(mean=mean, std=std),  # ImageNet normalization
            ]
        ),
    )

def videomae_transform():
    return VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")

