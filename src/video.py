#
import torch
import pandas as pd
from deepjuice.procedural.datasets import CustomDataset
from pytorchvideo.data.encoded_video import EncodedVideo
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda
from pytorchvideo.transforms import ShortSideScale, Normalize
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import ApplyTransformToKey, UniformTemporalSubsample


class VideoData(CustomDataset):
    def __init__(self, video_paths, clip_duration, transforms=None):
        self.videos = video_paths
        self.transforms = transforms
        self.clip_duration = clip_duration

    def __getitem__(self, index):
        video = EncodedVideo.from_path(self.videos[index]) # Initialize an EncodedVideo helper class
        video_data = video.get_clip(start_sec=0, end_sec=self.clip_duration) # Load the desired clip
            
        if self.transforms:
            video_data = self.transforms(video_data)
        return video_data
    
    def __len__(self):
        return len(self.videos)
    

def get_video_loader(video_set, transforms, clip_duration, batch_size = 64, **kwargs):
    if isinstance(video_set, pd.Series) or isinstance(video_set, list):
        return DataLoader(VideoData(video_set, clip_duration, transforms), batch_size, **kwargs)


def get_transform(model_name):
    if 'slowfast' in model_name:
        return slowfast_transform()

####################
# SlowFast transform
####################

class SlowFast_PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """
    def __init__(self):
        super().__init__()
        self.alpha = 4

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // self.alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


def slowfast_transform():
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    crop_size = 256
    num_frames = 32
    sampling_rate = 2
    frames_per_second = 30
   
    clip_duration = (num_frames * sampling_rate)/frames_per_second
    transform = ApplyTransformToKey(
        key="visdeo",
        transform=Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x/255.0),
                Normalize(mean, std),
                ShortSideScale(crop_size),
                SlowFast_PackPathway()
            ]
        ),
    )
    return transform, clip_duration