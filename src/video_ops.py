import torch
import av
import numpy as np
import pandas as pd
from deepjuice.procedural.datasets import CustomData
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import NormalizeVideo
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
)
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoProcessor
import transformers


class VideoData(CustomData):
    def __init__(self, video_paths, clip_duration,
                 transforms=None, device='cuda'):
        self.videos = video_paths
        self.clip_duration = clip_duration
        self.device = device
        self.transforms = transforms
        if isinstance(transforms, transformers.models.x_clip.processing_x_clip.XCLIPProcessor):
            self.model_type = 'xclip'
        elif isinstance(transforms, transformers.models.videomae.image_processing_videomae.VideoMAEImageProcessor):
            self.model_type = 'transformer'
        else:
            self.model_type = 'normal'
        self.data = self.videos

    def __getitem__(self, index):
        if self.model_type == 'xclip':
            container = av.open(self.videos[index])
            indices = self.sample_frame_indices(clip_len=8, frame_sample_rate=1,
                                                seg_len=container.streams.video[0].frames)
            video_data = self.read_video_pyav(container, indices)
            inputs = self.transforms(
                text=["people"],
                videos=list(video_data),
                return_tensors="pt",
                padding=True,
            )
            inputs['input_ids'] = inputs['input_ids'].squeeze(0)
            inputs['attention_mask'] = inputs['attention_mask'].squeeze(0)
            inputs['pixel_values'] = inputs['pixel_values'].squeeze(0)
            inputs = inputs.to(self.device)
            return inputs
        elif self.model_type == 'transformer':
            container = av.open(self.videos[index])
            indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
            video_data = self.video_to_list_arrays(container, indices)
            inputs = self.transforms(video_data, return_tensors="pt")
            inputs['pixel_values'] = inputs['pixel_values'].squeeze(0)
            inputs = inputs.to(self.device)
            return inputs
        else:
            video = EncodedVideo.from_path(self.videos[index])  # Initialize an EncodedVideo helper class
            video_data = video.get_clip(start_sec=0, end_sec=self.clip_duration)  # Load the desired clip
            video_data = self.transforms(video_data)  # Transform the
            inputs = video_data["video"]
            # Move to device
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(self.device)
            else:
                inputs = [x.to(self.device) for x in inputs]
            return inputs

    def __len__(self):
        return len(self.videos)

    def get_sample(self, index=None, show_original=False):
        index = self.get_sample_index(index, len(self))

        if show_original:
            print('not yet implemented')

        return self[index]  # the output of __getitem__

    def video_to_list_arrays(self, container, indices):
        '''
        Decode the video with PyAV decoder.
        Args:
            container (`av.container.input.InputContainer`): PyAV container.
            indices (`List[int]`): List of frame indices to decode.
        Returns:
            result (List[np.ndarray]): List of np arrays of decoded frames of shape (3, height, width).
        '''
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                # Convert the frame to an ndarray and transpose it
                frame_ndarray = frame.to_ndarray(format="rgb24")
                transposed_frame = np.transpose(frame_ndarray, (2, 0, 1))  # Reorder dimensions to C, H, W
                frames.append(transposed_frame)
        return frames

    def read_video_pyav(self, container, indices):
        '''
        Decode the video with PyAV decoder.
        Args:
            container (`av.container.input.InputContainer`): PyAV container.
            indices (`List[int]`): List of frame indices to decode.
        Returns:
            result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        '''
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])

    def sample_frame_indices(self, clip_len, frame_sample_rate, seg_len):
        '''
        Sample a given number of frame indices from the video.
        Args:
            clip_len (`int`): Total number of frames to sample.
            frame_sample_rate (`int`): Sample every n-th frame.
            seg_len (`int`): Maximum allowed index of sample's last frame.
        Returns:
            indices (`List[int]`): List of sampled frame indices
        '''
        converted_len = int(clip_len * frame_sample_rate)
        end_idx = np.random.randint(converted_len, seg_len)
        start_idx = end_idx - converted_len
        indices = np.linspace(start_idx, end_idx, num=clip_len)
        indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        return indices
    

def get_video_loader(video_set, clip_duration, transforms, batch_size=64, **kwargs):
    if isinstance(video_set, pd.Series) or isinstance(video_set, list):
        return DataLoader(VideoData(video_set, clip_duration, transforms), batch_size, **kwargs)


def get_transform(model_name):
    if model_name == 'slowfast_r50':
        sampling_rate = 2
        fps = 30
        num_frames = 32
        clip_duration = (num_frames * sampling_rate) / fps
        return slowfast_transform(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225], num_frames=num_frames, side_size=256), clip_duration

    elif 'x3d' in model_name:
        return x3d_transform(model_name, mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225], fps=30)

    elif model_name == 'slow_r50':
        num_frames = 8
        sampling_rate = 8
        fps = 30
        clip_duration = (num_frames * sampling_rate) / fps
        return slow_r50_transform(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225], side_size=256, num_frames=num_frames), clip_duration

    elif model_name == 'c2d_r50':
        num_frames = 8
        sampling_rate = 8
        fps = 30
        clip_duration = (num_frames * sampling_rate) / fps
        return c2d_r50_transform(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225], side_size=256, num_frames=num_frames), clip_duration

    elif model_name == 'i3d_r50':
        num_frames = 8
        sampling_rate = 8
        fps = 30
        clip_duration = (num_frames * sampling_rate) / fps
        return i3d_r50_transform(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225], side_size=256, num_frames=num_frames), clip_duration

    elif model_name == 'csn_r101':
        num_frames = 32
        sampling_rate = 2
        fps = 30
        clip_duration = (num_frames * sampling_rate) / fps
        return i3d_r50_transform(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225], side_size=256, num_frames=num_frames), clip_duration

    elif 'mvit' in model_name:
        num_frames = 16
        sampling_rate = 4
        fps = 30
        clip_duration = (num_frames * sampling_rate) / fps
        return mvit_transform(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225], side_size=256, num_frames=num_frames), clip_duration

    elif 'videomae' in model_name:
        return videomae_transform(), 3

    elif 'xclip' in model_name:
        return xclip_transform(), 3

    elif model_name == 'timesformer-base-finetuned-k400':
        return timesformer_transform(), 3

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


def slowfast_transform(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225], num_frames=32, side_size=256):
    return ApplyTransformToKey(
        key="video",
        transform=Compose([
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x/255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(size=side_size),
            PackPathway()
        ]
        )
    )

####################
# X3D transform
####################

def x3d_transform(model_name, mean, std, fps):
    model_transform_params = {
        "x3d_xs": {
            "side_size": 182,
            "crop_size": 182,
            "num_frames": 4,
            "sampling_rate": 12,
        },
        "x3d_s": {
            "side_size": 182,
            "crop_size": 182,
            "num_frames": 13,
            "sampling_rate": 6,
        },
        "x3d_m": {
            "side_size": 256,
            "crop_size": 256,
            "num_frames": 16,
            "sampling_rate": 5,
        }
    }
    # Get transform parameters based on model
    transform_params = model_transform_params[model_name]
    clip_duration = (transform_params["num_frames"] * transform_params["sampling_rate"]) / fps

    return ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(transform_params["num_frames"]),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(size=transform_params["side_size"])
            ]
        )
    ), clip_duration

####################
# slow_r50 transform
####################
def slow_r50_transform(mean, std, side_size, num_frames):
    return ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(size=side_size)
            ]
        )
    )

####################
# c2d_r50 transform
####################
def c2d_r50_transform(mean, std, side_size, num_frames):
    return ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(size=side_size)
            ]
        )
    )

####################
# i3d_r50 transform
####################
def i3d_r50_transform(mean, std, side_size, num_frames):
    return ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(size=side_size)
            ]
        )
    )

####################
# csn_r101 transform
####################
def csn_r101_transform(mean, std, side_size, num_frames):
    return ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(size=side_size)

            ]
        )
    )

####################
# mvit-b transform
####################
def mvit_transform(mean, std, side_size, num_frames):
    return ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(size=side_size)

            ]
        )
    )

####################
# videomae transform
####################
def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices
def videomae_transform():
    return AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")


####################
# xclip transform
####################
def xclip_transform():
    return AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")

####################
# xclip transform
####################
def timesformer_transform():
    return AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")