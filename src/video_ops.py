#
import pandas as pd
import os
from glob import glob
from tqdm import tqdm 
import os, cv2
from PIL import Image
from IPython.display import HTML
from deepjuice.structural import get_fn_kwargs
import torch


def visual_events(stimulus_data, video_dir, image_dir, 
                            key_frames=['F','L','M'],
                            target_index=None, **kwargs):

    # event data is our annotations:
    if isinstance(stimulus_data, str):
        stimulus_data = pd.read_csv(stimulus_data)

    # if isinstance(metadata, str):
    #     metadata = pd.read_csv(metadata)

    # if isinstance(response_data, str):
    #     response_data = pd.read_csv(response_data)

    stimulus_data.columns = [col.replace(' ', '_') for 
                          col in stimulus_data.columns]

    video_kwargs = get_fn_kwargs(parse_video_data, kwargs)
    video_data = parse_video_data(video_dir, **video_kwargs)

    process_kwargs = get_fn_kwargs(process_event_videos, kwargs)
    process_kwargs['key_frames'] = key_frames
    
    process_event_videos(video_data, image_dir, **process_kwargs)

    all_frame_paths = sorted(glob(f'{image_dir}/*.png'))
    
    event_index = {}
    image_paths = []

    def get_frame_index(frame_path):
        return int(frame_path.split('frame_')[-1].split('.')[0])

    frame_indices = [get_frame_index(path) for 
                     path in all_frame_paths]

    frame_indices = sorted(pd.unique(frame_indices).tolist())
                               
    if target_index is not None:
        if target_index in ['middle', 'median']:
            target_index = (int(len(frame_indices) // 2))
            
        if isinstance(target_index, int):
            frame_indices = [frame_indices[target_index]]
            
        else: # interpretable error
            raise ValueError("target_index should be None or one of {int, 'middle'}")

    stimulus_data = stimulus_data.merge(video_data)
            
    for video_id in stimulus_data.video_id:
        for frame_id in frame_indices:
            image_paths += sorted([path for path in all_frame_paths 
                                   if f'video_{video_id}' in path
                                   and f'frame_{frame_id}' in path])
                                  
            index_offset = (int(video_id)-1) * len(frame_indices)
            index_range = range(index_offset, index_offset + len(frame_indices))
            
            event_index[video_id] = torch.Tensor(index_range).long()

    return {'stimulus_data': stimulus_data, 
            'image_paths': image_paths,
            'group_indices': event_index}


def display_video(video_path):
    video_html = f"""
    <video width="500" autoplay="true" loop="true">
      <source src="{video_path}" type="video/mp4">
    </video>
    """
    return HTML(video_html)


def process_video(video_path, target_frames=None, verbose=False):
    # Load the video
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Could not open video")
        return

    # Obtain the frame rate
    fps = video.get(cv2.CAP_PROP_FPS)
    if verbose: print(f'FPS: {fps}')

    # Create a list to store the extracted frames
    extracted_frames = []
    
    # Frame index counter
    frame_idx = 0
    while True:
        # Read a frame from the video
        ret, frame = video.read()

        # If the frame was not correctly read, then we reached the end of the video
        if not ret:
            break

        # Convert frame to RGB and then to PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        # Check if this frame is in the target frames list
        if target_frames is None or frame_idx in target_frames:
            extracted_frames.append((frame_idx, frame_pil))

        # Update the frame index
        frame_idx += 1

    video.release()

    return extracted_frames


def count_total_frames(video_path):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()

    return total_frames
    

def get_sequence_idx(sequence_length=None, every_nth=1, *keys, 
                     first=False, last=False, middle=False):
    
    sequence = list(range(sequence_length))

    output_idx = [] # fill conditionally

    # Filter every nth element
    if every_nth is not None:
        output_idx += sequence[::every_nth]

    key_frames = [key[0].upper() for key in keys
                  if isinstance(key, str)]

    key_indices = [key for key in keys
                   if isinstance(key, int)]

    # Check first, last, middle, frames
    if first or 'F' in key_frames: 
        output_idx += [0]

    if last or 'L' in key_frames:
        output_idx += [sequence_length]

    if middle or 'M' in key_frames:
        output_idx += [sequence_length // 2]

    output_idx += [index for index in key_indices
                   if index not in output_idx]
    
    return list(sorted(set(output_idx)))

def extract_frames(video_path, video_id, output_dir=None, 
                   *key_frames, keep_every=1, **kwargs):

    if all(isinstance(index, int) for index in key_frames):
        frame_idx = list(key_frames)

    else: # parse  sequence indicators
        total_frames = count_total_frames(video_path)
        sequence_args = (total_frames, keep_every)
        frame_idx = get_sequence_idx(*sequence_args, *key_frames)

    if output_dir is None: # process directly
        return process_video(video_path, frame_idx)

    zfill_digits = kwargs.pop('zfill', True)

    if zfill_digits: # is not None or False
        zfill_digits = len(str(max(frame_idx)))

    pre = kwargs.pop('prefix', 'frame')
    file_head = f"video_{video_id}"

    output_files = {} # iterfill
    for _, frame_index in enumerate(frame_idx):
        frame_id = str(frame_index)
        if zfill_digits is not None:
            frame_id = frame_id.zfill(zfill_digits)

        file_name = f'{file_head}_{pre}_{frame_id}.png'
        file_path = os.path.join(output_dir, file_name)
        
        output_files[frame_index] = file_path

    if not all(os.path.exists(file) for file 
               in output_files.values()):

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        frames = process_video(video_path, frame_idx)
        
        for frame_id, frame in frames:
            output_file = output_files[frame_id]
            if not os.path.exists(output_file):
                frame.save(output_file)


def parse_video_data(video_directory, output_file=None, 
                     root=None, **kwargs):

    if not os.path.exists(video_directory):
        raise ValueError(f'video directory: {video_directory} does not exist')
    
    video_paths = sorted(glob(f'{video_directory}/*.mp4', recursive=True))

    video_uids = []
    video_data = []
    for _, video_path in enumerate(video_paths):        
        video_name = video_path.split('/')[-1]
        if video_name not in video_uids:
            video_uids.append(video_name)
    
        video_abspath = os.path.abspath(video_path)
        if root and root=='absolute':
            video_path = video_abspath

        if root is not None:
            video_path = os.path.join(root, video_path)
            
        video_data.append({'video_name': video_name, 'video_path': video_path})

    group = ['video_name']
    
    video_data = (pd.DataFrame(video_data)
                  .sort_values(by=group)
                  .reset_index(drop=True))
    
    video_ids = video_data.groupby(group).ngroup().values
    video_data.insert(2, 'video_id', [str(i+1).zfill(3) for i in video_ids])

    if output_file and not os.path.exists(output_file):
        video_data.to_csv(output_file, index = None)

    return video_data 


def process_event_videos(video_data, output_dir, key_frames=['F','L'], keep_every=12):
    for i, row in tqdm(video_data.iterrows(), total = video_data.shape[0],
                       desc = 'Processing Event Videos'):
        
        extract_frames(row['video_path'], row['video_id'], 
                       output_dir, *key_frames, keep_every=keep_every)