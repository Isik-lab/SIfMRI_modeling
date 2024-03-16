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
                  frame_idx=None,
                  target_index=None, **kwargs):

    # event data is our annotations:
    if isinstance(stimulus_data, str):
        stimulus_data = pd.read_csv(stimulus_data)

    stimulus_data['video_path'] = stimulus_data['video_name'].apply(lambda x:
                                                                     os.path.join(video_dir, x))
    stimulus_data = parse_video_data(stimulus_data)

    process_event_videos(stimulus_data, image_dir,
                         frame_idx=frame_idx)

    all_frame_paths = sorted(glob(f'{image_dir}/*.png'))
    
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
            
    image_paths = []
    for video_id in stimulus_data.video_id:
        for frame_id in frame_indices:
            paths = sorted([path for path in all_frame_paths
                            if f'video_{video_id}' in path
                            and f'frame_{frame_id}' in path])
            image_paths.append({'video_id': video_id, 'frame_id': frame_id, 'images': paths[0]})
    image_paths = pd.DataFrame(image_paths)
    return stimulus_data.merge(image_paths, on='video_id', how='outer')


def process_video(video_path, target_frames, verbose=False):
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
    

def extract_frames(video_path, video_id, frame_idx=None,
                   output_dir=None, **kwargs):

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


def parse_video_data(video_data, group=['video_name']):
    video_data_out = video_data.copy()
    video_data_out.set_index(group, inplace=True)
    video_data_out = video_data_out.reset_index(drop=False).reset_index(drop=False)
    video_ids = [str(index+1).zfill(3) for index in video_data_out['index']]
    video_data_out = video_data_out.drop(columns=['index'])
    video_data_out['video_id'] = video_ids
    return video_data_out 


def process_event_videos(video_data, output_dir, frame_idx=None):
    for _, row in tqdm(video_data.iterrows(), total=video_data.shape[0],
                       desc = 'Processing Event Videos'):
        
        extract_frames(row['video_path'], row['video_id'], 
                       output_dir=output_dir, frame_idx=frame_idx)