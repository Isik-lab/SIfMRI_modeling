import os, re, cv2

from PIL import Image
from IPython.display import HTML

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
