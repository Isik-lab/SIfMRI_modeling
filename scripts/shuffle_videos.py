import os
import cv2
import numpy as np
import random
from pathlib import Path


def shuffle_frames(input_video_path):
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Read all frames
    frames = []
    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    # Shuffle the frames
    random.shuffle(frames)
    
    cap.release()
    
    return frames, (frame_width, frame_height), fps


def process_videos(input_dir, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Loop through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Check for video files
            input_video_path = os.path.join(input_dir, filename)
            output_video_path = os.path.join(output_dir, filename)

            # Shuffle frames
            frames, (frame_width, frame_height), fps = shuffle_frames(input_video_path)

            # Write shuffled frames to a new video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

            for frame in frames:
                out.write(frame)
            
            out.release()
            print(f"Processed and saved: {output_video_path}")

# Example usage
input_directory = 'data/raw/videos'
output_directory = 'data/raw/shuffled_videos'
Path(output_directory).mkdir(exist_ok=True, parents=True)
process_videos(input_directory, output_directory)
