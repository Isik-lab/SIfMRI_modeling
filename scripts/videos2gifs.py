import os
from moviepy.editor import VideoFileClip

def mp4_to_gif(mp4_path, gif_path, fps=10):
    """
    Convert an MP4 file to a GIF.

    Parameters:
    - mp4_path (str): The path to the MP4 file.
    - gif_path (str): The path where the GIF should be saved.
    - fps (int, optional): The frames per second for the GIF. Default is 10.
    """
    with VideoFileClip(mp4_path) as clip:
        clip.write_gif(gif_path, fps=fps)

def convert_all_mp4s_in_directory(input_directory, output_directory, fps=10):
    """
    Convert all MP4 files in the input directory to GIFs in the output directory.

    Parameters:
    - input_directory (str): The path to the directory containing the MP4 files.
    - output_directory (str): The path to the directory where the GIFs should be saved.
    - fps (int, optional): The frames per second for the GIFs. Default is 10.
    """
    
    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Iterate over all files in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith(".mp4"):
            mp4_path = os.path.join(input_directory, filename)
            gif_path = os.path.join(output_directory, filename.replace(".mp4", ".gif"))
            mp4_to_gif(mp4_path, gif_path, fps=fps)
            print(f"Converted {filename} to GIF")

# Usage:
mp4_path = '/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_modeling/data/raw/videos'
gif_path = '/Users/emcmaho7/Dropbox/projects/SI_fmri/SIfMRI_modeling/data/raw/gifs'
convert_all_mp4s_in_directory(mp4_path, gif_path)

