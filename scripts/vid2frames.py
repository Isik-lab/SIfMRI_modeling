#/Applications/anaconda3/envs/pytorch/bin/python
import imageio
from PIL import Image
from glob import glob
from tqdm import tqdm
from pathlib import Path


def save_frames_from_video(top_dir, video_file, output='images'):
    file_name = Path(video_file).stem
    if output == 'frames':
        out_dir = Path(top_dir) / output / file_name
        out_dir.mkdir(exist_ok=True, parents=True)
    else: #'images'
        out_dir = Path(top_dir) / output
        out_dir.mkdir(exist_ok=True, parents=True)

    with imageio.get_reader(video_file, 'ffmpeg') as vid:
        if output == 'frames':
            frame_count = vid.get_meta_data()['nframes']
            for i in range(frame_count):
                im = Image.fromarray(vid.get_data(i))
                im.save(out_dir / f'frame-{str(i).zfill(2)}.png')
        else: #'images
            im = Image.fromarray(vid.get_data(0))
            im.save(out_dir / f'{file_name}.png')


if __name__ == '__main__':
    top_dir = Path('data/raw')
    vid_dir = top_dir / 'videos'
    videos = list(vid_dir.glob('*.mp4'))

    for video in tqdm(videos):
        save_frames_from_video(top_dir, video)
