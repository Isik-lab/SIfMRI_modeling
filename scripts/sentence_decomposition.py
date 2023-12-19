import os
import pandas as pd
from pathlib import Path
from src import lang_permute
from tqdm import tqdm
import argparse
import numpy as np


def load_caption_file(file):
    print('performing grammar correction')
    df = pd.read_csv(file)
    videos = df['video_name'].to_list()
    columns = ['caption' + str(i+1).zfill(2) for i in range(5)]
    caption_arr = df[columns].to_numpy().flatten()
    return videos, columns, caption_arr, df[columns].shape


class SentenceDecomposition:
    def __init__(self, args):
        self.process = 'SentenceDecomposition'
        self.data_dir = args.data_dir
        self.func_name = args.func_name
        self.overwrite = args.overwrite
        print(vars(self))

        #change the default cache location if defined
        if args.cache is not None:
            os.environ['TRANSFORMERS_CACHE'] = args.cache

        # get the function from the string
        if self.func_name == 'corrected_unmasked':
            self.func = lang_permute.strip_sentence
        else:
            self.func = getattr(lang_permute, self.func_name)
            
        #make the output path and set file names
        Path(f'{self.data_dir}/interim/{self.process}').mkdir(parents=True, exist_ok=True)
        self.out_file = f'{self.data_dir}/interim/{self.process}/{self.func_name}.csv'
        self.grammar_file = f'{self.data_dir}/interim/{self.process}/corrected_captions.csv'
        self.prompt = 'Rewrite the sentence, fix grammatical and spelling errors, and simplify the syntax: '
        self.columns = ['caption' + str(i+1).zfill(2) for i in range(5)]

    def loop_captions(self, df_):
        caption_arr = df_[self.columns].to_numpy().flatten()
        out = []
        for caption in tqdm(caption_arr, total=len(caption_arr), desc=self.func_name):
            perturbed_caption = self.func(caption)
            out.append(perturbed_caption)
        out = pd.DataFrame(np.array(out).reshape(df_[self.columns].shape), columns=self.columns)
        out['video_name'] = df_['video_name'].to_list()
        return pd.DataFrame(out)
    
    def save_ablations(self, out):
        annotations = pd.read_csv(f'{self.data_dir}/interim/ReorganziefMRI/stimulus_data.csv')
        annotations.drop(columns='captions', inplace=True)
        cap_annot = annotations.merge(out, on='video_name')
        cap_annot['captions'] = cap_annot[self.columns].apply(lambda row: row.dropna().tolist(), axis=1)
        cap_annot.drop(columns=self.columns, inplace=True)
        cap_annot.sort_values(by='video_name', inplace=True)
        cap_annot.to_csv(self.out_file, index=False)

    def get_correct_grammar(self):
        if not os.path.isfile(self.grammar_file): 
            # Load the raw captions and reorganize
            videos, columns, caption_arr, orig_shape = load_caption_file(f'{self.data_dir}/interim/CaptionData/captions.csv')

            # Load the grammar correction model
            tokenizer, gc_model = lang_permute.load_grammarcheck()

            # Loop through the captions to do the correction
            out  = []
            for caption in tqdm(caption_arr, total=len(caption_arr), desc='Grammar correction'):
                print(caption)
                corrected_caption = lang_permute.correct_grammar(self.prompt, caption, tokenizer, gc_model)
                out.append(corrected_caption)
            
            # Temporary saving for testing as insurance
            out = pd.DataFrame(np.array(out).reshape(orig_shape), columns=columns)
            out['video_name'] = videos
            out.to_csv(self.grammar_file, index=False)
        else: 
            print('loading the corrected captions')
            out = pd.read_csv(self.grammar_file)
        return out

    def run(self):
        df = self.get_correct_grammar()
        if self.func is not None:
            df = self.loop_captions(df)
        self.save_ablations(df)
        print('Finished!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--func_name', type=str, default='shuffle_sentence')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--data_dir', '-data', type=str,
                         default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIfMRI_modeling/data')        
    parser.add_argument('--cache', type=str, default='/home/emcmaho7/scratch4-lisik3/emcmaho7/cache')                
    args = parser.parse_args()
    SentenceDecomposition(args).run()


if __name__ == '__main__':
    main()