import pandas as pd

def parse_caption_data(dataset_file, format='long', output_file=None, **kwargs):

    captions_wide = pd.read_csv(dataset_file)

    def row_func(row):
        return [x for x in row['caption01':] if not pd.isna(x)]

    captions_list = captions_wide.apply(lambda row: row_func(row), axis=1)

    captions_dict = {captions_wide['video_name'].iloc[i]: 
                     captions_list.iloc[i] for i in range(len(captions_wide))}

    captions_long = pd.melt(captions_wide, id_vars=['video_name'], 
                            value_vars=[f'caption{i:02}' for 
                                        i in range(1, 12)],
                            var_name='caption_index', value_name='caption')
    
    captions_long['caption_index'] = (captions_long['caption_index'].str
                                      .replace('caption', '').astype(int))
    
    # Optional: removing rows where caption is NaN
    captions_long.dropna(subset=['caption'], inplace=True)

    caption_counts = (captions_long.groupby('video_name')['caption_index'].max()
                      .reset_index().rename(columns={'caption_index': 'caption_count'}))

    count_min = caption_counts['caption_count'].min()
    count_max = caption_counts['caption_count'].max()

    if kwargs.pop('show_counts', False):
        print(f'Caption Count Min-Max: ({count_min}, {count_max})')

    if format=='nested':
        data['captions'] = captions_list
        return data['video_name','captions']

    if format=='dict':
        return captions_dict

    if format=='wide':
        return captions_wide
        
    if format=='long':
        return captions_long

    raise ValueError('format must be one of {nested,dict,wide,list}')