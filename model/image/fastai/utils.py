import os
from tqdm import tqdm


def fix_image_path(input_df, relative_path):
    df = input_df.copy()
    for idx in tqdm(range(len(input_df))):
        filename = df.at[idx, 'image_path'].split('/')[1]
        if not filename.endswith('.jpg'):
            filename = filename + '.jpg'
        final_filename = os.path.join(relative_path, filename)
        df.at[idx, 'image_path'] = final_filename
    return df
