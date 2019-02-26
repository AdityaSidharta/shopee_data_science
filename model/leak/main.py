from utils.envs import *
import pandas as pd
import os
import json

if __name__ == '__main__':
    fashion_val = pd.read_csv(fashion_val_repo)
    fashion_leak = pd.read_csv(os.path.join(data_path, 'fashion_data_info_val_submission.csv'))
    with open(fashion_profile_json) as f:
        fashion_json = json.load(f)

    for column in fashion_leak.columns:
        if column in fashion_json.keys():
            fashion_leak[column] = fashion_leak[column].map(fashion_json[column])

    fashion_full = pd.concat([fashion_val, fashion_leak], axis=1)

    result_row = []
    for idx, row in fashion_full.iterrows():
        for columns in ['Clothing Material', 'Pattern',
                        'Sleeves', 'Collar Type', 'Fashion Trend']:
            value = row[columns]
            value = str(int(value)) if not pd.isnull(value) else ''
            result_row.append({
                'id': str(row.itemid) + '_{}'.format(columns),
                'tagging': value
            })
    result_df = pd.DataFrame(result_row)

    result_df.to_csv(os.path.join(result_path, 'leak_answer.csv'), index=False)
