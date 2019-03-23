from datetime import datetime
import os
import json

import pandas as pd

from model.heuristic.utilities import *
from utils.envs import *


def run_beauty_heuristic(beauty_train, beauty_profile, beauty_profile_secondary, beauty_profile_third, beauty_max_length):
    beauty_pred = beauty_train.filter(['itemid', 'title'])

    feature = 'Product_texture'
    libraries = [beauty_profile[feature], beauty_profile_secondary[feature]]
    level = beauty_max_length[feature]
    beauty_pred[feature] = beauty_train['title'].apply(get_feature, args=(libraries,level,))

    feature = 'Brand'
    libraries = [beauty_profile[feature], beauty_profile_secondary[feature]]
    level = beauty_max_length[feature]
    temp = beauty_train['title'].apply(get_feature_strip, args=(libraries,level,))
    beauty_pred['title_strip'] = temp.map(get_first)
    beauty_pred['Brand'] = temp.map(get_second)

    for feature in ['Colour_group', 'Benefits']:
        libraries = [beauty_profile[feature], beauty_profile_secondary[feature]]
        level = beauty_max_length[feature]
        beauty_pred[feature] = beauty_pred.title_strip.apply(get_feature, args=(libraries,level,))

    feature = 'Skin_type'
    libraries = [beauty_profile[feature], beauty_profile_secondary[feature], beauty_profile_third[feature]]
    level = beauty_max_length[feature]
    beauty_pred[feature] = beauty_pred.title_strip.apply(get_feature, args=(libraries,level,))

    return beauty_pred


PROJECT_PATH = '/Users/idawatibustan/Dev/shopee_data_science'
BEAUTY_LIBRARY_FILE = 'model/heuristic/beauty_library_20190322.json'


if __name__ == '__main__':
    beauty_library = load_library(BEAUTY_LIBRARY_FILE)
    beauty_profile = beauty_library.get('primary')
    beauty_profile_secondary = beauty_library.get('secondary')
    beauty_profile_third = beauty_library.get('third')
    beauty_max_length = beauty_library.get('length')


    beauty_val = pd.read_csv(beauty_val_repo)


    # run heuristic function
    start_time = datetime.now()


    beauty_val_pred = run_beauty_heuristic(
        beauty_val, 
        beauty_profile,
        beauty_profile_secondary,
        beauty_profile_third,
        beauty_max_length
    )

    duration = datetime.now() - start_time
    print(duration)


    # format answer for submission
    beauty_val_submission = pd.DataFrame(columns=['id', 'tagging'])

    for feature in beauty_profile.keys():
        temp = pd.DataFrame()
        temp['id'] = beauty_val_pred['itemid'].map(str)+"_"+feature
        temp['tagging'] = beauty_val_pred[feature]

        if len(beauty_val_submission) == 0:
            beauty_val_submission = temp
        else:
            beauty_val_submission = beauty_val_submission.append(temp, ignore_index=True)
        print(feature, len(beauty_val_submission))


    # export result to csv
    outpath = 'output/result'
    filename = 'beauty_info_val_submission_'+str(datetime.now().date()).replace('-', '')+'.csv'
    beauty_val_submission.to_csv(os.path.join(outpath,filename))
    print("Result saved: ", os.path.join(outpath,filename))
