#!/usr/bin/env python

import os
from datetime import datetime

import pandas as pd

from model.heuristic.utilities import *
from utils.envs import *


def run_fashion_heuristic(fashion_train, fashion_profile, fashion_profile_secondary, fashion_max_length):
    fashion_pred = fashion_train.filter(['itemid', 'title'])
    for feature in fashion_profile.keys():

        libraries = [fashion_profile[feature], fashion_profile_secondary[feature]]
        level = fashion_max_length[feature]

        fashion_pred[feature] = fashion_train.title.apply(get_feature, args=(libraries,level,))
    return fashion_pred


PROJECT_PATH = '/Users/idawatibustan/Dev/shopee_data_science'
FASHION_LIBRARY_FILE = 'model/heuristic/fashion_library_20190316.json'


fashion_library = load_library(FASHION_LIBRARY_FILE)
fashion_profile = fashion_library.get('primary')
fashion_profile_secondary = fashion_library.get('secondary')
fashion_max_length = fashion_library.get('length')


fashion_val = pd.read_csv(fashion_val_repo)


# run heuristic function
start_time = datetime.now()


fashion_val_pred = run_fashion_heuristic(
    fashion_val, 
    fashion_profile,
    fashion_profile_secondary,
    fashion_max_length
)

duration = datetime.now() - start_time
print(duration)


# format answer for submission
fashion_val_submission = pd.DataFrame(columns=['id', 'tagging'])

for feature in fashion_profile.keys():
    temp = pd.DataFrame()
    temp['id'] = fashion_val_pred['itemid'].map(str)+"_"+feature
    temp['tagging'] = fashion_val_pred[feature]
    
    if len(fashion_val_submission) == 0:
        fashion_val_submission = temp
    else:
        fashion_val_submission = fashion_val_submission.append(temp, ignore_index=True)
    print(feature, len(fashion_val_submission))


# export result to csv
outpath = 'output/result'
filename = 'fashion_info_val_submission_'+str(datetime.now().date()).replace('-', '')+'.csv'
fashion_val_submission.to_csv(os.path.join(outpath,filename))
print("Result saved: ", os.path.join(outpath,filename))
